"""Manages dynamic rebalancing between running and pending batches.

Supports TPOT_FIRST / E2E_FIRST scheduling modes, and the adaptive mode (E2E_FIRST order with
a band width that follows the KV pressure).
"""

import logging
import time

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS_ESTIMATION

logger = logging.getLogger(__name__)


class BatchRebalancer:
    """Manages dynamic rebalancing between running and pending batches.

    Supports TPOT_FIRST / E2E_FIRST scheduling modes.

    Adaptive mode: E2E_FIRST order; the band [th, th + alpha * base_unit * pp) is narrow
    (br_alpha_low, only the requests just past a base_unit boundary pause) unless the KV is
    under pressure, when it widens to br_alpha_high so that paused requests stop growing their
    KV and stop counting in the admission's reservation. Pressure: admission blocked by KV
    (requests waiting after a refused admission), or the KV budget with the remaining outputs
    of every live request reserved as PrefillAdder does (paused ones included) below
    br_kv_pressure of the pool. Both count only while a new (never retracted) request waits or
    one arrived within br_arrival_window_s, so a full KV with no new arrivals (e.g. a drained
    burst) pauses nobody. The wide band stays on br_hold_s after the last pressure, and while
    the slot is capped the band ends br_margin requests later (hysteresis). The decision is
    made on the rank that receives requests and sent to the others with the requests
    (Scheduler.recv_requests), so every rank rebalances alike.
    """

    def __init__(self, scheduler):
        self._sched = scheduler
        args = scheduler.server_args

        self.waiting_tok_threshold = args.br_waiting_tok_threshold
        self.load_alpha = args.br_load_alpha
        self.base_unit = args.br_base_unit
        self.kv_free_threshold = args.dc_kv_free_threshold
        self.br_mode = args.br_mode

        # Adaptive mode
        self.alpha_low = args.br_alpha_low
        self.alpha_high = args.br_alpha_high
        self.kv_pressure = args.br_kv_pressure
        self.hold_s = args.br_hold_s
        self.margin = args.br_margin
        self.arrival_window_s = args.br_arrival_window_s
        self.last_arrival = float("-inf")  # time.monotonic() of the last new request
        self._hi_until = float("-inf")
        self._in_band = False

    @property
    def adaptive(self):
        return self.br_mode == "adaptive"

    # ------------------------------------------------------------------
    # Adaptive mode: the wide-band decision (on the rank receiving requests)
    # ------------------------------------------------------------------

    def adaptive_signal(self) -> bool:
        """Whether the wide band (br_alpha_high) is on, from this rank's state."""
        now = time.monotonic()
        waiting = self._waiting_for_kv()
        recent = now - self.last_arrival <= self.arrival_window_s
        if recent or any(not req.br_retracted for req in waiting):
            if (waiting and self._admission_blocked()) or self._kv_pressure():
                self._hi_until = now + self.hold_s
        return now < self._hi_until

    def _waiting_for_kv(self):
        """The requests waiting for KV to be admitted."""
        return self._sched.waiting_queue

    def _admission_blocked(self) -> bool:
        """The last admission attempt was refused for lack of KV."""
        return self._sched.batch_is_full

    def _live_reqs(self):
        """Every request holding KV here: the running batches of all micro-batches, the
        batches in flight and the paused (pending) requests."""
        sched = self._sched
        batches = list(sched._running_batch) + list(sched._last_batch) + [sched.pending_batch]
        seen, reqs = set(), []
        for batch in batches:
            if batch is None:
                continue
            for req in batch.reqs:
                if id(req) not in seen and not req.finished():
                    seen.add(id(req))
                    reqs.append(req)
        return reqs

    def _kv_pressure(self) -> bool:
        """The admission budget with every live request's remaining output reserved
        (PrefillAdder's estimate, paused requests included) is below br_kv_pressure."""
        sched = self._sched
        available = sched.token_to_kv_pool.available_size() + sched.tree_cache.evictable_size()
        reserve = 0
        for req in self._live_reqs():
            reserve += min(
                req.sampling_params.max_new_tokens - req.output_ids_len,
                CLIP_MAX_NEW_TOKENS_ESTIMATION,
            )
        budget = available - reserve * sched.new_token_ratio
        return budget < self.kv_pressure * sched.max_total_num_tokens

    # ------------------------------------------------------------------
    # Main rebalancing
    # ------------------------------------------------------------------

    def rebalance(self):
        """Rebalance requests between running and pending batches."""
        sched = self._sched

        # Both empty — nothing to do
        if (sched.running_batch is None or sched.running_batch.is_empty()) and \
           (sched.pending_batch is None or sched.pending_batch.is_empty()):
            return

        WAITING_TOK_THRESHOLD = self.waiting_tok_threshold
        KV_FREE_THRESHOLD = self.kv_free_threshold
        alpha = self.load_alpha
        base_unit = self.base_unit
        TPOT_FIRST_THRESHOLD = sched.slo_tpot * 0.9

        if sched.running_batch is not None:
            initial_bs = sched.running_batch.batch_size()
            sched.running_batch.filter_batch()
            if sched.running_batch.batch_size() < initial_bs:
                sched.batch_is_full = False

        # Compute total request count
        n_reqs_total = 0
        for idx in range(sched.pp_size):
            if sched._last_batch[idx] is not None:
                n_reqs_total += sched._last_batch[idx].batch_size()
            if idx != sched.cur_pp.get() and sched._being_chunked_req[idx]:
                n_reqs_total -= 1

        if sched.pending_batch is not None:
            n_reqs_total += sched.pending_batch.batch_size()

        rebalance_metrics = getattr(sched, "rebalance_metrics", None)
        if rebalance_metrics is None:
            total_waiting_tokens = 0
            for r in sched.waiting_queue:
                total_waiting_tokens += len(r.origin_input_ids)

            available_size = sched.token_to_kv_pool.available_size() + sched.tree_cache.evictable_size()
            kv_free = available_size / sched.max_total_num_tokens

            tpot_signal = float(sched.tpot)
        else:
            tpot_signal, total_waiting_tokens, kv_free = rebalance_metrics

        has_kv_headroom = (
            total_waiting_tokens > WAITING_TOK_THRESHOLD and kv_free > KV_FREE_THRESHOLD
        )

        # Switch to TPOT_FIRST mode when tpot_signal exceeds threshold, otherwise stay E2E_FIRST.
        # br_mode overrides the automatic selection when set to tpot_first/e2e_first.
        # adaptive: E2E_FIRST, alpha from the wide-band decision received with the requests.
        if self.adaptive:
            use_tpot_first = False
            wide = sched.br_adaptive_wide
            alpha = self.alpha_high if wide else self.alpha_low
        elif self.br_mode == "tpot_first":
            use_tpot_first = True
        elif self.br_mode == "e2e_first":
            use_tpot_first = False
        else:
            use_tpot_first = tpot_signal >= TPOT_FIRST_THRESHOLD
        if self.adaptive:
            rebalance_mode = "ADAPTIVE_WIDE" if wide else "ADAPTIVE"
        else:
            rebalance_mode = "TPOT_FIRST" if use_tpot_first else "E2E_FIRST"

        def get_split_indices(total_size, move_size, is_take_from_pending):
            if use_tpot_first:
                if is_take_from_pending:
                    # TPOT_FIRST: take from the tail of pending and append to the tail of running.
                    take_idx = list(range(total_size - move_size, total_size))
                    keep_idx = list(range(total_size - move_size))
                else:
                    # TPOT_FIRST: take from the tail of running and prepend to pending.
                    keep_cnt = total_size - move_size
                    keep_idx = list(range(keep_cnt))
                    take_idx = list(range(keep_cnt, total_size))
            else:
                if is_take_from_pending:
                    take_idx = list(range(total_size - move_size, total_size))
                    keep_idx = list(range(total_size - move_size))
                else:
                    keep_idx = list(range(move_size, total_size))
                    take_idx = list(range(move_size))
            return keep_idx, take_idx

        # Base unit definition
        base_th = base_unit * sched.pp_size
        threshold_margin = alpha * base_th
        if self.adaptive and self._in_band:
            # hysteresis: a capped band ends br_margin requests later
            threshold_margin += self.margin

        # Pick the largest threshold (th) not exceeding n_reqs (0, base_th, 2*base_th ...)
        th = (n_reqs_total // base_th) * base_th
        current_local_limit = (th // sched.pp_size) + base_unit

        # --- Case 1: [th, th + alpha * 128) -> keep last at the previous tier (th level)
        # Skip this suppression band and fall through to Case 2 only when both waiting pressure and KV headroom are sufficient
        in_band = th > 0 and n_reqs_total < th + threshold_margin and not has_kv_headroom
        self._in_band = in_band
        if in_band:
            desired_last = th // sched.pp_size
            last_sz = 0 if (sched.running_batch is None) else sched.running_batch.batch_size()

            if last_sz > desired_last:
                move_cnt = last_sz - desired_last
                keep_idx, move_idx = get_split_indices(
                    last_sz, move_cnt, is_take_from_pending=False
                )
                batch_keep, batch_move = sched.running_batch.split_batch(keep_idx, move_idx)

                sched.running_batch = batch_keep

                if sched.pending_batch is None or sched.pending_batch.is_empty():
                    sched.pending_batch = batch_move
                else:
                    batch_move.merge_batch(sched.pending_batch)
                    sched.pending_batch = batch_move

                if sched.pp_rank == 0:
                    logger.info(
                        f"[CASE1_1 {th=} REBALANCE{sched.cur_pp.get()} {rebalance_mode}] "
                        f"last excess {move_cnt} -> pending "
                        f"(pending size={sched.pending_batch.batch_size()}, tpot_signal={tpot_signal:.3f})"
                    )

            elif last_sz < desired_last and sched.pending_batch and not sched.pending_batch.is_empty():
                need = desired_last - last_sz
                take = min(need, sched.pending_batch.batch_size())

                batch_sz = sched.pending_batch.batch_size()
                keep_idx, take_idx = get_split_indices(
                    batch_sz, take, is_take_from_pending=True
                )
                pend_keep, pend_take = sched.pending_batch.split_batch(keep_idx, take_idx)

                if pend_take:
                    if sched.running_batch is None or sched.running_batch.is_empty():
                        sched.running_batch = pend_take
                    else:
                        if use_tpot_first:
                            sched.running_batch.merge_batch(pend_take)
                        else:
                            pend_take.merge_batch(sched.running_batch)  # [LIFO]
                            sched.running_batch = pend_take

                if pend_keep is None or pend_keep.is_empty():
                    sched.pending_batch = None
                else:
                    sched.pending_batch = pend_keep
                if sched.pp_rank == 0:
                    logger.info(
                        f"[CASE1_2 {th=} REBALANCE{sched.cur_pp.get()} {rebalance_mode}] "
                        f"pending -> {take} -> last "
                        f"(remaining pending size={pend_keep.batch_size() if pend_keep else 0}, tpot_signal={tpot_signal:.3f})"
                    )

        # --- Case 2: >= (1+alpha)*th (next tier allowed) or KV headroom is sufficient
        else:
            desired_last = n_reqs_total // sched.pp_size

            last_sz = 0 if (sched.running_batch is None or sched.running_batch.is_empty()) else sched.running_batch.batch_size()
            pend_sz = 0 if (sched.pending_batch is None or sched.pending_batch.is_empty()) else sched.pending_batch.batch_size()

            need = desired_last - last_sz
            give = last_sz - current_local_limit

            # Allow aggressive rebalance only when KV headroom is sufficient
            is_over_limit = False if has_kv_headroom else (last_sz > current_local_limit)

            if need > 0 and pend_sz > 0:
                take = min(need, pend_sz)

                batch_sz = sched.pending_batch.batch_size()
                keep_idx, take_idx = get_split_indices(
                    batch_sz, take, is_take_from_pending=True
                )
                pend_keep, pend_take = sched.pending_batch.split_batch(keep_idx, take_idx)

                if sched.running_batch and not sched.running_batch.is_empty():
                    if use_tpot_first:
                        sched.running_batch.merge_batch(pend_take)
                    else:
                        pend_take.merge_batch(sched.running_batch)  # [LIFO]
                        sched.running_batch = pend_take
                else:
                    sched.running_batch = pend_take

                sched.pending_batch = None if (pend_keep is None or pend_keep.is_empty()) else pend_keep
                if sched.pp_rank == 0:
                    log_tag = "KV_HEADROOM" if has_kv_headroom else f"CASE2_1 {th=}"
                    logger.info(
                        f"[{log_tag} REBALANCE{sched.cur_pp.get()} {rebalance_mode}] "
                        f"pending -> {take} -> last "
                        f"(target={desired_last}, tpot_signal={tpot_signal:.3f})"
                    )

            elif (is_over_limit) and last_sz > 0:
                give = min(give, last_sz)

                keep_idx, give_idx = get_split_indices(
                    last_sz, give, is_take_from_pending=False
                )
                run_keep, run_give = sched.running_batch.split_batch(keep_idx, give_idx)

                if sched.pending_batch and not sched.pending_batch.is_empty():
                    run_give.merge_batch(sched.pending_batch)
                    sched.pending_batch = run_give
                else:
                    sched.pending_batch = run_give

                sched.running_batch = None if (run_keep is None or run_keep.is_empty()) else run_keep
                if sched.pp_rank == 0:
                    log_tag = "KV_HEADROOM" if has_kv_headroom else f"CASE2_2 {th=}"
                    logger.info(
                        f"[{log_tag} REBALANCE{sched.cur_pp.get()} {rebalance_mode}] "
                        f"last -> {give} -> pending "
                        f"(remaining pending size={sched.pending_batch.batch_size() if sched.pending_batch else 0}, "
                        f"tpot_signal={tpot_signal:.3f})"
                    )

        return
