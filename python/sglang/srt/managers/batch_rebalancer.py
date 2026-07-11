"""Manages dynamic rebalancing between running and pending batches.

Supports TPOT_FIRST / E2E_FIRST scheduling modes.
"""

import logging

logger = logging.getLogger(__name__)


class BatchRebalancer:
    """Manages dynamic rebalancing between running and pending batches.

    Supports TPOT_FIRST / E2E_FIRST scheduling modes.
    """

    def __init__(self, scheduler):
        self._sched = scheduler
        args = scheduler.server_args

        self.waiting_tok_threshold = args.br_waiting_tok_threshold
        self.load_alpha = args.br_load_alpha
        self.base_unit = args.br_base_unit
        self.kv_free_threshold = args.dc_kv_free_threshold

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
        use_tpot_first = tpot_signal >= TPOT_FIRST_THRESHOLD
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

        # Pick the largest threshold (th) not exceeding n_reqs (0, base_th, 2*base_th ...)
        th = (n_reqs_total // base_th) * base_th
        current_local_limit = (th // sched.pp_size) + base_unit

        # --- Case 1: [th, th + alpha * 128) -> keep last at the previous tier (th level)
        # Skip this suppression band and fall through to Case 2 only when both waiting pressure and KV headroom are sufficient
        if th > 0 and n_reqs_total < th + threshold_margin and not has_kv_headroom:
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
                    if use_tpot_first:
                        batch_move.merge_batch(sched.pending_batch)
                        sched.pending_batch = batch_move
                    else:
                        batch_move.merge_batch(sched.pending_batch)  # [LIFO]
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
                    if use_tpot_first:
                        run_give.merge_batch(sched.pending_batch)
                        sched.pending_batch = run_give
                    else:
                        run_give.merge_batch(sched.pending_batch)  # [LIFO]
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
