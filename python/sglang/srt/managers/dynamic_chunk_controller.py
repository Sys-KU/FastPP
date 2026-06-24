"""Dynamic chunked prefill size controller.

Manages EMA metrics and decides optimal chunk size using either
greedy rules or ALP model predictions.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DynamicChunkController:
    """Dynamic chunked prefill size controller.

    Manages EMA metrics and decides optimal chunk size using either
    greedy rules or ALP model predictions.
    """

    def __init__(self, scheduler):
        self._sched = scheduler
        args = scheduler.server_args

        # SLO targets
        self.slo_ttft = args.dc_slo_ttft
        self.slo_tpot = args.dc_slo_tpot

        # Greedy bounds
        self.base_chunk = args.dc_greedy_base_chunk
        self.max_chunk = args.dc_greedy_max_chunk
        self.min_chunk = args.dc_greedy_min_chunk

        # Tuning knobs (shared)
        self.kv_free_threshold = args.dc_kv_free_threshold

        # Tuning knobs (greedy)
        self.tpot_violation_ratio = args.dc_greedy_tpot_violation_ratio
        self.ttft_low_ratio = args.dc_greedy_ttft_low_ratio
        self.ttft_high_ratio = args.dc_greedy_ttft_high_ratio
        self.tpot_safe_ratio = args.dc_greedy_tpot_safe_ratio

        # Tuning knobs (ALP)
        self.alp_fresh_epsilon = args.dc_alp_fresh_epsilon
        self.alp_slo_tpot_coeff = args.dc_alp_slo_tpot_coeff
        self.alp_throughput_coeff = args.dc_alp_throughput_coeff

        # ALP model state
        self.alp_scheduler = scheduler.alp_scheduler
        self.cur_chunk = None
        self.next_chunk = None

        # Step tracking (used by event loop via pre_step / post_step)
        self.step_count: int = 0

        # KV pressure restore chunk state
        self._kv_pressure_chunk_restore: Dict[str, Optional[int]] = {}

        # Runtime exec sample counters (mirroring sglang scheduler state)
        self.exec_time_update_sum: float = 0.0
        self.exec_time_update_count: int = 0
        self._alp_runtime_table_warning_logged = False
        self._runtime_feature_snapshot = None

    def decide_chunk_size(self):
        """Decide chunk size at the start of a scheduler iteration."""
        if not self._sched.server_args.enable_dynamic_chunk:
            return
        sched = self._sched
        if sched.server_args.dynamic_chunk_strategy == "alp":
            if self.step_count % (sched.pp_size * 2) == 0:
                if sched.pp_rank == 0 and sched.tp_rank == 0:
                    self.decide_chunk_alp()
                self.sync_chunk_size()
        else:  # greedy
            if self.step_count % sched.pp_size == 0:
                if sched.pp_rank == 0 and sched.tp_rank == 0:
                    self.decide_chunk_greedy()
                self.sync_chunk_size()

    def pre_step(self, batch):
        """Pre-batch hook: stash runtime features and collect exec sample snapshot."""
        if self.alp_scheduler is None:
            return
        self._stash_batch_runtime_feature_snapshot(batch)
        self._runtime_feature_snapshot = self._get_runtime_exec_sample_snapshot(batch)

    def post_step(self, start_time, batch):
        """Post-batch hook: record exec sample (RLS update) and increment step count."""
        if self.alp_scheduler is not None:
            self._record_runtime_exec_sample(start_time, self._runtime_feature_snapshot, batch)
        self._runtime_feature_snapshot = None
        self.step_count += 1

    def reset(self):
        """Reset state (called on cache flush)."""
        self.step_count = 0
        self._kv_pressure_chunk_restore.clear()
        self.exec_time_update_sum = 0.0
        self.exec_time_update_count = 0
        self._alp_runtime_table_warning_logged = False

    def _set_chunked_prefill_size(self, chunk_size, source: str = ""):
        """Mirror sglang _set_chunked_prefill_size: keep max_prefill_tokens in sync."""
        sched = self._sched
        sched.chunked_prefill_size = sched.max_prefill_tokens = chunk_size

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _collect_base_metrics(self):
        """Collect base metrics common to greedy and ALP strategies.

        Returns a dict with:
            waiting_tok_num, running_req_num, ttft_vals, tpot_vals,
            last_idx, kv_free
        """
        sched = self._sched
        waiting_tok_num = [len(r.origin_input_ids) for r in sched.waiting_queue]
        running_req_num = []
        ttft_vals, tpot_vals = [], []
        last_idx = None

        for idx in range(sched.pp_size):
            if not sched._last_batch[idx]:
                running_req_num.append(0)
                continue
            running_req_num.append(len(sched._last_batch[idx].reqs))
            if sched._last_batch[idx].ttft:
                ttft_vals.extend(sched._last_batch[idx].ttft)
                sched._last_batch[idx].ttft = []
            if sched._last_batch[idx].tpot:
                tpot_vals.extend(sched._last_batch[idx].tpot)
                sched._last_batch[idx].tpot = []
                last_idx = idx

        available_size = sched.token_to_kv_pool.available_size() + sched.tree_cache.evictable_size()
        for idx in range(sched.pp_size):
            available_size -= running_req_num[idx]
        kv_free = available_size / sched.max_total_num_tokens

        return {
            'waiting_tok_num': waiting_tok_num,
            'running_req_num': running_req_num,
            'ttft_vals': ttft_vals,
            'tpot_vals': tpot_vals,
            'last_idx': last_idx,
            'kv_free': kv_free,
        }

    def _apply_safety_guard(self, next_chunk, running_req_num):
        """Apply safety guard: ensure chunk size exceeds running request count."""
        sched = self._sched
        if any(sched._last_batch[idx] and next_chunk <= len(sched._last_batch[idx].reqs)
               for idx in range(sched.pp_size)):
            max_run = max(running_req_num) if running_req_num else 0
            return (max_run // 128 + 1) * 128
        return next_chunk

    def _get_kv_pressure_restore_chunk(self, policy_name: str) -> Optional[int]:
        return self._kv_pressure_chunk_restore.get(policy_name)

    def _update_kv_pressure_restore_chunk(
        self,
        policy_name: str,
        kv_pressure_active: bool,
        cur_chunk: int,
        applied_chunk: int,
        restore_chunk: Optional[int],
    ) -> None:
        if kv_pressure_active:
            if restore_chunk is None and applied_chunk < cur_chunk:
                self._kv_pressure_chunk_restore[policy_name] = cur_chunk
            elif restore_chunk is not None and applied_chunk >= restore_chunk:
                self._kv_pressure_chunk_restore.pop(policy_name, None)
            return

        if restore_chunk is not None and applied_chunk == restore_chunk:
            self._kv_pressure_chunk_restore.pop(policy_name, None)

    # ------------------------------------------------------------------
    # Greedy strategy — logic ported from sglang
    # ------------------------------------------------------------------

    def decide_chunk_greedy(self):
        """Greedy adaptive chunk size decision."""
        sched = self._sched
        SLO_TTFT, SLO_TPOT = self.slo_ttft, self.slo_tpot
        base_chunk = self.base_chunk
        max_chunk = self.max_chunk
        min_chunk = self.min_chunk

        # 1. Metrics Collection
        m = self._collect_base_metrics()
        running_req_num = m['running_req_num']
        ttft_vals = m['ttft_vals']
        tpot_vals = m['tpot_vals']
        last_idx = m['last_idx']
        kv_free = m['kv_free']

        if not ttft_vals:
            if last_idx is not None:
                sched._last_batch[last_idx].tpot = tpot_vals
            return

        # 2. Update scheduler-level EMA Metrics
        if ttft_vals:
            sched.ttft = float(np.mean(ttft_vals))
        if tpot_vals:
            sched.tpot = float(np.mean(tpot_vals))

        cur_chunk = sched.chunked_prefill_size
        next_chunk = cur_chunk

        policy_name = "heu"
        restore_chunk = self._get_kv_pressure_restore_chunk(policy_name)
        kv_pressure_active = kv_free < self.kv_free_threshold

        # 3. Decision Logic (3-Step Algorithm)
        # [Step 1] KV Memory Safety Check
        if kv_pressure_active:
            next_chunk = ((min(running_req_num) + 64) // 128 + 1) * 128
        elif restore_chunk is not None:
            next_chunk = restore_chunk
        # [Step 2] Decrease Size (TPOT Violation OR Low Load)
        elif sched.tpot > SLO_TPOT * self.tpot_violation_ratio or sched.ttft < SLO_TTFT * self.ttft_low_ratio:
            next_chunk = max(cur_chunk - base_chunk, min_chunk)
        # [Step 3] Increase Size (High Load)
        elif sched.ttft > SLO_TTFT * self.ttft_high_ratio and sched.tpot < SLO_TPOT * self.tpot_safe_ratio:
            next_chunk = min(cur_chunk + base_chunk, max_chunk)

        # 4. Safety Guard
        next_chunk = self._apply_safety_guard(next_chunk, running_req_num)

        self._set_chunked_prefill_size(next_chunk, source="decide_chunk_greedy")

        self._update_kv_pressure_restore_chunk(
            policy_name=policy_name,
            kv_pressure_active=kv_pressure_active,
            cur_chunk=cur_chunk,
            applied_chunk=next_chunk,
            restore_chunk=restore_chunk,
        )

        # 5. Logging
        if sched.tp_rank == 0 and sched.pp_rank == 0:
            if kv_pressure_active:
                logger.info(f"[HEU] {cur_chunk} -> {next_chunk} (kv_pressure)")
            elif restore_chunk is not None:
                logger.info(f"[HEU] {cur_chunk} -> {next_chunk} (kv_restore)")
            else:
                logger.info(f"[HEU] {cur_chunk} -> {next_chunk}")

    # ------------------------------------------------------------------
    # ALP strategy — logic ported from sglang
    # ------------------------------------------------------------------

    def decide_chunk_alp(self):
        """ALP model-based adaptive chunk size decision."""
        if self.alp_scheduler is None:
            return

        sched = self._sched
        SLO_TTFT = float(self.slo_ttft)
        SLO_TPOT = float(self.slo_tpot)
        EPS = 1e-9

        FRESH_EPSILON = SLO_TTFT * self.alp_fresh_epsilon

        # 1. Empty state check
        empty_check = True
        if sched.waiting_queue:
            empty_check = False
        else:
            for idx in range(sched.pp_size):
                if sched._being_chunked_req[idx] and sched._being_chunked_req[idx].pp_idx == idx:
                    empty_check = False
                    break
        if empty_check:
            return

        # 2. State collection — base metrics (drains ttft/tpot)
        m = self._collect_base_metrics()
        ttft_vals = m['ttft_vals']
        tpot_vals = m['tpot_vals']
        last_idx = m['last_idx']

        # 2b. ALP-specific state collection (waiting_tok_num with remain_prefill like sglang)
        waiting_tok_num = []
        for r in sched.waiting_queue:
            fill_len = (
                len(r.fill_ids)
                if r.fill_ids is not None
                else (len(r.origin_input_ids) + len(r.output_ids))
            )
            prefix_len = len(r.prefix_indices)
            remain_prefill = max(fill_len - prefix_len, 0)
            waiting_tok_num.append(remain_prefill)

        # 2b. ALP-specific state collection
        cur_time = time.perf_counter()

        waiting_req_time = [cur_time - r.arrival_time for r in sched.waiting_queue]

        qps_tok = []
        window_sec = 10
        window_start_time = cur_time - window_sec
        window_sec = min(window_sec, cur_time - sched.system_start_time)
        for r in getattr(sched, "qps_queue", []):
            if r.arrival_time >= window_start_time:
                qps_tok.append(len(r.origin_input_ids) / window_sec)

        qps_tok_sum = float(np.sum(qps_tok)) if qps_tok else 0.0

        running_req_num = []
        running_token = []

        for idx in range(sched.pp_size):
            if not sched._last_batch[idx]:
                running_req_num.append(0)
                running_token.append(0)
                continue

            req_num = 0
            chunk_req = sched._being_chunked_req[idx]

            if chunk_req and chunk_req.pp_idx == idx:
                total_lens = len(chunk_req.origin_input_ids) + len(chunk_req.output_ids)
                chunked_lens = len(chunk_req.fill_ids)
                remain_tokens = max(total_lens - chunked_lens, 0)
                waiting_req_time.append(cur_time - chunk_req.arrival_time)
                waiting_tok_num.append(remain_tokens)

            for r in sched._last_batch[idx].reqs:
                if r is chunk_req:
                    continue
                req_num += 1

            running_req_num.append(req_num)
            running_token.append(len(sched._last_batch[idx].input_ids))

        # Recompute kv_free with ALP-specific running_req_num
        available_size = (sched.token_to_kv_pool.available_size() + sched.tree_cache.evictable_size())
        for idx in range(sched.pp_size):
            available_size -= running_req_num[idx]
        available_size = max(0, available_size)
        kv_free = (available_size / (sched.max_total_num_tokens))

        # 3. EMA update
        if ttft_vals:
            sched.ttft = float(np.mean(ttft_vals))
        if tpot_vals:
            sched.tpot = float(np.mean(tpot_vals))

        exec_time = self._consume_runtime_exec_samples()

        self.cur_chunk = sched.chunked_prefill_size

        decision_reason = ""
        raw_exec_time_pred = None

        alp_candidate_chunks = self.alp_scheduler.candidate_chunks

        if kv_free < self.kv_free_threshold:
            min_running_req = min(running_req_num) if running_req_num else 0
            self.next_chunk = ((min_running_req + 64) // 128 + 1) * 128
            decision_reason = "No KV Storage"
        else:
            # Stale Filter & Hybrid Velocity
            stale_tokens = 0.0
            stale_wait_times = []

            if waiting_req_time:
                for t, tok in zip(waiting_req_time, waiting_tok_num):
                    if t >= FRESH_EPSILON:
                        stale_tokens += tok
                        stale_wait_times.append(t)

            current_inflow_qps = qps_tok_sum

            representative_wait_time = 0.0
            if stale_wait_times:
                representative_wait_time = float(np.mean(stale_wait_times))

            time_budget = SLO_TTFT - representative_wait_time

            catch_up_speed = 0.0
            if stale_tokens > 0:
                if time_budget <= SLO_TPOT:
                    catch_up_speed = float('inf')
                else:
                    catch_up_speed = stale_tokens / time_budget

            maintenance_speed = current_inflow_qps

            required_throughput = 0.0
            if catch_up_speed == float('inf'):
                required_throughput = float('inf')
            else:
                required_throughput = catch_up_speed + maintenance_speed

            avg_running_req = float(np.mean(running_req_num)) if running_req_num else 0.0
            chosen_chunk, decision_reason, raw_exec_time_pred = (
                self._select_alp_chunk_with_lazy_prediction(
                    alp_candidate_chunks,
                    slo_tpot=SLO_TPOT * self.alp_slo_tpot_coeff,
                    required_throughput=required_throughput * self.alp_throughput_coeff,
                    avg_running_req=avg_running_req,
                    catch_up_speed=catch_up_speed,
                    maintenance_speed=maintenance_speed,
                    eps=EPS,
                )
            )
            if raw_exec_time_pred is None or chosen_chunk is None:
                return

            self.next_chunk = int(chosen_chunk)

        # 9. Safety Guard
        if any(sched._last_batch[idx] and self.next_chunk <= len(sched._last_batch[idx].reqs)
               for idx in range(sched.pp_size)):
            self.next_chunk = (max(running_req_num) // 128 + 1) * 128
            decision_reason += " [Safety Override]"

        self._set_chunked_prefill_size(self.next_chunk, source="decide_chunk_alp")

        # Logging
        if sched.tp_rank == 0 and sched.pp_rank == 0:
            logger.info(
                f"[ALP] {self.cur_chunk}->{self.next_chunk} reason={decision_reason}"
            )

    # ------------------------------------------------------------------
    # ALP helper methods ported from sglang Scheduler
    # ------------------------------------------------------------------

    def _select_alp_chunk_with_lazy_prediction(
        self,
        candidate_chunks: List[int],
        slo_tpot: float,
        required_throughput: float,
        avg_running_req: float,
        catch_up_speed: float,
        maintenance_speed: float,
        eps: float,
    ) -> Tuple[Optional[int], str, Optional[Dict[int, float]]]:
        prediction_context = self._get_runtime_prediction_context()
        raw_exec_time_pred: Dict[int, float] = {}
        last_safe_chunk = None

        for chunk in sorted(int(c) for c in candidate_chunks):
            raw_pred = self._predict_alp_exec_time_for_chunk(
                chunk, prediction_context=prediction_context
            )
            if raw_pred is None:
                return None, "", None

            raw_exec_time_pred[chunk] = raw_pred
            if raw_pred * self._sched.pp_size > slo_tpot:
                break

            last_safe_chunk = chunk
            pred_exec = max(float(raw_pred), eps)
            effective_tokens = max(float(chunk) - avg_running_req, 0.0)
            capacity = effective_tokens / pred_exec
            if capacity >= required_throughput:
                return (
                    chunk,
                    f"Optimal (Min Chunk Meeting Req:{required_throughput:.0f} "
                    f"[Stale:{catch_up_speed:.0f}+Maint:{maintenance_speed:.0f}])",
                    raw_exec_time_pred,
                )

        if last_safe_chunk is None:
            return min(candidate_chunks), "Fail_SLO_ExecTime", raw_exec_time_pred

        return last_safe_chunk, "Best_Effort (Max Safe Chunk)", raw_exec_time_pred

    def _predict_alp_exec_time_for_chunk(
        self,
        chunk: int,
        prediction_context = None,
    ) -> Optional[float]:
        try:
            context = prediction_context or self._get_runtime_prediction_context()
            features = self._get_candidate_runtime_prediction_features_for_chunk(
                int(chunk), context
            )
            return self.alp_scheduler.predict_exec_time_for_chunk(
                int(chunk),
                chunk_features=features,
            )
        except RuntimeError as e:
            if not getattr(self, "_alp_runtime_table_warning_logged", False):
                logger.warning(f"ALP runtime table incomplete: {e}")
                self._alp_runtime_table_warning_logged = True
            return None

    def _consume_runtime_exec_samples(self) -> float:
        val = 0.0
        if self.exec_time_update_count > 0:
            val = self.exec_time_update_sum / self.exec_time_update_count
        self.exec_time_update_sum = 0.0
        self.exec_time_update_count = 0
        return val

    # ------------------------------------------------------------------
    # Rebalance metrics & chunk size sync (ported from sglang recv_chunk_size)
    # ------------------------------------------------------------------

    def sync_chunk_size(self):
        """Propagate chunk size and rebalance_metrics to other workers via zmq."""
        sched = self._sched
        if (sched.tp_rank == 0 and sched.pp_rank == 0) or sched.server_args.enable_dp_attention:
            chunk_size = sched.chunked_prefill_size
            if sched.server_args.enable_dp_attention:
                sched.rebalance_metrics = None
                sched.req_sender.send_pyobj(chunk_size)
            else:
                total_waiting_tokens = 0
                for r in sched.waiting_queue:
                    total_waiting_tokens += len(r.origin_input_ids)
                available_size = (
                    sched.token_to_kv_pool.available_size() + sched.tree_cache.evictable_size()
                )
                kv_free = available_size / sched.max_total_num_tokens
                sched.rebalance_metrics = (float(sched.tpot), total_waiting_tokens, kv_free)
                sched.req_sender.send_pyobj((chunk_size, sched.rebalance_metrics))
        else:
            recv_obj = sched.req_receiver.recv_pyobj()
            if isinstance(recv_obj, tuple) and len(recv_obj) == 2:
                chunk_size, sched.rebalance_metrics = recv_obj
            else:
                chunk_size = recv_obj
                sched.rebalance_metrics = None
        self._set_chunked_prefill_size(chunk_size, source="recv_chunk_size")

    # ------------------------------------------------------------------
    # ALP runtime feature snapshot & exec sample recording (ported from sglang)
    # ------------------------------------------------------------------

    def _stash_batch_runtime_feature_snapshot(
        self, batch
    ):
        if batch is None:
            return None
        snapshot = self._get_batch_runtime_features(batch)
        batch.alp_runtime_feature_snapshot = snapshot
        return snapshot

    def _get_batch_runtime_features(self, batch):
        if batch is None or batch.input_ids is None:
            return None
        running_token_max = len(batch.input_ids)
        if running_token_max <= 0:
            return None

        if batch.forward_mode.is_decode():
            decode_reqs = list(batch.reqs)
            prefill_reqs = []
        else:
            decode_reqs = list(batch.decoding_reqs or [])
            decode_req_ids = {id(req) for req in decode_reqs}
            prefill_reqs = [req for req in batch.reqs if id(req) not in decode_req_ids]

        selected_decode_ctx_sum = sum(
            len(req.origin_input_ids) + len(req.output_ids) for req in decode_reqs
        )
        prefill_square_sum = 0
        prefill_prefix_prod_sum = 0
        for req in prefill_reqs:
            l_pre = int(req.extend_input_len)
            l_past = len(req.prefix_indices)
            if l_pre <= 0:
                continue
            prefill_square_sum += l_pre * l_pre
            prefill_prefix_prod_sum += l_pre * l_past

        return (
            float(running_token_max),
            float(selected_decode_ctx_sum),
            float(prefill_square_sum),
            float(prefill_prefix_prod_sum),
        )

    def _get_runtime_exec_sample_snapshot(self, batch):
        sched = self._sched
        cur_idx = sched.cur_pp.get()
        valid_batches = [
            batch if idx == cur_idx else last_batch
            for idx, last_batch in enumerate(sched._last_batch)
            if (batch if idx == cur_idx else last_batch) is not None
        ]
        if not valid_batches:
            return None

        if not any(
            valid_batch.forward_mode is not None
            and valid_batch.forward_mode.is_extend()
            for valid_batch in valid_batches
        ):
            return None

        feature_snapshots = []
        for valid_batch in valid_batches:
            snapshot = valid_batch.alp_runtime_feature_snapshot
            if snapshot is None:
                snapshot = self._stash_batch_runtime_feature_snapshot(valid_batch)
            if snapshot is not None:
                feature_snapshots.append(snapshot)

        return self._select_runtime_feature_snapshot(feature_snapshots)

    def _select_runtime_feature_snapshot(self, feature_snapshots):
        if not feature_snapshots:
            return None

        warmup_updates = 16
        candidate_batches = []
        max_update_chunk = None
        for features in feature_snapshots:
            if features is None:
                continue
            update_chunk = self._get_runtime_update_chunk_from_tokens(
                int(features[0])
            )
            if update_chunk is None:
                continue
            candidate_batches.append((features, int(update_chunk)))
            if max_update_chunk is None or int(update_chunk) > max_update_chunk:
                max_update_chunk = int(update_chunk)

        if not candidate_batches or max_update_chunk is None:
            return None

        candidate_batches = [
            item for item in candidate_batches if item[1] == max_update_chunk
        ]
        if len(candidate_batches) == 1:
            return candidate_batches[0][0]

        selected_features, _ = max(
            candidate_batches,
            key=lambda item: self._get_runtime_feature_selection_score(
                item[0],
                use_learned_theta=(
                    self.alp_scheduler is not None
                    and self.alp_scheduler.interference_updates >= warmup_updates
                ),
            ),
        )
        return selected_features

    def _get_runtime_update_chunk_from_tokens(self, observed_tokens: int):
        if self.alp_scheduler is None:
            return None
        candidate_chunks = [int(chunk) for chunk in self.alp_scheduler.candidate_chunks]
        if not candidate_chunks:
            return None
        observed_tokens = int(observed_tokens)
        if observed_tokens <= 0:
            sched = self._sched
            return (
                int(sched.chunked_prefill_size)
                if sched.chunked_prefill_size is not None
                else None
            )
        for chunk in candidate_chunks:
            if observed_tokens <= chunk:
                return chunk
        return int(candidate_chunks[-1])

    def _get_runtime_feature_selection_score(self, features, use_learned_theta: bool):
        if not use_learned_theta or self.alp_scheduler is None:
            return float(features[1]) + float(features[2]) + float(features[3])
        return float(
            np.dot(
                self.alp_scheduler.interference_theta,
                self.alp_scheduler._build_interference_features(
                    decode_ctx_sum=float(features[1]),
                    prefill_square_sum=float(features[2]),
                    prefill_prefix_prod_sum=float(features[3]),
                ),
            )
        )

    def _record_runtime_exec_sample(
        self,
        start_time: float,
        runtime_feature_snapshot,
        batch,
    ) -> None:
        if runtime_feature_snapshot is None:
            return
        exec_time = float(time.perf_counter() - start_time)
        if exec_time <= 0.0:
            return

        running_token_max = int(runtime_feature_snapshot[0])
        if running_token_max <= 0:
            return

        running_token_bucket = ((running_token_max - 1) // 128 + 1) * 128
        input_token_count = (
            len(batch.input_ids)
            if batch is not None and batch.input_ids is not None
            else 0
        )
        input_token_bucket = ((input_token_count - 1) // 128 + 1) * 128

        if running_token_bucket != input_token_bucket:
            return

        update_chunk = self._get_runtime_update_chunk_from_tokens(running_token_max)
        if update_chunk is None:
            return

        if self.alp_scheduler is not None:
            self.alp_scheduler.update_runtime_overhead(
                exec_time,
                runtime_chunk_size=update_chunk,
                decode_ctx_sum=float(runtime_feature_snapshot[1]),
                prefill_square_sum=float(runtime_feature_snapshot[2]),
                prefill_prefix_prod_sum=float(runtime_feature_snapshot[3]),
            )

        self.exec_time_update_sum += exec_time
        self.exec_time_update_count += 1

    # ------------------------------------------------------------------
    # ALP candidate-level feature simulation (ported from sglang)
    # ------------------------------------------------------------------

    def _get_runtime_prediction_context(self):
        sched = self._sched
        cur_idx = sched.cur_pp.get()
        stage_order = self._get_prediction_stage_order(cur_idx)
        stage_decode_ctx_sum = {}
        stage_mixed_decode_tokens = {}
        stage_chunked_lens = {}

        for stage_idx in stage_order:
            stage_batch = (
                sched._last_batch[stage_idx]
                if 0 <= stage_idx < sched.pp_size
                else None
            )
            decode_reqs = self._get_decode_reqs_for_runtime_prediction(stage_batch)
            stage_decode_ctx_sum[stage_idx] = float(
                sum(
                    len(req.origin_input_ids) + len(req.output_ids)
                    for req in decode_reqs
                )
            )
            stage_mixed_decode_tokens[stage_idx] = (
                len(decode_reqs) if sched.is_mixed_chunk else 0
            )

            chunked_req = self._get_prediction_chunked_req(stage_idx)
            stage_chunked_lens[stage_idx] = (
                self._get_chunked_req_prefill_lens_for_prediction(chunked_req)
                if chunked_req is not None
                else None
            )

        waiting_req_meta = []
        for req in sched.waiting_queue:
            prefix_len, prefill_len = self._get_waiting_req_prefill_lens_for_prediction(
                req
            )
            cannot_chunk_for_logprob = (
                req.return_logprob
                and req.normalized_prompt_logprob is None
                and req.logprob_start_len != len(req.origin_input_ids) - 1
            )
            waiting_req_meta.append(
                (int(prefix_len), int(prefill_len), bool(cannot_chunk_for_logprob))
            )

        return {
            "stage_order": stage_order,
            "stage_decode_ctx_sum": stage_decode_ctx_sum,
            "stage_mixed_decode_tokens": stage_mixed_decode_tokens,
            "stage_chunked_lens": stage_chunked_lens,
            "waiting_req_meta": waiting_req_meta,
        }

    def _get_prediction_stage_order(self, cur_idx: int):
        sched = self._sched
        return [((cur_idx + offset) % sched.pp_size) for offset in range(sched.pp_size)]

    def _get_decode_reqs_for_runtime_prediction(self, batch):
        if batch is None:
            return []
        if batch.forward_mode.is_decode():
            return list(batch.reqs)
        return list(batch.decoding_reqs or [])

    def _get_prediction_chunked_req(self, cur_idx: int):
        sched = self._sched
        chunked_req = sched._being_chunked_req[cur_idx]
        if chunked_req is not None and chunked_req.pp_idx == cur_idx:
            return chunked_req
        if sched.pp_size == 1:
            return None
        prev_idx = (cur_idx - 1) % sched.pp_size
        prev_req = sched._being_chunked_req[prev_idx]
        if prev_req is not None and prev_req.pp_idx == prev_idx:
            return prev_req
        return None

    def _get_waiting_req_prefill_lens_for_prediction(self, req):
        fill_len = (
            len(req.fill_ids)
            if req.fill_ids is not None
            else len(req.origin_input_ids) + len(req.output_ids)
        )
        prefix_len = len(req.prefix_indices)
        return prefix_len, max(fill_len - prefix_len, 0)

    def _get_chunked_req_prefill_lens_for_prediction(self, req):
        total_len = len(req.origin_input_ids) + len(req.output_ids)
        prefix_len = (
            len(req.fill_ids) if req.fill_ids is not None else len(req.prefix_indices)
        )
        return prefix_len, max(total_len - prefix_len, 0)

    def _get_candidate_runtime_prediction_features_for_chunk(
        self, chunk, prediction_context
    ):
        stage_order = prediction_context["stage_order"]
        stage_decode_ctx_sum = prediction_context["stage_decode_ctx_sum"]
        stage_mixed_decode_tokens = prediction_context["stage_mixed_decode_tokens"]
        stage_chunked_lens = prediction_context["stage_chunked_lens"]
        waiting_req_meta = prediction_context["waiting_req_meta"]
        queue_req_idx = 0
        queue_req_prefill_offset = 0
        stage_feature_candidates = []

        for stage_idx in stage_order:
            decode_ctx_sum = stage_decode_ctx_sum.get(stage_idx, 0.0)
            mixed_decode_tokens = stage_mixed_decode_tokens.get(stage_idx, 0)
            remaining_chunk_tokens = max(int(chunk) - mixed_decode_tokens, 0)
            prefill_square_sum = 0
            prefill_prefix_prod_sum = 0

            chunked_lens = stage_chunked_lens.get(stage_idx)
            if chunked_lens is not None and remaining_chunk_tokens > 0:
                prefix_len, remain_prefill = chunked_lens
                extend_len = min(remain_prefill, remaining_chunk_tokens)
                if extend_len > 0:
                    prefill_square_sum += extend_len * extend_len
                    prefill_prefix_prod_sum += extend_len * prefix_len
                    remaining_chunk_tokens -= extend_len

                if extend_len < remain_prefill:
                    stage_feature_candidates.append(
                        {
                            "running_token_max": float(
                                extend_len + mixed_decode_tokens
                            ),
                            "selected_decode_ctx_sum": float(decode_ctx_sum),
                            "prefill_square_sum": float(prefill_square_sum),
                            "prefill_prefix_prod_sum": float(
                                prefill_prefix_prod_sum
                            ),
                        }
                    )
                    continue

            packed_prefill_tokens = max(
                int(chunk) - mixed_decode_tokens - remaining_chunk_tokens, 0
            )
            local_req_idx = queue_req_idx
            local_req_prefill_offset = queue_req_prefill_offset
            while (
                local_req_idx < len(waiting_req_meta)
                and remaining_chunk_tokens > 0
            ):
                base_prefix_len, base_prefill_len, cannot_chunk_for_logprob = (
                    waiting_req_meta[local_req_idx]
                )
                consumed_offset = max(int(local_req_prefill_offset), 0)
                prefix_len = base_prefix_len + consumed_offset
                prefill_len = max(base_prefill_len - consumed_offset, 0)
                if prefill_len <= 0:
                    local_req_idx += 1
                    local_req_prefill_offset = 0
                    continue

                if (
                    prefill_len > remaining_chunk_tokens
                    and cannot_chunk_for_logprob
                ):
                    if packed_prefill_tokens > 0:
                        break
                    extend_len = prefill_len
                else:
                    extend_len = min(prefill_len, remaining_chunk_tokens)

                prefill_square_sum += extend_len * extend_len
                prefill_prefix_prod_sum += extend_len * prefix_len
                packed_prefill_tokens += extend_len
                remaining_chunk_tokens -= min(extend_len, remaining_chunk_tokens)

                if extend_len >= prefill_len:
                    local_req_idx += 1
                    local_req_prefill_offset = 0
                else:
                    local_req_prefill_offset += extend_len
                    break

            queue_req_idx = local_req_idx
            queue_req_prefill_offset = local_req_prefill_offset
            stage_feature_candidates.append(
                {
                    "running_token_max": float(
                        packed_prefill_tokens + mixed_decode_tokens
                    ),
                    "selected_decode_ctx_sum": float(decode_ctx_sum),
                    "prefill_square_sum": float(prefill_square_sum),
                    "prefill_prefix_prod_sum": float(prefill_prefix_prod_sum),
                }
            )

        if not stage_feature_candidates:
            return {
                "running_token_max": 0.0,
                "selected_decode_ctx_sum": 0.0,
                "prefill_square_sum": 0.0,
                "prefill_prefix_prod_sum": 0.0,
            }

        stage_count = float(len(stage_feature_candidates))
        return {
            "running_token_max": float(
                sum(
                    float(features["running_token_max"])
                    for features in stage_feature_candidates
                )
            ) / stage_count,
            "selected_decode_ctx_sum": float(
                sum(
                    float(features["selected_decode_ctx_sum"])
                    for features in stage_feature_candidates
                )
            ) / stage_count,
            "prefill_square_sum": float(
                sum(
                    float(features["prefill_square_sum"])
                    for features in stage_feature_candidates
                )
            ) / stage_count,
            "prefill_prefix_prod_sum": float(
                sum(
                    float(features["prefill_prefix_prod_sum"])
                    for features in stage_feature_candidates
                )
            ) / stage_count,
        }
