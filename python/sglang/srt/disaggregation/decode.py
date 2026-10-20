"""Decode server side of PD disaggregation."""

import copy
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.utils import (
    DisaggRebootstrapReq,
    get_transfer_port,
    next_bootstrap_room,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS_ESTIMATION
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def prepare_for_prebuilt_decode(batch: ScheduleBatch):
    """Fill the fields of a batch whose prompt KV was written by the prefill
    server, as if it had just finished a prefill, so that it can be merged
    into a running batch."""
    reqs = batch.reqs
    batch.forward_mode = ForwardMode.DECODE
    batch.req_pool_indices = torch.tensor(
        [req.req_pool_idx for req in reqs], dtype=torch.int32
    ).to(batch.device, non_blocking=True)
    # The KV received: the prompt, and the outputs of a retracted request
    seq_lens = [len(req.fill_ids) for req in reqs]
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(
        batch.device, non_blocking=True
    )
    batch.seq_lens_sum = sum(seq_lens)
    # The first token sampled by the prefill server is the input of the next decode.
    batch.output_ids = torch.tensor(
        [req.output_ids[-1] for req in reqs], dtype=torch.int32
    ).to(batch.device, non_blocking=True)
    if batch.return_logprob:
        batch.top_logprobs_nums = [req.top_logprobs_num for req in reqs]
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch,
        batch.model_config.vocab_size,
        enable_overlap_schedule=batch.enable_overlap,
    )


class SchedulerDisaggregationDecodeMixin:

    def disagg_decode_add(self: "Scheduler", req: Req):
        req.disagg_kv_len = len(req.origin_input_ids)
        self.disagg_prealloc_queue.append(req)

    def disagg_decode_rebootstrap(self: "Scheduler", reqs: List[Req]):
        """Send requests that retract_decode retracted (their KV is freed) back to the
        prefill server instead of prefilling them here: it computes the KV of the
        prompt and the outputs so far and samples the next token. They join the
        prealloc queue after the retracted requests already there and before the new
        requests.

        Every rank does the same (the retraction is replicated); rank 0 sends.
        Requests that did not come from a prefill server are prefilled here.
        """
        rebootstrapped = []
        for req in reqs:
            if req.bootstrap_room is None:
                self.waiting_queue.append(req)
                continue
            req.disagg_rebootstraps += 1
            req.bootstrap_room = next_bootstrap_room(req.bootstrap_room)
            req.disagg_pd = True
            req.disagg_kv_len = len(req.origin_input_ids) + len(req.output_ids)
            req.disagg_enqueue_time = time.time()
            rebootstrapped.append(req)
            if not self.disagg_is_rank0:
                continue
            sampling_params = copy.copy(req.sampling_params)
            # The outputs so far count for min_new_tokens.
            sampling_params.min_new_tokens = max(
                0, sampling_params.min_new_tokens - len(req.output_ids)
            )
            self.disagg_send_rebootstrap(
                DisaggRebootstrapReq(
                    TokenizedGenerateReqInput(
                        rid=f"{req.rid}-retract{req.disagg_rebootstraps}",
                        input_text=None,
                        input_ids=req.origin_input_ids + req.output_ids,
                        image_inputs=None,
                        sampling_params=sampling_params,
                        return_logprob=False,
                        logprob_start_len=-1,
                        top_logprobs_num=0,
                        stream=False,
                        lora_path=req.lora_path,
                        bootstrap_host=req.bootstrap_host,
                        bootstrap_port=req.bootstrap_port
                        or self.server_args.disaggregation_bootstrap_port,
                        bootstrap_room=req.bootstrap_room,
                    )
                )
            )
        queue = self.disagg_prealloc_queue
        i = next((k for k, r in enumerate(queue) if not r.disagg_rebootstraps), len(queue))
        queue[i:i] = rebootstrapped

    def _disagg_decode_num_reqs(self: "Scheduler") -> int:
        """Requests that hold or will need KV, counted from replicated state."""
        num = sum(len(batch.reqs) for batch in self._running_batch if batch is not None)
        # Prefills of retracted requests that are not merged yet
        num += sum(
            len(batch.reqs)
            for batch in self._last_batch
            if batch is not None and batch.forward_mode.is_extend()
        )
        if self.pending_batch is not None:
            num += len(self.pending_batch.reqs)
        return (
            num
            + len(self.waiting_queue)
            + len(self.disagg_transfer_reqs)
            + len(self.disagg_ready_reqs)
            + len(self.disagg_cancelling)
        )

    def disagg_decode_prealloc(self: "Scheduler"):
        """Preallocate the prompt KV of queued requests in FIFO order (the
        retracted requests sent back to the prefill server first).

        The decisions only depend on replicated state, so every rank preallocates
        the same requests at the same iteration. The local free lists cannot be
        used: each PP stage frees finished requests at a different iteration.
        As in upstream SGLang, every request keeps a reserve of tokens for its
        decoding, and the longest running request must still be able to finish
        (up to CLIP_MAX_NEW_TOKENS_ESTIMATION new tokens) if every other running
        request is retracted: the requests being transferred cannot be.
        """
        if not self.disagg_prealloc_queue:
            return
        reserved = self.server_args.disaggregation_decode_reserved_tokens
        clip = CLIP_MAX_NEW_TOKENS_ESTIMATION
        num_reqs = self._disagg_decode_num_reqs()
        running = [r for b in self._running_batch if b is not None for r in b.reqs]
        retractable = sum(len(r.origin_input_ids) + len(r.output_ids) for r in running)
        need_single = (
            max(
                len(r.origin_input_ids) + min(r.sampling_params.max_new_tokens, clip)
                for r in running
            )
            - retractable
            if running
            else 0
        )
        # The token budget comes from rank 0 (the first PP stage), which frees
        # finished requests last and therefore has the fewest free tokens.
        budget = self.available_size - max(reserved * num_reqs, need_single)
        num_free_req_slots = self.req_to_token_pool.size - num_reqs
        allocated = 0
        num_started = 0
        for req in self.disagg_prealloc_queue:
            num_tokens = req.disagg_kv_len
            left = req.sampling_params.max_new_tokens - len(req.output_ids)
            need = max(
                num_tokens + min(reserved, max(self.max_total_num_tokens - num_tokens, 0)),
                num_tokens + min(left, clip) - retractable,
            )
            if num_free_req_slots <= 0 or need > budget:
                break

            req.req_pool_idx = self.req_to_token_pool.alloc(1)[0]
            kv_loc = self.token_to_kv_pool.alloc(num_tokens)
            if kv_loc is None:
                raise RuntimeError("Out of KV cache while preallocating a PD request.")
            self.req_to_token_pool.write((req.req_pool_idx, slice(0, num_tokens)), kv_loc)
            # The prompt, and the outputs of a retracted request
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.prefix_indices = []
            req.extend_input_len = num_tokens
            req.disagg_prealloc_time = time.time()

            # Single node: the prefill server is on this machine whatever
            # bootstrap_host says.
            port = get_transfer_port(
                req.bootstrap_port or self.server_args.disaggregation_bootstrap_port,
                self.pp_rank,
                self.tp_rank,
                self.tp_size,
            )
            self.disagg_receiver.receive(
                req.bootstrap_room,
                port,
                kv_loc.to(torch.int64),
                self.disagg_ready_event(),
            )
            self.disagg_transfer_reqs[req.bootstrap_room] = req

            budget -= need
            num_free_req_slots -= 1
            allocated += num_tokens
            num_started += 1

        self.disagg_prealloc_queue = self.disagg_prealloc_queue[num_started:]
        # Later decisions of this iteration must see the preallocated tokens as used.
        self.available_size -= allocated

    def disagg_decode_received(self: "Scheduler", room: int, first_token: int):
        req = self.disagg_transfer_reqs.pop(room, None)
        if req is not None:
            req.disagg_first_token = first_token
            self.disagg_ready_reqs.append(req)

    def disagg_decode_merge_ready(self: "Scheduler"):
        """Merge the requests whose prompt KV has arrived into the running batch
        of the current micro-batch."""
        if not self.disagg_ready_reqs:
            return
        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        num = min(len(self.disagg_ready_reqs), self.max_running_requests - running_bs)
        if num <= 0:
            return
        reqs = self.disagg_ready_reqs[:num]
        self.disagg_ready_reqs = self.disagg_ready_reqs[num:]

        now = time.perf_counter()
        decoding = []
        for req in reqs:
            # From now on it is an ordinary running request (when retracted, it is
            # sent back to the prefill server again).
            req.disagg_pd = False
            # The first token, or the next one of a retracted request
            first = not req.output_ids
            req.output_ids.append(req.disagg_first_token)
            req.output_ids_len = len(req.output_ids)
            req.pp_idx = self.cur_pp.get()
            if req.start_time is None:
                req.start_time = now
            if first:
                req.ttft = now - req.arrival_time
            if req.return_logprob:
                # The logprobs of the prompt and of the tokens sampled by the
                # prefill server stay there.
                if first:
                    req.input_token_logprobs_val = []
                    req.input_token_logprobs_idx = []
                req.output_token_logprobs_val.append(None)
                req.output_token_logprobs_idx.append(req.disagg_first_token)
                if req.top_logprobs_num > 0:
                    if first:
                        req.input_top_logprobs_val = []
                        req.input_top_logprobs_idx = []
                    req.output_top_logprobs_val.append(None)
                    req.output_top_logprobs_idx.append(None)

            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)
            else:
                self.tree_cache.cache_unfinished_req(req)
                decoding.append(req)

        if decoding:
            batch = ScheduleBatch.init_new(
                decoding,
                self.req_to_token_pool,
                self.token_to_kv_pool,
                self.tree_cache,
                self.model_config,
                self.enable_overlap,
                self.cur_pp.get(),
            )
            prepare_for_prebuilt_decode(batch)
            if self.running_batch is None or self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)

        # Stream the first token.
        self.stream_output(reqs, any(req.return_logprob for req in reqs))

    def disagg_decode_cancel(self: "Scheduler", room: int, reason: str):
        """Stop the transfer of a room. Its KV stays allocated until every rank
        reports that it no longer writes it (see disagg_decode_release)."""
        req = self.disagg_transfer_reqs.pop(room, None)
        if req is None:
            return
        self.disagg_cancelling[room] = (req, reason)
        self.disagg_receiver.cancel(room)
        if self.disagg_is_rank0:
            self.disagg_consensus.discard(room)

    def disagg_decode_release(self: "Scheduler", room: int) -> Optional[tuple]:
        """Free the preallocated KV of a cancelled room."""
        entry = self.disagg_cancelling.pop(room, None)
        if entry is None:
            return None
        req, _ = entry
        num_tokens = req.disagg_kv_len
        self.token_to_kv_pool.free(
            self.req_to_token_pool.req_to_token[req.req_pool_idx, :num_tokens]
        )
        self.req_to_token_pool.free(req.req_pool_idx)
        return entry

    def disagg_decode_abort_request(self: "Scheduler", rid: str) -> bool:
        """A client abort of a request that is not in a batch yet."""
        for i, req in enumerate(self.disagg_prealloc_queue):
            if req.rid == rid:
                del self.disagg_prealloc_queue[i]
                self.disagg_finish_aborted(req, "aborted")
                return True
        for room, req in self.disagg_transfer_reqs.items():
            if req.rid == rid:
                self.disagg_decode_cancel(room, "aborted")
                return True
        for req in self.disagg_ready_reqs:
            if req.rid == rid:
                # Finished (and freed) when it is merged.
                req.to_abort = True
                return True
        return False
