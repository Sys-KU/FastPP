"""Prefill server side of PD disaggregation."""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.disaggregation.utils import DisaggRebootstrapReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerDisaggregationPrefillMixin:

    def disagg_prefill_add(self: "Scheduler", req: Req):
        # The prefill server only generates the first token; the decode server
        # generates the rest. This also keeps the memory estimate of PrefillAdder exact.
        req.sampling_params.max_new_tokens = 1
        # Prefill at once; the KV is sent when the decode server has preallocated it,
        # which rank 0 waits for until the timeout.
        if req.disagg_internal:
            # A retracted request goes before the new ones (in FCFS order), and it is
            # not an arrival.
            i = next(
                (k for k, r in enumerate(self.waiting_queue) if not r.disagg_internal),
                len(self.waiting_queue),
            )
            self.waiting_queue.insert(i, req)
        else:
            self.waiting_queue.append(req)
            self.qps_queue.append(req)
        if self.disagg_is_rank0:
            room = req.bootstrap_room
            if room in self.disagg_early_ready:
                del self.disagg_early_ready[room]
            else:
                self.disagg_awaiting_ready[room] = req.disagg_enqueue_time

    def disagg_prefill_rebootstrap(self: "Scheduler", obj: DisaggRebootstrapReq):
        """A request that the decode server retracted: prefill its prompt and outputs
        so far, like a new PD request, but stream nothing (stream_output skips it)."""
        self.handle_generate_request(obj.req, disagg_internal=True)

    def _disagg_remove_waiting(self: "Scheduler", match) -> Optional[Req]:
        """Remove the first PD request of the waiting queue that matches."""
        req = next((r for r in self.waiting_queue if r.disagg_pd and match(r)), None)
        if req is not None:
            self.waiting_queue.remove(req)
        return req

    def disagg_prefill_on_finish(self: "Scheduler", req: Req):
        """Called on every rank when the prefill of a PD request finishes on it.

        The KV stays locked in the cache and the output is held back until every
        rank has sent its part of the KV.
        """
        if req.finished_reason is None:
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        self.tree_cache.cache_unfinished_req(req)

        room = req.bootstrap_room
        reason = self.disagg_failed_rooms.pop(room, None)
        if reason is not None:
            # The request failed on another rank while it was prefilled here; the
            # caller streams the abort with the rest of the batch.
            self.tree_cache.cache_finished_req(req)
            self.disagg_sender.cancel(room, reason)
            req.finished_reason = FINISH_ABORT(f"PD disaggregation failed: {reason}")
            return

        # The rows not sent chunk by chunk: a cached prefix and the last chunk.
        num_tokens = len(req.origin_input_ids)
        if req.disagg_chunk_begin is None:
            ranges = [(0, num_tokens)]
        else:
            ranges = [(req.disagg_chunk_end, num_tokens)]
            if req.disagg_chunk_begin > 0:
                ranges.insert(0, (0, req.disagg_chunk_begin))
        rows = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        parts = [(start, rows[start:end].to(torch.int64)) for start, end in ranges]
        event = self.disagg_ready_event()
        first_token = req.output_ids[-1]
        if self.disagg_chunked_send:
            for k, (start, src) in enumerate(parts):
                last = k == len(parts) - 1
                self.disagg_sender.send_part(
                    room, start, src, event, last, first_token if last else None
                )
        else:
            self.disagg_sender.send(room, parts[0][1], first_token, event)
        req.disagg_transfer_pending = True
        self.disagg_inflight_reqs[room] = req

    def disagg_prefill_on_chunk(self: "Scheduler", req: Req, batch, i: int):
        """Called on every rank when a chunk of a chunked PD prefill finishes on it:
        send its KV while the next chunks are computed (direct transfer backends)."""
        if not self.disagg_chunked_send or req.disagg_chunk_disabled:
            return
        # The rows and KV slots of the chunk are in the batch, unless it was filtered.
        start, num = batch.prefix_lens[i], batch.extend_lens[i]
        if (
            batch.out_cache_loc is None
            or len(batch.extend_lens) != len(batch.reqs)
            or (req.disagg_chunk_end is not None and start != req.disagg_chunk_end)
        ):
            req.disagg_chunk_disabled = True
            return
        offset = sum(batch.extend_lens[:i])
        src = batch.out_cache_loc[offset : offset + num].to(torch.int64)
        self.disagg_sender.send_part(
            req.bootstrap_room, start, src, self.disagg_ready_event(), last=False
        )
        if req.disagg_chunk_begin is None:
            req.disagg_chunk_begin = start
        req.disagg_chunk_end = start + num

    def disagg_prefill_sent(self: "Scheduler", room: int):
        """Every rank has sent the KV: release it and return the request."""
        req = self.disagg_inflight_reqs.pop(room, None)
        if req is None:
            return
        req.disagg_transfer_pending = False
        self.tree_cache.cache_finished_req(req)
        self.stream_output([req], req.return_logprob)

    def disagg_prefill_abort(self: "Scheduler", room: int, reason: str) -> Optional[Req]:
        """Abort a failed request; the decode server is told to drop its KV.

        Failures happen before the decode server gets any KV (a timeout before it
        preallocated, or prompts of different lengths): the sender sends nothing more
        and tells the decode server once its pending copies, which only read the KV
        freed here, have completed.
        """
        self.disagg_sender.cancel(room, reason)
        if self.disagg_is_rank0:
            self.disagg_awaiting_ready.pop(room, None)
        req = self._disagg_remove_waiting(lambda r: r.bootstrap_room == room)
        if req is not None:
            return req
        req = self.disagg_inflight_reqs.pop(room, None)
        if req is not None:
            req.disagg_transfer_pending = False
            self.tree_cache.cache_finished_req(req)
            return req
        # The prefill has not finished on this rank yet.
        self.disagg_failed_rooms[room] = reason
        return None

    def disagg_prefill_abort_request(self: "Scheduler", rid: str) -> bool:
        """A client abort. Requests are dropped before their prefill starts; later
        they finish quickly anyway."""
        req = self._disagg_remove_waiting(lambda r: r.rid == rid)
        if req is None:
            return False
        room = req.bootstrap_room
        self.disagg_sender.cancel(room, "aborted")
        if self.disagg_is_rank0:
            self.disagg_awaiting_ready.pop(room, None)
            self.disagg_failed_history.add(room)
        self.disagg_finish_aborted(req, "aborted")
        return True
