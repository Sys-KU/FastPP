"""
Scheduler logic shared by the prefill and decode servers of PD disaggregation.

All scheduler ranks (every PP stage and TP rank) of a server replicate the same
scheduling decisions. KV transfers, however, progress independently on every
rank, so each rank reports its transfer events to rank 0, and rank 0 turns them
into a DisaggControl that is broadcast with the requests of an iteration. Every
rank then applies the same decision at the same iteration.

Life cycle of a PD request (one with a bootstrap room):
    prefill: waiting queue -> prefill -> inflight (KV locked) -(all ranks sent)
             -> finished. The prefill starts without waiting for the decode server;
             the KV is sent once the decode server has preallocated it.
    decode:  prealloc queue -(KV preallocated)-> transfer
             -(all ranks received)-> ready -> merged into a running batch
A prefill request that fails before its KV is sent tells the decode server,
which cancels the transfer and frees the preallocated KV once no rank writes it.

A request that the decode server retracts (out of KV) is not prefilled again
there: rank 0 sends it back to rank 0 of its prefill server with the outputs so
far appended to the prompt and a new room, the prefill server computes that KV
and samples the next token, and the decode server preallocates it before the new
requests, as a new request with a longer prompt. Its client stream goes on.
"""

import collections
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
import zmq

from sglang.srt.disaggregation.transfer import KVReceiver, KVSender
from sglang.srt.disaggregation.transfer_direct import (
    DIRECT_MODES,
    DirectKVReceiver,
    DirectKVSender,
)
from sglang.srt.disaggregation.utils import (
    DisaggControl,
    DisaggregationMode,
    DisaggRebootstrapReq,
    TransferConsensus,
    get_transfer_port,
    rebootstrap_endpoint,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.utils import get_zmq_socket

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import PortArgs

# Upper bound of an idle wait; requests and transfer reports end it at once.
_IDLE_WAIT_MS = 1
# READY reports kept for requests that have not reached the scheduler yet
_MAX_EARLY_READY = 100_000


class SchedulerDisaggregationMixin:

    def init_disaggregation(self: "Scheduler", port_args: "PortArgs", gpu_id: int):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        if self.disaggregation_mode == DisaggregationMode.NULL:
            return

        args = self.server_args
        rank = self.pp_rank * self.tp_size + self.tp_rank
        world_size = self.pp_size * self.tp_size
        self.disagg_is_rank0 = rank == 0
        world_info = {
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "pp_size": self.pp_size,
            "tp_size": self.tp_size,
        }
        report_endpoint = port_args.disagg_report_ipc_name

        if self.disagg_is_rank0:
            self.disagg_zmq_context = zmq.Context(1)
            self.disagg_reports = get_zmq_socket(
                self.disagg_zmq_context, zmq.PULL, report_endpoint
            )
            self.disagg_consensus = TransferConsensus(world_size)
            self.disagg_cancel_consensus = TransferConsensus(world_size)
            # Prefill: room -> enqueue time of the requests whose decode server has
            # not preallocated their KV yet, and rooms preallocated before their
            # request came (an ordered set).
            self.disagg_awaiting_ready: Dict[int, float] = {}
            self.disagg_early_ready = collections.OrderedDict()
            # Every rank reports a failure, but it is handled only once.
            self.disagg_failed_history: Set[int] = set()
            self.disagg_idle_poller = zmq.Poller()
            self.disagg_idle_poller.register(self.recv_from_tokenizer, zmq.POLLIN)
            self.disagg_idle_poller.register(self.disagg_reports, zmq.POLLIN)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # Requests that decode servers retracted come back here.
                self.disagg_rebootstrap_socket = get_zmq_socket(
                    self.disagg_zmq_context,
                    zmq.PULL,
                    rebootstrap_endpoint(args.disaggregation_bootstrap_port),
                )
                self.disagg_idle_poller.register(
                    self.disagg_rebootstrap_socket, zmq.POLLIN
                )
            else:
                # bootstrap port of a prefill server -> socket to its rank 0
                self.disagg_rebootstrap_sockets: Dict[int, zmq.Socket] = {}

        backend = args.disaggregation_transfer_backend
        # The direct backends send the KV of a chunked prefill chunk by chunk.
        self.disagg_chunked_send = backend in DIRECT_MODES
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_inflight_reqs: Dict[int, Req] = {}
            # Rooms that failed before their prefill finished on this rank.
            self.disagg_failed_rooms: Dict[int, str] = {}
            port = get_transfer_port(
                args.disaggregation_bootstrap_port,
                self.pp_rank,
                self.tp_rank,
                self.tp_size,
            )
            if backend in DIRECT_MODES:
                self.disagg_sender = DirectKVSender(
                    self.token_to_kv_pool,
                    gpu_id,
                    rank,
                    world_info,
                    port,
                    report_endpoint,
                    report_ready=self.disagg_is_rank0,
                    mode=backend,
                )
            else:
                self.disagg_sender = KVSender(
                    self.token_to_kv_pool,
                    gpu_id,
                    rank,
                    world_info,
                    port,
                    report_endpoint,
                    report_ready=self.disagg_is_rank0,
                    chunk_tokens=args.disaggregation_transfer_chunk_tokens,
                    num_slots=args.disaggregation_staging_slots,
                )
        else:
            self.disagg_prealloc_queue: List[Req] = []
            self.disagg_transfer_reqs: Dict[int, Req] = {}
            self.disagg_ready_reqs: List[Req] = []
            # room -> (request, reason): waiting for every rank to stop receiving
            self.disagg_cancelling: Dict[int, tuple] = {}
            if backend in DIRECT_MODES:
                self.disagg_receiver = DirectKVReceiver(
                    self.token_to_kv_pool,
                    gpu_id,
                    rank,
                    world_info,
                    report_endpoint,
                    mode=backend,
                )
            else:
                self.disagg_receiver = KVReceiver(
                    self.token_to_kv_pool,
                    gpu_id,
                    rank,
                    world_info,
                    report_endpoint,
                    chunk_tokens=args.disaggregation_transfer_chunk_tokens,
                    num_slots=args.disaggregation_staging_slots,
                    backend=backend,
                )

    def disagg_add_request(self: "Scheduler", req: Req) -> bool:
        """Take a request with a bootstrap room into the PD queues.

        Returns False if the request should go to the normal waiting queue.
        """
        if (
            self.disaggregation_mode == DisaggregationMode.NULL
            or req.bootstrap_room is None
        ):
            return False

        if req.grammar is not None:
            req.grammar = None
            req.finished_reason = FINISH_ABORT(
                "Constrained decoding is not supported for PD disaggregated requests."
            )
            self.waiting_queue.append(req)
            return True

        req.disagg_pd = True
        req.disagg_enqueue_time = time.time()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_add(req)
        else:
            self.disagg_decode_add(req)
        return True

    def disagg_abort_request(self: "Scheduler", rid: str) -> bool:
        """Abort a PD request that the normal abort logic does not see.

        Returns True if the request was found.
        """
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return self.disagg_prefill_abort_request(rid)
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return self.disagg_decode_abort_request(rid)
        return False

    def disagg_has_pending(self: "Scheduler") -> bool:
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return bool(self.disagg_inflight_reqs)
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return bool(
                self.disagg_prealloc_queue
                or self.disagg_transfer_reqs
                or self.disagg_ready_reqs
                or self.disagg_cancelling
            )
        return False

    def disagg_can_flush(self: "Scheduler") -> bool:
        """Whether the cache can be flushed, decided from replicated state."""
        if self.disaggregation_mode == DisaggregationMode.NULL:
            return True
        # The PP stages finish a prefill at different iterations, so the PD state
        # is only the same on every rank once no batch is in flight.
        return (
            all(batch is None for batch in self._last_batch)
            and not self.disagg_has_pending()
        )

    def disagg_idle_wait(self: "Scheduler"):
        """Called when the scheduler loop has nothing to run.

        The loop polls its sockets without blocking, so an idle scheduler releases
        the GIL only for a few microseconds at a time and the KV transfer thread of
        the process can wait seconds to run a chunk. Rank 0 waits for a request or a
        transfer report instead; the other ranks then wait for rank 0 in
        recv_requests.
        """
        if not self.disagg_is_rank0 or any(b is not None for b in self._last_batch):
            return
        self.disagg_idle_poller.poll(_IDLE_WAIT_MS)

    def disagg_recv_rebootstrap(self: "Scheduler") -> List[DisaggRebootstrapReq]:
        """Prefill rank 0: the retracted requests that decode servers sent back."""
        reqs = []
        while True:
            try:
                reqs.append(self.disagg_rebootstrap_socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.ZMQError:
                return reqs

    def disagg_send_rebootstrap(self: "Scheduler", obj: DisaggRebootstrapReq):
        """Decode rank 0: send a retracted request to rank 0 of its prefill server."""
        port = obj.req.bootstrap_port
        socket = self.disagg_rebootstrap_sockets.get(port)
        if socket is None:
            socket = get_zmq_socket(
                self.disagg_zmq_context, zmq.PUSH, rebootstrap_endpoint(port)
            )
            self.disagg_rebootstrap_sockets[port] = socket
        socket.send_pyobj(obj)

    def disagg_build_control(self: "Scheduler") -> Optional[DisaggControl]:
        """Rank 0: turn the transfer reports of all ranks into decisions."""
        is_decode = self.disaggregation_mode == DisaggregationMode.DECODE
        failures: Dict[int, str] = {}
        while True:
            try:
                msg = self.disagg_reports.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            kind = msg[0]
            if kind == "ready":
                room = msg[1]
                if self.disagg_awaiting_ready.pop(room, None) is None:
                    self.disagg_early_ready[room] = None
                    while len(self.disagg_early_ready) > _MAX_EARLY_READY:
                        self.disagg_early_ready.popitem(last=False)
            elif kind == "done":
                _, rank, room, first_token = msg
                if is_decode and room not in self.disagg_transfer_reqs:
                    continue  # Cancelled
                self.disagg_consensus.add(room, rank, first_token)
            elif kind == "cancelled":
                self.disagg_cancel_consensus.add(msg[2], msg[1], None)
            elif kind == "fail":
                if msg[1] not in self.disagg_failed_history:
                    self.disagg_failed_history.add(msg[1])
                    failures[msg[1]] = msg[2]
            elif kind == "crash":
                raise RuntimeError(f"The KV transfer thread crashed:\n{msg[1]}")

        ctrl = DisaggControl()
        deadline = time.time() - self.server_args.disaggregation_timeout
        if is_decode:
            # A failed or timed out transfer is cancelled on every rank first.
            cancel = {
                room: reason
                for room, reason in failures.items()
                if room in self.disagg_transfer_reqs
            }
            for room, req in self.disagg_transfer_reqs.items():
                if req.disagg_prealloc_time < deadline and room not in cancel:
                    cancel[room] = "timed out waiting for the KV cache"
            ctrl.cancel = list(cancel.items())
            ctrl.received = [
                (room, first_token)
                for room, first_token in self.disagg_consensus.pop_complete()
                if room in self.disagg_transfer_reqs and room not in cancel
            ]
            ctrl.released = [
                room for room, _ in self.disagg_cancel_consensus.pop_complete()
            ]
        else:
            for room, reason in failures.items():
                self.disagg_consensus.discard(room)
                self.disagg_awaiting_ready.pop(room, None)
                ctrl.failed.append((room, reason))
            ctrl.sent = [room for room, _ in self.disagg_consensus.pop_complete()]
            # Enqueue times only grow, so the timed out rooms come first.
            while self.disagg_awaiting_ready:
                room, enqueue_time = next(iter(self.disagg_awaiting_ready.items()))
                if enqueue_time >= deadline:
                    break
                del self.disagg_awaiting_ready[room]
                self.disagg_failed_history.add(room)
                ctrl.failed.append((room, "timed out waiting for the decode server"))

        return ctrl if ctrl else None

    def disagg_apply_control(self: "Scheduler", ctrl: DisaggControl):
        """All ranks: apply the decisions of rank 0."""
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for room in ctrl.sent:
                self.disagg_prefill_sent(room)
            for room, reason in ctrl.failed:
                req = self.disagg_prefill_abort(room, reason)
                if req is not None:
                    self.disagg_finish_aborted(req, reason)
        else:
            for room, first_token in ctrl.received:
                self.disagg_decode_received(room, first_token)
            for room, reason in ctrl.cancel:
                self.disagg_decode_cancel(room, reason)
            for room in ctrl.released:
                released = self.disagg_decode_release(room)
                if released is not None:
                    self.disagg_finish_aborted(*released)

    def disagg_finish_aborted(self: "Scheduler", req: Req, reason: str):
        req.finished_reason = FINISH_ABORT(f"PD disaggregation failed: {reason}")
        self.stream_output([req], req.return_logprob)

    def disagg_ready_event(self: "Scheduler") -> torch.cuda.Event:
        """An event after the GPU work issued so far by the scheduler thread."""
        event = torch.cuda.Event()
        event.record()
        return event
