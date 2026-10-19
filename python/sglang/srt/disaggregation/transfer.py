"""
KV cache transfer between the prefill and decode servers of a PD-disaggregated
deployment on a single node.

Each scheduler rank (pp_rank, tp_rank) of the prefill server is paired with the
rank of the same (pp_rank, tp_rank) on the decode server. Both servers run the
same model with the same TP/PP sizes, so a rank pair holds the KV of the same
layers and heads. The prefill rank pushes the prompt KV of a request to its
peer in chunks of tokens:

1. prefill GPU: gather the KV rows of the chunk into a local staging slot.
2. copy the slot into a staging slot of the decode rank. The decode staging is
   either GPU memory exported with CUDA IPC (backend "gpu"; a peer-to-peer copy
   if the GPUs support it) or shared pinned host memory (backend "host"; two
   PCIe copies, for systems where peer-to-peer is broken or disabled).
3. decode GPU: scatter the rows of the slot into the preallocated KV rows.

Each process launches kernels only on its own GPU and no GPU ever waits for the
other process; the two sides synchronize on the CPU through ZMQ messages. A
transfer therefore cannot deadlock with the TP/PP collectives of either server.

Protocol between a decode DEALER and a prefill ROUTER (JSON messages, over
loopback only):
    decode -> prefill: HELLO (layout, staging buffer), READY (room, num_tokens),
                       CREDIT (a staging slot that can be written again)
    prefill -> decode: HELLO_ACK, DATA (room, slot, token range, first token),
                       FAIL (room, reason)
A rejected HELLO (e.g., different models) is a configuration error and stops
the decode server.

Both sides report to scheduler rank 0 of their own server with a ZMQ PUSH socket:
    ("ready", room)                    prefill rank 0 only: decode preallocated KV
    ("done", rank, room, first_token)  this rank finished sending / receiving
    ("fail", room, reason)
    ("cancelled", rank, room)          decode: this rank no longer writes the KV
    ("crash", traceback)               the transfer thread died
"""

import atexit
import base64
import collections
import contextlib
import glob
import json
import logging
import os
import queue
import re
import stat
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import psutil
import torch
import zmq
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

logger = logging.getLogger(__name__)

# Single node: the prefill ranks only accept connections from this machine.
LOOPBACK = "127.0.0.1"

_ALIGN_BYTES = 256
_IDLE_POLL_MS = 100
_BUSY_SLEEP_S = 5e-5
_MAX_CANCELLED_ROOMS = 100_000


def _pack(msg: list) -> bytes:
    return json.dumps(msg).encode()


def _unpack(payload: bytes) -> list:
    return json.loads(payload)


def _b64(data: Optional[bytes]) -> Optional[str]:
    return None if data is None else base64.b64encode(data).decode()


def _unb64(data: Optional[str]) -> Optional[bytes]:
    return None if data is None else base64.b64decode(data)


def get_kv_buffers(token_to_kv_pool) -> List[torch.Tensor]:
    """The per-layer KV tensors of a token-to-KV pool, each of shape [size + 1, ...]."""
    buffers = []
    for name in ("k_buffer", "v_buffer", "label_buffer", "kv_buffer"):
        buffers.extend(getattr(token_to_kv_pool, name, None) or [])
    if not buffers:
        raise ValueError(
            f"{type(token_to_kv_pool).__name__} is not supported by PD disaggregation."
        )
    return buffers


class KVLayout:
    """Packing of the KV rows of a chunk of tokens into a uint8 staging slot.

    The rows of each buffer fill one region and the regions are back to back,
    so a chunk of n tokens occupies the first nbytes(n) bytes of a slot.
    """

    def __init__(self, buffers: List[torch.Tensor]):
        self.buffers = buffers
        self.token_shapes = [tuple(b.shape[1:]) for b in buffers]
        self.dtypes = [b.dtype for b in buffers]
        self.token_bytes = [b[0].numel() * b.element_size() for b in buffers]

    def signature(self) -> list:
        return [
            [list(shape), str(dtype)]
            for shape, dtype in zip(self.token_shapes, self.dtypes)
        ]

    def _offsets(self, num_tokens: int) -> Tuple[List[int], int]:
        offsets, end = [], 0
        for token_bytes in self.token_bytes:
            offsets.append(end)
            end += -(-token_bytes * num_tokens // _ALIGN_BYTES) * _ALIGN_BYTES
        return offsets, end

    def nbytes(self, num_tokens: int) -> int:
        return self._offsets(num_tokens)[1]

    def views(self, slot: torch.Tensor, num_tokens: int) -> List[torch.Tensor]:
        offsets, _ = self._offsets(num_tokens)
        return [
            slot[off : off + token_bytes * num_tokens].view(dtype).view(num_tokens, *shape)
            for off, token_bytes, shape, dtype in zip(
                offsets, self.token_bytes, self.token_shapes, self.dtypes
            )
        ]


def _device_uuid(index: int) -> Optional[str]:
    try:
        return str(torch.cuda.get_device_properties(index).uuid)
    except Exception:
        return None


def export_cuda_tensor(tensor: torch.Tensor) -> dict:
    """A JSON handle that lets another process on this node open a uint8 CUDA tensor."""
    assert tensor.dtype == torch.uint8 and tensor.is_contiguous()
    _, args = reduce_tensor(tensor)
    (
        tensor_cls,
        size,
        stride,
        offset,
        storage_cls,
        dtype,
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        _requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = args
    assert tensor_cls is torch.Tensor and storage_cls is torch.storage.TypedStorage
    return {
        "kind": "gpu",
        "size": list(size),
        "stride": list(stride),
        "offset": offset,
        "device": device,
        "handle": _b64(handle),
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "ref_counter_handle": _b64(ref_counter_handle),
        "ref_counter_offset": ref_counter_offset,
        "event_handle": _b64(event_handle),
        "event_sync_required": event_sync_required,
        "uuid": _device_uuid(tensor.device.index),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def import_cuda_tensor(handle: dict) -> torch.Tensor:
    device = handle["device"]
    if handle["uuid"] is not None:
        matches = [
            i
            for i in range(torch.cuda.device_count())
            if _device_uuid(i) == handle["uuid"]
        ]
        if not matches:
            raise RuntimeError(
                "The GPU of the peer server is not visible to this process. Launch the "
                "prefill and decode servers with the same CUDA_VISIBLE_DEVICES and place "
                "them on different GPUs with --base-gpu-id."
            )
        device = matches[0]
    elif handle["visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "The prefill and decode servers must use the same CUDA_VISIBLE_DEVICES."
        )
    return rebuild_cuda_tensor(
        torch.Tensor,
        torch.Size(handle["size"]),
        tuple(handle["stride"]),
        handle["offset"],
        torch.storage.TypedStorage,
        torch.uint8,
        device,
        _unb64(handle["handle"]),
        handle["storage_size_bytes"],
        handle["storage_offset_bytes"],
        False,
        _unb64(handle["ref_counter_handle"]),
        handle["ref_counter_offset"],
        _unb64(handle["event_handle"]),
        handle["event_sync_required"],
    )


_HOST_STAGING_PREFIX = "/dev/shm/fastpp-kv-staging-"
_HOST_STAGING_RE = re.compile(re.escape(_HOST_STAGING_PREFIX) + r"(\d+)-\d+")


def _unlink_quietly(path: str):
    with contextlib.suppress(OSError):
        os.unlink(path)


def _remove_stale_host_staging():
    """Remove the staging files left by processes that died before the handshake."""
    for path in glob.glob(_HOST_STAGING_PREFIX + "*"):
        match = _HOST_STAGING_RE.fullmatch(path)
        if match and not psutil.pid_exists(int(match.group(1))):
            _unlink_quietly(path)


def _pin_host_memory(tensor: torch.Tensor):
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel(), 0)
    if int(getattr(err, "value", err)) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


def create_host_staging(nbytes: int) -> Tuple[torch.Tensor, dict]:
    """Pinned host memory that another process on this node can map.

    The file is removed once the peer has mapped it (or at exit).
    """
    _remove_stale_host_staging()
    path = f"{_HOST_STAGING_PREFIX}{os.getpid()}-{time.monotonic_ns()}"
    tensor = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    atexit.register(_unlink_quietly, path)
    _pin_host_memory(tensor)
    return tensor, {"kind": "host", "path": path, "nbytes": nbytes}


def open_staging(handle: dict) -> torch.Tensor:
    """Open the staging buffer of a decode rank."""
    if handle["kind"] == "gpu":
        return import_cuda_tensor(handle)
    if handle["kind"] != "host":
        raise ValueError(f"Unknown staging kind {handle['kind']!r}")
    path, nbytes = handle["path"], handle["nbytes"]
    # Only map an existing staging file of the expected size.
    if not _HOST_STAGING_RE.fullmatch(path):
        raise ValueError(f"Not a staging file: {path!r}")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size != nbytes:
        raise ValueError(f"Unexpected staging file: {path!r}")
    tensor = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _pin_host_memory(tensor)
    return tensor


def _transfer_stream(device: torch.device) -> torch.cuda.Stream:
    """The stream of a transfer thread, with the highest priority. Its small gather and
    scatter kernels otherwise wait behind the long kernels of the model forward that
    runs on the same GPU at the same time."""
    return torch.cuda.Stream(device=device, priority=torch.cuda.Stream.priority_range()[1])


def _connect_push(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.connect(f"ipc://{endpoint}")
    return sock


def _wakeup_pair(ctx: zmq.Context) -> Tuple[zmq.Socket, zmq.Socket]:
    address = f"inproc://kv-transfer-wakeup-{id(ctx)}"
    rx = ctx.socket(zmq.PAIR)
    rx.bind(address)
    tx = ctx.socket(zmq.PAIR)
    tx.connect(address)
    return rx, tx


@dataclass
class _Cancel:
    room: int
    reason: str


class _TransferThread:
    """Common parts of the sender and the receiver: a daemon thread that owns the
    ZMQ sockets and a CUDA stream, fed by the scheduler thread through a queue."""

    def __init__(self, name: str, gpu_id: int, report_endpoint: str):
        self.device = torch.device("cuda", gpu_id)
        self.ctx = zmq.Context()
        self.report_socket = _connect_push(self.ctx, report_endpoint)
        self.wakeup_rx, self.wakeup_tx = _wakeup_pair(self.ctx)
        self.requests = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)

    def _submit(self, item):
        """Called by the scheduler thread."""
        self.requests.put(item)
        try:
            self.wakeup_tx.send(b"", zmq.NOBLOCK)
        except zmq.Again:
            pass  # The thread polls with a timeout anyway.

    def cancel(self, room: int, reason: str = "cancelled"):
        """Called by the scheduler thread; see the subclasses."""
        self._submit(_Cancel(room, reason))

    def _report(self, msg):
        self.report_socket.send_pyobj(msg)

    def _run(self):
        try:
            torch.cuda.set_device(self.device)
            self._loop()
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.thread.name} crashed:\n{tb}")
            self._report(("crash", tb))

    def _poll(self, poller: zmq.Poller, busy: bool) -> Dict[zmq.Socket, int]:
        if busy:
            events = dict(poller.poll(0))
            if not events:
                time.sleep(_BUSY_SLEEP_S)
            return events
        return dict(poller.poll(_IDLE_POLL_MS))

    def _drain_wakeups(self, events):
        if self.wakeup_rx in events:
            while True:
                try:
                    self.wakeup_rx.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

    def _drain_requests(self):
        while True:
            try:
                yield self.requests.get_nowait()
            except queue.Empty:
                return

    def _loop(self):
        raise NotImplementedError()


@dataclass
class _SendJob:
    room: int
    src_indices: torch.Tensor  # int64 KV indices of the prompt on this rank
    first_token: int
    ready_event: torch.cuda.Event  # The prompt KV is computed after this event
    num_tokens: int
    peer: Optional[bytes] = None
    sent_tokens: int = 0
    waited: bool = False


@dataclass
class _SendPeer:
    staging: torch.Tensor  # The staging buffer of the decode rank
    slot_bytes: int
    chunk_tokens: int
    credits: Deque[int]


@dataclass
class _SendChunk:
    event: torch.cuda.Event
    local_slot: int
    peer: bytes
    msg: list
    finished_room: Optional[int]


class KVSender(_TransferThread):
    """Prefill side of the KV transfer for one scheduler rank."""

    def __init__(
        self,
        token_to_kv_pool,
        gpu_id: int,
        rank: int,
        world_info: dict,
        port: int,
        report_endpoint: str,
        report_ready: bool,
        chunk_tokens: int,
        num_slots: int,
    ):
        super().__init__(f"kv-sender-{rank}", gpu_id, report_endpoint)
        self.layout = KVLayout(get_kv_buffers(token_to_kv_pool))
        self.rank = rank
        self.world_info = world_info
        self.report_ready = report_ready
        self.chunk_tokens = chunk_tokens
        self.num_slots = num_slots
        self.slot_bytes = self.layout.nbytes(chunk_tokens)
        self.staging = torch.empty(
            num_slots * self.slot_bytes, dtype=torch.uint8, device=self.device
        )

        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        try:
            self.router.bind(f"tcp://{LOOPBACK}:{port}")
        except zmq.ZMQError as e:
            raise RuntimeError(
                f"Cannot bind the KV transfer port {port} ({e}). "
                "Choose another --disaggregation-bootstrap-port."
            ) from e
        logger.info(
            f"KV sender listens on {LOOPBACK}:{port}, "
            f"staging {self.staging.numel() / (1 << 20):.1f} MiB"
        )
        self.thread.start()

    def send(
        self,
        room: int,
        src_indices: torch.Tensor,
        first_token: int,
        ready_event: torch.cuda.Event,
    ):
        """Send the prompt KV at `src_indices` once the decode rank is ready."""
        self._submit(
            _SendJob(room, src_indices, first_token, ready_event, len(src_indices))
        )

    # cancel(room, reason): the request failed on the prefill server before its
    # KV was sent. The decode rank gets a FAIL now, or when its READY arrives.

    def _loop(self):
        stream = _transfer_stream(self.device)
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)
        poller.register(self.wakeup_rx, zmq.POLLIN)

        free_slots: Deque[int] = deque(range(self.num_slots))
        peers: Dict[bytes, _SendPeer] = {}
        ready: Dict[int, Tuple[bytes, int]] = {}  # room -> (peer, num_tokens)
        jobs: Dict[int, _SendJob] = {}
        runnable: List[int] = []  # Rooms whose decode rank is ready, FIFO
        inflight: Deque[_SendChunk] = deque()
        cancelled = collections.OrderedDict()  # room -> reason, until READY arrives

        def reply(peer: bytes, msg: list):
            self.router.send_multipart([peer, _pack(msg)])

        def drop(room: int, reason: str, report: bool):
            """Drop a request whose KV has not been sent and tell the decode rank."""
            job = jobs.pop(room, None)
            peer = job.peer if job is not None else None
            if room in ready:
                peer = ready.pop(room)[0]
            if room in runnable:
                runnable.remove(room)
            if peer is not None:
                reply(peer, ["FAIL", room, reason])
            else:
                cancelled[room] = reason
                while len(cancelled) > _MAX_CANCELLED_ROOMS:
                    cancelled.popitem(last=False)
            if report:
                logger.error(f"KV transfer of room {room} failed: {reason}")
                self._report(("fail", room, reason))

        def try_start(room: int):
            if room not in jobs or room not in ready:
                return
            job = jobs[room]
            peer, num_tokens = ready[room]
            if num_tokens != job.num_tokens:
                drop(
                    room,
                    f"prompt length mismatch (prefill {job.num_tokens}, "
                    f"decode {num_tokens})",
                    report=True,
                )
                return
            del ready[room]
            job.peer = peer
            runnable.append(room)

        while True:
            events = self._poll(poller, busy=bool(inflight))
            self._drain_wakeups(events)

            for item in self._drain_requests():
                if isinstance(item, _Cancel):
                    job = jobs.get(item.room)
                    if job is None or job.sent_tokens == 0:
                        drop(item.room, item.reason, report=False)
                    # Otherwise the KV is being sent; let it finish.
                else:
                    jobs[item.room] = item
                    try_start(item.room)

            if self.router in events:
                while True:
                    try:
                        peer, payload = self.router.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    msg = _unpack(payload)
                    kind = msg[0]
                    if kind == "HELLO":
                        error = self._accept_peer(peer, msg[1], peers)
                        reply(peer, ["HELLO_ACK", error is None, error])
                    elif kind == "READY":
                        _, room, num_tokens = msg
                        if peer not in peers:
                            reply(peer, ["FAIL", room, "READY before HELLO"])
                        elif room in cancelled:
                            reply(peer, ["FAIL", room, cancelled.pop(room)])
                        else:
                            ready[room] = (peer, num_tokens)
                            if self.report_ready:
                                self._report(("ready", room))
                            try_start(room)
                    elif kind == "CREDIT":
                        peers[peer].credits.append(msg[1])
                    else:
                        logger.warning(f"KV sender got an unknown message {kind}")

            # Chunks complete in launch order because they share one stream.
            while inflight and inflight[0].event.query():
                chunk = inflight.popleft()
                free_slots.append(chunk.local_slot)
                reply(chunk.peer, chunk.msg)
                if chunk.finished_room is not None:
                    jobs.pop(chunk.finished_room, None)
                    self._report(("done", self.rank, chunk.finished_room, None))

            while free_slots:
                room = next(
                    (r for r in runnable if peers[jobs[r].peer].credits), None
                )
                if room is None:
                    break
                job = jobs[room]
                inflight.append(
                    self._launch_chunk(stream, job, peers[job.peer], free_slots.popleft())
                )
                if job.sent_tokens == job.num_tokens:
                    runnable.remove(room)

    def _accept_peer(self, peer: bytes, info: dict, peers: Dict[bytes, _SendPeer]):
        if info.get("mode") is not None:
            return (
                f"transfer backend mismatch (prefill gpu or host, decode {info['mode']}); "
                "launch both servers with the same --disaggregation-transfer-backend"
            )
        if info["world"] != self.world_info:
            return f"rank mismatch (prefill {self.world_info}, decode {info['world']})"
        if info["layout"] != self.layout.signature():
            return (
                "KV layout mismatch. The prefill and decode servers must serve the same "
                "model with the same --tp-size, --pp-size and --kv-cache-dtype."
            )
        try:
            staging = open_staging(info["staging"])
        except Exception as e:
            return f"cannot open the staging buffer of the decode rank: {e}"
        if staging.numel() < info["num_slots"] * info["slot_bytes"]:
            return "the staging buffer of the decode rank is too small"
        peers[peer] = _SendPeer(
            staging=staging,
            slot_bytes=info["slot_bytes"],
            chunk_tokens=min(self.chunk_tokens, info["chunk_tokens"]),
            credits=deque(range(info["num_slots"])),
        )
        logger.info(f"KV sender accepted a decode rank, staging on {staging.device}")
        return None

    def _launch_chunk(
        self, stream: torch.cuda.Stream, job: _SendJob, peer: _SendPeer, local_slot: int
    ) -> _SendChunk:
        start = job.sent_tokens
        num = min(peer.chunk_tokens, job.num_tokens - start)
        nbytes = self.layout.nbytes(num)
        credit = peer.credits.popleft()
        local_offset = local_slot * self.slot_bytes
        remote_offset = credit * peer.slot_bytes
        local = self.staging[local_offset : local_offset + nbytes]
        remote = peer.staging[remote_offset : remote_offset + nbytes]

        with torch.cuda.stream(stream):
            if not job.waited:
                stream.wait_event(job.ready_event)
                job.waited = True
            if num > 0:
                indices = job.src_indices[start : start + num]
                for buffer, view in zip(self.layout.buffers, self.layout.views(local, num)):
                    torch.index_select(buffer, 0, indices, out=view)
                remote.copy_(local, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)

        job.sent_tokens += num
        is_last = job.sent_tokens == job.num_tokens
        msg = ["DATA", job.room, credit, start, num, is_last, job.first_token]
        return _SendChunk(event, local_slot, job.peer, msg, job.room if is_last else None)


@dataclass
class _RecvJob:
    room: int
    peer: Tuple[str, int]
    dst_indices: torch.Tensor  # int64 preallocated KV indices on this rank
    ready_event: torch.cuda.Event  # The indices are valid after this event
    num_tokens: int
    waited: bool = False


@dataclass
class _RecvConn:
    socket: zmq.Socket
    staging: torch.Tensor
    # The file of a host staging buffer, removed once the prefill rank mapped it
    staging_path: Optional[str] = None
    acked: bool = False
    pending: Optional[List[int]] = None  # Rooms waiting for HELLO_ACK


@dataclass
class _RecvEvent:
    """GPU work of the receiver stream; handled when it completes, in order."""

    event: torch.cuda.Event
    peer: Optional[Tuple[str, int]] = None
    slot: Optional[int] = None  # The staging slot to give back
    report: Optional[tuple] = None  # Sent to rank 0 after the work completes
    # The job of a scatter: its index tensor must live until the scatter completes,
    # even after the job is dropped (last chunk, cancel or FAIL).
    job: Optional[_RecvJob] = None


class KVReceiver(_TransferThread):
    """Decode side of the KV transfer for one scheduler rank."""

    def __init__(
        self,
        token_to_kv_pool,
        gpu_id: int,
        rank: int,
        world_info: dict,
        report_endpoint: str,
        chunk_tokens: int,
        num_slots: int,
        backend: str = "gpu",
    ):
        super().__init__(f"kv-receiver-{rank}", gpu_id, report_endpoint)
        assert backend in ("gpu", "host"), backend
        self.layout = KVLayout(get_kv_buffers(token_to_kv_pool))
        self.rank = rank
        self.world_info = world_info
        self.chunk_tokens = chunk_tokens
        self.num_slots = num_slots
        self.backend = backend
        self.slot_bytes = self.layout.nbytes(chunk_tokens)
        # A host staging slot is copied to the GPU before it is scattered.
        self.gpu_slot = (
            torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
            if backend == "host"
            else None
        )
        # One staging buffer per prefill server; the first one is allocated now so
        # that it is in place before serving starts.
        self.spare_staging = [self._new_staging()]
        logger.info(
            f"KV receiver staging ({backend}) "
            f"{self.spare_staging[0][0].numel() / (1 << 20):.1f} MiB"
        )
        self.thread.start()

    def _new_staging(self) -> Tuple[torch.Tensor, dict]:
        nbytes = self.num_slots * self.slot_bytes
        if self.backend == "host":
            return create_host_staging(nbytes)
        staging = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        return staging, export_cuda_tensor(staging)

    def receive(
        self,
        room: int,
        port: int,
        dst_indices: torch.Tensor,
        ready_event: torch.cuda.Event,
    ):
        """Receive the prompt KV of `room` from the prefill rank on `port`."""
        self._submit(
            _RecvJob(room, (LOOPBACK, port), dst_indices, ready_event, len(dst_indices))
        )

    # cancel(room): stop writing the KV of `room`. The rank reports "cancelled"
    # once no scatter for the room is pending, so the KV can then be freed.

    def _loop(self):
        stream = _transfer_stream(self.device)
        poller = zmq.Poller()
        poller.register(self.wakeup_rx, zmq.POLLIN)

        conns: Dict[Tuple[str, int], _RecvConn] = {}
        socket_to_peer: Dict[zmq.Socket, Tuple[str, int]] = {}
        jobs: Dict[int, _RecvJob] = {}
        inflight: Deque[_RecvEvent] = deque()

        def send(peer, msg: list):
            conns[peer].socket.send(_pack(msg))

        def after_pending_scatters(report: tuple):
            """Report once the scatters launched so far have completed."""
            with torch.cuda.stream(stream):
                event = torch.cuda.Event()
                event.record(stream)
            inflight.append(_RecvEvent(event, report=report))

        while True:
            events = self._poll(poller, busy=bool(inflight))
            self._drain_wakeups(events)

            for item in self._drain_requests():
                if isinstance(item, _Cancel):
                    jobs.pop(item.room, None)
                    after_pending_scatters(("cancelled", self.rank, item.room))
                    continue
                jobs[item.room] = item
                conn = conns.get(item.peer)
                if conn is None:
                    conn = conns[item.peer] = self._connect(item.peer)
                    socket_to_peer[conn.socket] = item.peer
                    poller.register(conn.socket, zmq.POLLIN)
                if conn.acked:
                    send(item.peer, ["READY", item.room, item.num_tokens])
                else:
                    conn.pending.append(item.room)

            for sock, peer in socket_to_peer.items():
                if sock not in events:
                    continue
                conn = conns[peer]
                while True:
                    try:
                        msg = _unpack(sock.recv(zmq.NOBLOCK))
                    except zmq.Again:
                        break
                    kind = msg[0]
                    if kind == "DATA":
                        _, room, slot, start, num, is_last, first_token = msg
                        job = jobs.get(room)
                        if job is None:
                            send(peer, ["CREDIT", slot])
                            continue
                        event = self._scatter(stream, job, conn, slot, start, num)
                        report = None
                        if is_last:
                            jobs.pop(room)
                            report = ("done", self.rank, room, first_token)
                        inflight.append(_RecvEvent(event, peer, slot, report, job))
                    elif kind == "HELLO_ACK":
                        _, ok, error = msg
                        if conn.staging_path is not None:
                            # Both processes have mapped it now.
                            _unlink_quietly(conn.staging_path)
                            conn.staging_path = None
                        if not ok:
                            # A configuration error: stop the server.
                            raise RuntimeError(
                                f"The prefill rank at {peer} rejected this decode rank: {error}"
                            )
                        conn.acked = True
                        pending, conn.pending = conn.pending, []
                        for room in pending:
                            if room in jobs:
                                send(peer, ["READY", room, jobs[room].num_tokens])
                    elif kind == "FAIL":
                        _, room, reason = msg
                        if jobs.pop(room, None) is not None:
                            logger.error(f"KV transfer of room {room} failed: {reason}")
                            after_pending_scatters(("fail", room, reason))
                    else:
                        logger.warning(f"KV receiver got an unknown message {kind}")

            # Events complete in launch order because they share one stream.
            while inflight and inflight[0].event.query():
                done = inflight.popleft()
                if done.slot is not None:
                    send(done.peer, ["CREDIT", done.slot])
                if done.report is not None:
                    self._report(done.report)

    def _connect(self, peer: Tuple[str, int]) -> _RecvConn:
        host, port = peer
        sock = self.ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{host}:{port}")
        staging, handle = (
            self.spare_staging.pop() if self.spare_staging else self._new_staging()
        )
        sock.send(
            _pack(
                [
                    "HELLO",
                    {
                        "world": self.world_info,
                        "layout": self.layout.signature(),
                        "staging": handle,
                        "slot_bytes": self.slot_bytes,
                        "num_slots": self.num_slots,
                        "chunk_tokens": self.chunk_tokens,
                    },
                ]
            )
        )
        logger.info(f"KV receiver connects to the prefill rank at {host}:{port}")
        return _RecvConn(
            socket=sock, staging=staging, staging_path=handle.get("path"), pending=[]
        )

    def _scatter(
        self,
        stream: torch.cuda.Stream,
        job: _RecvJob,
        conn: _RecvConn,
        slot: int,
        start: int,
        num: int,
    ) -> torch.cuda.Event:
        with torch.cuda.stream(stream):
            if not job.waited:
                stream.wait_event(job.ready_event)
                job.waited = True
            if num > 0:
                offset = slot * self.slot_bytes
                nbytes = self.layout.nbytes(num)
                src = conn.staging[offset : offset + nbytes]
                if self.gpu_slot is not None:
                    self.gpu_slot[:nbytes].copy_(src, non_blocking=True)
                    src = self.gpu_slot[:nbytes]
                indices = job.dst_indices[start : start + num]
                for buffer, view in zip(self.layout.buffers, self.layout.views(src, num)):
                    buffer.index_copy_(0, indices, view)
            event = torch.cuda.Event()
            event.record(stream)
        return event
