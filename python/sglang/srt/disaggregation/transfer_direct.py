"""
KV transfer that writes straight into the KV pool of the decode rank (backends
"kernel" and "memcpy"), for GPUs with peer-to-peer access on one node.

The decode rank exports its KV buffers with CUDA IPC in its HELLO and sends the KV rows
that it preallocated for a request in the READY message. The prefill rank maps the
buffers once and copies the rows of a request from its own KV pool:
    kernel: one copy kernel for all buffers, as SGLang HiCache's transfer_kv_all_layer;
            the SMs of the prefill GPU write the rows to the peer GPU.
    memcpy: one cuMemcpyBatchAsync call with a copy per buffer and run of rows that are
            consecutive on both sides, as Mooncake's intra-node NVLink transport; the
            copy engines move the data.
No staging buffer is used and nothing is launched on the decode GPU. A chunked prefill
sends the KV of each chunk as soon as the chunk is computed, while the next chunks are
prefilled; DONE follows the copy of the last part.

Protocol between a decode DEALER and a prefill ROUTER (JSON messages, loopback only):
    decode -> prefill: HELLO (mode, layout, buffer handles), READY (room, rows),
                       CANCEL (room)
    prefill -> decode: HELLO_ACK, DONE (room, first token), FAIL (room, reason),
                       CANCELLED (room)
A decode rank frees the rows of a room that it cancelled after READY only on DONE,
CANCELLED or FAIL: the prefill rank answers once no copy of the room is pending.
The reports to scheduler rank 0 are those of transfer.py.
"""

import base64
import collections
import ctypes
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
import zmq

from sglang.srt.disaggregation.transfer import (
    _MAX_CANCELLED_ROOMS,
    LOOPBACK,
    KVLayout,
    _Cancel,
    _pack,
    _transfer_stream,
    _TransferThread,
    _unpack,
    export_cuda_tensor,
    get_kv_buffers,
)

logger = logging.getLogger(__name__)

DIRECT_MODES = ("kernel", "memcpy")
# Programs of the copy kernel; arbitrary, about the 64 warps of HiCache's default grid.
_KERNEL_PROGRAMS = int(os.environ.get("SGLANG_PD_KERNEL_PROGRAMS", "16"))
# Log the size and the time of every copy (prefill rank 0).
_LOG_COPIES = os.environ.get("SGLANG_PD_TRANSFER_LOG") == "1"


@triton.jit
def _copy_rows_kernel(
    src_bases,
    dst_bases,
    src_rows,
    dst_rows,
    num_rows,
    num_buffers,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row src_rows[i] of every source buffer to row dst_rows[i] of the matching
    # destination buffer; the bases are int64 addresses and a row is ROW_WORDS int64.
    offs = tl.arange(0, BLOCK)
    mask = offs < ROW_WORDS
    for r in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        src_row = tl.load(src_rows + r)
        dst_row = tl.load(dst_rows + r)
        for b in range(num_buffers):
            src = tl.load(src_bases + b).to(tl.pointer_type(tl.int64))
            dst = tl.load(dst_bases + b).to(tl.pointer_type(tl.int64))
            words = tl.load(src + src_row * ROW_WORDS + offs, mask=mask)
            tl.store(dst + dst_row * ROW_WORDS + offs, words, mask=mask)


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    # CUmemcpyAttributes_v1 of CUDA 12.8
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1


class _MemcpyBatch:
    """cuMemcpyBatchAsync of the driver, with its CUDA 12.8 signature. It works with an
    older CUDA runtime (PyTorch's) as long as the driver supports CUDA 12.8."""

    def __init__(self):
        u64p = ctypes.POINTER(ctypes.c_uint64)
        sizep = ctypes.POINTER(ctypes.c_size_t)
        self._fn = _driver_function(
            b"cuMemcpyBatchAsync",
            12080,
            ctypes.c_int,
            u64p,  # dsts
            u64p,  # srcs
            sizep,  # sizes
            ctypes.c_size_t,  # count
            ctypes.POINTER(_CUmemcpyAttributes),
            sizep,  # attrsIdxs
            ctypes.c_size_t,  # numAttrs
            sizep,  # failIdx
            ctypes.c_void_p,  # hStream
        )
        if self._fn is None:
            raise RuntimeError(
                "The memcpy transfer backend needs cuMemcpyBatchAsync, which requires an "
                "NVIDIA driver for CUDA 12.8 or later."
            )
        # The source is read in stream order, after the KV is computed.
        self._attrs = _CUmemcpyAttributes(srcAccessOrder=_CU_MEMCPY_SRC_ACCESS_ORDER_STREAM)
        self._attr_index = (ctypes.c_size_t * 1)(0)

    def __call__(self, dsts: np.ndarray, srcs: np.ndarray, sizes: np.ndarray, stream: int):
        count = len(dsts)
        fail = ctypes.c_size_t(count)
        err = self._fn(
            dsts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            srcs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            count,
            ctypes.byref(self._attrs),
            self._attr_index,
            1,
            ctypes.byref(fail),
            ctypes.c_void_p(stream),
        )
        if err != 0:
            raise RuntimeError(
                f"cuMemcpyBatchAsync failed with CUresult {err} at copy {fail.value} of {count}"
            )


def _driver_function(name: bytes, version: int, restype, *argtypes):
    """A CUDA driver API function with the signature of the given CUDA version."""
    try:
        get_proc = ctypes.CDLL("libcuda.so.1").cuGetProcAddress_v2
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"Cannot load the CUDA driver API: {e}") from e
    get_proc.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    get_proc.restype = ctypes.c_int
    fn, status = ctypes.c_void_p(), ctypes.c_int()
    if get_proc(name, ctypes.byref(fn), version, 0, ctypes.byref(status)) != 0 or not fn.value:
        return None
    return ctypes.CFUNCTYPE(restype, *argtypes)(fn.value)


class _CUipcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1


class _IpcMapper:
    """Maps the buffers of export_cuda_tensor into the current context. PyTorch would map
    them in a context of the GPU that owns the memory, where the kernels of this GPU
    cannot access them; this also enables peer access to that GPU."""

    def __init__(self):
        self._open = _driver_function(
            b"cuIpcOpenMemHandle",
            12000,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            _CUipcMemHandle,
            ctypes.c_uint,
        )
        if self._open is None:
            raise RuntimeError("Cannot find cuIpcOpenMemHandle in the CUDA driver.")
        self._bases: Dict[bytes, int] = {}  # One mapping per exported allocation

    def address(self, handle: dict) -> int:
        raw = base64.b64decode(handle["handle"])
        # PyTorch prefixes the cudaIpcMemHandle_t of a cudaMalloc block with a version
        # byte (1) and a kind byte ('c'); older versions send the bare handle.
        if len(raw) == ctypes.sizeof(_CUipcMemHandle) + 2 and raw[:2] == b"\x01c":
            raw = raw[2:]
        if len(raw) != ctypes.sizeof(_CUipcMemHandle):
            raise ValueError(
                "Unsupported CUDA IPC handle; expandable segments "
                "(PYTORCH_CUDA_ALLOC_CONF) are not supported."
            )
        if raw not in self._bases:
            ptr = ctypes.c_uint64()
            err = self._open(
                ctypes.byref(ptr),
                _CUipcMemHandle.from_buffer_copy(raw),
                _CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            )
            if err != 0:
                raise RuntimeError(
                    f"cuIpcOpenMemHandle failed with CUresult {err}. The kernel and memcpy "
                    "backends need GPU peer-to-peer; use the host backend instead."
                )
            self._bases[raw] = ptr.value
        return self._bases[raw] + handle["storage_offset_bytes"] + handle["offset"]


def _runs(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Starts and lengths of the runs of rows that are consecutive on both sides."""
    breaks = np.flatnonzero((np.diff(src) != 1) | (np.diff(dst) != 1)) + 1
    starts = np.concatenate(([0], breaks))
    lengths = np.diff(np.append(starts, len(src)))
    return src[starts], dst[starts], lengths


def _buffer_handles(token_to_kv_pool) -> List[dict]:
    return [export_cuda_tensor(b.view(torch.uint8).view(-1)) for b in get_kv_buffers(token_to_kv_pool)]


@dataclass
class _DirectPeer:
    dst_bases: np.ndarray  # uint64 addresses of the decode KV buffers, mapped with IPC
    dst_table: Optional[torch.Tensor]  # The same on this GPU, for the copy kernel


@dataclass
class _Part:
    """Rows [start, start + len(src_rows)) of the prompt KV of a room."""

    room: int
    start: int
    src_rows: torch.Tensor  # int64 KV indices of the rows on this rank
    ready_event: torch.cuda.Event  # The KV of the rows is computed after this event
    last: bool
    first_token: Optional[int] = None


@dataclass
class _Room:
    """The parts of a room; a prompt is sent in parts when its prefill is chunked."""

    queued: Deque[_Part]  # Submitted, not launched yet
    peer: Optional[bytes] = None
    dst_rows: Optional[list] = None  # Decode rows of the whole prompt, from READY
    pending: int = 0  # Parts launched and not completed
    rows_sent: int = 0  # Rows of the launched parts
    last_seen: bool = False
    fail: Optional[Tuple[str, bool]] = None  # (reason, report) once nothing is pending
    decode_cancelled: bool = False


@dataclass
class _Copy:
    """A launched part; handled when `done` completes, in launch order."""

    part: _Part
    done: torch.cuda.Event
    keep: tuple  # Tensors that the copy reads, kept alive until it completes
    start: Optional[torch.cuda.Event] = None
    t_launch: float = 0.0
    num_copies: int = 0


@dataclass
class _Staged:
    """memcpy: a part waiting for its source rows to reach the CPU."""

    part: _Part
    peer: bytes
    dst_rows: np.ndarray
    src_rows: torch.Tensor  # pinned CPU tensor
    event: torch.cuda.Event


class DirectKVSender(_TransferThread):
    """Prefill side: copies the KV of a request straight into the decode KV pool, in
    parts as the chunks of its prefill complete."""

    def __init__(
        self,
        token_to_kv_pool,
        gpu_id: int,
        rank: int,
        world_info: dict,
        port: int,
        report_endpoint: str,
        report_ready: bool,
        mode: str,
    ):
        super().__init__(f"kv-sender-{rank}", gpu_id, report_endpoint)
        assert mode in DIRECT_MODES, mode
        self.mode = mode
        self.layout = KVLayout(get_kv_buffers(token_to_kv_pool))
        self.rank = rank
        self.world_info = world_info
        self.report_ready = report_ready
        bases = [b.data_ptr() for b in self.layout.buffers]
        self.src_bases = np.array(bases, dtype=np.uint64)
        self.row_bytes = np.array(self.layout.token_bytes, dtype=np.uint64)
        if mode == "kernel":
            row_bytes = set(self.layout.token_bytes)
            if len(row_bytes) != 1 or next(iter(row_bytes)) % 8:
                raise ValueError(
                    "The kernel transfer backend needs KV buffers with equal rows of a "
                    "multiple of 8 bytes; use --disaggregation-transfer-backend memcpy."
                )
            self.row_words = next(iter(row_bytes)) // 8
            self.src_table = torch.tensor(bases, dtype=torch.int64, device=self.device)
        else:
            self.memcpy_batch = _MemcpyBatch()

        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        try:
            self.router.bind(f"tcp://{LOOPBACK}:{port}")
        except zmq.ZMQError as e:
            raise RuntimeError(
                f"Cannot bind the KV transfer port {port} ({e}). "
                "Choose another --disaggregation-bootstrap-port."
            ) from e
        logger.info(f"KV sender ({mode}) listens on {LOOPBACK}:{port}")
        self.thread.start()

    def send(
        self,
        room: int,
        src_indices: torch.Tensor,
        first_token: int,
        ready_event: torch.cuda.Event,
    ):
        """Copy the whole prompt KV at `src_indices` once the decode rank is ready."""
        self.send_part(room, 0, src_indices, ready_event, True, first_token)

    def send_part(
        self,
        room: int,
        start: int,
        src_rows: torch.Tensor,
        ready_event: torch.cuda.Event,
        last: bool,
        first_token: Optional[int] = None,
    ):
        """Copy prompt rows [start, start + len(src_rows)). The parts of a room may come
        in any order, but they must cover the prompt once, and `last` (with the first
        output token) marks the final one."""
        self._submit(_Part(room, start, src_rows, ready_event, last, first_token))

    def _loop(self):
        stream = _transfer_stream(self.device)
        self.ipc = _IpcMapper()
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)
        poller.register(self.wakeup_rx, zmq.POLLIN)

        peers: Dict[bytes, _DirectPeer] = {}
        rooms: Dict[int, _Room] = {}
        staged: Deque[_Staged] = deque()
        inflight: Deque[_Copy] = deque()
        cancel_after: Set[int] = set()  # Rooms the decode rank cancelled while copying
        # room -> reason: rooms failed or cancelled here; their later parts are dropped
        # and a later READY is answered with FAIL.
        cancelled = collections.OrderedDict()

        def reply(peer: bytes, msg: list):
            self.router.send_multipart([peer, _pack(msg)])

        def forget(room: int, reason: str):
            rooms.pop(room, None)
            cancelled[room] = reason
            while len(cancelled) > _MAX_CANCELLED_ROOMS:
                cancelled.popitem(last=False)

        def fail(room: int, state: _Room, reason: str, report: bool):
            """Stop sending a room; the decode rank hears of it once no copy is pending."""
            state.queued.clear()
            if report:
                logger.error(f"KV transfer of room {room} failed: {reason}")
            if state.pending:
                state.fail = (reason, report)
                return
            if state.peer is not None:
                reply(state.peer, ["FAIL", room, reason])
            if report:
                self._report(("fail", room, reason))
            forget(room, reason)

        def finish_cancelled(room: int):
            """The decode rank no longer needs the KV: the request is finished here."""
            rooms.pop(room, None)
            self._report(("done", self.rank, room, None))

        def launch(room: int, state: _Room):
            if state.dst_rows is None or state.fail is not None:
                return
            total = len(state.dst_rows)
            while state.queued:
                part = state.queued.popleft()
                num = part.src_rows.numel()
                if part.start + num > total or (part.last and state.rows_sent + num != total):
                    size = state.rows_sent + num if part.last else part.start + num
                    fail(room, state, f"prompt length mismatch (prefill {size}, decode {total})", True)
                    return
                state.rows_sent += num
                state.pending += 1
                if self.mode == "kernel":
                    inflight.append(self._launch_kernel(stream, part, peers[state.peer], state.dst_rows))
                else:
                    staged.append(self._stage(stream, part, state.peer, state.dst_rows))

        while True:
            events = self._poll(poller, busy=bool(inflight or staged))
            self._drain_wakeups(events)

            for item in self._drain_requests():
                room = item.room
                state = rooms.get(room)
                if isinstance(item, _Cancel):
                    if state is None or state.decode_cancelled:
                        forget(room, item.reason)
                    elif state.pending == 0 and state.rows_sent == 0:
                        # Nothing was copied: tell the decode rank, now or on READY.
                        if state.peer is not None:
                            reply(state.peer, ["FAIL", room, item.reason])
                        forget(room, item.reason)
                    else:
                        fail(room, state, item.reason, False)
                    continue
                if room in cancelled:
                    continue
                if state is None:
                    state = rooms[room] = _Room(deque())
                if state.decode_cancelled:
                    if item.last:
                        finish_cancelled(room)
                    continue
                state.queued.append(item)
                state.last_seen |= item.last
                launch(room, state)

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
                        _, room, rows = msg
                        if peer not in peers:
                            reply(peer, ["FAIL", room, "READY before HELLO"])
                        elif room in cancelled:
                            reply(peer, ["FAIL", room, cancelled.pop(room)])
                        else:
                            state = rooms.get(room)
                            if state is None:
                                state = rooms[room] = _Room(deque())
                            state.peer, state.dst_rows = peer, rows
                            if self.report_ready:
                                self._report(("ready", room))
                            launch(room, state)
                    elif kind == "CANCEL":
                        room = msg[1]
                        state = rooms.get(room)
                        if state is None:
                            reply(peer, ["CANCELLED", room])
                            continue
                        state.queued.clear()
                        state.decode_cancelled = True
                        if state.pending:
                            cancel_after.add(room)
                        else:
                            reply(peer, ["CANCELLED", room])
                            if state.last_seen:
                                finish_cancelled(room)
                    else:
                        logger.warning(f"KV sender got an unknown message {kind}")

            # memcpy: launch the parts whose source rows have reached the CPU.
            while staged and staged[0].event.query():
                inflight.append(self._launch_memcpy(stream, staged.popleft(), peers))

            # Copies complete in launch order because they share one stream.
            while inflight and inflight[0].done.query():
                copy = inflight.popleft()
                part = copy.part
                if _LOG_COPIES and self.rank == 0 and copy.start is not None:
                    logger.info(
                        f"PD copy ({self.mode}) room={part.room} start={part.start} "
                        f"rows={part.src_rows.numel()} last={int(part.last)} "
                        f"copies={copy.num_copies} gpu_ms={copy.start.elapsed_time(copy.done):.2f} "
                        f"total_ms={(time.perf_counter() - copy.t_launch) * 1000:.2f}"
                    )
                state = rooms.get(part.room)
                if state is None:
                    continue
                state.pending -= 1
                if state.pending:
                    continue
                if state.fail is not None:
                    reason, report = state.fail
                    reply(state.peer, ["FAIL", part.room, reason])
                    if report:
                        self._report(("fail", part.room, reason))
                    forget(part.room, reason)
                elif part.room in cancel_after:
                    cancel_after.discard(part.room)
                    reply(state.peer, ["CANCELLED", part.room])
                    if state.last_seen:
                        finish_cancelled(part.room)
                elif part.last:
                    reply(state.peer, ["DONE", part.room, part.first_token])
                    self._report(("done", self.rank, part.room, None))
                    rooms.pop(part.room, None)

    def _accept_peer(self, peer: bytes, info: dict, peers: Dict[bytes, _DirectPeer]):
        if info.get("mode") != self.mode:
            return (
                f"transfer backend mismatch (prefill {self.mode}, decode "
                f"{info.get('mode') or 'gpu or host'}); launch both servers with the same "
                "--disaggregation-transfer-backend"
            )
        if info["world"] != self.world_info:
            return f"rank mismatch (prefill {self.world_info}, decode {info['world']})"
        if info["layout"] != self.layout.signature():
            return (
                "KV layout mismatch. The prefill and decode servers must serve the same "
                "model with the same --tp-size, --pp-size and --kv-cache-dtype."
            )
        if len(info["buffers"]) != len(self.layout.buffers):
            return "the decode rank has a different number of KV buffers"
        try:
            bases = [self.ipc.address(h) for h in info["buffers"]]
        except Exception as e:
            return f"cannot open the KV buffers of the decode rank: {e}"
        peers[peer] = _DirectPeer(
            dst_bases=np.array(bases, dtype=np.uint64),
            dst_table=(
                torch.tensor(bases, dtype=torch.int64, device=self.device)
                if self.mode == "kernel"
                else None
            ),
        )
        logger.info("KV sender accepted a decode rank")
        return None

    def _timing_events(self, stream) -> Tuple[Optional[torch.cuda.Event], float]:
        if not (_LOG_COPIES and self.rank == 0):
            return None, 0.0
        start = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return start, time.perf_counter()

    def _launch_kernel(self, stream, part: _Part, peer: _DirectPeer, dst_rows: list) -> _Copy:
        num = part.src_rows.numel()
        with torch.cuda.stream(stream):
            stream.wait_event(part.ready_event)
            start, t_launch = self._timing_events(stream)
            dst = torch.tensor(dst_rows[part.start : part.start + num], dtype=torch.int64)
            dst = dst.to(self.device, non_blocking=True)
            if num > 0:
                block = triton.next_power_of_2(self.row_words)
                _copy_rows_kernel[(min(num, _KERNEL_PROGRAMS),)](
                    self.src_table,
                    peer.dst_table,
                    part.src_rows,
                    dst,
                    num,
                    len(self.layout.buffers),
                    ROW_WORDS=self.row_words,
                    BLOCK=block,
                    num_warps=max(1, min(4, block // 32)),
                )
            done = torch.cuda.Event(enable_timing=start is not None)
            done.record(stream)
        return _Copy(part, done, (part.src_rows, dst), start, t_launch, num * len(self.layout.buffers))

    def _stage(self, stream, part: _Part, peer_id: bytes, dst_rows: list) -> _Staged:
        num = part.src_rows.numel()
        with torch.cuda.stream(stream):
            stream.wait_event(part.ready_event)
            src_rows = torch.empty(num, dtype=torch.int64, pin_memory=True)
            src_rows.copy_(part.src_rows, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        dst = np.asarray(dst_rows[part.start : part.start + num], dtype=np.uint64)
        return _Staged(part, peer_id, dst, src_rows, event)

    def _launch_memcpy(self, stream, staged: _Staged, peers: Dict[bytes, _DirectPeer]) -> _Copy:
        peer = peers[staged.peer]
        src_rows = staged.src_rows.numpy().astype(np.uint64)
        num_copies = 0
        with torch.cuda.stream(stream):
            start, t_launch = self._timing_events(stream)
            if len(src_rows) > 0:
                src_start, dst_start, lengths = _runs(src_rows, staged.dst_rows)
                row = self.row_bytes[:, None]
                srcs = (self.src_bases[:, None] + src_start[None, :] * row).ravel()
                dsts = (peer.dst_bases[:, None] + dst_start[None, :] * row).ravel()
                sizes = (lengths.astype(np.uint64)[None, :] * row).ravel()
                num_copies = len(srcs)
                self.memcpy_batch(dsts, srcs, sizes, stream.cuda_stream)
            done = torch.cuda.Event(enable_timing=start is not None)
            done.record(stream)
        return _Copy(staged.part, done, (staged.part.src_rows,), start, t_launch, num_copies)

@dataclass
class _DirectRecvJob:
    room: int
    peer: Tuple[str, int]
    dst_indices: torch.Tensor  # int64 preallocated KV rows on this rank
    ready_event: torch.cuda.Event  # The rows are valid after this event
    num_tokens: int
    rows_cpu: Optional[torch.Tensor] = None
    rows_event: Optional[torch.cuda.Event] = None
    rows: Optional[list] = None
    sent_ready: bool = False
    cancelling: bool = False


@dataclass
class _DirectConn:
    socket: zmq.Socket
    acked: bool = False
    pending: Optional[List[int]] = None  # Rooms waiting for HELLO_ACK


class DirectKVReceiver(_TransferThread):
    """Decode side: hands out its KV pool and the rows of each request."""

    def __init__(
        self,
        token_to_kv_pool,
        gpu_id: int,
        rank: int,
        world_info: dict,
        report_endpoint: str,
        mode: str,
    ):
        super().__init__(f"kv-receiver-{rank}", gpu_id, report_endpoint)
        assert mode in DIRECT_MODES, mode
        self.mode = mode
        self.layout = KVLayout(get_kv_buffers(token_to_kv_pool))
        self.rank = rank
        self.world_info = world_info
        self.handles = _buffer_handles(token_to_kv_pool)
        logger.info(f"KV receiver ({mode}) exports {len(self.handles)} KV buffers")
        self.thread.start()

    def receive(
        self,
        room: int,
        port: int,
        dst_indices: torch.Tensor,
        ready_event: torch.cuda.Event,
    ):
        """Receive the prompt KV of `room` from the prefill rank on `port`."""
        self._submit(
            _DirectRecvJob(room, (LOOPBACK, port), dst_indices, ready_event, len(dst_indices))
        )

    def _loop(self):
        stream = _transfer_stream(self.device)
        poller = zmq.Poller()
        poller.register(self.wakeup_rx, zmq.POLLIN)

        conns: Dict[Tuple[str, int], _DirectConn] = {}
        socket_to_peer: Dict[zmq.Socket, Tuple[str, int]] = {}
        jobs: Dict[int, _DirectRecvJob] = {}
        fetching: Deque[_DirectRecvJob] = deque()  # Waiting for their rows on the CPU

        def send(peer, msg: list):
            conns[peer].socket.send(_pack(msg))

        def send_ready(job: _DirectRecvJob):
            send(job.peer, ["READY", job.room, job.rows])
            job.sent_ready = True

        while True:
            events = self._poll(poller, busy=bool(fetching))
            self._drain_wakeups(events)

            for item in self._drain_requests():
                if isinstance(item, _Cancel):
                    job = jobs.get(item.room)
                    if job is not None and job.sent_ready:
                        # The prefill rank may be writing the rows; wait for its answer.
                        if not job.cancelling:
                            job.cancelling = True
                            send(job.peer, ["CANCEL", item.room])
                    else:
                        jobs.pop(item.room, None)
                        self._report(("cancelled", self.rank, item.room))
                    continue
                jobs[item.room] = item
                if item.peer not in conns:
                    conn = conns[item.peer] = self._connect(item.peer)
                    socket_to_peer[conn.socket] = item.peer
                    poller.register(conn.socket, zmq.POLLIN)
                with torch.cuda.stream(stream):
                    stream.wait_event(item.ready_event)
                    item.rows_cpu = torch.empty(item.num_tokens, dtype=torch.int64, pin_memory=True)
                    item.rows_cpu.copy_(item.dst_indices, non_blocking=True)
                    item.rows_event = torch.cuda.Event()
                    item.rows_event.record(stream)
                fetching.append(item)

            while fetching and fetching[0].rows_event.query():
                job = fetching.popleft()
                if jobs.get(job.room) is not job:
                    continue  # Cancelled meanwhile
                job.rows = job.rows_cpu.tolist()
                job.rows_cpu = None
                if conns[job.peer].acked:
                    send_ready(job)
                else:
                    conns[job.peer].pending.append(job.room)

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
                    if kind == "DONE":
                        _, room, first_token = msg
                        job = jobs.pop(room, None)
                        if job is not None:
                            self._report(
                                ("cancelled", self.rank, room)
                                if job.cancelling
                                else ("done", self.rank, room, first_token)
                            )
                    elif kind in ("CANCELLED", "FAIL"):
                        room = msg[1]
                        job = jobs.pop(room, None)
                        if job is None:
                            continue
                        if job.cancelling:
                            self._report(("cancelled", self.rank, room))
                        elif kind == "FAIL":
                            logger.error(f"KV transfer of room {room} failed: {msg[2]}")
                            self._report(("fail", room, msg[2]))
                    elif kind == "HELLO_ACK":
                        _, ok, error = msg
                        if not ok:
                            # A configuration error: stop the server.
                            raise RuntimeError(
                                f"The prefill rank at {peer} rejected this decode rank: {error}"
                            )
                        conn.acked = True
                        pending, conn.pending = conn.pending, []
                        for room in pending:
                            job = jobs.get(room)
                            if job is not None and job.rows is not None:
                                send_ready(job)
                    else:
                        logger.warning(f"KV receiver got an unknown message {kind}")

    def _connect(self, peer: Tuple[str, int]) -> _DirectConn:
        host, port = peer
        sock = self.ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{host}:{port}")
        sock.send(
            _pack(
                [
                    "HELLO",
                    {
                        "mode": self.mode,
                        "world": self.world_info,
                        "layout": self.layout.signature(),
                        "buffers": self.handles,
                    },
                ]
            )
        )
        logger.info(f"KV receiver connects to the prefill rank at {host}:{port}")
        return _DirectConn(socket=sock, pending=[])
