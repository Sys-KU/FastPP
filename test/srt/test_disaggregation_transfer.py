"""Tests for the PD disaggregation KV transfer (CUDA IPC + ZMQ).

Runs a prefill process and a decode process on this node. With one GPU both use
cuda:0; with two or more GPUs they use cuda:0 and cuda:1, which also covers the
copy between GPUs. The staging backends ("gpu", "host") and the direct ones
("kernel", "memcpy") are tested. Run with:
    python test/srt/test_disaggregation_transfer.py -v
"""

import multiprocessing as mp
import socket
import tempfile
import time
import unittest

import torch
import zmq

from sglang.srt.disaggregation.transfer import KVReceiver, KVSender
from sglang.srt.disaggregation.transfer_direct import (
    DIRECT_MODES,
    DirectKVReceiver,
    DirectKVSender,
)

NUM_LAYERS = 3
POOL_SIZE = 4096
HEADS, HEAD_DIM = 2, 64
CHUNK_TOKENS = 64
NUM_SLOTS = 2
# Around the chunk boundaries, plus one long prompt.
PROMPT_LENS = [1, 63, 64, 65, 300, 1000]
ROOMS = [1000 + i for i in range(len(PROMPT_LENS))]
# The decode side expects one more token than the prefill side sends.
MISMATCH_ROOM, MISMATCH_LEN = 9999, 10
# The prefill side cancels this room before the decode side is ready.
PREFILL_CANCEL_ROOM, PREFILL_CANCEL_LEN = 8888, 20
# The decode side cancels this room; the prefill side still sends it.
DECODE_CANCEL_ROOM, DECODE_CANCEL_LEN = 7777, 200
# The decode side cancels this room after the first part of a chunked prefill; the
# rest comes after the cancel.
PARTIAL_CANCEL_ROOM, PARTIAL_CANCEL_LEN = 6666, 160
WORLD = {"pp_rank": 0, "tp_rank": 0, "pp_size": 1, "tp_size": 1}
TIMEOUT_S = 120


class FakePool:
    def __init__(self, device, seed=None):
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        def make():
            if gen is None:
                return torch.zeros(
                    POOL_SIZE + 1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
                )
            return (
                torch.randn(POOL_SIZE + 1, HEADS, HEAD_DIM, generator=gen)
                .to(torch.bfloat16)
                .to(device)
            )

        self.k_buffer = [make() for _ in range(NUM_LAYERS)]
        self.v_buffer = [make() for _ in range(NUM_LAYERS)]


def room_indices(seed):
    """Disjoint KV rows for every room, as int64 CPU tensors."""
    lens = PROMPT_LENS + [
        MISMATCH_LEN + 1,
        PREFILL_CANCEL_LEN,
        DECODE_CANCEL_LEN,
        PARTIAL_CANCEL_LEN,
    ]
    perm = torch.randperm(POOL_SIZE, generator=torch.Generator().manual_seed(seed)) + 1
    out, start = [], 0
    for n in lens:
        out.append(perm[start : start + n])
        start += n
    return out


ALL_ROOMS = ROOMS + [
    MISMATCH_ROOM,
    PREFILL_CANCEL_ROOM,
    DECODE_CANCEL_ROOM,
    PARTIAL_CANCEL_ROOM,
]
CANCELLED_ROOMS = [DECODE_CANCEL_ROOM, PARTIAL_CANCEL_ROOM]


def collect_reports(report, expected, deadline):
    """Receive report messages until `expected(msgs)` holds or the deadline passes."""
    msgs = []
    while not expected(msgs):
        if time.time() > deadline:
            break
        if report.poll(100):
            msgs.append(report.recv_pyobj())
    return msgs


def kinds(msgs, kind, field=1):
    return {m[field] for m in msgs if m[0] == kind}


def run_prefill(gpu, port, report_endpoint, results, exit_event, decode_cancelled, backend):
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    report = zmq.Context().socket(zmq.PULL)
    report.bind(f"ipc://{report_endpoint}")
    pool = FakePool(device, seed=1)
    if backend in DIRECT_MODES:
        sender = DirectKVSender(pool, gpu, 0, WORLD, port, report_endpoint, True, backend)
    else:
        sender = KVSender(
            pool, gpu, 0, WORLD, port, report_endpoint, True, CHUNK_TOKENS, NUM_SLOTS
        )
    sender.cancel(PREFILL_CANCEL_ROOM, "cancelled by the test")
    src = room_indices(seed=2)

    chunked = backend in DIRECT_MODES

    def send(i, room, part=None):
        indices = (src[i] if room != MISMATCH_ROOM else src[i][:MISMATCH_LEN]).to(device)
        event = torch.cuda.Event()
        event.record()
        n = len(indices)
        if room == PARTIAL_CANCEL_ROOM and chunked:
            # A chunked prefill: the first chunk now, the rest after the cancel.
            h = n // 2
            if part == 0:
                sender.send_part(room, 0, indices[:h], event, False)
            else:
                sender.send_part(room, h, indices[h:], event, True, 7 + i)
        elif chunked and n >= 300:
            # A chunked prefill with a cached prefix: the middle chunks as they
            # finish, then the prefix and the last chunk.
            a, b = n // 4, n - n // 8
            sender.send_part(room, a, indices[a:b], event, False)
            sender.send_part(room, 0, indices[:a], event, False)
            sender.send_part(room, b, indices[b:], event, True, 7 + i)
        else:
            sender.send(room, indices, 7 + i, event)

    for i, room in enumerate(ALL_ROOMS):
        if room not in (PREFILL_CANCEL_ROOM, *CANCELLED_ROOMS):
            send(i, room)
    if chunked:
        send(ALL_ROOMS.index(PARTIAL_CANCEL_ROOM), PARTIAL_CANCEL_ROOM, part=0)
    # Send what the decode side cancelled after its READY.
    decode_cancelled.wait(TIMEOUT_S)
    send(ALL_ROOMS.index(DECODE_CANCEL_ROOM), DECODE_CANCEL_ROOM)
    send(ALL_ROOMS.index(PARTIAL_CANCEL_ROOM), PARTIAL_CANCEL_ROOM, part=1)

    def expected(msgs):
        return kinds(msgs, "done", 2) == set(ROOMS + CANCELLED_ROOMS) and (
            MISMATCH_ROOM in kinds(msgs, "fail")
        )

    msgs = collect_reports(report, expected, time.time() + TIMEOUT_S)
    results.put(("prefill", msgs))
    exit_event.wait(TIMEOUT_S)


def run_decode(gpu, port, report_endpoint, results, exit_event, decode_cancelled, backend):
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    report = zmq.Context().socket(zmq.PULL)
    report.bind(f"ipc://{report_endpoint}")
    pool = FakePool(device)
    if backend in DIRECT_MODES:
        receiver = DirectKVReceiver(pool, gpu, 0, WORLD, report_endpoint, backend)
    else:
        receiver = KVReceiver(
            pool, gpu, 0, WORLD, report_endpoint, CHUNK_TOKENS, NUM_SLOTS, backend
        )
    dst = room_indices(seed=3)
    for i, room in enumerate(ALL_ROOMS):
        event = torch.cuda.Event()
        event.record()
        receiver.receive(room, port, dst[i].to(device), event)
    time.sleep(2)  # The READY messages have been sent by now.
    for room in CANCELLED_ROOMS:
        receiver.cancel(room)

    def expected(msgs):
        return (
            kinds(msgs, "done", 2) == set(ROOMS)
            and {MISMATCH_ROOM, PREFILL_CANCEL_ROOM} <= kinds(msgs, "fail")
            and set(CANCELLED_ROOMS) <= kinds(msgs, "cancelled", 2)
        )

    msgs = collect_reports(report, expected, time.time() + TIMEOUT_S)
    decode_cancelled.set()
    time.sleep(2)  # The data of the cancelled room arrives now; it must be dropped.

    torch.cuda.synchronize()
    reference = FakePool(device, seed=1)
    src = room_indices(seed=2)
    mismatches = []
    for i, room in enumerate(ROOMS):
        for name in ("k_buffer", "v_buffer"):
            for layer in range(NUM_LAYERS):
                got = getattr(pool, name)[layer][dst[i].to(device)]
                want = getattr(reference, name)[layer][src[i].to(device)]
                if not torch.equal(got, want):
                    mismatches.append((room, name, layer))
    # Rows sent after the cancel are not written; the first chunk of the partially
    # sent room may have been.
    partial = dst[ALL_ROOMS.index(PARTIAL_CANCEL_ROOM)]
    cancelled_rows = torch.cat(
        [dst[ALL_ROOMS.index(DECODE_CANCEL_ROOM)], partial[len(partial) // 2 :]]
    ).to(device)
    untouched = all(
        not getattr(pool, name)[layer][cancelled_rows].any()
        for name in ("k_buffer", "v_buffer")
        for layer in range(NUM_LAYERS)
    )
    results.put(("decode", msgs, mismatches, untouched))
    exit_event.wait(TIMEOUT_S)


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDisaggregationTransfer(unittest.TestCase):
    def test_gpu_staging(self):
        self._test_transfer("gpu")

    def test_host_staging(self):
        self._test_transfer("host")

    def test_direct_kernel(self):
        self._test_transfer("kernel")

    def test_direct_memcpy(self):
        self._test_transfer("memcpy")

    def _test_transfer(self, backend):
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        exit_event = ctx.Event()
        decode_cancelled = ctx.Event()
        decode_gpu = 1 if torch.cuda.device_count() > 1 else 0
        port = free_port()
        procs = [
            ctx.Process(
                target=run_prefill,
                args=(
                    0,
                    port,
                    tempfile.mktemp(),
                    results,
                    exit_event,
                    decode_cancelled,
                    backend,
                ),
            ),
            ctx.Process(
                target=run_decode,
                args=(
                    decode_gpu,
                    port,
                    tempfile.mktemp(),
                    results,
                    exit_event,
                    decode_cancelled,
                    backend,
                ),
            ),
        ]
        for p in procs:
            p.start()
        try:
            out = {}
            for _ in procs:
                item = results.get(timeout=TIMEOUT_S + 60)
                out[item[0]] = item[1:]
        finally:
            exit_event.set()
            for p in procs:
                p.join(30)
                if p.is_alive():
                    p.kill()

        prefill_msgs = out["prefill"][0]
        decode_msgs, mismatches, untouched = out["decode"]
        self.assertNotIn("crash", [m[0] for m in prefill_msgs + decode_msgs])

        # Prefill: READY of the cancelled room is answered with FAIL, not reported.
        self.assertEqual(
            kinds(prefill_msgs, "ready"), set(ALL_ROOMS) - {PREFILL_CANCEL_ROOM}
        )
        self.assertEqual(kinds(prefill_msgs, "done", 2), set(ROOMS + CANCELLED_ROOMS))
        self.assertEqual(kinds(prefill_msgs, "fail"), {MISMATCH_ROOM})

        first_tokens = {m[2]: m[3] for m in decode_msgs if m[0] == "done"}
        self.assertEqual(first_tokens, {room: 7 + i for i, room in enumerate(ROOMS)})
        self.assertEqual(
            kinds(decode_msgs, "fail"), {MISMATCH_ROOM, PREFILL_CANCEL_ROOM}
        )
        self.assertEqual(kinds(decode_msgs, "cancelled", 2), set(CANCELLED_ROOMS))
        self.assertEqual(mismatches, [])
        self.assertTrue(untouched, "data of a cancelled room was written")


if __name__ == "__main__":
    unittest.main()
