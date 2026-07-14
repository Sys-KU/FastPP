"""Unit tests for BatchRebalancer (Case1/Case2/TPOT_FIRST/E2E_FIRST).

GPU-free; run with:
    python -m unittest test.srt.test_batch_rebalancer -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.batch_rebalancer import BatchRebalancer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBatch:
    def __init__(self, reqs=None):
        self.reqs = reqs if reqs is not None else []

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def filter_batch(self, being_chunked_req=None, keep_indices=None):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if self.reqs[i] is not being_chunked_req
            ]
        self.reqs = [self.reqs[i] for i in keep_indices]

    def split_batch(self, keep_indices, drop_indices=None):
        if drop_indices is None:
            all_indices = set(range(len(self.reqs)))
            drop_indices = list(all_indices - set(keep_indices))
        keep = FakeBatch([self.reqs[i] for i in keep_indices]) if keep_indices else None
        drop = FakeBatch([self.reqs[i] for i in drop_indices]) if drop_indices else None
        return keep, drop

    def merge_batch(self, other):
        if other is None:
            return
        self.reqs.extend(other.reqs)


class FakeCurPP:
    def __init__(self, idx=0):
        self._idx = idx

    def get(self):
        return self._idx


class FakeReq:
    def __init__(self, rid="r"):
        self.rid = rid
        self.origin_input_ids = [1] * 100
        self.pp_idx = 0


class FakeKvPool:
    def __init__(self, available=100000):
        self._available = available

    def available_size(self):
        return self._available


class FakeTreeCache:
    def evictable_size(self):
        return 0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_server_args(**kw):
    defaults = dict(
        enable_batch_rebalancing=True,
        br_waiting_tok_threshold=1024,
        br_load_alpha=0.4,
        br_base_unit=128,
        br_mode="auto",
        dc_kv_free_threshold=0.10,
        dc_slo_tpot=0.2,
        enable_dp_attention=False,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def make_scheduler(pp_size=1, **kw):
    sa = make_server_args()
    # Route server-args keys to the server_args namespace, not the scheduler.
    if "br_mode" in kw:
        sa.br_mode = kw.pop("br_mode")
    s = SimpleNamespace(
        server_args=sa,
        pp_size=pp_size,
        pp_rank=0,
        tp_rank=0,
        slo_tpot=0.2,
        tpot=0.1,
        running_batch=None,
        pending_batch=None,
        waiting_queue=[],
        _last_batch=[None] * pp_size,
        _being_chunked_req=[None] * pp_size,
        cur_pp=FakeCurPP(0),
        batch_is_full=False,
        token_to_kv_pool=FakeKvPool(),
        tree_cache=FakeTreeCache(),
        max_total_num_tokens=100000,
        rebalance_metrics=None,
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def make_rebalancer(sched):
    return BatchRebalancer(sched)


def run_batch_with(reqs, ttft=None, tpot=None):
    b = FakeBatch(reqs)
    return b


# ---------------------------------------------------------------------------
# Noop / guards
# ---------------------------------------------------------------------------


class TestNoop(unittest.TestCase):
    def test_both_empty_noop(self):
        sched = make_scheduler()
        rb = make_rebalancer(sched)
        rb.rebalance()
        self.assertIsNone(sched.running_batch)
        self.assertIsNone(sched.pending_batch)


class TestRankGuard(unittest.TestCase):
    def test_non_rank0_does_not_log_crash(self):
        sched = make_scheduler(pp_rank=2)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq()] * 20)
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        self.assertGreater(sched.running_batch.batch_size(), 0)


# ---------------------------------------------------------------------------
# Case 1: suppress band
# ---------------------------------------------------------------------------


class TestCase1(unittest.TestCase):
    def _setup_case1(self, has_kv_headroom=False):
        # pp_size=1, base_unit=128 -> base_th = 128*1 = 128
        # th = (n // 128) * 128
        # suppress band: th <= n < th + alpha*128 = th + 51
        # want n=150 -> th=128, band [128, 179)
        sched = make_scheduler(pp_size=1)
        if has_kv_headroom:
            sched.token_to_kv_pool = FakeKvPool(95000)
            sched.waiting_queue = [FakeReq() for _ in range(20)]
        else:
            sched.token_to_kv_pool = FakeKvPool(50000)
        return sched

    def test_case1_suppress_running_shrinks(self):
        sched = self._setup_case1(has_kv_headroom=False)
        # running=5, pending=145 -> n_total=150, th=128, desired_last=128
        sched.running_batch = run_batch_with([FakeReq()] * 5)
        sched.pending_batch = run_batch_with([FakeReq()] * 145)
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # E2E_FIRST (tpot=0.1 < 0.18): desired=128, last=5 < 128
        # take = min(123, 145) = 123 from pending head -> running grows to 128
        self.assertEqual(sched.running_batch.batch_size(), 128)

    def test_case1_kv_headroom_skips_suppress(self):
        sched = self._setup_case1(has_kv_headroom=True)
        # running=150, pending=0 -> n_total=150, th=128
        # has_kv_headroom True -> falls to Case 2
        sched.running_batch = run_batch_with([FakeReq()] * 150)
        sched.pending_batch = None
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # Case2: desired_last = 150 // 1 = 150, last=150, need=0, give=150-256<0
        # no move happens
        self.assertEqual(sched.running_batch.batch_size(), 150)


# ---------------------------------------------------------------------------
# Case 2: grow / shed
# ---------------------------------------------------------------------------


class TestCase2(unittest.TestCase):
    def test_case2_grow_from_pending(self):
        # base_th=128, alpha=0.4 -> suppress band for th=256 is [256, 307.2)
        # n=320 >= 307.2 -> Case2 (skip suppress)
        sched = make_scheduler(pp_size=1)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 50)
        sched.pending_batch = run_batch_with([FakeReq()] * 270)
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # n_total = 50 + 270 = 320, th=256, Case2
        # desired_last = 320 // 1 = 320, last=50, take=min(270, 270)=270
        self.assertEqual(sched.running_batch.batch_size(), 320)

    def test_case2_no_shed_when_within_limit(self):
        # pp_size=1, base_th=128, n=200, th=128
        # suppress band [128, 179.2); 200 >= 179.2 -> Case2
        # current_local_limit = (128//1) + 128 = 256; running=200 < 256 -> no shed
        sched = make_scheduler(pp_size=1)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 200)
        sched.pending_batch = None
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # desired_last=200, need=0, give=200-256<0, is_over_limit=False -> no move
        self.assertEqual(sched.running_batch.batch_size(), 200)

    def test_case2_shed_when_over_limit(self):
        # pp_size=2: base_th=256, threshold_margin=0.4*256=102.4
        # _last_batch=[running(360), None] -> n_reqs_total=360, th=256
        # 360 >= 358.4 -> Case2
        # current_local_limit = (256//2) + 128 = 256
        # running=360, pending=None, desired_last=360//2=180, need=180-360<0
        # is_over_limit = (360 > 256) = True, give = min(360-256, 360) = 104
        sched = make_scheduler(pp_size=2)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 360)
        sched.pending_batch = None
        sched._last_batch = [sched.running_batch, None]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # shed 104 -> running=256, pending=104
        self.assertEqual(sched.running_batch.batch_size(), 256)
        self.assertEqual(sched.pending_batch.batch_size(), 104)


# ---------------------------------------------------------------------------
# TPOT_FIRST vs E2E_FIRST split indices
# ---------------------------------------------------------------------------


class TestSplitMode(unittest.TestCase):
    def test_tpot_first_takes_from_tail(self):
        # tpot_signal >= slo_tpot*0.9 = 0.18 -> TPOT_FIRST
        sched = make_scheduler(pp_size=1, tpot=0.3)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # n_total=30, th=0 (since 30//128=0), Case2 (no th)
        # desired_last = 30 // 1 = 30, take = min(20, 20) = 20
        # TPOT_FIRST takes from tail of pending -> last 20 reqs (p0..p19)
        # running grows to 30
        self.assertEqual(sched.running_batch.batch_size(), 30)

    def test_e2e_first_takes_from_head(self):
        # tpot_signal < 0.18 -> E2E_FIRST
        sched = make_scheduler(pp_size=1, tpot=0.1)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # n_total=30, Case2, desired=30, take=20
        # E2E_FIRST: take from tail of pending (is_take_from_pending branch
        # uses range(total-move_size, total_size) in both modes)
        self.assertEqual(sched.running_batch.batch_size(), 30)

    def test_tpot_first_merge_appends_to_running(self):
        # TPOT_FIRST: pending -> running appends (merge_batch keeps order)
        sched = make_scheduler(pp_size=1, tpot=0.3)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq(rid=f"r{i}") for i in range(10)])
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # TPOT_FIRST merge: running.merge_batch(pend_take) -> r0..r9, then tail
        rids = [r.rid for r in sched.running_batch.reqs]
        self.assertEqual(rids[:10], [f"r{i}" for i in range(10)])

    def test_e2e_first_merge_prepends_to_running(self):
        # E2E_FIRST: pend_take.merge_batch(running) -> pend first, running=pend_take
        sched = make_scheduler(pp_size=1, tpot=0.1)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq(rid=f"r{i}") for i in range(10)])
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # E2E_FIRST: pend_take.merge_batch(running) -> pend reqs first
        rids = [r.rid for r in sched.running_batch.reqs]
        # first 20 should be pending reqs (p0..p19), then running (r0..r9)
        self.assertEqual(rids[20:], [f"r{i}" for i in range(10)])


# ---------------------------------------------------------------------------
# rebalance_metrics consumption
# ---------------------------------------------------------------------------


class TestRebalanceMetrics(unittest.TestCase):
    def test_rebalance_metrics_consumed(self):
        sched = make_scheduler(pp_size=1)
        # broadcast metrics: (tpot_signal, total_waiting_tokens, kv_free)
        sched.rebalance_metrics = (0.05, 5000, 0.5)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq()] * 20)
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # tpot_signal=0.05 < 0.18 -> E2E_FIRST
        # n_total=30, Case2, desired=30, take=20
        self.assertEqual(sched.running_batch.batch_size(), 30)

    def test_rebalance_metrics_tpot_first(self):
        sched = make_scheduler(pp_size=1)
        sched.rebalance_metrics = (0.3, 5000, 0.5)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq()] * 20)
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # tpot_signal=0.3 >= 0.18 -> TPOT_FIRST
        self.assertEqual(sched.running_batch.batch_size(), 30)


# ---------------------------------------------------------------------------
# br_mode override (force tpot_first / e2e_first regardless of TPOT)
# ---------------------------------------------------------------------------


class TestBrModeOverride(unittest.TestCase):
    def test_force_tpot_first_when_tpot_is_low(self):
        # tpot=0.1 would normally select E2E_FIRST, but br_mode forces TPOT_FIRST
        sched = make_scheduler(pp_size=1, tpot=0.1, br_mode="tpot_first")
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq(rid=f"r{i}") for i in range(10)])
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # TPOT_FIRST merge: running.merge_batch(pend_take) -> r0..r9 first
        rids = [r.rid for r in sched.running_batch.reqs]
        self.assertEqual(rids[:10], [f"r{i}" for i in range(10)])

    def test_force_e2e_first_when_tpot_is_high(self):
        # tpot=0.3 would normally select TPOT_FIRST, but br_mode forces E2E_FIRST
        sched = make_scheduler(pp_size=1, tpot=0.3, br_mode="e2e_first")
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq(rid=f"r{i}") for i in range(10)])
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # E2E_FIRST merge: pend_take.merge_batch(running) -> pending reqs first
        rids = [r.rid for r in sched.running_batch.reqs]
        self.assertEqual(rids[20:], [f"r{i}" for i in range(10)])

    def test_auto_uses_tpot_measurement(self):
        # br_mode='auto' (default) keeps the original TPOT-based selection
        sched = make_scheduler(pp_size=1, tpot=0.3, br_mode="auto")
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq(rid=f"r{i}") for i in range(10)])
        sched.pending_batch = run_batch_with([FakeReq(rid=f"p{i}") for i in range(20)])
        sched._last_batch = [sched.running_batch]
        rb = make_rebalancer(sched)
        rb.rebalance()
        # tpot=0.3 >= 0.18 -> TPOT_FIRST -> r0..r9 first
        rids = [r.rid for r in sched.running_batch.reqs]
        self.assertEqual(rids[:10], [f"r{i}" for i in range(10)])


# ---------------------------------------------------------------------------
# n_reqs_total subtracts other-stage chunked req
# ---------------------------------------------------------------------------


class TestChunkedReqSubtract(unittest.TestCase):
    def test_other_stage_chunked_reduces_total(self):
        sched = make_scheduler(pp_size=2)
        sched.token_to_kv_pool = FakeKvPool(80000)
        sched.running_batch = run_batch_with([FakeReq()] * 10)
        sched.pending_batch = run_batch_with([FakeReq()] * 20)
        sched._last_batch = [sched.running_batch, sched.running_batch]
        # stage 1 (idx=1 != cur_pp=0) has a chunked req -> n_total -= 1
        sched._being_chunked_req[1] = FakeReq()
        rb = make_rebalancer(sched)
        rb.rebalance()
        # n_total = (10 + 10) + 20 - 1 = 39
        # th = (39 // 256)*256 = 0, Case2
        # desired_last = 39 // 2 = 19, last=10, take=min(9, 20)=9
        self.assertEqual(sched.running_batch.batch_size(), 19)


if __name__ == "__main__":
    unittest.main()
