import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_FIT_ON_STARTUP = True
DEFAULT_PROFILE_MODE = "scheduler_loop_runtime_aligned_no_attention"
DEFAULT_MODEL_PATH = "alp_wait_time_chunk_rls.json"
DEFAULT_PROFILE_WARMUP = 3
DEFAULT_PROFILE_REPEATS = 5
DEFAULT_CANDIDATE_CHUNK_MIN = 128
DEFAULT_CANDIDATE_CHUNK_MAX = 2048
DEFAULT_CANDIDATE_CHUNK_STEP = 128
DEFAULT_RUNTIME_OUTLIER_SKIP_THRESHOLD = 0.4

# Online RLS model hyperparameters.
DEFAULT_FORGET_FACTOR = 0.9975
DEFAULT_RLS_DELTA = 1.25

# Feature normalization scales.
DEFAULT_CTX_NORM_SCALE = 100000.0
DEFAULT_PREFILL_TOK_NORM_SCALE = 1024.0 * 1024.0
DEFAULT_PREFIX_SUM_NORM_SCALE = 1024.0 * 2048.0

MIN_RELATIVE_ERROR_DENOM = 1e-9
MIN_RLS_DENOM = 1e-12
INTERFERENCE_FEATURE_DIM = 4
INTERFERENCE_BIAS_FEATURE = 0.1


class ALPScheduler:
    """Chunk runtime table used by alp_dynamic.

    Startup profiling measures scheduler-style steady-state PP micro-steps for
    every candidate chunk size. During runtime, execution time is modeled as the
    startup micro-step time plus online RLS corrections from decode/prefill features.
    """

    def __init__(
        self,
        pp_size: int = 4,
        pp_rank: int = 0,
        tp_rank: int = 0,
        candidate_chunks: Optional[List[int]] = None,
        candidate_chunk_min: int = DEFAULT_CANDIDATE_CHUNK_MIN,
        candidate_chunk_max: int = DEFAULT_CANDIDATE_CHUNK_MAX,
        candidate_chunk_step: int = DEFAULT_CANDIDATE_CHUNK_STEP,
        coeff_path: str = DEFAULT_MODEL_PATH,
        fit_on_startup: bool = DEFAULT_FIT_ON_STARTUP,
        profile_warmup: int = DEFAULT_PROFILE_WARMUP,
        profile_repeats: int = DEFAULT_PROFILE_REPEATS,
    ):
        self.pp_size = int(pp_size)
        self.pp_rank = int(pp_rank)
        self.tp_rank = int(tp_rank)
        self.fit_on_startup = bool(fit_on_startup)
        self.coeff_path = coeff_path
        self.profile_warmup = int(profile_warmup)
        self.profile_repeats = int(profile_repeats)
        self.candidate_chunk_min = int(candidate_chunk_min)
        self.candidate_chunk_max = int(candidate_chunk_max)
        self.candidate_chunk_step = max(int(candidate_chunk_step), 1)

        default_chunks = list(
            range(
                self.candidate_chunk_min,
                self.candidate_chunk_max + 1,
                self.candidate_chunk_step,
            )
        )
        self.candidate_chunks = sorted(int(chunk) for chunk in (candidate_chunks or default_chunks))
        if self.candidate_chunks:
            self.candidate_chunk_min = self.candidate_chunks[0]
            self.candidate_chunk_max = self.candidate_chunks[-1]

        self.runtime_outlier_skip_threshold = DEFAULT_RUNTIME_OUTLIER_SKIP_THRESHOLD
        self.forget_factor = DEFAULT_FORGET_FACTOR
        self.rls_delta = DEFAULT_RLS_DELTA
        self.ctx_norm_scale = DEFAULT_CTX_NORM_SCALE
        self.prefill_tok_norm_scale = DEFAULT_PREFILL_TOK_NORM_SCALE
        self.prefix_sum_norm_scale = DEFAULT_PREFIX_SUM_NORM_SCALE
        self.prefill_exec_time_by_chunk = {
            int(chunk): 0.0 for chunk in self.candidate_chunks
        }
        self.interference_theta = np.zeros(INTERFERENCE_FEATURE_DIM, dtype=float)
        self.interference_cov = (
            np.eye(INTERFERENCE_FEATURE_DIM, dtype=float) * self.rls_delta
        )
        self.interference_updates = 0
        self.runtime_table: Dict[str, Any] = {}

        self.runtime_table = self._load_runtime_table()
        self._load_runtime_table_state(self.runtime_table)

    def _load_runtime_table(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.coeff_path):
            if self.fit_on_startup:
                logger.info(
                    "runtime predictor state not found yet. "
                    f"Startup profiling will create it: {self.coeff_path}"
                )
            return None

        with open(self.coeff_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        model_kind = str(raw.get("model_kind", "")).strip().lower()
        if model_kind != "chunk_rls":
            raise ValueError(
                f"unsupported runtime predictor kind in {self.coeff_path}: {model_kind!r}"
            )

        logger.info(f"chunk-RLS runtime table loaded: {self.coeff_path}")
        return raw

    def _load_runtime_table_state(self, bundle: Optional[Dict[str, Any]]) -> None:
        if not bundle:
            return

        self.runtime_outlier_skip_threshold = float(
            bundle.get(
                "runtime_outlier_skip_threshold",
                self.runtime_outlier_skip_threshold,
            )
        )
        self.forget_factor = float(bundle.get("forget_factor", self.forget_factor))
        self.rls_delta = float(bundle.get("rls_delta", self.rls_delta))
        self.ctx_norm_scale = float(bundle.get("ctx_norm_scale", self.ctx_norm_scale))
        self.prefill_tok_norm_scale = float(
            bundle.get("prefill_tok_norm_scale", self.prefill_tok_norm_scale)
        )
        self.prefix_sum_norm_scale = float(
            bundle.get("prefix_sum_norm_scale", self.prefix_sum_norm_scale)
        )
        raw_chunk_exec_times = bundle.get("chunk_exec_times", {})

        for chunk in self.candidate_chunks:
            chunk_key = int(chunk)
            raw_value = raw_chunk_exec_times.get(str(chunk_key), raw_chunk_exec_times.get(chunk_key))
            if raw_value is None:
                continue
            self.prefill_exec_time_by_chunk[chunk_key] = max(float(raw_value), 1e-9)

        theta = bundle.get("interference_theta", [0.0, 0.0, 0.0, 0.0])
        self.interference_theta = np.zeros(INTERFERENCE_FEATURE_DIM, dtype=float)
        loaded_theta = np.array(theta, dtype=float).reshape(-1)
        copy_theta = min(loaded_theta.shape[0], INTERFERENCE_FEATURE_DIM)
        if copy_theta > 0:
            self.interference_theta[:copy_theta] = loaded_theta[:copy_theta]

        cov = bundle.get("interference_cov")
        if cov is None:
            self.interference_cov = (
                np.eye(INTERFERENCE_FEATURE_DIM, dtype=float) * self.rls_delta
            )
        else:
            self.interference_cov = (
                np.eye(INTERFERENCE_FEATURE_DIM, dtype=float) * self.rls_delta
            )
            loaded_cov = np.array(cov, dtype=float)
            if loaded_cov.ndim == 2:
                copy_rows = min(loaded_cov.shape[0], INTERFERENCE_FEATURE_DIM)
                copy_cols = min(loaded_cov.shape[1], INTERFERENCE_FEATURE_DIM)
                self.interference_cov[:copy_rows, :copy_cols] = loaded_cov[
                    :copy_rows, :copy_cols
                ]

        self.interference_updates = int(bundle.get("interference_updates", 0))

    def _build_interference_features(
        self,
        decode_ctx_sum: float = 0.0,
        prefill_square_sum: float = 0.0,
        prefill_prefix_prod_sum: float = 0.0,
    ) -> np.ndarray:
        return np.array(
            [
                float(decode_ctx_sum) / float(self.ctx_norm_scale),
                float(prefill_square_sum) / float(self.prefill_tok_norm_scale),
                float(prefill_prefix_prod_sum) / float(self.prefix_sum_norm_scale),
                INTERFERENCE_BIAS_FEATURE,
            ],
            dtype=float,
        )

    def _normalize_interference_features(
        self,
        decode_ctx_sum: float = 0.0,
        prefill_square_sum: float = 0.0,
        prefill_prefix_prod_sum: float = 0.0,
    ) -> tuple[float, float, float, float]:
        return (
            float(decode_ctx_sum) / float(self.ctx_norm_scale),
            float(prefill_square_sum) / float(self.prefill_tok_norm_scale),
            float(prefill_prefix_prod_sum) / float(self.prefix_sum_norm_scale),
            INTERFERENCE_BIAS_FEATURE,
        )

    def update_runtime_overhead(
        self,
        observed_exec_time: float,
        runtime_chunk_size: Optional[int] = None,
        decode_ctx_sum: float = 0.0,
        prefill_square_sum: float = 0.0,
        prefill_prefix_prod_sum: float = 0.0,
    ) -> None:
        if runtime_chunk_size is None:
            return

        observed = float(observed_exec_time)
        if observed <= 0.0:
            return

        runtime_chunk_key = int(runtime_chunk_size)

        prefill_exec_time = float(
            self.prefill_exec_time_by_chunk.get(runtime_chunk_key, 0.0)
        )
        if prefill_exec_time <= 0.0:
            return

        x0, x1, x2, x3 = self._normalize_interference_features(
            decode_ctx_sum=decode_ctx_sum,
            prefill_square_sum=prefill_square_sum,
            prefill_prefix_prod_sum=prefill_prefix_prod_sum,
        )
        theta0 = float(self.interference_theta[0])
        theta1 = float(self.interference_theta[1])
        theta2 = float(self.interference_theta[2])
        theta3 = float(self.interference_theta[3])
        pred_total = float(
            prefill_exec_time
            + (theta0 * x0)
            + (theta1 * x1)
            + (theta2 * x2)
            + (theta3 * x3)
        )
        update_err = float(observed - pred_total)
        if self.runtime_outlier_skip_threshold > 0.0:
            pred_total_denom = max(abs(pred_total), MIN_RELATIVE_ERROR_DENOM)
            relative_error = abs(update_err) / pred_total_denom
            if relative_error > self.runtime_outlier_skip_threshold:
                max_delta = self.runtime_outlier_skip_threshold * pred_total_denom
                update_err = float(np.clip(update_err, -max_delta, max_delta))

        p00 = float(self.interference_cov[0, 0])
        p01 = float(self.interference_cov[0, 1])
        p02 = float(self.interference_cov[0, 2])
        p03 = float(self.interference_cov[0, 3])
        p10 = float(self.interference_cov[1, 0])
        p11 = float(self.interference_cov[1, 1])
        p12 = float(self.interference_cov[1, 2])
        p13 = float(self.interference_cov[1, 3])
        p20 = float(self.interference_cov[2, 0])
        p21 = float(self.interference_cov[2, 1])
        p22 = float(self.interference_cov[2, 2])
        p23 = float(self.interference_cov[2, 3])
        p30 = float(self.interference_cov[3, 0])
        p31 = float(self.interference_cov[3, 1])
        p32 = float(self.interference_cov[3, 2])
        p33 = float(self.interference_cov[3, 3])

        px0 = (p00 * x0) + (p01 * x1) + (p02 * x2) + (p03 * x3)
        px1 = (p10 * x0) + (p11 * x1) + (p12 * x2) + (p13 * x3)
        px2 = (p20 * x0) + (p21 * x1) + (p22 * x2) + (p23 * x3)
        px3 = (p30 * x0) + (p31 * x1) + (p32 * x2) + (p33 * x3)
        gain_denom = float(
            self.forget_factor + (x0 * px0) + (x1 * px1) + (x2 * px2) + (x3 * px3)
        )
        if gain_denom < MIN_RLS_DENOM:
            gain_denom = MIN_RLS_DENOM

        g0 = px0 / gain_denom
        g1 = px1 / gain_denom
        g2 = px2 / gain_denom
        g3 = px3 / gain_denom

        xp0 = (x0 * p00) + (x1 * p10) + (x2 * p20) + (x3 * p30)
        xp1 = (x0 * p01) + (x1 * p11) + (x2 * p21) + (x3 * p31)
        xp2 = (x0 * p02) + (x1 * p12) + (x2 * p22) + (x3 * p32)
        xp3 = (x0 * p03) + (x1 * p13) + (x2 * p23) + (x3 * p33)

        theta0 += g0 * update_err
        theta1 += g1 * update_err
        theta2 += g2 * update_err
        theta3 += g3 * update_err

        ff = float(self.forget_factor)
        p00 = (p00 - (g0 * xp0)) / ff
        p01 = (p01 - (g0 * xp1)) / ff
        p02 = (p02 - (g0 * xp2)) / ff
        p03 = (p03 - (g0 * xp3)) / ff
        p10 = (p10 - (g1 * xp0)) / ff
        p11 = (p11 - (g1 * xp1)) / ff
        p12 = (p12 - (g1 * xp2)) / ff
        p13 = (p13 - (g1 * xp3)) / ff
        p20 = (p20 - (g2 * xp0)) / ff
        p21 = (p21 - (g2 * xp1)) / ff
        p22 = (p22 - (g2 * xp2)) / ff
        p23 = (p23 - (g2 * xp3)) / ff
        p30 = (p30 - (g3 * xp0)) / ff
        p31 = (p31 - (g3 * xp1)) / ff
        p32 = (p32 - (g3 * xp2)) / ff
        p33 = (p33 - (g3 * xp3)) / ff

        self.interference_theta[0] = theta0
        self.interference_theta[1] = theta1
        self.interference_theta[2] = theta2
        self.interference_theta[3] = theta3
        self.interference_cov[0, 0] = p00
        self.interference_cov[0, 1] = p01
        self.interference_cov[0, 2] = p02
        self.interference_cov[0, 3] = p03
        self.interference_cov[1, 0] = p10
        self.interference_cov[1, 1] = p11
        self.interference_cov[1, 2] = p12
        self.interference_cov[1, 3] = p13
        self.interference_cov[2, 0] = p20
        self.interference_cov[2, 1] = p21
        self.interference_cov[2, 2] = p22
        self.interference_cov[2, 3] = p23
        self.interference_cov[3, 0] = p30
        self.interference_cov[3, 1] = p31
        self.interference_cov[3, 2] = p32
        self.interference_cov[3, 3] = p33
        self.interference_updates = int(self.interference_updates) + 1

    def reset_runtime_calibration(self) -> None:
        self.prefill_exec_time_by_chunk = {
            int(chunk): 0.0 for chunk in self.candidate_chunks
        }
        self.interference_theta = np.zeros(INTERFERENCE_FEATURE_DIM, dtype=float)
        self.interference_cov = (
            np.eye(INTERFERENCE_FEATURE_DIM, dtype=float) * self.rls_delta
        )
        self.interference_updates = 0
        self._load_runtime_table_state(self.runtime_table)

    def _is_profile_case_feasible(
        self, model_runner: Any, prefill_len: int, prefix_len: int = 0
    ) -> bool:
        req_slots = max(1, int(self.pp_size))
        kv_slots = int(prefill_len) + int(prefix_len)
        max_seq_len = int(prefill_len) + int(prefix_len)
        req_to_token_pool = model_runner.req_to_token_pool
        token_to_kv_pool = model_runner.token_to_kv_pool

        if req_slots > int(req_to_token_pool.size):
            return False
        if kv_slots > int(token_to_kv_pool.size):
            return False
        if max_seq_len > int(req_to_token_pool.max_context_len):
            return False
        return True

    def _aggregate_profile_outputs(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("No chunk-RLS profiling rows found to aggregate.")

        chunk_exec_times: Dict[str, float] = {}
        for chunk in self.candidate_chunks:
            chunk_key = int(chunk)
            values = [
                float(row["measured_exec_time"])
                for row in rows
                if int(row["chunk_size"]) == chunk_key
            ]
            if not values:
                raise ValueError(f"Missing startup profile for chunk={chunk_key}")
            chunk_exec_times[str(chunk_key)] = float(np.max(values))

        payload = {
            "model_kind": "chunk_rls",
            "profile_mode": DEFAULT_PROFILE_MODE,
            "profile_aggregation": "rank0_only",
            "runtime_outlier_skip_threshold": float(
                self.runtime_outlier_skip_threshold
            ),
            "forget_factor": float(self.forget_factor),
            "rls_delta": float(self.rls_delta),
            "ctx_norm_scale": float(self.ctx_norm_scale),
            "prefill_tok_norm_scale": float(self.prefill_tok_norm_scale),
            "prefix_sum_norm_scale": float(self.prefix_sum_norm_scale),
            "candidate_chunks": [int(chunk) for chunk in self.candidate_chunks],
            "chunk_exec_times": chunk_exec_times,
            "chunk_exec_time_updates": {
                str(int(chunk)): 0 for chunk in self.candidate_chunks
            },
            "interference_theta": [0.0, 0.0, 0.0, 0.0],
            "interference_cov": (
                np.eye(INTERFERENCE_FEATURE_DIM, dtype=float) * self.rls_delta
            ).tolist(),
            "interference_updates": 0,
            "num_stages": self.pp_size,
            "num_samples": int(len(rows)),
        }
        with open(self.coeff_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        logger.info(
            "aggregated chunk-RLS profiling outputs: "
            f"stages={self.pp_size}, samples={len(rows)}, coeffs={self.coeff_path}"
        )

    def predict_exec_time_for_chunk(
        self,
        chunk_size: int,
        chunk_features: Optional[Dict[str, float]] = None,
        decode_ctx_sum: float = 0.0,
        prefill_square_sum: float = 0.0,
        prefill_prefix_prod_sum: float = 0.0,
    ) -> float:
        chunk_key = int(chunk_size)
        prefill_exec_time = float(
            self.prefill_exec_time_by_chunk.get(chunk_key, 0.0)
        )
        if prefill_exec_time <= 0.0:
            raise RuntimeError(
                "chunk-RLS runtime table is missing startup measurements for "
                f"chunk={chunk_key}"
            )

        chunk_features = chunk_features or {}
        feature_decode_ctx_sum = float(
            chunk_features.get("selected_decode_ctx_sum", decode_ctx_sum)
        )
        feature_prefill_square_sum = float(
            chunk_features.get("prefill_square_sum", prefill_square_sum)
        )
        feature_prefill_prefix_prod_sum = float(
            chunk_features.get(
                "prefill_prefix_prod_sum", prefill_prefix_prod_sum
            )
        )
        x0, x1, x2, x3 = self._normalize_interference_features(
            decode_ctx_sum=feature_decode_ctx_sum,
            prefill_square_sum=feature_prefill_square_sum,
            prefill_prefix_prod_sum=feature_prefill_prefix_prod_sum,
        )
        theta0 = float(self.interference_theta[0])
        theta1 = float(self.interference_theta[1])
        theta2 = float(self.interference_theta[2])
        theta3 = float(self.interference_theta[3])
        contribution0 = theta0 * x0
        contribution1 = theta1 * x1
        contribution2 = theta2 * x2
        contribution3 = theta3 * x3
        interference = float(contribution0 + contribution1 + contribution2 + contribution3)
        prefill_exec_time = float(max(prefill_exec_time, 1e-9))
        exec_time_pred = float(max(prefill_exec_time + interference, 1e-9))
        return exec_time_pred
