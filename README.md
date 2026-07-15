# Revisiting Pipeline Parallelism for LLM Serving

> **OSDI 2026** · Soonjae Hwang, Jeongseob Ahn — Korea University
> Paper: <https://www.usenix.org/conference/osdi26/presentation/hwang>

This repository is the artifact for *Revisiting Pipeline Parallelism for LLM Serving*. It is a fork of [SGLang v0.4.1](https://github.com/sgl-project/sglang/tree/v0.4.1) that implements **Pipeline Parallelism (PP)** for LLM serving and adds two runtime optimizations on top of it.

## Overview

- **Pipeline Parallelism (PP)** — Multi-stage pipeline execution across GPUs via the `--pp` flag, with micro-batch cycling and per-stage layer partitioning.
- **Dynamic Chunk Sizing** — Runtime-adaptive chunked-prefill sizing that adjusts the chunk size based on system load, using either a greedy rule or an ALP (Adaptive Latency Predictor) model with online RLS adaptation.
- **Batch Rebalancing** — Dynamic request rebalancing between running/pending batches (TPOT_FIRST / E2E_FIRST) to improve throughput and SLO attainment under pipeline-parallel execution.

The two optimizations are **disabled by default** and can be enabled via `--enable-dynamic-chunk` and `--enable-batch-rebalancing` at server launch.

---

## Installation

```bash
pip install --upgrade pip
pip install uv
uv pip install -e "python[all]" -c constraints.txt \
    --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

---

## Server Launch

### 2.1 Baseline Server (PP only, optimization features OFF)

```bash
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-mixed-chunk \
    --chunked-prefill-size 2048 \
    --disable-overlap-schedule \
    --pp 4
```

### 2.2 Enable Dynamic Chunk (Greedy strategy)

Rule-based chunk size adjustment without an ALP model.

```bash
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-mixed-chunk \
    --chunked-prefill-size 128 \
    --disable-overlap-schedule \
    --pp 4 \
    --enable-dynamic-chunk \
    --dynamic-chunk-strategy greedy
```

### 2.3 Enable Dynamic Chunk (ALP strategy)

```bash
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-mixed-chunk \
    --chunked-prefill-size 128 \
    --disable-overlap-schedule \
    --pp 4 \
    --enable-dynamic-chunk \
    --dynamic-chunk-strategy alp
```

### 2.4 Enable Batch Rebalancing

```bash
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 --host 0.0.0.0 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-mixed-chunk \
    --chunked-prefill-size 128 \
    --disable-overlap-schedule \
    --pp 4 \
    --enable-dynamic-chunk \
    --dynamic-chunk-strategy alp \
    --enable-batch-rebalancing
```

---

## Client (Benchmark)

Run benchmark clients while the server is running.

### 3.1 Azure Trace (replay real traffic patterns)

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dataset-name azure_trace_conv_2023 \
    --output-file result/result_azure_conv_trace.jsonl \
    --arrival-mode trace \
    --trace-window-minutes 15 \
    --trace-sample-ratio 1 \
    --timestamp-speedup 0.9
```

### 3.2 Fixed Request Rate Sweep

```bash
python -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dataset-name azure_trace_conv_2023 \
    --num-prompt 400 \
    --multi \
    --output-file result/result_azure_conv.jsonl \
    --request-rate-range 1,11,1
```

```bash
python -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dataset-name cnn_r \
    --num-prompt 1000 \
    --multi \
    --output-file result/result_cnn_r.jsonl \
    --request-rate-range 1,12,1
```

---

## Server Args Reference

### Dynamic Chunk Sizing

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--enable-dynamic-chunk` | flag | `False` | Enable the feature |
| `--dynamic-chunk-strategy` | str | `alp` | `greedy` or `alp` |
| `--dc-slo-ttft` | float | `2.0` | SLO TTFT target (seconds) |
| `--dc-slo-tpot` | float | `0.2` | SLO TPOT target (seconds) |
| `--dc-kv-free-threshold` | float | `0.10` | KV cache free-ratio threshold for chunk shrink (shared by Dynamic Chunk & Batch Rebalancer) |
| `--dc-greedy-base-chunk` | int | `128` | Greedy: chunk adjustment step size |
| `--dc-greedy-max-chunk` | int | `2048` | Greedy: maximum chunk size |
| `--dc-greedy-min-chunk` | int | `128` | Greedy: minimum chunk size |
| `--dc-greedy-tpot-violation-ratio` | float | `1.08` | Greedy: TPOT violation multiplier |
| `--dc-greedy-ttft-low-ratio` | float | `0.1` | Greedy: TTFT low-load multiplier |
| `--dc-greedy-ttft-high-ratio` | float | `0.3` | Greedy: TTFT high-load multiplier |
| `--dc-greedy-tpot-safe-ratio` | float | `0.85` | Greedy: TPOT safe multiplier |
| `--dc-alp-model-path` | str | `alp_wait_time_chunk_rls.json` | ALP runtime table (JSON) path |
| `--dc-alp-train` | flag | `True` | Enable ALP startup profiling |
| `--dc-alp-chunk-min` | int | `128` | Minimum candidate chunk size for ALP |
| `--dc-alp-chunk-max` | int | `2048` | Maximum candidate chunk size for ALP |
| `--dc-alp-chunk-step` | int | `128` | Step between candidate chunk sizes for ALP |
| `--dc-alp-fresh-epsilon` | float | `0.25` | ALP: stale request threshold |
| `--dc-alp-slo-tpot-coeff` | float | `1.08` | ALP: SLO TPOT margin for chunk selection (noise-tolerant ceiling) |
| `--dc-alp-throughput-coeff` | float | `0.92` | ALP: required throughput margin for chunk selection (stabilization factor) |

### Batch Rebalancing

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--enable-batch-rebalancing` | flag | `False` | Enable the feature |
| `--br-waiting-tok-threshold` | int | `1024` | Overload detection token threshold |
| `--br-load-alpha` | float | `0.4` | Load balancing factor |
| `--br-base-unit` | int | `128` | Base unit for batch sizing |
| `--br-mode` | str | `auto` | Rebalance mode: `auto` (E2E_FIRST, switches to TPOT_FIRST when TPOT >= 90% of SLO), `tpot_first`, or `e2e_first` (forced) |

---

## License

Released under the [Apache License 2.0](LICENSE), inherited from the upstream SGLang project.
