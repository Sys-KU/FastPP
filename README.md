# Revisiting Pipeline Parallelism for LLM Serving

> **OSDI 2026** · Soonjae Hwang, Jeongseob Ahn — Korea University
> Paper: <https://www.usenix.org/conference/osdi26/presentation/hwang>

This repository is the artifact for *Revisiting Pipeline Parallelism for LLM Serving*. It is a fork of [SGLang v0.4.1](https://github.com/sgl-project/sglang/tree/v0.4.1) that implements **Pipeline Parallelism (PP)** for LLM serving and adds two runtime optimizations on top of it.

## Overview

- **Pipeline Parallelism (PP)** — Multi-stage pipeline execution across GPUs via the `--pp` flag, with micro-batch cycling and per-stage layer partitioning.
- **Dynamic Chunk Sizing** — Runtime-adaptive chunked-prefill sizing that adjusts the chunk size based on system load, using either a greedy rule or an ALP (Adaptive Latency Predictor) model with online RLS adaptation.
- **Batch Rebalancing** — Dynamic request rebalancing between running/pending batches (TPOT_FIRST / E2E_FIRST) to improve throughput and SLO attainment under pipeline-parallel execution. The default `adaptive` mode widens the rebalancing band only under KV pressure (see the Batch Rebalancing arguments).
- **PD Disaggregation** — Prefill and decode on separate servers of one node with the same TP/PP layout, the prompt KV copied rank to rank (GPU peer-to-peer by default), and a router (`sglang.srt.disaggregation.mini_lb`).

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
    --dynamic-chunk-strategy alp_deadline
```

`alp_deadline` (default) keeps ALP's predictor and picks the smallest chunk with which every waiting request meets its TTFT deadline, under a TPOT limit estimated from recent iterations.

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
    --dynamic-chunk-strategy alp_deadline \
    --enable-batch-rebalancing
```

This uses the default `--br-mode adaptive`.

### 2.5 PD Disaggregation (single node)

Prefill and decode on separate GPUs of one node (same model, TP and PP size), with a router in front.

```bash
# Prefill server (GPUs 0-3)
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 --host 127.0.0.1 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --chunked-prefill-size 2048 \
    --pp 4 --base-gpu-id 0 \
    --disaggregation-mode prefill

# Decode server (GPUs 4-7)
NCCL_P2P_DISABLE=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30001 --host 127.0.0.1 \
    --context-length 16384 \
    --mem-fraction-static 0.9 \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --disable-cuda-graph \
    --pp 4 --base-gpu-id 4 \
    --disaggregation-mode decode

# Router (send requests here)
python -m sglang.srt.disaggregation.mini_lb \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --port 8000
```

`scripts/launch_pd_single_node.sh` launches all three. Without GPU peer-to-peer, add `--disaggregation-transfer-backend host` to both servers.

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
| `--dynamic-chunk-strategy` | str | `alp_deadline` | `alp_deadline`, `alp` or `greedy` |
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
| `--dc-alp-fresh-epsilon` | float | `0.25` | `alp` only: stale request threshold |
| `--dc-alp-slo-tpot-coeff` | float | `1.05` | ALP: SLO TPOT margin for chunk selection (noise-tolerant ceiling) |
| `--dc-alp-throughput-coeff` | float | `1.0` | `alp` only: required throughput margin for chunk selection (stabilization factor) |
| `--dc-alp-window-sec` | float | `10.0` | ALP: time window (sec) for QPS measurement |

### Batch Rebalancing

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--enable-batch-rebalancing` | flag | `False` | Enable the feature |
| `--br-waiting-tok-threshold` | int | `1024` | Overload detection token threshold |
| `--br-load-alpha` | float | `0.4` | Load balancing factor (band width) of the `auto`, `tpot_first` and `e2e_first` modes |
| `--br-base-unit` | int | `128` | Base unit for batch sizing |
| `--br-mode` | str | `adaptive` | Rebalance mode: `adaptive` (E2E_FIRST with a band width that follows the KV pressure, below), `auto` (E2E_FIRST, switches to TPOT_FIRST when TPOT >= 90% of SLO), `tpot_first`, or `e2e_first` (forced) |
| `--br-alpha-low` | float | `0.2` | Adaptive: band width without KV pressure |
| `--br-alpha-high` | float | `0.8` | Adaptive: band width under KV pressure (admission blocked by KV, or the admission budget with the remaining outputs of all running and paused requests reserved below `--br-kv-pressure` of the KV pool) |
| `--br-kv-pressure` | float | `0.05` | Adaptive: KV pressure threshold (fraction of the KV pool) |
| `--br-hold-s` | float | `2.0` | Adaptive: seconds the wide band stays after the last KV pressure |
| `--br-margin` | int | `16` | Adaptive: requests a capped band extends by (hysteresis) |
| `--br-arrival-window-s` | float | `5.0` | Adaptive: KV pressure counts only while a new (never retracted) request waits or one arrived within this many seconds |

### PD Disaggregation

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--disaggregation-mode` | str | `null` | `prefill`, `decode`, or `null` (normal server) |
| `--disaggregation-bootstrap-port` | int | `8998` | Prefill rank (stage *i*, TP rank *j*) accepts its decode peer on 127.0.0.1, port + *i* × tp_size + *j* |
| `--disaggregation-transfer-backend` | str | `memcpy` | Same on both servers. `memcpy` / `kernel`: copy straight into the decode KV pool over GPU peer-to-peer (`cuMemcpyBatchAsync` / a copy kernel), one copy per prefill chunk. `gpu` / `host`: stage the KV in GPU or pinned host memory (`host` needs no peer-to-peer) |
| `--disaggregation-transfer-chunk-tokens` | int | `1024` | `gpu` / `host` only: tokens of KV per staging slot; a prefill chunk's KV is sent in pieces of this size |
| `--disaggregation-staging-slots` | int | `2` | `gpu` / `host` only: staging slots per GPU, allocated after the KV pool |
| `--disaggregation-decode-prealloc` | str | `safe` | Decode admission. `safe`: for ignore_eos requests without the radix cache, admit only if decoding every request to its end fits in the KV; others fall back to `reserve`. `reserve` (upstream SGLang): keep `--disaggregation-decode-reserved-tokens` per running request |
| `--disaggregation-decode-reserved-tokens` | int | `512` | `reserve` admission: KV tokens kept free per running request when preallocating a new request's KV |
| `--disaggregation-timeout` | float | `1800` | Seconds a PD request may wait for the other server before it is aborted |

Not supported with PD: multiple nodes, data parallelism, constrained decoding and `n > 1`. Batch rebalancing on the decode server has been checked in simulation only.

---

## License

Released under the [Apache License 2.0](LICENSE), inherited from the upstream SGLang project.
