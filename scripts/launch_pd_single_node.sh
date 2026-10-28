#!/usr/bin/env bash
# Launch a PD-disaggregated deployment on one node: a prefill server, a decode
# server and the router (sglang.srt.disaggregation.mini_lb). Ctrl-C stops all.
#
#   MODEL=Qwen/Qwen2.5-32B-Instruct PP=4 bash scripts/launch_pd_single_node.sh
#
# The prefill server uses PP*TP GPUs from PREFILL_GPU (default 0) and the decode
# server the next PP*TP GPUs. Benchmark against the router port (default 8000).
#
# Environment variables:
#   MODEL (required), PP, TP
#   PREFILL_PORT, DECODE_PORT, ROUTER_PORT, BOOTSTRAP_PORT
#   PREFILL_GPU, DECODE_GPU
#   COMMON_ARGS   args for both servers; add "--disaggregation-transfer-backend
#                 host" if GPU peer-to-peer copies are broken or disabled
#   PREFILL_ARGS  args for the prefill server only
#   DECODE_ARGS   args for the decode server only
set -euo pipefail

MODEL=${MODEL:?set MODEL to the model path}
PP=${PP:-1}
TP=${TP:-1}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-$((PREFILL_GPU + PP * TP))}
COMMON_ARGS=${COMMON_ARGS:-"--context-length 16384 --mem-fraction-static 0.9 --schedule-policy fcfs --disable-radix-cache --disable-cuda-graph"}
PREFILL_ARGS=${PREFILL_ARGS:-"--chunked-prefill-size 2048"}
DECODE_ARGS=${DECODE_ARGS:-""}

trap 'kill 0' EXIT INT TERM

python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PREFILL_PORT" \
    --tp-size "$TP" --pp-size "$PP" --base-gpu-id "$PREFILL_GPU" \
    --disaggregation-mode prefill --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    $COMMON_ARGS $PREFILL_ARGS &

python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$DECODE_PORT" \
    --tp-size "$TP" --pp-size "$PP" --base-gpu-id "$DECODE_GPU" \
    --disaggregation-mode decode \
    $COMMON_ARGS $DECODE_ARGS &

for port in "$PREFILL_PORT" "$DECODE_PORT"; do
    until curl -sf "http://127.0.0.1:$port/health_generate" > /dev/null; do
        sleep 5
    done
done

python -m sglang.srt.disaggregation.mini_lb \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --host 0.0.0.0 --port "$ROUTER_PORT" &

wait
