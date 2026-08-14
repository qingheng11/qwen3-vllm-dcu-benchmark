#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL_PATH:?请先设置 MODEL_PATH=/path/to/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --generation-config vllm

