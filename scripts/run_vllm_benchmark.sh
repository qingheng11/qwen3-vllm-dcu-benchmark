#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL_PATH:?请先设置 MODEL_PATH=/path/to/Qwen3-1.7B}"
RESULT_DIR="${PROJECT}/results/raw/vllm"

mkdir -p "$RESULT_DIR"

run_case() {
    local category="$1"
    local input_len="$2"
    local output_len="$3"
    local concurrency="$4"
    local repeat="$5"
    local prompts="$6"

    local name="${category}_in${input_len}_out${output_len}_c${concurrency}_r${repeat}"

    echo
    echo "============================================================"
    echo "开始：$name"
    echo "时间：$(date --iso-8601=seconds)"
    echo "============================================================"

    vllm bench serve \
      --backend vllm \
      --host 127.0.0.1 \
      --port 8000 \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --served-model-name qwen3-1.7b \
      --tokenizer "$MODEL" \
      --dataset-name random \
      --num-prompts "$prompts" \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --random-range-ratio 0 \
      --request-rate inf \
      --max-concurrency "$concurrency" \
      --ignore-eos \
      --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,95,99 \
      --save-result \
      --save-detailed \
      --result-dir "$RESULT_DIR" \
      --result-filename "${name}.json"

    rc=$?

    echo "结束：$name"
    echo "exit=$rc"
    echo "时间：$(date --iso-8601=seconds)"

    if [ "$rc" -ne 0 ]; then
        echo "测试失败：$name"
        return "$rc"
    fi

    sleep 10
}

echo "vLLM正式测试开始：$(date --iso-8601=seconds)"

# 并发扩展：固定输入512、输出128，每组50条，重复3次
for repeat in 1 2 3; do
    for concurrency in 1 2 4 8 16 32; do
        run_case \
          concurrency \
          512 \
          128 \
          "$concurrency" \
          "$repeat" \
          50 || exit $?
    done
done

# Prefill：改变输入长度
for input_len in 128 1024 2048; do
    run_case \
      prefill \
      "$input_len" \
      128 \
      1 \
      1 \
      50 || exit $?
done

# Decoding：改变输出长度
for output_len in 64 256; do
    run_case \
      decoding \
      512 \
      "$output_len" \
      1 \
      1 \
      50 || exit $?
done

echo "vLLM正式测试完成：$(date --iso-8601=seconds)"
