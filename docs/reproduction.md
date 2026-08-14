# 复现指南

## 1. 环境准备

准备兼容目标硬件的 PyTorch、Transformers、vLLM 和 `rocm-smi`，下载
Qwen3-1.7B 模型，但不要把模型权重提交到本仓库。

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B
export SERVED_MODEL_NAME=qwen3-1.7b
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

## 2. 运行 vLLM 基线

终端 1：

```bash
GPU_MEMORY_UTILIZATION=0.80 bash scripts/serve_vllm.sh \
  2>&1 | tee logs/vllm_server.log
```

终端 2：

```bash
bash scripts/monitor_dcu.sh >logs/vllm_dcu_monitor.log 2>&1 &
MONITOR_PID=$!
bash scripts/run_vllm_benchmark.sh
kill "$MONITOR_PID"
```

## 3. 运行 vLLM 优化配置

停止基线服务后，在终端 1 执行：

```bash
GPU_MEMORY_UTILIZATION=0.35 bash scripts/serve_vllm.sh \
  2>&1 | tee logs/vllm_mem035_server.log
```

终端 2：

```bash
bash scripts/monitor_dcu.sh >logs/vllm_mem035_dcu_monitor.log 2>&1 &
MONITOR_PID=$!
bash scripts/run_vllm_mem035_benchmark.sh
kill "$MONITOR_PID"
```

## 4. 运行串行 Transformers 基线

终端 1：

```bash
python -u scripts/transformers_server.py \
  2>&1 | tee logs/transformers_server.log
```

终端 2：

```bash
bash scripts/monitor_dcu.sh >logs/transformers_dcu_monitor.log 2>&1 &
MONITOR_PID=$!
bash scripts/run_transformers_benchmark.sh
kill "$MONITOR_PID"
```

三个服务使用相同的 `127.0.0.1:8000`，必须依次运行，不能同时启动。

## 5. 汇总结果

```bash
python -m pip install -r requirements-analysis.txt
bash scripts/run_analysis.sh
```

汇总 CSV 位于 `results/`，图表位于 `figures/`。

