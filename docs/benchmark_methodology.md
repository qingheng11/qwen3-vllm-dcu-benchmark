# 测试方法与公平性边界

## 目标与控制变量

实验比较串行 Transformers 服务、vLLM 基线配置（显存比例 0.80）和 vLLM 优化
配置（显存比例 0.35）。三组保持模型、单卡设备、BF16、最大上下文 4096、随机
种子 42、每组 50 个请求及输入/输出长度一致。

并发测试固定输入 512、输出 128 Token，并发为 1/2/4/8/16/32，各重复 3 次。
Prefill 测试固定输出 128 Token，输入为 128/1024/2048；Decoding 测试固定输入
512 Token，输出为 64/256。

## 指标

- Output Throughput：每秒生成的输出 Token 数；
- TTFT：首 Token 延迟；
- TPOT：后续每 Token 平均耗时；
- E2EL：端到端请求延迟；
- P50/P95/P99：延迟分位数；
- Peak VRAM：监控期间单卡最高显存占用。

## Transformers 基线边界

`transformers_server.py` 使用互斥锁串行执行 `model.generate()`，代表没有动态批处理
的简单在线服务，不代表 Transformers 能达到的最高批量推理性能。高并发时请求会
排队，而 vLLM 会连续批处理多个请求。因此高并发结果用于展示服务架构和调度差异，
不是底层算子的完全等价微基准。

## Token 统计标准化

vLLM 记录服务端生成 Token 数。自定义 Transformers 流式接口返回文本后，客户端
重新 Tokenize，特殊 Token 或空白可能造成 0～5.33% 的计数偏差。由于所有请求均
固定输出长度并忽略 EOS，跨引擎吞吐使用以下公式标准化：

```text
目标输出长度 × 完成请求数 ÷ 测试时长
```

原始 JSON 未删除，便于复核该差异。

## 显存口径

vLLM 显存包含模型权重、运行时缓冲区和预留 KV Cache。因此观测到的 51.81 GiB
不是模型权重体积，而是 `gpu_memory_utilization=0.80` 下的总体显存占用。降低参数
主要减少 KV Cache 预留空间。

## 局限性

- 仅测试 Qwen3-1.7B 和单张 K500SM_AI；
- 未测试量化、多卡 Tensor Parallel 和真实业务负载；
- Prefill/Decoding 单项仅执行一次；
- 并发 32 的优化结果可能包含缓存状态和运行波动；
- 不应将高并发加速比直接表述为算子加速比。

