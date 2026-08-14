# 最终实验结论

## 引擎对比

单并发下，vLLM 输出吞吐为 76.11 token/s，串行 Transformers 为
17.53 token/s，提升 4.34 倍；P95 TTFT 从 104.75 ms 降至 53.09 ms。

并发 32 时，vLLM 达到 923.12 token/s，串行 Transformers 为
17.54 token/s。后者的 P95 TTFT 上升至 226570.55 ms，主要原因是请求在互斥锁
前排队；vLLM 通过连续批处理将 P95 TTFT 控制在 271.32 ms。

## 显存优化

将 `gpu_memory_utilization` 从 0.80 降至 0.35 后，峰值显存从 53052 MiB
降至 23595 MiB，减少 29457 MiB（55.52%）。并发 1～16 的吞吐变化均小于
1%，P95 TTFT 变化除单并发改善外均在约 4% 以内。

优化配置提供约 17.35 GiB KV Cache，可容纳 162432 Token；在单请求最大
4096 Token 的假设下，理论并发约 39.66，覆盖本项目实测并发 32。

## 推荐配置

对于本实验的 Qwen3-1.7B、单张 K500SM_AI、最大并发 32 场景：

```text
dtype=bfloat16
max_model_len=4096
gpu_memory_utilization=0.35
```

该结论不应直接外推到更大模型、更长上下文或不同硬件。生产部署前应根据实际
请求长度和并发分布重新进行容量测试。

