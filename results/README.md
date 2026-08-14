# Results

- `raw/transformers/`：串行 Transformers 服务的 23 组原始结果；
- `raw/vllm/`：vLLM、显存比例 0.80 的 23 组原始结果；
- `raw/vllm_mem035/`：vLLM、显存比例 0.35 的 23 组原始结果；
- `*_run_summary.csv`：从原始 JSON 提取的逐轮指标；
- `engine_*comparison*.csv`：跨引擎标准化对比；
- `vllm_memory_optimization_comparison.csv`：vLLM 优化前后对比；
- `memory_summary.txt`：基线显存监控汇总；
- `vllm_mem035_memory_summary.txt`：优化配置显存汇总。

原始 JSON 的 SHA-256 清单位于
`verification/raw_results_sha256.txt`。

