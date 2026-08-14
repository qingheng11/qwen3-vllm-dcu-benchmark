#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

python scripts/summarize_vllm_results.py
python scripts/summarize_vllm_mem035_results.py
python scripts/summarize_transformers_results.py
python scripts/compare_engines_normalized.py
python scripts/compare_vllm_memory_configs.py
python scripts/plot_benchmark_results.py

