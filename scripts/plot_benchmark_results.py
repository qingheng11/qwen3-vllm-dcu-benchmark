import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_concurrency(path, normalized=False):
    grouped = defaultdict(list)

    for row in load_csv(path):
        if row["category"] != "concurrency":
            continue

        concurrency = int(row["concurrency"])

        if normalized:
            throughput = (
                int(row["num_prompts"])
                * int(row["output_len"])
                / float(row["duration_s"])
            )
        else:
            throughput = float(row["output_throughput_tps"])

        grouped[concurrency].append({
            "throughput": throughput,
            "ttft": float(row["p95_ttft_ms"]),
        })

    result = {}

    for concurrency, rows in grouped.items():
        result[concurrency] = {
            "throughput": statistics.mean(
                x["throughput"] for x in rows
            ),
            "ttft": statistics.mean(
                x["ttft"] for x in rows
            ),
        }

    return result


def aggregate_prefill(path):
    grouped = defaultdict(list)

    for row in load_csv(path):
        if row["category"] != "prefill":
            continue

        input_len = int(row["input_len"])
        grouped[input_len].append(float(row["p95_ttft_ms"]))

    return {
        input_len: statistics.mean(values)
        for input_len, values in grouped.items()
    }


transformers = aggregate_concurrency(
    RESULTS / "transformers_run_summary.csv",
    normalized=True,
)

vllm_baseline = aggregate_concurrency(
    RESULTS / "vllm_run_summary.csv"
)

vllm_optimized = aggregate_concurrency(
    RESULTS / "vllm_mem035_run_summary.csv"
)

concurrency = sorted(
    set(transformers)
    & set(vllm_baseline)
    & set(vllm_optimized)
)

# 图1：并发吞吐
plt.figure(figsize=(8, 5))
plt.plot(
    concurrency,
    [transformers[x]["throughput"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="Transformers serial",
)
plt.plot(
    concurrency,
    [vllm_baseline[x]["throughput"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.80",
)
plt.plot(
    concurrency,
    [vllm_optimized[x]["throughput"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.35",
)
plt.xlabel("Concurrency")
plt.ylabel("Output throughput (token/s)")
plt.title("Throughput under Different Concurrency Levels")
plt.xticks(concurrency)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES / "concurrency_throughput.png",
    dpi=200,
)
plt.close()

# 图2：并发P95 TTFT，使用对数纵轴
plt.figure(figsize=(8, 5))
plt.plot(
    concurrency,
    [transformers[x]["ttft"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="Transformers serial",
)
plt.plot(
    concurrency,
    [vllm_baseline[x]["ttft"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.80",
)
plt.plot(
    concurrency,
    [vllm_optimized[x]["ttft"] for x in concurrency],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.35",
)
plt.xlabel("Concurrency")
plt.ylabel("P95 TTFT (ms, log scale)")
plt.title("P95 Time to First Token")
plt.xticks(concurrency)
plt.yscale("log")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES / "concurrency_p95_ttft.png",
    dpi=200,
)
plt.close()

# 图3：不同输入长度下的P95 TTFT
tf_prefill = aggregate_prefill(
    RESULTS / "transformers_run_summary.csv"
)
vllm_prefill = aggregate_prefill(
    RESULTS / "vllm_run_summary.csv"
)
optimized_prefill = aggregate_prefill(
    RESULTS / "vllm_mem035_run_summary.csv"
)

input_lengths = sorted(
    set(tf_prefill)
    & set(vllm_prefill)
    & set(optimized_prefill)
)

plt.figure(figsize=(8, 5))
plt.plot(
    input_lengths,
    [tf_prefill[x] for x in input_lengths],
    marker="o",
    linewidth=2,
    label="Transformers serial",
)
plt.plot(
    input_lengths,
    [vllm_prefill[x] for x in input_lengths],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.80",
)
plt.plot(
    input_lengths,
    [optimized_prefill[x] for x in input_lengths],
    marker="o",
    linewidth=2,
    label="vLLM memory=0.35",
)
plt.xlabel("Input length (tokens)")
plt.ylabel("P95 TTFT (ms)")
plt.title("Prefill Latency under Different Input Lengths")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES / "prefill_p95_ttft.png",
    dpi=200,
)
plt.close()

# 图4：vLLM优化前后峰值显存
labels = ["vLLM 0.80", "vLLM 0.35"]
memory_gib = [53052 / 1024, 23595 / 1024]

plt.figure(figsize=(6, 5))
bars = plt.bar(
    labels,
    memory_gib,
    color=["#4472C4", "#70AD47"],
)

for bar, value in zip(bars, memory_gib):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value:.2f} GiB",
        ha="center",
    )

plt.ylabel("Peak VRAM usage (GiB)")
plt.title("vLLM Peak VRAM before and after Optimization")
plt.ylim(0, 60)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "vllm_peak_memory_optimization.png",
    dpi=200,
)
plt.close()

print("图表生成完成：")
for path in sorted(FIGURES.glob("*.png")):
    print(path)
