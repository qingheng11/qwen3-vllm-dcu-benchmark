import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

BASELINE = PROJECT / "results/vllm_run_summary.csv"
OPTIMIZED = PROJECT / "results/vllm_mem035_run_summary.csv"

OUTPUT = (
    PROJECT
    / "results/vllm_memory_optimization_comparison.csv"
)
REPORT = (
    PROJECT
    / "docs/vllm_memory_optimization_findings.md"
)

METRICS = [
    "output_throughput_tps",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "p95_e2el_ms",
]


def load(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


baseline = load(BASELINE)
optimized = load(OPTIMIZED)

assert len(baseline) == 23
assert len(optimized) == 23

base_by_file = {
    row["file"]: row
    for row in baseline
}
opt_by_file = {
    row["file"]: row
    for row in optimized
}

assert set(base_by_file) == set(opt_by_file)

groups = defaultdict(list)

for filename in sorted(base_by_file):
    base = base_by_file[filename]
    opt = opt_by_file[filename]

    assert int(base["completed"]) == 50
    assert int(opt["completed"]) == 50
    assert int(base["nonempty_errors"]) == 0
    assert int(opt["nonempty_errors"]) == 0
    assert (
        int(base["total_input_tokens"])
        == int(opt["total_input_tokens"])
    )
    assert (
        int(base["total_output_tokens"])
        == int(opt["total_output_tokens"])
    )

    key = (
        base["category"],
        int(base["input_len"]),
        int(base["output_len"]),
        int(base["concurrency"]),
    )

    groups[key].append((base, opt))


rows = []

for key, pairs in sorted(groups.items()):
    category, input_len, output_len, concurrency = key

    row = {
        "category": category,
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "runs": len(pairs),
    }

    for metric in METRICS:
        base_values = [
            float(base[metric])
            for base, _ in pairs
        ]
        opt_values = [
            float(opt[metric])
            for _, opt in pairs
        ]

        base_mean = statistics.mean(base_values)
        opt_mean = statistics.mean(opt_values)

        row[f"baseline_{metric}"] = base_mean
        row[f"optimized_{metric}"] = opt_mean
        row[f"{metric}_change_pct"] = (
            (opt_mean - base_mean)
            / base_mean
            * 100
        )

    rows.append(row)


with OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)


concurrency_rows = sorted(
    [
        row for row in rows
        if row["category"] == "concurrency"
    ],
    key=lambda row: row["concurrency"],
)

memory_base = 53052
memory_opt = 23595
memory_reduction = (
    (memory_base - memory_opt)
    / memory_base
    * 100
)

lines = [
    "# vLLM 显存优化对比",
    "",
    "## 配置",
    "",
    "- 基线：gpu_memory_utilization=0.80",
    "- 优化：gpu_memory_utilization=0.35",
    "- 其他模型、硬件和生成参数保持不变",
    "",
    "## 显存",
    "",
    f"- 基线峰值：{memory_base} MiB",
    f"- 优化峰值：{memory_opt} MiB",
    f"- 峰值下降：{memory_reduction:.2f}%",
    "",
    "## 并发性能",
    "",
    "|并发|基线tok/s|优化tok/s|吞吐变化|"
    "基线P95 TTFT(ms)|优化P95 TTFT(ms)|TTFT变化|",
    "|---:|---:|---:|---:|---:|---:|---:|",
]

for row in concurrency_rows:
    lines.append(
        f"|{row['concurrency']}|"
        f"{row['baseline_output_throughput_tps']:.2f}|"
        f"{row['optimized_output_throughput_tps']:.2f}|"
        f"{row['output_throughput_tps_change_pct']:+.2f}%|"
        f"{row['baseline_p95_ttft_ms']:.2f}|"
        f"{row['optimized_p95_ttft_ms']:.2f}|"
        f"{row['p95_ttft_ms_change_pct']:+.2f}%|"
    )

REPORT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("优化前后数据一致性：PASS")
print("对比CSV：", OUTPUT)
print("报告：", REPORT)
print()
print(
    "并发\t基线tok/s\t优化tok/s\t吞吐变化\t"
    "基线TTFT\t优化TTFT\tTTFT变化"
)

for row in concurrency_rows:
    print(
        row["concurrency"],
        f"{row['baseline_output_throughput_tps']:.2f}",
        f"{row['optimized_output_throughput_tps']:.2f}",
        f"{row['output_throughput_tps_change_pct']:+.2f}%",
        f"{row['baseline_p95_ttft_ms']:.2f}",
        f"{row['optimized_p95_ttft_ms']:.2f}",
        f"{row['p95_ttft_ms_change_pct']:+.2f}%",
        sep="\t",
    )

print()
print(f"峰值显存下降：{memory_reduction:.2f}%")
