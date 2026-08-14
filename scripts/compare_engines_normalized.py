import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

VLLM_CSV = PROJECT / "results/vllm_run_summary.csv"
TF_CSV = PROJECT / "results/transformers_run_summary.csv"

PAIR_CSV = (
    PROJECT / "results/engine_pair_comparison_normalized.csv"
)
AGG_CSV = (
    PROJECT / "results/engine_comparison_normalized.csv"
)
REPORT = (
    PROJECT / "docs/engine_comparison_initial.md"
)


def load(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return {row["file"]: row for row in rows}


vllm = load(VLLM_CSV)
transformers = load(TF_CSV)

assert len(vllm) == 23
assert len(transformers) == 23
assert set(vllm) == set(transformers)

pair_rows = []

for filename in sorted(vllm):
    v = vllm[filename]
    t = transformers[filename]

    category = v["category"]
    input_len = int(v["input_len"])
    output_len = int(v["output_len"])
    concurrency = int(v["concurrency"])
    repeat = int(v["repeat"])

    v_completed = int(v["completed"])
    t_completed = int(t["completed"])

    assert v_completed == 50
    assert t_completed == 50
    assert int(v["nonempty_errors"]) == 0
    assert int(t["nonempty_errors"]) == 0

    assert int(v["total_input_tokens"]) == int(
        t["total_input_tokens"]
    ), f"{filename}输入Token数不一致"

    expected_v_tokens = v_completed * output_len
    expected_t_tokens = t_completed * output_len

    v_duration = float(v["duration_s"])
    t_duration = float(t["duration_s"])

    v_ttft = float(v["mean_ttft_ms"])
    t_ttft = float(t["mean_ttft_ms"])

    v_e2e = float(v["mean_e2el_ms"])
    t_e2e = float(t["mean_e2el_ms"])

    # 使用配置要求的固定生成长度标准化吞吐。
    v_normalized_output_tps = (
        expected_v_tokens / v_duration
    )
    t_normalized_output_tps = (
        expected_t_tokens / t_duration
    )

    # 使用平均E2E减去平均TTFT，估算固定生成长度下
    # 每个后续Token的平均生成时间。
    v_normalized_tpot = (
        (v_e2e - v_ttft) / max(output_len - 1, 1)
    )
    t_normalized_tpot = (
        (t_e2e - t_ttft) / max(output_len - 1, 1)
    )

    row = {
        "file": filename,
        "category": category,
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "repeat": repeat,
        "vllm_completed": v_completed,
        "transformers_completed": t_completed,
        "expected_output_tokens": expected_v_tokens,
        "vllm_reported_output_tokens": int(
            v["total_output_tokens"]
        ),
        "transformers_reported_output_tokens": int(
            t["total_output_tokens"]
        ),
        "vllm_duration_s": v_duration,
        "transformers_duration_s": t_duration,
        "vllm_normalized_output_tps": (
            v_normalized_output_tps
        ),
        "transformers_normalized_output_tps": (
            t_normalized_output_tps
        ),
        "output_throughput_speedup_x": (
            v_normalized_output_tps
            / t_normalized_output_tps
        ),
        "vllm_request_throughput_rps": float(
            v["request_throughput_rps"]
        ),
        "transformers_request_throughput_rps": float(
            t["request_throughput_rps"]
        ),
        "vllm_p95_ttft_ms": float(v["p95_ttft_ms"]),
        "transformers_p95_ttft_ms": float(
            t["p95_ttft_ms"]
        ),
        "p95_ttft_advantage_x": (
            float(t["p95_ttft_ms"])
            / float(v["p95_ttft_ms"])
        ),
        "vllm_normalized_mean_tpot_ms": (
            v_normalized_tpot
        ),
        "transformers_normalized_mean_tpot_ms": (
            t_normalized_tpot
        ),
        "normalized_tpot_advantage_x": (
            t_normalized_tpot / v_normalized_tpot
        ),
        "vllm_p95_e2el_ms": float(v["p95_e2el_ms"]),
        "transformers_p95_e2el_ms": float(
            t["p95_e2el_ms"]
        ),
        "p95_e2el_advantage_x": (
            float(t["p95_e2el_ms"])
            / float(v["p95_e2el_ms"])
        ),
    }

    pair_rows.append(row)


with PAIR_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(pair_rows[0]),
    )
    writer.writeheader()
    writer.writerows(pair_rows)


groups = defaultdict(list)

for row in pair_rows:
    key = (
        row["category"],
        row["input_len"],
        row["output_len"],
        row["concurrency"],
    )
    groups[key].append(row)


def mean(rows, field):
    return statistics.mean(
        float(row[field]) for row in rows
    )


aggregate_rows = []

for key, rows in sorted(groups.items()):
    category, input_len, output_len, concurrency = key

    v_tps = mean(rows, "vllm_normalized_output_tps")
    t_tps = mean(
        rows,
        "transformers_normalized_output_tps",
    )

    v_ttft = mean(rows, "vllm_p95_ttft_ms")
    t_ttft = mean(
        rows,
        "transformers_p95_ttft_ms",
    )

    v_tpot = mean(
        rows,
        "vllm_normalized_mean_tpot_ms",
    )
    t_tpot = mean(
        rows,
        "transformers_normalized_mean_tpot_ms",
    )

    v_e2e = mean(rows, "vllm_p95_e2el_ms")
    t_e2e = mean(
        rows,
        "transformers_p95_e2el_ms",
    )

    aggregate_rows.append({
        "category": category,
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "runs": len(rows),
        "vllm_normalized_output_tps_mean": v_tps,
        "transformers_normalized_output_tps_mean": t_tps,
        "output_throughput_speedup_x": v_tps / t_tps,
        "vllm_p95_ttft_ms_mean": v_ttft,
        "transformers_p95_ttft_ms_mean": t_ttft,
        "p95_ttft_advantage_x": t_ttft / v_ttft,
        "vllm_normalized_mean_tpot_ms": v_tpot,
        "transformers_normalized_mean_tpot_ms": t_tpot,
        "normalized_tpot_advantage_x": t_tpot / v_tpot,
        "vllm_p95_e2el_ms_mean": v_e2e,
        "transformers_p95_e2el_ms_mean": t_e2e,
        "p95_e2el_advantage_x": t_e2e / v_e2e,
    })


with AGG_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(aggregate_rows[0]),
    )
    writer.writeheader()
    writer.writerows(aggregate_rows)


concurrency_rows = sorted(
    [
        row for row in aggregate_rows
        if row["category"] == "concurrency"
    ],
    key=lambda row: row["concurrency"],
)

lines = [
    "# Transformers 与 vLLM 初步性能对比",
    "",
    "## 测试口径",
    "",
    "- 模型：Qwen3-1.7B",
    "- 设备：单张 K500SM_AI",
    "- 精度：BF16",
    "- 随机种子：42",
    "- 每组请求数：50",
    "- 并发实验重复次数：3",
    "",
    "Transformers流式接口返回文本重新Tokenize后存在"
    "0～5.33%的计数偏差，因此吞吐采用实验配置中的固定"
    "目标输出长度进行标准化；原始统计数据仍完整保留。",
    "",
    "## 并发测试",
    "",
    "|并发|Transformers tok/s|vLLM tok/s|吞吐加速比|"
    "Transformers P95 TTFT(ms)|vLLM P95 TTFT(ms)|"
    "TTFT优势倍数|",
    "|---:|---:|---:|---:|---:|---:|---:|",
]

for row in concurrency_rows:
    lines.append(
        f"|{row['concurrency']}|"
        f"{row['transformers_normalized_output_tps_mean']:.2f}|"
        f"{row['vllm_normalized_output_tps_mean']:.2f}|"
        f"{row['output_throughput_speedup_x']:.2f}×|"
        f"{row['transformers_p95_ttft_ms_mean']:.2f}|"
        f"{row['vllm_p95_ttft_ms_mean']:.2f}|"
        f"{row['p95_ttft_advantage_x']:.2f}×|"
    )

REPORT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("标准化一致性检查：PASS")
print(f"逐轮对比：{PAIR_CSV}")
print(f"聚合对比：{AGG_CSV}")
print(f"报告：{REPORT}")
print()
print(
    "并发\tTF tok/s\tvLLM tok/s\t吞吐加速\t"
    "TF P95 TTFT\tvLLM P95 TTFT"
)

for row in concurrency_rows:
    print(
        row["concurrency"],
        f"{row['transformers_normalized_output_tps_mean']:.2f}",
        f"{row['vllm_normalized_output_tps_mean']:.2f}",
        f"{row['output_throughput_speedup_x']:.2f}x",
        f"{row['transformers_p95_ttft_ms_mean']:.2f}",
        f"{row['vllm_p95_ttft_ms_mean']:.2f}",
        sep="\t",
    )
