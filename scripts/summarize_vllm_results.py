import csv
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT / "results/raw/vllm"
OUTPUT = PROJECT / "results/vllm_run_summary.csv"

pattern = re.compile(
    r"(?P<category>\w+)_in(?P<input_len>\d+)"
    r"_out(?P<output_len>\d+)_c(?P<concurrency>\d+)"
    r"_r(?P<repeat>\d+)\.json"
)

fields = [
    "file",
    "category",
    "input_len",
    "output_len",
    "concurrency",
    "repeat",
    "num_prompts",
    "completed",
    "duration_s",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput_rps",
    "output_throughput_tps",
    "total_token_throughput_tps",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "nonempty_errors",
]

rows = []

for path in sorted(INPUT_DIR.glob("*.json")):
    match = pattern.fullmatch(path.name)

    if not match:
        raise ValueError(f"无法解析文件名：{path.name}")

    data = json.loads(path.read_text(encoding="utf-8"))

    errors = [
        error for error in data.get("errors", [])
        if error not in (None, "", False)
    ]

    row = {
        "file": path.name,
        **match.groupdict(),
        "num_prompts": data.get("num_prompts"),
        "completed": data.get("completed"),
        "duration_s": data.get("duration"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "request_throughput_rps": data.get("request_throughput"),
        "output_throughput_tps": data.get("output_throughput"),
        "total_token_throughput_tps": data.get(
            "total_token_throughput"
        ),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "p50_ttft_ms": data.get("p50_ttft_ms"),
        "p95_ttft_ms": data.get("p95_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "p50_tpot_ms": data.get("p50_tpot_ms"),
        "p95_tpot_ms": data.get("p95_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "mean_itl_ms": data.get("mean_itl_ms"),
        "p95_itl_ms": data.get("p95_itl_ms"),
        "p99_itl_ms": data.get("p99_itl_ms"),
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "p50_e2el_ms": data.get("p50_e2el_ms"),
        "p95_e2el_ms": data.get("p95_e2el_ms"),
        "p99_e2el_ms": data.get("p99_e2el_ms"),
        "nonempty_errors": len(errors),
    }

    rows.append(row)

assert len(rows) == 23, f"预期23组，实际{len(rows)}组"
assert all(row["completed"] == 50 for row in rows)
assert all(row["nonempty_errors"] == 0 for row in rows)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"已生成：{OUTPUT}")
print(f"实验组数：{len(rows)}")
print(f"成功请求：{sum(row['completed'] for row in rows)}")
