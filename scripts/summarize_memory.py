import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

files = {
    "vLLM": PROJECT / "logs/vllm_dcu_monitor.log",
    "Transformers": (
        PROJECT / "logs/transformers_dcu_monitor.log"
    ),
}

pattern = re.compile(
    r"DCU\[0\].*vram Total Used Memory \(MiB\):\s*(\d+)"
)

lines = [
    "推理引擎显存监控汇总",
    "",
]

for engine, path in files.items():
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    values = [
        int(match.group(1))
        for match in pattern.finditer(text)
    ]

    if not values:
        raise RuntimeError(
            f"{engine}未解析到DCU 0显存数据：{path}"
        )

    minimum = min(values)
    maximum = max(values)

    lines.extend([
        f"[{engine}]",
        f"采样次数：{len(values)}",
        f"最低显存：{minimum} MiB",
        f"峰值显存：{maximum} MiB",
        f"波动范围：{maximum - minimum} MiB",
        "",
    ])

report = "\n".join(lines)

output = PROJECT / "results/memory_summary.txt"
output.write_text(report, encoding="utf-8")

print(report)
print("已保存：", output)
