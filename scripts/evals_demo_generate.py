import os
import sys
import json
import argparse
import random
from datetime import datetime

try:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass

DEFAULT_PATH = os.environ.get("EVALS_LOG_PATH", os.path.join("eval_logs", "phoenix_evals.jsonl"))


def parse_args():
    parser = argparse.ArgumentParser(description="生成演示 Phoenix Evals JSONL 日志")
    parser.add_argument("--path", default=DEFAULT_PATH, help="输出 JSONL 文件路径")
    parser.add_argument("--count", type=int, default=10, help="生成记录条数（默认10）")
    return parser.parse_args()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


HELPFUL_LABELS = ["helpful", "not_helpful"]
HALLU_LABELS = ["no_hallucination", "hallucination"]


def make_record(i: int):
    q = f"演示问题 #{i}: 设备运行状态与报警统计情况如何？"
    a = (
        "演示回答：系统处于正常运行状态，近24小时报警较少。"
        if random.random() < 0.7
        else "演示回答：需要进一步确认具体设备与时间段。"
    )
    refs = [
        "point_name:主风机1 | code:VFAN-001 | table_name:points",
        "point_name:主风机2 | code:VFAN-002 | table_name:points",
    ] if random.random() < 0.8 else []
    helpful = random.choice(HELPFUL_LABELS)
    hallu = random.choice(HALLU_LABELS if refs else ["no_hallucination"])  # 无引用时默认不判幻觉
    return {
        "input": q,
        "output": a,
        "reference": refs,
        "meta": {
            "mode": "demo",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "evals": {
            "ok": True,
            "provider": "demo",
            "model": "demo",
            "helpfulness": helpful,
            "hallucination": hallu,
            "explanations": {
                "helpfulness": "演示标签，仅用于出报表效果。",
                "hallucination": "演示标签，非真实评估。",
            },
        },
    }


def main():
    args = parse_args()
    ensure_dir(args.path)
    count = max(1, args.count)
    with open(args.path, "a", encoding="utf-8") as f:
        for i in range(count):
            rec = make_record(i + 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] 已生成 {count} 条演示评估记录 -> {args.path}")


if __name__ == "__main__":
    main()