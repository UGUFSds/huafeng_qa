import os
import sys
import json
import argparse
from collections import Counter

# 保证从 scripts/ 运行时可访问项目根目录（但本脚本不依赖 app 包）
try:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Phoenix Evals 日志汇总")
    parser.add_argument(
        "--path",
        default=os.environ.get("EVALS_LOG_PATH", os.path.join("eval_logs", "phoenix_evals.jsonl")),
        help="评估日志 JSONL 文件路径（默认读取 EVALS_LOG_PATH 或 eval_logs/phoenix_evals.jsonl）",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多读取的记录条数，0 表示全部")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    limit = args.limit if args.limit and args.limit > 0 else None

    if not os.path.exists(path):
        print(f"[warn] 日志文件不存在：{path}")
        print("[hint] 若需启用评估，请设置 EVALS_ENABLED=1 并配置提供方 API Key。")
        return

    helpful = Counter()
    hallu = Counter()
    provider = Counter()
    total = 0
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            evals = rec.get("evals") or {}
            helpful_label = evals.get("helpfulness")
            hallu_label = evals.get("hallucination")
            prov = evals.get("provider")
            if helpful_label:
                helpful[helpful_label] += 1
            if hallu_label:
                hallu[hallu_label] += 1
            if prov:
                provider[prov] += 1
            if len(examples) < 3:
                examples.append({
                    "input": rec.get("input"),
                    "output": rec.get("output"),
                    "helpfulness": helpful_label,
                    "hallucination": hallu_label,
                })
            if limit and total >= limit:
                break

    print(f"[report] 文件: {path}")
    print(f"[report] 统计记录数: {total}")
    if provider:
        print("[report] 提供方使用分布:")
        for k, v in provider.items():
            print(f"  - {k}: {v}")
    if helpful:
        print("[report] Helpfulness 标签分布:")
        for k, v in helpful.items():
            print(f"  - {k}: {v}")
    else:
        print("[report] Helpfulness 暂无记录或评估未启用。")
    if hallu:
        print("[report] Hallucination 标签分布:")
        for k, v in hallu.items():
            print(f"  - {k}: {v}")
    else:
        print("[report] Hallucination 暂无记录或无参考数据。")
    if examples:
        print("[examples] 示例（最多3条）：")
        for i, ex in enumerate(examples, 1):
            print(f"  - #{i} helpfulness={ex['helpfulness']} hallucination={ex['hallucination']}")
            print(f"    Q: {ex['input']}")
            txt = ex["output"] or ""
            if len(txt) > 200:
                txt = txt[:200] + "…"
            print(f"    A: {txt}")


if __name__ == "__main__":
    main()