"""路由判断评测：微调的 encoder 在「真难样本」上能不能判断代码是否可完成。

与 eval/run_route_eval.py（旧的分流评测）不同，这里评的是**新策略**：
    句子 -> code / ask / extract / other

为什么要有这份人造难样本：
    训练集的路由标签是"跑一遍代码链路"生成的，标签和表面模式高度相关，
    所以模型和 TF-IDF 都能拿 99.98%——那个数字没有信息量。
    这份 40 条手写样本专门挑"规则会失手"的说法（一打年、四张、二叔）。

用法：
    $env:INTENT_DATASET = "route"
    python eval/run_router_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

GOLD = HERE / "route_hard.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in GOLD.open(encoding="utf-8-sig") if line.strip()]

    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    hits = 0
    confusion: Counter = Counter()
    print(f"{'句子':<30}{'金标':<9}{'模型':<9}")
    print("-" * 52)
    for r in rows:
        pred = p.predict(r["text"], top_k=1)["intent"]
        hits += pred == r["route"]
        confusion[(r["route"], pred)] += 1
        flag = "" if pred == r["route"] else "  ✗"
        print(f"{r['text'][:28]:<30}{r['route']:<9}{pred:<9}{flag}")

    n = len(rows)
    print("-" * 52)
    print(f"模型路由准确率：{hits}/{n} = {hits/n:.1%}\n")
    print("各标签召回：")
    for label in ("code", "ask", "extract", "other"):
        total = sum(1 for r in rows if r["route"] == label)
        got = sum(1 for (g, pr), c in confusion.items() if g == label and pr == label for _ in range(c))
        if total:
            print(f"   {label:<9}{got}/{total} = {got/total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
