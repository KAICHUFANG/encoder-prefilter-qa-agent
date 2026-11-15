"""门控评测：encoder 判断「该不该交给大模型」在难样本上准不准。

金标来自 eval/route_hard.jsonl（4 类），这里映射成二分类：
    code / ask      -> needs_llm = 0
    extract / other -> needs_llm = 1

为什么要单独评这份：
    训练集标签是"跑代码链路"生成的，和表面模式同源，模型能到 99%+，没有信息量。
    这 40 条手写样本专挑规则会失手的说法（一打年、四张、二叔），才看得出模型有没有学到东西。

用法：
    $env:INTENT_DATASET = "gate"
    python eval/run_gate_eval.py
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
TO_BINARY = {"code": 0, "ask": 0, "extract": 1, "other": 1}


def main() -> int:
    rows = [json.loads(line) for line in GOLD.open(encoding="utf-8-sig") if line.strip()]

    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    hits = tp = fp = fn = tn = 0
    errors = []
    print(f"{'句子':<30}{'金标':<8}{'模型':<8}")
    print("-" * 50)
    for r in rows:
        gold = TO_BINARY[r["route"]]
        pred = int(p.predict(r["text"], top_k=1)["intent"])
        ok = pred == gold
        hits += ok
        tp += (gold == 1 and pred == 1)
        fp += (gold == 0 and pred == 1)
        fn += (gold == 1 and pred == 0)
        tn += (gold == 0 and pred == 0)
        if not ok:
            errors.append((r["text"], r["route"], gold, pred))
        print(f"{r['text'][:28]:<30}{gold:<8}{pred:<8}{'' if ok else '  ✗'}")

    n = len(rows)
    print("-" * 50)
    print(f"二分类准确率：{hits}/{n} = {hits/n:.1%}")
    recall1 = tp / max(1, tp + fn)
    prec1 = tp / max(1, tp + fp)
    print(f"「该交大模型」这一类：召回 {recall1:.0%}  精确 {prec1:.0%}")
    print(f"  漏掉（该交没交）{fn} 条 —— 这些会答错")
    print(f"  多交（不必交却交）{fp} 条 —— 这些只是多花钱")

    # 对照：全走代码（不做判断）
    base = sum(1 for r in rows if TO_BINARY[r["route"]] == 0) / n
    print(f"\n对照：B0 全走代码（永不交模型）准确率 = {base:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
