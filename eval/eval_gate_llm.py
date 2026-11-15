"""在新标签（DeepSeek 依据问题+代码回答判定）上评估门控模型。

对比三样东西：
    ① 新训门控 models/crosswoz_gate_llm/intent-bert
    ② 旧门控   models/crosswoz_gate/intent-bert（用旧人工标签训的）
    ③ 我早先的人工标签（data/crosswoz/human_gate_merged.jsonl）与新标签的一致率

报告：准确率 / macro-F1 / 混淆矩阵 / 走代码比例 / 交模型召回率。

用法：
    $env:INTENT_DATASET = "crosswoz_gate_llm"
    python eval/eval_gate_llm.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

TEST = PROJ / "data" / "crosswoz_gate_llm" / "test.jsonl"
OLD_GATE = PROJ / "models" / "crosswoz_gate" / "intent-bert"
NEW_GATE = PROJ / "models" / "crosswoz_gate_llm" / "intent-bert"
HUMAN = PROJ / "data" / "crosswoz" / "human_gate_merged.jsonl"


def prf(conf: Counter) -> tuple[float, float, float]:
    tp = conf[(1, 1)]
    fp = conf[(0, 1)]
    fn = conf[(1, 0)]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(model_dir: Path, rows: list[dict], name: str) -> None:
    from app.model import IntentPredictor

    if not model_dir.exists():
        print(f"{name}: 模型不存在（{model_dir}）")
        return
    p = IntentPredictor(model_dir=model_dir)
    p.warmup()
    conf: Counter = Counter()
    hit = 0
    for r in rows:
        pred = int(p.predict(r["text"], top_k=1)["intent"])
        gold = int(r["intent"])
        conf[(gold, pred)] += 1
        hit += pred == gold
    n = len(rows)
    prec, rec, f1 = prf(conf)
    code_share = sum(v for (g, pr), v in conf.items() if pr == 0) / n
    gold_code = sum(1 for r in rows if int(r["intent"]) == 0) / n
    print(f"\n{name}")
    print(f"   准确率 {hit/n:.1%} | 交模型类 精确率 {prec:.2f} 召回率 {rec:.2f} F1 {f1:.2f}")
    print(f"   走代码 {code_share:.1%}（应然 {gold_code:.1%}）")
    print(f"   混淆 (gold,pred): {dict(sorted(conf.items()))}")


def main() -> int:
    rows = [json.loads(line) for line in TEST.open(encoding="utf-8") if line.strip()]
    print(f"测试集 {len(rows)} 条；标签分布 {dict(Counter(r['gate'] for r in rows))}")

    evaluate(NEW_GATE, rows, "① 新门控（本次重训）")
    evaluate(OLD_GATE, rows, "② 旧门控（旧人工标签训练）")

    if HUMAN.exists():
        human = {}
        for line in HUMAN.open(encoding="utf-8"):
            if not line.strip():
                continue
            h = json.loads(line)
            human[str(h.get("id"))] = h.get("gate")
        agree = total = 0
        for r in rows:
            g = human.get(str(r["id"]))
            if g is None:
                continue
            h_label = 0 if g in ("code", "template") else 1
            agree += h_label == int(r["intent"])
            total += 1
        if total:
            print(f"\n③ 我早先的人工标签 vs 新标签：一致 {agree}/{total} = {agree/total:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
