"""把大模型的回答质量并入门控指标，算真实的端到端答对率。

之前"总体答对 96.7%"是假设（默认交出去的请求模型都能答对）。
现在有真数据了：
    eval/llm_answers.jsonl  47 条模型回答
    eval/llm_grades.jsonl   人工判分（correct / partial / wrong / refuse）

端到端口径：
    代码侧：gate 测试集里，门控判 code 的样本 × 人工标签是否 code
    模型侧：llm_grades 里判 correct / partial 的比例
    总体   ：(代码侧答对 + 模型侧答对) / 全部样本

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/run_pipeline_metrics.py
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


def main() -> int:
    answers = [json.loads(l) for l in (HERE / "llm_answers.jsonl").open(encoding="utf-8") if l.strip()]
    grades = {g["i"]: g for g in (json.loads(l) for l in (HERE / "llm_grades.jsonl").open(encoding="utf-8") if l.strip())}
    gcount = Counter(g["label"] for g in grades.values())
    n_llm = len(answers)
    print(f"大模型侧：{n_llm} 条")
    for k in ("correct", "partial", "wrong", "refuse"):
        print(f"   {k:<9}{gcount[k]:>3}  {gcount[k]/n_llm:>6.1%}")
    useful = gcount["correct"] + gcount["partial"]
    print(f"   有用（correct+partial）= {useful/n_llm:.1%}；完全正确 = {gcount['correct']/n_llm:.1%}")

    # 代码侧：门控测试集
    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()
    test_rows = [json.loads(l) for l in config.TEST_JSONL.open(encoding="utf-8") if l.strip()]
    n_total = len(test_rows)
    code_routed = code_correct = 0
    for r in test_rows:
        out = p.predict(r["text"], top_k=1)
        if int(out["intent"]) == 0:
            code_routed += 1
            if r.get("gate") in ("code", "template"):
                code_correct += 1
    print(f"\n代码侧：{n_total} 条中门控放行 {code_routed} 条，其中答对 {code_correct} 条"
          f"（代码侧正确率 {code_correct/max(1,code_routed):.1%}）")

    # 端到端
    full = code_correct + gcount["correct"]
    partial = code_correct + useful
    print(f"\n端到端（{n_total} 条）：")
    print(f"   完全正确  {full}/{n_total} = {full/n_total:.1%}")
    print(f"   有用      {partial}/{n_total} = {partial/n_total:.1%}")
    print(f"\n对照（都不调大模型）：")
    base = sum(1 for r in test_rows if r.get("gate") in ("code", "template"))
    print(f"   全交代码  {base}/{n_total} = {base/n_total:.1%}（这就是代码上限）")
    print(f"\n结论：真调模型后，总体完全正确率 {full/n_total:.1%}，"
          f"比“假设模型全对”的 96.7% 低得多。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
