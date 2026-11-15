"""统计大模型侧准确率：区分「库里本来有答案」和「库里根本没有」。

用户要求：库里没有的就忽略，不统计。

做法：
    eval/llm_scope.jsonl 里逐条标了 kb_has（问题所需字段在库中是否存在）。
    只统计 kb_has=true 的样本；kb_has=false 的单独列出来，不计入准确率。

用法：
    python eval/run_llm_scope_metrics.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    grades = {g["i"]: g for g in load_jsonl(HERE / "llm_grades.jsonl")}
    scope = {s["i"]: s for s in load_jsonl(HERE / "llm_scope.jsonl")}

    inside = [i for i in grades if scope.get(i, {}).get("kb_has")]
    outside = [i for i in grades if not scope.get(i, {}).get("kb_has")]

    def report(name: str, ids: list[int]) -> Counter:
        c = Counter(grades[i]["label"] for i in ids)
        n = len(ids) or 1
        print(f"\n{name}：{len(ids)} 条")
        for k in ("correct", "partial", "wrong", "refuse"):
            print(f"   {k:<8}{c[k]:>3}  {c[k]/n:>6.1%}")
        print(f"   严格准确率 = {c['correct']/n:.1%}   有用率 = {(c['correct']+c['partial'])/n:.1%}")
        return c

    print("=" * 56)
    print("按「库里有没有答案」拆分（47 条门控判给大模型的样本）")
    print("=" * 56)
    report("① 库里本来有答案（计入统计）", inside)
    report("② 库里没有（按要求忽略）", outside)

    # 拒答原因归类
    why = Counter(scope[i]["why"] for i in outside)
    print("\n被忽略的样本原因：")
    for w, c in why.most_common():
        print(f"   {c:>3} 条  {w}")

    c_in = Counter(grades[i]["label"] for i in inside)
    n_in = len(inside) or 1
    print(f"\n{'='*56}")
    print(f"结论：可统计的 {len(inside)} 条里，大模型完全答对 {c_in['correct']} 条 = "
          f"{c_in['correct']/n_in:.1%}，含部分有用 {(c_in['correct']+c_in['partial'])/n_in:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
