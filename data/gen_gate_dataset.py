"""生成「二分类门控数据集」：这句话要不要交给大模型？

按用户定义修正：
    encoder-only 只输出一个判断 —— 该不该丢给大模型（needs_llm: 0/1）
    其余决策全部由代码完成：
        参数齐了      -> 代码执行（code）
        参数没给      -> 代码追问（ask）   ← 追问也是代码的事，0 token，不该叫大模型
        参数给了但抽不出 -> 这才是真的需要大模型（extract）
        制度外/求解释  -> 大模型（other）

标签映射（从 4 类并成 2 类）：
    code / ask    -> needs_llm = 0   （代码能推进：要么答，要么问）
    extract / other -> needs_llm = 1   （代码推不动）

为什么 ask 归到"不需要模型"：
    用户没给参数时，正确动作是**追问**，不是让大模型猜。
    追问是模板 + 缺失槽位，0 token；交给模型既浪费也可能编。

用法：
    python data/gen_gate_dataset.py
产出：
    data/raw/gate/questions.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

SRC = PROJ / "data" / "raw" / "route" / "questions.jsonl"
OUT_DIR = PROJ / "data" / "raw" / "gate"

KEEP = {"code": 0, "ask": 0, "extract": 1, "other": 1}


def main() -> int:
    if not SRC.exists():
        print(f"[失败] 缺 {SRC}，先跑 data/gen_route_dataset.py")
        return 1

    rows = []
    for line in SRC.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        route = obj.get("route")
        if route not in KEEP:
            continue
        rows.append({
            "question": obj["question"],
            "needs_llm": KEEP[route],
            "route": route,                 # 保留原始 4 类，方便错误分析
            "topic": obj.get("topic"),
            "slots": obj.get("slots", {}),
            "why": obj.get("why"),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "questions.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            r["id"] = f"gate-{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cnt = Counter(r["needs_llm"] for r in rows)
    total = len(rows)
    print(f"生成 {total} 条 -> {out}")
    print(f"   needs_llm = 0（代码可推进）: {cnt[0]:>6}  {cnt[0]/total:>6.1%}")
    print(f"   needs_llm = 1（需要大模型）: {cnt[1]:>6}  {cnt[1]/total:>6.1%}")
    print("\n来源分布：")
    for route, c in Counter(r["route"] for r in rows).most_common():
        print(f"   {route:<8}{c:>6}  -> needs_llm={KEEP[route]}")
    print("\n样例：")
    for r in rows[:3] + rows[-3:]:
        print(f"   [needs_llm={r['needs_llm']}] {r['question'][:36]:<38} ({r['why']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
