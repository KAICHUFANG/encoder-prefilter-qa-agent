"""把真实数据集的问题灌进代码，逐条检查结果是否正确，并记录标签。

流程（按用户定的方法）：
    1. 真实数据：data/crosswoz/test.jsonl（3525 条真人问句 + 1257 家商户）
    2. 完整代码：app/executor.py（抽约束 -> 查库 -> 渲染答案）
    3. 检查正确性：不是看答案"像不像"，而是机械校验三件事
           a) 覆盖：句子里提到的每类约束（价格/评分/时长/距离/关系），代码是否都解析了
           b) 满足：返回的每一行是否真的满足解析出的约束
           c) 有结果：库里有满足条件的行
    4. 记录标签：ok / parse_miss / empty / no_need

为什么要"覆盖"这一项：
    代码可以"忽略约束"然后返回一堆结果——看起来 ok，实际答错了。
    只有把"提到了但没解析"单独抓出来，才能发现这类静默错误。

用法：
    $env:INTENT_DATASET = "crosswoz"
    python eval/check_executor.py --limit 500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import executor  # noqa: E402

# 句子里的"约束信号"：出现这些词，说明用户表达了某类约束
MENTIONS = {
    "价格": re.compile(r"(人均|价格|元|块钱|预算|便宜|贵)"),
    "评分": re.compile(r"(评分|分数|好评|口碑)"),
    "时长": re.compile(r"(游玩时长|玩多久|小时|分钟)"),
    "距离": re.compile(r"(步行|公里|km|米)"),
    "关系": re.compile(r"(周边|附近|旁边|周围)"),
    "门票": re.compile(r"(门票|票价)"),
}

# 每种约束信号 -> 代码里应该出现哪个字段
EXPECTED_FIELD = {
    "价格": {"人均消费", "价格", "门票"},
    "评分": {"评分"},
    "时长": {"游玩时间"},
    "距离": {"距离"},
    "关系": {"name", "地址"},
    "门票": {"门票"},
}


def check_one(domain: str, text: str) -> dict:
    """跑一遍代码并检查结果，返回标签与原因。"""
    result = executor.execute(domain, text, top_k=3)
    fields = {c["field"] for c in result.get("constraints", [])}
    mentioned = [m for m, pat in MENTIONS.items() if pat.search(text)]

    # a) 覆盖检查：提到了但代码没解析 -> 静默漏约束
    missing = []
    for m in mentioned:
        if m == "关系":
            if not ({"name", "地址"} & fields):
                missing.append(m)
        elif m in ("价格", "门票"):
            # 价格和门票语义重叠，任一命中就算覆盖
            if not (fields & EXPECTED_FIELD["价格"]):
                missing.append(m)
        else:
            if not (fields & EXPECTED_FIELD.get(m, set())):
                missing.append(m)

    # b) 满足检查：返回行是否满足解析出的约束
    rows = result.get("rows") or []
    violated = sum(1 for r in rows if not executor._match(r, result.get("constraints", [])))

    # c) 有结果
    matched = result.get("matched") or 0

    if not mentioned:
        label = "no_need"          # 没有可校验的约束（闲聊/泛指）
    elif missing:
        label = "parse_miss"       # 提到了约束但代码没解析
    elif matched == 0:
        label = "empty"            # 解析了但库里没有
    elif violated:
        label = "bad_rows"         # 返回了不满足约束的行
    else:
        label = "ok"

    return {"text": text, "domain": domain, "label": label,
            "mentioned": mentioned, "parsed_fields": sorted(fields),
            "missing": missing, "matched": matched,
            "answer": (result.get("answer") or "")[:120],
            "reason": result.get("reason")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="data/crosswoz/labeled.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in config.TEST_JSONL.open(encoding="utf-8") if line.strip()]
    rows = rows[: args.limit]

    out_rows = []
    stat = Counter()
    examples = defaultdict(list)
    for r in rows:
        item = check_one(r["intent"], r["text"])
        item["gold_intent"] = r["intent"]
        out_rows.append(item)
        stat[item["label"]] += 1
        if len(examples[item["label"]]) < 3:
            examples[item["label"]].append(item)

    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in out_rows:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n = len(out_rows)
    print(f"灌入 {n} 条真实问句 -> {out}\n")
    print(f"{'标签':<12}{'条数':>7}{'占比':>8}  含义")
    meaning = {"ok": "约束全解析且答案满足", "parse_miss": "漏解析约束（静默答错）",
               "empty": "解析了但库里没有", "bad_rows": "返回了不满足约束的行",
               "no_need": "语句里没有可校验约束"}
    for label in ("ok", "parse_miss", "empty", "bad_rows", "no_need"):
        if stat[label]:
            print(f"{label:<12}{stat[label]:>7}{stat[label]/n:>8.1%}  {meaning[label]}")

    print("\n各类样例：")
    for label in ("parse_miss", "empty", "bad_rows"):
        for ex in examples[label]:
            print(f"   [{label}] {ex['text'][:34]}")
            print(f"        提到={ex['mentioned']} 解析={ex['parsed_fields']} 缺={ex['missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
