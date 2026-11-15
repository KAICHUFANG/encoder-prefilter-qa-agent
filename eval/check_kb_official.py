"""用官方库核一遍：60 条测试问题的「必要信息」到底有没有。

按问题类型分两种检查，避免误判：
    ① 指代型（"这家/它/周边"或明确提到门店）：
       找到参照门店 -> 检查问题所问字段在该行是否存在
    ② 筛选型（"推荐一个评分 4 分以上的餐馆"）：
       不需要参照门店，直接跑执行器看能否筛出结果

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/check_kb_official.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import executor  # noqa: E402

KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"

# 问句里的字段词 -> 库里的字段名
ASK_FIELDS = {
    "评分": ("评分", "几分", "评分是"),
    "门票": ("门票", "票价"),
    "电话": ("电话", "联系方式"),
    "地址": ("地址", "在哪"),
    "营业时间": ("营业时间", "几点"),
    "类型": ("类型", "几星"),
    "推荐菜": ("推荐菜", "菜品", "有什么菜", "招牌"),
    "人均消费": ("人均", "人均消费"),
    "价格": ("价格", "房价", "多少钱一晚"),
    "游玩时间": ("玩多久", "游玩时间", "游玩时长"),
}
FIELD_ALIAS = {"类型": "酒店类型"}
RELATION_FIELD = {"景点": "周边景点", "餐馆": "周边餐馆", "酒店": "周边酒店"}
DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "门票", "长城")}
PRONOUNS = ("这家", "这个", "它", "该", "周边", "附近", "旁边", "周围")


def guess_domain(text: str) -> str:
    hits = []
    for dom, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text:
                hits.append((dom, text.rfind(w)))
    return max(hits, key=lambda x: x[1])[0] if hits else "餐馆"


def main() -> int:
    executor.KB_PATH = KB
    executor.load_rows.cache_clear()
    kb = json.loads(KB.read_text(encoding="utf-8"))["rows"]
    by_name = {r["name"]: r for r in kb}
    names = sorted(by_name, key=len, reverse=True)

    rows = [json.loads(line) for line in config.TEST_JSONL.open(encoding="utf-8") if line.strip()]
    test_all = [json.loads(line) for line in (PROJ / "data" / "crosswoz" / "test.jsonl").open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))

    def context_of(row_id: str) -> list[str]:
        m = re.search(r"(\d+)$", row_id)
        idx = int(m.group(1)) if m else -1
        if idx < 0 or idx >= len(test_all):
            return []
        dlg = test_all[idx]["dialog_id"]
        msgs = raw.get(str(dlg), {}).get("messages", [])
        text = test_all[idx]["text"]
        pos = next((i for i, x in enumerate(msgs) if str(x.get("content", "")).strip() == text.strip()), None)
        if pos is None:
            return []
        return [str(x.get("content", "")) for x in msgs[max(0, pos - 6):pos]]

    stat = Counter()
    field_miss = Counter()
    problems = []

    for r in rows:
        text = r["text"]
        ctx = context_of(r["id"])
        asked = [f for f, ws in ASK_FIELDS.items() if any(w in text for w in ws)]
        needs_relation = any(w in text for w in ("周边", "附近", "旁边", "周围"))
        is_referent_type = needs_relation or any(w in text for w in PRONOUNS)

        # 参照门店：先找问句里出现的全名，再回溯最近一条上下文
        referent = next((n for n in names if n in text), None)
        if not referent:
            for msg in reversed(ctx):
                referent = next((n for n in names if n in msg), None)
                if referent:
                    break

        if referent is None:
            # 筛选型：不需要参照门店，跑执行器
            res = executor.execute(guess_domain(text), text, top_k=2,
                                   context=None)
            if res.get("ok"):
                stat["筛选型-可答"] += 1
            else:
                stat["筛选型-无结果"] += 1
                problems.append((r["id"], text, "筛选型无结果", "、".join(asked)))
            continue

        stat["指代型"] += 1
        row = by_name[referent]
        missing = []
        for f in asked:
            if f == "周边":
                continue
            key = FIELD_ALIAS.get(f, f)
            if not str(row.get(key, "")).strip():
                missing.append(key)
        if needs_relation:
            rel = RELATION_FIELD.get(guess_domain(text))
            if not str(row.get(rel, "")).strip():
                missing.append(rel)
        if missing:
            stat["指代型-字段不全"] += 1
            for m in missing:
                field_miss[m] += 1
            problems.append((r["id"], text, f"{referent} 缺 {'、'.join(missing)}", "、".join(asked)))
        else:
            stat["指代型-信息齐全"] += 1

    n = len(rows)
    print(f"官方库 {len(kb)} 家商户 · 测试 {n} 条\n")
    total_ok = stat["筛选型-可答"] + stat["指代型-信息齐全"]
    print(f"  信息齐全（能答）   {total_ok:>3}  {total_ok/n:>6.1%}")
    print(f"    其中 筛选型可答   {stat['筛选型-可答']:>3}")
    print(f"    其中 指代型齐全   {stat['指代型-信息齐全']:>3}")
    print(f"  缺字段/无结果       {stat['指代型-字段不全'] + stat['筛选型-无结果']:>3}"
          f"  {(stat['指代型-字段不全'] + stat['筛选型-无结果'])/n:>6.1%}")

    if field_miss:
        print("\n仍缺的字段：")
        for f, c in field_miss.most_common():
            print(f"   {f:<14}{c}")
    print(f"\n仍有问题的 {len(problems)} 条（前 10）：")
    for pid, text, why, asked in problems[:10]:
        print(f"   {pid} 『{text[:28]}』 问[{asked or '未识别'}] {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
