"""按「现在的代码能力」重新给门控打标签。

旧标签（2026-11）问的是"旧库 + 旧检索下代码能不能答"，当时只有 21.7% 能答。
现在换成官方库 + 周边关系表 + 约束过滤，代码能答 90%——旧标签已经过时。

新标签规则（确定性，可复现）：
    短句(<=12字)且无领域词          -> intent=0, gate=template
    执行器返回 ok                   -> intent=0, gate=code
    执行器返回不 ok                 -> intent=1, gate=llm

输出到 data/crosswoz_gate2/，不动原数据集（便于对比新旧）。
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

from app import executor  # noqa: E402

SRC = PROJ / "data" / "crosswoz_gate"
DST = PROJ / "data" / "crosswoz_gate2"
KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"

DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅", "小吃"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房费", "房价", "房间", "入住", "住下"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "长城", "乐园", "门票",
                         "票价", "逛")}
HOTEL_WORDS = ("会议室", "早餐", "24小时热水", "无烟房", "健身房", "游泳池", "温泉",
               "叫醒", "接机", "商务中心", "经济型", "舒适型", "高档型", "豪华型",
               "住宿", "住下", "入住")


def guess_domain(text: str) -> str:
    text = re.sub(r"周边(餐馆|景点|酒店|的餐馆|的景点|的酒店)", "", text)
    if any(w in text for w in HOTEL_WORDS):
        return "酒店"
    hits = []
    for dom, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text:
                hits.append((dom, text.rfind(w)))
    return max(hits, key=lambda x: x[1])[0] if hits else "餐馆"


def main() -> int:
    executor.KB_PATH = KB
    executor.load_rows.cache_clear()
    names = sorted({r["name"] for r in executor.load_rows() if r.get("name")}, key=len, reverse=True)

    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))

    def context_of(row_id: str) -> dict | None:
        m = re.search(r"(\d+)$", row_id)
        idx = int(m.group(1)) if m else -1
        if idx < 0 or idx >= len(all_items):
            return None
        dlg = all_items[idx]["dialog_id"]
        msgs = raw.get(str(dlg), {}).get("messages", [])
        text = all_items[idx]["text"]
        pos = next((i for i, x in enumerate(msgs)
                    if str(x.get("content", "")).strip() == text.strip()), None)
        if pos is None:
            return None
        for m2 in reversed(msgs[max(0, pos - 6):pos]):
            hit = next((n for n in names if n in str(m2.get("content", ""))), None)
            if hit:
                return {"last_name": hit}
        return None

    DST.mkdir(parents=True, exist_ok=True)
    total = Counter()
    for split in ("train", "dev", "test"):
        src = SRC / f"{split}.jsonl"
        if not src.exists():
            continue
        out = []
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["text"]
            if len(text) <= 12 and not any(w in text for w in DOMAIN_HINTS):
                label, gate = 0, "template"
            else:
                res = executor.execute(guess_domain(text), text, top_k=2,
                                       context=context_of(row["id"]))
                label, gate = (0, "code") if res.get("ok") else (1, "llm")
            out.append({"id": row["id"], "text": text, "intent": label,
                        "split": split, "gate": gate})
            total[f"{split}:{gate}"] += 1
        (DST / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        dist = Counter(r["gate"] for r in out)
        print(f"{split}: {dict(dist)}")

    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码/模板可推进 1=必须交大模型（按官方库+新检索重打）"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {DST}")
    for k in sorted(total):
        print(f"   {k}: {total[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
