"""导出待人工核对的样本（给"人来打标签"用）。

输出格式每行一条，包含判断所需的全部信息：
    编号 | 领域 | 问句 | 代码解析出的约束 | 命中数 | 首条答案摘要

为什么不用 checker 的自动标签：
    自动标签是"两套我写的规则互检"，会系统性高估代码能力
    （例如忽略"推荐菜"约束、把店名里的数字当金额）。
    训练门控模型需要的是"人看过答案后觉得合不合理"的标签。

用法：
    python data/dump_for_review.py --start 0 --count 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import executor  # noqa: E402

SHOW_FIELDS = ("评分", "人均消费", "门票", "游玩时间", "价格", "地址", "推荐菜", "营业时间")


def summarize_row(row: dict) -> str:
    parts = []
    for f in SHOW_FIELDS:
        v = str(row.get(f) or "").strip()
        if v:
            parts.append(f"{f}={v[:14]}")
    return " ".join(parts) or "（空）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=60)
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in config.TEST_JSONL.open(encoding="utf-8") if line.strip()]
    batch = rows[args.start: args.start + args.count]

    # 上下文：同一 dialog 内，记住最近一次提到的商户（用于"这家/它/周边"的指代）
    names = sorted({r["name"] for r in executor.load_rows() if r.get("name")},
                   key=len, reverse=True)
    last_by_dialog: dict[str, str] = {}
    for r in rows[: args.start]:
        hit = next((n for n in names if n in r["text"]), None)
        if hit:
            last_by_dialog[r["dialog_id"]] = hit

    for i, r in enumerate(batch, start=args.start):
        ctx = {"last_name": last_by_dialog[r["dialog_id"]]} if last_by_dialog.get(r["dialog_id"]) else None
        res = executor.execute(r["intent"], r["text"], top_k=1, context=ctx)
        hit = next((n for n in names if n in r["text"]), None)
        if hit:
            last_by_dialog[r["dialog_id"]] = hit
        elif res.get("rows"):
            last_by_dialog[r["dialog_id"]] = res["rows"][0]["name"]
        cons = res.get("constraints", [])
        cons_txt = "; ".join(
            f"{c['field']}{c['op']}{c.get('value', '')}"
            f"{('[' + str(round(c.get('lo', 0), 1)) + '-' + str(round(c.get('hi', 0), 1)) + ']') if c['op'] == 'range' else ''}"
            for c in cons) or "无"
        top = (res.get("rows") or [{}])[0]
        print(f"{i}|{r['intent']}|{r['text'][:52]}|{cons_txt}|"
              f"{res.get('matched')}|{summarize_row(top)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
