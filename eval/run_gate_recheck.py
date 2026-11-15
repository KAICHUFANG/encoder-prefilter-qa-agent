"""在新能力下重测门控（encoder-only）：它还能不能判断"该不该交给代码"。

背景：
    门控是用 400 条人工标签训的，而当时"代码能干什么"只包含旧库 + 旧检索
    （只能处理约 20% 的请求）。现在换成官方库、加了周边关系表和约束过滤，
    代码能力大幅提升——门控的输入没变，但"正确答案"变了。

做法：
    1. 用**现在的**代码链路重算每条问题"能不能答"（执行器有结果 = 能答；
       无结果 = 需要模型；短句无领域 = 模板）
    2. 拿这个新标签检验现有门控模型（crosswoz_gate）
    3. 报告覆盖率、代码侧正确率、以及"完美路由"的上限

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/run_gate_recheck.py
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

DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅", "小吃"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房费", "房价", "房间", "入住", "住下"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "长城", "乐园", "门票",
                         "票价", "逛")}


def guess_domain(text: str) -> str:
    text = re.sub(r"周边(餐馆|景点|酒店|的餐馆|的景点|的酒店)", "", text)
    if any(w in text for w in ("会议室", "早餐", "24小时热水", "无烟房", "健身房", "游泳池",
                               "温泉", "叫醒", "接机", "商务中心", "经济型", "舒适型",
                               "高档型", "豪华型", "住宿", "住下", "入住")):
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
    kb = executor.load_rows()
    names = sorted({r["name"] for r in kb if r.get("name")}, key=len, reverse=True)

    rows = [json.loads(line) for line in config.TEST_JSONL.open(encoding="utf-8") if line.strip()]
    test_all = [json.loads(line) for line in (PROJ / "data" / "crosswoz" / "test.jsonl").open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))

    def context_of(row_id: str) -> dict | None:
        m = re.search(r"(\d+)$", row_id)
        idx = int(m.group(1)) if m else -1
        if idx < 0 or idx >= len(test_all):
            return None
        dlg = test_all[idx]["dialog_id"]
        msgs = raw.get(str(dlg), {}).get("messages", [])
        text = test_all[idx]["text"]
        pos = next((i for i, x in enumerate(msgs) if str(x.get("content", "")).strip() == text.strip()), None)
        if pos is None:
            return None
        for m2 in reversed(msgs[max(0, pos - 6):pos]):
            hit = next((n for n in names if n in str(m2.get("content", ""))), None)
            if hit:
                return {"last_name": hit}
        return None

    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    stat = Counter()
    detail = []
    for r in rows:
        text, ctx = r["text"], context_of(r["id"])
        # 新标签：代码能不能答
        if len(text) <= 12 and not any(w in text for w in DOMAIN_HINTS):
            gold = "template"
        else:
            res = executor.execute(guess_domain(text), text, top_k=2, context=ctx)
            gold = "code" if res.get("ok") else "llm"

        pred_idx = int(p.predict(text, top_k=1)["intent"])
        pred = "code" if pred_idx == 0 else "llm"
        ok = pred == gold
        stat[f"{gold}->{pred}"] += 1
        stat["total"] += 1
        if ok:
            stat["hit"] += 1
        detail.append((r["id"], text, gold, pred, ok))

    n = stat["total"]
    code_n = sum(v for k, v in stat.items() if k.startswith("code->"))
    llm_n = sum(v for k, v in stat.items() if k.startswith("llm->"))
    tpl_n = sum(v for k, v in stat.items() if k.startswith("template->"))
    print(f"测试集 {n} 条（官方库 + 新检索下的真实能力）：")
    print(f"   代码能答 {code_n} 条（{code_n/n:.1%}）")
    print(f"   必须交模型 {llm_n} 条（{llm_n/n:.1%}）")
    print(f"   模板 {tpl_n} 条（{tpl_n/n:.1%}）")
    print(f"\n门控模型（旧标签训练）与新标签的一致率：{stat['hit']}/{n} = {stat['hit']/n:.1%}")
    print("\n混淆（新标签 -> 门控判定）：")
    for k in sorted(stat):
        if "->" in k:
            print(f"   {k:<20}{stat[k]}")

    wrong = [d for d in detail if not d[4]]
    print(f"\n门控判错的 {len(wrong)} 条（前 8）：")
    for rid, text, gold, pred, _ in wrong[:8]:
        print(f"   {rid} 『{text[:30]}』 实际={gold} 门控={pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
