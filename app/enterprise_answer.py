"""把「意图 + 槽位」变成确定性答案（不调模型）。

示例：
    intent=年休假, slots={工作年限: 12}
    -> 查《员工手册》第13条：满 10 年不满 20 年 = 10 天
    -> 答案：「累计工作满 10 年不满 20 年，年休假 10 天（《员工手册》第13条）」

为什么这一层必须存在：
    只给意图标签，工具没法算；只给槽位也不算。两者拼起来才能查表/计算/渲染。
    这一层全是纯代码：0 token、可复现、可测试。
"""
from __future__ import annotations

import re
from pathlib import Path

# 制度语料（复用制度问答项目的 58 条）
CORPUS_DIRS = [
    Path(r"C:\Users\xiong\Desktop\制度问答Agent\corpus"),
]

# 年休假档位（来自《员工手册》第13条）
ANNUAL_LEAVE_TIERS = [(20, 15), (10, 10), (1, 5)]   # (满N年, 天数)，从高到低匹配


def load_clauses() -> list[dict]:
    for d in CORPUS_DIRS:
        if not d.exists():
            continue
        clauses = []
        for md in sorted(d.glob("*.md")):
            text = md.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"^###\s*(第(\d+)条)\s*(.*)$", text, re.M):
                start = m.end()
                nxt = text.find("\n### ", start)
                body = text[start: nxt if nxt > 0 else len(text)].strip()
                clauses.append({"doc": md.stem, "number": f"第{m.group(2)}条",
                                "heading": m.group(3).strip(), "text": body})
        if clauses:
            return clauses
    return []


def annual_leave_days(years: float) -> int | None:
    for threshold, days in ANNUAL_LEAVE_TIERS:
        if years >= threshold:
            return days
    return None


PRESCIENT_SKIP = ("目的", "总则", "适用范围")


def find_clause(intent: str, clauses: list[dict]) -> dict | None:
    """按意图找最相关条款：标题命中优先，其次正文命中，跳过"目的/总则"这种前言。"""
    best, best_score = None, 0
    for c in clauses:
        if any(k in c["heading"] for k in PRESCIENT_SKIP):
            continue
        score = 0
        if intent in c["heading"]:
            score += 10
        if intent in c["text"]:
            score += 3
        if c["number"] == "第1条" and intent not in c["heading"]:
            score -= 5
        if score > best_score:
            best, best_score = c, score
    return best


# 住宿标准（《差旅与费用报销制度》第5条，一类/二类/三类城市）
CITY_CLASS = {"北京": "一类", "上海": "一类", "广州": "一类", "深圳": "一类",
              "杭州": "二类", "南京": "二类", "成都": "二类", "武汉": "二类",
              "西安": "二类", "厦门": "二类", "青岛": "二类", "大连": "二类"}
HOTEL_LIMIT = {"一类": 500, "二类": 400, "三类": 300}


def answer(intent: str, slots: dict, question: str = "") -> dict:
    """给定意图 + 槽位，返回结构化答案。查不到就 ok=False（交给模型）。"""
    # 1) 年休假：用工作年限查档位表（能真正算出数字）
    if intent == "年休假" and "工作年限" in slots:
        years = slots["工作年限"]
        days = annual_leave_days(float(years))
        if days is not None:
            return {
                "ok": True, "intent": intent, "slots": slots,
                "computed": {"年休假天数": days},
                "answer": (f"累计工作满 {int(years)} 年，年休假 {days} 天"
                           f"（满 10 年不满 20 年为 10 天，满 20 年为 15 天）。"
                           f"\n依据：《员工手册》第13条"),
                "source": "《员工手册》第13条",
                "mode": "computed",
            }

    # 2) 住宿：用城市类别 + 金额判断是否超标（也是纯计算）
    if intent == "住宿" and "金额" in slots:
        city = slots.get("城市")
        cls = CITY_CLASS.get(str(city), "三类")
        limit = HOTEL_LIMIT[cls]
        amount = float(slots["金额"])
        verdict = "未超标" if amount <= limit else f"超出 {amount - limit:.0f} 元"
        return {
            "ok": True, "intent": intent, "slots": slots,
            "computed": {"城市类别": cls, "上限": limit, "结论": verdict},
            "answer": (f"{city or '目的地'}属{cls}城市，住宿上限 {limit} 元/晚；"
                       f"你的 {amount:.0f} 元{verdict}。\n依据：《差旅与费用报销制度》第5条"),
            "source": "《差旅与费用报销制度》第5条",
            "mode": "computed",
        }

    # 2) 其他主题：用意图关键词找回对应条款，连同槽位一起返回
    clauses = load_clauses()
    hit = find_clause(intent, clauses)
    if hit:
        slot_text = "、".join(f"{k}={v}" for k, v in slots.items()) or "无"
        return {
            "ok": True, "intent": intent, "slots": slots,
            "answer": (f"《{hit['doc']}》{hit['number']} {hit['heading']}：\n"
                       f"{hit['text'][:220]}\n（识别到的参数：{slot_text}）"),
            "source": f"《{hit['doc']}》{hit['number']}",
            "mode": "clause",
        }

    return {"ok": False, "intent": intent, "slots": slots,
            "reason": f"制度里找不到「{intent}」对应条款，建议交给模型或人工"}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.slots import extract

    for intent, q in [("年休假", "我在这行干了 12 年，一年能歇多少天"),
                      ("年休假", "我干了 5 年，能休几天"),
                      ("住宿", "去上海住一晚要四百多，怎么算钱"),
                      ("合同", "有个合同要盖章，要怎么办")]:
        s = extract(q)
        r = answer(intent, s, q)
        print(f"\n『{q}』")
        print(f"   intent={intent}  slots={s}")
        print(f"   {r.get('answer') or r.get('reason')}")
