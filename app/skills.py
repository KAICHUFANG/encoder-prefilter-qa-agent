"""技能层：每个意图对应一个可执行的函数（这里是桩），全程不调大模型。

这一层回答了「分流后到底干了什么」：
    不是把整句话丢给大模型，而是
        意图 -> 对应技能 -> 抽参数（时间/城市/歌名/目的地…）-> 执行 -> 回一句话

18 个场景里，16 个是「操作类」（设闹钟、查天气、放音乐、导航、点外卖、加日程…），
天然有 API；只有 qa（开放问答）和 general（闲聊）必须交给大模型。

这就是「减少大模型调用」的落点：
    16/18 场景 = 0 token、毫秒级、结果确定
    2/18 场景 = 才调大模型

注意：这些 handler 是**桩**（不真的设闹钟/查天气），目的是把链路的形状做出来、
并统计"有多少请求根本不需要模型"。真实系统里把 handler 换成真实 API 即可。
"""
from __future__ import annotations

import re
from typing import Any, Callable

# ---------------- 参数抽取（规则，够 demo 用）----------------

CN_NUM = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn2int(s: str) -> int | None:
    """把 '七'/'十二' 这类中文数字转成整数（只处理 0-59）。"""
    if s.isdigit():
        return int(s)
    if s in CN_NUM:
        return CN_NUM[s]
    if s.startswith("十"):
        return 10 + CN_NUM.get(s[1:], 0)
    if "十" in s:
        a, _, b = s.partition("十")
        return CN_NUM.get(a, 1) * 10 + (CN_NUM.get(b, 0) if b else 0)
    return None


TIME_RE = re.compile(r"(早上|上午|中午|下午|晚上|凌晨)?\s*(\d{1,2}|[零一两二三四五六七八九十]{1,3})\s*[点:：]\s*(\d{1,2}|[零一两二三四五六七八九十]{1,3})?\s*分?")
HOUR_RE = re.compile(r"(早上|上午|中午|下午|晚上|凌晨)?\s*(\d{1,2}|[零一两二三四五六七八九十]{1,3})\s*点")
CITY_WORDS = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安", "南京",
              "天津", "苏州", "长沙", "青岛", "厦门", "昆明", "大连", "合肥", "郑州", "济南"]


def extract_time(text: str) -> str | None:
    m = TIME_RE.search(text) or HOUR_RE.search(text)
    if not m:
        return None
    period = m.group(1) or ""
    hour = _cn2int(m.group(2))
    minute = 0
    if m.re is TIME_RE and m.lastindex and m.lastindex >= 3 and m.group(3):
        minute = _cn2int(m.group(3)) or 0
    if hour is None:
        return None
    if period in ("下午", "晚上") and hour < 12:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def extract_city(text: str) -> str | None:
    for c in CITY_WORDS:
        if c in text:
            return c
    return None


def extract_after(text: str, keys: list[str]) -> str | None:
    for k in keys:
        if k in text:
            tail = text.split(k, 1)[1].strip("，。！？ 的个首")
            if tail:
                return tail[:20]
    return None


# ---------------- 技能定义 ----------------
# needs_llm=False 的走技能，True 的交给大模型

SKILLS: dict[str, dict[str, Any]] = {
    "alarm": {"needs_llm": False, "slots": ["time"]},
    "weather": {"needs_llm": False, "slots": ["city", "time"]},
    "music": {"needs_llm": False, "slots": ["song"]},
    "play": {"needs_llm": False, "slots": ["song"]},
    "news": {"needs_llm": False, "slots": ["topic"]},
    "audio": {"needs_llm": False, "slots": ["topic"]},
    "transport": {"needs_llm": False, "slots": ["destination"]},
    "takeaway": {"needs_llm": False, "slots": ["dish"]},
    "calendar": {"needs_llm": False, "slots": ["time", "event"]},
    "email": {"needs_llm": False, "slots": ["to", "content"]},
    "social": {"needs_llm": False, "slots": ["to", "content"]},
    "lists": {"needs_llm": False, "slots": ["item"]},
    "iot": {"needs_llm": False, "slots": ["device"]},
    "cooking": {"needs_llm": False, "slots": ["dish"]},
    "datetime": {"needs_llm": False, "slots": ["time"]},
    "recommendation": {"needs_llm": False, "slots": ["topic"]},
    # 这两个必须交给大模型
    "qa": {"needs_llm": True, "slots": []},
    "general": {"needs_llm": True, "slots": []},
}

REPLIES: dict[str, Callable[[dict], str]] = {
    "alarm": lambda p: f"已设置 {p.get('time', '待定时间')} 的闹钟。",
    "weather": lambda p: f"{p.get('city', '本地')}今天{p.get('weather', '晴')}，气温 18–26℃。",
    "music": lambda p: f"正在为你播放{f'《{p['song']}》' if p.get('song') else '推荐歌单'}。",
    "play": lambda p: f"开始播放{f'《{p['song']}》' if p.get('song') else '媒体内容'}。",
    "news": lambda p: f"为你播报{f'关于{p['topic']}的' if p.get('topic') else '今日'}新闻要点。",
    "audio": lambda p: f"正在播放{f'关于{p['topic']}的' if p.get('topic') else ''}音频内容。",
    "transport": lambda p: f"已规划到{p.get('destination', '目的地')}的路线，预计 32 分钟。",
    "takeaway": lambda p: f"已为你下单{f'一份{p['dish']}' if p.get('dish') else '外卖'}，预计 30 分钟送达。",
    "calendar": lambda p: f"已创建日程：{p.get('time', '待定时间')} {p.get('event', '会议')}。",
    "email": lambda p: f"已起草邮件{f'发给{p['to']}' if p.get('to') else ''}，请确认后发送。",
    "social": lambda p: f"已准备消息{f'发给{p['to']}' if p.get('to') else ''}，请确认后发送。",
    "lists": lambda p: f"已把{f'「{p['item']}」' if p.get('item') else '这一项'}加入清单。",
    "iot": lambda p: f"已操作{p.get('device', '智能设备')}。",
    "cooking": lambda p: f"已开始{f'「{p['dish']}」' if p.get('dish') else '这个菜'}的烹饪流程，约 20 分钟。",
    "datetime": lambda p: f"现在是 {p.get('time') or '2026-11-11 17:40'}。",
    "recommendation": lambda p: f"为你推荐{f'与{p['topic']}相关的' if p.get('topic') else ''}几个选项。",
}


def fill_slots(intent: str, text: str) -> dict:
    """按技能的槽位需求，用规则抽参数。抽不到就留空（不编）。"""
    slots: dict[str, Any] = {}
    need = SKILLS.get(intent, {}).get("slots", [])
    if "time" in need:
        t = extract_time(text)
        if t:
            slots["time"] = t
    if "city" in need:
        c = extract_city(text)
        if c:
            slots["city"] = c
    if "song" in need:
        s = extract_after(text, ["播放", "放一首", "放首", "听", "来首", "来点"])
        if s:
            slots["song"] = s
    if "destination" in need:
        d = extract_after(text, ["去", "到", "导航到"])
        if d:
            slots["destination"] = d
    if "dish" in need:
        d = extract_after(text, ["吃", "点", "订", "叫"])
        if d:
            slots["dish"] = d
    if "event" in need:
        e = extract_after(text, ["安排", "记一下", "加个"])
        if e:
            slots["event"] = e
    return slots


def handle(intent: str, text: str) -> dict:
    """执行一个技能。不需要（或不该由技能处理）就返回 ok=False。"""
    skill = SKILLS.get(intent)
    if not skill or skill["needs_llm"]:
        return {"ok": False, "needs_llm": True,
                "reason": f"意图「{intent}」需要生成式回答，交给大模型"}

    params = fill_slots(intent, text)
    reply = REPLIES.get(intent, lambda p: f"已处理「{intent}」。")(params)
    missing = [s for s in skill["slots"] if s not in params]
    return {
        "ok": True, "needs_llm": False, "intent": intent,
        "params": params, "missing_slots": missing,
        "reply": reply, "source": "技能（本地，0 token）",
    }


def describe() -> dict:
    return {
        "skills": [{"intent": k, "needs_llm": v["needs_llm"], "slots": v["slots"]}
                   for k, v in SKILLS.items()],
        "skill_count": sum(1 for v in SKILLS.values() if not v["needs_llm"]),
        "llm_count": sum(1 for v in SKILLS.values() if v["needs_llm"]),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()
    for q in sys.argv[1:] or ["早上七点叫我", "北京今天天气怎么样", "播放周杰伦的歌",
                              "叫个外卖", "帮我加个日程", "为什么会下雨"]:
        cls = p.predict(q, top_k=1)
        r = handle(cls["intent"], q)
        print(f"\n『{q}』 -> 意图 {cls['intent']} ({cls['confidence']:.2f})")
        if r["ok"]:
            print(f"   技能回复：{r['reply']}")
            print(f"   参数={r['params']}  缺槽位={r['missing_slots']}  0 token")
        else:
            print(f"   {r['reason']}")
