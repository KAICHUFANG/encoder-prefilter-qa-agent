"""槽位抽取：把「12 年」「四百多」「下周一」这类参数结构化出来。

为什么需要：
    只输出意图（年休假）工具没法算答案——
    「我在这行干了 12 年，一年能歇多少天」必须先拿到「工作年限=12」，
    才能查制度表（满 10 年不满 20 年 → 10 天）。这就是 joint NLU：intent + slots。

设计原则：
    1. 抽不到就不编（返回空，由上层追问或交模型）
    2. 每个槽位都要上下文约束，避免"两天后"被当成"天数"
    3. 中文数字、大写数字、口语缩写（一千二=1200）都要覆盖

支持槽位：
    工作年限 / 金额 / 天数 / 小时 / 人数 / 间数 / 时间 / 日期 / 城市 / 请假类型 / 亲属关系

用法：
    from app.slots import extract
    extract("我在这行干了 12 年，一年能歇多少天")   # {"工作年限": 12}
"""
from __future__ import annotations

import re

# ---------------- 数字解析 ----------------
CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "壹": 1, "两": 2, "二": 2, "贰": 2, "三": 3, "叁": 3,
            "四": 4, "肆": 4, "五": 5, "伍": 5, "六": 6, "陆": 6, "七": 7, "柒": 7,
            "八": 8, "捌": 8, "九": 9, "玖": 9}
CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000, "万": 10000,
           "亿": 100000000}

CN_CHARS = "".join(list(CN_DIGIT) + list(CN_UNIT))
NUM = rf"(\d+(?:\.\d+)?|[{CN_CHARS}]{{1,8}})"


def cn_to_arabic(s: str) -> float | None:
    """中文数字转阿拉伯数字。

    四百 -> 400      二十 -> 20       十五 -> 15
    一千二 -> 1200（口语省略末位单位）  四百五 -> 450
    """
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)

    total = section = num = 0
    last_unit = 0
    for ch in s:
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
            continue
        if ch in CN_UNIT:
            unit = CN_UNIT[ch]
            if num == 0:
                num = 1
            if unit >= 10000:                       # 万/亿：进位到大单位
                section = (section + num) * unit
                total += section
                section = 0
            else:
                section += num * unit
            num = 0
            last_unit = unit
            continue
        return None

    # 口语缩写：「一千二」= 1200、「四百五」= 450（末位数字省了单位）
    if num and last_unit >= 100:
        num = num * (last_unit // 10)
    return float(total + section + num) or None


def to_num(s: str) -> float | None:
    return cn_to_arabic(s) if s else None


# ---------------- 词表 ----------------
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安",
          "厦门", "青岛", "大连", "长沙", "苏州", "天津", "重庆", "郑州", "济南"]

LEAVE_TYPES = ["事假", "病假", "年假", "年休假", "婚假", "产假", "陪产假", "丧假",
               "调休", "工伤假", "探亲假"]

# 顺序重要：「配偶的祖辈」要排在「配偶」前面
KIN = [
    (r"(老婆|老公|爱人|配偶|妻子|丈夫|媳妇)的(奶奶|爷爷|外婆|外公)", "配偶祖辈"),
    (r"(爷爷|奶奶|外婆|外公|祖父|祖母|姥姥|姥爷)", "祖父母"),
    (r"(父亲|母亲|爸妈|父母|老爸|老妈|我爸|我妈|爸|妈)", "父母"),
    (r"(老婆|老公|爱人|配偶|妻子|丈夫|媳妇)", "配偶"),
    (r"(儿子|女儿|孩子|子女)", "子女"),
    (r"(哥哥|姐姐|弟弟|妹妹|兄弟|姐妹|我哥|我姐|我弟|我妹)", "兄弟姐妹"),
]

MONEY_HINTS = ("要", "花", "住", "贵", "便宜", "报", "付", "费用", "上限", "标准",
               "预算", "超过", "超标", "多", "元", "块", "发票", "金额", "人均", "封顶",
               "打车", "车费", "机票", "酒店")

# 数字后面跟这些量词，就不是金额（"一年""两间""三个人"）
NON_MONEY_UNITS = ("年", "天", "日", "个", "人", "间", "晚", "小时", "周", "月", "点",
                   "岁", "次", "分", "号", "度", "楼", "层", "台", "份")

DATE_ABS = {
    "大后天": "大后天", "今天": "今天", "今日": "今天", "明天": "明天", "明日": "明天",
    "后天": "后天", "昨天": "昨天", "昨日": "昨天", "前天": "前天",
    "上个月": "上个月", "上月": "上个月", "这个月": "这个月", "本月": "这个月",
    "下个月": "下个月", "下月": "下个月", "月底": "月底", "月初": "月初",
    "去年": "去年", "今年": "今年", "明年": "明年", "年底": "年底",
    "上周": "上周", "本周": "本周", "这周": "本周", "下周": "下周",
}

WEEKDAY = {"一": "周一", "二": "周二", "三": "周三", "四": "周四", "五": "周五",
           "六": "周六", "日": "周日", "天": "周日"}

WORK_HINTS = ("干", "工作", "工龄", "这行", "从业", "经验", "职场", "年资", "做了",
              "加起来", "总共", "一共", "累计")


# ---------------- 各槽位抽取 ----------------
def _ctx(text: str, start: int, end: int, pad: int = 8) -> str:
    return text[max(0, start - pad): end + pad]


def extract_work_years(text: str) -> float | None:
    """工作年限：必须有"干/工作/工龄/这行/从业"这类上下文；多值时取最大（累计工龄）。

    两个坑：
        「七年半」要算 7.5；「2015 年」是年份不是工龄（>60 直接丢弃）。
    """
    found: list[float] = []
    for m in re.finditer(rf"{NUM}\s*年半?", text):
        if any(h in _ctx(text, m.start(), m.end(), 6) for h in WORK_HINTS):
            v = to_num(m.group(1))
            if v is None or v > 60:               # 年份（2015）不是工龄
                continue
            if "年半" in m.group(0):
                v += 0.5
            found.append(v)
    return max(found) if found else None


def extract_money(text: str) -> float | None:
    """金额：优先「数字+元/块/万」，否则靠上下文词认「四百多」「超过 500」。"""
    m = re.search(rf"{NUM}\s*(?:元|块钱|块|万)", text)
    if m:
        v = to_num(m.group(1))
        if v is not None:
            return v * 10000 if "万" in m.group(0) else v

    # 阿拉伯数字 + 中文单位：「4 百」「3 千」
    m = re.search(r"(\d+(?:\.\d+)?)\s*([百千万])", text)
    if m:
        base = float(m.group(1)) * CN_UNIT[m.group(2)]
        return base

    for m in re.finditer(rf"{NUM}\s*(?:多|余|左右)?", text):
        raw = m.group(1)
        if not raw:
            continue
        # 后面跟量词的（"一年""两间""三个人"）不是金额
        if text[m.end(): m.end() + 1] in NON_MONEY_UNITS:
            continue
        window = _ctx(text, m.start(), m.end(), 6)
        if any(h in window for h in MONEY_HINTS):
            v = to_num(raw)
            # 没有"元/块"却小于 10 的，几乎都是被上下文词误伤的个位数（"下周一"的"一"）
            if v is not None and v >= 10:
                return v
    return None


def extract_days(text: str) -> float | None:
    """天数：排除「两天后」「三天前」这类日期表达；「半个月」算 15 天。"""
    if "半个月" in text:
        return 15.0
    for m in re.finditer(rf"{NUM}\s*(?:个)?\s*(?:天|日)", text):
        tail = text[m.end(): m.end() + 2]
        if tail.startswith(("后", "前", "之后", "以前")):
            continue
        v = to_num(m.group(1))
        if v is not None:
            return v
    return None


def extract_hours(text: str) -> float | None:
    m = re.search(rf"{NUM}\s*(?:个)?\s*(?:小时|钟头)", text)
    return to_num(m.group(1)) if m else None


def extract_people(text: str) -> float | None:
    m = re.search(rf"{NUM}\s*(?:个)?\s*人", text)
    if m:
        v = to_num(m.group(1))
        if v is not None and v <= 500:
            return v
    return None


def extract_rooms(text: str) -> float | None:
    m = re.search(rf"{NUM}\s*间", text)
    return to_num(m.group(1)) if m else None


def extract_time(text: str) -> str | None:
    """时间点：七点 / 八点半 / 下午 3 点 / 22:30。"""
    m = re.search(rf"(早上|上午|中午|下午|晚上|傍晚|凌晨)?\s*{NUM}\s*[点:：]\s*(半|{NUM})?\s*分?", text)
    if not m:
        return None
    period = m.group(1) or ""
    hour = to_num(m.group(2))
    if hour is None or hour >= 24:
        return None
    minute = 0.0
    tail = m.group(3)
    if tail == "半":
        minute = 30
    elif tail:
        minute = to_num(tail) or 0
    if period in ("下午", "晚上", "傍晚") and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    # 无时段词但有"加班/下班/夜里"上下文，且是 1-6 点以外的低调时，按 24 小时制补 12
    elif not period and hour <= 11 and any(h in _ctx(text, 0, len(text), 10)
                                           for h in ("加班", "下班", "夜里", "半夜")):
        hour += 12
    return f"{int(hour):02d}:{int(minute):02d}"


def extract_date(text: str) -> str | None:
    """日期：绝对表达优先（下周X / 上个月），其次相对表达（N 天后）。"""
    m = re.search(r"(上上|下下|上|下|这)?个?(?:周|星期|礼拜)([一二三四五六日天])", text)
    if m:
        day = WEEKDAY[m.group(2)]
        prefix = {"上上": "上上", "下下": "下下", "上": "上", "下": "下",
                  "这": "本"}.get(m.group(1) or "", "")
        return f"{prefix}{day}" if prefix else day
    for k in sorted(DATE_ABS, key=len, reverse=True):
        if k in text:
            return DATE_ABS[k]
    m = re.search(rf"{NUM}\s*(天|周|星期|礼拜|个月)\s*(后|以后|之后|前|以前|之前)", text)
    if m:
        v = to_num(m.group(1))
        if v is not None:
            unit = {"天": "天", "周": "周", "星期": "周", "礼拜": "周", "个月": "个月"}[m.group(2)]
            direction = "后" if m.group(3) in ("后", "以后", "之后") else "前"
            return f"{int(v)}{unit}{direction}"
    return None


def extract_leave_type(text: str) -> str | None:
    # 长词优先：「陪产假」不能先被「产假」截胡
    for t in sorted(LEAVE_TYPES, key=len, reverse=True):
        if t in text:
            return "年假" if t == "年休假" else t
    return None


def extract_kin(text: str) -> str | None:
    for pattern, label in KIN:
        if re.search(pattern, text):
            return label
    return None


def extract_city(text: str) -> str | None:
    for c in CITIES:
        if c in text:
            return c
    return None


def extract(text: str) -> dict:
    """抽全部槽位。抽不到就不写进返回（绝不编）。"""
    text = re.sub(r"\s+", "", text or "")
    out: dict = {}
    for key, value in (
        ("工作年限", extract_work_years(text)),
        ("金额", extract_money(text)),
        ("天数", extract_days(text)),
        ("小时", extract_hours(text)),
        ("人数", extract_people(text)),
        ("间数", extract_rooms(text)),
        ("时间", extract_time(text)),
        ("日期", extract_date(text)),
        ("城市", extract_city(text)),
        ("请假类型", extract_leave_type(text)),
        ("亲属关系", extract_kin(text)),
    ):
        if value is not None:
            out[key] = value
    return out


if __name__ == "__main__":
    import sys

    for q in sys.argv[1:] or [
        "我在这行干了 12 年，一年能歇多少天",
        "我工龄二十年了，年假多少天",
        "去上海住一晚要四百多，怎么算钱",
        "这次住宿一千二，超标了吧",
        "我爷爷过世了，能给几天",
        "晚上八点半下班",
        "两天后要去广州",
        "我和同事两个人一起住，能报两间吗",
    ]:
        print(f"『{q}』 -> {extract(q)}")
