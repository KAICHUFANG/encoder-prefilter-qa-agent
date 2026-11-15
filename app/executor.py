"""查商户库并渲染答案：全部本地、0 token。

数据源：data/crosswoz/kb_official.json（CrossWOZ 数据集自带的 database/ 转换而来）
    2549 家商户：餐馆 951 / 酒店 1133 / 景点 465
    字段：电话/地址 100%，评分 82%，地铁最近站 82%，周边景点 56%

支持的约束：
    数值：人均消费/价格/门票/评分/游玩时间（区间、以上以下、约）
    文本：名称、推荐菜、行政区、酒店设施、周边关系

用法：
    from app.executor import execute
    execute("餐馆", "评分 4.5 以上的餐馆")
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# ---------------- 字段与词表 ----------------
FIELD_WORDS = {
    "人均消费": ["人均", "人均消费", "消费"],
    "价格": ["价格", "房价", "多少钱一晚", "住一晚"],
    "门票": ["门票", "票价"],
    "评分": ["评分", "分数", "打分", "好评", "星级"],
    "游玩时间": ["游玩时长", "游玩时间", "玩多久", "玩多长时间", "逛多久", "游玩"],
    "酒店类型": ["酒店类型", "四星", "五星", "三星", "经济型", "快捷", "舒适型", "高档型", "豪华型"],
}

DOMAIN_DEFAULT_FIELD = {"餐馆": "人均消费", "酒店": "价格", "景点": "门票"}

DOMAIN_WORDS = {
    "餐馆": ["餐馆", "餐厅", "饭店", "吃饭", "好吃的", "餐饮"],
    "酒店": ["酒店", "宾馆", "住宿", "住一晚", "旅馆"],
    "景点": ["景点", "公园", "景区", "玩的地方", "游玩"],
}

FACILITY_WORDS = ["无烟房", "健身房", "游泳池", "温泉", "棋牌室", "会议室", "早餐",
                  "停车场", "接机", "桑拿", "酒吧", "洗衣", "wifi", "宽带", "商务中心",
                  "24小时热水", "行李寄存", "叫醒服务", "接站"]

RELATION_WORDS = ("周边", "附近", "旁边", "周围")
RELATION_FIELD = {"景点": "周边景点", "餐馆": "周边餐馆", "酒店": "周边酒店"}

RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|分钟|分|元|块钱|块)?\s*"
                      r"(?:-|~|～|到|至)\s*(\d+(?:\.\d+)?)")
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
UPPER_WORDS = ("以内", "以下", "一下", "不超过", "低于", "不到", "≤")
LOWER_WORDS = ("以上", "至少", "不低于", "高于", "超过", "≥")
AROUND_WORDS = ("左右", "上下")

UNIT_FIELD = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*分(?!钟)"), "评分"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|钟头)"), "游玩时间"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块钱|块)"), "__money__"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*星"), "酒店类型"),
]

ASK_FIELD_WORDS = {
    "电话": ["电话", "联系方式", "号码"],
    "地址": ["地址", "在哪", "位置", "怎么走"],
    "营业时间": ["营业时间", "几点开", "开门", "关门"],
    "评分": ["评分", "几分"],
    "人均消费": ["人均", "消费"],
    "门票": ["门票", "票价"],
    "游玩时间": ["游玩时间", "玩多久", "游玩时长"],
    "酒店类型": ["类型", "几星"],
}

DISTRICT_RE = re.compile(r"([\u4e00-\u9fa5]{2,4}(?:区|县))")
STRONG_DISH_RE = re.compile(
    r"(?:吃到|想吃|这道菜|这个菜|能做)([\u4e00-\u9fa5A-Za-z0-9]{2,10}?)(?:这道菜|这道|的|吗|呢|，|。|？|\?|$)")
NON_DISH = {"酒店", "宾馆", "餐馆", "餐厅", "景点", "房间", "服务", "早餐", "会议室",
            "停车场", "无烟房", "健身房", "住宿", "地方", "问题", "电话", "地址",
            "时间", "评分", "门票", "什么", "没有", "优惠", "活动"}
DISH_REJECT = ("哪些", "什么", "哪", "几", "多少", "吗", "呢", "啊", "么", "没有", "推荐")

# 允许调用方指定商户库路径（面板/评测时目录可能不同）
KB_PATH = None


@lru_cache(maxsize=1)
def load_rows() -> list[dict]:
    path = Path(KB_PATH) if KB_PATH else (config.DATA_DIR / "kb_official.json")
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def parse_number(text: str) -> float | None:
    m = NUM_RE.search(str(text))
    return float(m.group(1)) if m else None


def _district_of(row: dict) -> str | None:
    m = DISTRICT_RE.search(str(row.get("地址", "")))
    return m.group(1) if m else None


def detect_target_domain(query: str, default: str) -> str:
    """问句里提到别的领域词，就以那个领域为目标（跨域问句）。"""
    query = re.sub(r"周边(餐馆|景点|酒店)", "", query)   # 关系目标不算主领域
    hits = []
    for dom, words in DOMAIN_WORDS.items():
        for w in words:
            if w in query:
                hits.append((dom, w))
    if not hits:
        return default
    return max(hits, key=lambda x: query.rfind(x[1]))[0]


@lru_cache(maxsize=1)
def dish_vocab() -> set[str]:
    """从商户库的推荐菜里收集真实菜名，避免把"合适""啥好吃"当成菜。"""
    vocab: set[str] = set()
    try:
        for r in load_rows():
            raw = str(r.get("推荐菜") or "")
            for token in re.split(r"[、，,／/\s]+", raw):
                token = token.strip()
                if 2 <= len(token) <= 10:
                    vocab.add(token)
    except FileNotFoundError:
        pass
    return vocab


def extract_dish(query: str) -> str | None:
    """菜品约束：① 库内真实菜名（最长匹配）② 强菜名标记后的片段。"""
    for dish in sorted(dish_vocab(), key=len, reverse=True):
        if dish not in query:
            continue
        if any(w in dish or dish in w for w in FACILITY_WORDS):
            continue
        return dish
    for m in STRONG_DISH_RE.finditer(query):
        cand = m.group(1).strip()
        if not cand or cand in NON_DISH:
            continue
        if any(f in cand for f in FACILITY_WORDS):
            continue
        if any(w in cand for w in NON_DISH) or any(w in cand for w in DISH_REJECT):
            continue
        return cand
    return None


def _nearest_field(query: str, pos: int, pad: int = 16) -> str | None:
    lo, hi = max(0, pos - pad), min(len(query), pos + pad)
    window = query[lo:hi]
    best, best_dist = None, 10 ** 6
    for name, words in FIELD_WORDS.items():
        for w in words:
            start = window.find(w)
            while start >= 0:
                dist = abs((lo + start) - pos)
                if dist < best_dist:
                    best, best_dist = name, dist
                start = window.find(w, start + 1)
    return best


def _op_of(query: str, num_end: int) -> dict:
    tail = query[num_end: num_end + 8]
    if any(w in tail for w in UPPER_WORDS):
        return {"op": "<="}
    if any(w in tail for w in LOWER_WORDS):
        return {"op": ">="}
    return {"op": "around"}


def extract_constraints(domain: str, query: str, context: dict | None = None) -> list[dict]:
    """抽出全部约束：名称/菜品/设施/关系/数值。"""
    constraints: list[dict] = []
    seen: set[tuple] = set()
    default_field = DOMAIN_DEFAULT_FIELD.get(domain)

    def add(c: dict) -> None:
        v = c.get("value")
        if isinstance(v, list):
            v = tuple(v)          # 设施是 OR 约束，value 为字段列表
        key = (c["field"], c["op"], v, c.get("lo"), c.get("hi"))
        if key not in seen:
            seen.add(key)
            constraints.append(c)

    # ① 名称
    names = sorted({r["name"] for r in load_rows() if r.get("name")}, key=len, reverse=True)
    ref_name = next((n for n in names if n in query), None)
    is_relation = any(w in query for w in RELATION_WORDS)
    if ref_name:
        add({"field": "name", "op": "refer" if is_relation else "eq", "value": ref_name})
    elif context and context.get("last_name") and any(
            w in query for w in RELATION_WORDS + ("这家", "这个", "它", "该")):
        ref_name = context["last_name"]
        add({"field": "name", "op": "refer", "value": ref_name})

    # ② 菜品
    dish = extract_dish(query)
    if dish:
        add({"field": "推荐菜", "op": "contains", "value": dish})

    # ③ 酒店设施与类型
    #    库里字段名比口语更细（"早餐服务" vs "早餐"、"部分房间提供wifi" vs "wifi"），
    #    所以按子串把相关设施字段收集成一个 OR 约束。
    kb_facilities = sorted({k for r in load_rows() for k in r if k.startswith("酒店设施-")})
    for w in FACILITY_WORDS:
        if w not in query:
            continue
        # 完整字段名出现在问句里（"部分房间提供wifi"）就精确匹配，不要放宽成 OR
        exact = [k for k in kb_facilities if w in k and k.replace("酒店设施-", "") in query]
        fields = exact or [k for k in kb_facilities if w in k]
        if fields:
            add({"field": "__any__", "op": "any_nonempty", "value": fields})
        else:
            add({"field": f"酒店设施-{w}", "op": "nonempty"})
    for t in ("经济型", "舒适型", "高档型", "豪华型"):
        if t in query:
            add({"field": "酒店类型", "op": "contains", "value": t})

    # ④ 关系：参照对象行政区（弱信号）+ 排除参照对象本身
    if is_relation and ref_name:
        ref = next((r for r in load_rows() if r.get("name") == ref_name), None)
        if ref:
            district = _district_of(ref)
            if district:
                add({"field": "地址", "op": "contains", "value": district})
        add({"field": "name", "op": "ne", "value": ref_name})

    # 「1、2个小时」这种顿号写法统一成「1-2小时」，避免被当成两个单值
    query = re.sub(r"(\d+)\s*[、,，]\s*(\d+)", r"\1-\2", query)

    # ⑤ 数值：区间优先，其次按单位归属
    #    先屏蔽设施词里的数字（"24小时热水"不能被当成游玩时长 24 小时）
    facility_spans: list[tuple[int, int]] = []
    for w in FACILITY_WORDS:
        start = query.find(w)
        while start >= 0:
            facility_spans.append((start, start + len(w)))
            start = query.find(w, start + 1)

    def in_facility(pos: int) -> bool:
        return any(s <= pos < e for s, e in facility_spans)

    covered: list[tuple[int, int]] = []
    for m in RANGE_RE.finditer(query):
        if in_facility(m.start()):
            continue
        field = _nearest_field(query, m.start()) or default_field
        tail = query[m.end(): m.end() + 3]
        # 区间后面紧跟单位时，按单位定字段（"200-300元" -> 价格，不是"舒适型"）
        tail = tail[1:] if tail.startswith("个") else tail
        if tail.startswith(("元", "块", "万")):
            field = default_field
        elif tail.startswith("分") and not tail.startswith("分钟"):
            field = "评分"
        elif tail.startswith(("小时", "钟头")):
            field = "游玩时间"
        if field:
            add({"field": field, "op": "range",
                 "lo": float(m.group(1)), "hi": float(m.group(2))})
            covered.append((m.start(), m.end()))

    for pattern, field in UNIT_FIELD:
        for m in pattern.finditer(query):
            if in_facility(m.start()):
                continue
            if any(s <= m.start() < e for s, e in covered):
                continue
            value = float(m.group(1))
            target = default_field if field == "__money__" else field
            if not target:
                continue
            direction = _op_of(query, m.end())
            if direction["op"] == "around":
                add({"field": target, "op": "range",
                     "lo": round(value * 0.8, 1), "hi": round(value * 1.2, 1)})
            else:
                add({"field": target, "op": direction["op"], "value": value})
            covered.append((m.start(), m.end()))

    for m in NUM_RE.finditer(query):
        if in_facility(m.start()):
            continue
        if any(s <= m.start() < e for s, e in covered):
            continue
        field = _nearest_field(query, m.start()) or default_field
        if not field:
            continue
        value = float(m.group(1))
        direction = _op_of(query, m.end())
        if direction["op"] == "around":
            add({"field": field, "op": "range",
                 "lo": round(value * 0.8, 1), "hi": round(value * 1.2, 1)})
        else:
            add({"field": field, "op": direction["op"], "value": value})
    return constraints


def _match(row: dict, constraints: list[dict]) -> bool:
    for c in constraints:
        op = c["op"]
        if op == "eq":
            if c["value"] not in str(row.get(c["field"], "")):
                return False
            continue
        if op == "refer":
            continue
        if op == "ne":
            if str(row.get(c["field"], "")) == c["value"]:
                return False
            continue
        if op == "nonempty":
            if not str(row.get(c["field"], "")).strip():
                return False
            continue
        if op == "any_nonempty":
            if not any(str(row.get(f, "")).strip() for f in c["value"]):
                return False
            continue
        if op == "contains":
            if c["value"] not in str(row.get(c["field"], "")):
                return False
            continue
        value = parse_number(row.get(c["field"], ""))
        if value is None:
            return False
        if op == "<=" and value > c["value"]:
            return False
        if op == ">=" and value < c["value"]:
            return False
        if op == "range" and not (c["lo"] <= value <= c["hi"]):
            return False
    return True


def relation_answer(ref_name: str, target_domain: str, top_k: int = 5) -> dict | None:
    """"X 附近有什么 Y" —— 直接用数据集自带的「周边Y」关系表回答。"""
    rows = load_rows()
    ref = next((r for r in rows if r.get("name") == ref_name), None)
    if not ref:
        return None
    field = RELATION_FIELD.get(target_domain)
    if not field:
        return None
    names = [n.strip() for n in str(ref.get(field, "")).split("、") if n.strip()]
    if not names:
        return None
    by_name = {r["name"]: r for r in rows if r["domain"] == target_domain}
    picked = [by_name[n] for n in names if n in by_name][:top_k]
    if not picked:
        return None
    lines = [f"{ref_name} 的{field}（共 {len(names)} 个，取前 {len(picked)} 个）："]
    for i, r in enumerate(picked, 1):
        extra = "｜".join(f"{k} {r[k]}" for k in ("评分", "人均消费", "门票", "价格", "地址", "电话")
                          if r.get(k))
        lines.append(f"{i}. {r['name']}" + (f"｜{extra}" if extra else ""))
    return {"ok": True, "domain": target_domain,
            "constraints": [{"field": field, "op": "relation", "value": ref_name}],
            "matched": len(names), "rows": picked,
            "answer": "\n".join(lines), "source": "数据集周边关系表（0 token）"}


def asked_fields(query: str) -> list[str]:
    return [f for f, words in ASK_FIELD_WORDS.items() if any(w in query for w in words)]


def _fmt(row: dict, fields: list[str]) -> str:
    parts = [row["name"]]
    for f in fields:
        if row.get(f):
            parts.append(f"{f} {row[f]}")
    return "｜".join(parts)


PRONOUNS = ("这家", "这个", "该店", "该酒店", "它", "那家", "此店")


def _single_venue(row: dict, query: str) -> dict | None:
    """单店问答：问的是"这家/它"的某个属性时，直接答这一家，不要退化成全库筛选。"""
    wanted = asked_fields(query)
    if not wanted and any(w in query for w in FACILITY_WORDS):
        fac = [k.replace("酒店设施-", "") for k in row
               if k.startswith("酒店设施-") and str(row[k]).strip()]
        hit = [w for w in FACILITY_WORDS if w in query]
        ok = [w for w in hit if any(w in f for f in fac)]
        miss = [w for w in hit if w not in ok]
        parts = []
        if ok:
            parts.append(f"有{'、'.join(ok)}")
        if miss:
            parts.append(f"未标注{'、'.join(miss)}")
        return {"ok": True, "domain": row["domain"], "constraints": [],
                "matched": 1, "rows": [row], "answer": f"{row['name']}：" + "，".join(parts),
                "source": "商户库（本地，0 token）"}
    if not wanted:
        return None
    vals = [(f, row.get(f)) for f in wanted if row.get(f)]
    if not vals:
        return None
    return {"ok": True, "domain": row["domain"], "constraints": [], "matched": 1,
            "rows": [row],
            "answer": f"{row['name']}：" + "｜".join(f"{f} {v}" for f, v in vals),
            "source": "商户库（本地，0 token）"}


def execute(domain: str, query: str, top_k: int = 3,
            context: dict | None = None) -> dict:
    """查库并渲染答案。查不到返回 ok=False，调用方可以升级给大模型。"""
    target_domain = detect_target_domain(query, domain)
    explicit = next((r["name"] for r in load_rows()
                     if r.get("name") and r["name"] in query), None)
    has_pronoun = any(w in query for w in PRONOUNS)

    # 问的是"这家/它"，但解析不出指代对象时不能瞎答全库
    if has_pronoun and not explicit and not (context and context.get("last_name")):
        return {"ok": False, "domain": target_domain, "constraints": [], "matched": 0,
                "reason": "问句用“这家/它”指代，但无法确定指代的门店"}

    ref_row = None
    if explicit:
        ref_row = next((r for r in load_rows() if r.get("name") == explicit), None)
    elif has_pronoun and context:
        ref_row = next((r for r in load_rows() if r.get("name") == context.get("last_name")), None)

    # 单店属性问答：直接答这一家
    if ref_row is not None and has_pronoun:
        single = _single_venue(ref_row, query)
        if single and not any(w in query for w in RELATION_WORDS):
            return single

    # 关系问句优先走「周边X」关系表
    if any(w in query for w in RELATION_WORDS):
        ref = explicit or (context or {}).get("last_name")
        if ref:
            rel = relation_answer(ref, target_domain, top_k=max(top_k, 3))
            if rel:
                return rel
            return {"ok": False, "domain": target_domain, "constraints": [], "matched": 0,
                    "reason": f"库里没有 {ref} 的{RELATION_FIELD.get(target_domain, '周边')}数据"}

    constraints = extract_constraints(target_domain, query, context)
    # 约束里出现酒店设施/类型字段时，主领域一定是酒店（"去完景点，找个有早餐的"仍是找酒店）
    def _is_hotel_constraint(c: dict) -> bool:
        if c["field"] == "酒店类型":
            return True
        if c["field"].startswith("酒店设施-"):
            return True
        if c["field"] == "__any__":
            return any(str(f).startswith("酒店设施-") for f in c.get("value") or [])
        return False

    if any(_is_hotel_constraint(c) for c in constraints):
        target_domain = "酒店"
    rows = [r for r in load_rows() if r["domain"] == target_domain]
    matched = [r for r in rows if _match(r, constraints)]
    loosened = False

    if not matched and any(c["field"] == "地址" for c in constraints):
        loose = [c for c in constraints if c["field"] != "地址"]
        matched = [r for r in rows if _match(r, loose)]
        if matched:
            constraints, loosened = loose, True

    matched.sort(key=lambda r: parse_number(r.get("评分", "")) or 0.0, reverse=True)

    if not matched:
        return {"ok": False, "domain": target_domain, "constraints": constraints,
                "matched": 0,
                "reason": f"库里 {len(rows)} 条 {target_domain}，没有满足 {constraints} 的"}

    wanted = asked_fields(query)
    fields = []
    for f in wanted + ["评分", "人均消费", "价格", "门票", "营业时间", "推荐菜", "游玩时间"]:
        if f not in fields and any(f in r for r in matched[:top_k]):
            fields.append(f)

    yesno = any(w in query for w in ("是否", "有没有", "能提供", "可以吗", "么？", "吗？"))
    head = f"是，有 {len(matched)} 家满足条件。\n" if yesno else ""

    lines = [f"{head}找到 {len(matched)} 家{target_domain}，按评分推荐前 {min(top_k, len(matched))} 家："]
    if loosened:
        lines.insert(0, "（库里地址缺少行政区信息，已放宽到全市范围，距离未校验）")
    for i, row in enumerate(matched[:top_k], 1):
        lines.append(f"{i}. {_fmt(row, fields)}")

    return {"ok": True, "domain": target_domain, "constraints": constraints,
            "matched": len(matched), "rows": matched[:top_k],
            "answer": "\n".join(lines), "source": "商户库（本地，0 token）",
            "loosened": loosened}


if __name__ == "__main__":
    for q in sys.argv[1:] or ["帮我找人均50-100元的餐馆", "评分4.5以上的酒店",
                              "门票30元以内的景点", "5分的酒店"]:
        r = execute("餐馆" if "餐馆" in q else ("酒店" if "酒店" in q else "景点"), q)
        print(f"『{q}』\n  约束={r['constraints']}\n  {str(r.get('answer') or r.get('reason'))[:120]}\n")
