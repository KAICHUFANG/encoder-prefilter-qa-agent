"""Agent 式标注：判官可以自己查库，验证代码回答到底对不对。

为什么要有工具：
    只看"问题 + 代码回答"时，判官无法核实数字真假、也不知道"它"指谁；
    于是把错的答案判成对的（上一轮 60 条重判翻了 14 条，全是这个原因）。

判官能用的工具（只读，和线上执行器同一份冻结代码）：
    search_venues(query, domain)      语义检索商户
    get_venue(name)                   取某家完整档案（含周边表）
    filter_venues(...)                按评分/价格/门票/设施/菜品/区县筛选
    nearby(name, target)              取某家的周边景点/餐馆/酒店

判官流程：看上下文与代码回答 -> 需要核实就调工具 -> 最后输出 {"label":0/1,"reason":...}
    label 0 = 代码已经答对；1 = 代码没答对（该交给大模型）

输出：data/crosswoz_gate_v2/{train,dev,test}.jsonl（断点续传）

用法：
    python data/relabel_gate_agent.py --limit 5
    python data/relabel_gate_agent.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

from app import executor_frozen as executor  # noqa: E402
from app import vector_search  # noqa: E402

ENV_FILE = Path(r"C:\Users\xiong\Desktop\制度问答Agent\.env")
SRC = PROJ / "data" / "crosswoz_gate_llm"
PROGRESS = PROJ / "data" / "crosswoz_gate_v2" / "_progress.jsonl"
DST = PROJ / "data" / "crosswoz_gate_v2"
KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"
TURNS = 4
MAX_ROUNDS = 8

SYSTEM = """你是客服问答的质检员。给你一段对话、用户最后一问、以及**代码系统给出的回答**。
你的任务：判断代码回答是否**正确且完整**地回答了用户。

重要：你**不能**只凭代码回答里的说法下结论——必须用工具到库里核实。
判断重点：
1) 代码回答的对象是否正确（多轮里"它/这家"指的是哪家）
2) 回答里的事实是否与库一致（评分/门票/价格/电话/营业时间/周边关系）
3) 是否满足了用户提出的全部条件（评分阈值、价格区间、免费、时长、设施、菜品等）
4) 该说"没有符合条件的"时是否确有依据

工具用法提示：
- 不知道指哪家 -> get_venue / search_venues 核对该店是否存在
- 要验证筛选类回答 -> filter_venues 自己筛一遍，看代码给的对象是否真满足条件
- 周边类问题 -> nearby(name, target) 取真实周边清单对比
- 用户问的字段库里没有 -> 代码若给了数值就是编造，判 1

核验完成后，**只输出一行 JSON**：
{"label": 0 或 1, "reason": "30字以内的核验结论"}
label=0 表示代码已答对；label=1 表示代码没答对（该交给大模型）。"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_venues", "description": "按语义检索商户，返回名称与关键字段",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "domain": {"type": "string", "enum": ["餐馆", "酒店", "景点"]}},
            "required": ["query", "domain"]}}},
    {"type": "function", "function": {
        "name": "get_venue", "description": "取某家商户的完整档案（含周边表）",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "filter_venues", "description": "按条件筛选商户",
        "parameters": {"type": "object", "properties": {
            "domain": {"type": "string", "enum": ["餐馆", "酒店", "景点"]},
            "min_rating": {"type": "number"}, "max_rating": {"type": "number"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
            "free_ticket": {"type": "boolean"},
            "min_hours": {"type": "number"}, "max_hours": {"type": "number"},
            "facility": {"type": "string"}, "dish": {"type": "string"},
            "hotel_type": {"type": "string"}, "limit": {"type": "number"}},
            "required": ["domain"]}}},
    {"type": "function", "function": {
        "name": "nearby", "description": "取某家商户的周边景点/餐馆/酒店清单",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "target": {"type": "string", "enum": ["景点", "餐馆", "酒店"]}},
            "required": ["name", "target"]}}},
]

FIELD_KEYS = ("评分", "价格", "门票", "人均消费", "游玩时间", "营业时间", "电话", "地址",
              "酒店类型", "推荐菜", "地铁")


def brief(row: dict, with_nearby: bool = False) -> str:
    parts = [f"【{row.get('name','')}】{row.get('domain','')}"]
    for k in FIELD_KEYS:
        if row.get(k):
            parts.append(f"{k}={str(row[k])[:60]}")
    fac = [k.replace("酒店设施-", "") for k in row
           if k.startswith("酒店设施-") and str(row[k]).strip()][:10]
    if fac:
        parts.append("设施=" + "/".join(fac))
    if with_nearby:
        for f in ("周边景点", "周边餐馆", "周边酒店"):
            if row.get(f):
                names = str(row[f]).split("、")
                parts.append(f"{f}={('、'.join(names[:10]))}{'等' if len(names) > 10 else ''}")
    return "｜".join(parts)


def tool_search(args: dict) -> str:
    hits = vector_search.search(args["query"], top_k=6, kind="venue",
                                domain=args.get("domain"))
    return "\n".join(brief(h["row"]) for h in hits) or "（无结果）"


def tool_get(args: dict) -> str:
    row = next((r for r in executor.load_rows() if r.get("name") == args["name"]), None)
    return brief(row, with_nearby=True) if row else f"（库中无 {args['name']}）"


def _num(v):
    m = re.search(r"(\d+(?:\.\d+)?)", str(v or ""))
    return float(m.group(1)) if m else None


def tool_filter(args: dict) -> str:
    dom = args["domain"]
    rows = [r for r in executor.load_rows() if r["domain"] == dom]
    out = []
    for r in rows:
        rating = _num(r.get("评分"))
        if args.get("min_rating") is not None and (rating is None or rating < args["min_rating"]):
            continue
        if args.get("max_rating") is not None and (rating is None or rating > args["max_rating"]):
            continue
        if args.get("free_ticket"):
            if dom != "景点" or str(r.get("门票", "")).strip() not in ("0", "免费"):
                continue
        for lo, hi, field in (("min_price", "max_price", "价格"),
                              ("min_price", "max_price", None)):
            pass
        price = _num(r.get("价格")) if dom == "酒店" else None
        if dom == "酒店" and price is not None:
            if args.get("min_price") is not None and price < args["min_price"]:
                continue
            if args.get("max_price") is not None and price > args["max_price"]:
                continue
        ticket = _num(r.get("门票")) if dom == "景点" else None
        if dom == "景点" and ticket is not None:
            if args.get("min_price") is not None and ticket < args["min_price"]:
                continue
            if args.get("max_price") is not None and ticket > args["max_price"]:
                continue
        hours = _num(r.get("游玩时间")) if dom == "景点" else None
        if args.get("min_hours") is not None and (hours is None or hours < args["min_hours"]):
            continue
        if args.get("max_hours") is not None and (hours is None or hours > args["max_hours"]):
            continue
        if args.get("facility"):
            if not any(k.startswith("酒店设施-") and args["facility"] in k and str(r[k]).strip()
                       for k in r):
                continue
        if args.get("dish") and args["dish"] not in str(r.get("推荐菜", "")):
            continue
        if args.get("hotel_type") and args["hotel_type"] not in str(r.get("酒店类型", "")):
            continue
        out.append(brief(r))
        if len(out) >= int(args.get("limit") or 8):
            break
    return "\n".join(out) or "（无符合条件的商户）"


def tool_nearby(args: dict) -> str:
    row = next((r for r in executor.load_rows() if r.get("name") == args["name"]), None)
    if not row:
        return f"（库中无 {args['name']}）"
    field = {"景点": "周边景点", "餐馆": "周边餐馆", "酒店": "周边酒店"}[args["target"]]
    names = [n.strip() for n in str(row.get(field, "")).split("、") if n.strip()]
    return f"{field}（{len(names)} 个）：" + "、".join(names[:15]) if names else f"{field}：无数据"


TOOL_IMPL = {"search_venues": tool_search, "get_venue": tool_get,
             "filter_venues": tool_filter, "nearby": tool_nearby}


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def chat(base, key, model, messages) -> dict:
    payload = json.dumps({"model": model, "messages": messages,
                          "tools": TOOLS, "temperature": 0,
                          "max_tokens": 600}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def parse_label(content: str) -> tuple[int | None, str]:
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None, content[:60]
    try:
        obj = json.loads(m.group(0))
        return int(obj["label"]), str(obj.get("reason", ""))[:60]
    except Exception:
        return None, content[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    key = env.get("DEEPSEEK_API_KEY", "")
    base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("MODEL_NAME", "deepseek-chat")
    if not key:
        print("[失败] 没有 DEEPSEEK_API_KEY")
        return 1

    executor.KB_PATH = KB
    executor.load_rows.cache_clear()
    names = sorted({r["name"] for r in executor.load_rows() if r.get("name")}, key=len, reverse=True)
    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))

    old = {}
    for line in (SRC / "_progress.jsonl").open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            old[rec["id"]] = rec

    DST.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROGRESS.exists():
        for line in PROGRESS.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"已完成 {len(done)} 条")

    items = []
    for split in ("train", "dev", "test"):
        for line in (SRC / f"{split}.jsonl").open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                items.append({"id": r["id"], "text": r["text"], "split": split})
    if args.limit:
        items = items[: args.limit]
    print(f"待标注 {len(items)} 条（agent 可调工具核验）")

    written = 0
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as f:
        for idx, it in enumerate(items):
            if it["id"] in done:
                continue
            prev = old.get(it["id"], {})
            m = re.search(r"(\d+)$", it["id"])
            pos_i = int(m.group(1)) if m else -1
            ctx = []
            if 0 <= pos_i < len(all_items):
                dlg = all_items[pos_i]["dialog_id"]
                msgs = raw.get(str(dlg), {}).get("messages", [])
                p = next((i for i, x in enumerate(msgs)
                          if str(x.get("content", "")).strip() == it["text"].strip()), None)
                if p is not None:
                    for msg in msgs[max(0, p - TURNS * 2):p]:
                        role = "用户" if msg.get("role") == "usr" else "客服"
                        ctx.append(f"{role}：{str(msg.get('content',''))[:50]}")

            user = (f"【对话上下文】\n" + ("\n".join(ctx) if ctx else "（无）") +
                    f"\n\n【用户最后一问】{it['text']}" +
                    f"\n\n【代码系统的回答】\n{prev.get('code_answer','（无）')}")
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}]

            label, reason, rounds = None, "", 0
            try:
                for rounds in range(1, MAX_ROUNDS + 1):
                    data = chat(base, key, model, messages)
                    msg = data["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        label, reason = parse_label(msg.get("content") or "")
                        break
                    messages.append({"role": "assistant", "content": msg.get("content"),
                                     "tool_calls": calls})
                    for call in calls:
                        fn = call["function"]["name"]
                        try:
                            argv = json.loads(call["function"].get("arguments") or "{}")
                        except Exception:
                            argv = {}
                        try:
                            result = TOOL_IMPL[fn](argv)
                        except Exception as exc:
                            result = f"工具出错：{exc}"
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": str(result)[:1200]})
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"

            rec = {"id": it["id"], "text": it["text"], "split": it["split"],
                   "intent": label, "gate": "code" if label == 0 else ("llm" if label == 1 else "error"),
                   "reason": reason, "tool_rounds": rounds,
                   "old_label": prev.get("intent"), "code_answer": str(prev.get("code_answer", ""))[:300]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if idx % 20 == 0:
                print(f"[{idx+1}/{len(items)}] {it['text'][:22]:<24} -> {rec['gate']} 轮数{rounds} {reason[:24]}")
            time.sleep(0.1)

    rows = [json.loads(line) for line in PROGRESS.open(encoding="utf-8") if line.strip()]
    ok = [r for r in rows if r["intent"] in (0, 1)]
    from collections import Counter
    for split in ("train", "dev", "test"):
        sub = [r for r in ok if r["split"] == split]
        (DST / f"{split}.jsonl").write_text("\n".join(json.dumps(
            {"id": r["id"], "text": r["text"], "intent": r["intent"], "split": split,
             "gate": r["gate"], "reason": r["reason"], "tool_rounds": r.get("tool_rounds")},
            ensure_ascii=False) for r in sub) + "\n", encoding="utf-8")
        print(f"{split}: {len(sub)} 条 {dict(Counter(r['gate'] for r in sub))}")

    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "agent 式核验（可查库）后的标签：0=代码答对 1=代码没答对"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n新增 {written} 条，共 {len(ok)} 条 -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
