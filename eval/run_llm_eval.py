"""真调大模型：验证门控判给大模型的那批请求，模型到底答不答得对。

输入 = 对话上下文（CrossWOZ 原始 test.json 取前若干轮）
      + 参照门店档案（用户说"它/这家"指的是谁）
      + 检索候选（向量或规则）
模型 = DeepSeek（key 从 制度问答Agent/.env 读，不落盘不外传）
输出 = eval/llm_answers.jsonl；人工判分写到 eval/llm_grades.jsonl

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/run_llm_eval.py --retriever vector
    python eval/run_llm_eval.py --dry-run --limit 3 --retriever rule
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

import config  # noqa: E402

ENV_FILE = Path(r"C:\Users\xiong\Desktop\制度问答Agent\.env")
KB_OFFICIAL = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW_JSON = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
CTX_TURNS = 3

SYSTEM = (
    "你是旅游客服助手。资料里有两个关键区块，必须按优先级使用：\n"
    "1)【周边关系（来自数据集）】= 该门店真实的周边清单。用户问“附近/周边”时，"
    "直接从对应清单里挑 2-4 个回答，不要说自己不知道。\n"
    "2)【按约束筛选出的商户】= 系统已按用户的评分/价格/门票/设施/菜品条件筛过的结果。"
    "只要这个区块不为空，就说明库里存在符合条件的商户——必须从中推荐，并点明它满足哪个条件；"
    "禁止回答“没有”或“需要更多信息”。\n"
    "【语义相近的候选】只作补充参考，不能当作用户条件的证据。\n"
    "只有当上述区块都为空、且资料里确实没有用户要的字段时，才回答“需要更多信息”。\n"
    "不要编造资料里没有的名字、电话或数字。回答控制在 80 字以内。"
)

DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅", "小吃"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房费", "房价", "房间", "退房"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "长城", "乐园", "门票",
                         "票价", "景色", "逛")}


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


def guess_domain(text: str) -> str:
    # 「周边餐馆/周边景点」是关系目标，不是主领域，先剔掉再判断
    text = re.sub(r"周边(餐馆|景点|酒店|的餐馆|的景点|的酒店)", "", text)
    # 出现酒店设施/类型词时，主领域一定是酒店（"去完景点，找个有早餐的"仍是找酒店）
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


def load_raw_dialogues() -> dict[str, list[dict]]:
    if not RAW_JSON.exists():
        return {}
    data = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    return {str(k): v.get("messages", []) for k, v in data.items()}


def build_context(raw: dict, dialog_id: str, text: str, turns: int = CTX_TURNS) -> list[str]:
    msgs = raw.get(str(dialog_id), [])
    pos = next((i for i, m in enumerate(msgs)
                if str(m.get("content", "")).strip() == text.strip()), None)
    if pos is None:
        return []
    out = []
    for m in msgs[max(0, pos - turns * 2):pos]:
        role = "用户" if m.get("role") == "usr" else "客服"
        out.append(f"{role}：{str(m.get('content', ''))[:60]}")
    return out


def index_of(row_id: str) -> int | None:
    m = re.search(r"(\d+)$", str(row_id))
    return int(m.group(1)) if m else None


def kot_rows() -> list[dict]:
    return json.loads(KB_OFFICIAL.read_text(encoding="utf-8"))["rows"]


def referent_names(question: str, ctx_lines: list[str], limit: int = 2) -> list[str]:
    """从问句+上下文里找出被指代的门店（库内全名精确匹配，取最近提到的）。"""
    names = sorted({r["name"] for r in kot_rows() if r.get("name")}, key=len, reverse=True)
    found: list[str] = []
    for text in reversed(ctx_lines + [question]):
        for n in names:
            if n in text and n not in found:
                found.append(n)
                if len(found) >= limit:
                    return found
    return found


def venue_profile(name: str) -> str:
    """返回该门店的完整档案（一行），含数据集自带的「周边」关系表。"""
    row = next((r for r in kot_rows() if r.get("name") == name), None)
    if not row:
        return f"- 【{name}】（库中无此门店记录）"
    keys = ("评分", "价格", "门票", "人均消费", "游玩时间", "营业时间",
            "电话", "地址", "酒店类型", "推荐菜", "地铁")
    parts = [f"{k}={row[k]}" for k in keys if row.get(k)]
    fac = [k.replace("酒店设施-", "设施-") for k in row
           if k.startswith("酒店设施-") and str(row[k]).strip()][:8]
    parts += fac
    for f in ("周边景点", "周边餐馆", "周边酒店"):
        if row.get(f):
            names = str(row[f]).split("、")
            parts.append(f"{f}={('、'.join(names[:8]))}{'等' if len(names) > 8 else ''}")
    return f"- 【{name}】" + "｜".join(parts)


def relation_block(question: str, refs: list[str]) -> str:
    """周边问句：直接把参照店的周边表拉出来（不靠模型猜）。"""
    if not any(w in question for w in ("周边", "附近", "旁边", "周围")) or not refs:
        return ""
    from app import executor

    lines = []
    for ref in refs[:2]:
        row = next((r for r in kot_rows() if r.get("name") == ref), None)
        if not row:
            continue
        for field in ("周边景点", "周边餐馆", "周边酒店"):
            names = [n.strip() for n in str(row.get(field, "")).split("、") if n.strip()]
            if not names:
                continue
            detail = []
            for n in names[:6]:
                r2 = next((x for x in kot_rows() if x.get("name") == n), None)
                if r2:
                    extra = "｜".join(f"{k}{r2[k]}" for k in ("评分", "人均消费", "门票", "价格", "电话")
                                      if r2.get(k))
                    detail.append(f"{n}（{extra}）" if extra else n)
                else:
                    detail.append(n)
            lines.append(f"- {ref} 的{field}（{len(names)} 个）：" + "；".join(detail))
    return "\n".join(lines)


def constraint_block(question: str) -> str:
    """约束过滤：调用执行器把满足条件的行筛出来（评分/价格/设施/菜品）。"""
    from app import executor

    try:
        res = executor.execute(guess_domain(question), question, top_k=5)
    except Exception:
        return ""
    if not res.get("ok") or res.get("constraints", []) == []:
        return ""
    rows = res.get("rows") or []
    if not rows:
        return ""
    lines = []
    for r in rows[:5]:
        fields = [f"{k} {r[k]}" for k in ("评分", "人均消费", "门票", "价格", "电话", "地址", "酒店类型")
                  if r.get(k)]
        lines.append(f"- {r['name']}｜" + "｜".join(fields))
    cons = "、".join(f"{c['field']}{c['op']}{c.get('value','')}" for c in res["constraints"])
    return f"（按约束筛选：{cons}，命中 {res.get('matched')} 家）\n" + "\n".join(lines)


def retrieve(domain: str, text: str) -> str:
    """规则检索：执行器筛出候选，按评分排序。"""
    from app import executor

    try:
        res = executor.execute(domain, text, top_k=3)
    except Exception as exc:
        return f"（检索失败：{exc}）"
    rows = res.get("rows") or []
    if not rows:
        return "（无候选）"
    lines = []
    for r in rows:
        fields = [f"{k} {r[k]}" for k in ("评分", "人均消费", "价格", "门票", "地址", "电话", "营业时间")
                  if r.get(k)]
        lines.append(f"- {r.get('name')}｜" + "｜".join(fields))
    return "\n".join(lines)


def retrieve_hybrid(question: str, ctx_lines: list[str], top_k: int = 3) -> tuple[str, list[str]]:
    """三段式检索：周边关系表 + 约束过滤 + 向量召回。

    为什么这么拼：
        向量只按语义相似给 top-k，不保证满足约束、也不读「周边」表；
        而周边表就在库里、约束过滤代码也现成——不接进来等于白丢数据。
    """
    refs = referent_names(question, ctx_lines)
    blocks = []
    rel = relation_block(question, refs)
    if rel:
        blocks.append("【周边关系（来自数据集）】\n" + rel)
    cons = constraint_block(question)
    if cons:
        blocks.append("【按约束筛选出的商户】\n" + cons)
    vec_text, vec_refs = retrieve_vector(question, ctx_lines, top_k=top_k)
    if not refs:
        refs = vec_refs
    if vec_text:
        blocks.append("【语义相近的候选（仅供参考）】\n" + vec_text)
    return "\n\n".join(blocks), refs


def retrieve_vector(question: str, ctx_lines: list[str], top_k: int = 5) -> tuple[str, list[str]]:
    """向量检索：按语义相似度取候选；参照对象仍优先用精确店名。"""
    from app import vector_search

    query = "\n".join(ctx_lines[-2:] + [question])
    hits = vector_search.search(query, top_k=top_k, kind="venue", domain=guess_domain(question))
    lines = []
    for h in hits:
        row = h["row"]
        keys = ("评分", "价格", "门票", "人均消费", "游玩时间", "营业时间", "电话", "地址",
                "酒店类型", "推荐菜")
        parts = [f"{k}={row[k]}" for k in keys if row.get(k)]
        fac = [k.replace("酒店设施-", "设施-") for k in row
               if k.startswith("酒店设施-") and str(row[k]).strip()][:6]
        lines.append(f"- 【{h['name']}】" + "｜".join(parts + fac) + f"（相似度 {h['score']:.2f}）")
    refs = referent_names(question, ctx_lines)
    # 只有出现指代词（这家/它/该）时才把向量 top1 当参照，避免张冠李戴
    if not refs and hits and hits[0]["score"] >= 0.52 and any(
            w in question for w in ("这家", "这个", "它", "该", "那边", "这里")):
        refs = [hits[0]["name"]]
    return "\n".join(lines), refs


def call_llm(base: str, key: str, model: str, messages: list[dict], timeout: int = 60) -> str:
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": 0.2, "max_tokens": 200}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retriever", choices=["rule", "vector", "hybrid"], default="hybrid")
    ap.add_argument("--out", default="eval/llm_answers.jsonl")
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    key = env.get("DEEPSEEK_API_KEY", "")
    base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("MODEL_NAME", "deepseek-chat")
    if not key and not args.dry_run:
        print("[失败] 没读到 DEEPSEEK_API_KEY")
        return 1

    # 商户库固定指向官方库
    from app import executor

    executor.KB_PATH = KB_OFFICIAL
    executor.load_rows.cache_clear()

    rows = [json.loads(line) for line in (PROJ / "data" / "crosswoz" / "test.jsonl").open(encoding="utf-8")
            if line.strip()]
    raw = load_raw_dialogues()
    print(f"官方库 {len(kot_rows())} 家 · 原始对话 {len(raw)} 段（用于取上下文）")

    test_rows = [json.loads(line) for line in config.TEST_JSONL.open(encoding="utf-8") if line.strip()]
    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    targets = []
    for r in test_rows:
        out = p.predict(r["text"], top_k=1)
        if int(out["intent"]) == 1:
            targets.append(r)
    if args.limit:
        targets = targets[: args.limit]
    print(f"门控判给大模型：{len(targets)} 条，检索方式={args.retriever}")

    out_path = PROJ / args.out
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(targets):
            idx = index_of(r["id"])
            src = rows[idx] if idx is not None and idx < len(rows) else {}
            ctx = build_context(raw, str(src.get("dialog_id", "")), r["text"])
            if args.retriever == "hybrid":
                cands, refs = retrieve_hybrid(r["text"], ctx)
            elif args.retriever == "vector":
                cands, refs = retrieve_vector(r["text"], ctx)
            else:
                cands = retrieve(guess_domain(r["text"]), r["text"])
                refs = referent_names(r["text"], ctx)
            profiles = "\n".join(venue_profile(n) for n in refs) if refs else "（未识别到被指代的门店）"
            prompt = (
                f"【对话上下文】\n" + ("\n".join(ctx) if ctx else "（无）") +
                f"\n\n【参照门店档案】\n{profiles}"
                f"\n\n【检索候选】\n{cands}"
                f"\n\n【用户最后一句话】\n{r['text']}"
            )
            rec = {"id": r["id"], "text": r["text"], "context": ctx, "candidates": cands,
                   "referents": refs, "profiles": profiles, "prompt": prompt}
            if args.dry_run:
                print(f"\n--- {r['id']} ---\n{prompt[:500]}")
                continue
            try:
                rec["answer"] = call_llm(base, key, model, [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}])
                rec["ok"] = True
            except Exception as exc:
                rec["answer"] = None
                rec["ok"] = False
                rec["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(f"[{i+1}/{len(targets)}] {r['text'][:24]:<26} -> "
                  f"{(rec.get('answer') or rec.get('error') or '')[:46]}")
            time.sleep(0.4)

    if not args.dry_run:
        print(f"\n写出 {out_path}（{written} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
