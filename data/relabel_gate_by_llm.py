"""给门控打标签：DeepSeek 看「用户问题 + 代码回答」判代码答得对不对。

这是唯一正确的标签来源：
    · 判断"回答得对不对"需要理解能力，所以由更强的语言模型做
    · 但必须把代码的实际输出给它看，否则判的是"理论上能不能"，不是"实际能不能"
    · 标签含义：0 = 代码已经答对（该交给代码）；1 = 代码没答对（该交给大模型）

代码侧的回答来自当前执行器（官方库 + 周边关系表 + 约束过滤）。
为了让判官能核对事实，prompt 里同时给出代码用到的数据（命中的行 / 参照门店档案）。

输出：data/crosswoz_gate_llm/{train,dev,test}.jsonl
    每行 {"id","text","intent":0/1,"gate","reason","code_answer","split"}

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python data/relabel_gate_by_llm.py --limit 5
    python data/relabel_gate_by_llm.py
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

from app import executor_frozen as executor  # noqa: E402  # 冻结版本，保证标签可复现

ENV_FILE = Path(r"C:\Users\xiong\Desktop\制度问答Agent\.env")
SRC = PROJ / "data" / "crosswoz_gate"
DST = PROJ / "data" / "crosswoz_gate_llm"
PROGRESS = DST / "_progress.jsonl"
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

SYSTEM = """你是客服系统的质检员。输入包含：用户问题、代码系统给出的回答、代码用到的库数据。
请判断：这段代码回答是否已经**正确、完整**地回答了用户的问题。

判 0（代码已答对）：
- 事实与库数据一致，覆盖了用户所问的要点
- 筛选类问题给出了满足条件的对象
- 条件确实无解时，明确说明"没有符合条件的"（且库数据支持这一点）

判 1（代码没答对，该交给大模型）：
- 答非所问、漏掉用户问的关键信息
- 与库数据矛盾，或编造了库中没有的事实
- 代码说"查不到"，但库数据里其实有对应信息
- 只答了多问中的一部分

只输出 JSON，不要别的话：{"label": 0或1, "reason": "20字以内"}"""


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
    text = re.sub(r"周边(餐馆|景点|酒店|的餐馆|的景点|的酒店)", "", text)
    if any(w in text for w in HOTEL_WORDS):
        return "酒店"
    hits = []
    for dom, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text:
                hits.append((dom, text.rfind(w)))
    return max(hits, key=lambda x: x[1])[0] if hits else "餐馆"


def call(base: str, key: str, model: str, question: str, answer: str, evidence: str) -> tuple[int, str]:
    user = (f"用户问题：{question}\n\n代码回答：{answer}\n\n代码用到的库数据：\n{evidence or '（无）'}")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 120,
    }).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"].strip().strip("`")
    content = content.replace("json", "", 1).strip()
    obj = json.loads(content)
    return int(obj["label"]), str(obj.get("reason", ""))[:40]


def row_brief(r: dict) -> str:
    keys = ("评分", "价格", "门票", "人均消费", "游玩时间", "营业时间", "电话", "地址",
            "酒店类型", "推荐菜", "地铁", "周边景点", "周边餐馆", "周边酒店")
    parts = [f"{k}={str(r[k])[:40]}" for k in keys if r.get(k)]
    fac = [k.replace("酒店设施-", "设施-") for k in r
           if k.startswith("酒店设施-") and str(r[k]).strip()][:8]
    return f"- 【{r.get('name','')}】" + "｜".join(parts + fac)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    key = env.get("DEEPSEEK_API_KEY", "")
    base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("MODEL_NAME", "deepseek-chat")
    if not key:
        print("[失败] 没读到 DEEPSEEK_API_KEY")
        return 1

    executor.KB_PATH = KB
    executor.load_rows.cache_clear()
    kb_rows = executor.load_rows()
    names = sorted({r["name"] for r in kb_rows if r.get("name")}, key=len, reverse=True)
    by_name = {r["name"]: r for r in kb_rows}

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
    done = set()
    if PROGRESS.exists():
        for line in PROGRESS.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"已标注 {len(done)} 条")

    items = []
    for split in ("train", "dev", "test"):
        path = SRC / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                items.append({"id": r["id"], "text": r["text"], "split": split})
    if args.limit:
        items = items[: args.limit]
    print(f"待标注 {len(items)} 条（问题 + 代码回答 -> DeepSeek 判对错）")

    with PROGRESS.open("a", encoding="utf-8", newline="\n") as f:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            ctx = context_of(it["id"])
            res = executor.execute(guess_domain(it["text"]), it["text"], top_k=3, context=ctx)
            if res.get("ok"):
                code_answer = res.get("answer") or ""
                evidence = "\n".join(row_brief(r) for r in (res.get("rows") or [])[:3])
            else:
                code_answer = f"（代码未命中）{res.get('reason','')}"
                ref = (ctx or {}).get("last_name")
                evidence = row_brief(by_name[ref]) if ref and ref in by_name else ""

            label, reason = None, ""
            for _ in range(3):
                try:
                    label, reason = call(base, key, model, it["text"], code_answer, evidence)
                    break
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    time.sleep(1.5)
            rec = {**it, "intent": label, "gate": "code" if label == 0 else "llm",
                   "reason": reason, "code_answer": code_answer[:300]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{i+1}/{len(items)}] {it['text'][:24]:<26} -> {rec['gate']}  {reason[:20]}")
            time.sleep(0.15)

    rows = [json.loads(line) for line in PROGRESS.open(encoding="utf-8") if line.strip()]
    ok = [r for r in rows if r["intent"] in (0, 1)]
    from collections import Counter
    for split in ("train", "dev", "test"):
        sub = [r for r in ok if r["split"] == split]
        (DST / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"id": r["id"], "text": r["text"], "intent": r["intent"],
                                  "split": split, "gate": r["gate"], "reason": r["reason"]},
                                 ensure_ascii=False) for r in sub) + "\n", encoding="utf-8")
        print(f"{split}: {len(sub)} 条 {dict(Counter(r['gate'] for r in sub))}")

    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码已答对（交代码） 1=代码没答对（交大模型）；由 DeepSeek 依据问题+代码回答判定"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成，共 {len(ok)} 条 -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
