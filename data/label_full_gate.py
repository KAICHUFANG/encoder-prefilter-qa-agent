"""全量标注：把 test 集里全部 3,525 条问句都交给 agent 判官核验。

和之前 400 条版本的区别：
    · 之前：抽样 400 条（为省调用）
    · 现在：全量 3,525 条，全部标完，再按对话切 70/15/15

判官逻辑不变（与 relabel_gate_agent.py 同一套工具与 rubric）：
    看「对话上下文 + 问题 + 代码回答」-> 可调 search_venues/get_venue/filter_venues/nearby 查库核验
    -> 输出 {"label": 0/1, "reason": ...}
    label 0 = 代码已答对；1 = 代码没答对（该交大模型）

并发：默认 6 路；断点续传（_progress.jsonl）

输出：
    data/crosswoz_gate_full/_progress.jsonl
    data/crosswoz_gate_full/{train,dev,test}.jsonl + labels.json

用法：
    python data/label_full_gate.py --workers 6 --limit 5   # 冒烟
    python data/label_full_gate.py --workers 6             # 全量
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "data"))

import config  # noqa: E402
from app import executor_frozen as executor  # noqa: E402
from relabel_gate_agent import SYSTEM, TOOLS, TOOL_IMPL, load_env, chat, parse_label  # noqa: E402

ENV_FILE = Path(r"C:\Users\xiong\Desktop\制度问答Agent\.env")
KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"
OUT_DIR = PROJ / "data" / "crosswoz_gate_full"
PROGRESS = OUT_DIR / "_progress.jsonl"
TURNS = 4
MAX_ROUNDS = 8

DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅", "小吃"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房费", "房价", "房间", "入住", "住下"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "长城", "乐园", "门票",
                         "票价", "逛")}
HOTEL_WORDS = ("会议室", "早餐", "24小时热水", "无烟房", "健身房", "游泳池", "温泉",
               "叫醒", "接机", "商务中心", "经济型", "舒适型", "高档型", "豪华型",
               "住宿", "住下", "入住")


def guess_domain(text: str) -> str:
    text = re.sub(r"周边(餐馆|景点|酒店)", "", text)
    if any(w in text for w in HOTEL_WORDS):
        return "酒店"
    hits = []
    for dom, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text:
                hits.append((dom, text.rfind(w)))
    return max(hits, key=lambda x: x[1])[0] if hits else "餐馆"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
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

    items = [json.loads(line) for line in ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    print(f"全量问句 {len(items)} 条，对话 {len({i['dialog_id'] for i in items})} 段")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROGRESS.exists():
        for line in PROGRESS.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"已完成 {len(done)} 条，剩余 {len(items) - len(done)}")

    lock = threading.Lock()
    written = [0]

    def one(idx: int) -> dict | None:
        r = items[idx]
        rid = f"full-{idx:04d}"
        if rid in done:
            return None
        text = r["text"]
        dlg = str(r["dialog_id"])
        msgs = raw.get(dlg, {}).get("messages", [])
        pos = next((i for i, x in enumerate(msgs)
                    if str(x.get("content", "")).strip() == text.strip()), None)
        ctx_lines, ref = [], None
        if pos is not None:
            for msg in msgs[max(0, pos - TURNS * 2):pos]:
                role = "用户" if msg.get("role") == "usr" else "客服"
                content = str(msg.get("content", "")).strip()[:50]
                if content:
                    ctx_lines.append(f"{role}：{content}")
                hit = next((n for n in names if n in str(msg.get("content", ""))), None)
                if hit:
                    ref = hit

        res = executor.execute(guess_domain(text), text, top_k=3,
                               context={"last_name": ref} if ref else None)
        code_answer = res.get("answer") if res.get("ok") else f"（代码未命中）{res.get('reason','')}"
        user = (f"【对话上下文】\n" + ("\n".join(ctx_lines) if ctx_lines else "（无）") +
                f"\n\n【用户最后一问】{text}\n\n【代码系统的回答】\n{code_answer}")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]

        label, reason, rounds = None, "", 0
        for rounds in range(1, MAX_ROUNDS + 1):
            try:
                data = chat(base, key, model, messages)
            except Exception as exc:
                time.sleep(1.2)
                reason = f"{type(exc).__name__}: {exc}"
                continue
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

        return {"id": rid, "idx": idx, "text": text, "dialog_id": dlg,
                "intent": label, "reason": reason, "tool_rounds": rounds,
                "code_answer": str(code_answer)[:300]}

    todo = [i for i in range(len(items)) if f"full-{i:04d}" not in done]
    if args.limit:
        todo = todo[: args.limit]

    t0 = time.time()
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, i): i for i in todo}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:
                    continue
                if rec is None:
                    continue
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    written[0] += 1
                    if written[0] % 100 == 0:
                        ok = sum(1 for _ in PROGRESS.open(encoding="utf-8"))
                        print(f"  已写 {ok} 条 | 用时 {time.time()-t0:.0f}s", flush=True)

    print(f"\n本轮新增 {written[0]} 条，用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
