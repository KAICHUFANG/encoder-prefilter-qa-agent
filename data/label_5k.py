"""构建 5,000 条数据集并补标：复用已标的 test，补齐 train/dev 抽样。

组成：
    test 划分全部 3,525 条（已标 3,522 条，直接复用）
    + train/dev 按对话抽样约 1,478 条（需补标）

标签方式与之前完全一致（同一套 SYSTEM/TOOLS）：
    判官看「对话上下文 + 用户问题 + 代码回答」，可调 search_venues/get_venue/
    filter_venues/nearby 核验 -> 0=代码答对，1=代码没答对

输出：
    data/crosswoz_5k/_progress.jsonl          断点续传
    data/crosswoz_5k/{train,dev,test}.jsonl   按对话切 70/15/15
    data/crosswoz_5k/labels.json + stats.json

用法：
    python data/label_5k.py --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
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
from relabel_gate_agent import SYSTEM, TOOL_IMPL, load_env, chat, parse_label  # noqa: E402

ENV_FILE = Path(r"C:\Users\xiong\Desktop\制度问答Agent\.env")
KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = {"train": PROJ / "data/raw/crosswoz/extracted/train.json",
       "dev": PROJ / "data/raw/crosswoz/extracted/val.json",
       "test": PROJ / "data/raw/crosswoz/extracted/test.json"}
ITEMS = {s: PROJ / "data" / "crosswoz" / f"{s}.jsonl" for s in ("train", "dev", "test")}
PREV = PROJ / "data" / "crosswoz_gate_full" / "_progress.jsonl"   # 已有的 test 标注
OUT = PROJ / "data" / "crosswoz_5k"
PROGRESS = OUT / "_progress.jsonl"
TARGET_EXTRA = 1478
TURNS = 4
MAX_ROUNDS = 8

DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜", "烤鸭", "火锅", "小吃"),
                "酒店": ("酒店", "宾馆", "住宿", "住一晚", "房费", "房价", "房间", "入住", "住下"),
                "景点": ("景点", "公园", "景区", "游玩", "博物馆", "长城", "乐园", "门票", "票价", "逛")}
HOTEL_WORDS = ("会议室", "早餐", "24小时热水", "无烟房", "健身房", "游泳池", "温泉",
               "叫醒", "接机", "商务中心", "经济型", "舒适型", "高档型", "豪华型",
               "住宿", "住下", "入住")


def guess_domain(text: str) -> str:
    text = text
    for w in ("周边餐馆", "周边景点", "周边酒店"):
        text = text.replace(w, "")
    if any(w in text for w in HOTEL_WORDS):
        return "酒店"
    hits = []
    for dom, words in DOMAIN_HINTS.items():
        for w in words:
            if w in text:
                hits.append((dom, text.rfind(w)))
    return max(hits, key=lambda x: x[1])[0] if hits else "餐馆"


def build_sample() -> list[dict]:
    """test 全部 + train/dev 按对话抽样，共约 5,000 条。"""
    sample = []
    test_rows = [json.loads(l) for l in ITEMS["test"].open(encoding="utf-8") if l.strip()]
    for i, r in enumerate(test_rows):
        sample.append({"id": f"s-test-{i:05d}", "split_src": "test", "text": r["text"],
                       "dialog_id": r["dialog_id"], "turn": r.get("turn", 0)})

    extra = []
    for split in ("train", "dev"):
        rows = [json.loads(l) for l in ITEMS[split].open(encoding="utf-8") if l.strip()]
        by_dlg = defaultdict(list)
        for i, r in enumerate(rows):
            by_dlg[str(r["dialog_id"])].append({"id": f"s-{split}-{i:05d}", "split_src": split,
                                                "text": r["text"], "dialog_id": r["dialog_id"],
                                                "turn": r.get("turn", 0)})
        extra.append((split, by_dlg))

    # 交替从 train/dev 抽对话，直到够数
    rng = random.Random(config.SEED + 7)
    pools = []
    for split, by_dlg in extra:
        dlgs = sorted(by_dlg)
        rng.shuffle(dlgs)
        pools.append([(split, d, by_dlg[d]) for d in dlgs])
    i = 0
    added = 0
    while added < TARGET_EXTRA and any(pools):
        for k, pool in enumerate(pools):
            if not pool:
                continue
            split, dlg, items = pool.pop()
            sample.extend(items)
            added += len(items)
            if added >= TARGET_EXTRA:
                break
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    key = env.get("DEEPSEEK_API_KEY", "")
    base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("MODEL_NAME", "deepseek-chat")

    executor.KB_PATH = KB
    executor.load_rows.cache_clear()
    names = sorted({r["name"] for r in executor.load_rows() if r.get("name")}, key=len, reverse=True)

    prev = {}
    if PREV.exists():
        for line in PREV.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("intent") in (0, 1):
                    prev[r["text"]] = r
    print(f"可复用标注 {len(prev)} 条")

    sample = build_sample()
    print(f"样本 {len(sample)} 条（test {sum(1 for s in sample if s['split_src']=='test')} / "
          f"train {sum(1 for s in sample if s['split_src']=='train')} / "
          f"dev {sum(1 for s in sample if s['split_src']=='dev')}）")

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROGRESS.exists():
        for line in PROGRESS.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"已完成 {len(done)} 条")

    raw_cache = {}
    raw_lock = threading.Lock()

    def raw_of(split: str) -> dict:
        with raw_lock:
            if split not in raw_cache:
                raw_cache[split] = json.loads(RAW[split].read_text(encoding="utf-8"))
            return raw_cache[split]

    def one(t: dict) -> dict | None:
        if t["id"] in done:
            return None
        if t["text"] in prev:
            p = prev[t["text"]]
            return {**t, "intent": p["intent"], "reason": p.get("reason", ""),
                    "tool_rounds": p.get("tool_rounds", 0),
                    "code_answer": str(p.get("code_answer", ""))[:300], "reused": True}
        raw = raw_of(t["split_src"])
        msgs = raw.get(str(t["dialog_id"]), {}).get("messages", [])
        pos = int(t["turn"]) if 0 <= int(t["turn"]) < len(msgs) else None
        ctx, ref = [], None
        if pos is not None:
            for msg in msgs[max(0, pos - TURNS * 2):pos]:
                role = "用户" if msg.get("role") == "usr" else "客服"
                c = str(msg.get("content", "")).strip()[:50]
                if c:
                    ctx.append(f"{role}：{c}")
                hit = next((n for n in names if n in str(msg.get("content", ""))), None)
                if hit:
                    ref = hit
        res = executor.execute(guess_domain(t["text"]), t["text"], top_k=3,
                               context={"last_name": ref} if ref else None)
        code_answer = res.get("answer") if res.get("ok") else f"（代码未命中）{res.get('reason','')}"
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"【对话上下文】\n" + ("\n".join(ctx) if ctx else "（无）") +
                     f"\n\n【用户最后一问】{t['text']}\n\n【代码系统的回答】\n{code_answer}"}]
        label, reason, rounds = None, "", 0
        for rounds in range(1, MAX_ROUNDS + 1):
            try:
                data = chat(base, key, model, messages)
            except Exception as exc:
                time.sleep(1.0)
                reason = f"{type(exc).__name__}: {exc}"
                continue
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                label, reason = parse_label(msg.get("content") or "")
                break
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
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
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)[:1200]})
        if label not in (0, 1):
            return None
        return {**t, "intent": label, "reason": reason, "tool_rounds": rounds,
                "code_answer": str(code_answer)[:300], "reused": False}

    todo = [t for t in sample if t["id"] not in done]
    print(f"本轮待处理 {len(todo)} 条（其中可复用 {sum(1 for t in todo if t['text'] in prev)} 条）")
    lock = threading.Lock()
    written = [0]
    t0 = time.time()
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, t) for t in todo]
        for fut in as_completed(futures):
            try:
                out = fut.result()
            except Exception:
                continue
            if out is None:
                continue
            with lock:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                written[0] += 1
                if written[0] % 200 == 0:
                    print(f"  已写 {written[0]}/{len(todo)} | {written[0]/max(1,time.time()-t0):.1f} 条/秒", flush=True)

    rows = [json.loads(line) for line in PROGRESS.open(encoding="utf-8") if line.strip()]
    print(f"\n本轮新增 {written[0]} 条，累计 {len(rows)} 条 | {(time.time()-t0)/60:.1f} 分钟")
    print("标签分布:", dict(Counter(r["intent"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
