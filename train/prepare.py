"""把 KUAKE-QIC 原始文件转成统一的 jsonl。

用法：
    uv run python train/prepare.py --inspect     # 第一步：只看原始文件长什么样
    uv run python train/prepare.py               # 第二步：转换 + 打印类别分布
    uv run python train/prepare.py --limit 200   # 冒烟测试：每个划分只取 200 条
    uv run python train/prepare.py --dedup       # 顺手删掉跨划分的重复文本

为什么要先 --inspect：原始数据可能是 json、tsv、单个文件或三个文件，
四种情况转换代码完全不同。先看，再转，别猜。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

SUFFIXES = {".json", ".jsonl", ".txt", ".tsv", ".csv"}
TEXT_HINTS = ("text", "query", "question", "sentence", "内容", "问句", "问题", "检索词")
LABEL_HINTS = ("label", "intent", "category", "class", "标签", "意图", "类别")
SPLIT_HINTS = {
    "train": ("train", "训练"),
    "dev": ("dev", "valid", "val", "验证"),
    "test": ("test", "测试"),
}


# ---------------------------------------------------------------- 读文件

def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _from_list(items: list):
    if not items:
        return None
    if isinstance(items[0], dict):
        header: list[str] = []
        for d in items:
            for k in d.keys():
                if k not in header:
                    header.append(k)
        rows = [[_cell(d.get(k)) for k in header] for d in items]
        return header, rows
    if isinstance(items[0], (list, tuple)):
        return None, [[_cell(x) for x in r] for r in items]
    return None


def _from_json_obj(obj):
    if isinstance(obj, list):
        return _from_list(obj)
    if isinstance(obj, dict):
        # A: {"text": [...], "label": [...]} —— 列存
        if obj and all(isinstance(v, list) for v in obj.values()):
            keys = list(obj.keys())
            n = max(len(v) for v in obj.values())
            rows = [
                [_cell(obj[k][i]) if i < len(obj[k]) else "" for k in keys]
                for i in range(n)
            ]
            return keys, rows
        # B: {"data": [ {...}, ... ]} —— 取第一个「元素是字典的列表」
        for value in obj.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return _from_list(value)
    return None


def load_table(path: Path):
    """返回 (header, rows)；header 可能是 None。读不出来返回 None。"""
    text = read_text(path)
    suffix = path.suffix.lower()

    if suffix in (".json", ".jsonl"):
        probe = text.lstrip()[:1]
        if probe in ("[", "{"):
            try:
                parsed = _from_json_obj(json.loads(text))
                if parsed and parsed[1]:
                    return parsed
            except json.JSONDecodeError:
                pass
        rows_dicts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows_dicts.append(obj)
        if rows_dicts:
            return _from_list(rows_dicts)

    # 分隔符文本（tsv / csv / 空格分隔）
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    delim = max(["\t", ",", ";"], key=lambda d: lines[0].count(d))
    if delim == "\t" and lines[0].count("\t") == 0:
        delim = "," if lines[0].count(",") >= lines[0].count("\t") else "\t"
    rows = [[c.strip() for c in ln.split(delim)] for ln in lines]

    header = None
    first = rows[0]
    joined = " ".join(first).lower()
    if max(len(c) for c in first) <= 12 and any(
        h in joined for h in TEXT_HINTS + LABEL_HINTS
    ):
        header, rows = first, rows[1:]
    return header, rows


def pick_columns(header, rows):
    """猜哪一列是问句、哪一列是标签。猜不出就报错并要求贴样例。"""
    ncol = len(rows[0])
    if header:
        low = [str(h).lower() for h in header]
        ti = next((i for i, h in enumerate(low) if any(k in h for k in TEXT_HINTS)), None)
        li = next((i for i, h in enumerate(low) if any(k in h for k in LABEL_HINTS)), None)
        if ti is not None and li is not None and ti != li:
            return ti, li

    stats = []
    for i in range(ncol):
        values = [r[i] for r in rows if i < len(r)]
        if not values:
            continue
        avg_len = sum(len(v) for v in values) / len(values)
        stats.append({"col": i, "avg_len": avg_len, "distinct": len(set(values))})

    if len(stats) < 2:
        raise SystemExit(
            "只读到一列，没法区分问句和标签。\n"
            "把这个文件的前 5 行贴给教练看，再决定怎么解析。"
        )

    text_col = max(stats, key=lambda s: s["avg_len"])["col"]
    cands = [s for s in stats if s["col"] != text_col and 2 <= s["distinct"] <= 80]
    if not cands:
        raise SystemExit(
            "猜不出标签列（没有 2~80 个不同值的列）。\n"
            "把这个文件的前 5 行贴给教练看。"
        )
    label_col = min(cands, key=lambda s: s["distinct"])["col"]
    return text_col, label_col


def guess_split(path: Path) -> str | None:
    name = path.name.lower()
    for split, hints in SPLIT_HINTS.items():
        if any(h in name for h in hints):
            return split
    return None


def stratified_split(pool: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """按标签分层切 8:1:1。

    为什么必须分层：随机切会让小类别整类落进 train，test 里一条都没有，
    那一类的 F1 直接算 0，macro-F1 被冤枉。分层能保证每类在每个划分里都有。
    """
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for it in pool:
        by_label.setdefault(it["intent"], []).append(it)

    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for _label, items in sorted(by_label.items()):
        items = items[:]           # 复制，别动原列表
        rng.shuffle(items)
        n = len(items)
        if n >= 3:
            n_dev = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            if n_dev + n_test >= n:      # 太少，保底全给 train
                n_dev = n_test = 0
        else:
            n_dev = n_test = 0
        out["train"] += items[: n - n_dev - n_test]
        out["dev"] += items[n - n_dev - n_test: n - n_test]
        out["test"] += items[n - n_test:]
    for key in out:
        rng.shuffle(out[key])
    return out


# ---------------------------------------------------------------- 主流程

def do_inspect() -> int:
    files = sorted(p for p in config.RAW_DIR.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not files:
        print(f"[空] {config.RAW_DIR} 下面没有可读的数据文件")
        zips = list(config.RAW_DIR.rglob("*.zip"))
        if zips:
            print("发现压缩包，先解压：")
            for z in zips:
                print(f"  {z}")
        return 1
    print(f"共 {len(files)} 个候选文件：\n")
    for p in files:
        print("=" * 78)
        print(f"{p.relative_to(config.RAW_DIR)}  ({p.stat().st_size} 字节)")
        head = read_text(p)[:400].replace("\r", "")
        print("-" * 78)
        print(head)
        print()
    print("看完把上面的文件名列表和前 5 行贴给教练，再决定怎么转。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只看原始文件长什么样")
    ap.add_argument("--limit", type=int, default=0, help="每个划分最多取多少条（冒烟用）")
    ap.add_argument("--dedup", action="store_true", help="删掉跨划分重复的文本")
    ap.add_argument("--pattern", default="*", help="只读文件名匹配这个通配符的原始文件")
    args = ap.parse_args()

    if args.inspect:
        return do_inspect()

    files = sorted(
        p for p in config.RAW_DIR.rglob("*")
        if p.suffix.lower() in SUFFIXES and fnmatch.fnmatch(p.name, args.pattern)
    )
    if not files:
        print(f"[失败] {config.RAW_DIR} 下面没有匹配 {args.pattern} 的数据文件。")
        print(f"       先跑：uv run python data/fetch.py")
        return 1

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": [], "unknown": []}
    seen_text: set[str] = set()
    skipped_dup = 0
    for path in files:
        parsed = load_table(path)
        if not parsed:
            print(f"[跳过] 读不出表格：{path.name}")
            continue
        header, rows = parsed
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            print(f"[跳过] 没有数据行：{path.name}")
            continue
        text_i, label_i = pick_columns(header, rows)
        split = guess_split(path) or "unknown"
        kept = 0
        for r in rows:
            if len(r) <= max(text_i, label_i):
                continue
            text, label = _cell(r[text_i]), _cell(r[label_i])
            if not text or not label:
                continue
            if text in seen_text:      # 多个文件常常互相包含（dev 和 dev_100）
                skipped_dup += 1
                continue
            seen_text.add(text)
            buckets[split].append({"text": text, "intent": label})
            kept += 1
        print(f"[读入] {path.name:<32} split={split:<7} 行数={kept:<6} "
              f"列={header} 问句列={text_i} 标签列={label_i}")
    if skipped_dup:
        print(f"[去重] 多个文件之间有 {skipped_dup} 条重复文本，已只保留第一次出现的")

    # 没有文件名线索的部分，按 8:1:1 切
    if buckets["unknown"]:
        pool = buckets["unknown"]
        print(f"\n[提示] {len(pool)} 条数据没有文件名线索，按 8:1:1 划分")
        n = len(pool)
        n_dev = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        buckets["train"] += pool[: n - n_dev - n_test]
        buckets["dev"] += pool[n - n_dev - n_test: n - n_test]
        buckets["test"] += pool[n - n_test:]

    # 只拿到 dev/test 划分（没有 train）：全部合并后按 8:1:1 分层重切
    if not buckets["train"]:
        pool = buckets["train"] + buckets["dev"] + buckets["test"]
        print(f"\n[提示] 没有 train 划分（公开镜像只放了官方 dev）。"
              f"把 {len(pool)} 条合并后按 8:1:1 **分层**重切；报告里要写明这一点。")
        buckets.update(stratified_split(pool))

    # dev 为空时，从 train 末尾切 10%（只动 train，绝不碰 test 的官方划分）
    if not buckets["dev"] and len(buckets["train"]) > 20:
        n_dev = max(1, int(len(buckets["train"]) * 0.1))
        buckets["dev"] = buckets["train"][-n_dev:]
        buckets["train"] = buckets["train"][:-n_dev]
        print(f"\n[提示] 原始数据没有 dev 划分，从 train 末尾切 {n_dev} 条当 dev"
              f"（test 不动，仍是官方划分）")

    # 跨划分重复检测（数据泄漏最常见的原因）
    def keyset(items: list[dict]) -> set[str]:
        return {it["text"] for it in items}

    train_keys = keyset(buckets["train"])
    dup_dev = keyset(buckets["dev"]) & train_keys
    dup_test = keyset(buckets["test"]) & train_keys
    if dup_dev or dup_test:
        print(f"\n[警告] 跨划分重复：dev {len(dup_dev)} 条、test {len(dup_test)} 条与 train 完全相同")
        if args.dedup:
            buckets["dev"] = [it for it in buckets["dev"] if it["text"] not in train_keys]
            buckets["test"] = [it for it in buckets["test"] if it["text"] not in train_keys]
            print("[已删除] 用 --dedup 去掉了这些重复")
        else:
            print("[建议] 加 --dedup 重跑一遍，否则测试分数会虚高")

    labels = sorted({it["intent"] for s in buckets.values() for it in s})
    if not labels:
        print("[失败] 一条数据都没解析出来，把 --inspect 的输出贴给教练")
        return 1

    # 写文件
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    plan = [("train", config.TRAIN_JSONL), ("dev", config.DEV_JSONL), ("test", config.TEST_JSONL)]
    for split, out in plan:
        items = buckets[split]
        if args.limit:
            items = items[: args.limit]
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for i, it in enumerate(items):
                f.write(json.dumps(
                    {"id": f"{split}-{i:04d}", "text": it["text"],
                     "intent": it["intent"], "split": split},
                    ensure_ascii=False) + "\n")
        print(f"[写出] {out.name:<12} {len(items):>6} 条")

    config.LABELS_FILE.write_text(
        json.dumps({"labels": labels,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "KUAKE-QIC"},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[写出] {config.LABELS_FILE.name:<12} {len(labels)} 个标签：{labels}")

    # 分布表
    print("\n" + "=" * 70)
    print("类别分布（行=意图，列=划分）")
    print("=" * 70)
    header_line = f"{'意图':<18}{'train':>8}{'dev':>8}{'test':>8}{'合计':>8}"
    print(header_line)
    print("-" * len(header_line))
    for lb in labels:
        c = [sum(1 for it in buckets[s] if it["intent"] == lb) for s in ("train", "dev", "test")]
        print(f"{lb[:16]:<18}{c[0]:>8}{c[1]:>8}{c[2]:>8}{sum(c):>8}")
    totals = [len(buckets[s]) for s in ("train", "dev", "test")]
    print("-" * len(header_line))
    print(f"{'合计':<18}{totals[0]:>8}{totals[1]:>8}{totals[2]:>8}{sum(totals):>8}")

    lens = [len(it["text"]) for s in ("train", "dev", "test") for it in buckets[s]]
    print(f"\n问句长度：最短 {min(lens)} 字，最长 {max(lens)} 字，"
          f"平均 {sum(lens) / len(lens):.1f} 字")
    for split in ("train", "dev", "test"):
        if not buckets[split]:
            print(f"[警告] {split} 划分是空的，训练/评测会出问题")
    print(f"标签数：{len(labels)}；训练集最小类：{min(Counter(it['intent'] for it in buckets['train']).values())} 条")
    print("\n下一步：uv run python train/baseline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
