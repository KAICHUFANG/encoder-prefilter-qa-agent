"""把 CrossWOZ 的对话数据转成单轮意图分类数据。

**为什么这份数据适合验证「分流」**：
每个用户话语都带四元组标注 `[动作, 领域, 槽位, 值]`，例如

    ['Request', '餐馆', '营业时间', '']      -> 问餐馆营业时间：库里就有，确定性可答
    ['General', 'greet', 'none', 'none']    -> 打招呼：模板

领域（餐馆/酒店/景点/地铁/出租）每一个都对应一份真实的属性库 ——
**这就是「分到某个意图之后，真的有一个执行者能接住」**，而不是全丢给大模型。

标签设计（7 类）：
    餐馆 / 酒店 / 景点 / 地铁 / 出租 / General / 多域
    单域 -> 查该域数据库（确定性执行）
    General -> 模板话术
    多域（一句话涉及 >=2 个领域）-> 单个分类器搞不定，天然就是「交给大模型拆解」的样本

用法：
    uv run python train/prepare_crosswoz.py
    uv run python train/prepare_crosswoz.py --max-train 20000 --max-dev 3000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

MULTI = "多域"
GENERAL = "General"
# 领域名 -> 中文标签（保持原样，数据里就是中文）
SKIP_DOMAINS = {"none", "", "通用", "General"}


# ---------------------------------------------------------------- 读取

def iter_dialogues(path: Path):
    """逐个产出 (dialog_id, dialogue)。

    train.json 有 390MB，直接 json.load 要吃好几 GB 内存，
    所以优先用 ijson 流式读；没装 ijson 才退回全量加载。
    """
    if not path.exists():
        raise SystemExit(f"找不到 {path}\n先下载 CrossWOZ（见 data/README.md）")

    try:
        import ijson

        with path.open("rb") as f:
            for key, dialogue in ijson.kvitems(f, ""):
                if isinstance(dialogue, dict):
                    yield str(key), dialogue
        return
    except ImportError:
        print(f"[提示] 没装 ijson，退回归全量加载（{path.name} 很大，注意内存）")
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, dialogue in data.items():
            if isinstance(dialogue, dict):
                yield str(key), dialogue


def label_of(dialog_acts: list) -> str | None:
    """从 dialog_act 列表里推出这一轮的标签。"""
    domains: set[str] = set()
    for item in dialog_acts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        act, domain = str(item[0]), str(item[1])
        if act == "General" or domain in SKIP_DOMAINS:
            continue
        if "-" in domain:                     # 有些标注写成 "餐馆-营业时间"
            domain = domain.split("-")[0]
        domains.add(domain)

    if len(domains) >= 2:
        return MULTI
    if len(domains) == 1:
        return next(iter(domains))
    return GENERAL


def build_split(paths_and_splits, caps: dict[str, int]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for path, split in paths_and_splits:
        cap = caps.get(split, 0)
        n_dlg = 0
        skipped_short = 0
        for dialog_id, dialogue in iter_dialogues(path):
            for i, msg in enumerate(dialogue.get("messages") or []):
                if msg.get("role") not in ("usr", "user"):
                    continue
                text = str(msg.get("content") or "").strip()
                if len(text) < 2:
                    skipped_short += 1
                    continue
                acts = msg.get("dialog_act") or []
                label = label_of(acts)
                if label is None:
                    continue
                act_type = ""
                if acts and isinstance(acts[0], (list, tuple)):
                    act_type = str(acts[0][0])
                out[split].append({
                    "text": text,
                    "intent": label,
                    "act": act_type,
                    "dialog_id": dialog_id,
                    "turn": i,
                    "split": split,
                })
            n_dlg += 1
            if cap and len(out[split]) >= cap:
                print(f"[截断] {path.name}: 到 {cap} 条就停（覆盖 {n_dlg} 个对话）")
                break
        print(f"[读入] {path.name:<12} split={split:<6} 对话 {n_dlg:<6} "
              f"话语 {len(out[split]):<7} 过短跳过 {skipped_short}")
    return out


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=20000, help="0 表示不限制")
    ap.add_argument("--max-dev", type=int, default=3000)
    ap.add_argument("--max-test", type=int, default=0, help="0 表示用全部测试集")
    args = ap.parse_args()

    raw = config.RAW_DIR
    candidates = {
        "train": [raw / "extracted" / "train.json", raw / "train.json"],
        "dev": [raw / "extracted" / "val.json", raw / "val.json", raw / "extracted" / "dev.json"],
        "test": [raw / "extracted" / "test.json", raw / "test.json"],
    }
    paths: list[tuple[Path, str]] = []
    for split, options in candidates.items():
        found = next((p for p in options if p.exists()), None)
        if found is None:
            print(f"[缺失] {split} 找不到，试过：{[str(p) for p in options]}")
            return 1
        paths.append((found, split))

    caps = {"train": args.max_train, "dev": args.max_dev, "test": args.max_test}
    buckets = build_split(paths, caps)

    # 训练集打散（原顺序是按对话来的，同类会聚集）
    random.Random(config.SEED).shuffle(buckets["train"])
    random.Random(config.SEED + 1).shuffle(buckets["dev"])
    random.Random(config.SEED + 2).shuffle(buckets["test"])

    # 跨划分去重（同一句话在 train/test 都出现会让分数虚高）
    seen: set[str] = set(buckets["train"] and [r["text"] for r in buckets["train"]] or [])
    for split in ("dev", "test"):
        before = len(buckets[split])
        buckets[split] = [r for r in buckets[split] if r["text"] not in seen]
        if before != len(buckets[split]):
            print(f"[去重] {split}: 去掉 {before - len(buckets[split])} 条与 train 重复的样本")

    labels = sorted({r["intent"] for s in buckets.values() for r in s})
    if not labels:
        print("[失败] 一条都没解析出来")
        return 1

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    plan = [("train", config.TRAIN_JSONL), ("dev", config.DEV_JSONL), ("test", config.TEST_JSONL)]
    for split, out_path in plan:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(buckets[split]):
                f.write(json.dumps({
                    "id": f"{split}-{i:05d}",
                    "text": r["text"],
                    "intent": r["intent"],
                    "act": r["act"],
                    "dialog_id": r["dialog_id"],
                    "split": split,
                }, ensure_ascii=False) + "\n")
        print(f"[写出] {out_path.name:<12} {len(buckets[split]):>6} 条  -> {out_path}")

    config.LABELS_FILE.write_text(
        json.dumps({"labels": labels,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "CrossWOZ (Apache-2.0)"},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[写出] labels.json   {len(labels)} 个标签：{labels}")

    # 分布表
    print("\n" + "=" * 72)
    print("标签分布（行=标签，列=划分）")
    print("=" * 72)
    header = f"{'标签':<12}{'train':>9}{'dev':>9}{'test':>9}{'合计':>9}"
    print(header)
    print("-" * len(header))
    for lb in labels:
        c = [sum(1 for r in buckets[s] if r["intent"] == lb) for s in ("train", "dev", "test")]
        print(f"{lb:<12}{c[0]:>9}{c[1]:>9}{c[2]:>9}{sum(c):>9}")
    totals = [len(buckets[s]) for s in ("train", "dev", "test")]
    print("-" * len(header))
    print(f"{'合计':<12}{totals[0]:>9}{totals[1]:>9}{totals[2]:>9}{sum(totals):>9}")

    multi_ratio = sum(1 for s in buckets.values() for r in s if r["intent"] == MULTI) / max(1, sum(totals))
    print(f"\n多域样本占比：{multi_ratio * 100:.1f}%  （这批就是「单个分类器搞不定、要交给大模型拆解」的样本）")
    lens = [len(r["text"]) for s in buckets.values() for r in s]
    print(f"问句长度：最短 {min(lens)}，最长 {max(lens)}，平均 {sum(lens) / len(lens):.1f} 字")
    acts = Counter(r["act"] for s in buckets.values() for r in s)
    print("对话动作分布（前 8）：" + "  ".join(f"{k}={v}" for k, v in acts.most_common(8)))
    print("\n下一步：uv run python train/baseline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
