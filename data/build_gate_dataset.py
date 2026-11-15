"""把 400 条人工门控标签转成训练格式（二分类）。

标签定义（用户定义）：
    encoder 只回答一个问题：这句话该不该交给大模型？
        gate = code / template  -> needs_llm = 0   （代码或模板能推进，0 token）
        gate = llm              -> needs_llm = 1   （只有模型能推进）

切分：
    **按 dialog_id 切**，不按句子随机切。
    同一段对话里的前后句高度相关（"它家电话多少" 依赖上文），
    随机切会让测试集混入训练对话，指标虚高。

用法：
    python data/build_gate_dataset.py
产出：
    data/crosswoz_gate/{train,dev,test}.jsonl + labels.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

BASE = PROJ / "data" / "crosswoz"
BATCHES = ["human_labels_batch1_v2.jsonl", "human_gate_batch2.jsonl",
           "human_gate_batch3.jsonl", "human_gate_batch4.jsonl",
           "human_gate_batch5.jsonl", "human_gate_batch6.jsonl",
           "human_gate_batch7.jsonl", "human_gate_batch8.jsonl",
           "human_gate_batch9.jsonl", "human_gate_batch10.jsonl"]


def main() -> int:
    labels: dict[int, dict] = {}
    for name in BATCHES:
        p = BASE / name
        if not p.exists():
            print(f"[跳过] 缺 {name}")
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                labels[r["id"]] = r

    rows = [json.loads(line) for line in (BASE / "test.jsonl").open(encoding="utf-8") if line.strip()]
    data = []
    for i, r in enumerate(rows):
        if i not in labels:
            continue
        gate = labels[i]["label"]
        gate = {"ok": "code", "chitchat": "template"}.get(gate, gate)
        needs_llm = 0 if gate in ("code", "template") else 1
        data.append({"id": f"gate-{i:04d}", "text": r["text"], "intent": needs_llm,
                     "gate": gate, "dialog_id": r["dialog_id"],
                     "reason": labels[i]["reason"]})

    # 按 dialog_id 切分，避免同段对话跨划分
    by_dialog: dict[str, list[dict]] = defaultdict(list)
    for d in data:
        by_dialog[d["dialog_id"]].append(d)

    rng = random.Random(config.SEED)
    dialogs = sorted(by_dialog)
    rng.shuffle(dialogs)
    n = len(data)
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for dlg in dialogs:
        if len(buckets["train"]) < n * 0.7:
            buckets["train"] += by_dialog[dlg]
        elif len(buckets["dev"]) < n * 0.15:
            buckets["dev"] += by_dialog[dlg]
        else:
            buckets["test"] += by_dialog[dlg]

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"train": config.TRAIN_JSONL, "dev": config.DEV_JSONL, "test": config.TEST_JSONL}
    for split, path in paths.items():
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for d in buckets[split]:
                f.write(json.dumps({"id": d["id"], "text": d["text"], "intent": d["intent"],
                                    "split": split, "gate": d["gate"]},
                                   ensure_ascii=False) + "\n")
        c = Counter(d["intent"] for d in buckets[split])
        print(f"[写出] {split:<6}{len(buckets[split]):>4} 条  needs_llm=0:{c[0]:>3} =1:{c[1]:>3}")

    config.LABELS_FILE.write_text(
        json.dumps({"labels": [0, 1], "note": "0=代码/模板可推进 1=必须交大模型"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n合计 {len(data)} 条人工标签，对话 {len(dialogs)} 段")
    print(f"正样本（该交大模型）占比：{sum(d['intent'] for d in data)/len(data):.1%}")
    print(f"朴素基线：全判 0 -> {sum(1 for d in data if d['intent']==0)/len(data):.1%}；"
          f"全判 1 -> {sum(d['intent'] for d in data)/len(data):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
