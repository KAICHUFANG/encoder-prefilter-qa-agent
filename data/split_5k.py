"""把 5,000 条按对话切分：train/dev/test = 70/15/15。

用法：
    python data/split_5k.py
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

SRC = PROJ / "data" / "crosswoz_5k" / "_progress.jsonl"
OUT = PROJ / "data" / "crosswoz_5k"


def main() -> int:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("intent") in (0, 1)]
    print(f"可用标注 {len(rows)} 条")

    by_dialog: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dialog[str(r["dialog_id"])].append(r)
    dialogs = sorted(by_dialog)
    random.Random(config.SEED).shuffle(dialogs)

    total = len(rows)
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for dlg in dialogs:
        if len(buckets["train"]) < total * 0.70:
            buckets["train"] += by_dialog[dlg]
        elif len(buckets["dev"]) < total * 0.15:
            buckets["dev"] += by_dialog[dlg]
        else:
            buckets["test"] += by_dialog[dlg]

    for split, items in buckets.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in items:
                f.write(json.dumps({"id": r["id"], "text": r["text"],
                                    "intent": r["intent"], "split": split,
                                    "gate": "code" if r["intent"] == 0 else "llm",
                                    "reason": r.get("reason", "")},
                                   ensure_ascii=False) + "\n")
        c = Counter(r["intent"] for r in items)
        print(f"{split:<6}{len(items):>5} 条  代码答对 {c[0]:>4} / 需交模型 {c[1]:>4}")

    (OUT / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码答对（交代码） 1=需交大模型；与 crosswoz_gate_full 同一标注方式"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps({
        "total": total, "dialogs": len(dialogs),
        "splits": {k: len(v) for k, v in buckets.items()},
        "code_rate": sum(1 for r in rows if r["intent"] == 0) / total,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n对话 {len(dialogs)} 段，代码答对率 {sum(1 for r in rows if r['intent']==0)/total:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
