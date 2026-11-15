"""给门控数据补上对话上下文——"这家/它"不带上文根本没法判断。

现状问题：
    原门控只喂最后一句（MAX_LEN=64），
    但 CrossWOZ 里大量请求是"这家酒店周边有什么景点"这种指代句，
    没有上下文时，模型既不知道指哪家，也学不到任何语境信号。

做法：
    从原始 test.json 里取该句之前的若干轮（默认 4 轮），拼成：
        客服：… / 用户：… / 客服：… /【当前】用户：<最后一问>
    标签沿用 DeepSeek 按「问题+代码回答」判出的 intent。

输出：data/crosswoz_gate_ctx/{train,dev,test}.jsonl
    {"id","text"(含上下文),"query","intent","gate","split"}

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python data/build_gate_ctx.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

SRC = PROJ / "data" / "crosswoz_gate_llm"
DST = PROJ / "data" / "crosswoz_gate_ctx"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"
TURNS = 4


def main() -> int:
    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    DST.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        src = SRC / f"{split}.jsonl"
        if not src.exists():
            continue
        out = []
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            m = re.search(r"(\d+)$", r["id"])
            idx = int(m.group(1)) if m else -1
            ctx_lines = []
            if 0 <= idx < len(all_items):
                dlg = all_items[idx]["dialog_id"]
                msgs = raw.get(str(dlg), {}).get("messages", [])
                pos = next((i for i, x in enumerate(msgs)
                            if str(x.get("content", "")).strip() == r["text"].strip()), None)
                if pos is not None:
                    for msg in msgs[max(0, pos - TURNS * 2):pos]:
                        role = "用户" if msg.get("role") == "usr" else "客服"
                        content = str(msg.get("content", "")).strip()[:40]
                        if content:
                            ctx_lines.append(f"{role}：{content}")
            text = ("\n".join(ctx_lines) + "\n" if ctx_lines else "") + f"【当前】用户：{r['text']}"
            out.append({"id": r["id"], "text": text, "query": r["text"],
                        "intent": r["intent"], "gate": r["gate"], "split": split})
        (DST / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        with_ctx = sum(1 for x in out if "\n" in x["text"])
        print(f"{split}: {len(out)} 条，其中带上下文 {with_ctx} 条")

    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码已答对（交代码） 1=代码没答对（交大模型）；标签来自 crosswoz_gate_llm"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
