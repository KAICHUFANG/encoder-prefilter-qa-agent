"""证据感知门控：输入 = 上下文 + 问题 + 代码给出的答案，由模型判断"够不够"。

为什么要这么设计：
    · 纯文本门控 61.7%——它看不到代码到底答了什么
    · 带上下文门控 61.7%——上下文只解决"指代谁"，解决不了"这个答案对不对"
    · 而"答案对不对"正是要判的东西，所以必须把答案本身作为输入

注意：判断仍由**训练出来的语言模型**做，代码只负责产出答案草稿，不参与评判。

标签：沿用 crosswoz_gate_llm（DeepSeek 依据问题+代码回答判定）。

输出：data/crosswoz_gate_ev/{train,dev,test}.jsonl
     {"id","text"(上下文+问题+代码答案),"query","code_answer","intent","gate","split"}

用法：
    python data/build_gate_evidence.py
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
PROGRESS = SRC / "_progress.jsonl"
DST = PROJ / "data" / "crosswoz_gate_ev"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"
TURNS = 4


def main() -> int:
    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    code_answers = {}
    if PROGRESS.exists():
        for line in PROGRESS.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                code_answers[r["id"]] = r.get("code_answer", "")

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
            answer = str(code_answers.get(r["id"], "")).strip()[:260]
            text = (("\n".join(ctx_lines) + "\n") if ctx_lines else "")
            text += f"【用户问题】{r['text']}\n【代码给出的答案】{answer or '（无）'}"
            out.append({"id": r["id"], "text": text, "query": r["text"],
                        "code_answer": answer, "intent": r["intent"],
                        "gate": r["gate"], "split": split})
        (DST / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        print(f"{split}: {len(out)} 条")

    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码答案够用 1=代码答案不够用（标签来自 DeepSeek 判定）"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
