"""D 方案：把「本来要喂给大模型的材料」也喂给 encoder-only。

与前面几版的区别：
    A 只给最后一问
    B 给上下文 + 问题
    C 给上下文 + 问题 + 代码最终答案
    D 给上下文 + 问题 + **检索材料**（周边关系表 / 约束筛选结果 / 向量候选）
      —— 也就是大模型实际看到的那份输入，只是不包含代码的结论

标签不变：DeepSeek 依据「问题 + 代码回答」判的「该不该交大模型」。

输出：data/crosswoz_gate_mat/{train,dev,test}.jsonl

用法：
    python data/build_gate_material.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

sys.path.insert(0, str(PROJ / "eval"))
import run_llm_eval as R  # noqa: E402  # 复用它的三段式检索

SRC = PROJ / "data" / "crosswoz_gate_llm"
DST = PROJ / "data" / "crosswoz_gate_mat"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"
TURNS = 4


def main() -> int:
    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    DST.mkdir(parents=True, exist_ok=True)

    lengths = []
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
                        content = str(msg.get("content", "")).strip()[:36]
                        if content:
                            ctx_lines.append(f"{role}：{content}")

            # 紧凑版材料：每个区块只保留最关键的 2 条，总长控制在 400 字内
            parts = []
            rel = R.relation_block(r["text"], R.referent_names(r["text"], ctx_lines))
            if rel:
                parts.append("周边：" + rel.replace("\n", " ")[:170])
            cons = R.constraint_block(r["text"])
            if cons:
                parts.append("筛选：" + cons.replace("\n", " ")[:170])
            vec_text, _vrefs = R.retrieve_vector(r["text"], ctx_lines, top_k=2)
            if vec_text:
                parts.append("候选：" + vec_text.replace("\n", " ")[:120])
            material = "｜".join(parts) if parts else "（无）"
            ctx_short = " ".join(x[:26] for x in ctx_lines[-2:])
            text = (f"上下文：{ctx_short}\n" if ctx_short else "")
            text += f"问题：{r['text']}\n材料：{material[:460]}"
            lengths.append(len(text))
            out.append({"id": r["id"], "text": text, "query": r["text"],
                        "intent": r["intent"], "gate": r["gate"], "split": split})
        (DST / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        print(f"{split}: {len(out)} 条")

    lengths.sort()
    print(f"\n输入长度：中位数 {lengths[len(lengths)//2]} 字，95 分位 {lengths[int(len(lengths)*0.95)]} 字，最大 {lengths[-1]} 字")
    (DST / "labels.json").write_text(json.dumps(
        {"labels": [0, 1], "note": "0=代码答案够用 1=不够用；标签来自 DeepSeek，输入含检索材料"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
