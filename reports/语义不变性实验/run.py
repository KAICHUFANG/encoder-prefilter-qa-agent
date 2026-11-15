"""语义不变性实验：同一意思、不同句法，模型给不给同一个答案。

运行：
    python reports/语义不变性实验/run.py
会依次跑三组对照，结果写到同目录 results.json：
    crosswoz + rbt3(3层)   /  crosswoz + MacBERT(12层)  /  massive + MacBERT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

MAX_LEN = 64

RUNS = [
    {"dataset": "crosswoz", "model_dir": "models/crosswoz/intent-bert-rbt3",
     "label": "CrossWOZ + rbt3（3 层 / 38M）"},
    {"dataset": "crosswoz", "model_dir": "models/crosswoz/intent-bert",
     "label": "CrossWOZ + MacBERT（12 层 / 102M）"},
    {"dataset": "massive", "model_dir": "models/massive/intent-bert",
     "label": "MASSIVE + MacBERT（12 层 / 102M）"},
]


def predict_all(model_dir: Path, texts: list[str]) -> list[tuple[str, float]]:
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True).eval()
    id2label = model.config.id2label
    out: list[tuple[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i: i + 32], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            for c, j in zip(conf.tolist(), idx.tolist()):
                out.append((id2label[j], round(c, 4)))
    return out


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    results = []

    for run in RUNS:
        model_dir = PROJ / run["model_dir"]
        if not model_dir.exists():
            print(f"[跳过] {run['label']}：{model_dir} 不存在")
            continue

        groups = cases[run["dataset"]]["groups"]
        texts = [t for group in groups.values() for t in group]
        preds = predict_all(model_dir, texts)

        cursor = 0
        detail = {}
        consistent = 0
        for name, group in groups.items():
            chunk = preds[cursor: cursor + len(group)]
            cursor += len(group)
            labels = [p[0] for p in chunk]
            same = len(set(labels)) == 1
            consistent += same
            detail[name] = {"predictions": labels, "confidences": [p[1] for p in chunk],
                            "consistent": same}

        n = len(groups)
        row = {"label": run["label"], "dataset": run["dataset"],
               "model_dir": run["model_dir"],
               "groups": n, "consistent_groups": consistent,
               "invariance": round(consistent / n, 4), "detail": detail}
        results.append(row)
        print(f"{run['label']}: 一致 {consistent}/{n} = {consistent / n:.0%}")

    # 汇总
    summary = {
        "metric": "syntax_invariance",
        "definition": "同一意思的 5 种不同说法中，预测完全一致的组所占比例",
        "runs": [{k: v for k, v in r.items() if k != "detail"} for r in results],
        "detail": {r["label"]: r["detail"] for r in results},
    }
    (HERE / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已写入 {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
