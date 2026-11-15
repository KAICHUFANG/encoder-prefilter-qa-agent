"""两个基线，报告里必须都出现。

    1) 多数类基线：永远猜训练集里最多的那一类 —— 这是「下限」，
       任何模型都必须明显超过它，否则说明模型什么也没学到。
    2) TF-IDF + 逻辑回归：传统文本分类的强基线 —— 微调必须赢过它，
       赢不过就说明「上大模型」在这个任务上是不必要的。

理由：没有基线的「微调后 91%」等于什么都没证明。

用法：
    uv run python train/baseline.py
    uv run python train/baseline.py --limit 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import metrics as metrics_mod  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"缺少 {path}，先跑：uv run python train/prepare.py")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train = load_jsonl(config.TRAIN_JSONL)
    test = load_jsonl(config.TEST_JSONL)
    if args.limit:
        train, test = train[: args.limit], test[: args.limit]

    labels = json.loads(config.LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    x_train = [r["text"] for r in train]
    y_train = [r["intent"] for r in train]
    x_test = [r["text"] for r in test]
    y_test = [r["intent"] for r in test]

    print(f"train {len(train)} 条 / test {len(test)} 条 / {len(labels)} 类")
    print(f"{config.describe()}\n")

    results = {}

    # ---------- 基线 1：多数类 ----------
    major = Counter(y_train).most_common(1)[0][0]
    y_pred_major = [major] * len(y_test)
    r1 = metrics_mod.compute(y_test, y_pred_major, labels)
    results["majority"] = {
        "name": "多数类基线",
        "predict": major,
        **{k: r1[k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class_f1")},
    }
    print("=" * 70)
    print(f"[基线1] 多数类（永远猜「{major}」）")
    print(f"  准确率 {r1['accuracy']:.4f} | macro-F1 {r1['macro_f1']:.4f}")

    # ---------- 基线 2：TF-IDF + 逻辑回归 ----------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    t0 = time.perf_counter()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                  min_df=2, max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(x_train, y_train)
    y_pred_lr = list(pipe.predict(x_test))
    fit_seconds = round(time.perf_counter() - t0, 1)
    r2 = metrics_mod.compute(y_test, y_pred_lr, labels)
    results["tfidf_logreg"] = {
        "name": "TF-IDF + 逻辑回归",
        "fit_seconds": fit_seconds,
        **{k: r2[k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class_f1")},
    }
    print("=" * 70)
    print(f"[基线2] TF-IDF(char 1-3) + 逻辑回归   训练耗时 {fit_seconds}s")
    print(metrics_mod.format_summary(r2))
    print("\n混淆矩阵：")
    print(metrics_mod.format_confusion(r2))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.BASELINE_JSON.write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "train_size": len(train), "test_size": len(test),
                    "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n基线结果已写入 {config.BASELINE_JSON}")
    print(f"微调时必须同时超过：准确率 {max(r1['accuracy'], r2['accuracy']):.4f}、"
          f"macro-F1 {max(r1['macro_f1'], r2['macro_f1']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
