"""离线评测：加载微调后的模型，在 test.jsonl 上算全套指标。

和 eval/run_eval.py 的分工：
    train/evaluate.py   直接调模型（快，用来比模型）
    eval/run_eval.py    走 HTTP 接口（慢一点，用来证明接口层没问题）

用法：
    uv run python train/evaluate.py
    uv run python train/evaluate.py --limit 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import metrics as metrics_mod  # noqa: E402
from app.model import IntentPredictor  # noqa: E402


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

    test = load_jsonl(config.TEST_JSONL)
    if args.limit:
        test = test[: args.limit]

    labels = json.loads(config.LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    texts = [r["text"] for r in test]
    y_true = [r["intent"] for r in test]

    # 拿训练时记下的真实底座模型名（不然表里会写死成 config 里的默认值）
    meta_path = config.MODELS_DIR / "train_meta.json"
    train_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model_label = train_meta.get("base_model", config.MODEL_NAME)

    problems = config.check_model()
    if problems:
        for p in problems:
            print(f"[缺少] {p}")
        return 1

    print(f"模型 {config.MODEL_OUT}")
    print(f"测试集 {len(test)} 条，{len(labels)} 类\n")

    predictor = IntentPredictor()
    predictor.warmup()          # 先把模型读进内存，别把加载时间算进延迟
    t0 = time.perf_counter()
    y_pred = predictor.predict_batch(texts)
    seconds = round(time.perf_counter() - t0, 1)
    per_item_ms = round(seconds * 1000 / max(1, len(texts)), 2)

    result = metrics_mod.compute(y_true, y_pred, labels)
    print(metrics_mod.format_summary(result))
    print("\n混淆矩阵：")
    print(metrics_mod.format_confusion(result))
    print(f"\n推理耗时：{seconds}s，平均 {per_item_ms} ms/条（设备 {predictor._device}）")

    # 和基线对比（有就比，没有就提示）
    baseline = None
    if config.BASELINE_JSON.exists():
        baseline = json.loads(config.BASELINE_JSON.read_text(encoding="utf-8"))
        base = baseline["results"]["tfidf_logreg"]
        print("\n" + "=" * 70)
        print("与基线对比")
        print(f"{'方法':<22}{'准确率':>10}{'macro-F1':>12}")
        print("-" * 44)
        print(f"{'多数类':<22}{baseline['results']['majority']['accuracy']:>10.4f}"
              f"{baseline['results']['majority']['macro_f1']:>12.4f}")
        print(f"{'TF-IDF + 逻辑回归':<22}{base['accuracy']:>10.4f}{base['macro_f1']:>12.4f}")
        print(f"{('微调 ' + model_label)[:21]:<22}{result['accuracy']:>10.4f}"
              f"{result['macro_f1']:>12.4f}")
        delta = result["macro_f1"] - base["macro_f1"]
        verdict = "赢过基线" if delta > 0 else "没赢过基线，要解释为什么"
        print(f"\nmacro-F1 相对 TF-IDF 基线 {delta:+.4f} -> {verdict}")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "offline",
        "test_size": len(test),
        "model_dir": str(config.MODEL_OUT),
        "train_meta": train_meta,
        "latency_ms_per_item": per_item_ms,
        "metrics": result,
        "baseline": baseline["results"] if baseline else None,
    }
    config.REPORT_OFFLINE_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n报告已写入 {config.REPORT_OFFLINE_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
