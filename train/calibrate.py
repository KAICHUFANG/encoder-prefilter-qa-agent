"""置信度校准：让「0.99」真的意味着「99% 对」。

问题：模型在不认识的输入上会给 0.999 的错误答案（miscalibration，校准不良）。
      而分流系统的兜底逻辑依赖置信度——置信度不可信，护栏就是纸做的。

做法（temperature scaling，温度缩放）：
    logits' = logits / T ，T 在验证集上拟合，使负对数似然最小。
    只有一个参数，不会改变预测结果（argmax 不变），只改变置信度的含义。

为什么只在验证集上拟合：测试集要留给最终评测；用测试集拟合温度=偷看答案。

用法：
    uv run python train/calibrate.py
    uv run python train/calibrate.py --split dev      # 默认
产出：
    models/<dataset>/calibration.json   {"temperature": 1.83, "ece_before": ..., "ece_after": ...}
    app/model.py 读到它就会自动除温度
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"缺少 {path}，先跑 prepare")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    probs = softmax(logits / temperature)
    picked = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error（期望校准误差）：把置信度分箱，看每箱的「实际准确率 vs 平均置信度」。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc_hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece_value = 0.0
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(acc_hit[mask].mean())
        ece_value += n / total * abs(avg_acc - avg_conf)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n,
                      "avg_conf": round(avg_conf, 3), "avg_acc": round(avg_acc, 3),
                      "gap": round(avg_acc - avg_conf, 3)})
    return round(ece_value, 4), table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    args = ap.parse_args()

    from app.model import IntentPredictor
    from app import labels as label_mod

    path = {"train": config.TRAIN_JSONL, "dev": config.DEV_JSONL, "test": config.TEST_JSONL}[args.split]
    rows = load_jsonl(path)
    label2id = label_mod.label2id()
    texts = [r["text"] for r in rows]
    y = np.array([label2id[r["intent"]] for r in rows])

    predictor = IntentPredictor()
    predictor.warmup()
    print(f"数据集 {config.DATASET} | 拟合用 {args.split}（{len(rows)} 条）")
    logits = np.array(predictor.logits_batch(texts))

    # 拟合温度：在 log 空间里搜，保证 T > 0
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda t: nll(logits, y, np.exp(t)),
                          bounds=(-2.0, 3.0), method="bounded")
    temperature = float(np.exp(res.x))

    probs_before = softmax(logits)
    probs_after = softmax(logits / temperature)
    ece_before, _ = ece(probs_before, y)
    ece_after, table = ece(probs_after, y)
    acc = float((probs_before.argmax(axis=1) == y).mean())

    print(f"  当前准确率      {acc:.4f}（温度缩放不改变预测，只改置信度）")
    print(f"  平均置信度      {probs_before.max(axis=1).mean():.4f} -> {probs_after.max(axis=1).mean():.4f}")
    print(f"  ECE（越小越好）  {ece_before:.4f} -> {ece_after:.4f}")
    print(f"  拟合出的温度 T  {temperature:.3f}")

    print("\n校准后的可靠性表（gap 越接近 0 越好）：")
    print(f"  {'置信区间':<12}{'条数':>6}{'平均置信':>10}{'实际准确':>10}{'gap':>8}")
    for row in table:
        print(f"  {row['bin']:<12}{row['n']:>6}{row['avg_conf']:>10.3f}"
              f"{row['avg_acc']:>10.3f}{row['gap']:>+8.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MODELS_DIR / "calibration.json"
    out.write_text(json.dumps({
        "temperature": round(temperature, 4),
        "fitted_on": args.split,
        "n": len(rows),
        "accuracy": round(acc, 4),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability": table,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out}")
    print("app/model.py 读到它会自动把 logits 除以温度。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
