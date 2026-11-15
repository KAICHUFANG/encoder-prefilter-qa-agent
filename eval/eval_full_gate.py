"""全量数据集（2,469/534/519）上的集成门控评估。

流程（避免偷看测试集）：
    1. dev 上扫阈值（4/4 一致 + min P ≥ t），选工作点
    2. test 上只跑一次，报告覆盖率 / 路由准确率 / 错给代码率

用法：
    $env:INTENT_DATASET = "crosswoz_gate_full"
    python eval/eval_full_gate.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

DATA = PROJ / "data" / "crosswoz_gate_full"
ENS = PROJ / "models" / "_ensemble_full"
SEEDS = (7, 13, 21, 33)
OUT = PROJ / "eval" / "ensemble_routing_full.jsonl"


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def load(split: str) -> tuple[list[dict], list[list[float]]]:
    from app.model import IntentPredictor

    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").open(encoding="utf-8") if line.strip()]
    texts = [r["text"] for r in rows]
    p_codes = []
    for seed in SEEDS:
        p = IntentPredictor(model_dir=ENS / f"seed{seed}")
        p.warmup()
        logits = p.logits_batch(texts)
        p_codes.append([softmax(z)[0] for z in logits])
    # 转成每行一个列表
    per_item = [[p_codes[k][i] for k in range(len(SEEDS))] for i in range(len(rows))]
    return rows, per_item


def rule(probs: list[float], threshold: float) -> bool:
    votes = [0 if v >= 0.5 else 1 for v in probs]
    return all(v == 0 for v in votes) and min(probs) >= threshold


def stats(rows: list[dict], probs: list[list[float]], threshold: float) -> dict:
    routed, ok = 0, 0
    for r, ps in zip(rows, probs):
        if rule(ps, threshold):
            routed += 1
            ok += (r["intent"] == 0)
    n = len(rows)
    return {"n": n, "routed": routed, "ok": ok,
            "coverage": routed / n, "precision": ok / max(1, routed),
            "err_rate": (routed - ok) / n,
            "waste": sum(1 for r, ps in zip(rows, probs)
                         if not rule(ps, threshold) and r["intent"] == 0) / n}


def main() -> int:
    print("加载 4 个全量模型...")
    dev_rows, dev_p = load("dev")
    test_rows, test_p = load("test")
    print(f"dev {len(dev_rows)} | test {len(test_rows)}")

    print(f"\n{'阈值':>6} {'dev覆盖':>8} {'dev准确':>8} {'dev错给':>8}")
    best = None
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        s = stats(dev_rows, dev_p, t)
        print(f"{t:>6.2f} {s['coverage']:>7.0%} {s['precision']:>8.0%} {s['err_rate']:>7.1%}")
        if best is None or (s["precision"] >= 0.85 and s["coverage"] > best[1]["coverage"]):
            if s["precision"] >= 0.85:
                best = (t, s)

    threshold = best[0] if best else 0.90
    print(f"\n选定阈值：{threshold}（dev 上路由准确率首次达到 85% 的工作点）")

    s_dev = stats(dev_rows, dev_p, threshold)
    s_test = stats(test_rows, test_p, threshold)
    print(f"\ndev  : 覆盖 {s_dev['coverage']:.0%} | 路由准确率 {s_dev['precision']:.1%} | 错给代码 {s_dev['err_rate']:.1%}")
    print(f"test : 覆盖 {s_test['coverage']:.0%} | 路由准确率 {s_test['precision']:.1%} | 错给代码 {s_test['err_rate']:.1%}")

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for split, rows, ps in (("dev", dev_rows, dev_p), ("test", test_rows, test_p)):
            for r, p in zip(rows, ps):
                f.write(json.dumps({
                    "id": r["id"], "split": split, "label": r["intent"],
                    "route": "code" if rule(p, threshold) else "llm",
                    "min_p": min(p), "votes": [0 if v >= 0.5 else 1 for v in p],
                }, ensure_ascii=False) + "\n")
    print(f"\n逐条结果写出 {OUT}")
    print(json.dumps({"threshold": threshold, "dev": s_dev, "test": s_test},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
