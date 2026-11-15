"""门控的真实指标：覆盖率 × 代码侧正确率。

为什么不用分类准确率：
    "准确率 83.3%" 混了两件完全不同的事——
        判对的"该交模型"（省不了 token，也不影响正确性）
        判对的"该交代码"（这才省钱，但判错会答错）
    业务上只关心两个数：
        ① 覆盖率：多少请求被交给代码（决定省多少 token）
        ② 代码侧正确率：交给代码的那些里，真正答对了多少（决定错多少）

金标来自 400 条人工标签：
    gate == "code"     代码能正确作答
    gate == "template" 闲聊，走固定话术（也不该给大模型）
    gate == "llm"      代码答错/答不了，必须交模型

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/run_gate_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402


def load(split_path: Path) -> list[dict]:
    return [json.loads(line) for line in split_path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    for split_name, path in (("test", config.TEST_JSONL), ("dev", config.DEV_JSONL)):
        rows = load(path)
        if not rows:
            continue
        # 每条的代码能力（金标）
        gold_code = [1 if r.get("gate") in ("code", "template") else 0 for r in rows]
        n = len(rows)
        base_code_share = sum(gold_code) / n

        preds, confs = [], []
        for r in rows:
            out = p.predict(r["text"], top_k=1)
            preds.append(0 if int(out["intent"]) == 0 else 1)   # 0=交代码 1=交模型
            confs.append(out["confidence"])

        print(f"\n{'='*58}\n{split_name} 集 {n} 条（代码实际能处理 {base_code_share:.1%}）\n{'='*58}")

        # ---------- 三个策略 ----------
        def eval_strategy(name: str, route_code: list[bool]) -> dict:
            covered = sum(route_code)
            correct = sum(1 for i, rc in enumerate(route_code) if rc and gold_code[i])
            wrong = covered - correct
            # 交模型的假设 100% 答对（本次不测大模型）
            overall = (correct + (n - covered)) / n
            return {"name": name, "coverage": covered / n, "code_acc": correct / covered if covered else 0.0,
                    "wrong": wrong, "overall": overall, "covered": covered, "correct": correct}

        strategies = [
            eval_strategy("全交代码", [True] * n),
            eval_strategy("全交模型", [False] * n),
            eval_strategy("门控模型 argmax", [pr == 0 for pr in preds]),
        ]

        print(f"{'策略':<14}{'覆盖率':>8}{'代码侧正确率':>12}{'答错条数':>10}{'总体答对':>10}")
        print("-" * 58)
        for s in strategies:
            print(f"{s['name']:<14}{s['coverage']:>8.1%}{s['code_acc']:>12.1%}"
                  f"{s['wrong']:>10}{s['overall']:>10.1%}")

        # ---------- 阈值扫描：只有模型很有把握判"代码"才真交给代码 ----------
        print(f"\n阈值扫描（交代码需要 判代码 的置信度 ≥ 阈值）：")
        print(f"{'阈值':<8}{'覆盖率':>8}{'代码侧正确率':>12}{'答错条数':>10}{'总体答对':>10}")
        print("-" * 58)
        for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
            route = [(preds[i] == 0 and confs[i] >= th) for i in range(n)]
            s = eval_strategy(f">={th:.1f}", route)
            print(f"{th:<8.1f}{s['coverage']:>8.1%}{s['code_acc']:>12.1%}"
                  f"{s['wrong']:>10}{s['overall']:>10.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
