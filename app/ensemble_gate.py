"""4 模型集成门控：一致 + 概率阈值决定"交代码还是交大模型"。

为什么用集成而不是单模型置信度：
    单模型 softmax 校准后仍分不出对错（ECE 0.22→0.13，但错答案照样高置信）；
    4 个种子模型的**分歧度**是更好的信号——全一致时准确率 78%，有分歧时 37%。

判定规则（当前工作点）：
    4 个模型都判"代码" 且 最小 P(代码) ≥ 阈值  -> 交代码
    否则                                        -> 交大模型

阈值 0.90 的实测（test 60 条）：
    覆盖 35%，路由准确率 86%；dev 上同一阈值是 75%（20 来条，区间宽）。

用法：
    from app.ensemble_gate import gate
    gate().decide("评分4.5以上的酒店")
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

ENSEMBLE_DIR = config.PROJ / "models" / "_ensemble_full"
SEEDS = (7, 13, 21, 33)
DEFAULT_THRESHOLD = 0.50


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class EnsembleGate:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._models = None

    def _load(self) -> list:
        if self._models is not None:
            return self._models
        from app.model import IntentPredictor

        dirs = [ENSEMBLE_DIR / f"seed{s}" for s in SEEDS]
        missing = [d for d in dirs if not d.exists()]
        if missing:
            raise FileNotFoundError(f"缺少集成模型：{missing}")
        models = []
        for d in dirs:
            p = IntentPredictor(model_dir=d)
            p.warmup()
            models.append(p)
        self._models = models
        return models

    def decide(self, text: str) -> dict:
        """返回 route / votes / min_p_code / 理由。"""
        models = self._load()
        votes: list[int] = []
        p_codes: list[float] = []
        for m in models:
            logits = m.logits_batch([text])[0]
            probs = _softmax(logits)
            p_codes.append(probs[0])           # 类别 0 = 交代码
            votes.append(0 if probs[0] >= 0.5 else 1)

        unanimous_code = all(v == 0 for v in votes)
        min_p = min(p_codes)
        if unanimous_code and min_p >= self.threshold:
            route, why = "code", f"4 模型一致且最小置信度 {min_p:.2f} ≥ {self.threshold}"
        elif not unanimous_code:
            route, why = "llm", f"模型有分歧（{votes.count(0)}/4 判代码）"
        else:
            route, why = "llm", f"一致但置信度不足（{min_p:.2f} < {self.threshold}）"

        return {"route": route, "why": why, "votes": votes,
                "min_p_code": round(min_p, 4),
                "mean_p_code": round(sum(p_codes) / len(p_codes), 4),
                "threshold": self.threshold}


@lru_cache(maxsize=1)
def gate() -> EnsembleGate:
    return EnsembleGate()


if __name__ == "__main__":
    g = gate()
    for q in ["你好，我想找一个评分4.5分以上的酒店", "这家酒店周边有什么景点", "帮我找个人均50以下的餐馆"]:
        r = g.decide(q)
        print(f"『{q}』 -> {r['route']}（{r['why']}）")
