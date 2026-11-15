"""「这段话我见过吗」检测器。

为什么需要它：
    模型对不认识的词（比如没在训练集出现过的品牌名）会给出 0.999 的自信错误。
    温度缩放只能修正「整体偏高」，修不了「没见过还很自信」——因为那是全局缩放，
    0.999 除以 1.35 仍然是 0.99。

原理（字符二元组覆盖率）：
    把训练集所有文本切成字符二元组，形成一个「见过的片段」词表。
    对一句新输入，看它的二元组有多大比例出现在训练集里。
    没见过品牌名 → 里面有大量从没出现过的片段 → 覆盖率低 → 判为 unseen。

    为什么用二元组而不是分词：中文不用分词器也能切，而且对错别字/新词更敏感。

用法：
    from app.ood import detector
    cov = detector().coverage("帮我订个希尔顿花园")   # 0~1
    detector().is_unseen("帮我订个希尔顿花园")        # True/False
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import config

# 低于这个覆盖率判为「没见过」。由 eval/run_ood_eval.py 在数据上标定。
COVERAGE_FLOOR = 0.70


class SeenVocabulary:
    def __init__(self, texts: list[str]) -> None:
        self.bigrams: set[str] = set()
        for t in texts:
            t = t.replace(" ", "")
            for i in range(len(t) - 1):
                self.bigrams.add(t[i: i + 2])
        self.n_texts = len(texts)

    def coverage(self, text: str) -> float:
        """输入里的字符二元组，有多大比例在训练集出现过。"""
        t = (text or "").replace(" ", "")
        if len(t) < 2:
            return 0.0
        grams = [t[i: i + 2] for i in range(len(t) - 1)]
        hit = sum(1 for g in grams if g in self.bigrams)
        return hit / len(grams)

    def unseen_fragments(self, text: str) -> list[str]:
        t = (text or "").replace(" ", "")
        return [t[i: i + 2] for i in range(len(t) - 1) if t[i: i + 2] not in self.bigrams]

    def max_unseen_run(self, text: str) -> int:
        """最长的一段「连续没见过的二元组」——用来抓长句里夹着的陌生实体。

        为什么要它：覆盖率是比例，长句里夹一个陌生品牌名，覆盖率仍然很高
        （比如「推荐一下希尔顿花园」里大部分二元组都见过）。
        但陌生品牌名会形成一段连续未见的片段，这个信号更尖锐。

        返回的是覆盖的字符数：连续 k 个未见二元组 ≈ k+1 个字没见过的名字。
        """
        t = (text or "").replace(" ", "")
        if len(t) < 2:
            return 0
        run = best = 0
        for i in range(len(t) - 1):
            if t[i: i + 2] not in self.bigrams:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best + 1 if best else 0


@lru_cache(maxsize=1)
def detector() -> SeenVocabulary:
    if not config.TRAIN_JSONL.exists():
        raise FileNotFoundError(f"缺少 {config.TRAIN_JSONL}，OOD 检测需要训练集文本")
    texts = [json.loads(line)["text"] for line in config.TRAIN_JSONL.open(encoding="utf-8") if line.strip()]
    return SeenVocabulary(texts)


# 判定「没见过」的阈值——**必须按数据集标定，不能跨数据集移植**
#   crosswoz：句子长（平均 27 字），覆盖率 P5=0.885 -> 0.80
#   massive ：句子短（平均 10.5 字），覆盖率 P5=0.50 -> 0.50
# 用错阈值的实测后果：在 MASSIVE 上沿用 0.80，会把 26.4% 的正常请求误判成"没见过"，
# 相当于白白多调 26% 的大模型——分流就白做了。
COVERAGE_FLOORS = {"crosswoz": 0.80, "massive": 0.50, "kuake": 0.80}
COVERAGE_FLOOR = COVERAGE_FLOORS.get(config.DATASET, 0.70)

# 「连续未见片段」这个信号实测无效：域内误杀 8.6%，却没有多拦下任何一条高置信度错误，
# 所以默认关闭。留在这里是为了留下实验记录——试过、不划算、不用。
USE_UNSEEN_RUN = False
UNSEEN_RUN_FLOOR = 4


def is_unseen(text: str, coverage_floor: float | None = None) -> bool:
    coverage_floor = COVERAGE_FLOOR if coverage_floor is None else coverage_floor
    det = detector()
    if det.coverage(text) < coverage_floor:
        return True
    if USE_UNSEEN_RUN and det.max_unseen_run(text) >= UNSEEN_RUN_FLOOR:
        return True
    return False
