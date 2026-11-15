"""标签的唯一真相源。

关键设计：标签**不是手写死的**，而是 train/prepare.py 从数据里统计出来的
（写到 data/labels.json）。好处是换数据集版本时，标签自动跟着变，
不会出现「代码里 11 类、数据里 12 类」这种对不上的事故。
"""
from __future__ import annotations

import json
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def load() -> dict:
    if not config.LABELS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {config.LABELS_FILE}\n"
            f"先跑：uv run python train/prepare.py"
        )
    data = json.loads(config.LABELS_FILE.read_text(encoding="utf-8"))
    labels: list[str] = data["labels"]
    return {
        "labels": labels,
        "label2id": {name: i for i, name in enumerate(labels)},
        "id2label": {i: name for i, name in enumerate(labels)},
    }


def names() -> list[str]:
    return load()["labels"]


def label2id() -> dict[str, int]:
    return load()["label2id"]


def id2label() -> dict[int, str]:
    return load()["id2label"]


def count() -> int:
    return len(load()["labels"])
