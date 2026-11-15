"""评测指标：准确率、macro-F1、每类 F1、混淆矩阵。

为什么要专门算 macro-F1：类别不平衡时，总准确率会被多数类吃掉。
一个「永远猜最常见那类」的模型也能有不错的 accuracy，但 macro-F1 会很惨。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """算全套指标。y_true / y_pred 都是标签原文（中文）。"""
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    true_arr = np.array(y_true)
    result = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
        "weighted_f1": round(weighted, 4),
        "labels": labels,
        "per_class_f1": {lb: round(float(s), 4) for lb, s in zip(labels, per)},
        "support": {lb: int((true_arr == lb).sum()) for lb in labels},
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "total": len(y_true),
    }
    result["top_confusions"] = top_confusions(result, k=5)
    return result


def top_confusions(result: dict, k: int = 5) -> list[dict]:
    """最大的几处「真实 A 被判成 B」——报告里最有信息量的一张表。"""
    labels = result["confusion"]["labels"]
    matrix = result["confusion"]["matrix"]
    pairs: list[dict] = []
    for i, row in enumerate(matrix):
        for j, n in enumerate(row):
            if i != j and n > 0:
                pairs.append({"true": labels[i], "pred": labels[j], "count": int(n)})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:k]


def format_confusion(result: dict) -> str:
    """把混淆矩阵打成可读的文本表（终端能直接看，也能贴进报告）。"""
    labels: list[str] = result["confusion"]["labels"]
    matrix: list[list[int]] = result["confusion"]["matrix"]
    short = [str(lb)[:7] for lb in labels]
    width = max(10, max(len(s) for s in short) + 2)

    lines = ["真实 \\ 预测".ljust(width) + "".join(s.rjust(width) for s in short)]
    for i, row in enumerate(matrix):
        lines.append(short[i].ljust(width) + "".join(str(v).rjust(width) for v in row))
    return "\n".join(lines)


def format_summary(result: dict) -> str:
    """一行摘要 + 每类 F1 + 最大混淆对。"""
    lines = [
        f"样本 {result['total']} 条 | 准确率 {result['accuracy']:.4f} "
        f"| macro-F1 {result['macro_f1']:.4f} | weighted-F1 {result['weighted_f1']:.4f}",
        "",
        "每类 F1（括号内为样本数）：",
    ]
    for lb in result["labels"]:
        lines.append(
            f"  {str(lb)[:14]:<16} F1={result['per_class_f1'][lb]:.4f}  "
            f"n={result['support'][lb]}"
        )
    if result["top_confusions"]:
        lines.append("")
        lines.append("最大混淆（真实 -> 判错成）：")
        for p in result["top_confusions"]:
            lines.append(f"  {str(p['true'])[:14]:<16} -> {str(p['pred'])[:14]:<16} {p['count']} 次")
    return "\n".join(lines)
