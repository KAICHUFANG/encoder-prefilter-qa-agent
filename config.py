"""集中配置：数据集切换、路径、模型、训练超参。

**支持多数据集共存**：用环境变量切换，两套数据/模型/报告互不覆盖，方便对比。

    $env:INTENT_DATASET = "crosswoz"    # 默认
    $env:INTENT_DATASET = "kuake"       # 之前那份医疗意图数据

目录约定：
    data/<dataset>/train.jsonl      统一格式的数据
    data/raw/<dataset>/             原始下载文件（不提交 git）
    models/<dataset>/intent-bert/   训练产物
    eval/<dataset>/report*.json     评测报告
"""
from __future__ import annotations

import os
from pathlib import Path

PROJ = Path(__file__).resolve().parent

# ---------------- 数据集选择 ----------------
DATASET = os.getenv("INTENT_DATASET", "crosswoz")

DATA_DIR = PROJ / "data" / DATASET
RAW_DIR = PROJ / "data" / "raw" / DATASET
TRAIN_JSONL = DATA_DIR / "train.jsonl"
DEV_JSONL = DATA_DIR / "dev.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"
LABELS_FILE = DATA_DIR / "labels.json"      # 由 prepare*.py 从数据统计生成
ZHIDU_TEST = DATA_DIR / "test_zhidu20.jsonl"  # 迁移测试（可选）

# ---------------- 模型产物 ----------------
MODELS_DIR = PROJ / "models" / DATASET
MODEL_OUT = MODELS_DIR / "intent-bert"
BASELINE_JSON = MODELS_DIR / "baseline.json"
TRAIN_META = MODELS_DIR / "train_meta.json"

# ---------------- 评测产物 ----------------
EVAL_DIR = PROJ / "eval" / DATASET
REPORT_JSON = EVAL_DIR / "report.json"
REPORT_OFFLINE_JSON = EVAL_DIR / "report_offline.json"
REPORT_ROUTING_JSON = EVAL_DIR / "report_routing.json"

# ---------------- 模型与训练超参 ----------------
MODEL_NAME = os.getenv("INTENT_MODEL", "bert-base-chinese")
MAX_LEN = int(os.getenv("INTENT_MAX_LEN", "64"))
BATCH_SIZE = int(os.getenv("INTENT_BATCH", "16"))
EPOCHS = 3
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = int(os.getenv("INTENT_SEED", "42"))
DEVICE = os.getenv("INTENT_DEVICE", "auto")

# ---------------- HuggingFace 镜像 ----------------
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# 关掉下载/加载进度条：它们往 stderr 写，Windows PowerShell 会把 stderr 当成错误，
# 界面上会出现一堆 "NativeCommandError"，看起来像程序崩了，其实没事。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def describe() -> str:
    return (f"dataset={DATASET} model={MODEL_NAME} device={DEVICE} "
            f"max_len={MAX_LEN} batch={BATCH_SIZE} epochs={EPOCHS} lr={LR}")


def check_data() -> list[str]:
    problems: list[str] = []
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        problems.append(f"{RAW_DIR} 是空的：先下载原始数据（见 README）")
    for p in (TRAIN_JSONL, DEV_JSONL, TEST_JSONL, LABELS_FILE):
        if not p.exists():
            problems.append(f"缺少 {p.name}：先跑对应的 train/prepare*.py")
    return problems


def check_model() -> list[str]:
    problems: list[str] = []
    if not MODEL_OUT.exists():
        problems.append(f"{MODEL_OUT} 不存在：先跑 uv run python train/finetune.py")
    if not LABELS_FILE.exists():
        problems.append("缺少 labels.json：先跑 prepare")
    return problems
