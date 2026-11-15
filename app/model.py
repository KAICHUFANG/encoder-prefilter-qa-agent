"""模型加载与推理。

两个设计点：

1) 懒加载：服务启动时不读模型（启动快、没模型也能起来），
   第一次真正要预测时才加载。和制度问答里 Retriever 的思路一致。
2) 推理不调外部 API：模型在本地，一次推理几毫秒到几十毫秒，不花 token。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import config
from app import labels as label_mod


class IntentPredictor:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = str(model_dir or config.MODEL_OUT)
        self._tok = None
        self._model = None
        self._device = "cpu"
        self.load_seconds: float | None = None
        self.loaded_at: str | None = None

    # ---------------- 加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:  # 关掉加载进度条，避免 stderr 噪音
            from transformers.utils import logging as tf_logging

            tf_logging.disable_progress_bar()
        except Exception:
            pass

        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"模型目录不存在：{self.model_dir}\n"
                f"先训练：uv run python train/finetune.py"
            )

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        use_cuda = (
            config.DEVICE in ("auto", "cuda") and torch.cuda.is_available()
        )
        self._device = "cuda" if use_cuda else "cpu"
        self._model.to(self._device)
        self._model.eval()

        # 置信度校准：train/calibrate.py 拟合出的温度，让 0.9 真的意味着 90% 对
        self._temperature = 1.0
        calib = Path(self.model_dir).parent / "calibration.json"
        if calib.exists():
            try:
                import json as _json

                self._temperature = float(
                    _json.loads(calib.read_text(encoding="utf-8"))["temperature"]
                )
            except Exception:
                self._temperature = 1.0

        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """先把模型加载进内存。测延迟前必须调一次，否则第一条会把加载时间算进去。"""
        self._ensure_loaded()

    # ---------------- 推理 ----------------
    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        import torch

        self._ensure_loaded()
        t0 = time.perf_counter()

        enc = self._tok(
            text,
            truncation=True,
            max_length=config.MAX_LEN,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits / getattr(self, "_temperature", 1.0), dim=-1).cpu().tolist()

        id2label = label_mod.id2label()
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        k = max(1, min(int(top_k), len(order)))
        top = [
            {"intent": id2label[i], "score": round(float(probs[i]), 4)}
            for i in order[:k]
        ]

        return {
            "text": text,
            "intent": top[0]["intent"],
            "confidence": top[0]["score"],
            "top_k": top,
            "device": self._device,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ---------------- 批量推理（评测用） ----------------
    def predict_batch(self, texts: list[str], batch_size: int | None = None) -> list[str]:
        """只返回每条的第一名标签。评测两千条时比逐条调快得多。"""
        import torch

        self._ensure_loaded()
        bs = batch_size or config.BATCH_SIZE
        id2label = label_mod.id2label()
        out: list[str] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i: i + bs]
            enc = self._tok(
                chunk,
                truncation=True,
                max_length=config.MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            for p in logits.argmax(dim=-1).cpu().tolist():
                out.append(id2label[p])
        return out

    def logits_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """返回原始 logits（未经 softmax）。校准温度要用它。"""
        import torch

        self._ensure_loaded()
        bs = batch_size or config.BATCH_SIZE
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            chunk = texts[i: i + bs]
            enc = self._tok(
                chunk, truncation=True, max_length=config.MAX_LEN,
                padding=True, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            out.extend(logits.cpu().tolist())
        return out


# 全局单例：整个进程只加载一次模型
predictor = IntentPredictor()
