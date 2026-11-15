"""微调 encoder-only 模型做意图分类。

为什么用 encoder（BERT 系）而不是生成模型：
    分类任务只需要「理解」，不需要「生成」。encoder 参数少 10-100 倍，
    推理是毫秒级，CPU 都能训。这就是「选对模型规模」而不是「越大越好」。

用法：
    uv run python train/finetune.py
    uv run python train/finetune.py --epochs 2 --limit 500        # 冒烟测试
    $env:INTENT_MODEL="hfl/rbt3"; uv run python train/finetune.py  # 换小模型，CPU 更快

产出：
    models/intent-bert/      模型 + 分词器
    models/train_meta.json   本次训练的超参和最好成绩
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from app import metrics as metrics_mod  # noqa: E402


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
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--limit", type=int, default=0, help="每个划分最多用多少条（冒烟用）")
    ap.add_argument("--no-class-weight", action="store_true",
                    help="关掉类别权重（默认开，因为基线用了 balanced，比较必须公平）")
    ap.add_argument("--save-last", action="store_true",
                    help="总是保存最后一轮（当 dev 与训练同分布、会饱和到 1.0 时，按 dev 选轮次失效）")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:  # 关掉进度条，避免 stderr 噪音
        from transformers.utils import logging as tf_logging

        tf_logging.disable_progress_bar()
    except Exception:
        pass

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    train = load_jsonl(config.TRAIN_JSONL)
    dev = load_jsonl(config.DEV_JSONL)
    if args.limit:
        train, dev = train[: args.limit], dev[: args.limit]

    labels = json.loads(config.LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for i, name in enumerate(labels)}

    device = "cuda" if (config.DEVICE in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    print(f"模型 {config.MODEL_NAME} | 设备 {device} | 类别 {len(labels)}")
    print(f"train {len(train)} / dev {len(dev)} | epochs {args.epochs} "
          f"| batch {args.batch_size} | lr {args.lr}")
    print(f"HF_ENDPOINT {config.HF_ENDPOINT}\n")

    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    class IntentDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.texts = [r["text"] for r in rows]
            self.ys = [label2id[r["intent"]] for r in rows]
            self.enc = tok(self.texts, truncation=True, max_length=config.MAX_LEN)

        def __len__(self) -> int:
            return len(self.ys)

        def __getitem__(self, i: int):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.ys[i]
            return item

    def _collate(batch: list[dict]):
        """动态补齐：同一批里按最长的那条补 pad，比全局补到 64 快。"""
        feats = [{k: v for k, v in item.items() if k != "labels"} for item in batch]
        out = tok.pad(feats, padding=True, return_tensors="pt")
        out["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return out

    train_loader = DataLoader(
        IntentDataset(train), batch_size=args.batch_size,
        shuffle=True, collate_fn=_collate,
    )
    dev_loader = DataLoader(
        IntentDataset(dev), batch_size=args.batch_size,
        shuffle=False, collate_fn=_collate,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    # 类别权重：w_c = N / (K * n_c)，和 sklearn 的 class_weight="balanced" 同一个公式。
    # 为什么必须有：数据最大类 676 条、最小类 28 条，不加权模型会放弃小类，
    # macro-F1 会崩——而基线是加过权的，不加权就是不公平比较。
    counts = Counter(r["intent"] for r in train)
    n_all, k_all = len(train), len(labels)
    class_weights = [n_all / (k_all * max(1, counts[lb])) for lb in labels]
    if args.no_class_weight:
        criterion = torch.nn.CrossEntropyLoss()
        print("类别权重：关闭")
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float, device=device)
        )
        print("类别权重：" + " ".join(
            f"{lb}={w:.2f}" for lb, w in zip(labels, class_weights)))

    total_steps = max(1, len(train_loader) * args.epochs)
    try:
        from transformers import get_linear_schedule_with_warmup

        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
            num_training_steps=total_steps,
        )
    except Exception:  # 版本差异兜底：没有调度器也能训
        sched = None

    def evaluate() -> dict:
        model.eval()
        preds: list[str] = []
        trues: list[str] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                for p in logits.argmax(dim=-1).cpu().tolist():
                    preds.append(id2label[p])
        trues = [r["intent"] for r in dev]
        return metrics_mod.compute(trues, preds, labels)

    best_f1 = -1.0
    best_epoch = 0
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch.pop("labels")
            out = model(**batch)
            loss = criterion(out.logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad()
            running += float(loss.detach())
            if step % 50 == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}", flush=True)

        dev_res = evaluate()
        print(f"[epoch {epoch}] 用时 {time.perf_counter() - t0:.0f}s | "
              f"dev 准确率 {dev_res['accuracy']:.4f} | dev macro-F1 {dev_res['macro_f1']:.4f}",
              flush=True)

        if args.save_last or dev_res["macro_f1"] > best_f1:
            best_f1 = dev_res["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(config.MODEL_OUT)
            tok.save_pretrained(config.MODEL_OUT)
            print(f"  [保存] 第 {epoch} 轮，模型写入 {config.MODEL_OUT}", flush=True)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": config.MODEL_NAME,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": round(best_f1, 4),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": config.MAX_LEN,
        "device": device,
        "class_weight": (not args.no_class_weight),
        "class_weights": {lb: round(w, 4) for lb, w in zip(labels, class_weights)},
        "train_size": len(train),
        "dev_size": len(dev),
        "labels": labels,
    }
    (config.MODELS_DIR / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"训练结束：最好的一轮是第 {best_epoch} 轮，dev macro-F1 {best_f1:.4f}")
    print(f"模型在 {config.MODEL_OUT}")
    print("下一步：uv run python train/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
