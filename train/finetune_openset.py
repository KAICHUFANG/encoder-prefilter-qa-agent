"""训练侧的解法：加「未知」类 + 域外负样本（Outlier Exposure）+ 标签平滑。

为什么要这个，而不是后处理补丁：
    补丁（温度缩放、覆盖率护栏）是在模型外面加检查，模型本身还是"什么都要答"。
    训练侧的做法是**改变模型的目标**：让它学会第 8 个类——"这不是我的活"。

两类域外负样本（都标成「未知」）：
    1) 别的领域问句：复用 KUAKE-QIC 的医疗问句（和我们 5 个领域毫无关系）
    2) 陌生实体模板：随机生成假品牌名填进常见句式（"帮我订个{假名}"）
       —— 这正是模型翻车的场景：句式熟悉、实体陌生

损失函数：
    L = 交叉熵(样本, 标签) + λ · 均匀分布损失(域外样本)
    第二项是 Outlier Exposure 的核心：对域外样本，把预测拉到"均匀分布"，
    等于告诉模型"这些都不是答案，别挑一个最像的"。

标签平滑（0.1）顺带压住过度自信。

用法：
    uv run python train/finetune_openset.py
    uv run python train/finetune_openset.py --epochs 3 --oe-weight 0.5
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

UNKNOWN = "未知"
FAKE_CHARS = "阿尔法贝塔伽马德尔塔西格玛欧米伽布拉格维也纳萨尔斯堡京都首尔曼谷"
OOD_DIR = PROJ / "data" / "kuake" / "train.jsonl"   # 借另一份数据集当域外样本


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def make_ood_texts(n_fake: int = 2500, seed: int = 0) -> list[str]:
    """域外文本：医疗问句 + 随机假品牌名模板。"""
    rng = random.Random(seed)
    out: list[str] = []

    medical = load_jsonl(OOD_DIR)
    for r in medical:
        out.append(r["text"])

    templates = ["帮我订个{n}", "我想去{n}", "推荐一下{n}", "{n}", "{n}怎么样",
                 "{n}附近有什么", "怎么去{n}", "{n}贵不贵"]
    pool = list(FAKE_CHARS)
    for _ in range(n_fake):
        name = "".join(rng.choice(pool) for _ in range(rng.randint(2, 4)))
        out.append(rng.choice(templates).format(n=name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--oe-weight", type=float, default=0.5, help="域外损失权重 λ")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
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
    all_labels = labels + [UNKNOWN]
    label2id = {n: i for i, n in enumerate(all_labels)}
    id2label = {i: n for n, i in label2id.items()}

    device = "cuda" if (config.DEVICE in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    print(f"open-set 训练：{len(labels)} 个已知类 + 1 个「{UNKNOWN}」类")
    print(f"模型 {config.MODEL_NAME} | 设备 {device} | train {len(train)} / dev {len(dev)}")
    print(f"标签平滑 {args.label_smoothing} | OE 权重 {args.oe_weight}")

    ood_texts = make_ood_texts()
    print(f"域外负样本 {len(ood_texts)} 条（医疗问句 + 假品牌名模板）")
    print("  例：", ood_texts[-3:])

    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=len(all_labels), id2label=id2label, label2id=label2id,
    ).to(device)

    class DS(Dataset):
        def __init__(self, texts: list[str], ys: list[int] | None = None):
            self.enc = tok(texts, truncation=True, max_length=config.MAX_LEN, padding=False)
            self.ys = ys

        def __len__(self) -> int:
            return len(self.enc["input_ids"])

        def __getitem__(self, i: int):
            item = {k: v[i] for k, v in self.enc.items()}
            if self.ys is not None:
                item["labels"] = self.ys[i]
            return item

    def collate(batch):
        feats = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        out = tok.pad(feats, padding=True, return_tensors="pt")
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out

    train_ds = DS([r["text"] for r in train], [label2id[r["intent"]] for r in train])
    ood_ds = DS(ood_texts, [label2id[UNKNOWN]] * len(ood_texts))
    dev_ds = DS([r["text"] for r in dev], [label2id[r["intent"]] for r in dev])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    counts = Counter(r["intent"] for r in train)
    weights = [len(train) / (len(labels) * max(1, counts.get(lb, 1))) for lb in labels]
    weights.append(1.0)  # 未知类权重固定 1
    ce = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float, device=device),
        label_smoothing=args.label_smoothing,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * args.epochs)
    try:
        from transformers import get_linear_schedule_with_warmup

        sched = get_linear_schedule_with_warmup(
            opt, int(total_steps * config.WARMUP_RATIO), total_steps)
    except Exception:
        sched = None

    def evaluate() -> tuple[float, float]:
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                targets = batch.pop("labels")
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(targets.cpu().tolist())
        from sklearn.metrics import f1_score

        acc = float(np.mean(np.array(preds) == np.array(trues)))
        f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
        return acc, f1

    best = (-1.0, 0)
    out_dir = config.MODELS_DIR / "intent-bert-openset"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_ce = running_oe = 0.0
        ood_iter = iter(ood_loader)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch.pop("labels")
            logits = model(**batch).logits
            loss_ce = ce(logits, targets)

            # Outlier Exposure：对域外样本，把预测拉到均匀分布
            try:
                ood_batch = next(ood_iter)
            except StopIteration:
                ood_iter = iter(ood_loader)
                ood_batch = next(ood_iter)
            ood_batch = {k: v.to(device) for k, v in ood_batch.items()}
            ood_batch.pop("labels")
            ood_logits = model(**ood_batch).logits
            loss_oe = -torch.log_softmax(ood_logits, dim=-1).mean()

            loss = loss_ce + args.oe_weight * loss_oe
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad()
            running_ce += float(loss_ce)
            running_oe += float(loss_oe)
            if step % 200 == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"ce {running_ce / step:.4f} oe {running_oe / step:.4f}", flush=True)

        acc, f1 = evaluate()
        print(f"[epoch {epoch}] {time.perf_counter() - t0:.0f}s | dev 准确率 {acc:.4f} "
              f"| dev macro-F1 {f1:.4f}", flush=True)
        if f1 > best[0]:
            best = (f1, epoch)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            (out_dir / "labels.json").write_text(json.dumps(
                {"labels": all_labels, "unknown": UNKNOWN}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"  [保存] -> {out_dir}", flush=True)

    print(f"\n训练结束：最好 macro-F1 {best[0]:.4f}（第 {best[1]} 轮）")
    print(f"模型 {out_dir}")
    print("下一步：uv run python eval/run_openset_eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
