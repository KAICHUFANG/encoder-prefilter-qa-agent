"""向量检索：用 BGE 中文小模型把商户/条款编码，查询走余弦相似度。

为什么用向量：
    规则检索只认精确字面——用户说「香江戴斯」（简称）匹配不到全名
    「北京香江戴斯酒店」；也不能按语义相关度排序。
    向量检索能解决简称/近义/语序问题。

为什么不用 faiss：
    2549 条 × 512 维 ≈ 5 MB，numpy 点积就够（<1ms），上 faiss 是过度设计。

模型：BAAI/bge-small-zh-v1.5（95 MB，512 维，中文检索常用）
    查询侧加官方指令前缀，文档侧不加。

数据源：data/crosswoz/kb_official.json（数据集自带的 database/）

用法：
    from app.vector_search import build, search
    build()                              # 首次建索引
    for hit in search("香江戴斯是几星", top_k=3):
        print(hit["score"], hit["name"])
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

MODEL_DIR = config.PROJ / "models" / "_pretrained" / "bge-small-zh-v1.5"
INDEX_DIR = config.PROJ / "data" / "crosswoz"
VEC_FILE = INDEX_DIR / "vectors.npz"
DOC_FILE = INDEX_DIR / "vector_docs.jsonl"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
MAX_LEN = 256


@lru_cache(maxsize=1)
def model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"缺少 {MODEL_DIR}，先下载 bge-small-zh-v1.5")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    enc = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, enc.to(device).eval(), device


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """BGE 的正规用法：CLS 池化 + L2 归一化。"""
    import torch

    tok, enc, device = model()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                         return_tensors="pt").to(device)
            hidden = enc(**inputs).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            out.append(hidden.cpu().numpy().astype("float32"))
    return np.vstack(out)


VENUE_FIELDS = ("domain", "评分", "价格", "门票", "人均消费", "游玩时间", "营业时间",
                "电话", "地址", "酒店类型", "推荐菜", "地铁")


def venue_doc(row: dict) -> str:
    parts = [f"{row.get('domain', '')} {row.get('name', '')}"]
    for k in VENUE_FIELDS[1:]:
        if row.get(k):
            parts.append(f"{k} {row[k]}")
    fac = [k.replace("酒店设施-", "") for k in row if k.startswith("酒店设施-") and str(row[k]).strip()]
    if fac:
        parts.append("设施 " + " ".join(fac[:12]))
    return "｜".join(parts)


def venue_docs() -> list[dict]:
    from app import executor

    executor.load_rows.cache_clear()
    docs = []
    for i, r in enumerate(executor.load_rows()):
        docs.append({"id": f"venue-{i}", "kind": "venue", "name": r.get("name", ""),
                     "domain": r.get("domain", ""), "text": venue_doc(r), "row": r})
    return docs


def clause_docs() -> list[dict]:
    from app import enterprise_answer

    docs = []
    for i, c in enumerate(enterprise_answer.load_clauses()):
        text = f"{c['doc']} {c['number']} {c['heading']} {c['text']}"
        docs.append({"id": f"clause-{i}", "kind": "clause", "name": f"{c['number']} {c['heading']}",
                     "domain": "制度", "text": text, "row": c})
    return docs


def build(force: bool = False) -> dict:
    """建索引并落盘；已存在且文档数一致就复用。"""
    docs = venue_docs() + clause_docs()
    if not force and VEC_FILE.exists() and DOC_FILE.exists():
        old = sum(1 for _ in DOC_FILE.open(encoding="utf-8"))
        if old == len(docs):
            return {"reused": True, "docs": old}

    vec = encode([d["text"] for d in docs])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VEC_FILE, vec=vec)
    with DOC_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"reused": False, "docs": len(docs), "dim": int(vec.shape[1])}


@lru_cache(maxsize=1)
def index():
    if not VEC_FILE.exists() or not DOC_FILE.exists():
        build()
    vec = np.load(VEC_FILE)["vec"]
    docs = [json.loads(line) for line in DOC_FILE.open(encoding="utf-8") if line.strip()]
    return vec, docs


def search(query: str, top_k: int = 5, kind: str | None = None,
           domain: str | None = None) -> list[dict]:
    """余弦检索。kind: venue/clause；domain: 餐馆/酒店/景点。"""
    vec, docs = index()
    q = encode([QUERY_PREFIX + query])[0]
    scores = vec @ q
    order = np.argsort(-scores)
    hits = []
    for i in order:
        d = docs[int(i)]
        if kind and d["kind"] != kind:
            continue
        if domain and d["kind"] == "venue" and d["domain"] != domain:
            continue
        hits.append({**d, "score": float(scores[int(i)])})
        if len(hits) >= top_k:
            break
    return hits


if __name__ == "__main__":
    print(build(force=True))
    for q in ["香江戴斯是几星", "评分4.5以上人均100-150的餐馆", "年假怎么算"]:
        print(f"\n『{q}』")
        for h in search(q, top_k=3):
            print(f"   {h['score']:.3f}  [{h['kind']}] {h['name'][:28]}")
