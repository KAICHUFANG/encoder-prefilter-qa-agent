"""把 CrossWOZ 官方 db 转成项目用的扁平行（kb_official.json）。

为什么重建：
    data/crosswoz/kb.json 是**从对话标注里挖**出来的——只有对话里被提过的商户和属性，
    电话 26%、营业时间 10%、周边 0%，直接导致大模型 45% 的题"库里没有"。
    官方 db（data/raw/crosswoz/db/*.json）字段覆盖接近 100%，而且**带「周边景点/餐馆/酒店」关系表**。

官方库结构：list of [名称, {字段...}]
    餐馆 951 / 酒店 1133 / 景点 465 / 地铁 2549

用法：
    python train/build_kb_official.py
产出：
    data/crosswoz/kb_official.json   {"rows": [...], "stats": {...}}
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

DB_DIR = PROJ / "data" / "raw" / "crosswoz" / "db"
OUT = PROJ / "data" / "crosswoz" / "kb_official.json"

SOURCES = [("restaurant_db.json", "餐馆"), ("hotel_db.json", "酒店"),
           ("attraction_db.json", "景点"), ("metro_db.json", "地铁")]

# 列表字段展开成顿号分隔的字符串；设施单独展平成 酒店设施-XX 列
LIST_FIELDS = {"推荐菜", "周边景点", "周边餐馆", "周边酒店", "酒店设施"}


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(v) for v in value if str(v).strip())
    if isinstance(value, dict):
        return "、".join(f"{k}" for k, v in value.items() if v)
    return str(value).strip()


def main() -> int:
    rows = []
    for filename, domain in SOURCES:
        path = DB_DIR / filename
        if not path.exists():
            print(f"[跳过] 缺 {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            name, fields = (entry[0], entry[1]) if isinstance(entry, list) else (None, entry)
            row = {"domain": domain, "name": as_text(fields.get("名称") or name)}
            for k, v in fields.items():
                if k in ("领域", "名称"):
                    continue
                if k == "酒店设施":
                    for fac in (v if isinstance(v, list) else []):
                        fac = as_text(fac)
                        if fac:
                            row[f"酒店设施-{fac}"] = "有"
                    row["设施列表"] = as_text(v)
                    continue
                row[k] = as_text(v)
            rows.append(row)

    # 同一家店可能同时出现在多个库里（例如酒店里也有同名餐厅行），按名称合并：
    # 保留字段最多的那条 domain，其余字段并入；避免"空行覆盖有数据的行"。
    merged: dict[str, dict] = {}
    dup_count = 0
    for r in rows:
        name = r["name"]
        if name not in merged:
            merged[name] = r
            continue
        dup_count += 1
        keep, other = merged[name], r
        if len([k for k, v in other.items() if str(v).strip()]) > \
           len([k for k, v in keep.items() if str(v).strip()]):
            keep, other = other, keep
        for k, v in other.items():
            if str(v).strip() and not str(keep.get(k, "")).strip():
                keep[k] = v
        merged[name] = keep
    rows = list(merged.values())
    print(f"[合并] 同名跨库 {dup_count} 条，去重后 {len(rows)} 家")

    stats = {
        "total": len(rows),
        "by_domain": dict(Counter(r["domain"] for r in rows)),
        "coverage": {},
    }
    for field in ("电话", "地址", "营业时间", "推荐菜", "评分", "价格", "门票", "人均消费",
                  "游玩时间", "酒店类型", "周边景点", "周边餐馆", "周边酒店"):
        n = sum(1 for r in rows if str(r.get(field, "")).strip())
        stats["coverage"][field] = f"{n / max(1, len(rows)):.0%}"

    OUT.write_text(json.dumps({"rows": rows, "stats": stats}, ensure_ascii=False),
                   encoding="utf-8")

    print(f"写出 {OUT}（{OUT.stat().st_size/1024:.0f} KB）")
    print(f"总行数 {len(rows)}  按领域 {stats['by_domain']}")
    print("字段覆盖:", stats["coverage"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
