"""校验标签稳定性：给判官补上「对话上下文 + 参照门店完整档案」，重判 60 条 test。

要回答的问题：
    原来的标注只给了「单句问题 + 代码回答 + 代码用到的行」，
    多轮指代（"它/这家"）时判官看不到上下文，标签可能偏。
    补上上下文后重判，看标签会不会变、一致率多少。

用法：
    $env:INTENT_DATASET = "crosswoz_gate"
    python eval/check_label_context.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

sys.path.insert(0, str(PROJ / "data"))
from relabel_gate_by_llm import SYSTEM, load_env, ENV_FILE  # noqa: E402

TEST = PROJ / "data" / "crosswoz_gate_llm" / "test.jsonl"
PROGRESS = PROJ / "data" / "crosswoz_gate_llm" / "_progress.jsonl"
KB = PROJ / "data" / "crosswoz" / "kb_official.json"
RAW = PROJ / "data" / "raw" / "crosswoz" / "extracted" / "test.json"
ALL_ITEMS = PROJ / "data" / "crosswoz" / "test.jsonl"


def row_brief(r: dict) -> str:
    keys = ("评分", "价格", "门票", "人均消费", "游玩时间", "营业时间", "电话", "地址",
            "酒店类型", "推荐菜", "地铁", "周边景点", "周边餐馆", "周边酒店")
    parts = [f"{k}={str(r[k])[:50]}" for k in keys if r.get(k)]
    fac = [k.replace("酒店设施-", "设施-") for k in r
           if k.startswith("酒店设施-") and str(r[k]).strip()][:8]
    return "- " + "｜".join(parts + fac)


def call(base, key, model, user_text) -> tuple[int, str]:
    payload = json.dumps({"model": model,
                          "messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user_text}],
                          "temperature": 0, "max_tokens": 120}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"].strip().strip("`").replace("json", "", 1).strip()
    obj = json.loads(content)
    return int(obj["label"]), str(obj.get("reason", ""))[:40]


def main() -> int:
    env = load_env(ENV_FILE)
    key = env.get("DEEPSEEK_API_KEY", "")
    base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("MODEL_NAME", "deepseek-chat")

    kb_rows = json.loads(KB.read_text(encoding="utf-8"))["rows"]
    by_name = {r["name"]: r for r in kb_rows}
    names = sorted(by_name, key=len, reverse=True)

    answers = {}
    for line in PROGRESS.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            answers[rec["id"]] = rec

    all_items = [json.loads(line) for line in ALL_ITEMS.open(encoding="utf-8") if line.strip()]
    raw = json.loads(RAW.read_text(encoding="utf-8"))

    rows = [json.loads(line) for line in TEST.open(encoding="utf-8") if line.strip()]
    agree = Counter()
    flips = []
    for r in rows:
        prev = answers.get(r["id"], {})
        m = re.search(r"(\d+)$", r["id"])
        idx = int(m.group(1)) if m else -1
        ctx_lines, ref = [], None
        if 0 <= idx < len(all_items):
            dlg = all_items[idx]["dialog_id"]
            msgs = raw.get(str(dlg), {}).get("messages", [])
            pos = next((i for i, x in enumerate(msgs)
                        if str(x.get("content", "")).strip() == r["text"].strip()), None)
            if pos is not None:
                for msg in msgs[max(0, pos - 8):pos]:
                    role = "用户" if msg.get("role") == "usr" else "客服"
                    ctx_lines.append(f"{role}：{str(msg.get('content',''))[:50]}")
                    hit = next((n for n in names if n in str(msg.get("content", ""))), None)
                    if hit:
                        ref = hit
        evidence = row_brief(by_name[ref]) if ref else "（无）"
        user_text = (f"【对话上下文】\n" + ("\n".join(ctx_lines) if ctx_lines else "（无）") +
                     f"\n\n用户问题：{r['text']}" +
                     f"\n\n代码回答：{prev.get('code_answer','')}" +
                     f"\n\n参照门店完整档案：\n{evidence}")
        try:
            label, reason = call(base, key, model, user_text)
        except Exception as exc:
            print(f"  失败 {r['id']}: {exc}")
            continue
        same = label == int(r["intent"])
        agree["same" if same else "diff"] += 1
        if not same:
            flips.append((r["id"], r["text"][:26], int(r["intent"]), label, reason))
        time.sleep(0.15)

    n = sum(agree.values())
    print(f"\n重判 {n} 条：与原标签一致 {agree['same']} 条 = {agree['same']/n:.1%}")
    print(f"翻转 {len(flips)} 条：")
    for fid, text, old, new, why in flips[:8]:
        print(f"   {fid} 『{text}』 原={old} 新={new} | {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
