"""生成「路由数据集」：让 encoder 判断「这句话代码能不能直接完成」。

策略变更（用户提出）：
    旧：encoder 分 25 个制度主题 -> 查条款 / 算数
    新：encoder 判断「代码能不能完成」
            - code     ：规则能抽全参数 -> 代码直接执行（0 token）
            - ask      ：参数没给，先追问（0 token）
            - extract  ：参数提到了但规则抽不出 -> 交大模型抽参数，再由代码执行
            - other    ：制度外/闲聊 -> 大模型接管

为什么这样更好：
    1. 编码器擅长二/三分类，不擅长细分 25 类（实测准确率 58.6%）
    2. 分类错误不再等于答错——只要判"能不能"对，后路是可用的
    3. 大模型只在"抽参数"这一小步被调用，执行仍然确定

标签是怎么来的（可复现，不靠人标）：
    对每条问句实际跑一遍代码链路，看它能不能给出完整答案：
        需要的槽位齐了           -> code
        需要槽位但句子里没给      -> ask
        句子里有数字/亲属词但抽不出 -> extract
        制度外/闲聊               -> other

用法：
    python data/gen_route_dataset.py
产出：
    data/raw/route/questions.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

from app import enterprise_answer  # noqa: E402
from app.slots import extract  # noqa: E402

SRC = PROJ / "data" / "raw" / "enterprise_hard" / "questions.jsonl"
OUT_DIR = PROJ / "data" / "raw" / "route"

# 每个主题**在问数量时**才需要额外参数（问"能跨年吗"就不需要工龄）
REQUIRED_WHEN = {
    "年休假": (re.compile(r"(几天|多少天|几天假|能休多久|歇多少天|一共多少天)"), ["工作年限"]),
    "住宿": (re.compile(r"(超标|超了|能报|上限|够不够|报多少)"), ["金额"]),
}

# "参数在句子里出现过"的判定（按槽位语义分开，不能用一个大正则）
SLOT_MENTION = {
    "工作年限": re.compile(
        r"(干|工作|工龄|入职|从业|这行|做了)[^，。？！]{0,8}"
        r"([0-9零一二两三四五六七八九十百千万]+年|一打年|两位数)"
        r"|(二零零八|二零一|一九九)年"),
    "金额": re.compile(
        r"[0-9零一二两三四五六七八九十百千万]+\s*(元|块|百|千|万)"
        r"|(一晚|住一晚)[^，。？！]{0,8}[0-9零一二两三四五六七八九十百千万]+"
        r"|(四张|半千)"),
}
KINISH = re.compile(r"(叔|姑|舅|姨|侄|甥|堂|表|岳|公公|婆婆|继)")
# 明显需要生成式回答的（制度里不写这些）
ANSWERISH = re.compile(r"(为什么|怎么会|是不是可以理解|举个例子|解释一下|你觉得)")


def label_of(topic: str, text: str) -> tuple[str, dict, str]:
    """跑一遍代码链路，根据结果打标签。"""
    slots = extract(text)

    if topic in ("out_of_scope", "chitchat") or ANSWERISH.search(text):
        return "other", slots, "制度外/闲聊/求解释"

    rule, need = REQUIRED_WHEN.get(topic, (None, []))
    if not need or not (rule and rule.search(text)):
        # 不需要算数，查条款即可
        return "code", slots, "查条款可完成"

    missing = [s for s in need if s not in slots]
    if not missing:
        return "code", slots, "参数齐了，可算数"

    if any(SLOT_MENTION.get(s) and SLOT_MENTION[s].search(text) for s in missing) or KINISH.search(text):
        return "extract", slots, f"提到参数但规则抽不出 {missing}"
    return "ask", slots, f"未提供 {missing}"


def main() -> int:
    if not SRC.exists():
        print(f"[失败] 缺 {SRC}")
        return 1

    rows = []
    for line in SRC.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        topic = obj.get("topic") or obj.get("label")
        label, slots, why = label_of(topic, obj["question"])
        rows.append({"question": obj["question"], "topic": topic,
                     "route": label, "why": why, "slots": slots})

    # 补一批「规则已知抽不出」的硬样本，让 extract 类有足够信号
    HARD = [
        ("请假", "我二叔过世了，要请几天", "extract"),
        ("请假", "我岳父住院，得请几天假", "extract"),
        ("年休假", "年假怎么算", "ask"),
        ("住宿", "住宿标准是多少", "ask"),
        ("住宿", "我住的那晚超标了吗", "ask"),
        ("差旅", "出差怎么申请", "code"),
        ("报销", "报销流程是什么", "code"),
        ("合同", "合同要盖章找谁", "code"),
        ("考勤", "忘记打卡怎么办", "code"),
    ]
    for topic, text, gold in HARD:
        label, slots, why = label_of(topic, text)
        rows.append({"question": text, "topic": topic, "route": gold,
                     "why": why, "slots": slots, "hard": True})

    # **系统性扰动**：把规则能抽的数字换成抽不出的说法，逼出 extract 类
    # 例：「我干了 12 年，年假几天」->「我干了一打年，年假几天」（参数还在，规则抽不到）
    PERTURB = [
        (re.compile(r"(\d+)\s*年"), ["一打年", "两位数年头", "十二个年头", "十来年"]),
        (re.compile(r"(\d+)\s*元"), ["四张", "半千", "好几张", "几张"]),
        (re.compile(r"(\d+)\s*天"), ["小十天", "小半周"]),
    ]
    extra = []
    for r in rows[:]:
        if r["route"] != "code" or r.get("hard"):
            continue
        topic = r["topic"]
        rule, need = REQUIRED_WHEN.get(topic, (None, []))
        if not need or not (rule and rule.search(r["question"])):
            continue
        for pattern, repls in PERTURB:
            if not pattern.search(r["question"]):
                continue
            for repl in repls:
                newq = pattern.sub(repl, r["question"], count=1)
                label, slots, why = label_of(topic, newq)
                if label == "extract":
                    extra.append({"question": newq, "topic": topic, "route": "extract",
                                  "why": why, "slots": slots})
    rows.extend(extra)
    print(f"[扰动] 生成 extract 样本 {len(extra)} 条")

    rng = random.Random(42)
    rng.shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "questions.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            r["id"] = f"route-{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cnt = Counter(r["route"] for r in rows)
    print(f"生成 {len(rows)} 条 -> {out}")
    print("路由标签分布：")
    for k in ("code", "ask", "extract", "other"):
        print(f"   {k:<10}{cnt.get(k, 0):>6}  {cnt.get(k, 0)/len(rows):>6.1%}")

    print("\n各标签样例：")
    seen = set()
    for r in rows:
        if r["route"] in seen:
            continue
        seen.add(r["route"])
        print(f"   [{r['route']:<8}] {r['question'][:34]:<36} ({r['why']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
