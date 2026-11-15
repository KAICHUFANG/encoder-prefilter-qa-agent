"""生成「企业内部问题」语料（离线、模板组合，不调大模型）。

需求：企业内部问答没有公开数据集，但我们要一份"像员工真的会问"的语料。
做法：拿制度条款当知识源，按六维模板 + 真实语料的措辞习惯组合生成。

参考的三份真实语料（用来对齐"人怎么说话"）：
    CrossWOZ    平均 27.3 字，口语、常带"帮我/我想/呢/啊"
    MASSIVE     平均 10.5 字，命令式、"设个/放首/查下"
    KUAKE-QIC   平均 16.2 字，疑问式、"…吗/怎么/多少"

四个标签（对应之前设计的分流策略）：
    ask_clause    查条款（事实/数字/标准）      -> kb
    ask_process   问流程（怎么做/找谁/要什么）   -> kb（模板）
    out_of_scope  制度里没写的                  -> llm / 拒答
    chitchat      打招呼/试探                    -> template

用法：
    python data/gen_enterprise_faq.py --n 3000
产出：
    data/raw/enterprise_faq/questions.jsonl
    data/raw/enterprise_faq/stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

# 制度条款来源：优先用制度问答项目里的真实语料
CORPUS_DIRS = [
    Path(r"C:\Users\xiong\Desktop\制度问答Agent\corpus"),
    PROJ / "data" / "raw" / "enterprise_corpus",
]

# ---------------- 措辞素材（参考三份真实语料的说法）----------------
OPENERS = ["", "", "", "请问", "麻烦问下", "想问一下", "我想问", "咨询一下", "在吗，"] + \
          ["你好，", "打扰一下，", "", ""]
ENDINGS = ["？", "呢？", "啊？", "呀？", "", "嘛？"]
FILLERS_MID = ["", "", "大概", "一般来说", "具体", "现在", "目前"]

# 手写问句库：每个主题 5 条「查条款」+ 4 条「问流程」。
# 为什么手写：模板拼出来的「考勤是多少」「差旅有几天」语义不通。
# 手写 + 口语化变体 + 情境前缀，才能接近真人提问。
KEYWORD_QUESTIONS: dict[str, dict[str, list[str]]] = {
    "年休假": {
        "ask_clause": ["年假有几天", "工作12年了年假多少天", "年假能跨年吗", "没休完的年假怎么折算", "试用期有年假吗"],
        "ask_process": ["年假怎么申请", "年假找谁批", "年假要提前几天提", "年假在考勤系统哪里提交"],
    },
    "请假": {
        "ask_clause": ["事假有几天", "病假要什么证明", "请假扣不扣工资", "请假会影响全勤吗"],
        "ask_process": ["请假流程是什么", "请假找谁审批", "请假在哪个系统提", "请假要提前多久"],
    },
    "加班": {
        "ask_clause": ["加班费怎么算", "加班到几点算加班", "周末加班怎么调休", "加班有上限吗"],
        "ask_process": ["加班要申请吗", "加班怎么报备", "加班记录在哪填", "调休找谁批"],
    },
    "考勤": {
        "ask_clause": ["迟到几次算旷工", "忘记打卡怎么办", "外勤怎么打卡", "考勤周期是到几号"],
        "ask_process": ["补卡怎么申请", "考勤异常找谁处理", "外勤打卡在哪登记"],
    },
    "住宿": {
        "ask_clause": ["出差住宿一晚能报多少", "住宿标准按什么级别分", "住宿超标了怎么办", "住宿费有上限吗"],
        "ask_process": ["住宿费怎么报销", "住宿要保留什么单据", "住宿超标需要谁批"],
    },
    "差旅": {
        "ask_clause": ["出差补贴一天多少", "出差能坐高铁一等座吗", "差旅费包括哪些", "出差天数怎么算"],
        "ask_process": ["出差要提前申请吗", "出差审批走什么流程", "出差回来怎么报销", "出差申请找谁批"],
    },
    "报销": {
        "ask_clause": ["报销有金额上限吗", "发票要什么样的", "报销多久能到账", "哪些费用不能报"],
        "ask_process": ["报销流程怎么走", "报销单找谁签", "报销在哪提交", "报销要附什么材料"],
    },
    "交通": {
        "ask_clause": ["打车费能报吗", "机票能报经济舱吗", "市内交通怎么算", "交通费有没有上限"],
        "ask_process": ["交通费怎么报", "打车要留什么凭证", "交通报销找谁审"],
    },
    "餐补": {
        "ask_clause": ["餐补一天多少", "加班有餐补吗", "出差餐补怎么算", "餐补发到工资里吗"],
        "ask_process": ["餐补怎么申请", "餐补在哪登记", "餐补找谁核实"],
    },
    "补贴": {
        "ask_clause": ["有哪些补贴", "高温补贴发多少", "补贴什么时候发", "补贴要交税吗"],
        "ask_process": ["补贴怎么申领", "补贴找哪个部门", "补贴要什么材料"],
    },
    "采购": {
        "ask_clause": ["采购多少钱以内不用招标", "采购要几家比价", "采购周期多久", "采购有金额分级吗"],
        "ask_process": ["采购怎么申请", "采购单找谁批", "采购流程是什么", "采购在哪提需求"],
    },
    "合同": {
        "ask_clause": ["合同审批要几级", "合同用印要什么材料", "合同金额多少要法务审", "合同能自己签吗"],
        "ask_process": ["合同怎么走审批", "用印申请在哪提", "合同找谁审", "合同归档怎么弄"],
    },
    "保密": {
        "ask_clause": ["保密期限多久", "哪些算商业秘密", "离职后还要保密吗", "违反保密有什么后果"],
        "ask_process": ["保密协议在哪签", "涉密资料怎么处理", "保密事项找谁确认"],
    },
    "离职": {
        "ask_clause": ["离职要提前几天", "试用期离职提前几天", "离职有补偿吗", "年假没休完怎么办"],
        "ask_process": ["离职流程是什么", "离职找谁批", "离职交接要做什么", "离职证明怎么开"],
    },
    "入职": {
        "ask_clause": ["入职要带什么材料", "试用期多久", "试用期工资多少", "转正标准是什么"],
        "ask_process": ["入职手续怎么办", "入职找谁对接", "转正申请怎么提"],
    },
    "调岗": {
        "ask_clause": ["调岗需要本人同意吗", "调岗后工资怎么定", "调岗有试用期吗"],
        "ask_process": ["调岗怎么申请", "调岗找谁批", "调岗流程是什么"],
    },
    "绩效": {
        "ask_clause": ["绩效怎么算", "绩效等级有几档", "绩效和年终奖有关系吗", "绩效不合格会怎样"],
        "ask_process": ["绩效申诉怎么提", "绩效面谈什么时候", "绩效目标找谁定"],
    },
    "培训": {
        "ask_clause": ["培训费用能报吗", "外部培训要签协议吗", "培训期间算工时吗"],
        "ask_process": ["培训怎么申请", "培训预算找谁批", "培训报名在哪"],
    },
    "办公用品": {
        "ask_clause": ["办公用品有额度吗", "哪些用品能领", "领用要登记吗"],
        "ask_process": ["办公用品怎么领", "领用找谁", "办公用品在哪申请"],
    },
    "设备": {
        "ask_clause": ["电脑多久能换新", "设备损坏要赔吗", "离职设备要还吗"],
        "ask_process": ["设备申请怎么提", "设备领用找谁", "设备维修在哪报"],
    },
}
# 为什么带第二个字段：不加约束会生成「差旅有几天」「考勤按什么比例」这种语义错配的问句。
DIMENSIONS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "ask_clause": [
        ("{kw}是多少", ("元", "金额", "标准", "上限", "天", "比例", "百分")),
        ("{kw}怎么算", ("计算", "折算", "算", "比例", "百分")),
        ("{kw}的标准是什么", ("标准", "元", "金额", "上限", "比例")),
        ("{kw}有几天", ("天", "日", "假期")),
        ("{kw}能报多少", ("报销", "报", "元", "金额", "上限")),
        ("{kw}是什么规定", ()),
        ("第{num}条讲的什么", ()),
        ("{kw}的范围包括哪些", ("包括", "范围", "以下", "情形")),
        ("{kw}上限多少", ("上限", "不得超过", "元", "金额")),
        ("{kw}按什么比例", ("比例", "百分", "％", "%")),
        ("{kw}有没有明确数字", ("元", "天", "比例", "百分", "金额")),
    ],
    "ask_process": [
        ("{kw}怎么申请", ("申请", "提交", "报批", "提出")),
        ("{kw}的流程是什么", ("流程", "程序", "步骤", "审批")),
        ("{kw}找谁批", ("审批", "批准", "负责人", "主管", "上级")),
        ("{kw}要提交什么材料", ("材料", "证明", "单据", "发票", "凭证")),
        ("{kw}在哪个系统办", ("系统", "平台", "线上", "OA")),
        ("{kw}需要提前多久", ("提前", "工作日", "日内")),
        ("{kw}走什么审批", ("审批", "批准", "签")),
        ("{kw}怎么操作", ("操作", "办理", "申请", "提交")),
        ("{kw}要填什么表", ("表", "单据", "模板")),
        ("{kw}需要哪些证明", ("证明", "材料", "凭证", "发票")),
        ("{kw}找哪个部门", ("部门", "人事", "行政", "财务", "负责人")),
    ],
}

# 从条款标题/正文里抽「主题词」用的词表（制度里的高频对象）
KW_PATTERNS = [
    ("年休假", ["年假", "年休假"]), ("请假", ["请假", "事假", "病假"]), ("加班", ["加班"]),
    ("考勤", ["打卡", "考勤", "迟到"]), ("住宿", ["住宿", "酒店", "房费"]),
    ("差旅", ["出差", "差旅"]), ("报销", ["报销", "发票"]), ("交通", ["交通", "打车", "高铁"]),
    ("餐补", ["餐补", "伙食"]), ("补贴", ["补贴", "补助"]), ("采购", ["采购", "申购"]),
    ("合同", ["合同", "用印"]), ("保密", ["保密", "竞业"]), ("离职", ["离职", "辞职"]),
    ("入职", ["入职", "试用期", "转正"]), ("调岗", ["调岗", "调薪"]), ("绩效", ["绩效", "考核"]),
    ("培训", ["培训"]), ("办公用品", ["办公用品", "领用"]), ("设备", ["电脑", "设备", "资产"]),
]

# 制度里没有的话题 -> out_of_scope（真实员工会问，但公司制度不管）
OOS = [
    ("期权怎么行权", "股权期权"), ("股票什么时候能卖", "股权期权"),
    ("裁员赔偿怎么算", "劳动仲裁"), ("我被辞退有赔偿吗", "劳动仲裁"),
    ("社保基数怎么调", "社保"), ("公积金能提多少", "公积金"),
    ("个税专项扣除怎么填", "个税"), ("婚假多少天", "假期"),
    ("产假能休多久", "假期"), ("陪产假有几天", "假期"),
    ("公司什么时候上市", "公司信息"), ("今年年终奖发多少", "薪酬"),
    ("能不能远程办公", "办公方式"), ("可以带宠物上班吗", "办公环境"),
    ("停车位怎么申请", "后勤"), ("食堂什么时候开饭", "后勤"),
    ("体检什么时候安排", "福利"), ("团建经费怎么用", "福利"),
    ("能不能内部转岗到算法组", "职业发展"), ("晋升答辩什么时候", "职业发展"),
    ("工资什么时候发", "薪酬"), ("加班费怎么算的", "薪酬"),
    ("五险一金交的比例", "社保"), ("离职证明怎么开", "离职"),
]

CHITCHAT = ["你好", "在吗", "有人吗", "你是谁", "这是机器人吗", "谢谢", "好的谢谢",
            "再见", "帮我转人工", "你叫什么名字", "在不在", "嗯嗯"]

# 情境前缀：真实员工提问很少干巴巴一句「报销要什么材料」，
# 往往带一个具体处境。加了之后长度和自然度都更接近 CrossWOZ。
# 情境前缀：必须和条款主题匹配，否则会出现「我换了工位，那离职是什么标准」这种错配。
SITUATIONS: dict[str, list[str]] = {
    "出差": ["我上个月出差回来", "我下周要去上海出差", "我这次出差住得比较贵"],
    "差旅": ["我上个月出差回来", "我下周要去上海出差"],
    "住宿": ["我上周出差住了一晚", "我周六要过去住一晚"],
    "报销": ["我上个月出差回来", "我攒了一堆发票", "我在系统里提交了报销"],
    "请假": ["我这个月请假有点多", "我下个月要结婚", "我家里有点事"],
    "年休假": ["我今年还没休年假", "我打算下个月休年假", "我去年剩了几天年假"],
    "加班": ["我昨天加班到十点", "我这个月加了好几次班"],
    "考勤": ["我最近老是迟到", "我上周忘打卡了"],
    "入职": ["我刚入职还没转正", "我下周一入职"],
    "离职": ["我准备离职", "我提了离职申请"],
    "采购": ["我们组想买台设备", "我要采购一批办公用品"],
    "合同": ["有个合同要走流程", "我要盖章"],
    "保密": ["我离职后还要保密吗", "我想问下竞业的事"],
    "绩效": ["我这次考核结果不太好", "我们组开始评绩效了"],
    "培训": ["我想报个外部培训", "公司有培训预算吗"],
    "办公用品": ["我买了点办公用品", "我们组缺打印纸了"],
    "设备": ["我电脑坏了想换一台", "我领的设备要还吗"],
    "调岗": ["我想申请调岗", "我们组要调整"],
    "补贴": ["我上个月有几天在外面跑", "夏天太热了"],
    "餐补": ["我加班到很晚", "我中午在外面吃"],
    "交通": ["我打车去的客户那边", "我坐高铁去的"],
}
SITUATION_TAIL = ["，", "，那", "，请问", "，想问下"]

TYPO_MAP = {"报销": "报消", "制度": "制渡", "年假": "年休", "申请": "申清",
            "标准": "标谁", "报销单": "报消单"}


def load_clauses() -> list[dict]:
    """从制度问答项目的 corpus/*.md 里切出条款。"""
    for d in CORPUS_DIRS:
        if not d.exists():
            continue
        clauses = []
        for md in sorted(d.glob("*.md")):
            text = md.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"^###\s*(第(\d+)条)\s*(.*)$", text, re.M):
                start = m.end()
                nxt = text.find("\n### ", start)
                body = text[start: nxt if nxt > 0 else len(text)].strip()
                clauses.append({"doc": md.stem, "number": f"第{m.group(2)}条",
                                "heading": m.group(3).strip(), "text": body})
        if clauses:
            print(f"[语料] {d} -> {len(clauses)} 条条款")
            return clauses
    raise SystemExit("找不到制度语料，请检查 CORPUS_DIRS")


def keywords_of(clause: dict) -> list[str]:
    blob = clause["heading"] + " " + clause["text"]
    kws = []
    for canonical, aliases in KW_PATTERNS:
        if any(a in blob for a in aliases):
            kws.append(canonical)
    if not kws and clause["heading"]:
        kws.append(clause["heading"][:6])
    return kws or ["这个"]


def humanize(text: str, rng: random.Random, allow_opener: bool = True) -> str:
    """加口语成分：开头语气词、结尾助词、偶发错别字。

    短句不加长前缀（"想问一下你好呀？"就是拼接痕迹），闲聊单独处理。
    """
    opener = rng.choice(OPENERS) if (allow_opener and len(text) > 6) else ""
    ending = rng.choice(ENDINGS)
    out = opener + text + ending
    if rng.random() < 0.06:
        for wrong, right in TYPO_MAP.items():
            if wrong in out and rng.random() < 0.5:
                out = out.replace(wrong, right, 1)
                break
    return out


def pick_template(clause: dict, label: str, rng: random.Random) -> str | None:
    """按条款内容挑一个语义上说得通的模板；挑不出返回 None。"""
    blob = clause["heading"] + " " + clause["text"]
    usable = [tpl for tpl, reqs in DIMENSIONS[label]
              if not reqs or any(r in blob for r in reqs)]
    return rng.choice(usable) if usable else None


def pick_question(clause: dict, label: str, kw: str, rng: random.Random) -> str:
    """优先从手写问句库里取；没有再退回模板。"""
    bank = KEYWORD_QUESTIONS.get(kw, {}).get(label, [])
    if bank:
        return rng.choice(bank)
    tpl = pick_template(clause, label, rng)
    if tpl is None:                       # 模板全都不匹配：用万能的
        tpl = "{kw}是什么规定" if label == "ask_clause" else "{kw}怎么操作"
    return tpl.format(kw=kw, num=clause["number"].replace("第", "").replace("条", ""))


def raw_core(clause: dict, label: str, rng: random.Random) -> str:
    """只出核心问句，不加口语成分（用于复合句的第二问）。"""
    kw = rng.choice(keywords_of(clause))
    return pick_question(clause, label, kw, rng)


def gen_question(clause: dict, label: str, rng: random.Random,
                 with_situation: bool = False) -> tuple[str, str]:
    """返回 (问句, 主题词)。主题词作为细粒度标签（20 类），比 4 个粗动作难得多。"""
    kws = keywords_of(clause)
    kw = rng.choice(kws)
    core = pick_question(clause, label, kw, rng)
    if with_situation:
        pool = SITUATIONS.get(kw, [])      # 只取和这个词匹配的情境
        if pool:
            core = rng.choice(pool) + rng.choice(SITUATION_TAIL) + core
        else:
            with_situation = False
    return humanize(core, rng, allow_opener=not with_situation), kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="总条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clauses = load_clauses()

    # 配比：查条款 40%、问流程 30%、超出范围 20%、闲聊 10%
    quotas = {
        "ask_clause": int(args.n * 0.40),
        "ask_process": int(args.n * 0.30),
        "out_of_scope": int(args.n * 0.20),
        "chitchat": args.n - int(args.n * 0.40) - int(args.n * 0.30) - int(args.n * 0.20),
    }

    rows: list[dict] = []
    for label in ("ask_clause", "ask_process"):
        for _ in range(quotas[label]):
            clause = rng.choice(clauses)
            q, topic = gen_question(clause, label, rng, with_situation=(rng.random() < 0.35))
            # 15% 生成复合问句（真实员工经常一句话问两件事）
            if rng.random() < 0.15:
                extra = raw_core(rng.choice(clauses), "ask_clause", rng)
                q = q.rstrip("？?") + "，" + rng.choice(["另外", "还有", "顺带问下"]) + extra + "？"
            rows.append({"question": q, "label": label, "topic": topic,
                         "source_clause": f"{clause['doc']} {clause['number']}",
                         "synthetic": True})

    for _ in range(quotas["out_of_scope"]):
        q, topic = rng.choice(OOS)
        rows.append({"question": humanize(q, rng), "label": "out_of_scope",
                     "topic": topic, "synthetic": True})

    for _ in range(quotas["chitchat"]):
        base = rng.choice(CHITCHAT)
        # 打招呼/试探才带问号；道谢/告别带句号或不带
        if base in ("你好", "在吗", "有人吗", "你是谁", "这是机器人吗", "你叫什么名字", "在不在"):
            punctuation = rng.choice(["？", "?", "？", ""])
        else:
            punctuation = rng.choice(["", "。", "！", "~"])
        rows.append({"question": base + punctuation, "label": "chitchat", "synthetic": True})

    rng.shuffle(rows)
    out_dir = PROJ / "data" / "raw" / "enterprise_faq"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "questions.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            r["id"] = f"efaq-{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["question"]) for r in rows]
    stats = {
        "total": len(rows),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "length": {"min": min(lens), "max": max(lens),
                   "mean": round(statistics.mean(lens), 1),
                   "median": statistics.median(lens),
                   "p90": sorted(lens)[int(len(lens) * 0.9)]},
        "reference": {"CrossWOZ_mean": 27.3, "MASSIVE_mean": 10.5, "KUAKE-QIC_mean": 16.2},
        "with_typo": sum(1 for r in rows if any(w in r["question"] for w in TYPO_MAP.values())),
        "composite": sum(1 for r in rows if "另外" in r["question"] or "还有" in r["question"]
                         or "顺带" in r["question"]),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    print(f"\n生成 {len(rows)} 条 -> {out}")
    print("标签分布:", stats["by_label"])
    print(f"长度：平均 {stats['length']['mean']} 字（中位 {stats['length']['median']}，"
          f"P90 {stats['length']['p90']}）")
    print(f"参考：CrossWOZ 27.3 / KUAKE-QIC 16.2 / MASSIVE 10.5")
    print(f"含错别字 {stats['with_typo']} 条，复合问句 {stats['composite']} 条")
    print("\n各类样例：")
    seen = set()
    for r in rows:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        print(f"   [{r['label']:<12}] {r['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
