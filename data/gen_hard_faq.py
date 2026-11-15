"""生成「难版」企业内部问题：问句里**不出现主题词**，只能靠情境推断。

对比「易版」（gen_enterprise_faq.py）：
    易：问「年假有几天」        -> 词面就是标签，TF-IDF 都能 98%
    难：问「我在这行干了 12 年，一年能歇多少天」 -> 没有"年假"二字，要推断

难度从哪来（也是这个数据集的价值）：
    1. 问句里不含主题词
    2. 切分按「情境」分组 —— 测试集的情境训练时完全没见过
    3. 同一情境可以用不同问法（能休几天/最多几天/可以歇多久）

标注规则：每条情境只对应一个主题，人工预先指定，不依赖模型判断。

用法：
    python data/gen_hard_faq.py --n 30000
产出：
    data/raw/enterprise_hard/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

# 问法模板：按「答案类型」分组，避免「能休几天」问到报销头上
ASKS = {
    "days": ["能休几天", "最多几天", "有几天假", "能给几天", "可以歇多久", "一共多少天"],
    "money": ["能报多少", "有没有上限", "怎么算钱", "能给多少", "标准是多少", "按什么比例"],
    "procedure": ["要怎么办", "该找谁", "流程是什么", "要准备哪些材料", "在哪提交",
                  "需要提前说吗", "怎么操作", "走什么手续", "找哪个部门"],
    "eligibility": ["我可以吗", "符合条件吗", "有没有资格", "能不能办", "算不算"],
}

# 情境字典：主题 -> [(不含主题词的情境, 允许的问法类型)]
SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "年休假": [
        ("我在这行干了 12 年", "days"), ("我今年一天都没歇过", "days"),
        ("去年剩了三天没歇完", "days"), ("我下个月想连着休一周", "procedure"),
        ("我换过工作，之前的工龄算吗", "eligibility"), ("公司让我年底前清掉", "procedure"),
        ("我手里活太多走不开", "eligibility"), ("一年最多能攒多少天", "days"),
        ("我打算春节前休掉", "procedure"), ("没歇完的会不会作废", "days"),
    ],
    "请假": [
        ("我明天要陪家人去医院", "procedure"), ("我孩子发烧了", "procedure"),
        ("我下午有点急事", "eligibility"), ("我连着两天没去公司", "procedure"),
        ("我要回老家办点事", "procedure"), ("我牙疼要去看医生", "procedure"),
        ("我上午晚点到", "procedure"), ("我要照顾生病的老人", "procedure"),
        ("我房子过户得去一趟", "procedure"), ("我感冒了不想传染同事", "eligibility"),
    ],
    "婚假": [
        ("我下个月办婚礼", "days"), ("我准备领证了", "days"),
        ("我婚礼在外地办", "procedure"), ("我对象户口在老家", "procedure"),
        ("我们打算旅行结婚", "eligibility"), ("婚礼在下周，来不及提前说", "procedure"),
        ("我父母要过来", "procedure"), ("办完还要回门", "days"),
    ],
    "产假与陪产假": [
        ("我爱人预产期在三月", "days"), ("我老婆刚生完", "days"),
        ("我想在孩子出生后请一段", "eligibility"), ("我老婆要剖腹产", "days"),
        ("孩子早产了", "procedure"), ("我这边需要陪一段时间", "days"),
        ("我老婆在老家待产", "procedure"), ("生完还要复查几次", "days"),
    ],
    "丧假": [
        ("我爷爷过世了", "days"), ("家里老人走了，我要回趟老家", "procedure"),
        ("我外婆去世，在外省", "days"), ("我要去参加葬礼", "procedure"),
        ("是我爱人的奶奶", "eligibility"), ("路上要两天", "days"),
    ],
    "加班": [
        ("昨晚干到十一点才走", "money"), ("周末来公司赶进度", "money"),
        ("这个月加了好几次", "money"), ("领导让我留下来", "procedure"),
        ("我在家晚上也在处理工作", "money"), ("节假日被叫去值班", "money"),
        ("干的能换成调休吗", "procedure"), ("我干了六小时算不算", "money"),
        ("上线那周天天到半夜", "money"), ("早上五点起来处理故障", "money"),
    ],
    "考勤": [
        ("今早地铁故障，我晚了半小时", "procedure"), ("我忘了按指纹", "procedure"),
        ("我最近老是被记迟到", "procedure"), ("我在外面跑客户，没打卡", "procedure"),
        ("我手机没电，没打上卡", "procedure"), ("我出差那几天怎么算", "procedure"),
        ("系统显示我缺卡", "procedure"), ("我早退了十分钟", "procedure"),
        ("我在家办公怎么记", "procedure"), ("公司闸机刷不上", "procedure"),
    ],
    "差旅": [
        ("下周要去广州三天", "procedure"), ("我明天飞上海", "procedure"),
        ("这趟要待一周", "procedure"), ("老板临时让我出差", "eligibility"),
        ("我要去两个城市", "procedure"), ("我提前一天过去", "procedure"),
        ("客户那边催得急", "procedure"), ("我这次要带同事一起", "procedure"),
        ("行程改了一次", "procedure"), ("要在外地过周末", "procedure"),
    ],
    "住宿": [
        ("去上海住一晚要四百多", "money"), ("我住的酒店超标了", "money"),
        ("出差那晚我住朋友家", "money"), ("对方安排好了房间", "money"),
        ("我订的是标间", "money"), ("会议酒店太贵", "eligibility"),
        ("我多住了一晚自费", "money"), ("两个人住一间怎么算", "money"),
    ],
    "交通": [
        ("昨天打车去客户那边", "money"), ("我坐高铁去的", "money"),
        ("回来叫了个网约车", "money"), ("机场大巴的票还在", "money"),
        ("我开车去的，油费怎么算", "money"), ("我改签了一次", "money"),
        ("我从机场打车回家", "money"), ("地铁票能一起交吗", "money"),
    ],
    "报销": [
        ("我攒了两个月发票", "procedure"), ("上个月的单子还没交", "procedure"),
        ("财务说我贴错了", "procedure"), ("这个月花了不少", "procedure"),
        ("发票抬头开错了", "procedure"), ("电子票怎么交", "procedure"),
        ("我垫的钱什么时候到", "money"), ("有些单子过期了吗", "procedure"),
        ("发票丢了怎么办", "procedure"), ("一张单子超了五千", "procedure"),
    ],
    "补贴": [
        ("上个月在外面跑了十几天", "money"), ("夏天车间太热", "money"),
        ("我一直在外地驻场", "money"), ("我上夜班", "money"),
        ("我住的地方离公司很远", "money"), ("我在高原项目上", "money"),
    ],
    "合同": [
        ("有个合同要盖章", "procedure"), ("对方催着签", "procedure"),
        ("这份要法务看吗", "eligibility"), ("金额不大能不能简化", "eligibility"),
        ("我要用公章", "procedure"), ("合同要寄给对方", "procedure"),
        ("我签了但没盖章", "procedure"), ("对方改了条款", "procedure"),
    ],
    "保密": [
        ("我离职后还能提公司的项目吗", "eligibility"), ("客户资料我能带走吗", "eligibility"),
        ("有人问我们内部数据", "procedure"), ("我在外面讲课能举例吗", "eligibility"),
        ("我不小心发错群了", "procedure"), ("合作方要我们的名单", "procedure"),
    ],
    "离职": [
        ("我打算月底走", "procedure"), ("我提了申请", "procedure"),
        ("我还有年假没休", "money"), ("我想尽快办完手续", "procedure"),
        ("我下家催我入职", "procedure"), ("我还在试用期", "procedure"),
        ("电脑要还给谁", "procedure"), ("我走之前要交接给谁", "procedure"),
        ("社保怎么停", "procedure"), ("我要开个证明给下家", "procedure"),
    ],
    "入职": [
        ("我下周一报到", "procedure"), ("我还没拿到工牌", "procedure"),
        ("我试用期快满了", "eligibility"), ("我要带什么来", "procedure"),
        ("我档案还在学校", "procedure"), ("我体检报告没交", "procedure"),
        ("我银行卡还没登记", "procedure"), ("我工位还没分", "procedure"),
    ],
    "调岗": [
        ("部门想让我换个组", "eligibility"), ("我想申请去别的团队", "procedure"),
        ("调过去之后工资怎么算", "money"), ("我不想去", "eligibility"),
        ("新岗位要重新试用吗", "eligibility"), ("我领导同意了", "procedure"),
        ("换城市的话怎么算", "money"),
    ],
    "绩效": [
        ("这次考核结果我不服", "procedure"), ("我的评分比同事低", "procedure"),
        ("考核影响年终奖吗", "money"), ("目标是谁定的", "procedure"),
        ("我上半年休了长假", "eligibility"), ("我想申诉", "procedure"),
        ("结果什么时候公布", "procedure"), ("我能看到别人的吗", "eligibility"),
    ],
    "培训": [
        ("我想报个外面的课", "money"), ("公司出钱让我去学", "procedure"),
        ("培训要签协议吗", "eligibility"), ("周末的课算加班吗", "money"),
        ("我考证的费用能报吗", "money"), ("学完要考试", "procedure"),
    ],
    "办公用品": [
        ("我们组没打印纸了", "procedure"), ("我想领几支笔", "procedure"),
        ("我们组的订书机找不到了", "procedure"), ("文件夹用完了", "procedure"),
        ("我想申请个白板", "procedure"), ("东西要送到工位吗", "procedure"),
    ],
    "设备": [
        ("我电脑开不了机", "procedure"), ("显示器坏了", "procedure"),
        ("我想换台新的", "eligibility"), ("键盘进水了", "procedure"),
        ("我要外接一个屏幕", "procedure"), ("设备要登记吗", "procedure"),
        ("机器太卡了影响干活", "procedure"),
    ],
    "工作时间": [
        ("我们几点上班", "procedure"), ("中午休息多久", "procedure"),
        ("周末要上班吗", "eligibility"), ("可以弹性一点吗", "eligibility"),
        ("周五能提前走吗", "eligibility"), ("外面下大雨能不能晚到", "procedure"),
    ],
    "办公秩序": [
        ("会议室能吃东西吗", "eligibility"), ("工位能放加湿器吗", "eligibility"),
        ("办公室能抽烟吗", "eligibility"), ("能在工位上午睡吗", "eligibility"),
        ("可以外放音乐吗", "eligibility"), ("带宠物来行不行", "eligibility"),
    ],
    "利益冲突": [
        ("我亲戚开了家供应商", "eligibility"), ("我在外面接了个私活", "eligibility"),
        ("朋友公司想跟我们合作", "eligibility"), ("我买了点竞品的股票", "eligibility"),
        ("我兼职做咨询", "eligibility"), ("我老婆在做同类生意", "eligibility"),
    ],
    "客户接待": [
        ("客户来公司要请吃饭", "money"), ("下个月有客户来参观", "procedure"),
        ("要接待一批外地来的客户", "money"), ("客户要住一晚", "money"),
        ("要送点纪念品", "money"), ("客户一行五个人", "money"),
    ],
}

OOS = ["社保基数怎么调", "公积金能提多少", "期权怎么行权", "裁员赔偿怎么算",
       "公司什么时候上市", "能不能远程办公", "停车位怎么申请", "食堂几点开饭",
       "体检什么时候安排", "团建经费怎么用", "今年年终奖发多少", "个税专项扣除怎么填",
       "公司有没有班车", "能不能带娃来上班"]
CHITCHAT = ["你好", "在吗", "有人吗", "你是谁", "谢谢", "好的谢谢", "再见", "帮我转人工"]
OPENERS = ["", "", "", "", "请问", "麻烦问下", "想问一下", "咨询一下"]
ENDINGS = ["？", "呢？", "", "啊？", "呀？", "嘛？"]
TYPO = {"报销": "报消", "申请": "申清", "材料": "材枓"}

# 情境层的同义替换：只换**主题中立**的词。
# 教训：之前把「加班」换成「留下来」、「发票」换成「单据」，这些替换词本身是
#       另一个主题的强信号 —— 等于给样本换了标签，模型"学错"是我们造成的。
SYNONYMS = [
    ("过世", "去世"), ("爷爷", "祖父"), ("外婆", "外祖母"), ("老婆", "爱人"),
    ("电脑", "笔记本"), ("显示器", "屏幕"), ("开不了机", "启动不了"),
    ("坏了", "故障了"), ("发烧", "发热"), ("医院", "诊所"), ("老家", "家乡"),
]


def apply_synonyms(text: str, rng: random.Random) -> str:
    if rng.random() > 0.35:
        return text
    pairs = [(a, b) for a, b in SYNONYMS if a in text]
    if not pairs:
        return text
    a, b = rng.choice(pairs)
    return text.replace(a, b, 1)


def humanize(text: str, rng: random.Random) -> str:
    out = rng.choice(OPENERS) + text + rng.choice(ENDINGS)
    if rng.random() < 0.08:
        for right, wrong in TYPO.items():
            if right != wrong and right in out and rng.random() < 0.5:
                out = out.replace(right, wrong, 1)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    topics = list(SCENARIOS)
    n_scenarios = sum(len(v) for v in SCENARIOS.values())
    n_oos = int(args.n * 0.08)
    n_chit = int(args.n * 0.04)
    n_in = args.n - n_oos - n_chit

    rows: list[dict] = []
    for _ in range(n_in):
        topic = rng.choice(topics)
        scenario, kind = rng.choice(SCENARIOS[topic])
        scenario_v = apply_synonyms(scenario, rng)
        q = f"{scenario_v}，{rng.choice(ASKS[kind])}"
        # 不再生成复合问句：一句话两个主题却只标一个，是标签歧义，
        # 会让"错误"里混进大量本来就没有唯一答案的样本。
        rows.append({"question": humanize(q, rng), "topic": topic, "label": "in_scope",
                     "scenario": scenario, "difficulty": "hard", "synthetic": True})

    for _ in range(n_oos):
        rows.append({"question": humanize(rng.choice(OOS), rng), "topic": "out_of_scope",
                     "label": "out_of_scope", "difficulty": "hard", "synthetic": True})
    for _ in range(n_chit):
        rows.append({"question": rng.choice(CHITCHAT) + rng.choice(["", "？", "。", "！"]),
                     "topic": "chitchat", "label": "chitchat", "difficulty": "hard",
                     "synthetic": True})

    rng.shuffle(rows)
    out_dir = PROJ / "data" / "raw" / "enterprise_hard"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "questions.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            r["id"] = f"hard-{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["question"]) for r in rows]
    topics_count = Counter(r["topic"] for r in rows)
    stats = {
        "total": len(rows),
        "topics": len(topics),
        "scenarios": n_scenarios,
        "by_topic": dict(topics_count.most_common()),
        "length": {"mean": round(statistics.mean(lens), 1), "median": statistics.median(lens),
                   "p90": sorted(lens)[int(len(lens) * 0.9)], "max": max(lens)},
        "note": "问句不含主题词；切分需按 scenario 分组",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    print(f"生成 {len(rows)} 条（难版）-> {path}")
    print(f"主题 {len(topics)} 个，情境 {n_scenarios} 个")
    print(f"长度：平均 {stats['length']['mean']} 字，中位 {stats['length']['median']}，"
          f"P90 {stats['length']['p90']}")
    print("\n样例：")
    for r in rows[:8]:
        print(f"   [{r['topic']:<10}] {r['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
