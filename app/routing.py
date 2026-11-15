"""动作策略：意图 -> 谁来干活。

**意图本身不产生价值，「分流」才产生价值。**

    用户问句
      -> 意图分类（本地模型，十几毫秒，0 token）
      -> 动作策略（本文件）
           kb       查知识库/数据库  -> 0 token，毫秒级
           template 固定话术         -> 0 token，毫秒级
           llm      调大模型         -> 花钱、1 秒级
           human    转人工           -> 最贵，只留给必须由人负责的事

策略按数据集分开（`config.DATASET` 切换）——因为这正是本项目的核心论点：
**同一个分类器框架，标签背后有没有执行者，决定了分流率。**

    kuake（医疗搜索问句）：11 个标签都是「主题」，背后没有执行者 -> 分流率约 19%
    crosswoz（任务型对话）：7 个标签里 5 个是「领域」，每个领域背后有真实的属性库 -> 分流率约 95%

三条设计原则（被质疑过，改出来的）：

1) 本地模型不确定，不等于要转人工。低置信度只打 needs_review 标，仍交给模型兜底，
   供人事后抽检。把「分类器不确定」丢给最贵的人力，是设计错误。
2) 前置分流不该产生 human。human 的触发条件是「内容有责任风险」（投诉、赔付、纠纷），
   属于后置判定，不是主题分类能决定的。
3) 动作按「答案是不是确定的」来分，不按「话题重不重要」来分。
"""
from __future__ import annotations

import config

# 低于这个置信度：仍然走策略，但打 needs_review 标，供人工抽检
REVIEW_FLOOR = 0.55

# 这些意图必须由人负责（投诉、赔付、纠纷…）。两个数据集都没有这类标签，先留空。
HUMAN_INTENTS: set[str] = set()

ACTIONS = ("kb", "skill", "template", "llm", "human")

# ---------------- 数据集 A：KUAKE-QIC（医疗搜索问句，主题分类）----------------
# 特点：11 个标签描述「问的是哪方面」，背后没有可直接执行的库 ->
#       只有「指标区间/价目/说明书」这几类勉强能查，其余必须生成。
KUAKE_POLICY: dict[str, dict[str, str]] = {
    "指标解读": {"action": "kb", "why": "检验指标有参考区间表，查表即可"},
    "功效作用": {"action": "kb", "why": "药品/项目功效是固定说明"},
    "医疗费用": {"action": "kb", "why": "价目表固定"},
    "就医建议": {"action": "template", "why": "科室指引、挂号流程是固定话术"},
    "注意事项": {"action": "template", "why": "通用注意事项可模板化（仍需免责）"},
    "治疗方案": {"action": "llm", "why": "需按个体情况组织语言"},
    "病情诊断": {"action": "llm", "why": "需理解描述，且必须加免责"},
    "病因分析": {"action": "llm", "why": "需推理与解释"},
    "疾病表述": {"action": "llm", "why": "需理解患者自述"},
    "后果表述": {"action": "llm", "why": "预后必须条件化表述"},
    "其他": {"action": "llm", "why": "内容泛，用模型兜底"},
}

# ---------------- 数据集 B：CrossWOZ（任务型对话，领域 + 动作）----------------
# 特点：5 个领域背后各有一份真实属性库（名称/门票/人均消费/评分/出发地…），
#       「分到哪个领域」就等于「知道该查哪个库」。
CROSSWOZ_POLICY: dict[str, dict[str, str]] = {
    "餐馆": {"action": "kb", "why": "餐馆库：人均消费/评分/推荐菜/营业时间"},
    "酒店": {"action": "kb", "why": "酒店库：类型/设施/价格/评分"},
    "景点": {"action": "kb", "why": "景点库：门票/游玩时间/评分/周边"},
    "地铁": {"action": "kb", "why": "地铁线路库：出发地到目的地"},
    "出租": {"action": "kb", "why": "出租计价/路线：出发地到目的地"},
    "General": {"action": "template", "why": "问候/道谢/告别，固定话术即可"},
    "多域": {"action": "llm", "why": "一句话涉及多个领域，单标签分类器拆不开，交给模型"},
}

# ---------------- 数据集 C：MASSIVE（18 个语音助手场景）----------------
# 特点：绝大多数场景是「操作类」——设闹钟、查天气、放音乐、导航、点外卖、加日程，
#       每个都有确定性的技能可执行；只有 qa（开放问答）和 general（闲聊）必须生成。
MASSIVE_POLICY: dict[str, dict[str, str]] = {
    **{k: {"action": "skill", "why": f"{k}：本地技能可直接执行，0 token"}
       for k in ("alarm", "weather", "music", "play", "news", "audio", "transport",
                 "takeaway", "calendar", "email", "social", "lists", "iot",
                 "cooking", "datetime", "recommendation")},
    "qa": {"action": "llm", "why": "开放问答，需生成式回答"},
    "general": {"action": "llm", "why": "闲聊/杂项，需生成式回答"},
}

# ---------------- 数据集 D：企业内部制度问答（25 个主题）----------------
# 特点：每个主题背后都有对应的制度条款（58 条），能直接查条款回答，0 token。
#       out_of_scope（制度没写）才需要模型/拒答；chitchat 走模板。
ENTERPRISE_TOPICS = ("年休假", "请假", "婚假", "产假与陪产假", "丧假", "加班", "考勤",
                     "差旅", "住宿", "交通", "报销", "补贴", "合同", "保密", "离职",
                     "入职", "调岗", "绩效", "培训", "办公用品", "设备", "工作时间",
                     "办公秩序", "利益冲突", "客户接待")
ENTERPRISE_POLICY: dict[str, dict[str, str]] = {
    **{k: {"action": "kb", "why": f"{k}：查对应制度条款，0 token"} for k in ENTERPRISE_TOPICS},
    "out_of_scope": {"action": "llm", "why": "制度里没写，需生成/拒答"},
    "chitchat": {"action": "template", "why": "问候/闲聊，固定话术"},
}

POLICIES: dict[str, dict[str, dict[str, str]]] = {
    "kuake": KUAKE_POLICY,
    "crosswoz": CROSSWOZ_POLICY,
    "massive": MASSIVE_POLICY,
    "enterprise": ENTERPRISE_POLICY,
    "enterprise_hard": ENTERPRISE_POLICY,
}

# 未知意图：用模型兜底，不转人工
DEFAULT: dict[str, str] = {"action": "llm", "why": "未定义意图，用模型兜底"}


def policy() -> dict[str, dict[str, str]]:
    """当前数据集的策略表。数据集名不认识时返回空表（全部走默认兜底）。"""
    return POLICIES.get(config.DATASET, {})


def decide(intent: str, confidence: float, coverage: float | None = None,
           unseen_run: int = 0) -> dict:
    """给定意图、置信度、输入的新鲜度信号，返回该走哪条路。

    coverage（0~1）：这句话的字符二元组有多大比例在训练集出现过。
    unseen_run：最长的一段连续没见过的片段（字符数），用来抓长句里夹的陌生实体。

    任一信号触发，说明输入超出训练分布——这时候**置信度不可信**，
    不管模型多自信都要升级给大模型。
    """
    if intent in HUMAN_INTENTS:
        return {
            "action": "human",
            "needs_llm": False,
            "needs_review": False,
            "unseen": False,
            "reason": "该意图必须由人负责（投诉/赔付类）",
        }

    entry = policy().get(intent, DEFAULT)
    needs_review = confidence < REVIEW_FLOOR

    from app import ood

    unseen = False
    why = []
    if coverage is not None and coverage < ood.COVERAGE_FLOOR:
        unseen = True
        why.append(f"覆盖率 {coverage:.2f} < {ood.COVERAGE_FLOOR}")
    if ood.USE_UNSEEN_RUN and unseen_run >= ood.UNSEEN_RUN_FLOOR:
        unseen = True
        why.append(f"连续未见 {unseen_run} 字 ≥ {ood.UNSEEN_RUN_FLOOR}")

    if unseen:
        return {
            "action": "llm",
            "needs_llm": True,
            "needs_review": True,
            "unseen": True,
            "coverage": None if coverage is None else round(float(coverage), 3),
            "reason": "输入疑似训练集外表达（" + "；".join(why) + "），置信度不可信，交给大模型",
        }

    reason = entry["why"]
    if needs_review:
        reason += f"（置信度 {confidence:.2f} 低于 {REVIEW_FLOOR}，标记待抽检）"

    return {
        "action": entry["action"],
        "needs_llm": entry["action"] == "llm",
        "needs_review": needs_review,
        "unseen": False,
        "coverage": None if coverage is None else round(float(coverage), 3),
        "reason": reason,
    }


def describe() -> dict:
    return {
        "dataset": config.DATASET,
        "review_floor": REVIEW_FLOOR,
        "human_intents": sorted(HUMAN_INTENTS),
        "actions": {
            "kb": "查知识库/数据库，0 token，毫秒级",
            "skill": "调本地技能（设闹钟/查天气/放音乐…），0 token，毫秒级",
            "template": "固定话术填空，0 token，毫秒级",
            "llm": "调大模型，1 秒级，花钱",
            "human": "转人工坐席（只用于必须由人负责的场景，前置分流不产生）",
        },
        "table": [{"intent": name, **entry} for name, entry in policy().items()],
        "default": DEFAULT,
        "note": "置信度低于 review_floor 不改变动作，只标记 needs_review。",
    }
