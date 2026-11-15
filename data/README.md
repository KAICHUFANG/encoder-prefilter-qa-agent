# 数据说明

本项目用**两个数据集**，目的不是"多做一个实验"，而是证明一件事：

> **分流价值由「标签背后有没有执行者」决定，不由分类器精度决定。**

| | CrossWOZ（主） | KUAKE-QIC（对比） |
|---|---|---|
| 类型 | 任务型对话（订餐/订房/查景点/路线） | 医疗搜索问句（主题分类） |
| 标签 | **领域 5 类**（餐馆/酒店/景点/地铁/出租）+ General + 多域 | 11 个主题（治疗方案/病情诊断/医疗费用…） |
| 标签背后有执行者吗 | **有**——每个领域对应一份真实属性库 | **没有**——只能交大模型生成 |
| 预期分流率 | 高（实测约 95%） | 低（实测 19.4%） |
| 许可 | Apache-2.0 | CC-BY-NC 4.0 |

---

## 一、CrossWOZ（主数据集）

### 来源

- HuggingFace 镜像：`GEM/CrossWOZ`（备选 `ConvLab/crosswoz`）
- 原始论文：Zhu et al., *CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset*, TACL 2020
- 规模：6012 个对话、10.2 万句用户话语、5 个领域
- 许可：**Apache-2.0**

### 下载

```powershell
uv run python data/fetch.py --dataset crosswoz
```

会下载 `data.zip`（16 MB）并解压出 `train.json`(390MB) / `val.json`(39MB) / `test.json`(39MB)。
`train.json` 太大，`prepare_crosswoz.py` 用 `ijson` 流式解析，不会吃爆内存。

### 原始结构

顶层是 `{对话ID: 对话对象}`，对话对象里有 `messages`，每条消息：

```json
{
  "role": "usr",
  "content": "营业时间是什么时间？",
  "dialog_act": [["Request", "餐馆", "营业时间", ""]]
}
```

**四元组 = [动作, 领域, 槽位, 值]**。这是本数据集的核心价值：
领域告诉你"该查哪个库"，动作告诉你"要查还是要答"。

### 转成单轮分类数据

```powershell
uv run python train/prepare_crosswoz.py
```

标签设计（7 类）：

| 标签 | 含义 | 动作 |
|---|---|---|
| 餐馆 / 酒店 / 景点 | 单领域查询 | `kb`（查该域属性库） |
| 地铁 / 出租 | 路线查询 | `kb`（路线/计价库） |
| General | 问候/道谢/告别 | `template` |
| **多域** | 一句话涉及 ≥2 个领域 | `llm`（单标签分类器拆不开） |

实测规模：train 20002 / dev 2568 / test 3525；长度 2–89 字，平均 27.3 字。

### 必须写进报告的局限

1. **只做意图，不做槽位**。真实系统里"查库"还需要抽槽位（人均消费<100、附近、评分>4）。
   本项目把槽位抽取当作下游能力，属于**假设**，要在局限性里写明。
2. **多域只占 2.3%**（话语级）。对话级跨域占 35%，但拆成单轮后比例大幅下降。
3. **官方 train.json 只用前 20000 条**（覆盖 2360 个对话），为了 CPU 训练时间可控；
   想跑全量：`uv run python train/prepare_crosswoz.py --max-train 0`。

---

## 二、KUAKE-QIC（对比数据集）

### 来源与获取

- HuggingFace 镜像：`wyp/cblue-qic`（**OpenI 那个下载页已失效**，别再用了）
- 只有官方 dev 划分的 1955 条；11 类；**CC-BY-NC 4.0**

```powershell
uv run python data/fetch.py --dataset kuake
uv run python train/prepare.py --pattern "KUAKE-QIC_dev.json"
```

1955 条按标签分层切成 train 1563 / dev 196 / test 196（种子 42）。
报告里要写明："公开镜像仅含 dev 划分，本研究按标签分层 8:1:1 重切，数字不可与 CBLUE 榜单直接比较。"

### 类别分布（严重不平衡）

| 意图 | train | dev | test | 合计 |
|---|---|---|---|---|
| 治疗方案 | 540 | 68 | 68 | 676 |
| 其他 | 315 | 40 | 40 | 395 |
| 病情诊断 | 230 | 29 | 29 | 288 |
| 疾病表述 | 126 | 16 | 16 | 158 |
| 就医建议 | 108 | 13 | 13 | 134 |
| 注意事项 | 96 | 12 | 12 | 120 |
| 医疗费用 | 40 | 5 | 5 | 50 |
| 后果表述 | 37 | 4 | 4 | 45 |
| 指标解读 | 26 | 3 | 3 | 32 |
| 病因分析 | 23 | 3 | 3 | 29 |
| 功效作用 | 22 | 3 | 3 | 28 |

最大类 676、最小类 28，**差 24 倍** → 必须看 macro-F1。

---

## 三、统一格式

`prepare*.py` 输出 JSONL（一行一个 JSON，UTF-8 无 BOM）：

```json
{"id": "train-00001", "text": "帮我推荐一个人均50的餐馆", "intent": "餐馆", "act": "Request", "dialog_id": "2303", "split": "train"}
```

| 字段 | 说明 |
|---|---|
| `id` | `split-序号`，全局唯一 |
| `text` | 原始话语，**不清洗**（口语、错别字都是真实特征） |
| `intent` | 标签原文 |
| `act` | CrossWOZ 的对话动作（Request/Inform/General…），只做分析，不参与训练 |
| `split` | `train` / `dev` / `test` |

`data/<dataset>/labels.json` 由 prepare 从数据统计生成——**不硬编码**，避免代码和数据对不上。

---

## 四、切换数据集

```powershell
$env:INTENT_DATASET = "crosswoz"    # 默认
$env:INTENT_DATASET = "kuake"       # 对比用
```

两套数据/模型/报告互不覆盖：

```
data/<dataset>/*.jsonl        models/<dataset>/        eval/<dataset>/report*.json
```
