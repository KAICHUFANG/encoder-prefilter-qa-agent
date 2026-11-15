# 客服意图识别 Agent · encoder-only 预筛

用 encoder-only 模型（MacBERT）判断：**用户这句话该交给确定性代码，还是交给大模型**。
目标是让问答型 Agent 在“能查库”的请求上做到 0 token，同时把错误答案率压到可控范围。

> 一句话定位：**这是一个用 encoder-only 模型做问答 Agent 初步筛查（pre-filter / router）的工程实验。**
> 模型不负责回答，只负责分流；回答由执行器（查库）或大模型完成。

---

## 1. 为什么需要预筛

客服/问答场景里，大量请求其实是**确定性查询**：

- “这家酒店的电话是多少” → 查字段
- “评分 4.5 以上、有早餐的酒店” → 按条件筛选
- “它周边有什么景点” → 查关系表

这类请求交给大模型，既慢又贵，还可能编造。
但也不是所有请求都能用代码处理——问到库外信息、需要比较解释、多意图混合时，必须交给模型。

**所以核心问题变成：怎么判断哪条走哪边。** 本项目就是在这个问题上做实验。

---

## 2. 架构

```
用户请求
   │
   ├─ encoder-only 门控（4 × MacBERT-large 集成）
   │     ├─ 4/4 一致且 P(代码) ≥ 0.50  → 执行器（查库，0 token）
   │     └─ 否则                        → 大模型
   │
   └─ 执行器：官方商户库 2,549 家
        字段查询 / 区间筛选 / 设施匹配 / 周边关系表 / 澄清反问
```

门控不做回答，只做分流；判错的代价是“错答案发出去”或“多花 token”。

---

## 3. 数据与标注

| 项 | 数值 |
|---|---|
| 评测问句 | **3,525 条**（CrossWOZ test，已去重/过滤） |
| 有效标注 | **3,522 条**（DeepSeek agent 判官逐条核验） |
| 划分（按对话切） | train 2,469 / dev 534 / test 519 |
| 标签含义 | 0 = 代码已答对；1 = 代码没答对（该交大模型） |
| 代码答对率 | **49.5%** |
| 标注成本 | 每条平均 2.3 轮工具调用，全量约 1.2 万次 API 请求 |

判官不是“看问题打分”，而是**可以调工具查库核验**：
`search_venues` / `get_venue` / `filter_venues` / `nearby`，核对代码回答是否属实、是否满足全部条件。

---

## 4. 关键结果

阈值在 **dev 上选**，test 只跑一次：

| 指标 | dev（534） | **test（519）** |
|---|---|---|
| 覆盖率（交代码） | 43% | **44%** |
| 代码路由准确率 | 88.6% | **88.6%** |
| **错给代码（危险）** | 4.9% | **5.0%** |
| 该给代码却给模型（浪费） | 14.6% | 13.1% |

留出合计 1,053 条：错给代码 **4.9%（52/1053）**、代码路由准确率 **89%**、覆盖率 **43%**。

对照组：

| 方案 | 错误答案比例 |
|---|---|
| 全交代码 | **50.5%** |
| 4 模型集成 + 0.50 | **4.9%** |

阈值取舍（dev）：

| 阈值 | 覆盖率 | 路由准确率 | 错给代码 |
|---|---|---|---|
| 0.50 | 43% | 88.6% | 4.9% |
| 0.70 | 39% | 92% | 3.2% |
| 0.85 | 32% | 92% | 2.4% |
| 0.90 | 24% | 95% | 1.1% |

全量 3,525 条（无标签，只看覆盖与延时）：走代码 **44.7%**，延时中位 **67 ms**（p95 85 ms）。

---

## 5. 为什么用集成，而不是单模型置信度

单模型 softmax **分不出对错**（温度校准后 ECE 0.22→0.13，错答案仍高置信）。
4 个种子模型的分歧度才是有效信号（全量留出集）：

| 规则 | 走代码 | 路由准确率 | 错给代码（占全部） |
|---|---|---|---|
| **4/4 一致** | 457 | **89%** | **4.9%** |
| ≥3/4 | 543 | 85% | 7.6% |
| ≥2/4 | 612 | 81% | 11.1% |

---

## 6. 错判集中在哪（52 条）

| 特征 | 错判组 | 判对组 | 说明 |
|---|---|---|---|
| **多条件筛选** | **56%** | 38% | 评分/价格/时长组合筛不干净 |
| 菜名 | 17% | 13% | 库里无此菜时静默漏筛 |
| 指代 | 21% | 37% | 指代错对象 |

**主要矛盾从“指代”变成了“多条件筛选”**——执行器在组合条件上有漏筛/误筛，门控从措辞上看不出来。

---

## 7. 目录

```
app/
  executor_frozen.py   查库执行器（冻结版，标签基准，SHA256 EB3C97FD…）
  ensemble_gate.py     4 模型集成门控（阈值 0.50）
  main.py              FastAPI：/intent /answer /api/gate /dashboard
data/
  crosswoz/kb_official.json       官方商户库 2,549 家
  crosswoz_gate_full/             全量标注 train 2,469 / dev 534 / test 519
eval/
  ensemble_routing_full.jsonl     留出集逐条路由结果
  ensemble_routing_3525.jsonl     全量路由结果
  eval_full_gate.py               dev 选阈值 + test 报告
data/label_full_gate.py           agent 判官全量标注（可查库）
reports/
  dashboard.html                  可视化面板（直接打开）
  门控实验/集成门控-结果.md        详细报告
```

## 8. 复现

```powershell
# 1) 依赖
uv sync

# 2) 生成面板
$env:INTENT_DATASET="crosswoz_gate_full"
.venv\Scripts\python.exe data\build_dashboard.py
start reports\dashboard.html

# 3) 起服务
$env:INTENT_DATASET="crosswoz_gate_full"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8010
# 访问 http://127.0.0.1:8010/dashboard
```

## 9. 说明

- 商户库来自 CrossWOZ 数据集自带的 `database/`，属性由数据集作者构造，未经现实核验；
  因此所有指标衡量的是“与参考库的一致性”。
- 大型文件（原始数据、模型权重、`.venv`）不入库，见 `.gitignore`。

## 10. 没有模型权重怎么看结果

仓库包含**逐条路由结果**，不重训也能核对结论：

- `eval/ensemble_routing_full.jsonl`：留出 1,053 条的 route / votes / min_p / 标签
- `eval/ensemble_routing_3525.jsonl`：全量 3,525 条的 route 与置信度
- `reports/dashboard.html`：所有指标由上面两份数据生成

要完整复现（重新训练门控）：

```powershell
# 1) 下载基座模型（约 1.2 GB，hf-mirror）
curl.exe -L -o models/_pretrained/chinese-macbert-large/pytorch_model.bin `
  https://hf-mirror.com/hfl/chinese-macbert-large/resolve/main/pytorch_model.bin

# 2) 训练 4 个种子模型（全量 2,469 条，约 50 分钟）
$env:INTENT_DATASET="crosswoz_gate_full"; $env:INTENT_MODEL="models/_pretrained/chinese-macbert-large"
foreach ($s in 7,13,21,33) { $env:INTENT_SEED="$s"; python train/finetune.py --epochs 6 }
```
