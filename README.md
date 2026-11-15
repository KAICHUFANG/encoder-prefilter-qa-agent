# 客服意图识别 Agent · encoder-only 预筛

用一个小型 encoder-only 模型（MacBERT）判断：**用户这句话该交给确定性代码，还是交给大模型**。
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
   │     ├─ 4/4 一致且 P(代码) ≥ 0.90  → 执行器（查库，0 token）
   │     └─ 否则                        → 大模型
   │
   └─ 执行器：官方商户库 2549 家
        字段查询 / 区间筛选 / 设施匹配 / 周边关系表 / 澄清反问
```

门控不做回答，只做分流；判错的代价是“错答案发出去”或“多花 token”。

---

## 3. 关键结果

数据：CrossWOZ 真实问句 400 条（train 280 / dev 60 / test 60），留出集 120 条评测。
标签：DeepSeek agent 判官（可调用查库工具核验）判断“代码有没有答对”。
代码：`app/executor_frozen.py`（SHA256 冻结，标签与训练都基于这一份）。

| 指标 | 数值 | 口径 |
|---|---|---|
| **错给代码（危险）** | **7.5%** | 占全部请求：用户会收到错误答案 |
| 代码路由准确率 | 80% | 走代码的请求里答对的比例 |
| 覆盖率 | 38% | 由代码处理、0 token 的请求比例 |
| 该给代码却给模型 | 24% | 浪费 token，但不影响正确性 |
| 单次判定延时 | 77 ms | 4 模型串行，p95 111 ms |

对照组：

| 方案 | 错误答案比例 |
|---|---|
| 全交代码 | 41.7% |
| 单模型门控 | 约 20% |
| **4 模型集成 + 0.90** | **7.5%** |
| 全交大模型 | 取决于模型 |

全量测试集（3,525 条，无标签）：走代码 30.8%，交模型 69.2%。

> 局限：留出集仅 120 条，7.5% 的 95% 置信区间约 2.8%–12.2%。

---

## 4. 为什么用集成，而不是单模型置信度

单模型的 softmax 置信度**分不出对错**：

- 温度校准把 ECE 从 0.22 压到 0.13，但错答案照样高置信；
- 阈值从 0.5 扫到 0.95，错误放行只从 12 条降到 11 条。

换成 4 个种子模型后，**分歧度**成了有效信号：

| 一致度 | 准确率 |
|---|---|
| 4/4 全一致 | 78% |
| 3/4 | 36% |
| ≤2/4 | 38% |

---

## 5. 错判集中在哪

9 条“错给代码”的共同点：

1. **置信度极高**（0.93–0.99），阈值拦不住；
2. **指代类占 56%**（“它/这家”指错对象）；
3. **菜名类占 22%**（库里没有该菜，代码静默漏筛）。

这说明门控看不到执行器的实际行为——“看起来能查”不等于“查得对”。

---

## 6. 目录

```
app/
  executor_frozen.py   查库执行器（冻结版，标签基准）
  ensemble_gate.py     4 模型集成门控
  main.py              FastAPI：/intent /answer /api/gate /dashboard
  model.py             单模型推理包装
data/
  crosswoz/kb_official.json       官方商户库 2549 家
  crosswoz_gate_v2/               400 条标签（agent 判官）
eval/
  ensemble_routing_400.jsonl      逐条路由结果
  ensemble_routing_3525.jsonl     全量路由结果
  relabel_gate_agent.py           agent 式标注（可查库）
reports/
  dashboard.html                  可视化面板（直接打开）
  门控实验/集成门控-结果.md        详细报告
```

## 7. 复现

```powershell
# 1) 依赖
uv sync

# 2) 生成面板
$env:INTENT_DATASET="crosswoz_gate_v2"
.venv\Scripts\python.exe data\build_dashboard.py
start reports\dashboard.html

# 3) 起服务
$env:INTENT_DATASET="crosswoz_gate_v2"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8010
# 访问 http://127.0.0.1:8010/dashboard
```

## 8. 说明

- 商户库来自 CrossWOZ 数据集自带的 `database/`，属性由数据集作者构造，未经现实核验；
  因此所有指标衡量的是“与参考库的一致性”。
- 大型文件（原始数据、模型权重、`.venv`）不入库，见 `.gitignore`。

## 9. 没有模型权重怎么看结果

仓库里已包含**逐条路由结果**，不需要重训就能核对结论：

- `eval/ensemble_routing_400.jsonl`：400 条的 route / votes / min_p / 标签
- `eval/ensemble_routing_3525.jsonl`：全量 3,525 条的 route 与置信度
- `reports/dashboard.html`：直接双击打开，所有指标都由上面两份数据生成

要完整复现（重新训练门控）：

```powershell
# 1) 下载基座模型（约 1.2 GB）
#    仓库未入库权重，可从 hf-mirror 拉取
curl.exe -L -o models/_pretrained/chinese-macbert-large/pytorch_model.bin `
  https://hf-mirror.com/hfl/chinese-macbert-large/resolve/main/pytorch_model.bin

# 2) 训练 4 个种子模型
$env:INTENT_DATASET="crosswoz_gate_v2"; $env:INTENT_MODEL="models/_pretrained/chinese-macbert-large"
foreach ($s in 7,13,21,33) { $env:INTENT_SEED="$s"; python train/finetune.py --epochs 6 }
```
