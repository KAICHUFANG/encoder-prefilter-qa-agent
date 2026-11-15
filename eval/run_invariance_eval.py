"""语义不变性测试：同一意思、不同句法，模型给不给同一个答案。

为什么这个指标重要：
    CrossWOZ 上做过同样测试，只有 17% 的组一致（模型跟着"先出现的领域词"走）。
    如果 MASSIVE 是真·语义稳定的数据集，这个数字应该显著更高——
    因为它的标签只取决于"用户想干什么"，与句子怎么组织无关。

用法：
    uv run python eval/run_invariance_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

# 每个组：同一意图的五种说法（句法差异大，关键词也有替换）
GROUPS = {
    "alarm": [
        "早上七点叫我",
        "帮我定个早上七点的闹钟",
        "七点叫醒我",
        "设一个七点的闹钟",
        "我要七点起床，定个闹钟",
    ],
    "weather": [
        "今天天气怎么样",
        "今天下雨吗",
        "帮我看下今天天气",
        "外面气温多少度",
        "明天冷不冷",
    ],
    "music": [
        "放首歌",
        "我想听音乐",
        "来点音乐",
        "播放周杰伦的歌",
        "给我放点歌听",
    ],
    "transport": [
        "怎么去机场",
        "去机场怎么走",
        "到机场要多久",
        "帮我导航到机场",
        "我想去机场",
    ],
    "takeaway": [
        "我要点外卖",
        "帮我订份饭",
        "叫个外卖",
        "点个披萨",
        "我想吃外卖",
    ],
    "calendar": [
        "帮我加个日程",
        "明天下午三点开个会，记一下",
        "在日历里加个事件",
        "把明天的会议加到日程",
        "安排个会议",
    ],
}


def main() -> int:
    from app.model import IntentPredictor

    p = IntentPredictor()
    p.warmup()

    consistent = 0
    total_items = 0
    correct = 0
    print(f"{'组':<12}{'预测结果'}")
    for expected, texts in GROUPS.items():
        preds = [p.predict(t, top_k=1)["intent"] for t in texts]
        total_items += len(texts)
        correct += sum(1 for x in preds if x == expected)
        same = len(set(preds)) == 1
        consistent += same
        flag = "✅一致" if same else "❌不一致"
        print(f"{expected:<12}{preds}  {flag}")

    n = len(GROUPS)
    print(f"\n同一意思、五种句法：**完全一致的组 {consistent}/{n} = {consistent / n:.0%}**")
    print(f"逐条准确率（vs 预期场景）：{correct}/{total_items} = {correct / total_items:.0%}")
    print("\n对照：CrossWOZ 上同口径为 17%（rbt3）、0%（MacBERT）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
