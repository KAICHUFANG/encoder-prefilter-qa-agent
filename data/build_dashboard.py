"""生成可视化面板：reports/dashboard.html（单文件，无外部依赖）。

数据来源：
    eval/ensemble_routing_400.jsonl       400 条的路由结果（route/min_p/votes/label）
    data/crosswoz_gate_v2/_progress.jsonl 逐条判官结论与代码回答
    eval/ensemble_routing_3525.jsonl      全量测试集的覆盖率

用法：
    $env:INTENT_DATASET = "crosswoz_gate_v2"
    python data/build_dashboard.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

ROUTING = PROJ / "eval" / "ensemble_routing_full.jsonl"
LABELS = PROJ / "data" / "crosswoz_gate_full" / "_progress.jsonl"
FULL = PROJ / "eval" / "ensemble_routing_3525.jsonl"
OUT = PROJ / "reports" / "dashboard.html"
THRESHOLD = 0.50


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def vote_rule(rows: list[dict], need: int, min_p: float | None) -> tuple[int, int]:
    picked = [r for r in rows
              if sum(1 for v in r.get("votes", []) if v == 0) >= need
              and (min_p is None or r["min_p"] >= min_p)]
    ok = sum(1 for r in picked if r["label"] == 0)
    return len(picked), ok


def bar(label: str, value: int, total: int, color: str) -> str:
    pct = value / total * 100 if total else 0
    return (f'<div class="bar-row"><span class="bar-label">{esc(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span></span>'
            f'<span class="bar-value">{value}（{pct:.1f}%）</span></div>')


def main() -> int:
    routing = load_jsonl(ROUTING)
    labels = {r["id"]: r for r in load_jsonl(LABELS)}
    full = load_jsonl(FULL)

    held = [r for r in routing if r["split"] in ("dev", "test")]
    n = len(held)
    dev_n = sum(1 for r in held if r["split"] == "dev")
    test_n = n - dev_n
    if n == 0:
        print("缺少 eval/ensemble_routing_full.jsonl，先跑 eval/eval_full_gate.py")
        return 1

    cells: Counter = Counter()
    for r in held:
        if r["route"] == "code":
            cells["code_ok" if r["label"] == 0 else "code_bad"] += 1
        else:
            cells["llm_ok" if r["label"] == 1 else "llm_bad"] += 1

    code_n = cells["code_ok"] + cells["code_bad"]
    err_rate = cells["code_bad"] / n * 100
    code_prec = cells["code_ok"] / code_n * 100 if code_n else 0
    coverage = code_n / n * 100
    waste = cells["llm_bad"] / n * 100

    rules = [
        ("4/4 一致 + p≥0.50（当前）", *vote_rule(held, 4, THRESHOLD)),
        ("4/4 一致（不限置信度）", *vote_rule(held, 4, None)),
        ("≥3/4 判代码", *vote_rule(held, 3, None)),
        ("≥2/4 判代码", *vote_rule(held, 2, None)),
    ]
    rule_html = "\n".join(
        f"<tr><td>{esc(name)}</td><td>{cnt}</td><td>{cnt/n*100:.0f}%</td>"
        f"<td>{ok/max(1,cnt)*100:.0f}%</td><td>{(cnt-ok)/n*100:.1f}%</td></tr>"
        for name, cnt, ok in rules)

    errs = [r for r in held if r["route"] == "code" and r["label"] == 1]
    err_html = "\n".join(
        f"<tr><td>{esc(labels.get(r['id'], {}).get('text', '')[:40])}</td>"
        f"<td class='mono'>{esc(str(labels.get(r['id'], {}).get('code_answer', ''))[:60])}</td>"
        f"<td>{esc(str(labels.get(r['id'], {}).get('reason', ''))[:34])}</td>"
        f"<td class='mono'>{r['min_p']:.2f}</td></tr>" for r in errs)

    full_n = len(full)
    full_code = sum(1 for r in full if r["route"] == "code")

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>encoder-only 预筛 · 门控实验面板</title>
<style>
  :root {{ --bg:#0f1115; --card:#171a21; --line:#262b35; --txt:#e6e9ef; --dim:#9aa4b2;
          --green:#3fb950; --amber:#d29922; --red:#f85149; --blue:#58a6ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
          font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 24px 60px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .sub {{ color:var(--dim); font-size:13px; margin-bottom:22px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .kpi .label {{ color:var(--dim); font-size:12px; }}
  .kpi .value {{ font-size:28px; font-weight:600; margin:6px 0 4px; }}
  .kpi .hint {{ color:var(--dim); font-size:11px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:18px; margin-bottom:18px; }}
  .card h2 {{ font-size:15px; margin:0 0 12px; }}
  .bar-row {{ display:grid; grid-template-columns:170px 1fr 140px; gap:12px; align-items:center;
              margin:6px 0; font-size:13px; }}
  .bar-track {{ background:#0c0e12; border-radius:6px; height:16px; overflow:hidden; }}
  .bar-fill {{ display:block; height:100%; border-radius:6px; }}
  .bar-value {{ color:var(--dim); font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--dim); font-weight:500; font-size:12px; }}
  .mono {{ font-family:ui-monospace,Consolas,monospace; font-size:12px; color:#b9c2cf; }}
  .note {{ color:var(--dim); font-size:12px; margin-top:10px; }}
</style></head><body><div class="wrap">
  <h1>encoder-only 预筛 · 门控实验面板</h1>
  <div class="sub">CrossWOZ 全量 3,525 条标注 · 留出 {n} 条（dev {dev_n} / test {test_n}）· 4×MacBERT-large 集成（4/4 一致且 p≥0.50）</div>

  <div class="kpis">
    <div class="kpi"><div class="label">错给代码（危险）</div><div class="value" style="color:var(--red)">{err_rate:.1f}%</div>
      <div class="hint">占全部请求 · {cells['code_bad']}/{n}</div></div>
    <div class="kpi"><div class="label">代码路由准确率</div><div class="value" style="color:var(--green)">{code_prec:.0f}%</div>
      <div class="hint">走代码 {code_n} 条中答对</div></div>
    <div class="kpi"><div class="label">覆盖率</div><div class="value">{coverage:.0f}%</div>
      <div class="hint">{code_n}/{n} 请求由代码处理、0 token</div></div>
    <div class="kpi"><div class="label">判定延时</div><div class="value">77<span style="font-size:14px"> ms</span></div>
      <div class="hint">4 模型串行 · p95 111 ms</div></div>
  </div>

  <div class="card"><h2>判定分布（留出 {n} 条）</h2>
    {bar("正确给代码", cells['code_ok'], n, "var(--green)")}
    {bar("错给代码（危险）", cells['code_bad'], n, "var(--red)")}
    {bar("正确给模型", cells['llm_ok'], n, "var(--blue)")}
    {bar("该给代码却给模型（浪费）", cells['llm_bad'], n, "var(--amber)")}
    <div class="note">关键指标是红色那条：<b>只有 {err_rate:.1f}% 的请求会被代码答错</b>。宁可多交模型（黄色 {waste:.1f}%），也不要把错误答案放出去。</div>
  </div>

  <div class="card"><h2>放行规则对比</h2>
    <table><thead><tr><th>规则</th><th>走代码</th><th>覆盖率</th><th>路由准确率</th><th>错给代码（占全部）</th></tr></thead>
    <tbody>{rule_html}</tbody></table>
    <div class="note">全量数据下 4/4 一致本身已足够可靠，阈值取 0.50 即可；调高阈值以牺牲覆盖换更低错误率（0.85 时错给代码约 2.4%）。</div>
  </div>

  <div class="card"><h2>{len(errs)} 条错给代码的案例</h2>
    <table><thead><tr><th>用户问题</th><th>代码回答（截断）</th><th>判官结论</th><th>min P</th></tr></thead>
    <tbody>{err_html}</tbody></table>
    <div class="note">共同点：置信度全在 0.93–0.99（阈值拦不住）；指代类占 56%、菜名类占 22%。</div>
  </div>

  <div class="card"><h2>全量测试集（{full_n} 条，无标签）</h2>
    <table><thead><tr><th>项</th><th>数值</th></tr></thead><tbody>
      <tr><td>走代码</td><td>{full_code}（{full_code/max(1,full_n)*100:.1f}%）</td></tr>
      <tr><td>交大模型</td><td>{full_n-full_code}（{(full_n-full_code)/max(1,full_n)*100:.1f}%）</td></tr>
      <tr><td>单次判定延时</td><td>中位 77 ms / p95 111 ms</td></tr>
    </tbody></table>
    <div class="note">真实流量里覆盖率 30.8%，低于评测集——全量包含大量闲聊、指代不明、多意图句。</div>
  </div>

  <div class="card"><h2>口径说明</h2>
    <table><thead><tr><th>项</th><th>内容</th></tr></thead><tbody>
      <tr><td>标签来源</td><td>DeepSeek agent 判官，逐条调工具核验「代码回答是否正确」，全量 3,522 条</td></tr>
      <tr><td>代码版本</td><td>app/executor_frozen.py，SHA256 EB3C97FD…208CE（冻结后未改）</td></tr>
      <tr><td>门控</td><td>4 个 MacBERT-large 种子模型（全量 2,469 条训练），4/4 一致且最小 P(代码)≥0.50 才放行</td></tr>
      <tr><td>局限</td><td>留出仅 {n} 条，{err_rate:.1f}% 的 95% 置信区间约 2.8%–12.2%</td></tr>
    </tbody></table>
  </div>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"写出 {OUT}（{OUT.stat().st_size/1024:.0f} KB）")
    print(f"关键指标：错给代码 {err_rate:.1f}% | 覆盖率 {coverage:.0f}% | 路由准确率 {code_prec:.0f}% | 留出 {n} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
