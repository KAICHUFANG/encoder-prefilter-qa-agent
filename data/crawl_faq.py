"""爬取「企业内部问题」语料：企业软件的帮助中心 / FAQ。

为什么选这类站点：
    企业内部制度问答没有公开数据集（企业内部数据不外流）。
    但企业软件（钉钉/飞书/金蝶/用友/北森）的**帮助中心**，内容就是
    「怎么请假」「报销单在哪提交」「审批流怎么配」——这正是员工会问的问题，
    而且页面公开、结构规整、措辞接近真实提问。

合规（必须遵守，写在代码里）：
    1) 先读 robots.txt，Disallow 的路径一律不碰
    2) 每个域名限速 1 请求/秒
    3) 只取公开页面，不登录、不绕验证码
    4) 保留来源 URL，非商业用途（课程作业）
    5) 只存「问题标题 + 答案首段」，不整站镜像

用法：
    python data/crawl_faq.py --check                      # 只看 robots 是否允许
    python data/crawl_faq.py --seed https://open.dingtalk.com/document/ --limit 20
    python data/crawl_faq.py --seed <URL> --limit 50 --out data/raw/enterprise_faq.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

UA = "Mozilla/5.0 (compatible; CourseProjectBot/1.0; +educational-use)"
DELAY_SECONDS = 1.0

# 候选站点（按「问题像不像员工提问」排序）
CANDIDATES = [
    "https://open.dingtalk.com/document/",     # 钉钉开放平台文档（审批/考勤/报销）
    "https://open.feishu.cn/document/",        # 飞书文档（审批/人事）
    "https://www.kingdee.com/products/",       # 金蝶产品页（ERP 流程）
    "https://www.dingtalk.com/",               # 钉钉官网（含常见问题）
]

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_last_request: dict[str, float] = {}


def robots_for(url: str) -> urllib.robotparser.RobotFileParser:
    host = urllib.parse.urlsplit(url).netloc
    if host in _robots_cache:
        return _robots_cache[host]
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"https://{host}/robots.txt")
    try:
        rp.read()
    except Exception as exc:                       # 读不到就按"不允许"处理，保守
        print(f"[robots] {host} 读取失败：{exc}（按不允许处理）")
        rp.disallow_all = True
    _robots_cache[host] = rp
    return rp


def allowed(url: str) -> bool:
    return robots_for(url).can_fetch(UA, url)


def polite_get(url: str, timeout: int = 20) -> str | None:
    """限速 + robots 检查后取页面；返回 HTML 文本。"""
    import urllib.request

    if not allowed(url):
        print(f"[跳过] robots 不允许：{url}")
        return None
    host = urllib.parse.urlsplit(url).netloc
    wait = DELAY_SECONDS - (time.time() - _last_request.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read()
        _last_request[host] = time.time()
        return body.decode(charset, errors="replace")
    except Exception as exc:
        print(f"[失败] {url} -> {type(exc).__name__}: {exc}")
        _last_request[host] = time.time()
        return None


TAG_RE = re.compile(r"<[^>]+>")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
H_RE = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.S | re.I)
LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.S | re.I)
P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
QUESTION_HINT = re.compile(r"[？?]|怎么|如何|在哪|能不能|是否|流程|步骤|配置|设置")


def clean(fragment: str) -> str:
    text = TAG_RE.sub("", fragment)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_questions(html_text: str, url: str, site: str) -> list[dict]:
    """从一页里抽「像问题」的条目：标题/列表项/段落里的疑问句。"""
    out: list[dict] = []
    page_title = clean(TITLE_RE.search(html_text).group(1)) if TITLE_RE.search(html_text) else ""

    candidates: list[str] = [clean(m.group(1)) for m in H_RE.finditer(html_text)]
    candidates += [clean(m.group(1)) for m in LI_RE.finditer(html_text)]
    # 段落里以问号结尾的短句，往往是 FAQ 的问题行
    candidates += [clean(m.group(1)) for m in P_RE.finditer(html_text)]

    seen: set[str] = set()
    for text in candidates:
        if not (4 <= len(text) <= 60):
            continue
        if not QUESTION_HINT.search(text):
            continue
        norm = re.sub(r"\s+", "", text)
        if norm in seen:
            continue
        seen.add(norm)
        out.append({"question": text, "page_title": page_title, "url": url, "site": site})
    return out


def links_on(html_text: str, base_url: str, same_host_only: bool = True) -> list[str]:
    base_host = urllib.parse.urlsplit(base_url).netloc
    links = []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html_text):
        href = html.unescape(m.group(1)).strip()
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        parts = urllib.parse.urlsplit(absolute)
        if parts.scheme not in ("http", "https"):
            continue
        if same_host_only and parts.netloc != base_host:
            continue
        links.append(absolute.split("#")[0])
    # 去重保序
    seen, uniq = set(), []
    for link in links:
        if link not in seen:
            seen.add(link)
            uniq.append(link)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=CANDIDATES[0], help="起始页")
    ap.add_argument("--limit", type=int, default=20, help="最多抓多少个页面")
    ap.add_argument("--out", default="data/raw/enterprise_faq.jsonl")
    ap.add_argument("--check", action="store_true", help="只检查 robots，不抓取")
    args = ap.parse_args()

    if args.check:
        for url in CANDIDATES:
            print(f"{url:<45} robots 允许: {allowed(url)}")
        return 0

    site = urllib.parse.urlsplit(args.seed).netloc
    queue = [args.seed]
    visited: set[str] = set()
    rows: list[dict] = []
    pages = 0

    while queue and pages < args.limit:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        html_text = polite_get(url)
        if not html_text:
            continue
        pages += 1
        found = extract_questions(html_text, url, site)
        rows.extend(found)
        print(f"[{pages}/{args.limit}] {url[-60:]:<60} 抽到 {len(found)} 条问题")
        if len(rows) < args.limit * 20:          # 队列别爆炸
            for link in links_on(html_text, url)[:15]:
                if link not in visited:
                    queue.append(link)

    # 去重（按问题文本哈希）
    seen, uniq = set(), []
    for r in rows:
        key = hashlib.md5(re.sub(r"\s+", "", r["question"]).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n共抓 {pages} 页，得到 {len(uniq)} 条不重复问题 -> {out}")
    if uniq:
        print("\n样例：")
        for r in uniq[:8]:
            print(f"   [{r['site']}] {r['question'][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
