"""下载原始数据（两个数据集都走 HuggingFace 镜像）。

为什么不用 OpenI / 官网直链：OpenI 那个 KUAKE-QIC 下载页已经拿不到文件；
官网直链国内经常断。hf-mirror.com 可以直连。

用法：
    uv run python data/fetch.py --dataset crosswoz     # 推荐：任务型对话，有真实数据库
    uv run python data/fetch.py --dataset kuake        # 对比用：医疗搜索问句（主题分类）
    uv run python data/fetch.py --dataset crosswoz --list    # 只看仓库里有什么

下载后：
    crosswoz: data/raw/crosswoz/data.zip + extracted/{train,val,test}.json
    kuake:    data/raw/kuake/KUAKE-QIC_dev.json

注意：crosswoz 的 train.json 有 390MB，prepare_crosswoz.py 用 ijson 流式读，
      不会把内存吃爆。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import config  # noqa: E402

REPOS = {
    "crosswoz": ["GEM/CrossWOZ", "ConvLab/crosswoz"],
    "kuake": ["wyp/cblue-qic", "CHIHCI/CBlue-KUAKE-QIC", "wyp/CBlue-KUAKE-QIC"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="crosswoz", choices=sorted(REPOS))
    ap.add_argument("--repo", default=None, help="覆盖默认仓库")
    ap.add_argument("--list", action="store_true", help="只列文件，不下载")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    candidates = [args.repo] if args.repo else REPOS[args.dataset]
    target = PROJ / "data" / "raw" / args.dataset
    target.mkdir(parents=True, exist_ok=True)

    api = HfApi(endpoint=config.HF_ENDPOINT)
    print(f"镜像 {config.HF_ENDPOINT}\n目标 {target}\n")

    for repo in candidates:
        print(f"== 尝试 {repo} ==")
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
        except Exception as exc:                      # 网络抖动就换下一个源
            print(f"   [跳过] 列不出文件：{type(exc).__name__}: {exc}")
            continue

        print(f"   文件 {len(files)} 个：{files[:8]}{' ...' if len(files) > 8 else ''}")
        if args.list:
            continue

        try:
            path = snapshot_download(
                repo_id=repo, repo_type="dataset", local_dir=str(target),
                endpoint=config.HF_ENDPOINT, max_workers=4,
            )
        except Exception as exc:
            print(f"   [失败] {type(exc).__name__}: {exc}")
            print("   继续试下一个源……")
            continue

        print(f"   [完成] {path}")
        # crosswoz 需要解压 data.zip
        zips = list(Path(path).rglob("data.zip"))
        for z in zips:
            out = Path(path) / "extracted"
            out.mkdir(exist_ok=True)
            print(f"   解压 {z.name} -> {out}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    print(f"      {f.name}  {f.stat().st_size:,} 字节")

        print("\n下一步：")
        if args.dataset == "crosswoz":
            print("  uv run python train/prepare_crosswoz.py")
        else:
            print('  uv run python train/prepare.py --pattern "KUAKE-QIC_dev.json"')
        return 0

    if args.list:
        return 0
    print("\n[全部失败] 网络问题。可以先试 --repo 换源，或者过几分钟再跑。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
