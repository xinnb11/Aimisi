#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 publish 里最新的 conditioning.json / manifest.json 增量推到魔搭仓库。

令牌从环境变量 MS_TOKEN 读取，绝不写进仓库文件或凭据文件。
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import requests

REPO_ID = "xinxin111x/Aimisi"
PUBLISH = Path(r"D:\爱弥斯\_work\publish")
FILES = [
    PUBLISH / "conditioning" / "conditioning.json",
    PUBLISH / "manifest.json",
]
API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    token = os.environ.get("MS_TOKEN", "").strip()
    if not token:
        print("缺少 MS_TOKEN 环境变量", file=sys.stderr)
        return 2

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    print("login ok")

    for path in FILES:
        if not path.is_file():
            print(f"本地缺失：{path}", file=sys.stderr)
            return 3
        relative = path.relative_to(PUBLISH).as_posix()
        api.upload_file(
            repo_id=REPO_ID,
            path_or_fileobj=str(path),
            path_in_repo=relative,
            commit_message=f"更新 {relative}（开关全开 + manifest 重算）",
        )
        print(f"uploaded {relative} ({path.stat().st_size} B)")

    # 远端校验：重新下载并比对 SHA256
    ok = True
    for path in FILES:
        relative = path.relative_to(PUBLISH).as_posix()
        url = API.format(repo=REPO_ID, path=relative)
        response = requests.get(url, timeout=300)
        if response.status_code != 200:
            print(f"远端读取失败 {relative}: HTTP {response.status_code}")
            ok = False
            continue
        remote = hashlib.sha256(response.content).hexdigest().upper()
        local = sha256_of(path)
        same = remote == local
        ok = ok and same
        print(f"verify {relative}: remote={remote[:16]}… local={local[:16]}… {'OK' if same else 'MISMATCH'}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
