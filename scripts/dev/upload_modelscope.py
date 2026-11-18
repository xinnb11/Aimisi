# -*- coding: utf-8 -*-
"""把 `publish/` 上传到魔搭 `xinxin111x/Aimisi`，然后逐文件校验远端 SHA256。

用法：
    python scripts/dev/upload_modelscope.py --token ms-xxxx
    python scripts/dev/upload_modelscope.py --token ms-xxxx --verify-only

校验方式：从仓库把 `manifest.json` 拉回来，按里面的 `files[]`（path/bytes/sha256）
逐条与本地产物比对；再随机抽 5 个文件真实下载并算 SHA256。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
REPO = "xinxin111x/Aimisi"


def upload(token: str) -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    started = time.time()
    last = {"pct": -1}

    def progress(info: dict) -> None:
        total = info.get("total_bytes") or 1
        done = info.get("uploaded_bytes") or 0
        pct = int(done * 100 / total)
        if pct != last["pct"] and pct % 5 == 0:
            last["pct"] = pct
            print(f"  upload {pct:3d}%  {done/1e6:8.1f}/{total/1e6:.1f} MB", flush=True)

    api.upload_folder(
        repo_id=REPO,
        folder_path=str(PUBLISH),
        commit_message="v2.4: layer21 BERT + 798-word polyphone lexicon + audio_embedding",
        commit_description=(
            "统一 bert/model.int8.onnx 为第21层 hidden_states[-3]；"
            "新增 conditioning/audio_embedding.f32(1025x512)、g2p/tone_table.json、"
            "g2p/lexicon.json(798 词)；4 个情绪 ref_bert.f32 按第21层重算。"
        ),
        progress_callback=progress,
    )
    print(f"[upload] done in {time.time() - started:.1f}s")


def verify(token: str) -> int:
    import requests

    local_manifest = json.loads((PUBLISH / "manifest.json").read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {token}"}
    base = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo"

    def fetch(path: str):
        resp = requests.get(base, params={"Revision": "master", "FilePath": path}, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp

    remote_manifest = json.loads(fetch("manifest.json").content.decode("utf-8"))
    local_files = {f["path"]: f for f in local_manifest["files"]}
    remote_files = {f["path"]: f for f in remote_manifest.get("files", [])}

    missing = sorted(set(local_files) - set(remote_files))
    extra = sorted(set(remote_files) - set(local_files))
    bad = [
        p for p in sorted(set(local_files) & set(remote_files))
        if local_files[p]["sha256"] != remote_files[p]["sha256"]
        or local_files[p]["bytes"] != remote_files[p]["bytes"]
    ]
    print(f"[verify] 本地 {len(local_files)} / 远端 manifest {len(remote_files)}")
    print(f"[verify] 缺失 {len(missing)} 多出 {len(extra)} manifest 内哈希不符 {len(bad)}")
    for p in (missing + extra + bad)[:10]:
        print("   !!", p)

    probes = ["conditioning/g2p/lexicon.json", "conditioning/conditioning.json",
              "conditioning/audio_embedding.f32", "bert/vocab.txt",
              "conditioning/平静/ref_bert.f32", "bert/model.int8.onnx"]
    ok = 0
    for rel in probes:
        if rel not in local_files:
            continue
        try:
            body = fetch(rel).content
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] {rel}: 下载失败 {exc}")
            continue
        digest = hashlib.sha256(body).hexdigest()
        same = digest == local_files[rel]["sha256"] and len(body) == local_files[rel]["bytes"]
        ok += 1 if same else 0
        print(f"[probe] {rel}: {'OK' if same else 'MISMATCH'} ({len(body)} B)")

    good = not missing and not extra and not bad and ok == len(probes)
    print("[verify]", "全部一致 ✅" if good else "存在差异 ❌")
    return 0 if good else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        upload(args.token)
    return verify(args.token)


if __name__ == "__main__":
    sys.exit(main())
