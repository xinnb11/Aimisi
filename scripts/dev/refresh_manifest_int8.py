#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 publish/ 里 T2S 三图换成 Int8 后，重算 manifest（大小 + SHA256 + 总字节），
并把结果同步到工程的 models/aemeath-sovits-pro/manifest.json。

m05844 第 3 条：仓库里主模型从 FP32 换成 Int8 混合包（t2s_* = int8_dynamic，
vits = fp32），这样手机端下载量从 1.17 GB 降到约 700 MB。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
PROJECT_MANIFEST = Path(r"D:\语音模型项目\models\aemeath-sovits-pro\manifest.json")

TEXT_SUFFIXES = {".json", ".md", ".txt"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_newlines(path: Path) -> None:
    """魔搭/git 提交会把 CRLF 规整成 LF；本地先转好，manifest 的 sha 才能与远端一致。"""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    raw = path.read_bytes()
    fixed = raw.replace(b"\r\n", b"\n")
    if fixed != raw:
        path.write_bytes(fixed)
        print(f"[lf] {path.relative_to(PUBLISH)}")


def main() -> None:
    manifest_path = PUBLISH / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for entry in manifest["files"]:
        target = PUBLISH / entry["path"]
        if target.exists():
            normalize_newlines(target)
    normalize_newlines(manifest_path)

    files = []
    total = 0
    for entry in manifest["files"]:
        path = PUBLISH / entry["path"]
        if not path.exists():
            raise SystemExit(f"[FAIL] manifest 里列了但磁盘上没有：{entry['path']}")
        size = path.stat().st_size
        files.append({"path": entry["path"], "bytes": size, "sha256": sha256_of(path)})
        total += size

    manifest["files"] = files
    manifest["total_files"] = len(files)
    manifest["total_bytes"] = total
    manifest["quantization"] = {
        "onnx_main": "int8_dynamic (t2s_encoder / t2s_fsdec / t2s_sdec)",
        "vocoder": "fp32 (vits.onnx)",
        "bert": "int8_dynamic",
        "note": "前端语义大胆压 Int8，后端声码器严守 FP32/FP16——桌面 A/B 实测 Int8 三图"
                "话质更好（无开头结巴）、F0 与谱质心无损，包体从 1.17 GB 降到 "
                f"{total / 1024 / 1024:.1f} MB",
    }
    manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["notes"] = (
        "v2.6：T2S 三图改为 int8_dynamic（桌面 A/B：Int8 话质更好、F0/谱质心无损、"
        "端到端 14.3s vs 16.6s），声码器 vits.onnx 保持 FP32；其余同 v2.5"
        "（第21层 BERT、audio_embedding、tone_table、798 词 lexicon、XNNPACK 默认关）。"
    )

    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(payload, encoding="utf-8", newline="\n")

    project_copy = PROJECT_MANIFEST
    project_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, project_copy)

    print(f"[ok] manifest: {len(files)} files / {total} B ({total / 1024 / 1024:.1f} MiB)")
    for name in ("t2s_encoder.onnx", "t2s_fsdec.onnx", "t2s_sdec.onnx", "vits.onnx"):
        size = (PUBLISH / name).stat().st_size
        print(f"     {name:<18} {size:>12,} B")
    print(f"[ok] 同步到 {project_copy}")


if __name__ == "__main__":
    main()
