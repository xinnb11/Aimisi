#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 publish 里的文本文件统一成 LF，剔除 .ms_upload_cache，并重算 manifest。

魔搭（git）在提交时会把 CRLF 规整成 LF，所以 manifest 里 sha256 / bytes 必须按
「入库后的字节」计算，否则端侧下载器逐文件校验必然报哈希不符。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
TEXT_SUFFIXES = {".json", ".txt", ".md"}
SKIP_NAMES = {".ms_upload_cache", "manifest.json"}


def normalize_text_files() -> list[str]:
    changed: list[str] = []
    for path in PUBLISH.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in SKIP_NAMES:
            continue
        raw = path.read_bytes()
        fixed = raw.replace(b"\r\n", b"\n")
        if fixed != raw:
            path.write_bytes(fixed)
            changed.append(str(path.relative_to(PUBLISH)))
    return changed


def collect_files() -> list[Path]:
    result = []
    for path in PUBLISH.rglob("*"):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.name.startswith("."):
            continue
        result.append(path)
    return sorted(result, key=lambda p: p.relative_to(PUBLISH).as_posix())


def main() -> int:
    cache = PUBLISH / ".ms_upload_cache"
    if cache.exists():
        cache.unlink()
        print("removed local .ms_upload_cache")

    changed = normalize_text_files()
    print(f"LF-normalized {len(changed)} text file(s)")
    for name in changed:
        print("  -", name)

    manifest_path = PUBLISH / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = []
    total = 0
    for path in collect_files():
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(PUBLISH).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        total += len(data)

    manifest["files"] = files
    manifest["total_bytes"] = total
    manifest["total_files"] = len(files)
    manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest: {len(files)} files / {total} bytes ({total / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
