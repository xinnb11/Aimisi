# -*- coding: utf-8 -*-
"""开启三个声学开关、重算 publish manifest、打包真机推送 tar。

用法：
    python scripts/dev/enable_switches_and_pack.py [--pack]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
HYBRID = Path(r"D:\爱弥斯\_work\hybrid")
ONNX = Path(r"D:\爱弥斯\_work\onnx")
TAR = Path(r"D:\语音模型项目\scripts\dev\_device_update.tar")

FLAGS = {
    "sync_y_emb": True,
    "eos_guard": True,
    "tone_sandhi": True,
    "lexicon_override": True,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enable_flags() -> None:
    for root in (PUBLISH, HYBRID):
        p = root / "conditioning" / "conditioning.json"
        if not p.is_file():
            print("skip missing", p)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        changed = {k: (data.get(k), v) for k, v in FLAGS.items() if data.get(k) != v}
        data.update(FLAGS)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"updated {p} ({len(changed)} changed): {sorted(changed)}")


def rebuild_manifest() -> None:
    manifest_path = PUBLISH / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    total = 0
    for item in manifest.get("files", []):
        rel = item["path"]
        f = PUBLISH / rel
        if not f.is_file():
            raise SystemExit(f"missing file: {f}")
        size = f.stat().st_size
        total += size
        entry = dict(item)
        entry["bytes"] = size
        entry["sha256"] = sha256(f)
        files.append(entry)
    manifest["files"] = files
    manifest["total_bytes"] = total
    manifest["total_files"] = len(files)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest rebuilt: {len(files)} files / {total / (1024 ** 2):.1f} MiB")

    project_copy = Path(r"D:\语音模型项目\models\aemeath-sovits-pro\manifest.json")
    project_copy.parent.mkdir(parents=True, exist_ok=True)
    project_copy.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    print("synced", project_copy)


def pack_device() -> None:
    if TAR.exists():
        TAR.unlink()
    entries: list[tuple[Path, str]] = [
        (ONNX / "t2s_encoder.int8.onnx", "t2s_encoder.int8.onnx"),
        (ONNX / "t2s_fsdec.int8.onnx", "t2s_fsdec.int8.onnx"),
        (ONNX / "t2s_sdec.int8.onnx", "t2s_sdec.int8.onnx"),
        (PUBLISH / "bert" / "model.int8.onnx", "bert/model.int8.onnx"),
        (PUBLISH / "conditioning" / "conditioning.json", "conditioning/conditioning.json"),
        (PUBLISH / "conditioning" / "g2p" / "lexicon.json", "conditioning/g2p/lexicon.json"),
        (PUBLISH / "conditioning" / "g2p" / "tone_table.json", "conditioning/g2p/tone_table.json"),
        (PUBLISH / "conditioning" / "audio_embedding.f32", "conditioning/audio_embedding.f32"),
    ]
    for emotion in ("平静", "开心", "温柔", "调侃"):
        entries.append(
            (PUBLISH / "conditioning" / emotion / "ref_bert.f32", f"conditioning/{emotion}/ref_bert.f32")
        )
    with tarfile.open(TAR, "w") as tar:
        for src, arc in entries:
            if not src.is_file():
                raise SystemExit(f"missing: {src}")
            tar.add(src, arcname=arc)
    print(f"tar: {TAR} {TAR.stat().st_size:,} B, {len(entries)} entries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()
    enable_flags()
    rebuild_manifest()
    if args.pack:
        pack_device()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
