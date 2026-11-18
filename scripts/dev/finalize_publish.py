# -*- coding: utf-8 -*-
"""收尾 publish/：统一 BERT 命名、补齐 conditioning.json 契约、重算 manifest 并同步到工程。"""

import hashlib
import json
from pathlib import Path

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
COND = PUBLISH / "conditioning"
PROJECT_MANIFEST = Path(r"D:\语音模型项目\models\aemeath-sovits-pro\manifest.json")

FLAGS = {
    # 端侧采样 token 与图内 y_emb 目前一致（图输出的 y 就是图内 token），先不开启覆盖。
    "sync_y_emb": False,
    # EOS 护栏会强行延长句子，桌面参考不需要，默认关。
    "eos_guard": False,
    # 变调表已随包下发，但桌面参考实现未启用以保持与训练侧一致，默认关。
    "tone_sandhi": False,
    # 多音字词组表已扩充到 798 条，默认启用。
    "lexicon_override": True,
}


def main() -> int:
    cond_path = COND / "conditioning.json"
    cond = json.loads(cond_path.read_text(encoding="utf-8"))

    cond["bert_model"] = "bert/model.int8.onnx"
    cond["bert_model_layer21"] = "bert/model.int8.onnx"
    cond["bert_source"] = "repo"
    cond["audio_embedding"] = {
        "file": "audio_embedding.f32",
        "dtype": "float32",
        "shape": [1025, 512],
        "eos_token": 1024,
    }
    g2p = cond.setdefault("g2p", {})
    g2p["tone_table"] = "g2p/tone_table.json"
    g2p["lexicon"] = "g2p/lexicon.json"
    cond.update(FLAGS)
    cond_path.write_text(json.dumps(cond, ensure_ascii=False, indent=2), encoding="utf-8")

    lexicon = json.loads((COND / "g2p" / "lexicon.json").read_text(encoding="utf-8"))["table"]
    tone = json.loads((COND / "g2p" / "tone_table.json").read_text(encoding="utf-8"))

    manifest_path = PUBLISH / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = []
    total = 0
    for path in sorted(PUBLISH.rglob("*")):
        if path.is_dir() or path.name == "manifest.json":
            continue
        data = path.read_bytes()
        total += len(data)
        files.append(
            {
                "path": path.relative_to(PUBLISH).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    manifest["files"] = files
    manifest["total_bytes"] = total
    manifest["arch"] = "GPT-SoVITS v2Pro"
    manifest["arch_alias"] = "GPT-SoVITS v2"
    manifest["bert"] = {
        "model": "bert/model.int8.onnx",
        "vocab": "bert/vocab.txt",
        "tokenizer": "bert/tokenizer.json",
        "feature_layer": "hidden_states[-3] (第21层)",
    }
    manifest["conditioning"] = {
        "root": "conditioning",
        "manifest": "conditioning/conditioning.json",
        "emotions": ["平静", "开心", "温柔", "调侃"],
        "g2p": {"symbols": "g2p/symbols.json", "char2phone": "g2p/char2phone.json",
                "tone_table": "g2p/tone_table.json", "lexicon": "g2p/lexicon.json"},
        "audio_embedding": "conditioning/audio_embedding.f32",
        "lexicon_entries": len(lexicon),
        "tone_variants": len(tone.get("tone_variants", {})),
    }
    manifest["notes"] = (
        "v2.4：BERT 统一为第21层 hidden_states[-3]（model.int8.onnx）；"
        "新增 audio_embedding.f32(1025x512)、g2p/tone_table.json、g2p/lexicon.json(798 词)；"
        "4 个情绪 ref_bert.f32 已按第21层重算；端侧默认关闭 XNNPACK。"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    PROJECT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[manifest] {len(files)} files / {total/1e6:.1f} MB -> publish & project")
    print(f"[lexicon] {len(lexicon)} 词 / [tones] {len(tone.get('tone_variants', {}))} 变体")
    for probe in ("bert/model.int8.onnx", "conditioning/audio_embedding.f32", "conditioning/g2p/lexicon.json"):
        row = next((f for f in files if f["path"] == probe), None)
        print(" ", probe, row["bytes"] if row else "MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
