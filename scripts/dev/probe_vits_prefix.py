# -*- coding: utf-8 -*-
"""探测 vits.onnx 是否支持「语义码前缀」合成（边算边播的前提）。"""

from __future__ import annotations

import sys
from pathlib import Path

import time\n\nimport numpy as np

sys.path.insert(0, str(Path(r"D:\语音模型项目\scripts")))
import verify_onnx_tts as V  # noqa: E402

MODEL = Path(r"D:\爱弥斯\_work\publish")
TEXT = "今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。"
EMOTION = "温柔"


def main() -> int:
    manifest = V._load_json(MODEL / "manifest.json")
    params = manifest.get("sampling_params") or {}
    temperature = float(params.get("temperature", 0.85))
    top_k = int(params.get("top_k", 15))
    top_p = float(params.get("top_p", 0.9))
    penalty = float(params.get("repetition_penalty", 1.35))
    stop_token = int(manifest.get("eos_token", 1024))

    frontend = V.TextFrontend(MODEL)
    phone_ids, _ = frontend.phones(TEXT)
    text_bert, _ = frontend.bert_features(TEXT)

    cond_dir = MODEL / "conditioning" / EMOTION
    cond = V._load_json(MODEL / "conditioning" / "conditioning.json")["emotions"][EMOTION]
    ref_seq = np.fromfile(cond_dir / cond["ref_seq"]["file"], dtype="<i8").reshape(cond["ref_seq"]["shape"])
    ref_bert = np.fromfile(cond_dir / cond["ref_bert"]["file"], dtype="<f4").reshape(cond["ref_bert"]["shape"])
    ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(cond["ssl_content"]["shape"])
    ref_audio = np.fromfile(cond_dir / cond["ref_audio"]["file"], dtype="<f4").reshape(cond["ref_audio"]["shape"])

    enc = V.Session(MODEL / "t2s_encoder.onnx")
    e = enc.run({
        V._pick(enc, "ref_seq"): ref_seq.astype(np.int64),
        V._pick(enc, "text_seq"): np.array([phone_ids], dtype=np.int64),
        V._pick(enc, "ref_bert"): ref_bert.astype(np.float32),
        V._pick(enc, "text_bert"): text_bert.astype(np.float32),
        V._pick(enc, "ssl_content"): ssl.astype(np.float32),
    })
    x = e[V._pick_out(enc, "x")]
    prompts = e[V._pick_out(enc, "prompts")]

    fs = V.Session(MODEL / "t2s_fsdec.onnx")
    f = fs.run({V._pick(fs, "x"): x, V._pick(fs, "prompts"): prompts})
    y = f[V._pick_out(fs, "y")].astype(np.int64)
    k = f[V._pick_out(fs, "k")]
    v = f[V._pick_out(fs, "v")]
    y_emb = f[V._pick_out(fs, "y_emb")]
    x_ex = f[V._pick_out(fs, "x_example")]
    n_prompt = int(prompts.shape[1])

    sd = V.Session(MODEL / "t2s_sdec.onnx")
    rng = np.random.default_rng(7)
    codes = [int(t) for t in y[0, n_prompt:].tolist() if 0 <= int(t) < stop_token]
    recent = list(codes)
    for _ in range(300):
        o = sd.run({
            V._pick(sd, "iy"): y, V._pick(sd, "ik"): k, V._pick(sd, "iv"): v,
            V._pick(sd, "iy_emb"): y_emb, V._pick(sd, "ix_example"): x_ex,
        })
        raw = o[V._pick_out(sd, "logits")]
        logits = raw.reshape(-1, raw.shape[-1])[-1].astype(np.float64)
        token = V.sample_token(logits, recent, temperature, top_k, top_p, penalty, rng, False)
        if token == stop_token or token >= logits.shape[0] - 1:
            break
        codes.append(token)
        recent.append(token)
        y = o[V._pick_out(sd, "y")].astype(np.int64)
        y[0, -1] = token
        y_emb = o[V._pick_out(sd, "y_emb")]
        k = o[V._pick_out(sd, "k")]
        v = o[V._pick_out(sd, "v")]

    print("phones", len(phone_ids), "codes", len(codes))
    vit = V.Session(MODEL / "vits.onnx")
    for n in (16, 32, 64, len(codes)):
        part = codes[:n]
        try:
            a = vit.run({
                V._pick(vit, "text_seq"): np.array([phone_ids], dtype=np.int64),
                V._pick(vit, "pred_semantic"): np.array([[part]], dtype=np.int64),
                V._pick(vit, "ref_audio"): ref_audio.astype(np.float32),
            })
            wav = a[V._pick_out(vit, "audio")].astype(np.float32).reshape(-1)
            print(f"prefix={n:3d} -> samples={wav.size:6d} ({wav.size/32000:.2f}s) peak={float(np.abs(wav).max()):.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"prefix={n:3d} -> ERROR {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

