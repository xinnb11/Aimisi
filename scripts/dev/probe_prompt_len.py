"""参考音语义码前缀长度对「复读参考音」的影响（m07150 实验）。

现象：短文本合成时，开头会念出参考音台词的一部分（「也有一些危险刺激」）。
桌面参考实现（scripts/verify_onnx_tts.py）同样复现，说明是模型行为，不是设备 bug：
sdec 的 `prompts` 是 118 个参考音语义码，作为自回归前缀喂给模型，模型会把它们当
「已经说过的话」继续往下衔接，短文本时更容易顺着参考音内容往下说。

本脚本把 `prompts` 截到不同长度（只保留尾部 k 个码）重新跑，看复读是否消失、
码数是否回到与文本长度相称的量级。用法：

    python scripts/dev/probe_prompt_len.py --text "..." --emotion 温柔
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "scripts" / "verify_onnx_tts.py"


def load_verify():
    spec = importlib.util.spec_from_file_location("verify_onnx_tts", VERIFY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"D:\爱弥斯\_work\publish")
    ap.add_argument("--text", default="模型验证，爱弥斯语音合成正常，这条音频可以正常播放。")
    ap.add_argument("--emotion", default="温柔")
    ap.add_argument("--out-dir", default="scripts/dev/fixtures")
    ap.add_argument("--lengths", default="118,80,48,24")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    v = load_verify()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复用参考实现的文本前端 + 条件包读取，只替换自回归前缀长度
    front = v.TextFrontend(model_dir)
    cond = v.load_conditioning(model_dir, args.emotion)
    phone_ids = front.phones(args.text)
    text_bert = front.bert_features(args.text, phone_ids)

    encoder = v.Session(model_dir / "t2s_encoder.onnx")
    outs = encoder.run(
        {
            v._pick(encoder, "ref_seq"): cond["ref_seq"].astype(np.int64),
            v._pick(encoder, "text_seq"): np.array([phone_ids], dtype=np.int64),
            v._pick(encoder, "ref_bert"): cond["ref_bert"].astype(np.float32),
            v._pick(encoder, "text_bert"): text_bert.astype(np.float32),
            v._pick(encoder, "ssl_content"): cond["ssl_content"].astype(np.float32),
        }
    )
    x = outs[v._pick_out(encoder, "x")].astype(np.float32)
    prompts = outs[v._pick_out(encoder, "prompts")].astype(np.int64)
    total = int(prompts.shape[1])
    print(f"phones={len(phone_ids)} prompts={total} x={tuple(x.shape)}")

    fsdec = v.Session(model_dir / "t2s_fsdec.onnx")
    sdec = v.Session(model_dir / "t2s_sdec.onnx")
    vocoder = v.Session(model_dir / "vits.onnx")

    for length in [int(item) for item in args.lengths.split(",") if item.strip()]:
        k = max(8, min(total, length))
        prefix = prompts[:, total - k:]
        first = fsdec.run(
            {
                v._pick(fsdec, "x"): x,
                v._pick(fsdec, "prompts"): prefix,
            }
        )
        y = first[v._pick_out(fsdec, "y")].astype(np.int64)
        k_cache = first[v._pick_out(fsdec, "k")]
        v_cache = first[v._pick_out(fsdec, "v")]
        y_emb = first[v._pick_out(fsdec, "y_emb")]
        x_example = first[v._pick_out(fsdec, "x_example")]

        rng = np.random.default_rng(args.seed)
        generated: list[int] = [
            int(tok) for tok in y[0, k:].tolist() if 0 <= int(tok) < v.stop_token
        ]
        recent: list[int] = list(generated)
        steps = 0
        while steps < 400:
            steps += 1
            out = sdec.run(
                {
                    v._pick(sdec, "iy"): y.astype(np.int64),
                    v._pick(sdec, "ik"): k_cache,
                    v._pick(sdec, "iv"): v_cache,
                    v._pick(sdec, "iy_emb"): y_emb,
                    v._pick(sdec, "ix_example"): x_example,
                }
            )
            raw = out[v._pick_out(sdec, "logits")]
            logits = raw.reshape(-1, raw.shape[-1])[-1].astype(np.float64)
            token = v.sample_token(logits, recent, 0.85, 15, 0.9, 1.35, rng, False)
            if token == v.stop_token or token >= logits.shape[0] - 1:
                break
            generated.append(token)
            recent.append(token)
            y = out[v._pick_out(sdec, "y")].astype(np.int64)
            y[0, -1] = token
            y_emb = out[v._pick_out(sdec, "y_emb")]
            k_cache = out[v._pick_out(sdec, "k")]
            v_cache = out[v._pick_out(sdec, "v")]

        audio = vocoder.run(
            {
                v._pick(vocoder, "text_seq"): np.array([phone_ids], dtype=np.int64),
                v._pick(vocoder, "pred_semantic"): np.array([[generated]], dtype=np.int64),
                v._pick(vocoder, "ref_audio"): cond["ref_audio"].astype(np.float32),
            }
        )[v._pick_out(vocoder, "audio")].astype(np.float32).reshape(-1)
        path = out_dir / f"_promptlen_{k}.wav"
        v.write_wav(path, audio, 32000)
        print(f"prefix={k} steps={steps} codes={len(generated)} audio={audio.size / 32000:.2f}s -> {path.name}")


if __name__ == "__main__":
    sys.exit(main())
