"""临时调试：对比桌面与真机 sdec 第一步 logits。

用法：
  python scripts/dev/dbg_step1.py "文本" [--emotion 平静]

输出前 12 个 token 的 logits 排序、y/y_emb 形状，用于和真机 logcat 的
`step=1 top5=... iy=... iyEmb=... k=...` 行逐位对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_onnx_tts as V  # noqa: E402


def main() -> None:
    model_dir = Path(r"D:\爱弥斯\_work\publish")
    text = sys.argv[1] if len(sys.argv) > 1 else "今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。"
    emotion = "平静"
    for i, arg in enumerate(sys.argv):
        if arg == "--emotion" and i + 1 < len(sys.argv):
            emotion = sys.argv[i + 1]

    frontend = V.TextFrontend(model_dir)
    phone_ids, _ = frontend.phones(text)
    text_bert, _ = frontend.bert_features(text)
    cond = V._load_json(model_dir / "conditioning" / "conditioning.json")["emotions"][emotion]
    cond_dir = model_dir / "conditioning" / emotion
    ref_seq = np.fromfile(cond_dir / cond["ref_seq"]["file"], dtype="<i8").reshape(cond["ref_seq"]["shape"])
    ref_bert = np.fromfile(cond_dir / cond["ref_bert"]["file"], dtype="<f4").reshape(cond["ref_bert"]["shape"])
    ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(cond["ssl_content"]["shape"])

    enc = V.Session(model_dir / "t2s_encoder.onnx")
    out = enc.run(
        {
            V._pick(enc, "ref_seq"): ref_seq.astype(np.int64),
            V._pick(enc, "text_seq"): np.array([phone_ids], dtype=np.int64),
            V._pick(enc, "ref_bert"): ref_bert.astype(np.float32),
            V._pick(enc, "text_bert"): text_bert.astype(np.float32),
            V._pick(enc, "ssl_content"): ssl.astype(np.float32),
        }
    )
    x = out[V._pick_out(enc, "x")]
    prompts = out[V._pick_out(enc, "prompts")]
    print("phones", len(phone_ids), "x", x.shape, "prompts", prompts.shape)
    print("x[0,0,:4]", np.round(x[0, 0, :4], 4).tolist(), "prompts[0,:4]", prompts[0, :4].tolist())

    fs = V.Session(model_dir / "t2s_fsdec.onnx")
    first = fs.run({V._pick(fs, "x"): x.astype(np.float32), V._pick(fs, "prompts"): prompts.astype(np.int64)})
    y = first[V._pick_out(fs, "y")].astype(np.int64)
    y_emb = first[V._pick_out(fs, "y_emb")]
    k = first[V._pick_out(fs, "k")]
    v = first[V._pick_out(fs, "v")]
    x_ex = first[V._pick_out(fs, "x_example")]
    print("fsdec y", y.shape, "yEmb", y_emb.shape, "k", k.shape)
    print("y first4", y[0, :4].tolist(), "y last4", y[0, -4:].tolist())
    print("yEmb[0,0,:4]", np.round(y_emb[0, 0, :4], 4).tolist())

    sd = V.Session(model_dir / "t2s_sdec.onnx")
    step = sd.run(
        {
            V._pick(sd, "iy"): y,
            V._pick(sd, "ik"): k,
            V._pick(sd, "iv"): v,
            V._pick(sd, "iy_emb"): y_emb,
            V._pick(sd, "ix_example"): x_ex,
        }
    )
    logits = step[V._pick_out(sd, "logits")]
    logits = logits.reshape(-1, logits.shape[-1])[-1]
    top = np.argsort(-logits)[:8]
    print("step1 top8", [(int(i), round(float(logits[i]), 3)) for i in top])
    out_y = step[V._pick_out(sd, "y")]
    out_emb = step[V._pick_out(sd, "y_emb")]
    print("sdec y", out_y.shape, "last4", out_y[0, -4:].tolist())
    print("sdec yEmb last4x4", np.round(out_emb[0, -1, :4], 4).tolist(), "shape", out_emb.shape)
    print("samples", step[V._pick_out(sd, "samples")].tolist() if "samples" in sd.outputs else "n/a")


if __name__ == "__main__":
    main()
