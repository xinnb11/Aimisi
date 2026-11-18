"""裁掉 t2s_sdec 尾部的图内采样链（m07388 第 3 条）。

背景：`t2s_sdec.onnx` 末尾挂着 Softmax→ScatterElements→Where→ArgMax→Cast 的
“图内贪婪解码”，输出里还带 `samples` / `y` / `y_emb` 三个大张量。自回归循环里
我们其实只需要 `logits`（自己采样）+ `k` / `v`（KV cache 直传）。

本脚本用 `onnx.utils.extract_model` 保留到 `["logits", "k", "v"]` 为止的子图，
产出一个更小的裁剪版；`y` 与 `y_emb` 由端侧自己维护（y_emb 末行用
`conditioning/audio_embedding.f32` 查表，与 graph 内 `ar_audio_embedding` 一致，
实测余弦 0.99902）。

用法：
  python scripts/trim_sdec_tail.py --src D:\\爱弥斯\\_work\\publish\\t2s_sdec.onnx \
      --dst D:\\爱弥斯\\_work\\publish\\t2s_sdec.trim.onnx
"""

from __future__ import annotations

import argparse
import os
import time

import onnx
from onnx.utils import extract_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--keep", default="logits,k,v")
    parser.add_argument("--verify", action="store_true", help="裁剪后跑一次数值等价性检查")
    args = parser.parse_args()

    keep = [name.strip() for name in args.keep.split(",") if name.strip()]
    started = time.time()
    # onnx.utils.extract_model 要求显式给出输入名（传 None 会报 "Input tensor names shall not be empty!"）
    src_model = onnx.load(args.src, load_external_data=False)
    input_names = [i.name for i in src_model.graph.input]
    extract_model(
        args.src,
        args.dst,
        input_names=input_names,
        output_names=keep,
        check_model=True,
    )
    src_size = os.path.getsize(args.src)
    dst_size = os.path.getsize(args.dst)
    print(f"裁剪完成：{args.src} ({src_size/1e6:.1f} MB) -> {args.dst} ({dst_size/1e6:.1f} MB)")
    print(f"保留输出：{keep}；耗时 {time.time()-started:.1f}s")

    trimmed = onnx.load(args.dst)
    print("裁剪后图输出：", [o.name for o in trimmed.graph.output])
    if args.verify:
        verify(args.src, args.dst)
    return 0


def verify(src: str, dst: str) -> None:
    """用同一组随机但稳定的输入，比较裁剪前后 logits/k/v 是否一致。"""
    import numpy as np
    import onnxruntime as ort

    full = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    trimmed = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(20241118)
    steps, dim, layers = 8, 512, 24
    feed = {
        "iy": rng.integers(1, 1024, size=(1, steps), dtype=np.int64),
        "ik": rng.standard_normal((layers, steps, 1, dim), dtype=np.float32) * 0.1,
        "iv": rng.standard_normal((layers, steps, 1, dim), dtype=np.float32) * 0.1,
        "iy_emb": rng.standard_normal((1, steps, dim), dtype=np.float32) * 0.1,
        "ix_example": np.zeros((1, 94), dtype=np.float32),
    }
    out_full = full.run(None, feed)
    out_trim = trimmed.run(None, feed)
    names_full = [o.name for o in full.get_outputs()]
    names_trim = [o.name for o in trimmed.get_outputs()]
    for name in names_trim:
        a = out_full[names_full.index(name)]
        b = out_trim[names_trim.index(name)]
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        cos = float(
            (a.ravel() @ b.ravel())
            / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
        )
        print(f"  {name}: shape={a.shape} max|diff|={diff:.6g} cos={cos:.8f}")


if __name__ == "__main__":
    raise SystemExit(main())
