"""按 GPT-SoVITS 官方口径重算 ref_bert：

- BERT 取 `hidden_states[-3]`（第 21 层 LayerNorm 输出，24 层中的倒数第三层），
  与官方 `1-get-text.py` 的 `torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]` 等价；
- 去掉首尾 `[CLS]`/`[SEP]`，按 `word2ph`（每个字的音素数）repeat 到音素级。

用法：
    python scripts/dev/recompute_ref_bert.py            # 只体检 + 生成 layer21 onnx
    python scripts/dev/recompute_ref_bert.py --write    # 并把 ref_bert.f32 覆盖进 conditioning/
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
COND = PUBLISH / "conditioning"
BERT = PUBLISH / "bert" / "model.int8.onnx"
LAYER_TENSOR = "/model/encoder/layer.21/output/LayerNorm/LayerNormalization_output_0"
CLS_ID, SEP_ID, UNK_ID = 101, 102, 100


def build_layer_model(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    if dst.exists():
        print(f"[skip] {dst.name} 已存在")
        return
    # 用 extract_model 裁剪出「输入 -> 第21层输出」子图
    onnx.utils.extract_model(
        str(src),
        str(dst),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=[LAYER_TENSOR],
    )
    print(f"[ok] 已导出 {dst} ({dst.stat().st_size/1e6:.1f} MB)")


def load_vocab(path: Path) -> dict:
    vocab = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        piece = line.rstrip("\n")
        if piece:
            vocab[piece] = index
    return vocab


def wordpiece(text: str, vocab: dict) -> list:
    ids = []
    for ch in text:
        piece = ch
        if piece not in vocab:
            piece = "[UNK]"
        ids.append(vocab.get(piece, UNK_ID))
    return ids


def main() -> int:
    write = "--write" in sys.argv
    layer_model = PUBLISH / "bert" / "model.layer21.int8.onnx"
    build_layer_model(BERT, layer_model)

    cond = json.loads((COND / "conditioning.json").read_text(encoding="utf-8"))
    g2p = cond["g2p"]
    char2phone = json.loads((COND / g2p["char2phone"]["file"]).read_text(encoding="utf-8"))
    if isinstance(char2phone, dict) and "table" in char2phone:
        char2phone = char2phone["table"]

    vocab = load_vocab(PUBLISH / "bert" / "vocab.txt")
    session = ort.InferenceSession(str(layer_model), providers=["CPUExecutionProvider"])
    out_name = session.get_outputs()[0].name
    print("[model] layer21 output:", out_name, session.get_outputs()[0].shape)

    for emotion, meta in cond["emotions"].items():
        text = meta.get("ref_text") or meta.get("text") or ""
        if not text:
            print(f"[warn] {emotion}: conditioning.json 里没有 ref_text，跳过")
            continue
        counts = [len(char2phone.get(ch, [])) for ch in text]
        ids = [CLS_ID] + wordpiece(text, vocab) + [SEP_ID]
        n = len(ids)
        mask = [1] * n
        types = [0] * n
        hidden = session.run(
            [out_name],
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
                "token_type_ids": np.array([types], dtype=np.int64),
            },
        )[0][0][1:-1]  # 去掉 [CLS]/[SEP]

        rows = []
        for index, count in enumerate(counts):
            if count <= 0:
                continue
            rows.append(np.repeat(hidden[index][None, :], count, axis=0))
        feature = np.concatenate(rows, axis=0).astype(np.float32)

        old_path = COND / emotion / "ref_bert.f32"
        old = np.fromfile(old_path, dtype=np.float32).reshape(-1, 1024) if old_path.exists() else None
        print(f"\n[{emotion}] text={text}")
        print("  音素数:", sum(counts), " per-char:", counts)
        print("  新特征:", feature.shape, "norm=", float(np.linalg.norm(feature)))
        if old is not None:
            same = np.allclose(old, feature)
            dup_rows = int((np.linalg.norm(old[20:] - old[20], axis=1) < 1e-6).sum())
            print("  旧特征:", old.shape, " norm=", float(np.linalg.norm(old)),
                  " 与新版一致:", same, " 旧版第20行之后重复行数:", dup_rows)
        if write:
            feature.tofile(old_path)
            print("  [write]", old_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
