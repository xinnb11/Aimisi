#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端校验 / 参考实现：用已发布的 ONNX 包在桌面端跑通「文本 -> 语音」。

这个脚本有两个用途：

1. **验证发布包自洽**：读 ``conditioning/``、``conditioning/g2p/``、``bert/`` 三套产物，
   不依赖 GPT-SoVITS 仓库代码、不依赖 FunASR/CNHubert，纯 ``onnxruntime`` + ``numpy``。
2. **给 Android 端做行为基准**：``GptSoVitsOnnxEngine.kt`` 里的分词、BERT 特征对齐、
   自回归采样循环都必须与本脚本逐字节等价；任何一端改了语义，另一端必须同步。

用法::

    python verify_onnx_tts.py --model-dir D:\\爱弥斯\\_work\\publish ^
        --text "那些故事大多都有趣曲折，也有一些危险刺激。" --out preview.wav

    python verify_onnx_tts.py --model-dir <pkg> --text "叮，模型加载完毕" --emotion 开心 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

# BERT 词表里的特殊 token
CLS_ID = 101
SEP_ID = 102
PAD_ID = 0
UNK_ID = 100
BERT_DIM = 1024


# --------------------------------------------------------------------------- #
# WordPiece 分词（与 bert/tokenizer.json 的 Chinese 分支等价的最小子集）
# --------------------------------------------------------------------------- #
def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x3000 <= code <= 0x303F  # CJK 标点（，。！？ 等在这里或下面）
        or 0xFF00 <= code <= 0xFFEF  # 全角
    )


def _is_punct(ch: str) -> bool:
    if ch.isalnum():
        return False
    code = ord(ch)
    # ASCII 标点 / 通用标点 / CJK 标点
    return (
        33 <= code <= 47
        or 58 <= code <= 64
        or 91 <= code <= 96
        or 123 <= code <= 126
        or 0x2000 <= code <= 0x206F
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


class WordPiece:
    def __init__(self, vocab_path: Path, do_lower_case: bool = True) -> None:
        self.vocab: Dict[str, int] = {}
        with vocab_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                token = line.rstrip("\n")
                self.vocab[token] = index
        self.do_lower_case = do_lower_case
        self.max_chars = 100
        self.max_word_chars = 100

    def _wordpiece(self, word: str) -> List[str]:
        if not word:
            return []
        if len(word) > self.max_word_chars:
            return ["[UNK]"]
        pieces: List[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            matched: Optional[str] = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = "##" + candidate
                if candidate in self.vocab:
                    matched = candidate
                    break
                end -= 1
            if matched is None:
                return ["[UNK]"]
            pieces.append(matched)
            start = end
        return pieces

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for ch in text:
            if ch.isspace() or ch in "\u200b\ufeff":
                continue
            if _is_cjk(ch):
                tokens.append(ch)
            elif _is_punct(ch):
                tokens.append(ch)
            else:
                word = ch.lower() if self.do_lower_case else ch
                tokens.extend(self._wordpiece(word))
        return tokens[: self.max_chars]

    def encode(self, text: str) -> List[int]:
        ids = [self.vocab.get(token, UNK_ID) for token in self.tokenize(text)]
        return [CLS_ID] + ids + [SEP_ID]


# --------------------------------------------------------------------------- #
# ONNX 会话
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, path: Path, threads: int = 4) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.inputs = {item.name: item for item in self.sess.get_inputs()}
        self.outputs = {item.name: item for item in self.sess.get_outputs()}

    def run(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        names = [item.name for item in self.sess.get_outputs()]
        values = self.sess.run(names, feed)
        return dict(zip(names, values))


def _pick(session: Session, *candidates: str) -> str:
    for name in candidates:
        if name in session.inputs:
            return name
    raise SystemExit(
        f"{session.sess.get_modelmeta().custom_metadata_map.get('model', '?')} 缺少张量名 {candidates}；"
        f"实际输入={sorted(session.inputs)}"
    )


def _pick_out(session: Session, *candidates: str) -> str:
    for name in candidates:
        if name in session.outputs:
            return name
    # 单输出模型（如单独导出的第 21 层）直接用它唯一的输出。
    if len(session.outputs) == 1:
        return next(iter(session.outputs))
    raise SystemExit(f"缺少输出名 {candidates}；实际输出={sorted(session.outputs)}")


# --------------------------------------------------------------------------- #
# 文本前端：char2phone 字表 + BERT 特征按音素重复对齐
# --------------------------------------------------------------------------- #
class TextFrontend:
    def __init__(self, model_dir: Path, threads: int = 4) -> None:
        self.model_dir = model_dir
        conditioning = _load_json(model_dir / "conditioning" / "conditioning.json")
        g2p = conditioning.get("g2p") or {}
        char_rel = (g2p.get("char2phone") or {}).get("file") or "g2p/char2phone.json"
        char_path = _resolve_under(model_dir, "conditioning", char_rel)
        self.char2phone: Dict[str, List[int]] = {
            key: [int(v) for v in values]
            for key, values in _load_json(char_path).items()
        }
        self.unknown_id = int((g2p.get("symbols") or {}).get("unknown_id", 0))

        bert_entry = conditioning.get("bert") or {}
        bert_rel = (
            bert_entry.get("model")
            or conditioning.get("bert_model")
            or "bert/model.int8.onnx"
        )
        self.bert_path = _resolve_under(model_dir, ".", bert_rel)
        vocab_rel = bert_entry.get("vocab") or "bert/vocab.txt"
        vocab_path = _resolve_under(model_dir, ".", vocab_rel)
        self.wordpiece = WordPiece(vocab_path)
        self.bert: Optional[Session] = None
        self.bert_threads = threads

    def _ensure_bert(self) -> Session:
        if self.bert is None:
            self.bert = Session(self.bert_path, self.bert_threads)
        return self.bert

    def phones(self, text: str) -> Tuple[List[int], List[int]]:
        """返回 (音素 id 序列, 每个「有音素的字」的音素个数)。"""
        phone_ids: List[int] = []
        counts: List[int] = []
        for ch in text:
            if ch.isspace():
                continue
            ids = self.char2phone.get(ch)
            if ids is None and (ch.isalnum()):
                ids = [self.unknown_id]
            if not ids:
                continue
            phone_ids.extend(ids)
            counts.append(len(ids))
        return phone_ids, counts

    def bert_features(self, text: str) -> Tuple[np.ndarray, List[int]]:
        """返回 (音素级特征 [T,1024], 每字音素数)。"""
        phone_ids, counts = self.phones(text)
        if not phone_ids:
            raise SystemExit(f"文本无法转成音素：{text!r}")
        keep = self._keep_chars(text)
        if len(keep) != len(counts):
            raise SystemExit(
                f"字符-音素对齐不一致：可用音素字 {len(counts)} 个，BERT 对齐字 {len(keep)} 个"
            )
        session = self._ensure_bert()
        ids = session.inputs
        encoded = self.wordpiece.encode(text)
        feed = {
            _pick(session, "input_ids"): np.array([encoded], dtype=np.int64),
            _pick(session, "attention_mask"): np.ones((1, len(encoded)), dtype=np.int64),
            _pick(session, "token_type_ids"): np.zeros((1, len(encoded)), dtype=np.int64),
        }
        hidden = session.run(feed)[_pick_out(session, "last_hidden_state")][0]  # [S,1024]
        hidden = hidden[1 : 1 + len(self.wordpiece.tokenize(text))]

        # 字 -> BERT token 索引区间：CJK/标点 1 个 token，ASCII 单词可能多个
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for ch in text:
            if ch.isspace():
                continue
            if _is_cjk(ch) or _is_punct(ch):
                spans.append((cursor, cursor + 1))
                cursor += 1
            else:
                pieces = self.wordpiece._wordpiece(ch.lower())
                spans.append((cursor, cursor + len(pieces)))
                cursor += len(pieces)

        char_vectors: List[np.ndarray] = []
        for index, (begin, end) in enumerate(spans):
            vector = hidden[begin:end].mean(axis=0) if end > begin else hidden[0] * 0
            char_vectors.append(vector)

        # 只保留「有音素」的字，与 counts 一一对应
        kept: List[np.ndarray] = []
        char_cursor = 0
        for ch in text:
            if ch.isspace():
                continue
            ids_for_char = self.char2phone.get(ch)
            if ids_for_char is None and ch.isalnum():
                ids_for_char = [self.unknown_id]
            has_phones = bool(ids_for_char)
            if has_phones:
                kept.append(char_vectors[char_cursor])
            char_cursor += 1

        repeated = [np.repeat(vector[None, :], count, axis=0) for vector, count in zip(kept, counts)]
        features = np.concatenate(repeated, axis=0).astype(np.float32)
        return features, counts

    @staticmethod
    def _keep_chars(text: str) -> List[str]:
        return [ch for ch in text if not ch.isspace()]


# --------------------------------------------------------------------------- #
# 采样器（与 AemeathSampler.kt 同口径）
# --------------------------------------------------------------------------- #
def sample_token(
    logits: np.ndarray,
    recent: Sequence[int],
    temperature: float,
    top_k: int,
    top_p: float,
    penalty: float,
    rng: np.random.Generator,
    greedy: bool,
) -> int:
    scores = logits.astype(np.float64).copy()
    if penalty != 1.0:
        for token in set(recent[-64:]):
            if 0 <= token < scores.shape[0]:
                scores[token] = scores[token] / penalty if scores[token] > 0 else scores[token] * penalty
    if greedy:
        return int(np.argmax(scores))
    if temperature > 0:
        scores = scores / temperature
    scores -= scores.max()
    probs = np.exp(scores)
    total = probs.sum()
    probs = probs / total

    k = max(1, min(int(top_k), probs.shape[0]))
    order = np.argsort(-probs)
    kept = order[:k]
    kept_probs = probs[kept]
    if top_p < 1.0:
        cumulative = np.cumsum(kept_probs)
        cut = int(np.searchsorted(cumulative, top_p)) + 1
        cut = max(2, min(cut, kept.shape[0]))
        kept = kept[:cut]
        kept_probs = kept_probs[:cut]
    kept_probs = kept_probs / kept_probs.sum()
    return int(rng.choice(kept, p=kept_probs))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def synthesize(
    model_dir: Path,
    text: str,
    emotion: str,
    seed: int,
    greedy: bool,
    max_steps: int,
    recompute_ref_bert: bool = False,
) -> Tuple[np.ndarray, int, dict]:
    manifest = _load_json(model_dir / "manifest.json")
    params = manifest.get("sampling_params") or {}
    temperature = float(params.get("temperature", 0.85))
    top_k = int(params.get("top_k", 15))
    top_p = float(params.get("top_p", 0.9))
    penalty = float(params.get("repetition_penalty", 1.35))
    stop_token = int(manifest.get("eos_token", 1024))

    frontend = TextFrontend(model_dir)
    phone_ids, _ = frontend.phones(text)
    text_bert, _ = frontend.bert_features(text)
    if text_bert.shape[0] != len(phone_ids):
        raise SystemExit(f"text_bert 行数 {text_bert.shape[0]} != 音素数 {len(phone_ids)}")

    cond_dir = model_dir / "conditioning" / emotion
    cond = _load_json(model_dir / "conditioning" / "conditioning.json")["emotions"][emotion]
    ref_seq = np.fromfile(cond_dir / cond["ref_seq"]["file"], dtype="<i8").reshape(cond["ref_seq"]["shape"])
    ref_bert = np.fromfile(cond_dir / cond["ref_bert"]["file"], dtype="<f4").reshape(cond["ref_bert"]["shape"])
    if recompute_ref_bert:
        ref_text = cond.get("ref_text") or ""
        ref_bert, _ = frontend.bert_features(ref_text)
        if ref_bert.shape[0] != ref_seq.shape[1]:
            raise SystemExit(
                f"重算 ref_bert 行数 {ref_bert.shape[0]} != ref_seq 长度 {ref_seq.shape[1]}"
            )
    ssl_shape = cond["ssl_content"]["shape"]
    ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(ssl_shape)

    # 实验开关（m07150 复读取证）：把 ssl_content 截断为前 N 秒（25 Hz 帧率 → N*25 帧）。
    # 用来判断「模型开头照念参考音」是否由过长的参考音内容引起。
    ssl_seconds = float(os.environ.get("AEMEATH_SSL_SECONDS", "0") or 0)
    if 0 < ssl_seconds:
        frames = min(ssl.shape[-1], int(ssl_seconds * 25))
        if frames < ssl.shape[-1]:
            ssl = ssl[:, :, :frames]
            print(f"[probe] ssl_content 截断为前 {ssl_seconds}s（{frames} 帧）", file=sys.stderr)
    ref_shape = cond["ref_audio"]["shape"]
    ref_audio = np.fromfile(cond_dir / cond["ref_audio"]["file"], dtype="<f4").reshape(ref_shape)

    # 实验开关（m07150 复读取证）：把 ref_audio 裁剪成前 N 秒，用来判断「复读」到底由
    # 声码器侧的参考音引起，还是由 encoder 侧的 ssl_content 引起。默认不裁剪。
    ref_seconds = float(os.environ.get("AEMEATH_REF_AUDIO_SECONDS", "0") or 0)
    if 0 < ref_seconds and ref_audio.shape[-1] > int(ref_seconds * 32000):
        keep = int(ref_seconds * 32000)
        ref_audio = ref_audio[:, :keep]
        print(f"[probe] ref_audio 裁剪为前 {ref_seconds}s（{keep} 采样点）", file=sys.stderr)

    encoder = Session(model_dir / "t2s_encoder.onnx")
    # 实验开关（m07150 复读取证）：把 ref_seq / ref_bert 也换成「目标文本」的音素与 BERT，
    # 只保留参考音的音色（ref_audio）与内容条件（ssl_content）。若复读消失，说明复读来自
    # 「参考文本 + 参考音」这对条件，而不是来自声码器或采样。
    if os.environ.get("AEMEATH_REF_TEXT_AS_TARGET", "0") == "1":
        ref_seq_feed = np.array([phone_ids], dtype=np.int64)
        ref_bert_feed = text_bert.astype(np.float32)
        print("[probe] ref_seq/ref_bert 改用目标文本", file=sys.stderr)
    elif os.environ.get("AEMEATH_REF_TEXT_NEUTRAL"):
        neutral = os.environ["AEMEATH_REF_TEXT_NEUTRAL"]
        n_ids, _n_counts = frontend.phones(neutral)
        n_bert, _ = frontend.bert_features(neutral)
        ref_seq_feed = np.array([n_ids], dtype=np.int64)
        ref_bert_feed = n_bert.astype(np.float32)
        print(f"[probe] ref_seq/ref_bert 改用中性文本「{neutral}」（{len(n_ids)} 音素）", file=sys.stderr)
    else:
        ref_seq_feed = ref_seq.astype(np.int64)
        ref_bert_feed = ref_bert.astype(np.float32)
    # 实验开关（m07150 复读取证）：放大 text_bert 的权重，看能否让文本条件压过
    # 参考音内容（复读表现为模型开头照念参考音台词）。
    text_scale = float(os.environ.get("AEMEATH_TEXT_BERT_SCALE", "1") or 1)
    text_bert_feed = text_bert.astype(np.float32) * text_scale
    if text_scale != 1.0:
        print(f"[probe] text_bert 缩放 ×{text_scale}", file=sys.stderr)
    enc = encoder.run(
        {
            _pick(encoder, "ref_seq"): ref_seq_feed,
            _pick(encoder, "text_seq"): np.array([phone_ids], dtype=np.int64),
            _pick(encoder, "ref_bert"): ref_bert_feed,
            _pick(encoder, "text_bert"): text_bert_feed,
            _pick(encoder, "ssl_content"): ssl.astype(np.float32),
        }
    )
    x = enc[_pick_out(encoder, "x")]
    prompts = enc[_pick_out(encoder, "prompts")]

    # 实验开关（m07150）：参考音语义码前缀过长时，短文本会被模型顺着参考音内容
    # 继续往下说（真机/桌面都能复现「也有一些危险刺激」）。设 AEMEATH_PROMPT_LIMIT=N
    # 只保留 prompts 的尾部 N 个码，用来量它对复读的影响；默认 0 表示不截断。
    prompt_limit = int(os.environ.get("AEMEATH_PROMPT_LIMIT", "0") or 0)
    if 0 < prompt_limit < prompts.shape[1]:
        prompts = prompts[:, prompts.shape[1] - prompt_limit:]
        print(f"[probe] prompts 截断为尾部 {prompt_limit} 码", file=sys.stderr)

    fsdec = Session(model_dir / "t2s_fsdec.onnx")
    first = fsdec.run(
        {
            _pick(fsdec, "x"): x.astype(np.float32),
            _pick(fsdec, "prompts"): prompts.astype(np.int64),
        }
    )
    y = first[_pick_out(fsdec, "y")].astype(np.int64)
    k_cache = first[_pick_out(fsdec, "k")]
    v_cache = first[_pick_out(fsdec, "v")]
    y_emb = first[_pick_out(fsdec, "y_emb")]
    x_example = first[_pick_out(fsdec, "x_example")]
    prompt_count = int(prompts.shape[1])

    sdec = Session(model_dir / "t2s_sdec.onnx")
    rng = np.random.default_rng(seed)
    generated: List[int] = [
        int(token) for token in y[0, prompt_count:].tolist() if 0 <= int(token) < stop_token
    ]
    recent: List[int] = list(generated)
    steps = 0
    while steps < max_steps:
        steps += 1
        out = sdec.run(
            {
                _pick(sdec, "iy"): y.astype(np.int64),
                _pick(sdec, "ik"): k_cache,
                _pick(sdec, "iv"): v_cache,
                _pick(sdec, "iy_emb"): y_emb,
                _pick(sdec, "ix_example"): x_example,
            }
        )
        raw_logits = out[_pick_out(sdec, "logits")]
        logits = raw_logits.reshape(-1, raw_logits.shape[-1])[-1].astype(np.float64)
        token = sample_token(logits, recent, temperature, top_k, top_p, penalty, rng, greedy)
        if token == stop_token or token >= logits.shape[0] - 1:
            break
        generated.append(token)
        recent.append(token)
        y = out[_pick_out(sdec, "y")].astype(np.int64)
        y[0, -1] = token  # 用运行时采样替换图内 argmax，KV cache 才会跟随采样 token
        y_emb = out[_pick_out(sdec, "y_emb")]
        k_cache = out[_pick_out(sdec, "k")]
        v_cache = out[_pick_out(sdec, "v")]

    if not generated:
        raise SystemExit("自回归循环没有产生任何语义码")

    vocoder = Session(model_dir / "vits.onnx")
    audio = vocoder.run(
        {
            _pick(vocoder, "text_seq"): np.array([phone_ids], dtype=np.int64),
            _pick(vocoder, "pred_semantic"): np.array([[generated]], dtype=np.int64),
            _pick(vocoder, "ref_audio"): ref_audio.astype(np.float32),
        }
    )
    wav = audio[_pick_out(vocoder, "audio")].astype(np.float32).reshape(-1)

    # 诊断（m07150）：统计生成码里「与参考音 prompts 相同的 5-gram」占比，
    # 用来判断「复读参考音」是否真的复用了参考音的语义码序列。
    prompt_seq = [int(tok) for tok in prompts.reshape(-1).tolist()]
    echo_windows = 0
    if len(prompt_seq) >= 5 and len(generated) >= 5:
        prompt_grams = {
            tuple(prompt_seq[i : i + 5]) for i in range(len(prompt_seq) - 4)
        }
        for i in range(len(generated) - 4):
            if tuple(generated[i : i + 5]) in prompt_grams:
                echo_windows += 1
    info = {
        "phones": len(phone_ids),
        "steps": steps,
        "codes": len(generated),
        "first_codes": generated[:8],
        "prompt_gram_hits": echo_windows,
        "peak": float(np.abs(wav).max()) if wav.size else 0.0,
        "samples": int(wav.size),
        "sample_rate": int(manifest.get("sample_rate", 32000)),
        "emotion": emotion,
        "greedy": greedy,
    }
    return wav, int(manifest.get("sample_rate", 32000)), info


def write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    scale = (0.89 / peak) if peak > 0.89 else 1.0
    data = np.clip(wav * scale, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_under(root: Path, sub: str, relative: str) -> Path:
    candidate = (root / sub / relative).resolve()
    return candidate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用已发布的 ONNX 包做端到端语音合成校验")
    parser.add_argument("--model-dir", required=True, help="含 manifest.json 的模型包目录")
    parser.add_argument("--text", required=True, help="待合成文本")
    parser.add_argument("--emotion", default="平静", help="平静/开心/温柔/调侃")
    parser.add_argument("--out", default="verify_tts.wav", help="输出 wav 路径")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true", help="强制 argmax（仅用于排障对比）")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--recompute-ref-bert", action="store_true",
                        help="用包内 BERT 重算参考文本特征（修正发布包里 ref_bert 的重复行缺陷）")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    wav, sample_rate, info = synthesize(
        model_dir, args.text, args.emotion, args.seed, args.greedy, args.max_steps,
        recompute_ref_bert=args.recompute_ref_bert,
    )
    write_wav(Path(args.out), wav, sample_rate)
    duration = info["samples"] / sample_rate if sample_rate else 0.0
    print(json.dumps({**info, "seconds": round(duration, 3), "out": str(Path(args.out).resolve())},
                     ensure_ascii=False, indent=2))
    if info["peak"] < 0.05 or duration < 0.3:
        print("[warn] 输出疑似静音：检查 text_bert 是否有真实 BERT 特征（全零会立刻 EOS）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
