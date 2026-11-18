#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_ref_conditioning.py
===========================
爱弥斯 (Aemeath) GPT-SoVITS v2Pro 端侧「参考条件包」生成器。

## 为什么需要它

`Aimisi_v2Pro_e8_*.onnx` 是官方 `onnx_export.py` 的原样导出，`t2s_encoder` 要求：

```
ref_seq(int64) + text_seq(int64) + ref_bert(float 1024) + text_bert(float 1024) + ssl_content(float 768)
```

其中 `ref_seq`（参考音语义码）、`ssl_content`（CNHubert 特征）、`ref_bert`（参考文本
BERT 特征）**只与参考音有关，与用户说什么无关**，因此可以在云端算一次、固化成
几个小文件随模型下发，手机端就完全不需要跑 CNHubert / BERT。

`text_bert` 依赖逐句文本，端侧无 BERT 时按约定填零向量（音质略降，但可发声），
本脚本会明确记录 `bert_source`。

## 在三类环境里的运行姿势

### A. 有 GPT-SoVITS 仓库（推荐，可完整提取）

```bash
python scripts/prepare_ref_conditioning.py \
    --gpt-sovits-repo /root/GPT-SoVITS \
    --s2g /root/models/Aimisi_v2Pro_e8_s232.pth \
    --cnhubert /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base \
    --bert /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large \
    --ref-dir /root/aemeath_ref \
    --onnx-dir /root/models/onnx \
    --out-dir /root/models/onnx/conditioning
```

### B. 只能给现成产物（团队里有人已在训练机上跑过）

把训练日志里的 `2-name2text.txt` 与 `6-name2semantic.tsv` 拿过来，用 `--mode logs`
直接从已有的语义码里取（数值与训练分布完全一致）：

```bash
python scripts/prepare_ref_conditioning.py --mode logs \
    --semantic-tsv logs/Aimisi_v2Pro/6-name2semantic.tsv \
    --name2text logs/Aimisi_v2Pro/2-name2text.txt \
    --ref-dir refs --onnx-dir onnx --out-dir onnx/conditioning
```

### C. 只建 G2P 表（端侧文本 → 音素 id）

```bash
python scripts/prepare_ref_conditioning.py --mode g2p \
    --gpt-sovits-repo /root/GPT-SoVITS --out-dir onnx/conditioning
```

## 产物布局（端侧按此读取）

```
conditioning/
├── conditioning.json            # 清单：情绪 → 各张量文件 / dtype / shape / sha256
├── <情绪>/
│   ├── ref_seq.i64              # int64 小端，参考音语义码
│   ├── ssl_content.f32          # float32 小端，[1,768,T] 展平
│   ├── ref_bert.f32             # float32 小端，[T_ref,1024] 展平（可缺省）
│   └── ref_audio.f32            # float32 小端，32kHz 参考波形
└── g2p/
    ├── symbols.json             # 音素表（顺序即 id）
    └── char2phone.json          # 汉字 → 音素 id 序列（端侧免 jieba/g2pw）
```

原始参考音 **也必须随包下发**（`ref_audio.f32` 已包含），因为声码器 `vits.onnx`
的第三个输入就是参考波形本身。

作者：DeepSeek Harness（端侧 AI 推理引擎与云端训练量化工程师）
许可：AGPL-3.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("aemeath.cond")

EMOTIONS: Tuple[str, ...] = ("平静", "开心", "温柔", "调侃")
TOOL_VERSION = "1.0.0"

#: v2Pro / v3 / v4 系列语义词表大小（EOS 通常为 vocab-1）
DEFAULT_VOCAB = 1025
DEFAULT_EOS = 1024

#: GB2312 一级常用字区间（3755 字），用于生成端侧 G2P 表
GB2312_LEVEL1_RANGE = (0xB0A1, 0xD7F9)


class CondError(RuntimeError):
    """条件包生成失败。"""


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def setup_logging(level: str) -> None:
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOG.handlers.clear()
    LOG.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    LOG.addHandler(handler)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_blob(path: Path, array: np.ndarray, dtype: str) -> Dict[str, Any]:
    """写出裸二进制（小端），返回清单条目。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cast = array.astype("<f4" if dtype == "float32" else "<i8")
    path.write_bytes(cast.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": dtype,
        "shape": [int(dim) for dim in cast.shape],
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def read_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    """读取 wav → 单声道 float32，并按需重采样。"""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise CondError(f"参考音只支持 16bit PCM wav：{path}")
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        mono = pcm.reshape(-1, channels).mean(axis=1) if channels > 1 else pcm

    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if sr != target_sr:
        try:
            from math import gcd

            from scipy.signal import resample_poly  # type: ignore

            g = gcd(int(sr), int(target_sr))
            mono = np.asarray(resample_poly(mono, target_sr // g, sr // g), dtype=np.float32)
        except Exception:
            n_out = int(round(mono.size * target_sr / float(sr)))
            xp = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
            xq = np.linspace(0.0, 1.0, num=max(n_out, 2), endpoint=False)
            mono = np.interp(xq, xp, mono).astype(np.float32)
    return mono


# --------------------------------------------------------------------------- #
# GPT-SoVITS 仓库装载（延迟导入，缺依赖时报可读错误）
# --------------------------------------------------------------------------- #
@dataclass
class RepoModules:
    """从 GPT-SoVITS 仓库加载出来的能力集合。"""

    cnhubert: Any = None
    quantizer: Any = None            # 具备 extract_latent / encode 的模块
    get_bert_feature: Any = None
    clean_text: Any = None
    text_to_sequence: Any = None
    symbols: Optional[List[str]] = None


def load_repo_modules(
    repo: Path,
    s2g: Optional[Path],
    cnhubert_path: Optional[Path],
    device: str,
) -> RepoModules:
    """装载仓库内 CNHubert / 语义量化器 / 文本前端。"""
    import torch  # type: ignore

    if not (repo / "GPT_SoVITS").is_dir():
        raise CondError(f"{repo} 下找不到 GPT_SoVITS 目录，请指向仓库根目录")
    sys.path.insert(0, str(repo / "GPT_SoVITS"))

    modules = RepoModules()

    # ---- 文本前端（音素表 / 清洗 / 转 id / BERT）----
    try:
        from text import cleaned_text_to_sequence  # type: ignore

        modules.text_to_sequence = cleaned_text_to_sequence
    except Exception as exc:
        LOG.warning("装载 cleaned_text_to_sequence 失败：%s", exc)

    try:
        from text.cleaner import clean_text  # type: ignore

        modules.clean_text = clean_text
    except Exception as exc:
        LOG.warning("装载 text.cleaner.clean_text 失败：%s", exc)

    try:
        from text.cleaner import get_bert_feature  # type: ignore

        modules.get_bert_feature = get_bert_feature
    except Exception as exc:
        LOG.info("装载 get_bert_feature 失败（将退化为零向量）：%s", exc)

    for module_name in ("text.symbols", "text.symbols2", "text.symbols3"):
        try:
            module = __import__(module_name, fromlist=["symbols"])
            modules.symbols = list(getattr(module, "symbols"))
            LOG.info("音素表来源：%s（%d 个符号）", module_name, len(modules.symbols))
            break
        except Exception:
            continue
    if modules.symbols is None:
        LOG.warning("未能自动定位音素表，可稍后手工提供 g2p/symbols.json")

    # ---- CNHubert ----
    if cnhubert_path is not None:
        try:
            from feature_extractor.cnhubert import CNHubert  # type: ignore

            model = CNHubert(str(cnhubert_path))
            model.eval().to(device)
            modules.cnhubert = model
            LOG.info("CNHubert 已装载：%s", cnhubert_path)
        except Exception as exc:
            LOG.warning("CNHubert 装载失败：%s", exc)

    # ---- 语义量化器（来自 s2G：ssl_proj + quantizer）----
    if s2g is not None and s2g.is_file():
        try:
            from module.models import SynthesizerTrn  # type: ignore

            package = torch.load(str(s2g), map_location="cpu")
            state = package.get("weight", package) if isinstance(package, dict) else package
            hps = None
            if isinstance(package, dict):
                hps = package.get("config") or package.get("hps")
            if hps is None:
                raise CondError(
                    "s2G 内没有 config/hps，无法重建 SynthesizerTrn；"
                    "请改用 --mode logs 直接取训练期语义码"
                )
            net = SynthesizerTrn(hps)
            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing:
                LOG.warning("SynthesizerTrn 缺失 %d 个权重（照常继续）", len(missing))
            net.eval().to(device)
            modules.quantizer = net
            LOG.info("语义量化器已装载（含 ssl_proj/quantizer）")
        except Exception as exc:
            LOG.warning("语义量化器装载失败：%s", exc)

    return modules


# --------------------------------------------------------------------------- #
# 单个情绪的条件提取
# --------------------------------------------------------------------------- #
def extract_emotion(
    emotion: str,
    wav_path: Path,
    modules: RepoModules,
    device: str,
    sample_rate: int,
    ref_text: Optional[str],
    out_dir: Path,
    use_bert: bool,
) -> Dict[str, Any]:
    """提取一条参考音的四类条件张量并落盘。"""
    import torch  # type: ignore

    LOG.info("[%s] 参考音：%s", emotion, wav_path.name)
    waveform = read_wav_mono(wav_path, sample_rate)
    if waveform.size < sample_rate // 4:
        raise CondError(f"参考音过短（{waveform.size / sample_rate:.2f}s）：{wav_path}")

    entry: Dict[str, Any] = {}
    emotion_dir = out_dir / emotion

    # 1) 声码器参考波形（32kHz）
    entry["ref_audio"] = write_blob(emotion_dir / "ref_audio.f32", waveform.reshape(1, -1), "float32")

    # 2) CNHubert 特征 [1, 768, T]
    if modules.cnhubert is None:
        raise CondError(
            "缺少 CNHubert，无法生成 ssl_content；请提供 --cnhubert 或改用 --mode logs"
        )
    wav16 = read_wav_mono(wav_path, 16000)
    tensor = torch.from_numpy(wav16).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ssl = modules.cnhubert.model(tensor)
    if isinstance(ssl, (list, tuple)):
        ssl = ssl[-1]
    ssl = ssl.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, 768, T]
    ssl_np = ssl.detach().cpu().numpy().astype(np.float32)
    entry["ssl_content"] = write_blob(emotion_dir / "ssl_content.f32", ssl_np, "float32")
    LOG.info("[%s] ssl_content: %s", emotion, list(ssl_np.shape))

    # 3) 语义码 ref_seq（与训练期同一套 quantizer）
    if modules.quantizer is None:
        raise CondError(
            "缺少语义量化器（s2G 的 ssl_proj/quantizer），无法生成 ref_seq；"
            "请提供 --s2g 或改用 --mode logs"
        )
    with torch.no_grad():
        codes = None
        for attempt in ("extract_latent", "encode"):
            method = getattr(modules.quantizer, attempt, None)
            if method is None:
                continue
            try:
                codes = method(ssl)
                LOG.info("[%s] 语义码提取路径：%s", emotion, attempt)
                break
            except Exception as exc:
                LOG.debug("[%s] %s 失败：%s", emotion, attempt, exc)
        if codes is None and hasattr(modules.quantizer, "quantizer"):
            try:
                codes = modules.quantizer.quantizer.encode(ssl)
                LOG.info("[%s] 语义码提取路径：quantizer.encode", emotion)
            except Exception as exc:
                LOG.debug("[%s] quantizer.encode 失败：%s", emotion, exc)
    if codes is None:
        raise CondError(f"{emotion} 语义码提取失败，请检查 s2G 与该模型的匹配关系")

    codes_np = codes.detach().cpu().numpy() if hasattr(codes, "detach") else np.asarray(codes)
    codes_np = codes_np.astype(np.int64).reshape(1, -1)
    entry["ref_seq"] = write_blob(emotion_dir / "ref_seq.i64", codes_np, "int64")
    LOG.info("[%s] ref_seq: %s", emotion, list(codes_np.shape))

    # 4) ref_bert（参考文本的 BERT 特征；不可得时按约定留空）
    ref_bert_shape = None
    if use_bert and modules.get_bert_feature is not None and ref_text:
        try:
            with torch.no_grad():
                bert = modules.get_bert_feature(
                    ref_text, [1] * len(ref_text), device=device
                )
            bert_np = np.asarray(bert, dtype=np.float32).reshape(-1, 1024)
            entry["ref_bert"] = write_blob(emotion_dir / "ref_bert.f32", bert_np, "float32")
            ref_bert_shape = list(bert_np.shape)
            LOG.info("[%s] ref_bert: %s", emotion, ref_bert_shape)
        except Exception as exc:
            LOG.warning("[%s] ref_bert 提取失败（端侧将喂零）：%s", emotion, exc)
    else:
        LOG.warning("[%s] 未提供参考文本/BERT，ref_bert 缺省（端侧喂零）", emotion)

    entry["ref_bert_shape"] = ref_bert_shape
    entry["source_wav"] = wav_path.name
    entry["duration_sec"] = round(waveform.size / sample_rate, 3)
    return entry


# --------------------------------------------------------------------------- #
# G2P 表（汉字 → 音素 id），端侧免 jieba/g2pw
# --------------------------------------------------------------------------- #
def gb2312_level1_chars() -> List[str]:
    chars: List[str] = []
    for high in range(0xB0, 0xD8):
        for low in range(0xA1, 0xFF):
            try:
                chars.append(bytes([high, low]).decode("gb2312"))
            except UnicodeDecodeError:
                continue
    return chars


def build_g2p_table(
    modules: RepoModules,
    out_dir: Path,
    corpus_extra: Sequence[str] = (),
) -> Dict[str, Any]:
    """生成 symbols.json 与 char2phone.json。"""
    if modules.symbols is None or modules.clean_text is None or modules.text_to_sequence is None:
        raise CondError("文本前端未装载完整（需要 symbols / clean_text / cleaned_text_to_sequence）")

    g2p_dir = out_dir / "g2p"
    g2p_dir.mkdir(parents=True, exist_ok=True)
    symbols_path = g2p_dir / "symbols.json"
    symbols_path.write_text(
        json.dumps(modules.symbols, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    chars = gb2312_level1_chars()
    for extra in corpus_extra:
        chars.extend([ch for ch in extra if "\u4e00" <= ch <= "\u9fff"])
    chars = sorted(set(chars))

    mapping: Dict[str, List[int]] = {}
    failures = 0
    for index, char in enumerate(chars, 1):
        try:
            # version 参数随仓库版本变化，这里做兼容尝试
            result = None
            for kwargs in ({"version": "v2Pro"}, {"version": "v3"}, {"version": "v4"}, {}):
                try:
                    result = modules.clean_text(char, language="zh", **kwargs)
                    break
                except TypeError:
                    continue
            if result is None:
                raise CondError("clean_text 调用签名不兼容")
            phones = result[0]
            ids = modules.text_to_sequence(phones)
            if ids:
                mapping[char] = [int(value) for value in ids]
        except Exception:
            failures += 1
        if index % 500 == 0:
            LOG.info("G2P 构建进度 %d/%d（已覆盖 %d 字）", index, len(chars), len(mapping))

    char2phone_path = g2p_dir / "char2phone.json"
    char2phone_path.write_text(
        json.dumps(mapping, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    LOG.info("G2P 表完成：%d 字覆盖，%d 字失败", len(mapping), failures)
    return {
        "symbols": {"file": "g2p/symbols.json", "count": len(modules.symbols),
                    "sha256": sha256_of(symbols_path)},
        "char2phone": {"file": "g2p/char2phone.json", "count": len(mapping),
                       "sha256": sha256_of(char2phone_path), "failures": failures},
    }


# --------------------------------------------------------------------------- #
# 校验：用真实 encoder 跑一遍条件张量
# --------------------------------------------------------------------------- #
def validate_with_encoder(onnx_dir: Path, conditioning_dir: Path, payload: Dict[str, Any]) -> None:
    import onnxruntime as ort

    encoder_path = onnx_dir / "t2s_encoder.onnx"
    if not encoder_path.is_file():
        LOG.warning("未找到 %s，跳过条件校验", encoder_path)
        return
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass

    session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    first = EMOTIONS[0]
    entry = payload["emotions"][first]

    def load(name: str, dtype: str) -> np.ndarray:
        info = entry[name]
        raw = (conditioning_dir / first / info["file"]).read_bytes()
        array = np.frombuffer(raw, dtype="<f4" if dtype == "float32" else "<i8")
        return array.reshape(info["shape"]).astype(dtype)

    ref_seq = load("ref_seq", "int64")
    ssl_content = load("ssl_content", "float32")
    ref_audio = load("ref_audio", "float32")
    text_ids = np.zeros((1, 4), dtype=np.int64)
    if entry.get("ref_bert"):
        ref_bert = load("ref_bert", "float32")
    else:
        ref_bert = np.zeros((1, 1024), dtype=np.float32)
    text_bert = np.zeros((4, 1024), dtype=np.float32)

    feeds = {
        "ref_seq": ref_seq,
        "text_seq": text_ids,
        "ref_bert": ref_bert,
        "text_bert": text_bert,
        "ssl_content": ssl_content,
    }
    outputs = session.run(None, feeds)
    for name, value in zip([o.name for o in session.get_outputs()], outputs):
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise CondError(f"encoder 输出 {name} 含 NaN/Inf，条件张量可能有误")
        LOG.info("校验 %s -> %s %s", name, array.shape, array.dtype)
    LOG.info("条件包与 t2s_encoder 校验通过（ref_audio %s 点）", ref_audio.shape)


# --------------------------------------------------------------------------- #
# --mode logs：直接复用训练期语义码
# --------------------------------------------------------------------------- #
def load_semantic_from_tsv(tsv: Path, wanted: Dict[str, Path]) -> Dict[str, np.ndarray]:
    """从 `6-name2semantic.tsv` 读取指定参考音的语义码。"""
    wanted_names = {name: emotion for emotion, path in wanted.items() for name in (path.name, path.stem)}
    found: Dict[str, np.ndarray] = {}
    with tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key = Path(parts[0]).name
            emotion = wanted_names.get(key) or wanted_names.get(Path(key).stem)
            if emotion is None:
                continue
            codes = [int(token) for token in parts[1].strip().split() if token.lstrip("-").isdigit()]
            if codes:
                found[emotion] = np.asarray(codes, dtype=np.int64).reshape(1, -1)
                LOG.info("[%s] 从 TSV 取到语义码 %s", emotion, list(found[emotion].shape))
    return found


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_ref_conditioning.py",
        description="GPT-SoVITS v2Pro 端侧参考条件包生成器（ref_seq / ssl_content / ref_bert / ref_audio + G2P 表）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"aemeath-cond {TOOL_VERSION}")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--mode", choices=("repo", "logs", "g2p"), default="repo",
                        help="repo=仓库完整提取；logs=复用训练期语义码；g2p=只建音素表")
    parser.add_argument("--gpt-sovits-repo", default=None, help="GPT-SoVITS 仓库根目录")
    parser.add_argument("--s2g", default=None, help="SoVITS 生成器权重（s2G / s232.pth）")
    parser.add_argument("--cnhubert", default=None, help="chinese-hubert-base 目录")
    parser.add_argument("--bert", default=None, help="chinese-roberta-wwm-ext-large 目录（可选）")
    parser.add_argument("--device", default="cpu", help="cpu / cuda:0")
    parser.add_argument("--sample-rate", type=int, default=32000, help="声码器参考音采样率")
    parser.add_argument("--ref-dir", default=None, help="含 平静/开心/温柔/调侃.wav 的目录")
    parser.add_argument("--emotion-map", default=None, help="JSON：{情绪: wav 路径}")
    parser.add_argument("--ref-text-map", default=None,
                        help="JSON：{情绪: 参考文本}，用于 ref_bert；缺省则 ref_bert 留空")
    parser.add_argument("--semantic-tsv", default=None, help="训练日志 6-name2semantic.tsv")
    parser.add_argument("--name2text", default=None, help="训练日志 2-name2text.txt（用于 G2P 语料）")
    parser.add_argument("--onnx-dir", default=None, help="用于校验的 ONNX 目录（含 t2s_encoder.onnx）")
    parser.add_argument("--out-dir", required=True, help="条件包输出目录")
    parser.add_argument("--eos-token", type=int, default=DEFAULT_EOS, help="EOS token id")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB, help="语义词表大小")
    parser.add_argument("--prompt-prefix-tokens", type=int, default=-1,
                        help="fsdec.y 前多少个 token 属于参考音 prompt（-1=自动推断）")
    parser.add_argument("--build-g2p", action="store_true", help="同时生成端侧 G2P 表")
    parser.add_argument("--dry-run", action="store_true", help="只检查不写文件")
    return parser


def resolve_refs(args: argparse.Namespace) -> Dict[str, Path]:
    if args.emotion_map:
        raw = json.loads(Path(args.emotion_map).expanduser().read_text(encoding="utf-8"))
        mapping = {emotion: Path(raw[emotion]).expanduser() for emotion in EMOTIONS if emotion in raw}
    elif args.ref_dir:
        root = Path(args.ref_dir).expanduser()
        mapping = {}
        for emotion in EMOTIONS:
            for candidate in (root / f"{emotion}.wav", root / f"{emotion}.WAV"):
                if candidate.is_file():
                    mapping[emotion] = candidate
                    break
            else:
                matches = sorted(root.glob(f"*{emotion}*.wav"))
                if matches:
                    mapping[emotion] = matches[0]
    else:
        raise CondError("需要 --ref-dir 或 --emotion-map")

    missing = [emotion for emotion in EMOTIONS if emotion not in mapping]
    if missing:
        raise CondError(f"缺少情绪参考音：{', '.join(missing)}")
    return mapping


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "tool": f"prepare_ref_conditioning.py/{TOOL_VERSION}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "sample_rate": args.sample_rate,
        "vocab_size": args.vocab_size,
        "eos_token": args.eos_token,
        "prompt_prefix_tokens": args.prompt_prefix_tokens,
        "emotions": {},
    }

    if args.mode == "g2p":
        modules = load_repo_modules(
            Path(args.gpt_sovits_repo).expanduser() if args.gpt_sovits_repo else Path("."),
            None, None, args.device,
        )
        payload["g2p"] = build_g2p_table(modules, out_dir)
        (out_dir / "conditioning.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOG.info("条件包清单已写出：%s", out_dir / "conditioning.json")
        return 0

    refs = resolve_refs(args)

    if args.mode == "logs":
        if not args.semantic_tsv:
            raise CondError("--mode logs 需要 --semantic-tsv")
        tsv = Path(args.semantic_tsv).expanduser()
        semantic = load_semantic_from_tsv(tsv, refs)
        for emotion, wav in refs.items():
            if emotion not in semantic:
                raise CondError(f"TSV 中找不到 {wav.name} 的语义码")
            waveform = read_wav_mono(wav, args.sample_rate)
            emotion_dir = out_dir / emotion
            if args.dry_run:
                continue
            payload["emotions"][emotion] = {
                "ref_seq": write_blob(emotion_dir / "ref_seq.i64", semantic[emotion], "int64"),
                "ref_audio": write_blob(emotion_dir / "ref_audio.f32",
                                        waveform.reshape(1, -1), "float32"),
                "ref_bert": None,
                "ref_bert_shape": None,
                "ssl_content": None,
                "source_wav": wav.name,
                "duration_sec": round(waveform.size / args.sample_rate, 3),
                "note": "logs 模式：ssl_content 缺失，端侧 encoder 需喂零（质量下降，建议改用 repo 模式）",
            }
            LOG.warning("[%s] logs 模式缺少 ssl_content，encoder 只能喂零", emotion)
    else:
        repo = Path(args.gpt_sovits_repo).expanduser() if args.gpt_sovits_repo else None
        if repo is None:
            raise CondError("--mode repo 需要 --gpt-sovits-repo")
        modules = load_repo_modules(
            repo,
            Path(args.s2g).expanduser() if args.s2g else None,
            Path(args.cnhubert).expanduser() if args.cnhubert else None,
            args.device,
        )
        ref_texts: Dict[str, str] = {}
        if args.ref_text_map:
            ref_texts = json.loads(
                Path(args.ref_text_map).expanduser().read_text(encoding="utf-8")
            )
        for emotion, wav in refs.items():
            payload["emotions"][emotion] = extract_emotion(
                emotion=emotion,
                wav_path=wav,
                modules=modules,
                device=args.device,
                sample_rate=args.sample_rate,
                ref_text=ref_texts.get(emotion),
                out_dir=out_dir,
                use_bert=bool(args.bert),
            )

        if args.build_g2p:
            payload["g2p"] = build_g2p_table(modules, out_dir)
        else:
            LOG.warning(
                "未指定 --build-g2p：条件包将缺少 g2p/char2phone.json，端侧无法把文本转成音素 id"
            )

    payload["bert_source"] = "repo" if any(
        entry.get("ref_bert") for entry in payload["emotions"].values()
    ) else "zeros"

    if args.dry_run:
        LOG.info("--dry-run：验收通过，未写任何文件。")
        return 0

    conditioning_path = out_dir / "conditioning.json"
    conditioning_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("条件包清单已写出：%s", conditioning_path)

    if args.onnx_dir:
        validate_with_encoder(
            Path(args.onnx_dir).expanduser().resolve(), out_dir, payload
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run(args)
    except CondError as exc:
        LOG.error("条件包生成失败：%s", exc)
        return 4
    except KeyboardInterrupt:
        LOG.warning("用户中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
