#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autodl_prep_and_asr.py
======================
爱弥斯 (Aemeath) 语音数据集「云端清洗 + FunASR 自动打标」一体化流水线。

设计目标（AutoDL 32GB 显存实例 / 任意 Linux 或 Windows 主机均可运行）
--------------------------------------------------------------------------
1. 扫描输入目录下所有 ``.wav``（48kHz / 16bit / 单声道 解包素材，命名如
   ``aimisi_zh_003_6264182.wav``）。
2. 逐条做客观声学质检（QC），自动剔除：
   - 时长 < 1.8s 的战斗短促喊叫（同时按 180KB 原始文件体积二次兜底）；
   - 峰值削波 / 数字满刻度爆音（clip ratio 超阈值）；
   - 信噪比过低（底噪、电流声、远距离录音）；
   - 近乎全静音或有效语音占比过低；
   - 重复 take（同一条台词的多份拷贝，避免训练集过拟合）。
3. 对合格音频做清洗：去直流、60Hz 高通、首尾静音裁剪、峰值归一化、
   重采样（默认 48k -> 32k，符合 GPT-SoVITS v2 训练采样率）、边缘 5ms 淡入淡出。
4. 调用阿里达摩院 FunASR（从魔搭 ModelScope 离线加载 Paraformer-zh +
   FSMN-VAD + CT-Punc）批量识别中文台词。
5. 文本后处理：全半角归一、ASR 标点规整、阿拉伯数字转中文、专有名词同音错字
   自动纠正（爱弥斯 / 漂泊者 / 拉海洛 / 星炬学院 等）。
6. 输出符合 GPT-SoVITS v2 与 Bert-VITS2 训练标准的标注文件::

       <cleaned_wav_abs_path>|aemeath|ZH|台词文本

   并附带机器可读的 ``prep_report.json``、``dataset_meta.jsonl`` 与
   ``rejected_files.csv``，便于回归审计与多机分片并行。

运行示例
--------
::

    # 标准用法（AutoDL GPU）
    python scripts/autodl_prep_and_asr.py \
        --input-dir /root/autodl-tmp/aemeath_raw \
        --output-dir /root/autodl-tmp/aemeath_dataset \
        --device cuda:0

    # 仅质检不识别（先摸清素材成色）
    python scripts/autodl_prep_and_asr.py -i raw -o ds --asr-backend none

    # 四机并行分片打标（每台处理 1/4 文件）
    python scripts/autodl_prep_and_asr.py -i raw -o ds_shard0 --shard-index 0 --shard-count 4

    # 断点续跑
    python scripts/autodl_prep_and_asr.py -i raw -o ds --resume

依赖
----
必需：``numpy``；读取/写出 wav 优先使用 ``soundfile``（缺失时自动回退标准库
``wave``，仅支持 PCM）。打标需 ``funasr`` + ``modelscope``（``pip install -r
scripts/requirements-autodl.txt``）。``scipy`` / ``soxr`` / ``pypinyin`` /
``tqdm`` 均为可选增强项，缺失时自动降级，不会中断流水线。

作者：DeepSeek Harness（端侧 AI 推理引擎与云端训练量化工程师）
许可：AGPL-3.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("aemeath.prep")

AUDIO_EXTS: Tuple[str, ...] = (".wav", ".wave")

EMOTIONS: Tuple[str, ...] = ("平静", "开心", "温柔", "调侃")

# --------------------------------------------------------------------------- #
# 一、专有名词纠正词典
# --------------------------------------------------------------------------- #
# 说明：每一项为 (规范写法, 既往观察到的高频误识别变体)。
# 除显式变体外，脚本还会在安装 pypinyin 时启用「同音字滑动窗口纠错」，
# 因此无需把所有可能的同音错字都穷举进来。
PROPER_NOUNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("爱弥斯", ("艾米斯", "爱米斯", "爱迷斯", "艾弥斯", "爱弥丝", "埃米斯", "艾密斯", "爱密斯", "艾米丝")),
    ("漂泊者", ("漂泊着", "飘泊者", "飘泊着", "漂迫者", "漂泊折", "飘迫者", "漂泼者")),
    ("拉海洛", ("拉海罗", "拉海落", "拉海若", "拉哈罗", "拉海络", "拉海洛特")),
    ("星炬学院", ("星距学院", "星具学院", "星聚学院", "星炬学园", "星巨学院", "星距学园")),
    ("鸣潮", ("明潮", "鸣嘲", "名潮", "鸣朝")),
    ("声骸", ("生骸", "声核", "声害", "声孩")),
    ("共鸣者", ("共明者", "共鸣着", "共名者", "共鸣折")),
    ("守岸人", ("首岸人", "守暗人", "手岸人")),
    ("今州", ("金州", "今洲", "金洲")),
]

# 送入 FunASR 作为热词提示的规范名词（仅上下文热词模型生效，其余模型自动忽略）
HOTWORD_TERMS: Tuple[str, ...] = tuple(n for n, _ in PROPER_NOUNS)


# --------------------------------------------------------------------------- #
# 二、基础异常与工具函数
# --------------------------------------------------------------------------- #
class PrepError(RuntimeError):
    """流水线内部可恢复/不可恢复错误的统一异常类型。"""


def db(x: float, floor: float = 1e-10) -> float:
    """线性幅度 -> dBFS（带安全下限，避免 log10(0)）。"""
    return 20.0 * math.log10(max(float(x), floor))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b else default


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 三、音频读写（soundfile 优先，标准库 wave 兜底）
# --------------------------------------------------------------------------- #
def _read_wav_builtin(path: Path) -> Tuple[np.ndarray, int, int]:
    """使用标准库 wave 读取 PCM wav，返回 (mono float32, sample_rate, channels)。"""
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        triples = np.frombuffer(raw, dtype=np.uint8)
        triples = triples[: (triples.size // 3) * 3].reshape(-1, 3).astype(np.int32)
        ints = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        ints = np.where(ints >= (1 << 23), ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    else:
        raise PrepError(f"不支持的 PCM 位宽 {sw * 8} bit：{path}")

    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr, ch


def load_audio(path: Path) -> Tuple[np.ndarray, int, int]:
    """稳健读取音频为单声道 float32，返回 (samples, sample_rate, channels)。"""
    try:
        import soundfile as sf  # type: ignore

        raw, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if raw.size == 0:
            raise PrepError(f"空音频：{path}")
        channels = int(raw.shape[1])
        data = raw.mean(axis=1) if channels > 1 else raw[:, 0]
        return np.ascontiguousarray(data, dtype=np.float32), int(sr), channels
    except PrepError:
        raise
    except Exception:
        # soundfile 未安装 / 文件头异常时回退
        return _read_wav_builtin(path)


def write_wav_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """写出 16bit 单声道 PCM wav（GPT-SoVITS / Bert-VITS2 训练标准格式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), pcm, int(sample_rate), subtype="PCM_16", format="WAV")
        return
    except Exception:
        pass
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """多相/插值重采样。soxr -> scipy -> 线性插值三级降级。"""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        import soxr  # type: ignore

        return np.asarray(soxr.resample(x, sr_in, sr_out, quality="HQ"), dtype=np.float32)
    except Exception:
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore

        g = gcd(int(sr_in), int(sr_out))
        return np.asarray(
            resample_poly(x, int(sr_out // g), int(sr_in // g)), dtype=np.float32
        )
    except Exception:
        pass

    n_out = int(round(x.size * float(sr_out) / float(sr_in)))
    if n_out <= 1:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(xq, xp, x).astype(np.float32)


def highpass_60hz(x: np.ndarray, sr: int, cutoff: float = 60.0) -> np.ndarray:
    """一阶高通滤除低频直流/电源嗡鸣。scipy 可用时更平滑，否则仅去均值。"""
    try:
        from scipy.signal import butter, sosfilt  # type: ignore

        sos = butter(2, cutoff / (sr / 2.0), btype="highpass", output="sos")
        return np.asarray(sosfilt(sos, x), dtype=np.float32)
    except Exception:
        return (x - float(np.mean(x))).astype(np.float32)


def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    """按固定帧长切分计算 RMS 序列（用于 VAD / SNR 估计）。"""
    hop = max(1, int(round(sr * frame_ms / 1000.0)))
    n = x.size // hop
    if n <= 0:
        return np.abs(x).astype(np.float32)
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 四、客观质检（QC）
# --------------------------------------------------------------------------- #
@dataclass
class QcResult:
    """单条音频的客观指标快照。"""

    path: str
    sample_rate: int
    channels: int
    duration_sec: float
    file_bytes: int
    peak_dbfs: float
    rms_dbfs: float
    clip_ratio: float
    silence_ratio: float
    snr_db: Optional[float]
    hiss_db: float
    dc_offset: float
    speech_ratio: float
    fingerprint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateConfig:
    """质检阈值（均可由命令行覆盖）。"""

    min_duration: float = 1.8
    min_bytes: Optional[int] = None
    max_clip_ratio: float = 0.005
    min_snr_db: float = 12.0
    max_hiss_db: float = -12.0
    max_silence_ratio: float = 0.85
    min_peak_dbfs: float = -35.0
    min_speech_ratio: float = 0.12


def estimate_snr(frame_rms_seq: np.ndarray) -> Optional[float]:
    """基于「静音帧噪声底 / 语音帧峰值」估计 SNR。

    连续无停顿的长句无法从时域动态范围估计噪声底，此时返回 ``None``
    （不参与淘汰判定，改由谱域嘶声比 hiss_db 兜底），避免误杀好素材。
    """
    if frame_rms_seq.size < 8:
        return None
    p90 = float(np.percentile(frame_rms_seq, 90))
    if p90 <= 1e-7:
        return None
    quiet = frame_rms_seq[frame_rms_seq < p90 * 0.10]
    if quiet.size / frame_rms_seq.size < 0.05:
        return None
    noise = max(float(np.median(quiet)), 1e-8)
    # 静音段为数字零时理论 SNR 无穷大，上限收敛到 60dB 以保持报表可读
    return min(db(p90) - db(noise), 60.0)


def estimate_hiss_db(x: np.ndarray, sr: int) -> float:
    """谱域嘶声比：7.5k~11kHz 高频噪声密度 相对 1k~4kHz 语音密度（dB）。

    白噪声/电流声约为 0dB，干净语音通常低于 -20dB；对采样率不足 16kHz
    的素材无高频分量，返回 -120dB 视为无嘶声。
    """
    if x.size < 512 or sr < 16000:
        return -120.0
    n = min(x.size, 1 << 18)
    segment = x[:n].astype(np.float64)
    spectrum = np.fft.rfft(segment * np.hanning(n))
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    def band_density(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(np.mean(power[mask]))

    mid = band_density(1000.0, 4000.0)
    high = band_density(7500.0, 11000.0)
    if mid <= 1e-20:
        return -120.0
    if high <= 1e-20:
        return -120.0
    return float(10.0 * math.log10(high / mid))


def waveform_fingerprint(x: np.ndarray, sr: int) -> str:
    """粗粒度包络指纹，用于识别重复 take（容忍轻微幅值差异）。"""
    if x.size == 0:
        return "empty"
    segments = np.array_split(x, 16)
    env = np.array(
        [float(np.sqrt(np.mean(np.square(s, dtype=np.float64)))) for s in segments if s.size > 0]
    )
    peak = float(env.max()) if env.size else 0.0
    coarse = np.round(env / peak * 12.0).astype(np.int32) if peak > 1e-9 else np.zeros_like(env)
    h = hashlib.md5()
    h.update(str(int(round(float(x.size) / max(sr, 1) * 10.0))).encode("ascii"))
    h.update(coarse.astype(np.int16).tobytes())
    return h.hexdigest()


def evaluate_quality(
    path: Path,
    x: np.ndarray,
    sr: int,
    channels: int,
    file_bytes: int,
    gate: GateConfig,
) -> Tuple[QcResult, List[str]]:
    """计算指标并给出淘汰原因列表（空列表 = 合格）。"""
    duration = _safe_div(x.size, sr)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0

    # 削波占比：满刻度采样点比例（-0.02dBFS 以上视为削波）
    clip_ratio = float(np.mean(np.abs(x) >= 0.998)) if x.size else 1.0

    rms_seq = frame_rms(x, sr)
    p90 = float(np.percentile(rms_seq, 90)) if rms_seq.size else 0.0
    silence_thr = max(p90 * 0.08, 1e-5)
    silence_ratio = float(np.mean(rms_seq < silence_thr)) if rms_seq.size else 1.0
    speech_ratio = 1.0 - silence_ratio
    snr = estimate_snr(rms_seq)
    hiss = estimate_hiss_db(x, sr)
    dc_offset = float(np.mean(x)) if x.size else 0.0

    # 原始文件体积兜底：48kHz/16bit/mono 下 1.8s ≈ 172.8KB，
    # 用户口径约 180KB；若采样率/声道不同则按字节率等比缩放。
    if gate.min_bytes is None:
        min_bytes = int(gate.min_duration * max(sr, 1) * 2 * max(channels, 1))
    else:
        min_bytes = int(gate.min_bytes)

    qc = QcResult(
        path=str(path),
        sample_rate=int(sr),
        channels=int(channels),
        duration_sec=round(duration, 4),
        file_bytes=int(file_bytes),
        peak_dbfs=round(db(peak), 2),
        rms_dbfs=round(db(rms), 2),
        clip_ratio=round(clip_ratio, 6),
        silence_ratio=round(silence_ratio, 4),
        snr_db=(round(snr, 2) if snr is not None else None),
        hiss_db=round(hiss, 2),
        dc_offset=round(dc_offset, 5),
        speech_ratio=round(speech_ratio, 4),
        fingerprint=waveform_fingerprint(x, sr),
    )

    reasons: List[str] = []
    if duration < gate.min_duration:
        reasons.append(f"duration_too_short({duration:.2f}s<{gate.min_duration:.2f}s)")
    if file_bytes < min_bytes:
        reasons.append(f"file_too_small({file_bytes}B<{min_bytes}B)")
    if clip_ratio > gate.max_clip_ratio:
        reasons.append(f"clipping({clip_ratio * 100:.2f}%>{gate.max_clip_ratio * 100:.2f}%)")
    if snr is not None and snr < gate.min_snr_db:
        reasons.append(f"low_snr({snr:.1f}dB<{gate.min_snr_db:.1f}dB)")
    if hiss > gate.max_hiss_db:
        reasons.append(f"hiss_noise({hiss:.1f}dB>{gate.max_hiss_db:.1f}dB)")
    if silence_ratio > gate.max_silence_ratio:
        reasons.append(f"mostly_silence({silence_ratio * 100:.0f}%)")
    if qc.peak_dbfs < gate.min_peak_dbfs:
        reasons.append(f"too_quiet(peak{qc.peak_dbfs:.1f}dBFS)")
    if speech_ratio < gate.min_speech_ratio:
        reasons.append(f"little_speech({speech_ratio * 100:.0f}%)")
    return qc, reasons


# --------------------------------------------------------------------------- #
# 五、清洗（去直流 / 高通 / 裁静音 / 归一化 / 重采样 / 淡入淡出）
# --------------------------------------------------------------------------- #
@dataclass
class CleanConfig:
    target_sr: int = 32000
    trim_silence: bool = True
    trim_pad_ms: float = 50.0
    target_peak_dbfs: float = -1.0
    fade_ms: float = 5.0
    highpass: bool = True


def trim_silence(x: np.ndarray, sr: int, pad_ms: float = 50.0) -> np.ndarray:
    """基于 10ms 帧能量裁剪首尾长静音，保留 pad_ms 呼吸感。"""
    hop = max(1, int(sr * 0.01))
    n = x.size // hop
    if n < 3:
        return x
    frames = x[: n * hop].reshape(n, hop)
    energies = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = float(energies.max())
    if peak <= 1e-7:
        return x
    voiced = np.flatnonzero(energies > peak * 0.06)
    if voiced.size == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, int(voiced[0]) * hop - pad)
    end = min(x.size, (int(voiced[-1]) + 1) * hop + pad)
    return x[start:end]


def apply_fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = int(sr * fade_ms / 1000.0)
    if n <= 1 or x.size < 2 * n:
        return x
    ramp = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def clean_waveform(x: np.ndarray, sr: int, cfg: CleanConfig) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """返回 (清洗后波形, 采样率, 处理摘要)。"""
    info: Dict[str, Any] = {"input_sr": sr, "input_duration": round(_safe_div(x.size, sr), 4)}

    y = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    y = y - float(np.mean(y))  # 去直流
    if cfg.highpass:
        y = highpass_60hz(y, sr, 60.0)
    if cfg.trim_silence:
        y = trim_silence(y, sr, cfg.trim_pad_ms)
    if sr != cfg.target_sr:
        y = resample_audio(y, sr, cfg.target_sr)
        sr = cfg.target_sr
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        target = 10.0 ** (cfg.target_peak_dbfs / 20.0)
        y = (y * (target / peak)).astype(np.float32)
    y = apply_fade(y, sr, cfg.fade_ms)
    y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

    info["output_sr"] = sr
    info["output_duration"] = round(_safe_div(y.size, sr), 4)
    info["normalized_peak_dbfs"] = cfg.target_peak_dbfs
    return y, sr, info


# --------------------------------------------------------------------------- #
# 六、文本后处理（数字 / 标点 / 专有名词同音纠错）
# --------------------------------------------------------------------------- #
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")


def int_to_chinese(value: int) -> str:
    """整数转中文读法（0 ~ 9999_9999），用于避免训练集出现裸阿拉伯数字。"""
    if value < 0:
        return "负" + int_to_chinese(-value)
    if value == 0:
        return "零"
    if value >= 10000:
        high, low = divmod(value, 10000)
        text = int_to_chinese(high) + "万"
        if low:
            text += ("零" if low < 1000 else "") + int_to_chinese(low)
        return text
    digits = str(value)
    out: List[str] = []
    zero_pending = False
    total = len(digits)
    for idx, ch in enumerate(digits):
        d = int(ch)
        pos = total - idx - 1
        if d == 0:
            zero_pending = True
            continue
        if zero_pending and out:
            out.append("零")
        zero_pending = False
        out.append(_CN_DIGITS[d] + _CN_UNITS[pos])
    text = "".join(out)
    if text.startswith("一十"):
        text = text[1:]
    return text


def digits_to_chinese(text: str) -> str:
    """百分比 / 小数 / 整数 的阿拉伯数字中文化。"""
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        lambda m: "百分之" + (int_to_chinese(int(float(m.group(1)))) if "." not in m.group(1)
                              else _decimal_to_chinese(m.group(1))),
        text,
    )

    def _decimal(match: "re.Match[str]") -> str:
        return _decimal_to_chinese(match.group(0))

    text = re.sub(r"\d+\.\d+", _decimal, text)
    text = re.sub(r"\d+", lambda m: int_to_chinese(int(m.group(0))), text)
    return text


def _decimal_to_chinese(token: str) -> str:
    whole, _, frac = token.partition(".")
    head = int_to_chinese(int(whole)) if whole else "零"
    tail = "".join(_CN_DIGITS[int(c)] for c in frac)
    return f"{head}点{tail}"


def to_halfwidth(text: str) -> str:
    """全角数字/字母转半角，保留中文标点（列表格式更易读且兼容前端正则）。"""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


class TextPostProcessor:
    """ASR 文本规整 + 专有名词同音纠错。"""

    _PUNCT_MAP = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ";": "；",
        ":": "：",
        "~": "～",
        "…": "…",
    }

    def __init__(self, use_pinyin: bool = True, number_style: str = "chinese") -> None:
        self.use_pinyin = use_pinyin
        self.number_style = number_style
        self._pinyin: Any = None
        self._pinyin_ready = False
        self._canonical_pinyin: Dict[str, List[str]] = {}

    # -- pypinyin 懒加载 ---------------------------------------------------- #
    def _ensure_pinyin(self) -> bool:
        if self._pinyin_ready:
            return self._pinyin is not None
        self._pinyin_ready = True
        if not self.use_pinyin:
            return False
        try:
            from pypinyin import lazy_pinyin  # type: ignore

            self._pinyin = lazy_pinyin
            for canonical, _variants in PROPER_NOUNS:
                self._canonical_pinyin[canonical] = list(
                    lazy_pinyin(canonical, errors="ignore")
                )
            return True
        except Exception:
            LOG.info("pypinyin 未安装，专有名词纠错退化为显式变体词典模式。")
            self._pinyin = None
            return False

    # -- 主入口 ------------------------------------------------------------- #
    def process(self, raw: str) -> str:
        if not raw:
            return ""
        text = unicodedata.normalize("NFKC", raw).strip()
        text = to_halfwidth(text)
        text = text.replace("|", "，").replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", "", text)
        text = self._normalize_punctuation(text)
        if self.number_style == "chinese":
            text = digits_to_chinese(text)
        text = self._replace_variants(text)
        if self._ensure_pinyin():
            text = self._replace_homophones(text)
        text = self._drop_asr_noise(text)
        text = self._finalize_punctuation(text)
        return text.strip()

    def _normalize_punctuation(self, text: str) -> str:
        # NFKC 会把中文省略号「……」展开成连续点号，这里先折叠回省略号
        text = re.sub(r"\.{3,}|。{3,}|·{3,}", "…", text)
        for src, dst in self._PUNCT_MAP.items():
            if src == ".":
                continue
            text = text.replace(src, dst)
        text = text.replace(".", "。")
        text = re.sub(r"…{2,}", "…", text)
        text = re.sub(r"([，。！？；：、]){2,}", r"\1", text)
        text = re.sub(r"[，、]+([。！？…])", r"\1", text)
        return text

    def _replace_variants(self, text: str) -> str:
        # 长词优先，避免「星炬」被「星炬学院」的规则重复处理
        ordered = sorted(PROPER_NOUNS, key=lambda item: len(item[0]), reverse=True)
        for canonical, variants in ordered:
            for variant in variants:
                if variant and variant in text:
                    text = text.replace(variant, canonical)
                    LOG.debug("专名纠正(变体)：%s -> %s", variant, canonical)
        return text

    def _replace_homophones(self, text: str) -> str:
        """按拼音完全相同的窗口替换，实现「同音错字」通用纠正。"""
        if self._pinyin is None:
            return text
        ordered = sorted(PROPER_NOUNS, key=lambda item: len(item[0]), reverse=True)
        for canonical, _variants in ordered:
            target_py = self._canonical_pinyin.get(canonical)
            if not target_py:
                continue
            width = len(canonical)
            if width < 2 or len(text) < width:
                continue
            chars = list(text)
            idx = 0
            while idx <= len(chars) - width:
                window = "".join(chars[idx : idx + width])
                if window == canonical:
                    idx += width
                    continue
                # 仅当窗口全是汉字时才做拼音比对，避免误伤数字/标点
                if not all("\u4e00" <= c <= "\u9fff" for c in window):
                    idx += 1
                    continue
                window_py = list(self._pinyin(window, errors="ignore"))
                if window_py == target_py:
                    chars[idx : idx + width] = list(canonical)
                    LOG.debug("专名纠正(同音)：%s -> %s", window, canonical)
                    idx += width
                else:
                    idx += 1
            text = "".join(chars)
        return text

    @staticmethod
    def _drop_asr_noise(text: str) -> str:
        # 去除常见 ASR 幻觉尾巴与孤立的重复语气词标记
        text = re.sub(r"(字幕由.{0,12}提供|请不吝点赞.{0,20}|关注.{0,10}订阅)", "", text)
        text = re.sub(r"[♪♫♬]+", "", text)
        text = re.sub(r"[（(][^）)]{0,4}[）)]", "", text)  # 去掉（笑）（呼吸）等副语言标记
        return text

    @staticmethod
    def _finalize_punctuation(text: str) -> str:
        text = text.strip("，、；：")
        if not text:
            return ""
        if text[-1] not in "。！？…～":
            text += "。"
        return text


# --------------------------------------------------------------------------- #
# 七、ASR 打标后端（FunASR / Mock / None）
# --------------------------------------------------------------------------- #
class AsrTagger:
    """ASR 后端抽象。"""

    name: str = "base"

    def transcribe(self, wav_paths: Sequence[Path]) -> Dict[str, str]:
        raise NotImplementedError


class NoneTagger(AsrTagger):
    """仅质检模式：不产生文本。"""

    name = "none"


class MockTagger(AsrTagger):
    """流水线联调用假打标器：不下载任何模型，按文件名生成占位台词。"""

    name = "mock"

    def transcribe(self, wav_paths: Sequence[Path]) -> Dict[str, str]:
        return {str(p): f"爱弥斯占位台词 {int(time.time()) % 1000}。" for p in wav_paths}


class FunAsrTagger(AsrTagger):
    """阿里达摩院 FunASR（Paraformer-zh + FSMN-VAD + CT-Punc），魔搭离线加载。"""

    name = "funasr"

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        device: str = "cuda:0",
        batch_size_s: int = 300,
        cache_dir: Optional[str] = None,
        hotword: bool = True,
    ) -> None:
        self.model_name = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.device = device
        self.batch_size_s = batch_size_s
        self.cache_dir = cache_dir
        self.use_hotword = hotword
        self._model: Any = None

    # -- 模型装配 ----------------------------------------------------------- #
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失路径
            raise PrepError(
                "未安装 FunASR，请先执行：pip install -r scripts/requirements-autodl.txt"
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "device": self.device,
            "disable_update": True,
            "disable_pbar": True,
        }
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        if self.punc_model:
            kwargs["punc_model"] = self.punc_model
        if self.cache_dir:
            os.environ.setdefault("MODELSCOPE_CACHE", self.cache_dir)
            os.environ.setdefault("HF_HOME", self.cache_dir)
            kwargs["cache_dir"] = self.cache_dir

        LOG.info("正在从魔搭社区加载 FunASR 离线模型：%s (device=%s)", self.model_name, self.device)
        self._model = AutoModel(**kwargs)
        LOG.info("FunASR 模型加载完成。")
        return self._model

    # -- 推理 --------------------------------------------------------------- #
    def _generate_batch(self, wav_paths: Sequence[Path]) -> List[str]:
        model = self._ensure_model()
        kwargs: Dict[str, Any] = {
            "input": [str(p) for p in wav_paths],
            "batch_size_s": int(self.batch_size_s),
            "use_itn": True,
            "disable_pbar": True,
        }
        if self.use_hotword:
            kwargs["hotword"] = " ".join(HOTWORD_TERMS)
        results = model.generate(**kwargs)
        if isinstance(results, dict):
            results = [results]
        texts: List[str] = []
        for item in results:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
            else:
                texts.append(str(item))
        return texts

    def _generate_single(self, wav_path: Path) -> str:
        model = self._ensure_model()
        kwargs: Dict[str, Any] = {
            "input": str(wav_path),
            "use_itn": True,
            "disable_pbar": True,
        }
        if self.use_hotword:
            kwargs["hotword"] = " ".join(HOTWORD_TERMS)
        results = model.generate(**kwargs)
        if isinstance(results, dict):
            results = [results]
        if not results:
            return ""
        item = results[0]
        return str(item.get("text", "")) if isinstance(item, dict) else str(item)

    def transcribe(self, wav_paths: Sequence[Path]) -> Dict[str, str]:
        if not wav_paths:
            return {}
        try:
            texts = self._generate_batch(wav_paths)
            if len(texts) == len(wav_paths):
                return {str(p): t for p, t in zip(wav_paths, texts)}
            LOG.warning("批量识别返回条目数不匹配（%d != %d），回退逐条识别。", len(texts), len(wav_paths))
        except Exception as exc:
            LOG.warning("批量识别异常（%s），尝试切换 CPU 逐条重试…", exc)
            if "cuda" in str(self.device).lower():
                try:
                    self.device = "cpu"
                    self._model = None
                except Exception:  # pragma: no cover
                    pass
            else:
                self._model = None

        out: Dict[str, str] = {}
        for idx, path in enumerate(wav_paths, 1):
            try:
                out[str(path)] = self._generate_single(path)
            except Exception as exc:  # 单条失败不应中断整批
                LOG.error("识别失败 [%d/%d] %s：%s", idx, len(wav_paths), path.name, exc)
                out[str(path)] = ""
        return out


def build_tagger(args: argparse.Namespace) -> AsrTagger:
    if args.asr_backend == "none":
        return NoneTagger()
    if args.asr_backend == "mock":
        return MockTagger()
    return FunAsrTagger(
        model=args.asr_model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        device=args.device,
        batch_size_s=args.batch_size_s,
        cache_dir=args.cache_dir,
        hotword=args.hotword,
    )


# --------------------------------------------------------------------------- #
# 八、文件发现与分片
# --------------------------------------------------------------------------- #
def discover_wavs(input_dir: Path, recursive: bool = True) -> List[Path]:
    if not input_dir.exists():
        raise PrepError(f"输入目录不存在：{input_dir}")
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in sorted(input_dir.glob(pattern))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    return files


def apply_shard(files: Sequence[Path], shard_index: int, shard_count: int) -> List[Path]:
    if shard_count <= 1:
        return list(files)
    if not (0 <= shard_index < shard_count):
        raise PrepError(f"非法分片参数：index={shard_index}, count={shard_count}")
    return [p for i, p in enumerate(files) if i % shard_count == shard_index]


# --------------------------------------------------------------------------- #
# 九、主流程
# --------------------------------------------------------------------------- #
@dataclass
class AcceptedItem:
    source: str
    cleaned: str
    duration: float
    sample_rate: int
    text: str = ""
    qc: Dict[str, Any] = field(default_factory=dict)
    clean_info: Dict[str, Any] = field(default_factory=dict)


def setup_logging(level: str, log_file: Optional[Path]) -> None:
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOG.handlers.clear()
    LOG.propagate = False  # 避免第三方库 basicConfig 导致日志重复输出
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOG.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autodl_prep_and_asr.py",
        description="爱弥斯语音素材云端清洗 + FunASR 自动打标（GPT-SoVITS v2 / Bert-VITS2 训练格式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 输入输出
    p.add_argument("-i", "--input-dir", required=True, help="原始 wav 素材目录（含子目录）")
    p.add_argument("-o", "--output-dir", default="./aemeath_dataset", help="输出数据集目录")
    p.add_argument("--list-name", default="aemeath.list", help="标注文件名")
    p.add_argument("--speaker", default="aemeath", help="标注文件中的说话人标识")
    p.add_argument("--language", default="ZH", help="标注文件中的语种标识")
    p.add_argument("--path-mode", choices=("abs", "rel"), default="abs",
                   help="标注文件中的音频路径形式（abs=绝对路径，rel=相对输出目录）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录内容")
    p.add_argument("--resume", action="store_true", help="断点续跑：复用已有 list 中的文本")

    # 质检阈值
    p.add_argument("--min-duration", type=float, default=1.8, help="最短时长（秒），低于则剔除")
    p.add_argument("--min-bytes", type=int, default=None,
                   help="最小文件字节数（默认按时长与采样率自动换算，48k/16bit/单声道约 180KB）")
    p.add_argument("--min-snr-db", type=float, default=12.0,
                   help="最低信噪比（dB，连续无停顿长句无法估计时自动跳过该判定）")
    p.add_argument("--max-hiss-db", type=float, default=-12.0,
                   help="允许的谱域嘶声比上限（dB，越高越脏；白噪声/电流声约 0dB）")
    p.add_argument("--max-clip-ratio", type=float, default=0.005, help="允许的削波采样点占比上限")
    p.add_argument("--max-silence-ratio", type=float, default=0.85, help="允许的静音帧占比上限")
    p.add_argument("--min-peak-dbfs", type=float, default=-35.0, help="最低峰值电平（dBFS）")
    p.add_argument("--no-dedupe", action="store_true", help="关闭重复 take 去重")

    # 清洗参数
    p.add_argument("--resample-to", type=int, default=32000,
                   help="清洗后重采样目标采样率（GPT-SoVITS v2 用 32000；0 表示不重采样）")
    p.add_argument("--target-peak-dbfs", type=float, default=-1.0, help="峰值归一化目标电平")
    p.add_argument("--no-trim-silence", action="store_true", help="不裁剪首尾静音")
    p.add_argument("--no-highpass", action="store_true", help="不做 60Hz 高通滤波")

    # ASR
    p.add_argument("--asr-backend", choices=("funasr", "mock", "none"), default="funasr",
                   help="打标后端：funasr=真实识别，mock=占位文本联调，none=仅质检")
    p.add_argument("--asr-model", default="paraformer-zh", help="FunASR 模型（魔搭模型 ID 或别名）")
    p.add_argument("--vad-model", default="fsmn-vad", help="VAD 模型；留空则关闭")
    p.add_argument("--punc-model", default="ct-punc", help="标点恢复模型；留空则关闭")
    p.add_argument("--device", default="cuda:0", help="FunASR 设备（cuda:0 / cpu）")
    p.add_argument("--asr-batch-files", type=int, default=8, help="每批送入 ASR 的文件数")
    p.add_argument("--batch-size-s", type=int, default=300, help="FunASR batch_size_s")
    p.add_argument("--cache-dir", default=None, help="魔搭模型缓存目录（MODELSCOPE_CACHE）")
    p.add_argument("--no-hotword", action="store_true", help="关闭专名热词提示")

    # 文本后处理
    p.add_argument("--number-style", choices=("chinese", "keep"), default="chinese",
                   help="阿拉伯数字处理方式（chinese=转中文读法，TTS 更稳）")
    p.add_argument("--no-pinyin-correction", action="store_true", help="关闭同音字纠错")
    p.add_argument("--min-chars", type=int, default=2, help="台词最少汉字数，过低视为无效识别")

    # 其它
    p.add_argument("--recursive", action="store_true", default=True, help="递归扫描子目录")
    p.add_argument("--max-files", type=int, default=0, help="最多处理文件数（0=不限，调试用）")
    p.add_argument("--shard-index", type=int, default=0, help="多机并行分片序号")
    p.add_argument("--shard-count", type=int, default=1, help="多机并行分片总数")
    p.add_argument("--copy-rejected", action="store_true", help="把废片复制到 rejected/ 便于人工复核")
    p.add_argument("--log-level", default="INFO", help="日志级别")
    p.add_argument("--version", action="version", version="aemeath-prep 1.0.0")
    return p


def load_resume_texts(list_path: Path) -> Dict[str, str]:
    """读取已有 list，返回 {音频绝对路径: 文本}。"""
    mapping: Dict[str, str] = {}
    if not list_path.exists():
        return mapping
    try:
        for line in list_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[0].strip():
                mapping[str(Path(parts[0]).expanduser())] = "|".join(parts[3:])
        LOG.info("断点续跑：从 %s 复用 %d 条既有标注。", list_path.name, len(mapping))
    except Exception as exc:
        LOG.warning("读取既有 list 失败（%s），忽略续跑缓存。", exc)
    return mapping


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cleaned_dir = output_dir / "cleaned"
    rejected_dir = output_dir / "rejected"
    log_dir = output_dir / "logs"

    if output_dir.exists() and args.overwrite:
        LOG.warning("--overwrite：清空既有输出目录 %s", output_dir)
        for sub in (cleaned_dir, rejected_dir):
            if sub.exists():
                for child in sub.iterdir():
                    if child.is_file():
                        child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = discover_wavs(input_dir, recursive=args.recursive)
    files = apply_shard(files, args.shard_index, args.shard_count)
    if args.max_files and len(files) > args.max_files:
        files = files[: args.max_files]
    LOG.info("发现 %d 个 wav 文件（分片 %d/%d）。", len(files), args.shard_index, args.shard_count)
    if not files:
        LOG.error("输入目录中没有 .wav 文件：%s", input_dir)
        return 2

    gate = GateConfig(
        min_duration=args.min_duration,
        min_bytes=args.min_bytes,
        max_clip_ratio=args.max_clip_ratio,
        min_snr_db=args.min_snr_db,
        max_hiss_db=args.max_hiss_db,
        max_silence_ratio=args.max_silence_ratio,
        min_peak_dbfs=args.min_peak_dbfs,
    )
    clean_cfg = CleanConfig(
        target_sr=args.resample_to if args.resample_to > 0 else 0,
        trim_silence=not args.no_trim_silence,
        target_peak_dbfs=args.target_peak_dbfs,
        highpass=not args.no_highpass,
    )

    list_path = output_dir / args.list_name
    resume_texts = load_resume_texts(list_path) if args.resume else {}

    accepted: List[AcceptedItem] = []
    rejected: List[Dict[str, Any]] = []
    seen_fingerprints: Dict[str, str] = {}
    total_duration = 0.0

    # ---------------- 阶段 1：质检 + 清洗 ---------------- #
    for idx, src in enumerate(files, 1):
        stem = src.stem
        try:
            file_bytes = src.stat().st_size
            data, sr, channels = load_audio(src)
            qc, reasons = evaluate_quality(src, data, sr, channels, file_bytes, gate)
        except Exception as exc:
            LOG.error("[%d/%d] 读取失败 %s：%s", idx, len(files), src.name, exc)
            rejected.append({"file": str(src), "reasons": [f"read_error:{exc}"], "qc": {}})
            continue

        if reasons:
            LOG.info("[%d/%d] 淘汰 %s -> %s", idx, len(files), src.name, ", ".join(reasons))
            rejected.append({"file": str(src), "reasons": reasons, "qc": qc.as_dict()})
            if args.copy_rejected:
                rejected_dir.mkdir(parents=True, exist_ok=True)
                dst = rejected_dir / src.name
                if not dst.exists():
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        pass
            continue

        if not args.no_dedupe:
            prev = seen_fingerprints.get(qc.fingerprint)
            if prev is not None:
                LOG.info("[%d/%d] 淘汰 %s -> duplicate_of(%s)", idx, len(files), src.name, prev)
                rejected.append(
                    {"file": str(src), "reasons": [f"duplicate_of:{prev}"], "qc": qc.as_dict()}
                )
                continue
            seen_fingerprints[qc.fingerprint] = str(src)

        try:
            clean_cfg.target_sr = args.resample_to if args.resample_to > 0 else sr
            washed, out_sr, clean_info = clean_waveform(data, sr, clean_cfg)
            dst = cleaned_dir / f"{stem}.wav"
            write_wav_pcm16(dst, washed, out_sr)
        except Exception as exc:
            LOG.error("[%d/%d] 清洗失败 %s：%s", idx, len(files), src.name, exc)
            rejected.append({"file": str(src), "reasons": [f"clean_error:{exc}"], "qc": qc.as_dict()})
            continue

        total_duration += float(clean_info["output_duration"])
        accepted.append(
            AcceptedItem(
                source=str(src),
                cleaned=str(dst),
                duration=float(clean_info["output_duration"]),
                sample_rate=out_sr,
                qc=qc.as_dict(),
                clean_info=clean_info,
            )
        )
        LOG.info(
            "[%d/%d] 合格 %s | %.2fs | SNR %s | 峰值 %.1fdBFS",
            idx, len(files), src.name, clean_info["output_duration"],
            ("%.1fdB" % qc.snr_db) if qc.snr_db is not None else "n/a",
            qc.peak_dbfs,
        )

    LOG.info(
        "质检完成：合格 %d 条（总时长 %.1fs），淘汰 %d 条。",
        len(accepted), total_duration, len(rejected),
    )
    if not accepted:
        LOG.error("没有任何音频通过质检，请放宽阈值或检查素材采样率/声道。")
        return 2

    # ---------------- 阶段 2：ASR 打标 ---------------- #
    post = TextPostProcessor(
        use_pinyin=not args.no_pinyin_correction, number_style=args.number_style
    )
    tagger = build_tagger(args)

    if tagger.name != "none":
        batch_size = max(1, args.asr_batch_files)
        for start in range(0, len(accepted), batch_size):
            chunk = accepted[start : start + batch_size]
            chunk_paths = [Path(item.cleaned).resolve() for item in chunk]
            LOG.info(
                "ASR 进度 %d-%d/%d …", start + 1, start + len(chunk), len(accepted)
            )
            try:
                texts = tagger.transcribe(chunk_paths)
            except Exception as exc:
                LOG.error("ASR 批次失败（%s），本批标记为空文本。", exc)
                texts = {}
            for item in chunk:
                key = str(Path(item.cleaned).resolve())
                raw = texts.get(key, "") or resume_texts.get(key, "")
                text = post.process(raw)
                if len(text) < args.min_chars:
                    LOG.info("丢弃（文本过短）：%s -> %r", Path(item.cleaned).name, text)
                    item.text = ""
                else:
                    item.text = text

    # ---------------- 阶段 3：落盘 ---------------- #
    path_root = output_dir if args.path_mode == "rel" else None
    lines: List[str] = []
    meta_lines: List[str] = []
    for item in accepted:
        if not item.text:
            continue
        audio_path = Path(item.cleaned).resolve()
        shown = os.path.relpath(str(audio_path), str(path_root)) if path_root else str(audio_path)
        shown = shown.replace("\\", "/")
        lines.append(f"{shown}|{args.speaker}|{args.language}|{item.text}")
        meta_lines.append(
            json.dumps(
                {
                    "audio": shown,
                    "text": item.text,
                    "duration": round(item.duration, 3),
                    "sample_rate": item.sample_rate,
                    "snr_db": item.qc.get("snr_db"),
                    "peak_dbfs": item.qc.get("peak_dbfs"),
                    "source": item.source,
                },
                ensure_ascii=False,
            )
        )

    if tagger.name == "none":
        LOG.warning("--asr-backend none：仅生成质检报告，不写 aemeath.list。")
    else:
        list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (output_dir / "dataset_meta.jsonl").write_text(
            "\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8"
        )
        LOG.info("标注文件已写出：%s（%d 行）", list_path, len(lines))

    rejected_csv = output_dir / "rejected_files.csv"
    with rejected_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "reasons", "duration_sec", "snr_db", "hiss_db", "peak_dbfs", "clip_ratio", "silence_ratio"])
        for row in rejected:
            qc = row.get("qc") or {}
            writer.writerow(
                [
                    row["file"],
                    ";".join(row["reasons"]),
                    qc.get("duration_sec", ""),
                    qc.get("snr_db", "") if qc.get("snr_db") is not None else "n/a",
                    qc.get("hiss_db", ""),
                    qc.get("peak_dbfs", ""),
                    qc.get("clip_ratio", ""),
                    qc.get("silence_ratio", ""),
                ]
            )

    report = {
        "tool": "autodl_prep_and_asr.py",
        "version": "1.0.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "list_file": str(list_path) if tagger.name != "none" else None,
        "speaker": args.speaker,
        "language": args.language,
        "asr": {
            "backend": tagger.name,
            "model": args.asr_model if tagger.name == "funasr" else None,
            "device": args.device if tagger.name == "funasr" else None,
        },
        "gate": asdict(gate),
        "clean": asdict(clean_cfg),
        "stats": {
            "scanned": len(files),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "listed": len(lines),
            "accepted_duration_sec": round(total_duration, 2),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "accepted": [
            {
                "source": item.source,
                "cleaned": item.cleaned,
                "text": item.text,
                "duration": round(item.duration, 3),
                "qc": item.qc,
            }
            for item in accepted
            if item.text
        ],
        "rejected": rejected,
    }
    report_path = output_dir / "prep_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("=" * 72)
    LOG.info("扫描 %d | 合格 %d | 标注 %d | 淘汰 %d | 用时 %.1fs",
             len(files), len(accepted), len(lines), len(rejected), time.time() - t0)
    LOG.info("标注文件：%s", list_path if tagger.name != "none" else "（未生成）")
    LOG.info("质检报告：%s", report_path)
    LOG.info("淘汰明细：%s", rejected_csv)
    LOG.info("=" * 72)
    return 0 if (lines or tagger.name == "none") else 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    log_file = output_dir / "logs" / f"prep_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)

    try:
        return run(args)
    except PrepError as exc:
        LOG.error("流水线中断：%s", exc)
        return 4
    except KeyboardInterrupt:
        LOG.warning("用户中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
