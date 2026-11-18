"""净化参考音底噪（m07045 第 4 条）。

参考音 `conditioning/<emotion>/ref_audio.f32` 是 32 kHz 单声道 float32，
会被 vits 当作音色条件；静音段的底噪（空调声、麦克风自噪）会被一并
编进音色里，听起来就是「沙沙的底噪 + 句首一点点嘶声」。

做法（保守，只动静音段，不动语音段）：
1. 去直流（减去整段均值）。
2. 按 20 ms 帧算 RMS，取其中的「语音帧」（RMS > 峰值 RMS 的 5%）作为基准。
3. 静音帧按比例衰减到 15%，并做 5 ms 交叉渐变，避免出现台阶噪声。
4. 峰值归一到 0.95（不改变波形形状，只整体缩放）。

用法：
    python scripts/dev/denoise_ref_audio.py            # 处理 publish 下 4 个情绪
    python scripts/dev/denoise_ref_audio.py --check    # 只报告不写回
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\爱弥斯\_work\publish\conditioning")
EMOTIONS = ("平静", "开心", "温柔", "调侃")
SAMPLE_RATE = 32000
FRAME = 640  # 20 ms


def load_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise SystemExit(f"{path} 不是 4 字节对齐的 float32")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def save_f32(path: Path, data: np.ndarray) -> None:
    path.write_bytes(data.astype("<f4").tobytes())


def denoise(samples: np.ndarray) -> tuple[np.ndarray, dict]:
    x = samples.astype(np.float32).copy()
    x -= float(np.mean(x))

    frames = len(x) // FRAME
    trimmed = x[: frames * FRAME].reshape(frames, FRAME)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12)
    speech_rms = float(np.percentile(rms, 90))
    threshold = speech_rms * 0.05

    gains = np.where(rms < threshold, 0.15, 1.0).astype(np.float32)
    # 5 ms 交叉渐变，避免硬切换产生新的咔哒声
    fade = max(1, int(0.005 * SAMPLE_RATE / FRAME))
    smoothed = gains.copy()
    for i in range(len(gains)):
        lo, hi = max(0, i - fade), min(len(gains), i + fade + 1)
        smoothed[i] = float(np.mean(gains[lo:hi]))

    # 逐采样展开成增益包络并做线性插值
    env = np.repeat(smoothed, FRAME)
    if len(env) < len(x):
        env = np.concatenate([env, np.full(len(x) - len(env), smoothed[-1], dtype=np.float32)])
    env = env[: len(x)]

    quiet = int(np.sum(gains < 0.5))
    y = x * env

    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak * 0.95).astype(np.float32)

    stats = {
        "samples": int(len(y)),
        "seconds": round(len(y) / SAMPLE_RATE, 3),
        "speech_rms": round(speech_rms, 5),
        "quiet_frames": quiet,
        "quiet_ratio": round(quiet / max(1, frames), 3),
        "dc_before": round(float(np.mean(samples)), 6),
        "peak_after": round(float(np.max(np.abs(y))), 4),
        "noise_floor_before": round(float(np.percentile(rms, 10)), 6),
        "noise_floor_after_est": round(float(np.percentile(rms, 10)) * 0.15, 6),
    }
    return y, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告不写回")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    reference = None
    for emotion in EMOTIONS:
        path = root / emotion / "ref_audio.f32"
        if not path.is_file():
            print(f"[skip] {path} 不存在")
            continue
        data = load_f32(path)
        cleaned, stats = denoise(data)
        print(f"[{emotion}] {stats}")
        if reference is None:
            reference = cleaned
        elif not np.allclose(reference, cleaned, atol=1e-5):
            print(f"  ⚠ {emotion} 的参考音与「平静」不一致（长度 {len(data)} vs {len(reference)}）")
        if not args.check:
            save_f32(path, cleaned)
    print("完成（--check 模式未写回）" if args.check else "完成，已写回 4 个情绪目录")


if __name__ == "__main__":
    main()
