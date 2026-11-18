"""用 sherpa-onnx SenseVoice 对合成 WAV 做中文 ASR，并计算 F0/谱质心。

用法：
    python scripts/dev/asr_check.py 合成.wav [--ref golden_ref.wav] [--model-dir D:\tools\asr\sv]
"""

import argparse
import math
import sys
import wave

import numpy as np
import sherpa_onnx


def read_wav(path):
    with wave.open(path, "rb") as fp:
        rate = fp.getframerate()
        channels = fp.getnchannels()
        width = fp.getsampwidth()
        data = fp.readframes(fp.getnframes())
    if width != 2:
        raise SystemExit(f"仅支持 16bit PCM：{path} width={width}")
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio, rate, target=16000):
    if rate == target:
        return audio
    n = int(round(len(audio) * target / rate))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def recognizer(model_dir):
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{model_dir}/model.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        num_threads=4,
        use_itn=True,
        language="zh",
        debug=False,
    )


def spectral_centroid(audio, rate):
    frame, hop = 1024, 256
    if len(audio) < frame:
        return 0.0
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    total, weight = 0.0, 0.0
    for start in range(0, len(audio) - frame, hop):
        seg = audio[start:start + frame] * window
        mag = np.abs(np.fft.rfft(seg))
        total += float((mag * freqs).sum())
        weight += float(mag.sum())
    return total / weight if weight > 0 else 0.0


def f0_mean(audio, rate, fmin=70, fmax=500):
    """自相关法估 F0（逐帧取 40–160ms 窗口），返回有声帧均值。"""
    frame = int(rate * 0.04)
    hop = int(rate * 0.02)
    lo, hi = int(rate / fmax), int(rate / fmin)
    values = []
    for start in range(0, max(0, len(audio) - frame), hop):
        seg = audio[start:start + frame]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.01:
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window = corr[lo:hi]
        if len(window) == 0:
            continue
        lag = lo + int(np.argmax(window))
        if corr[lag] < 0.3:
            continue
        values.append(rate / lag)
    return float(np.mean(values)) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--model-dir", default=r"D:\tools\asr\sv")
    args = ap.parse_args()

    rec = recognizer(args.model_dir)
    audio, rate = read_wav(args.wav)
    stream = rec.create_stream()
    stream.accept_waveform(16000, resample(audio, rate))
    rec.decode_stream(stream)
    text = stream.result.text.strip()
    print("ASR:", text)
    print(f"audio: {len(audio)/rate:.2f}s @{rate}Hz peak={float(np.max(np.abs(audio))):.3f}")
    print(f"spectral_centroid={spectral_centroid(audio, rate):.0f}Hz f0_mean={f0_mean(audio, rate):.1f}Hz")

    if args.ref:
        ref, rrate = read_wav(args.ref)
        print(f"ref {args.ref}: {len(ref)/rrate:.2f}s @{rrate}Hz")
        print(f"ref spectral_centroid={spectral_centroid(ref, rrate):.0f}Hz f0_mean={f0_mean(ref, rrate):.1f}Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
