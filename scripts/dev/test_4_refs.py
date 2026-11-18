import sys
from pathlib import Path
sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V
import numpy as np
import onnxruntime as ort

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
vocoder = ort.InferenceSession(str(PUBLISH / "vits.onnx"))

frontend = V.TextFrontend(PUBLISH)
phone_ids, _ = frontend.phones("漂泊者，今天我们去星炬学院看看吧。")

# Synthesize semantic codes once with seed 42
wav, sr, info = V.synthesize(PUBLISH, "漂泊者，今天我们去星炬学院看看吧。", "平静", seed=42, greedy=False, max_steps=300)
codes = info["first_codes"] # we want real generated codes, let's extract them properly

ref_paths = {
    "平静": r"D:\爱弥斯\参考音频\平静\zh_vo_MAIN_YHX_2_7.wav",
    "调侃": r"D:\爱弥斯\参考音频\调侃\zh_vo_MAIN_YHX_6_29.wav",
    "温柔": r"D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav",
    "愤怒生气": r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav"
}

# Let's inspect read_wav_mono on all 4
for emo, path in ref_paths.items():
    mono = V.read_wav_mono(Path(path), 32000)
    print(f"{emo}: {path} -> {len(mono)} samples ({len(mono)/32000:.2f}s)")
