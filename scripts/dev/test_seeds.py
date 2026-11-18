import sys
from pathlib import Path
sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V
import numpy as np

MODEL = Path(r"D:\爱弥斯\_work\publish")
text = "漂泊者，今天我们去星炬学院看看吧。"
for s in [7, 11, 23, 37, 42]:
    wav, sr, info = V.synthesize(MODEL, text, "平静", seed=s, greedy=False, max_steps=300)
    out_path = Path(f"test_seed_{s}.wav")
    V.write_wav(out_path, wav, sr)
    print(f"Seed {s}: steps={info['steps']}, codes={info['codes']}, first8={info['first_codes']}")
