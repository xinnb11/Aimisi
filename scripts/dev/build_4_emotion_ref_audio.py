import json, hashlib, wave
from pathlib import Path
import numpy as np

def read_wav_mono_slice(path: str, max_dur: float = None, target_sr: int = 32000) -> np.ndarray:
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        if max_dur is not None:
            n = min(n, int(sr * max_dur))
        frames = w.readframes(n)
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        n_out = int(round(len(pcm) * target_sr / sr))
        xp = np.linspace(0, 1, len(pcm), endpoint=False)
        xq = np.linspace(0, 1, n_out, endpoint=False)
        pcm = np.interp(xq, xp, pcm).astype(np.float32)
    return pcm.astype(np.float32)

def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

PUBLISH_COND = Path(r"D:\爱弥斯\_work\publish\conditioning")

configs = {
    "平静": {
        "src_wav": r"D:\爱弥斯\参考音频\平静\zh_vo_MAIN_YHX_2_7.wav",
        "max_dur": None, # 5.18s
        "text": "同桌，你好。我是拉贝尔学部的爱弥斯，和你一样。"
    },
    "调侃": {
        "src_wav": r"D:\爱弥斯\参考音频\调侃\zh_vo_MAIN_YHX_6_29.wav",
        "max_dur": None, # 8.98s
        "text": "啊！刚才没来得及说。那时候，您勇救莫宁教授的身姿，非常酷哦！"
    },
    "温柔": {
        "src_wav": r"D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav",
        "max_dur": 8.0, # 8.0s first sentence
        "text": "在那些听到你叹息的夜晚，我悄悄走到桌子旁时，偶尔会听到你在和他低声通讯。"
    },
    "愤怒生气": {
        "src_wav": r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav",
        "max_dur": 6.5, # 6.5s first sentence
        "text": "知道那些真相的时候，我感受到的第一种情绪，是愤怒。"
    },
    "开心": {
        "src_wav": r"D:\爱弥斯\参考音频\调侃\zh_vo_MAIN_YHX_6_29.wav",
        "max_dur": None,
        "text": "啊！刚才没来得及说。那时候，您勇救莫宁教授的身姿，非常酷哦！"
    }
}

cond_json_path = PUBLISH_COND / "conditioning.json"
cond_data = json.loads(cond_json_path.read_text(encoding="utf-8"))

for emo, cfg in configs.items():
    emo_dir = PUBLISH_COND / emo
    emo_dir.mkdir(parents=True, exist_ok=True)
    
    pcm = read_wav_mono_slice(cfg["src_wav"], cfg["max_dur"], 32000)
    # Peak normalization to 0.89 to ensure loudness consistency
    peak = float(np.abs(pcm).max())
    if peak > 0:
        pcm = pcm * (0.89 / max(peak, 0.89))
    
    pcm_bytes = pcm.tobytes()
    out_f32 = emo_dir / "ref_audio.f32"
    out_f32.write_bytes(pcm_bytes)
    sha256 = sha256_of_bytes(pcm_bytes)
    
    # If the emotion doesn't have ref_seq, copy from 平静
    for aux in ["ref_seq.i64", "ref_bert.f32", "ssl_content.f32"]:
        if not (emo_dir / aux).exists():
            src_aux = PUBLISH_COND / "平静" / aux
            if src_aux.exists():
                (emo_dir / aux).write_bytes(src_aux.read_bytes())
    
    # Update conditioning.json
    if emo not in cond_data["emotions"]:
        cond_data["emotions"][emo] = json.loads(json.dumps(cond_data["emotions"]["平静"]))
    
    cond_data["emotions"][emo]["ref_audio"] = {
        "file": "ref_audio.f32",
        "dtype": "float32",
        "shape": [1, len(pcm)],
        "bytes": len(pcm_bytes),
        "sha256": sha256
    }
    cond_data["emotions"][emo]["ref_text"] = cfg["text"]
    cond_data["emotions"][emo]["source_wav"] = Path(cfg["src_wav"]).name
    cond_data["emotions"][emo]["duration_sec"] = round(len(pcm) / 32000.0, 2)
    print(f"[{emo}] ref_audio.f32: {len(pcm)} samples ({len(pcm)/32000:.2f}s), peak={np.abs(pcm).max():.3f}, sha256={sha256[:12]}...")

cond_json_path.write_text(json.dumps(cond_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Updated {cond_json_path}")
