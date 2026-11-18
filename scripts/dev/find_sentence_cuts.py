import wave
import numpy as np

for name, path in [
    ("温柔", r"D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav"),
    ("愤怒生气", r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav")
]:
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        frames = w.getnframes()
        pcm = np.frombuffer(w.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
    
    # 50ms windows
    win = sr // 20
    rms = [np.sqrt(np.mean(pcm[i:i+win]**2)) for i in range(0, len(pcm)-win, win)]
    # find silences where rms < 0.01 for > 300ms
    silences = []
    in_sil = False
    start_sil = 0
    for idx, r in enumerate(rms):
        t = idx * 0.05
        if r < 0.01:
            if not in_sil:
                in_sil = True
                start_sil = t
        else:
            if in_sil:
                in_sil = False
                if t - start_sil >= 0.25:
                    silences.append((round(start_sil, 2), round(t, 2)))
    print(f"[{name}] {frames/sr:.2f}s silences:", silences[:8])
