import wave

def slice_wav(src: str, dst: str, duration: float):
    with wave.open(src, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = int(sr * duration)
        data = w.readframes(n)
    with wave.open(dst, 'wb') as w:
        w.setframerate(sr)
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.writeframes(data)

slice_wav(r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav", "slice_angry_65.wav", 6.5)
