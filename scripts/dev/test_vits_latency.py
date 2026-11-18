import sys, wave, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def read_wav_mono(path: str, target_sr: int = 32000) -> np.ndarray:
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        frames = w.readframes(n)
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        n_out = int(round(len(pcm) * target_sr / sr))
        xp = np.linspace(0, 1, len(pcm), endpoint=False)
        xq = np.linspace(0, 1, n_out, endpoint=False)
        pcm = np.interp(xq, xp, pcm).astype(np.float32)
    return pcm

sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V

MODEL = Path(r"D:\爱弥斯\_work\publish")
frontend = V.TextFrontend(MODEL)
phone_ids, _ = frontend.phones("漂泊者，今天我们去星炬学院看看吧。")

# Capture the generated codes
# Run sdec with verify_onnx_tts
wav, sr, info = V.synthesize(MODEL, "漂泊者，今天我们去星炬学院看看吧。", "平静", seed=42, greedy=False, max_steps=300)
# We can run vocoder with different ref_audio!
# Let's see what happens if we patch verify_onnx_tts or run vocoder directly
vocoder = ort.InferenceSession(str(MODEL / "vits.onnx"))

ref_paths = {
    "平静": r"D:\爱弥斯\参考音频\平静\zh_vo_MAIN_YHX_2_7.wav",
    "调侃": r"D:\爱弥斯\参考音频\调侃\zh_vo_MAIN_YHX_6_29.wav",
    "温柔_full": r"D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav",
    "温柔_6s": (r"D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav", 6.0),
    "愤怒生气_full": r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav",
    "愤怒生气_6s": (r"D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav", 6.0),
}

# In verify_onnx_tts, info had 'codes' length.
# Let's extract the actual codes:
cond_dir = MODEL / "conditioning" / "平静"
cond = V._load_json(MODEL / "conditioning" / "conditioning.json")["emotions"]["平静"]
ref_seq = np.fromfile(cond_dir / cond["ref_seq"]["file"], dtype="<i8").reshape(cond["ref_seq"]["shape"])
ref_bert = np.fromfile(cond_dir / cond["ref_bert"]["file"], dtype="<f4").reshape(cond["ref_bert"]["shape"])
ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(cond["ssl_content"]["shape"])
text_bert, _ = frontend.bert_features("漂泊者，今天我们去星炬学院看看吧。")

encoder = V.Session(MODEL / "t2s_encoder.onnx")
enc = encoder.run({
    V._pick(encoder, "ref_seq"): ref_seq.astype(np.int64),
    V._pick(encoder, "text_seq"): np.array([phone_ids], dtype=np.int64),
    V._pick(encoder, "ref_bert"): ref_bert.astype(np.float32),
    V._pick(encoder, "text_bert"): text_bert.astype(np.float32),
    V._pick(encoder, "ssl_content"): ssl.astype(np.float32),
})
prompts = enc[V._pick_out(encoder, "prompts")]

fsdec = V.Session(MODEL / "t2s_fsdec.onnx")
first = fsdec.run({
    V._pick(fsdec, "x"): enc[V._pick_out(encoder, "x")].astype(np.float32),
    V._pick(fsdec, "prompts"): prompts.astype(np.int64),
})
y = first[V._pick_out(fsdec, "y")].astype(np.int64)
k_cache = first[V._pick_out(fsdec, "k")]
v_cache = first[V._pick_out(fsdec, "v")]
y_emb = first[V._pick_out(fsdec, "y_emb")]
x_example = first[V._pick_out(fsdec, "x_example")]
prompt_count = int(prompts.shape[1])

sdec = V.Session(MODEL / "t2s_sdec.onnx")
rng = np.random.default_rng(42)
generated = [int(token) for token in y[0, prompt_count:].tolist() if 0 <= int(token) < 1024]
recent = list(generated)
for _ in range(300):
    out = sdec.run({
        V._pick(sdec, "iy"): y.astype(np.int64),
        V._pick(sdec, "ik"): k_cache,
        V._pick(sdec, "iv"): v_cache,
        V._pick(sdec, "iy_emb"): y_emb,
        V._pick(sdec, "ix_example"): x_example,
    })
    logits = out[V._pick_out(sdec, "logits")][0]
    k_cache = out[V._pick_out(sdec, "k")]
    v_cache = out[V._pick_out(sdec, "v")]
    y_emb = out[V._pick_out(sdec, "y_emb")]
    sampled = V.sample_token(logits, recent, 0.85, 15, 0.9, 1.35, rng, False)
    if sampled == 1024:
        break
    generated.append(sampled)
    recent.append(sampled)
    y = out[V._pick_out(sdec, "y")]
    y[0, -1] = sampled

print(f"Generated semantic codes: {len(generated)}")

for name, item in ref_paths.items():
    if isinstance(item, tuple):
        path, dur = item
        pcm = read_wav_mono(path, 32000)[:int(dur * 32000)]
    else:
        pcm = read_wav_mono(item, 32000)
    
    t0 = time.time()
    audio = vocoder.run(
        None,
        {
            "text_seq": np.array([phone_ids], dtype=np.int64),
            "pred_semantic": np.array([[generated]], dtype=np.int64),
            "ref_audio": pcm.reshape(1, -1).astype(np.float32),
        }
    )
    cost = time.time() - t0
    out_wav = audio[0].flatten()
    out_file = Path(f"out_vits_{name}.wav")
    V.write_wav(out_file, out_wav, 32000)
    print(f"[{name}] ref={len(pcm)/32000:.2f}s -> vits took {cost*1000:.1f}ms, audio={len(out_wav)/32000:.2f}s, peak={np.abs(out_wav).max():.3f}")
