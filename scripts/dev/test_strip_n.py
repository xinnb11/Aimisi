import sys, wave
from pathlib import Path
sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V
import numpy as np

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

MODEL = Path(r"D:\爱弥斯\_work\publish")
frontend = V.TextFrontend(MODEL)

text_target = "漂泊者，今天我们去星炬学院看看吧。"
t_phones, _ = frontend.phones(text_target)

# Let's test slicing generated from test_new_cond.py
# Let's re-run sdec once to get generated
ref_text = "同桌，你好。我是拉贝尔学部的爱弥斯，和你一样。"
r_phones, r_counts = frontend.phones(ref_text)
r_bert, _ = frontend.bert_features(ref_text)
t_bert, _ = frontend.bert_features(text_target)

cond_dir = MODEL / "conditioning" / "平静"
cond = V._load_json(MODEL / "conditioning" / "conditioning.json")["emotions"]["平静"]
ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(cond["ssl_content"]["shape"])

encoder = V.Session(MODEL / "t2s_encoder.onnx")
enc = encoder.run({
    V._pick(encoder, "ref_seq"): np.array([r_phones], dtype=np.int64),
    V._pick(encoder, "text_seq"): np.array([t_phones], dtype=np.int64),
    V._pick(encoder, "ref_bert"): r_bert.astype(np.float32),
    V._pick(encoder, "text_bert"): t_bert.astype(np.float32),
    V._pick(encoder, "ssl_content"): ssl.astype(np.float32),
})

prompts = enc[V._pick_out(encoder, "prompts")]
x = enc[V._pick_out(encoder, "x")]

fsdec = V.Session(MODEL / "t2s_fsdec.onnx")
first = fsdec.run({
    V._pick(fsdec, "x"): x.astype(np.float32),
    V._pick(fsdec, "prompts"): prompts.astype(np.int64),
})
y = first[V._pick_out(fsdec, "y")]

sdec = V.Session(MODEL / "t2s_sdec.onnx")
rng = np.random.default_rng(42)
k_cache = first[V._pick_out(fsdec, "k")]
v_cache = first[V._pick_out(fsdec, "v")]
y_emb = first[V._pick_out(fsdec, "y_emb")]
x_example = first[V._pick_out(fsdec, "x_example")]
prompt_count = int(prompts.shape[1])
generated = [int(token) for token in y[0, prompt_count:].tolist() if 0 <= int(token) < 1024]
recent = list(generated)
for step in range(300):
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

vocoder = V.Session(MODEL / "vits.onnx")
ref_wav = read_wav_mono(r"D:\爱弥斯\参考音频\平静\zh_vo_MAIN_YHX_2_7.wav", 32000)

for strip_n in [0, 15, 25, 30, 35]:
    sub_gen = generated[strip_n:]
    audio = vocoder.run({
        V._pick(vocoder, "text_seq"): np.array([t_phones], dtype=np.int64),
        V._pick(vocoder, "pred_semantic"): np.array([[sub_gen]], dtype=np.int64),
        V._pick(vocoder, "ref_audio"): ref_wav.reshape(1, -1).astype(np.float32),
    })
    wav = audio[V._pick_out(vocoder, "audio")].flatten()
    out_name = f"test_strip_{strip_n}.wav"
    V.write_wav(Path(out_name), wav, 32000)
    print(f"Saved {out_name} (codes: {len(sub_gen)})")
