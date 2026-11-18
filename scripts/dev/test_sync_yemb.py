import sys
from pathlib import Path
sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V
import numpy as np

MODEL = Path(r"D:\爱弥斯\_work\publish")
frontend = V.TextFrontend(MODEL)
phone_ids, _ = frontend.phones("漂泊者，今天我们去星炬学院看看吧。")
text_bert, _ = frontend.bert_features("漂泊者，今天我们去星炬学院看看吧。")

cond_dir = MODEL / "conditioning" / "平静"
cond = V._load_json(MODEL / "conditioning" / "conditioning.json")["emotions"]["平静"]
ref_seq = np.fromfile(cond_dir / cond["ref_seq"]["file"], dtype="<i8").reshape(cond["ref_seq"]["shape"])
ref_bert = np.fromfile(cond_dir / cond["ref_bert"]["file"], dtype="<f4").reshape(cond["ref_bert"]["shape"])
ssl = np.fromfile(cond_dir / cond["ssl_content"]["file"], dtype="<f4").reshape(cond["ssl_content"]["shape"])

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

table = np.fromfile(MODEL / "conditioning" / "audio_embedding.f32", dtype=np.float32).reshape(-1, 512)

# Compare sync_y_emb=False vs sync_y_emb=True
for do_sync in [False, True]:
    sdec = V.Session(MODEL / "t2s_sdec.onnx")
    rng = np.random.default_rng(42)
    cur_y = y.copy()
    cur_k = k_cache.copy()
    cur_v = v_cache.copy()
    cur_y_emb = y_emb.copy()
    generated = [int(token) for token in cur_y[0, prompt_count:].tolist() if 0 <= int(token) < 1024]
    recent = list(generated)
    for step in range(250):
        out = sdec.run({
            V._pick(sdec, "iy"): cur_y.astype(np.int64),
            V._pick(sdec, "ik"): cur_k,
            V._pick(sdec, "iv"): cur_v,
            V._pick(sdec, "iy_emb"): cur_y_emb,
            V._pick(sdec, "ix_example"): x_example,
        })
        logits = out[V._pick_out(sdec, "logits")][0]
        cur_k = out[V._pick_out(sdec, "k")]
        cur_v = out[V._pick_out(sdec, "v")]
        fresh_y_emb = out[V._pick_out(sdec, "y_emb")]
        sampled = V.sample_token(logits, recent, 0.85, 15, 0.9, 1.35, rng, False)
        if sampled == 1024:
            break
        generated.append(sampled)
        recent.append(sampled)
        cur_y = out[V._pick_out(sdec, "y")]
        cur_y[0, -1] = sampled
        if do_sync:
            patched = fresh_y_emb.copy()
            patched[0, -1, :] = table[sampled]
            cur_y_emb = patched
        else:
            cur_y_emb = fresh_y_emb
    print(f"do_sync={do_sync}: steps={len(generated)}, first8={generated[:8]}")
