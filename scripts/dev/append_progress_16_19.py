import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
### 16.19 参考音复读抑制 + 速度/音质体检（2026-11-17 20:06）

用户 m06781 连问四件事：是否分流合成、为什么慢、为什么开头多一句「也有一些危险刺激」、为什么音质与语气不对。以下是取证结论与修复。

#### 16.19.1 不是分流合成，慢在自回归步数
- `LocalOnnxTtsBridge` 是**严格串行单队列**：`processQueue()` 一次只处理一条，`splitForSpeech` 只在**句子之间**按标点切（≥18 字才切），单句内**只跑一次** T2S 自回归 + 一次 vits。
- 播放侧节奏由 `vits` 决定：每个语义码固定 **1280 采样点**（32 kHz 下 40 ms/码 = **25 码/秒**）。
- 计算侧实测每码约 **0.1 s**（真机 4 线程、纯 CPU、int8 T2S），即比实时慢约 **2.5×**；23 字句产生 140 码 → 音频 5.6 s、计算约 14 s；28 字整句端到端 17.1 s。
- 真机 logcat 四条实测（改前）：23 字→201 码/8.04 s 音频、25 字→170/6.8 s、31 字→237/9.48 s、33 字→182/7.28 s。

#### 16.19.2 开头那句「也有一些危险刺激」是参考音复读
- 条件包参考音台词是「那些故事大多都有趣曲折，也有一些危险刺激。」，其语义码就是 `t2s_encoder` 输出的 `prompts`（118 个）。
- **v2Pro 的接口是「参考音 → prompts → 自回归」**，没有情绪向量输入；句子偏短时模型会先照抄参考音尾部若干码再进入正文，于是听起来「每次开口都先念一句参考音」。
- 修复（`GptSoVitsOnnxEngine.kt` 两道防线，随 APK 21,856,638 B 装机）：
  1. 采样前——前 24 步对 `promptTokenIds`（prompts 中出现过的非 EOS token）做 `logits[id] -= 1.5f` 软惩罚；
  2. 生成后——用 5-gram 哈希（`gramHash`，`h = h * 1000003L + (data[i] + 1L)`）把与 `prompts` 重叠的**前缀**剥掉（上限 40 码，且至少保留 `ECHO_GRAM + 8` 码）。
- 真机验收（同一句「漂泊者，语音链路已经接通。现在可以正常对话啦！」）：码数 **201 → 140**；ASR「漂泊者语音链路已经接通，现在可以正常对换了。」**不再含参考音内容**；`5.60 s / peak 0.626 / spectral_centroid 5483 Hz / F0 243.6 Hz`（`golden_ref.wav` 6427 Hz / 249.7 Hz）。证据 `scripts/dev/fixtures/_echo_a.wav`。

#### 16.19.3 音质与语气：四个情绪用的是同一段参考音
- `conditioning/平静|开心|温柔|调侃` 四个目录的 `ssl_content.f32` **SHA256 完全相同**——当初是从同一段 4.74 s 参考音复制的。
- 又因为禁止对 vits 输出做变速（会改基频与共振峰），所以**情绪开关目前只改文案、不改声音**；「语气不对」不是错觉，是条件包本身没有四套参考音。
- 其余影响音质的客观因素：T2S 走 int8 动态量化、BERT 走 int8（301 MB）、CPU 4 线程、采样 `top_k=15 / T=0.85 / 重复惩罚 1.35`——这些是为了在手机上跑得动所做的取舍，不是「推理偷懒」。

#### 16.19.4 需要用户提供的东西
要把「语气」真正修好，需要 **4 段情绪参考音 + 对应台词**（平静 / 开心 / 温柔 / 调侃）：
- 每段 **3–10 s**、单人、干净无 BGM、16 kHz 以上、wav/mp3 均可；
- 每段给出**准确台词文本**（用于 `ref_seq` 音素与 `ref_bert` 对齐）；
- 拿到后我用 `scripts/dev/recompute_ref_bert.py` + `scripts/prepare_ref_conditioning.py` 重建四个目录的 `ref_seq.i64 / ref_bert.f32 / ssl_content.f32 / ref_audio.f32`，重算 `manifest.json`，再走一遍真机 ASR 验收。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "### 16.19 参考音复读抑制" in text:
    print("SKIP: already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"
text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
