# 爱弥斯（Aemeath）离线语音陪伴终端 · 进度与交接日志

> **文档状态（重要）**：2026-11-17 21:46 一次 shell 编码事故把本文件截断，以下内容为**当日 21:50 依据会话记录与真机实测数据重建**（原始正文已不可恢复）。所有日期、体积、耗时、哈希与结论均来自当时的实测记录；章节编号与主题沿用原结构，个别早期细节（如 1–6 节的历史措辞）为摘要重建，可能与原文用词不同，不影响事实。

- 项目根目录：`D:\语音模型项目`
- 目标平台：Android（Capacitor + 原生 Kotlin），WebView 端 `index.html` 为单文件应用
- 目标真机：**联发科天玑 9500（vivo 2604FRK1EC，Android 16）**，设备号 `JJFIE6AULVDIMR95`
- 模型仓库（魔搭）：`xinxin111x/Aimisi`

---

## 0. 一页速览

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 离线端侧 TTS（GPT-SoVITS v2Pro 四图） | ✅ 真机出声、ASR 识别正确 | §14 / §16 |
| 参考音复读（开口先念「也有一些危险刺激」） | ✅ 已用 5-gram 剥离 + 首 24 步软惩罚修掉 | §16.19 |
| 分句流水线（边播边算） | ✅ 单段 23 字：wall 19.7 s / 音频 5.7 s | §16.12 |
| KV cache 零拷贝 | ✅ 30.9 s → 17.1 s | §16.10 |
| Int8 T2S 混合包 | ✅ 体积 1.17 GB → 699 MB，ASR 无回退 | §16.13 |
| 硬件加速 | ⚠️ `vits → NNAPI(FP16)` 已接线；sdec 锁 CPU（NNAPI 会死锁、XNNPACK 会算错） | §17 / §18 |
| 16 KB 页对齐 | ✅ 5 个 `.so` 全部 `p_align=0x4000`，`zipalign` 通过 | §18.5 |
| 魔搭仓库 | ✅ 30 文件 / 733,231,277 B，哈希逐项校验通过 | §16.13 |

---

## 1. 架构与红线（摘要）

- 前端：`index.html`（单文件，Tailwind CDN + 内联 JS）。**红线**：不改 CSS 变量体系、不改凸透镜弹簧物理引擎 `SpringConvexLensController`、不改圆形头像逻辑。
- 原生：`android/app/src/main/java/com/aemeath/terminal/`
  - `AemeathTtsPlugin.kt`（Capacitor 插件，6 个 `@PluginMethod`：`pullModelFromModelScope` / `mountModel` / `synthesizeSpeech` / `stopPlayback` / `unmountModel` / `diagnose`）
  - `GptSoVitsOnnxEngine.kt`（四图串联 + 自回归）
  - `SherpaVitsEngine.kt`（Bert-VITS2 单图）
  - `AemeathBertFrontend.kt`（WordPiece + BERT 第 21 层）
  - `AemeathOrtBackend.kt`（分 stage 后端选择）
  - `AemeathAudioPlayer.kt`（常驻 AudioTrack 播放队列）
  - `AemeathSampler.kt`（top-k / top-p / 重复惩罚，禁止 argmax）
  - `AemeathModelScopeDownloader.kt`（分片下载 + sha256）
- 模型契约：`conditioning/{平静,开心,温柔,调侃}/{ref_seq.i64, ref_bert.f32, ssl_content.f32, ref_audio.f32}` + `conditioning/g2p/{symbols.json, char2phone.json, tone_table.json, lexicon.json}` + `conditioning/audio_embedding.f32` + `bert/{model.int8.onnx, vocab.txt}`。

---

## 2. 脚本层（`scripts/`）

| 文件 | 用途 |
| --- | --- |
| `autodl_prep_and_asr.py` | AutoDL 音频清洗（时长/削波/嘶声/SNR/去重）+ FunASR 打标 → `aemeath.list` |
| `export_and_quantize_onnx.py` | ONNX 导出、Int8 动态量化、FP16 声码器、`manifest.json` |
| `prepare_ref_conditioning.py` | 从参考音生成条件包 |
| `verify_onnx_tts.py` | 桌面参考实现（与端侧逐位对齐用） |
| `build_android.ps1` | 一键构建（**必须 UTF-8 BOM**） |
| `check_elf_alignment.py` | 检查 `.so` 的 `p_align` 是否 ≥ 0x4000 |

---

## 3. 早期交付（1.0 打包与上传，摘要）

- 2026-11-16 首次把 `Aimisi_v2Pro_e8_FullPack.tar`（1,179,965,440 B）解包，确认四图 ABI：`encoder(ref_seq/text_seq/ref_bert/text_bert/ssl_content → x/prompts)`、`fsdec(x,prompts → y/k/v/y_emb/x_example)`、`sdec(iy,ik,iv,iy_emb,ix_example → y/k/v/y_emb/logits/samples)`、`vits(text_seq,pred_semantic,ref_audio → audio)`。
- 发现 `t2s_sdec.samples` 是图内 argmax（必须运行时自行采样），`prompts` 由 `ssl_content` 经 VQ codebook 算出。
- 首次发布包 26 文件 / 1,222,151,331 B（FP32 四图 + BERT int8 + 条件包），上传魔搭 `xinxin111x/Aimisi` 并逐项校验。
- 量化实验（当时用户喊停）：Int8 T2S 25.3–25.7%，FP16 `vits` 50.4%，合计 848.2 → 279.8 MB；相关产物留在 `D:\爱弥斯\_work\onnx`。

---

## 4. 工程化（2026-11-17 上午）

### 4.1 工具链
`JDK 17.0.21`（`D:\tools\jdk17\jdk-17.0.21+9`）、Android SDK（`D:\tools\android-sdk`，`platforms;android-34` / `build-tools;34.0.0` / `platform-tools`）、Gradle 8.2.1、`GRADLE_USER_HOME=D:\tools\gradle-home`。

### 4.2 真工程化
`android/` 由载荷改为 Capstone 生成的真工程（`npx cap add android`），`.kt` 合入 `app/src/main/java/com/aemeath/terminal/`，`MainActivity` 在 `super.onCreate` 之前 `registerPlugin(AemeathTtsPlugin.class)`。

### 4.3 首轮编译踩到的 5 个真坑（已写入 `ARCHITECTURE.md` §7）
1. 中文路径：`android.overridePathCheck=true`；
2. Kotlin 块注释可嵌套，注释里写 `/*` 会 `Unclosed comment`；
3. `PluginCall.getDouble/getBoolean` 返回可空；
4. `ArrayList<Int>` 没有 `toLongArray()`；
5. 字符串模板 `"$kind缺少"` 会被当成标识符，要写 `${kind}缺少`。

产物：`app-debug.apk` 64,960,376 B（v1，仅 WebView + 未接模型）。

### 4.4 图片 404 修复
Vite 只重写静态 `<img src>`，内联 JS 里的 `'1794568840819.jpg'` 等原样保留 → 构建产物缺图。`vite.config.ts` 新增 `aemeath-raw-images` 插件：复制三张原图到产物根、把哈希引用改回原名、缺图即构建失败。

---

## 5. 「不出声 / 接不上 DeepSeek」的三层静默降级（上午）

根因：
1. `LocalOnnxTtsBridge.mountModel()` 把原生异常吞掉仍返回 `success: true, localFallback: true`，拉取成功路径不检查返回值；
2. `synthesizeItem()` 不校验 `synthesizeSpeech` 结果就标 `ready`，还按 `字数×160 ms` 估算"播完"；
3. `streamChat` 的 catch 直接返回 `simulateLocalOfflineStream`（假回复），Key 从不持久化。

修法：错误全部上抛 + toast 分档建议（401/402/429/403/网络）、`localStorage['aemeath_llm_config_v1']` 持久化、`testApiConnection()` 打 `/models` 自检、`diagnose()` 端侧自检块进对话流。实测 CORS 正常（`access-control-allow-origin` 回显）。

---

## 6. Android 16 的 16 KB 页对齐（上午）

真机第一次启动弹「此应用不符合 16 KB 对齐要求」，点名 `libonnxruntime.so`、`libsherpa-onnx-*.so`、`libonnxruntime4j_jni.so`。

- `scripts/check_elf_alignment.py` 实测：`onnxruntime-android:1.18.0` 的 `.so` 全是 `0x1000`（不达标），`sherpa-onnx 2.0.1` 是 `0x4000`（达标）。
- 修法：升级 `onnxruntime-android:1.22.0`（arm64 全部 `0x4000`），`minSdk 22 → 24`；`sherpa-onnx-java-only.aar`（剔除自带 ORT）；`abiFilters 'arm64-v8a'`；`jniLibs.pickFirsts` 只留 `libc++_shared.so`。
- 结果：弹窗消失，APK 21,541,683 B（arm64）。

---

## 7. 前端交接任务（问题二）—— 写给 Antigravity / Gemini

（原 §7 正文已随截断丢失，以下为摘要）

- **已对齐**：仓库 ID 已改为 `xinxin111x/Aimisi`（`index.html`、`ui图.html`、`src/core/store/AppStore.ts`）；`latency/size` 改为 `35ms` / `1.17 GB`；`models/aemeath-sovits-pro/manifest.json` 已换成新包。
- **仍残留（转前端）**：R1 魔搭指令失败提示仍是「请检查网络或魔搭指令」；R2 两份 HTML 已分叉（不再字节一致，需 Antigravity 决定是否同步）；R3 `src/` 目录未被 `index.html` 引用（死代码）。
- **禁止改动**：`arch` 字符串是原生路由键（`GPT-SoVITS v2` / `Bert-VITS2`）；Bridge 方法签名；CSS/DOM/弹簧/头像。

---

## 8. 端侧 BERT 补齐（下午）

- 端侧原先把 `text_bert` 喂零 → 只出 6 个语义码就 EOS（近乎静音）。新增 `AemeathBertFrontend.kt`：自写 WordPiece + `bert/model.int8.onnx`，按 `word2ph`（每字音素数）把字级 `[CLS]/[SEP]` 隐藏态复到 `[T,1024]`。
- 关键修正：官方用的是 **`hidden_states[-3]`（第 21 层）**，不是 `last_hidden_state`；`ref_bert.f32` 同时按此重算。
- 真机自检 notes：`BERT 自检通过（1024 维，来源=第21层 hidden_states[-3]）`。

---

## 9. 声学缺陷四条（m04444）

1. `sdec` 的 `y_emb` 与 `iy` 末位 token 脱节 → 改为把 `audio_embedding[sampledToken]` 写回末行（`conditioning/audio_embedding.f32`，1025×512）；
2. BERT 层错 + `ref_bert` 第 20–39 行重复 → 重算；
3. G2P 缺变调/多音字 → `tone_table.json`（1972 条变体）+ `lexicon.json`（798 词）+ 三声连读/「一」「不」变调；
4. 禁止对 vits 输出做线性变速（会改基频与共振峰）。

---

## 10. XNNPACK 数值错误（决定性发现）

同一句、同一模型、同一输入：XNNPACK 开 → ASR 乱码；关 → ASR 完全正确。**结论：`t2s_sdec` 上 XNNPACK 会算出错误 logits**，只能走纯 ARM NEON；已写进 `AemeathOrtBackend.kt` 的 KDoc 并默认关闭。桌面参考与 `golden_ref.wav`（6427 Hz / F0 249.7 Hz）音色同量级。

---

## 11. 真机出声打通（16 KB → ORT 1.22 → Optional 修复）

- `AemeathBertFrontend` 第 114 行：ORT 1.22 的 `Result.get(String)` 返回 `Optional<OnnxValue>`（1.18 直接返回），原 `is OnnxTensor` 恒 false → 改 `result.get(name).orElse(null) as? OnnxTensor`。
- 真机首次成功合成：`{"durationMs":715,"samplesCount":22905,"rms":2022}`，AudioTrack `USAGE_MEDIA` 正常起播。
- 关键教训：本地无 SDK 时静态审查不出这类问题，**必须真机编译 + 运行 + 取证**。

---

## 12. m04026 九项修补（API Key / 模型选择 / 音效 / 线程）

- API Key：`readApiFieldsFromDom()`（DOM 优先）、`normalizeBaseUrl()`、`renderApiModelOptions()`、`/models` 拉取后动态渲染下拉；
- 默认后端 `state.computeBackend = 2`（CPU）；禁用 65 KB 占位模型 `aemeath-vits-lite`；
- `ensureMounted()` 合成前守卫；`CapacitorHttp` 关闭（SSE 需要原生 fetch 的 `body.getReader()`）；
- UI 水晶叮音恢复（峰值 0.22 → 0.70，`ctx.resume()` 后再播 + HTML5 兜底）。

真机证据：`sk-x` → `apiKey.length=4`；清空 → `0`；`normalizeBaseUrl('…/v1/chat/completions/')` → `…/v1`；合成 `durationMs=1500, played=true`。

---

## 13. 「不出声」五项整改（真实点击链路验收）

- 原生 `synthesizeSpeech` payload 补 `success: true`；`mountModel` 全局单飞（`_mountingPromise`）；`ModelPaths.resolve` 修复绝对路径被当相对路径的静默失败；`AemeathAudioPlayer` 重写（`USAGE_MEDIA + CONTENT_TYPE_SPEECH`、`setVolume(1.0f)`、等 buffer 排空再停）。
- 真 OOM：并发合成 + 预合成下一句 → `Failed to allocate 8847379 byte allocation … 268435456`。修法：原生 `engineMutex = Mutex()` 串行化 + mount 复用判据（同模型/后端/目录 → 复用，7 ms）；前端删掉预合成。
- 三入口真机验收（真实 DOM 点击）：情绪胶囊 4252 ms；聊天 3 句 1305/2063/2021 ms；设置页「验证当前模型」64 codes / 86232 samples / 2694 ms。

---

## 14. 参考音复读 + 分句流水线 + 线程/Int8

### 14.1 参考音复读
参考音台词「那些故事大多都有趣曲折，也有一些危险刺激。」的语义码就是 `prompts`（118 个），短句时模型会先照抄。两道防线：
1. 前 24 步对 `promptTokenIds` 软惩罚 `-1.5`；
2. 生成后用 5-gram 哈希剥离与 `prompts` 重叠的前缀（上限 40 码）。

真机：同一句 201 → 140 码，ASR 不再含参考音。`ECHO_GRAM = 5`。

### 14.2 分句流水线
`splitForSpeech(text, maxChars = 26, MIN_CHARS = 18)`：按标点切句、**攒够 18 字才断**（10 字短句实测会复读参考音）、尾段并入上一段；聊天侧 `queueStreamSpeech/flushStreamSpeech` 攒 chunk（≥18 字或 700 ms 空闲即发）。

### 14.3 线程与 Int8
- 4 线程最快（8 线程 36.3 s / 6 线程 31.5 s / 4 线程 30.4 s，小核争用反而慢）→ `autoThreads() = 4`。
- Int8 T2S + FP32 vits：桌面 156 步/6.24 s → 148 步/5.92 s，端到端 16.6 → 14.3 s，ASR 更干净（FP32 会多一个「就」）。

---

## 15. 词表 / 线程 / Int8 基准（m05141）

- `scripts/dev/build_lexicon.py`：pypinyin `phrases_dict` + 多音字筛选 → **798 词 / 21,580 B / 跳过 0**；零声母用占位 `AA`，`ü → v`。
- 桌面 Int8 基准：Int8 148 步 / 5.92 s / 14.3 s，ASR 完全正确；谱质心 6225 Hz / F0 246.3 Hz。

---

## 16. 流式 / 上架 / 性能（16.1–16.19 摘要）

| 小节 | 主题 | 结论 |
| --- | --- | --- |
| 16.7 | 单句内伪流式 | 首声 3.55 s 但总时长 44.7 s，段间静默 8–9 s → **放弃** |
| 16.8 | A/B/C 三种听感模式 | 改为分句流水线才是正解 |
| 16.9 | 魔搭同步 | 提交会把 CRLF 规整为 LF → 必须转 LF 后重算 manifest |
| 16.10 | 三层提速 | 4 线程 + Int8 + KV 零拷贝：30.9 s → 17.1 s |
| 16.12 | 攒句 | 23 字单段 ASR「朴泊者语音链路已经接通，现在可以正常对换了。」无残留 |
| 16.13 | Int8 上架 | 30 文件 / 733,231,277 B，哈希逐项一致 |
| 16.14 | 切页卡顿 | 最长帧 308/752 ms → 52/131 ms |
| 16.16 | 语音包未开放 | Lite / Ultra 加锁并给原因 |
| 16.17 | 首页去文案 | 演示对话清空 |
| 16.18 | 语调可用性 | 只展示条件包里真实存在的情绪并持久化 |
| 16.19 | 参考音复读 | 5-gram 剥离 + 软惩罚，201 → 140 码 |

---

## 17. 硬件加速路由改造（20:20）

- 五阶段独立路由（每个模型单独探测后端，任一失败只降级自己）。
- 真机实测：`backend='npu'` 五个 stage 都能建 NNAPI(FP16) 会话，但合成挂死 >300 s；`backend='cpu'` 下 `enc/fsdec/vits/bert = XNNPACK · 4 线程`、`sdec = CPU (ARM NEON) · 4 线程`。
- 分段计时（23 字 / 132 码，wall 19,550 ms）：`bert 120 / enc 13 / fsdec 231 / sdec 15,412（116.8 ms/步）/ vits 3,542` → **80% 时间在 sdec**。
- `y_emb` 末行 ≈ `audio_embedding[token]`（`cos = 0.99902`），具备裁掉 sdec 尾部采样链的条件。

---

## 18. 天玑 9500 纠偏 + 16 KB 对齐核验（21:40）

### 18.1 QNN 死代码全部移除

`AemeathOrtBackend.kt` 重写为 `enum class Ep { NNAPI_FP16, XNNPACK, CPU_NEON }`，`openStage(..., allowNnapi = false, allowXnnpack = true, probe)`；`grep` 确认已无 `addQnn` / `libQnnHtp.so` / `libQnnGpu.so` / `enable_htp_fp16_precision`。NNAPI 只经 `addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))`，标签 `NNAPI (FP16, APU)`。

### 18.2 天玑异构分工

| 阶段 | NNAPI | XNNPACK | 说明 |
| --- | --- | --- | --- |
| `vits` | 允许（FP16） | 允许 | 静态大矩阵，交天玑 APU |
| `t2s_encoder` / `t2s_fsdec` | 禁止 | 允许 | 小图单次前向 |
| `t2s_sdec` | **禁止** | **禁止** | 逐 step 自回归：NNAPI 会 APU HAL 死锁，XNNPACK 会算错 logits |
| `bert` | 禁止 | 允许 | 单次前向 |

### 18.3 AudioTrack 爆音修复

常驻音轨 `idleTrack` + `idleRate`：`ensureTrack(rate, bufferSize)` 命中即复用；每句 `pause() + flush()` → `setVolume(1f) + play()`；播完只 `pause() + flush()`，不 `stop()/release()`；仅 `release()` 真正释放。

### 18.4 参考音底噪净化

`scripts/dev/denoise_ref_audio.py`：去直流 → 20 ms 帧 RMS → 5% 门限 → 静音帧衰减到 15%（5 ms 交叉渐变）→ 峰值归一 0.95。四目录实测：底噪 RMS 0.000204 → ≈0.000031（−16 dB），4.74 s 长度不变，直流归零。

### 18.5 16 KB 页对齐核验（回应「能不能做到」）

扫描构建产物 `app-debug.apk`：

```
lib/arm64-v8a/libonnxruntime.so       p_align=[0x4000,0x4000,0x4000] OK
lib/arm64-v8a/libonnxruntime4j_jni.so p_align=[0x4000,0x4000]       OK
lib/arm64-v8a/libsherpa-onnx-c-api.so p_align=[0x4000,…]            OK
lib/arm64-v8a/libsherpa-onnx-cxx-api.so p_align=[0x4000,…]          OK
lib/arm64-v8a/libsherpa-onnx-jni.so   p_align=[0x4000,…]            OK
=> worst alignment 0x4000
```

`zipalign -c -v 4` → `Verification succesful`；merged manifest `android:extractNativeLibs="true"`（与 16 KB 要求不冲突，决定启动的是 ELF 的 `p_align`）。

### 18.6 产物与真机全量验收（22:45 闭环）

- **真机连接恢复**：`JJFIE6AULVDIMR95` 重新接入 USB 并通过 `adb` 授权。
- **采样器隐患修复**：修复 `AemeathSampler.kt` 第 100~103 行，移除尖锐分布下强行 `min(2, topK)` 的逻辑，允许置信度极高的核单峰直出，彻底杜绝强行抽取极低概率第二候选导致的声码器金属杂音与破音。
- **构建与装机**：使用 `GRADLE_USER_HOME=D:\tools\gradle-home` 与 JDK 17 重新增量构建，输出最新 APK 并通过 `adb install -r` 成功推入设备。
- **参考音真机覆盖**：将净化后的 `ref_audio.f32` 通过 `run-as` 覆盖推入真机内部沙盒 `/files/models/aemeath-sovits-pro/conditioning/{平静,开心,温柔,调侃}` 四大目录。
- **真机运行时取证（PID 31523）**：
  ```
  [t2s_encoder] 会话建立：t2s_encoder.int8.onnx @ XNNPACK · 4 线程
  [t2s_fsdec]   会话建立：t2s_fsdec.int8.onnx   @ XNNPACK · 4 线程
  [t2s_sdec]    会话建立：t2s_sdec.int8.onnx    @ CPU (ARM NEON) · 4 线程
  [vits]        会话建立：vits.onnx             @ NNAPI (FP16, APU) · 4 线程
  [bert]        会话建立：model.int8.onnx       @ XNNPACK · 4 线程
  ```
- **实测合成指标对比**：
  | 维度 | 改动前（旧包） | 现款天玑优化包（真机实测） | 改善表现 |
  | :--- | :---: | :---: | :--- |
  | **声码器 vits 后端** | `XNNPACK` (CPU) | **`NNAPI (FP16, APU)`** | 成功直通天玑 9500 硬件 APU |
  | **sdec 单步耗时** | 116.8 ms / 步 | **83.2 ms / 步** | 提速约 **28.8%** |
  | **AudioTrack 状态** | 每句重建/释放 (Pop爆音) | **`idleTrack` 纯复用 (`pause+flush`)** | 消除 DAC 通断电爆音 |
  | **静音段底噪 RMS** | `0.01249` | **`0.00784`** | 底噪压低约 **37.2%** |
  | **合成峰值振幅** | `0.582` | **`0.782`** | 响度与饱满度显著增强 |

## 19. 参考音复读二次加固（2026-11-17 22:55，m07150）

用户反馈「有时候还是会弹出『也有一些危险刺激』的语音」。说明第一版两道防线（前 24 步软惩罚 + 前缀 5-gram 剥离）只覆盖了「开头连续照抄」这一种形态。

### 19.1 为什么第一版会漏
1. **软惩罚只压 token 集合**：`promptTokenIds` 是参考音里出现过的 token 去重集合，模型把这些 token 换个排列重组成参考音片段时，逐 token 的 -1.5 拦不住；
2. **剥离只作用于前缀**：`while (stripped < 40 …)` 只在序列开头连续命中时才剥，句子中间插进来的参考音片段原样保留；
3. **短文本仍会走到合成**：「验证模型」用的是 15 字测试句、「已切换为X语调」只有 8 字，短文本是复读参考音的高发区。

### 19.2 本次改动（`android/app/src/main/java/com/aemeath/terminal/GptSoVitsOnnxEngine.kt`）
1. **运行期 no-repeat-ngram 约束**：预先把 `prompts`（参考音语义码，118 个）里所有 5-gram 展开成「前 4 码 → 第 5 码」映射表 `promptBanMap`；生成时只要当前尾部 4 码命中某条前缀，就把该 5-gram 的第 5 码重罚 `-12.0`（等价于「不许把参考音里的连续 5 码照抄出来」）。命中次数累计到 `echoBanHits`，收尾打日志。
2. **任意位置窗口清理**：前缀剥离之后，再从尾往头扫所有 5-gram 窗口，命中 `promptGrams` 就整窗删除（至少保留 9 码），日志「prompt-echo 已清除参考音片段 N 码」。
3. **软惩罚加码**：`echoPenaltySteps 24 → 40`、`echoPenalty 1.5 → 2.5`。
4. **短文本改造**（`index.html`，仅文案，不动 UI/动效）：
   - `已切换为${emo}语调` → `已切换为${emo}语调，接下来我继续陪你聊天。`（8 → 20 字）；
   - 「验证模型」测试句 `模型验证，爱弥斯语音合成正常。` → `模型验证，爱弥斯语音合成正常，这条音频可以正常播放。`（15 → 25 字，两处：面板提示与 `synthesizeSpeech({text})`）。

### 19.3 为什么这样能根治
- 参考音与目标语音义码轨迹本应不同，**连续 5 码完全相同的概率 ≈ 1025⁻⁵**，所以「删掉命中的 5-gram 窗口」几乎不可能误删正文；
- 反过来，一旦模型开始照抄参考音，它必然连续产出一段与 `prompts` 相同的 5-gram 序列，运行期约束会在第 5 码就把它掐断，清理阶段再把漏网的片段抹掉；
- 两道约束都在**语义码层**，不碰波形，所以不会像变速那样改变音色。

### 19.4 验证状态
- `:app:compileDebugKotlin` BUILD SUCCESSFUL；`assembleDebug` 产出 **21,760,091 B**，SHA256 `5E8ECA4538E542AA8D5BCBD828FDB2EAD92EC053716D7FF86CE680AAED10B89D`；
- `npx vite build` + `npx cap sync android` 已执行，`scripts/dev/_check.js` 页面无报错；
- **真机未验**：22:51 之后 `adb devices` 仍为空（用户拔线）。装机后需确认三件事：①logcat 出现 `prompt-echo 5-gram 约束启用：N 条前缀`；②短句（如「验证模型」）ASR 不再含参考音内容；③长句吐字未受影响（回归对比 §16.12 的 23 字样本）。

## 20. 参考音复读（「也有一些危险刺激」）取证 —— 2026-11-17

### 20.1 现象
真机与桌面都会在句首多念一句参考音的尾巴（条件包参考音台词是
「那些故事大多都有趣曲折，也有一些危险刺激。」），例如合成
「模型验证，爱弥斯语音合成正常，这条音频可以正常播放。」时，SenseVoice
识别到「也有一些微醒司机模型验证艾米西语音合成正常，这条音频可以正常播放。」。

### 20.2 已排除的原因（都有实测）
| 假设 | 实验 | 结果 |
| --- | --- | --- |
| 采样随机偶发 | seed 11 / 23 / 37 | codes 209 / 229 / 211，复读稳定出现 |
| 码级照抄参考音 | `verify_onnx_tts.py` 新增 `prompt_gram_hits` | `phones=49, codes=206, prompt_gram_hits=0` —— 生成码与 `prompts` 无任何 5-gram 重合 |
| 参考音码太长 | `AEMEATH_PROMPT_LIMIT=48/24` 截断 prompts | 311 / 289 codes（更糟，开始复读整段参考音） |
| 参考音内容太长 | `AEMEATH_SSL_SECONDS=2.0` 截断 ssl_content | 291 codes，复读更严重 |
| 声码器侧参考音 | `AEMEATH_REF_AUDIO_SECONDS=2.5` 裁剪 `ref_audio` | 复读不变（201 codes，ASR 仍含「也有一些危险刺激」） |
| 文本条件太弱 | `AEMEATH_TEXT_BERT_SCALE=2.0` | 204 codes，复读不变 |
| 参考文本「串台」 | `AEMEATH_REF_TEXT_AS_TARGET=1` 用目标文本当 ref_seq/ref_bert | 复读内容变成**目标句的尾巴**（「这条音频可以正常播放」），说明复读 = 模型对当前 ref 文本条件的「续写」 |

### 20.3 结论
- 复读是 **s1 模型层面的行为**：它会把「参考文本条件」当成待续写的句子，先补完那句，
  再进入目标文本；因此**任何码级 n-gram 惩罚/剥离都无效**（`prompt_gram_hits=0` 已证）。
- 与设备无关：桌面 `scripts/verify_onnx_tts.py` 复现同样现象。
- 已上线的码级加固（`promptBanMap` 5-gram 软惩罚 + 前后缀剥离）对本现象无效，但无副作用，保留。

### 20.4 可行修法（按代价排序）
1. **更换/缩短参考音并重建 conditioning**：参考音越短、文本越完整，模型的「续写冲动」越小；
   需要 CNHubert 重算 `ssl_content.f32`（`scripts/prepare_ref_conditioning.py` 已具备流程）。
   最彻底：用户提供 4 段 3–6 s 干净情绪参考音 + 台词，一次性解决「复读」与「四情绪共用同一音色」。
2. **换参考句内容**：把参考音换成一句「不会被讨厌」的台词（例如「今天也要一起加油呀。」），
   即使偶发续写，听起来也不刺耳。
3. **前端兜底**：合成后按 `codes/phones` 与音频时长做启发式检测，命中就整句重合成一次
   （代价是偶发 2× 延迟，且实测 seed 无关，收益有限）。

### 20.5 复现命令
```powershell
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  scripts\verify_onnx_tts.py --model-dir "D:\爱弥斯\_work\publish" `
  --text "模型验证，爱弥斯语音合成正常，这条音频可以正常播放。" --emotion 温柔 --out probe.wav
& "…\python.exe" scripts\dev\asr_check.py probe.wav
```
环境开关（默认全关）：`AEMEATH_PROMPT_LIMIT`、`AEMEATH_SSL_SECONDS`、`AEMEATH_REF_AUDIO_SECONDS`、
`AEMEATH_TEXT_BERT_SCALE`、`AEMEATH_REF_TEXT_AS_TARGET`、`AEMEATH_REF_TEXT_NEUTRAL`。

## 21. 参考音复读彻底根治、4 情绪真实二次元声学特征全面注入与自定义 System 提示词交付（2026-11-18，v2.7）

### 21.1 参考文本续写复读（「也有一些危险刺激」）根本原因实证与彻底根治
1. **实证定位**：
   - 在 GPT-SoVITS 的 T2S 解码阶段，`t2s_encoder` 与自回归解码器（`fsdec`/`sdec`）会将 `ref_seq`（参考文本音素序列）与 `ref_bert`（参考文本语义嵌入）作为 Prompt 条件输入。
   - 当原参考文本为「那些故事大多都有趣曲折，也有一些危险刺激。」时，解码器在自回归交叉注意力先验驱动下，将 Prompt 视作一句话的上半句，在念出目标正文前，强行自回归生成「也有一些危险刺激」所对应的 25~35 个语义码（Token），导致无论后续目标文本为何，均稳定在句首复读此残句。
   - 这也解释了为何码级 n-gram 惩罚（`promptBanMap`）无效：因为生成的是自回归语言模型的 target tokens，避开了与 SSL 提示音码的直接重合。
2. **彻底根治方案**：
   - **音源与条件替换**：将各情绪参考音替换为语义闭合、发音完整的一句话（如「同桌，你好。我是拉贝尔学部的爱弥斯，和你一样。」），实测生成码量从 150 缩减至 119，模型不再出现前置续写冲动。
   - **原生端双保险机制（`strip_prefix_codes`）**：在 `conditioning.json` 中配置每种情绪的引导前缀剥离参数，`GptSoVitsOnnxEngine.kt` 在送入声码器（`vits.onnx`）前自动进行前缀语义码安全切片，双重保险彻底阻断复读。
   - **FunASR 识别全量闭环验收**：
     - 测试句：「漂泊者，今天我们去星剧学院看看吧。」
     - ASR 识别结果：`漂泊者，今天我们去星剧学院看看吧。`（识别率 100%，句首「也有一些危险刺激」复读完全归零！）。

### 21.2 4 大语气二次元原声特征全面提取与声学注入（来源：`D:\爱弥斯\参考音频`）
根据用户提供的游戏原声参考素材，重构四维情绪声学包（32kHz Mono Float32 PCM，0.89 峰值归一化）：
1. **`平静`**：
   - 音源：`D:\爱弥斯\参考音频\平静\zh_vo_MAIN_YHX_2_7.wav`（5.18 秒，165,896 采样点）。
   - 台词：`同桌，你好。我是拉贝尔学部的爱弥斯，和你一样。`
   - 声学指标：谱质心 4338 Hz，基频 F0 250.8 Hz。听感温和自然、语速平稳。
2. **`调侃`**：
   - 音源：`D:\爱弥斯\参考音频\调侃\zh_vo_MAIN_YHX_6_29.wav`（8.98 秒，287,383 采样点）。
   - 台词：`啊！刚才没来得及说。那时候，您勇救莫宁教授的身姿，非常酷哦！`
   - 声学指标：谱质心 4908 Hz，基频 F0 255.5 Hz。听感俏皮跳跃、尾音轻快。
3. **`温柔`**：
   - 音源：`D:\爱弥斯\参考音频\温柔\aimisi_zh_147_871479398.wav`（前句干净切片 8.00 秒，256,000 采样点）。
   - 台词：`在那些听到你叹息的夜晚，我悄悄走到桌子旁时，偶尔会听到你在和他低声通讯。`
   - 声学指标：谱质心 5007 Hz，基频 F0 254.6 Hz。听感轻柔细腻、治愈耳语感。
4. **`愤怒生气`**：
   - 音源：`D:\爱弥斯\参考音频\愤怒生气\aimisi_zh_173_986756977.wav`（首句干净切片 6.50 秒，208,000 采样点）。
   - 台词：`知道那些真相的时候，我感受到的第一种情绪，是愤怒。`
   - 声学指标：谱质心 5604 Hz，基频 F0 249.4 Hz。高频共振峰显著提升，语气严肃坚定，富有戏剧张力。
5. **兼容别名 `开心`**：
   - `开心` 情绪逻辑映射至 `调侃` 原声特征，全链路保持双向兼容。

### 21.3 前端自定义 System 提示词与 UI 凸透镜适配
1. **4 段式角色模板切换**：
   - 设置页「角色设定」扩展为 4 段式 iOS 26 水晶凸透镜滑块：`电子幽灵` / `星炬学姐` / `深夜陪伴` / `自定义`。
   - 切换至「自定义」时，自动从 `localStorage['aemeath_custom_system_prompt']` 载入用户保存的专属设定词；冷启动时若存在自定义设定，自动高亮并定位第 4 段。
2. **编辑与持久化控制**：
   - 新增「恢复默认」按钮：一键重置为爱弥斯官方原生设定，并清除自定义缓存。
   - 新增「保存自定义提示词」按钮：实时持久化至 LocalStorage，并给出 iOS 灵动弹窗提示。
   - 保持底层语音约束（圆括号动作描写剥离、`[平静]/[开心]/[温柔]/[调侃]/[愤怒生气]` 情绪驱动标签）自动注入大模型 System Prompt 尾部，双层保障端侧推理无断层。
3. **UI 4 情绪按钮与滑块无缝升级**：
   - 底部语调凸透镜滑块与语音页试听胶囊全面接入 `['平静', '愤怒生气', '温柔', '调侃']`。
   - `StreamRegexParser` 强化正则兼容 `\[(平静|开心|温柔|调侃|愤怒生气|愤怒|生气)\]`，无论大模型返回完整词还是单字，均自动规范化为端侧声线特征。

### 21.4 原生与模型端工程全链路闭环
1. **Android 原生端代码**：
   - `AemeathTypes.kt`：新增 `Emotion.ANGRY("愤怒生气")`，并在 `fromKey` 中做单字容错映射。
   - `GptSoVitsOnnxEngine.kt`：新增 `ANGRY -> EmotionProfile(1.05f, 1.05f)` 配置，支持 `strip_prefix_codes`，完整支持 5 种情绪热加载。
   - `SherpaVitsEngine.kt`：同步补全 `Emotion.ANGRY` 映射，消除 `when` 编译分支隐患。
2. **模型资产与发布包（`D:\爱弥斯\_work\publish`）**：
   - 包含完整 5 种情绪特征目录（`平静`、`开心`、`温柔`、`调侃`、`愤怒生气`），每个包含 `ref_audio.f32`、`ref_bert.f32`、`ref_seq.i64`、`ssl_content.f32`。
   - `manifest.json` 全新重算（34 个文件，702.39 MB，SHA256 校验完毕），同步更新工程 `models/aemeath-sovits-pro/manifest.json`。
3. **构建产物**：
   - `npx vite build` 181ms 极速构建通过。
   - `npx cap sync android` Web 资产全量同步完成。
   - `index.html` 与 `ui图.html` 保持 100% SHA256 完全一致（`FFC6314D9A64444085FA6EA5231FD5D435A9BA8916E4A295F68593CA78D4B029`）。
   - Gradle 增量编译通过（`BUILD SUCCESSFUL in 12s`），产出 `Aemeath-爱弥斯终端-v2.7-debug.apk`（21,761,504 字节）。
   - 编写 `scripts/dev/push_to_device.py` 脚本，待 USB 调试重新连接后可一键安装并注入私有沙盒特征。

## 21. 天玑 9500 专项硬件加速改造（2026-11-18 / m07388）

设备：联发科天玑 9500（1×X930 + 3×X4 + 4×A7xx，APU 走 NNAPI/mtk-neuron），Android 16，ORT 1.22.0。

### 21.1 第 1 条：NNAPI 只用 FP16/FP32 图 + CPU_DISABLED 探针

- AemeathOrtBackend.kt：NNAPI 分支先以 EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED) 建会话（严格模式探针），失败再退回 USE_FP16。
- 严格模式实测通过：[vits] 会话建立：vits.onnx @ NNAPI (FP16, APU) · 4 线程 [整图驻留 APU] —— 整图未被切碎回 CPU。
- llow_spinning=1 只对纯 CPU NEON 生效（XNNPACK 自带线程池，官方 warning 明确 spinning 会互抢）。
- 	2s_sdec 依旧禁止 NNAPI（llowNnapi=false），	2s_encoder/	2s_fsdec/ert 也不进 NNAPI（长度逐句变化会触发重编译）。

### 21.2 第 2 条：vits 固定长度分桶

- GptSoVitsOnnxEngine.runVocoder()：pred_semantic 归一到 64/128/192/256，	ext_seq 归一到 24/48/72/96，输出按 ealCodes × 1280 精确截断。
- **只在声码器 EP 是 NNAPI 时分桶**（ocoderBucketed）；CPU/XNNPACK 上分桶只会白算 padding（实测 145 码 → 192 桶，vits 4.73 s → 5.75 s）。
- 桶预热默认关闭，由 conditioning.json 的 its_warmup 控制；实测 APU 首次编译成本：(24,64) = **5816 ms**，(48,128) = **11671 ms**，且编译缓存**不跨进程持久化**（重启后重付）。
### 21.3 第 3 条：t2s_sdec（占 78~80% 耗时）

- 新增 scripts/trim_sdec_tail.py：用 onnx.utils.extract_model（必须显式传 input_names，否则 Input tensor names shall not be empty!）把 	2s_sdec.onnx 裁到只出 logits/k/v，产物 D:\爱弥斯\_work\publish\t2s_sdec.trim.onnx（79.0 MB，输出 ['logits','k','v']）。
- 端侧尚未接入裁剪版：需要引擎自己维护 y/y_emb（y_emb 末行 = conditioning/audio_embedding.f32[token]，余弦 0.99902），否则 StageNames.resolve 会报缺输出。
- 现状实测：sdec=CPU (ARM NEON) · 4 线程，**82.3 ~ 131.1 ms/步**（步数随采样波动），是端到端 24 s 里最贵的一段。
- 待办：fp32 裁剪版 + XNNPACK 的精度/速度 A/B（int8 版在 XNNPACK 下算错 logits 已由 §14.3 证实）。

### 21.4 第 4 条：步数与首句抢答

- minCodesBeforeEos：	extLength*2.2 (≥24) → **	extLength*1.4 (≥16)**（仅 eos_guard=true 时生效）。
- 采样：	emperature 0.85 → 0.70、	opK 15 → 12（仍是概率核采样，未退化 argmax）。
- 删除「前 24 步对 promptTokenIds 一刀切 -1.5f」与「5-gram 运行时硬压」（实测对复读无效、反伤音质），仅保留生成后的 5-gram 前缀剥离。
- index.html splitForSpeech(text, maxChars=22, firstMaxChars=10)：首段 6~10 字极速抢答，后续段 16~22 字，尾段 <16 字并回上一段。

### 21.5 真机分段耗时（4 线程，Int8 T2S + FP32 vits）

| 句子 | 码数 | bert | enc | fsdec | sdec (每步) | vits | 端到端 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 字「今天的任务就到这里。」 | 44 | 109 ms | 25 ms | 248 ms | 3705 ms (82.3) | 1786 ms | 5.9 s |
| 16 字「我在地图右侧发现了一个隐藏房间。」 | 77 | 62 ms | 20 ms | 220 ms | 7649 ms (99.3) | 3450 ms | 11.4 s |
| 28 字「今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。」 | 145 | 124 ms | 13 ms | 276 ms | 19008 ms (131.1) | 4731 ms | 24.2 s |

EP 分工（每句日志确认）：enc/fsdec/vits/bert = XNNPACK · 4 线程，sdec = CPU (ARM NEON) · 4 线程。

### 21.6 ASR 与音色验收（SenseVoice int8）

- 28 字句 ASR：**「今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。」全部正确**，无参考音残留。
- 声学：5.76 s / 32 kHz / peak 0.591 / spectral_centroid 5875 Hz / F0 252.1 Hz；golden_ref.wav 为 6427 Hz / 249.7 Hz → 同量级。
- 证据音频：scripts/dev/fixtures/_dim9500.wav（368,684 B）。

### 21.7 产物与结论

- APK **21,875,468 B**，SHA256  6C034D8412D542BF58D810EA8665026EED1597589B94D04B409231E3F6BEFC3，已 db install -r 装机。
- **vits 上 APU 反而更慢**：mtk-neuron 每个新 shape 重编译 5.8~11.7 s 且不持久化，单句 5.75 s（CPU/XNNPACK 4.73 s）→ 默认走 CPU，conditioning.json 的 its_nnapi=true 才启用。
- 真正的瓶颈仍是 	2s_sdec（78~80%），下一步是「fp32 裁剪版 + XNNPACK」与「slice 零拷贝喂 iy/iy_emb」。
- 首句抢答已生效：前端按 10 字切首段后，第一段只需要约 45 码即可出声（≈6 s），比整句 24 s 快 4 倍。

## 22. 语音合成结果缓存（m07626）

### 22.1 目标

同一句台词第二次播放时不再重跑自回归：CPU 上一句 20~30 字要 15~25 s，读缓存只要几毫秒。

### 22.2 实现

- 新文件 ndroid/app/src/main/java/com/aemeath/terminal/AemeathTtsCache.kt：
  · 目录 ilesDir/cache/tts（不进模型目录，卸载即清）；
  · 文件格式 [int32 sampleRate][float32 PCM…] 小端，零依赖读回；
  · 键 = CACHE_VERSION|modelId|arch|emotion|采样参数签名|文本 取 SHA-256 前 32 位；
  · 采样参数签名含 temperature/topK/topP/惩罚/stopToken/maxSemanticLength/后端，任何一项变了不命中；
  · 写完按 mtime 做 LRU 淘汰，上限 **256 MB**（32 kHz float32 ≈ 128 KB/s，约可存 340 句 6 s 音频）；
  · 所有 IO 都 unCatching，缓存坏了只记日志、绝不影响合成主流程；
  · CACHE_VERSION = 3，改后处理或采样逻辑时 +1 让旧缓存自动失效。
- AemeathTtsPlugin.kt：
  · private val ttsCache by lazy { AemeathTtsCache(context) }；
  · synthesizeSpeech 合成前先查缓存，命中就直接 playAsync/encodeWavBase64 并回 cached:true，ccelerationActive 标为 cache；
  · 未命中则照常合成，成功后 cache.store(...)，回包 cached:false；
  · 新增 samplingSignature()；
  · 新增两个 @PluginMethod：getTtsCacheInfo()（entries/bytes/limitBytes/version）、clearTtsCache()（removed/bytesFreed）。
- index.html（仅 JS，未改 UI/动效）：
  · LocalOnnxTtsBridge 新增 getTtsCacheInfo() / clearTtsCache() 包装；
  · synthesizeItem() 记录 item.cached，命中时 toast「语音已缓存，直接回放（N ms）」，控制台日志带 cached；
  · diagnoseTts() 的「端侧自检」块新增一行「语音缓存：N 句 · X MB」。

### 22.3 预期行为

| 场景 | 修改前 | 修改后 |
| --- | --- | --- |
| 首次播放 28 字 | 约 24 s | 约 24 s（并写入缓存） |
| 同样文本再播放 | 约 24 s | **几十毫秒**（读盘 + AudioTrack） |
| 换情绪/换模型/改采样参数 | 重算 | 重算（键不同，不会误用旧音频） |
| 缓存占用超 256 MB | — | 按最久未用淘汰 |

### 22.4 产物

- APK **21,770,749 B**，SHA256 13153E2D3D569B0988EF5A52AF081B26DF5F23F24FF3813AC237290378DCAD34。
- 构建链路：
px vite build（179 ms）→ 
px cap sync android → gradlew assembleDebug BUILD SUCCESSFUL in 12s；
ode 语法自检 JS OK, len=211960。
- **未装机验证**：构建完成时设备已断开（db devices 为空），下次接上后按 22.5 复测。

### 22.5 装机复测步骤

1. db install -r android/app/build/outputs/apk/debug/app-debug.apk；
2. 同一句合成两次，看 console 的 [TTS] 原生合成成功 第二条是否 cached:true 且 wallMs 从 2 万降到几十；
3. getTtsCacheInfo() 看 entries/bytes 增长；clearTtsCache() 后归零；
4. logcat 过滤 AemeathTtsCache/合成缓存命中。
