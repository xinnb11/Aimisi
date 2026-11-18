# 问题定位：端侧 TTS 句首复读参考音（「也有一些危险刺激」）

> 交接对象：接手修复的工程师 / Agent。以下每条都可用文末命令复现，数字均为实测。

## 1. 现象

合成任意中文句子时，音频开头**偶发**多出一段参考音的内容，最长可到一整个短句。

| 场景 | 输入文本 | SenseVoice ASR 结果 | 时长 |
| --- | --- | --- | --- |
| 真机（天玑 9500） | 模型验证，爱弥斯语音合成正常，这条音频可以正常播放。 | **也有一些微醒司机**模型验证艾米西语音合成正常，这条音频可以正常播放。 | 8.32 s |
| 桌面参考实现 | 同上 | **也有一些我醒司机，我行**验证艾米斯语音合成正常，这条音频可以正常播放。 | 9.24 s |
| 真机 | 模型验证，爱弥斯语音合成正常。 | **也有一些危险刺激，**无形验证艾米斯语音合成正常。 | 5.72 s |
| 真机 | 今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。 | 今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。 | 6.16 s ✅ |
| 真机 | 你好。 | 你好。 | 0.84 s ✅ |

- 多出来的那句话，正好是**条件包参考音台词的最后一段**：`conditioning/*/ref_text` =
  「那些故事大多都有趣曲折，**也有一些危险刺激。**」
- 不是每次都有：短句（`你好。`）与部分长句干净，26 字、14 字两类句子稳定复读 → 用户感知为「有时候会」。

## 2. 端侧调用链（谁给谁喂什么）

```
conditioning/{情绪}/ssl_content.f32   (1,768,236)  ← CNHubert 特征，50 Hz × 236 帧 = 4.72 s
conditioning/{情绪}/ref_seq.i64       (40)         ← 参考音台词的音素 id（symbols2.py，732 项）
conditioning/{情绪}/ref_bert.f32      (40,1024)    ← 参考文本层 21（hidden_states[-3]）特征
conditioning/{情绪}/ref_audio.f32     (1,151680)   ← 参考音波形 4.74 s @32 kHz

t2s_encoder.onnx  (ref_seq, text_seq, ref_bert, text_bert, ssl_content) → x[1,T,512], prompts[1,118]
t2s_fsdec.onnx    (x, prompts) → y, k, v, y_emb, x_example
t2s_sdec.onnx     (iy, ik, iv, iy_emb, ix_example) → y, k, v, y_emb, logits[·,1025], samples
vits.onnx         (text_seq, pred_semantic, ref_audio) → audio[1,L]  # L = codes × 1280 @32 kHz
```

- `prompts`（118 个）= 参考音的语义码，由 encoder 图内 `ssl_proj/Conv` + VQ codebook ArgMax 得到。
- 自回归循环在 `t2s_sdec.onnx` 上跑，**运行时采样**（top_k=15 / T=0.85 / top_p=0.9 / penalty=1.35），
  严禁退化为 argmax；每步把「图输出的 y 末位」替换成采样 token 后再进下一步（KV cache 才会跟随）。
- 相关 Kotlin 实现（`android/app/src/main/java/com/aemeath/terminal/GptSoVitsOnnxEngine.kt`）：
  - `:368` `echoPenaltySteps = 40`、`:375` `promptBanMap`、`:387` `echoBanPenalty = 12.0f`
  - `:455` 开头 40 步对 `promptTokenIds` 软惩罚、`:467` 运行期 5-gram no-repeat 约束
  - `:561` 生成后剥离与 `prompts` 重合的前缀、`:577` 反向整窗清除
  - `:1278` `ECHO_GRAM = 5`
  > 以上四道防线**对本问题全部无效**（原因见 §3），但无副作用，保留即可。

## 3. 已经排除的原因（都有实测）

| 假设 | 实验/开关 | 结果 | 结论 |
| --- | --- | --- | --- |
| 采样随机、重试就好 | seed 11 / 23 / 37 | codes 209 / 229 / 211（基线 206/213），全部复读 | ❌ 采样无关 |
| 码级照抄参考音 | `verify_onnx_tts.py` 的 `prompt_gram_hits` 诊断 | `phones=49, codes=206, prompt_gram_hits=0` | ❌ 生成码与 `prompts` **零** 5-gram 重合 |
| 参考音码太长 | `AEMEATH_PROMPT_LIMIT=48 / 24` | 311 / 289 codes（更长） | ❌ 反而更糟 |
| ssl 内容太长 | `AEMEATH_SSL_SECONDS=2.0` | 291 codes，复读更严重 | ❌ 反而更糟 |
| 声码器参考音串味 | `AEMEATH_REF_AUDIO_SECONDS=2.5` | 201 codes，复读不变 | ❌ 与 vits 无关 |
| 文本条件太弱 | `AEMEATH_TEXT_BERT_SCALE=2.0` | 204 codes，复读不变 | ❌ 无关 |
| 设备/ORT/线程问题 | 桌面 `verify_onnx_tts.py` 同句 | 同样复读（9.24 s） | ❌ 非设备特有 |
| **参考文本被“续写”** | `AEMEATH_REF_TEXT_AS_TARGET=1`（把 `ref_seq/ref_bert` 换成目标文本） | 复读内容随之变成**目标句的尾巴**（“这条音频可以正常播放”） | ✅ **根因在此** |

## 4. 根因判断

模型（Aimisi_v2Pro_e8，GPT-SoVITS v2Pro 架构，25 Hz 语义码、32 kHz 声码器）把
`ref_seq/ref_bert`（参考文本条件）当成**一句待续写的话**：先补完这句，再开始念 `text_seq/text_bert`（目标文本）。
因此：

1. 复读的内容 = 参考文本的尾部，与采样、码级重复无关；
2. 参考音/参考文本越长、越“完整”，模型越容易补完 → 4.74 s / 21 字的参考音属偏长；
3. 只要参考条件不变，任何**解码侧**手段（惩罚、剥离、重采样、换 seed）都无效；
4. 目标文本较短时更容易触发（`你好。` 不触发，14–26 字触发）。

## 5. 可行修法（按代价排序）

### 方案 A（推荐）：重建 conditioning，换成短而干净的参考音
- 需要：每个情绪一段 **3–6 s** 干净人声（`wav`，单声道，24/32 kHz）+ **逐字对应**的台词文本。
- 做法：`scripts/prepare_ref_conditioning.py`（已具备流程）→ 用 CNHubert(`chinese-hubert-base`) 重算
  `ssl_content.f32`，用 BERT 第 21 层重算 `ref_bert.f32`、用 `symbols2.py + char2phone.json` 生成 `ref_seq.i64`，
  同步写 `conditioning.json` 与 `manifest.json`（含 SHA256）。
- 附带收益：现在四个情绪目录的 `ssl_content.f32` **SHA256 完全相同**（同一段参考音），
  所以「情绪切换」对音色没有任何影响；换 4 段独立参考音可一并修复「语气不对」。

### 方案 B（应急）：截短现有参考音重算
- 取 `D:\爱弥斯\_work\pack\Aimisi_e8_Backup\golden_ref.wav`（32 kHz / 4.74 s）前 **2.5 s**，重算
  `ssl_content/ref_seq/ref_bert/ref_audio` 四件套（需重新下载 CNHubert ≈1.9 GB）。
- 代价：音色略变；收益：复读内容缩短（仍可能残留前半句）。

### 方案 C（纯兜底，不推荐单独使用）
- 前端检测 `codes / phones > 3.0` 或音频时长异常 → 整句重合成。实测 seed 无关，收益有限。

## 6. 验收标准（必须用 ASR，不许只看 `samplesCount > 0`）

准备 5 句测试集（含长短、含标点、含多音字）：

1. 今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。
2. 模型验证，爱弥斯语音合成正常，这条音频可以正常播放。
3. 漂泊者，语音链路已经接通。现在可以正常对话啦！
4. 觉得累了就靠一会儿，我在星炬学院等你回来。
5. 嗯，走吧。

逐条 `adb pull` 合成 WAV → `scripts/dev/asr_check.py` 识别，要求：

- 5/5 **ASR 与输入文本一致**（允许「爱弥斯/艾米斯」这类同音差异），**句首不得出现参考音内容**；
- `codes / phones ≤ 3.0`（干净样本实测 2.3–2.6，复读样本 4.2–4.7）；
- 音色对齐 `golden_ref.wav`：谱质心 5,500–6,600 Hz、F0 均值 230–270 Hz（当前实测 5,481–6,014 Hz / 234–261 Hz）；
- 4 个情绪各自合成同一句，谱质心/F0 必须有可测差异（现在四者完全相同）。

## 7. 复现命令

```powershell
# 桌面参考实现（不需要设备）
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  scripts\verify_onnx_tts.py --model-dir "D:\爱弥斯\_work\publish" `
  --text "模型验证，爱弥斯语音合成正常，这条音频可以正常播放。" --emotion 温柔 --out probe.wav
& "C:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  scripts\dev\asr_check.py probe.wav         # SenseVoice int8，模型在 D:\tools\asr\sv

# 实验开关（默认全关，见 scripts/verify_onnx_tts.py）
#   AEMEATH_PROMPT_LIMIT / AEMEATH_SSL_SECONDS / AEMEATH_REF_AUDIO_SECONDS
#   AEMEATH_TEXT_BERT_SCALE / AEMEATH_REF_TEXT_AS_TARGET / AEMEATH_REF_TEXT_NEUTRAL

# 真机（设备 JJFIE6AULVDIMR95）
$env:Path="D:\tools\android-sdk\platform-tools;$env:Path"
adb forward --remove-all
adb forward tcp:9333 localabstract:webview_devtools_remote_$((adb shell pidof com.aemeath.terminal).Trim())
# 再用 scripts/dev/_dev.js 调 window.AppController.tts.synthesizeSpeech / getNativePlugin().synthesizeSpeech
```

## 8. 约束（改代码时不许违反）

1. 不许改 `index.html` 的 CSS / DOM 结构 / 凸透镜弹簧物理 / 圆形头像逻辑（改内联 JS 逻辑可以）。
2. 解码**不许退化为 argmax**（`top_k=15`、`temperature=0.85`、`top_p=0.9`、`repetition_penalty=1.35` 必须保留。
3. 不许对 `vits.onnx` 输出的 PCM 做线性插值变速（会改基频/共振峰）。
4. XNNPACK 必须保持**关闭**（在 `t2s_sdec` 上会算错 logits：实测开→乱码、关→完全正确）。
5. `t2s_sdec` 不许进 NNAPI（天玑 APU 会死锁，实测挂死 >300 s）；`vits` 可以走 `NNAPI(FP16)`。
6. 模型文件、`conditioning/*` 的任何改动都要同步 `manifest.json` 的 sha256 / size（端侧按它校验）。

## 9. 相关文件

| 路径 | 说明 |
| --- | --- |
| `D:\爱弥斯\_work\publish\` | 模型发布目录（30 files / 733,231,277 B / 699.3 MiB） |
| `…\publish\conditioning\conditioning.json` | 情绪 → 四件套文件 + `ref_text` + 四个开关（`sync_y_emb`/`eos_guard`/`tone_sandhi`/`lexicon_override`，当前全 true） |
| `…\publish\conditioning\{平静,开心,温柔,调侃}\` | `ssl_content.f32` / `ref_seq.i64` / `ref_bert.f32` / `ref_audio.f32` |
| `…\publish\manifest.json` | 端侧下载校验清单（sha256 + size + 采样参数） |
| `scripts\verify_onnx_tts.py` | 桌面参考实现 + 诊断（`prompt_gram_hits` 等） |
| `scripts\prepare_ref_conditioning.py` | 参考音 → 四件套 的重建脚本（方案 A/B 用） |
| `scripts\dev\asr_check.py` | SenseVoice ASR + 谱质心 + F0 |
| `scripts\dev\build_lexicon.py` | 798 词多音字表（`g2p/lexicon.json`） |
| `android\…\GptSoVitsOnnxEngine.kt` | 端侧四段推理 + 采样 + 现有（无效的）码级加固 |
| `PROGRESS.md` §20 | 本问题的取证记录 |

## 10. 环境坑

- `adb` 要显式加 PATH：`D:\tools\android-sdk\platform-tools`。
- Windows 控制台是 GBK：中文输出用 `$env:PYTHONIOENCODING="utf-8"` + `Out-File -Encoding utf8` 再读；
  写 `PROGRESS.md` 这类中文文件请用 UTF-8 直写（曾因 PowerShell here-string 把 1,848 行截断成 72 行）。
- `AppController` 在页面里先是 class，起完 Portal 才是实例；CDP 里一律用 `window.AppController`。
- devtools 端口每次重启应用都要 `adb forward --remove-all` 重建。
