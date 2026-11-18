"""追加 PROGRESS.md §20：参考音复读取证（m07150）。UTF-8 直写，避免 PowerShell GBK 事故。"""
from pathlib import Path

P = Path(r"D:\语音模型项目\PROGRESS.md")

SECTION = """
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
& "C:\\Users\\Administrator\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" `
  scripts\\verify_onnx_tts.py --model-dir "D:\\爱弥斯\\_work\\publish" `
  --text "模型验证，爱弥斯语音合成正常，这条音频可以正常播放。" --emotion 温柔 --out probe.wav
& "…\\python.exe" scripts\\dev\\asr_check.py probe.wav
```
环境开关（默认全关）：`AEMEATH_PROMPT_LIMIT`、`AEMEATH_SSL_SECONDS`、`AEMEATH_REF_AUDIO_SECONDS`、
`AEMEATH_TEXT_BERT_SCALE`、`AEMEATH_REF_TEXT_AS_TARGET`、`AEMEATH_REF_TEXT_NEUTRAL`。
"""

text = P.read_text(encoding="utf-8")
if "## 20. 参考音复读" not in text:
    P.write_text(text.rstrip() + "\n" + SECTION, encoding="utf-8")
    print("appended")
else:
    print("already present")
print(len(P.read_text(encoding="utf-8").splitlines()), "lines")
