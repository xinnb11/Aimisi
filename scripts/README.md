# `scripts/` — AutoDL 云端预处理 · 打标 · ONNX 混合量化流水线

本目录是「爱弥斯 (Aemeath) 离线语音陪伴终端」的**云端训练与量化侧**工具链，
由 DeepSeek Harness 维护。产出物直接对接 `models/` 与 Android 原生
`AemeathTtsPlugin`（见 `ARCHITECTURE.md` §2）。

| 脚本 | 职责 | 关键产物 |
| :--- | :--- | :--- |
| `autodl_prep_and_asr.py` | 48kHz 原声质检清洗 + FunASR 中文打标 + 专名纠错 | `aemeath.list`、`cleaned/*.wav`、`prep_report.json` |
| `export_and_quantize_onnx.py` | ONNX 导出驱动 + Int8/FP16 混合量化 + 情绪向量固化 + 魔搭清单 | `*.int8.onnx`、`vits_vocoder.fp16.onnx`、`emotions/*.bin`、`manifest.json`、`model_io.json` |
| `prepare_ref_conditioning.py` | 从 4 条情绪参考音预计算 `ref_seq/ssl_content/ref_bert/ref_audio` + 离线 G2P 字表 | `conditioning/**`、`conditioning.json` |
| `requirements-autodl.txt` | AutoDL 依赖清单 | — |

---

## 1. 一键安装（AutoDL 实例）

```bash
cd /root/你的项目目录
pip install -r scripts/requirements-autodl.txt
# torch / torchaudio 请按镜像 CUDA 版本单独装，例如：
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. 任务一 A：`autodl_prep_and_asr.py`

### 2.1 它替你挡掉哪些废片

| 质检项 | 默认阈值 | 说明 |
| :--- | :--- | :--- |
| 时长 | `< 1.8s` 剔除 | 战斗短促喊叫，训练集里只会污染韵律 |
| 文件体积 | `< 1.8s × 采样率 × 2` 字节（48k/16bit 单声道 ≈ 172.8KB，用户口径 180KB） | 时长阈值的二次兜底 |
| 削波 | 满刻度采样点占比 `> 0.5%` 剔除 | 防止爆音被学进声码器 |
| 信噪比 | `< 12dB` 剔除（连续无停顿长句自动跳过该判定） | 见下方「关于 SNR」 |
| 谱域嘶声 | `7.5–11kHz` 相对 `1–4kHz` 能量密度 `> -12dB` 剔除 | 白噪声 / 电流声的真实判据 |
| 静音占比 | `> 85%` 剔除 | 有效语音太少 |
| 峰值电平 | `< -35dBFS` 剔除 | 近乎无声 |
| 重复 take | 包络指纹去重 | 同一句多份拷贝会导致过拟合 |

> **关于 SNR**：`--min-snr-db` 用「静音帧噪声底 vs 语音帧峰值」估计，只有在录音里
> 存在可识别停顿时才成立；连续长句无法估计时该判定自动跳过，改由谱域嘶声比
> `--max-hiss-db` 兜底。这避免了「误杀一口气念完的好素材」。

### 2.2 它替你做哪些清洗

去直流 → 60Hz 高通 → 首尾静音裁剪（保留 50ms 呼吸）→ 重采样到 **32kHz**
（GPT-SoVITS v2 训练采样率）→ 峰值归一化到 **-1dBFS** → 5ms 淡入淡出 →
写 16bit 单声道 PCM WAV。

### 2.3 常用命令

```bash
# 标准全流程（GPU 打标）
python scripts/autodl_prep_and_asr.py \
    --input-dir /root/autodl-tmp/aemeath_raw \
    --output-dir /root/autodl-tmp/aemeath_dataset \
    --device cuda:0

# 先只做质检，摸清素材成色（不下载任何 ASR 模型）
python scripts/autodl_prep_and_asr.py -i raw -o ds --asr-backend none

# 断点续跑 / 四机并行分片
python scripts/autodl_prep_and_asr.py -i raw -o ds --resume
python scripts/autodl_prep_and_asr.py -i raw -o ds_shard0 --shard-index 0 --shard-count 4

# 流水线联调（不下载模型，用占位台词）
python scripts/autodl_prep_and_asr.py -i raw -o ds --asr-backend mock
```

### 2.4 产物格式

`aemeath.list`（GPT-SoVITS v2 / Bert-VITS2 通用）：

```
/root/.../cleaned/aimisi_zh_003_6264182.wav|aemeath|ZH|爱弥斯，你终于回来啦。
```

* 字段：`音频绝对路径|说话人|语种|台词`；
* 文本已做：全半角归一、标点规整、数字中文化、专有名词同音纠错
  （`艾米斯→爱弥斯`、`漂泊着→漂泊者`、`拉海罗→拉海洛`、`星距学院→星炬学院` 等）；
* `prep_report.json` / `rejected_files.csv` 记录每条素材的指标与淘汰原因，可回归审计。

---

## 3. 任务一 B：`export_and_quantize_onnx.py`

### 3.1 防电音铁律（本脚本的核心设计）

| 模块 | 精度 | 实现 | 理由 |
| :--- | :--- | :--- | :--- |
| `t2s_encoder.onnx` | **Int8 动态** | `quantize_dynamic(QUInt8)` | 自回归矩阵小，量化误差不累积，体积降 70% |
| `t2s_fsdec.onnx` | **Int8 动态** | 同上 | 同上 |
| `t2s_sdec.onnx` | **Int8 动态** | 同上 | 同上 |
| `text_encoder.onnx`（Bert-VITS2 文本流） | **Int8 动态** | 同上 | 前端语义对单点误差不敏感 |
| `vits_vocoder.onnx` | **FP16** | `convert_float_to_float16(..., keep_io_types=True)` | 逐样本波形反卷积，Int8 噪声会直接刻进波形 → 金属电流声 |

额外加固：

* **数值敏感算子保 FP32**：`Div/Exp/Log/Pow/Sqrt/Softmax/ReduceMean/Clip/Tanh/...`
  列入 `op_block_list`，只让大矩阵 GEMM/Conv 走半精度。
* **I/O 仍为 Float32**：`keep_io_types=True`，且转换后会复核 I/O 类型，若被
  半精度「穿透」直接判失败。
* **三级转换器兜底**：`onnxconverter_common+blocklist` → `onnxruntime.transformers+blocklist`
  → `onnxconverter_common` 纯 FP16；每一档都用 ORT 真实加载验证，只有能跑通的
  产物才落盘（实测 `onnxconverter_common` 的 blocklist 分支在部分图上会生成非法图）。
* **拓扑重排**：转换器插入 Cast 后不会重排节点，脚本统一做 Kahn 拓扑排序，
  保证 `onnx.checker` 与端侧加载同时通过。
* **量化后校验**：`onnx.checker` + ORT dummy 前向 + 与 FP32 的余弦相似度
  （低于 0.95 会在报告里标红）。

### 3.2 五条常用命令

```bash
# ① 从 GPT-SoVITS 权重导出 FP32 ONNX（调用官方 onnx_export.py，不重造轮子）
python scripts/export_and_quantize_onnx.py export \
    --gpt-sovits-repo /root/GPT-SoVITS \
    --ckpt-dir /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/aemeath \
    --out-dir /root/autodl-tmp/aemeath_pro/fp32

# ② 一条命令出包：量化 + 情绪向量 + 校验 + 魔搭清单 + zip DLC
python scripts/export_and_quantize_onnx.py all \
    --fp32-dir /root/autodl-tmp/aemeath_pro/fp32 \
    --out-dir /root/autodl-tmp/aemeath_pro/onnx \
    --reference-dir /root/autodl-tmp/aemeath_ref \
    --arch "GPT-SoVITS v2" --sample-rate 32000 --bundle

# ③ 已有 FP32 ONNX，只做量化（先看计划再执行）
python scripts/export_and_quantize_onnx.py quantize --fp32-dir fp32 --out-dir onnx --print-plan
python scripts/export_and_quantize_onnx.py quantize --fp32-dir fp32 --out-dir onnx --compare-fp32

# ④ 换了参考音，只重跑情绪向量
python scripts/export_and_quantize_onnx.py emotions --reference-dir refs --out-dir onnx/emotions

# ⑤ 校验 + 生成原生绑定契约 model_io.json
python scripts/export_and_quantize_onnx.py verify --target onnx --compare-fp32 --fp32-dir fp32
```

### 3.3 情绪向量（`.bin`）规范

* 4 条代表性参考原声 → `平静.bin` / `开心.bin` / `温柔.bin` / `调侃.bin`；
* 严格 **256 维 float32 小端**，即每个文件**恰好 1024 字节**（脚本会强制校验）；
* 提取路径：CNHubert（`TencentGameMate/chinese-hubert-base`）mean+std 池化 1536 维
  → 固定种子正交投影到 256 维 → 跨情绪均值去相关 → 单位化；
* 无 GPU / 无 transformers 时自动降级为「4 窗长 × 32 mel 带 × (均值,标准差)」
  谱域嵌入，仍输出 1024 字节，并在 `emotion_report.json` 标注 `dsp_mel_statistics`；
* 两两余弦相似度 > 0.985 会在报告里告警（情绪区分度不足）。

> 端侧因此**无需**运行 CNHubert/Wav2Vec2 前处理模型，手机只读 1KB 向量。

### 3.4 魔搭上传

```bash
python scripts/export_and_quantize_onnx.py manifest \
    --out-dir onnx --repo-id aemeath/gpt-sovits-aemeath-pro-onnx --bundle --upload-cmd
# 打印出的命令示例：
# modelscope upload aemeath/gpt-sovits-aemeath-pro-onnx onnx . --commit-message '...'
```

`manifest.json` 含每个文件的体积与 SHA256、精度、以及 `sampling_params`
（`temperature=0.85 / top_k=15 / top_p=0.90 / repetition_penalty=1.35`），
Android 下载器据此逐文件校验。

### 3.5 子命令一览

| 子命令 | 作用 |
| :--- | :--- |
| `export` | 驱动 GPT-SoVITS 官方 `onnx_export.py` 产出 FP32 ONNX；`--skip-export` 可跳过 |
| `quantize` | Int8 动态量化 + FP16 声码器转换 + 自动校验，写 `quant_report.json` |
| `emotions` | 固化 4 组情绪向量 `.bin`，写 `emotion_report.json` |
| `manifest` | 生成 `manifest.json`（含 SHA256），可 `--bundle` 打 zip DLC |
| `verify` | ORT 加载 / dummy 前向 / 与 FP32 比对，写 `model_io.json` |
| `all` | 串联 ②，一条命令出完整语音包 |

---

## 4. 参考条件包：`prepare_ref_conditioning.py`（GPT-SoVITS v2Pro 实机路线）

真实权重 `Aimisi_v2Pro_e8_*.onnx` 的 `t2s_encoder` 需要 5 路输入：

```
ref_seq(int64) + text_seq(int64) + ref_bert(1024) + text_bert(1024) + ssl_content(768)
```

其中 **4 路只与参考音有关**，可以在云端算一次固化下发；`text_bert` 逐句依赖中文 BERT，
端侧按约定喂零。这样手机端**完全不跑 CNHubert / BERT**，只多读几 MB 条件文件。

### 4.1 三种模式

```bash
# ① 有 GPT-SoVITS 仓库与权重（完整，推荐）
python scripts/prepare_ref_conditioning.py --mode repo \
    --gpt-sovits-repo /root/GPT-SoVITS \
    --s2g /root/models/Aimisi_v2Pro_e8_s232.pth \
    --cnhubert /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base \
    --bert /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large \
    --ref-dir /root/aemeath_ref --ref-text-map /root/aemeath_ref/ref_texts.json \
    --build-g2p --onnx-dir /root/models/onnx --out-dir /root/models/onnx/conditioning

# ② 只有训练日志（复用训练期语义码，最快；缺 ssl_content，端侧只能喂零 → 仅联调用）
python scripts/prepare_ref_conditioning.py --mode logs \
    --semantic-tsv logs/Aimisi_v2Pro/6-name2semantic.tsv \
    --ref-dir refs --onnx-dir onnx --out-dir onnx/conditioning

# ③ 只重建端侧 G2P 字表
python scripts/prepare_ref_conditioning.py --mode g2p \
    --gpt-sovits-repo /root/GPT-SoVITS --out-dir onnx/conditioning
```

### 4.2 产物

```
conditioning/
├── conditioning.json          # vocab_size / eos_token / sample_rate / bert_source / g2p
├── <情绪>/{ref_seq.i64, ssl_content.f32, ref_bert.f32, ref_audio.f32}
└── g2p/{symbols.json, char2phone.json}    # 汉字 → 音素 id（端侧免 jieba/g2pw）
```

* `ref_audio.f32` 必须存在——声码器 `vits.onnx` 的第三个输入就是参考波形本身；
* `ref_seq` 由 `s232.pth` 内的 `ssl_proj + quantizer` 复现，保证与训练分布一致；
* 生成后自带校验：用真实 `t2s_encoder.onnx` 跑一遍，检查输出形状与是否含 NaN/Inf。

### 4.3 端侧对这张图的特殊处理（务必知悉）

`t2s_sdec` 的 `samples` 输出实测为
`Softmax → Div → ScatterElements → Less/Where → Div → Softmax → ArgMax → Cast → Unsqueeze`，
即**导出时图内就做了 argmax**。若原生直接取 `samples`，红线 3 会被绕过。
`GptSoVitsOnnxEngine` 因此只读 `logits`，用 `AemeathSampler` 采样后喂回 `iy`。

`conditioning.json` 里可调、无需改代码的项：

| 字段 | 默认 | 含义 |
| :--- | :--- | :--- |
| `vocab_size` | 1025 | 语义码表大小 |
| `eos_token` | 1024 | 结束符 id |
| `prompt_prefix_tokens` | -1 | `fsdec.y` 前多少 token 属于 prompt；-1=自动（`y.size > ref_seq.size` 时丢前 `ref_seq.size` 个） |

---

## 5. 与端侧的接口

* 原生按 `ARCHITECTURE.md` §5.1 的真实张量名**精确绑定**，缺名即报错（不再依赖模糊别名）；
* `manifest.json.models` 给出 `semantic_encoder` / `semantic_decoder_first` /
  `semantic_decoder_second` / `vocoder` 等角色到文件的映射；
* 情绪 `.bin` 路径固定为 `emotions/<情绪>.bin`；参考条件包固定为 `conditioning/`。

完整契约见 `ARCHITECTURE.md` §2、§4、§5、§6。

---

## 6. 端侧构建辅助：`build_android.ps1`

不是 Python 脚本，是给本机 Windows 用的**一键出 APK** 脚本（vite build → cap sync → gradle assembleDebug）。

```powershell
# 默认：用 D:\tools 下的 JDK17 / Android SDK / Gradle 缓存
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_android.ps1

# 编完直接装到已连接的真机
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_android.ps1 -Install

# 彻底重编
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_android.ps1 -Clean
```

可覆盖的工具链路径：`-JdkHome`、`-AndroidSdk`、`-GradleUserHome`。
脚本会自己检查三件事：JDK 是否存在、SDK 是否存在、`android/` 工程是否存在；任一步失败即以非零退出并打印原因。
产物固定为 `android\app\build\outputs\apk\debug\app-debug.apk`（本机实测 64,960,376 B / 61.95 MB）。

> 注意：该脚本带 UTF-8 BOM——Windows PowerShell 5.1 若不读 BOM 会把中文注释按 ANSI 解析并报
> 「Missing closing '}'」。修改时请保持 BOM 编码。

