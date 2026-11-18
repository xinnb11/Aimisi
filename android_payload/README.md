> 🗄 **本目录已归档（2026-11-17）**：8 个 Kotlin 已合入仓库根目录的 `android/`（真 Gradle 工程），
> 并已 `assembleDebug` 编译通过（APK 61.95 MB）。本目录仅作历史参考，**不要再修改它**；
> 现行说明见 `../android/README.md`，构建记录见 `../PROGRESS.md` §10。

# `android/` — 爱弥斯原生推理桥接（Capacitor Android）

本目录是 **原生落地载荷（source drop-in payload）**：把 Kotlin 插件源码放进
`npx cap add android` 生成的 Android 工程即可。它不包含完整 Android 工程，
以免与 Capacitor 生成的 Gradle / Manifest 冲突。

```
android/
└── app/
    ├── build.gradle.snippet                  # 依赖与打包片段（手工合并到 app/build.gradle）
    └── src/main/java/com/aemeath/terminal/
        ├── AemeathTtsPlugin.kt               # Capacitor 插件门面（契约实现）
        ├── AemeathTypes.kt                   # 情绪/后端/采样/挂载结果等数据契约 + 沙盒路径
        ├── AemeathSampler.kt                 # Top-K/Top-P/Temperature/Repetition + 多项式采样
        ├── AemeathOrtBackend.kt              # CPU(XNNPACK)/GPU(Vulkan)/NPU(NNAPI) 调度与回退
        ├── SherpaVitsEngine.kt               # arch == "Bert-VITS2" → Sherpa-ONNX OfflineTts
        ├── GptSoVitsOnnxEngine.kt            # arch == "GPT-SoVITS v2" → ORT 四阶段串联
        ├── AemeathModelScopeDownloader.kt    # 魔搭分片下载 + Range 续传 + SHA256
        └── AemeathAudioPlayer.kt             # AudioTrack 流式播放 + PCM 后处理
```

## 1. 接入步骤

`npx cap add android` 要求 `android/` 目录尚不存在，而本仓库的 `android/` 就是
「原生落地载荷」。因此按下面的顺序合并（一次即可，合并后本目录内容已在 git 里）：

```bash
# 1) 给载荷让位
mv android android_payload          # Windows: ren android android_payload

# 2) 生成真正的 Android 工程（会读取根目录 capacitor.config.json）
npm install
npx cap add android

# 3) 把插件源码与 Gradle 片段并入生成工程（包路径完全一致，直接覆盖）
cp android_payload/app/src/main/java/com/aemeath/terminal/*.kt \
   android/app/src/main/java/com/aemeath/terminal/
#    然后按第 3 节合并 android_payload/app/build.gradle.snippet

# 4) 同步并打开
npx cap sync android
npx cap open android

# 5) 清理载荷副本（可选，源码已在 android/app/... 内）
rm -rf android_payload
```

> 生成工程的默认包名就是 `com.aemeath.terminal`，与载荷路径一一对应，
> 合并时不需要改任何 `package` 声明。

## 2. 注册插件（Capacitor 6 必须显式注册）

`MainActivity.kt`：

```kotlin
package com.aemeath.terminal

import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        registerPlugin(AemeathTtsPlugin::class.java)
        super.onCreate(savedInstanceState)
    }
}
```

> 顺序很重要：`registerPlugin` 必须在 `super.onCreate` **之前**调用，
> 否则 `window.Capacitor.Plugins.AemeathTtsPlugin` 不存在，前端会静默降级到
> Web Worker（这不会报错，只是没走原生推理）。

## 3. Gradle 依赖

把 `app/build.gradle.snippet` 的内容合并进 `android/app/build.gradle`，核心是：

| 依赖 | 作用 |
| :--- | :--- |
| `com.microsoft.onnxruntime:onnxruntime-android:1.18.0` | ONNX Runtime Mobile（含 NNAPI / XNNPACK） |
| `sherpa-onnx AAR`（放 `app/libs/`） | Sherpa-ONNX C++ JNI（`OfflineTts`） |
| `com.squareup.okhttp3:okhttp:4.12.0` | 魔搭分片下载（分块读写 + 超时控制） |
| `org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0` | 后台推理协程 |

Sherpa-ONNX 的 AAR 需要从官方 release 页下载后放 `android/app/libs/`：

```
https://github.com/k2-fsa/sherpa-onnx/releases   →   sherpa-onnx-vX.Y.Z.aar
```

`abis` 建议同时保留 `arm64-v8a` 与 `armeabi-v7a`；`x86_64` 仅在模拟器调试时加。

## 4. 模型目录约定

| 场景 | 沙盒路径 |
| :--- | :--- |
| 魔搭下载默认目录 | `context.filesDir/models/aemeath/<modelId>/` |
| 前端传入 `localDir` | `context.filesDir/<localDir>`（如 `models/aemeath-vits-lite`） |
| 绝对路径 | 仅允许落在 `filesDir` 内，否则自动收敛到沙盒根目录（防路径穿越） |

一个已量化模型目录的内容：

```
models/aemeath-sovits-pro/
├── manifest.json            # 量化脚本生成，含 sample_rate / sampling_params / SHA256
├── t2s_encoder.int8.onnx    # Int8 动态量化（前端语义）
├── t2s_fsdec.int8.onnx      # Int8 动态量化
├── t2s_sdec.int8.onnx       # Int8 动态量化（自回归，只读 logits 采样）
├── vits.fp16.onnx           # FP16 声码器（I/O 仍为 Float32）
├── conditioning/            # ★ 参考条件包（GPT-SoVITS 必需，见 ARCHITECTURE §5.2）
│   ├── conditioning.json
│   ├── 平静/{ref_seq.i64, ssl_content.f32, ref_bert.f32, ref_audio.f32}
│   ├── 开心/  温柔/  调侃/
│   └── g2p/{symbols.json, char2phone.json}
├── tokens.txt               # 词表（Bert-VITS2 用）
├── lexicon.txt              # 中文音素词典（Bert-VITS2 用）
└── emotions/                # 256 维情绪向量（Bert-VITS2 快路径用）
    ├── 平静.bin             # 256 × float32 LE = 1024 字节
    ├── 开心.bin / 温柔.bin / 调侃.bin
```

> **GPT-SoVITS 与 Bert-VITS2 的差异**：前者靠 `conditioning/` 里的参考音特征
> （`ssl_content` / `ref_seq` / `ref_audio`）驱动音色与韵律，`emotions/*.bin` 只是
> 前端展示与快路径；后者靠 `emotions/*.bin` + `tokens.txt` + `lexicon.txt`。
> 两者都放在同一模型目录下不冲突，引擎按 `arch` 各取所需。

## 5. 插件方法契约（对齐 `ARCHITECTURE.md` §2）

| 方法 | 入参要点 | 返回 |
| :--- | :--- | :--- |
| `pullModelFromModelScope` | `repoId, modelId, arch, localDir`（可选 `revision, token`） | `success, modelId, savedDir, totalBytes, hashVerified, message` |
| `mountModel` | `modelId, arch, engine, backend, modelDir, emotionFiles?, samplingConfig?` | `success, engine, sampleRate, channels, accelerationActive, memoryFootprintMb, notes` |
| `synthesizeSpeech` | `id, text, emotion, speed, volume, playImmediately` | `id, durationMs, sampleRate, samplesCount, played, audioBase64?` |
| `stopPlayback` | — | `stopped` |
| `unmountModel` | — | `success` |
| `getEngineStatus` | — | 装载状态 / 实际加速器 / 当前采样参数 |

持续事件：`modelDownloadProgress`（`progress, stage, downloadedBytes, totalBytes, speed`）、
`speechPlaybackStart`（`id, emotion`）。

## 6. 后端调度行为

| `backend` | 首选 | 实际降级路径 |
| :--- | :--- | :--- |
| `cpu` | `setIntraOpNumThreads(4)` + XNNPACK | XNNPACK 不存在 → 纯 ARM NEON 4 线程 |
| `gpu` | Vulkan EP（反射探测） | 当前 ORT Android 构建多未暴露 Vulkan → CPU/XNNPACK，并在 `notes` 说明 |
| `npu` | `addNnapi()` | NNAPI 不支持某算子导致建会话失败 → 自动剔除 NNAPI 用 CPU 重建 |

`accelerationActive` 返回的是**声码器实际生效**的加速器（大矩阵才是 GPU/NPU 的
真正受益者）；若 T2S 与声码器后端不同，会在 `notes` 里注明。

## 7. 调试

```bash
adb logcat -s AemeathTtsPlugin AemeathSoVits AemeathSherpa AemeathOrt AemeathDownloader AemeathAudio
```

前端页面里随时可查：

```js
const status = await window.Capacitor.Plugins.AemeathTtsPlugin.getEngineStatus();
console.log(status.accelerationActive, status.sampleRate, status.topK);
```

## 8. 已知限制（务必在真机验证）

1. **未在本仓库执行过编译**：本环境没有 Android SDK / Gradle，Kotlin 代码是
   静态审查产物，首次构建请在 Android Studio 中同步并修正依赖版本差异。
2. **Vulkan EP**：`onnxruntime-android` 官方包不包含 Vulkan EP。若必须 GPU 加速，
   需要自编译带 Vulkan 的 ORT 并在 `AemeathOrtBackend` 中暴露 `addVulkan`；
   在此之前 `backend='gpu'` 会安全回退 CPU 并在 `notes` 里说明。
3. **GPT-SoVITS 张量绑定**：不同 GPT-SoVITS 版本的导出张量名差异较大，
   端侧采用「`model_io.json` 优先 + 语义别名兜底」策略；若某版本无法绑定，
   `mountModel` 的 `notes` 会明确指出缺失的张量名，请据此在导出侧对齐。
4. **Sherpa-ONNX 的 provider**：Android 上一般只有 `cpu`；`npu/gpu` 请求会被
   构建期捕获并回退，`accelerationActive` 反映真实结果。
