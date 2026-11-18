# `android/` — 爱弥斯原生推理桥接（Capacitor 6 · 已编译通过）

> **状态（2026-11-17）**：本目录已不再是「载荷」，而是 `npx cap add android` 生成的真实
> Gradle 工程，8 个 Kotlin 插件已合入并**编译成功**：
>
> ```
> BUILD SUCCESSFUL in 25s
> app-debug.apk  64,960,376 B (61.95 MB)   package com.aemeath.terminal  label 爱弥斯终端
> ```
>
> 原始载荷副本保留在仓库根目录 `android_payload/`（仅作历史参考，**不要再改它**）。

---

## 1. 工程结构

```
android/
├── app/
│   ├── build.gradle                         # 已加 kotlin-android + 4 个依赖 + packagingOptions
│   ├── libs/sherpa-onnx.aar                 # 47.2 MB，来自 sherpa-onnx v2.0.1 release
│   ├── src/main/assets/public/              # ← dist/ 的构建产物（cap sync 自动同步）
│   └── src/main/java/com/aemeath/terminal/
│       ├── MainActivity.java                # registerPlugin(AemeathTtsPlugin.class) 在 super.onCreate 之前
│       ├── AemeathTtsPlugin.kt              # Capacitor 插件门面（6 个 @PluginMethod）
│       ├── AemeathTypes.kt                  # 情绪/后端/采样/挂载结果等数据契约 + 沙盒路径
│       ├── AemeathSampler.kt                # Top-K/Top-P/Temperature/Repetition + 多项式采样（无 argmax）
│       ├── AemeathOrtBackend.kt             # CPU(XNNPACK)/GPU(Vulkan)/NPU(NNAPI) 调度与回退
│       ├── SherpaVitsEngine.kt              # arch == "Bert-VITS2" → Sherpa-ONNX OfflineTts
│       ├── GptSoVitsOnnxEngine.kt           # arch == "GPT-SoVITS v2" → ORT 四阶段串联
│       ├── AemeathModelScopeDownloader.kt   # 魔搭分片下载 + Range 续传 + SHA256
│       └── AemeathAudioPlayer.kt            # AudioTrack 流式播放 + PCM 后处理
├── build.gradle                             # buildscript 加 kotlin-gradle-plugin:1.9.24
├── gradle.properties                        # android.overridePathCheck=true（非 ASCII 路径必需）
├── local.properties                         # sdk.dir=D:\tools\android-sdk
└── gradlew / gradlew.bat                    # Gradle 8.2.1 wrapper
```

## 2. 构建环境（本机实测路径）

| 项 | 值 |
| :--- | :--- |
| JDK | `D:\tools\jdk17\jdk-17.0.21+9`（Temurin 17.0.21） |
| Android SDK | `D:\tools\android-sdk`（platform-tools 37.0.1 / platforms;android-34 / build-tools;34.0.0） |
| Gradle 缓存 | `D:\tools\gradle-home` |
| Gradle 版本 | 8.2.1（wrapper，AGP 8.2.1） |
| Node / Capacitor | `v24.21.0` / Capacitor 6.0.x，CLI 在 `node_modules/@capacitor/cli` |

`local.properties` 已写好 `sdk.dir`；换机器只需改这一行（或用 `ANDROID_HOME`）。

## 3. 构建命令（PowerShell，可直接复制）

> 最省事：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_android.ps1`
> （等价于下面三步，还会自动校验工具链；加 `-Install` 直接 adb 装机，加 `-Clean` 彻底重编）。

```powershell
cd "D:\语音模型项目\android"
$env:JAVA_HOME      = "D:\tools\jdk17\jdk-17.0.21+9"
$env:PATH           = "$env:JAVA_HOME\bin;" + $env:PATH
$env:ANDROID_HOME   = "D:\tools\android-sdk"
$env:GRADLE_USER_HOME = "D:\tools\gradle-home"

.\gradlew.bat assembleDebug --no-daemon --console=plain
```

产物：`android/app/build/outputs/apk/debug/app-debug.apk`（61.95 MB）。

每次都改前端时，先同步 web 资源再打包：

```powershell
cd "D:\语音模型项目"
npx vite build          # → dist/
npx cap sync android    # → android/app/src/main/assets/public/
cd android; .\gradlew.bat assembleDebug --no-daemon --console=plain
```

真机安装与日志：

```powershell
D:\tools\android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
D:\tools\android-sdk\platform-tools\adb.exe logcat -s AemeathTtsPlugin AemeathSoVits AemeathSherpa AemeathOrt AemeathDownloader AemeathAudio
```

## 4. 首轮编译实际踩到并修掉的 5 个坑

| # | 现象 | 根因 | 修法 |
| :--- | :--- | :--- | :--- |
| 1 | `Your project path contains non-ASCII characters` | AGP 8 对工程路径的 ASCII 校验 | `gradle.properties` 加 `android.overridePathCheck=true` |
| 2 | `Unclosed comment` at `GptSoVitsOnnxEngine.kt:731`，连带 `AemeathTtsPlugin.kt` 找不到 `GptSoVitsOnnxEngine` | Kotlin **块注释可嵌套**，KDoc 里写了 `conditioning/<情绪>/*.f32`，`/*` 变成嵌套注释开端 | 把该处写成 `conditioning/<情绪>/ 下的 *.f32|i64`，去掉字面 `/*` |
| 3 | `AemeathTtsPlugin.kt:261` 类型不匹配 | Capacitor `PluginCall.getDouble/getBoolean` 返回 **可空** 类型 | `call.getDouble("speed", 1.0)?.toFloat() ?: 1.0f`、`call.getBoolean(...) == true` |
| 4 | `GptSoVitsOnnxEngine.kt:333` 无 `toLongArray()` | `generated` 是 `ArrayList<Int>`，不是 `Long` 集合 | `LongArray(generated.size) { generated[it].toLong() }` |
| 5 | `Unresolved reference: kind缺少` | 字符串模板 `"$stage 的$kind缺少"` 把中文并进了标识符（Kotlin 标识符允许中文） | `${kind}缺少` |

> 这 5 个问题在「没有 SDK 只做静态审查」阶段全部漏掉了——所以 §8 的「已知限制」里
> 「未编译过」是真风险，不是免责声明。现在已用真实编译闭环。

## 5. 编译产物核验（可复现）

| 项 | 结果 |
| :--- | :--- |
| APK | `64,960,376 B`（61.95 MB），`package com.aemeath.terminal`，`versionName 1.0`，`label 爱弥斯终端` |
| 原生库 | `lib/{arm64-v8a,armeabi-v7a,x86,x86_64}` 下含 `libonnxruntime.so`、`libonnxruntime4j_jni.so`、`libsherpa-onnx-{c,cxx,jni}*.so` |
| 插件类 | dex 内命中：`AemeathTtsPlugin`×49、`GptSoVitsOnnxEngine`×70、`AemeathModelScopeDownloader`×66、`SherpaVitsEngine`×29、`AemeathAudioPlayer`×13、`AemeathOrtBackend`×13、`AemeathTypes`×10、`AemeathSampler`×6 |
| Web 资源 | `assets/public/index.html` 存在且与 `dist/index.html` 同为 328,892 B |

## 6. 模型目录约定
| 场景 | 沙盒路径 |
| :--- | :--- |
| 魔搭下载默认目录 | `context.filesDir/models/aemeath/<modelId>/` |
| 前端传入 `localDir` | `context.filesDir/<localDir>`（如 `models/aemeath-sovits-pro`） |
| 绝对路径 | 仅允许落在 `filesDir` 内，否则自动收敛到沙盒根目录（防路径穿越） |

一个已发布模型目录的内容（`xinxin111x/Aimisi` 的实际布局）：

```
models/aemeath-sovits-pro/
├── manifest.json            # 26 文件 + SHA256 + total_bytes=1222151331 + sampling_params
├── t2s_encoder.onnx         # 10.9 MB 
├── t2s_fsdec.onnx           # 293.3 MB
├── t2s_sdec.onnx            # 293.3 MB（图内 samples 是 argmax，运行时只读 logits 自行采样）
├── vits.onnx                # 250.6 MB（声码器）
├── bert/                    # ★ 有它才出声（缺了 → 6 码 EOS → 近乎静音）
│   ├── model.int8.onnx      # 311.2 MB，中文 roberta-wwm-ext-large 的 int8 动态量化
│   ├── vocab.txt
│   └── tokenizer.json
└── conditioning/            # ★ 参考条件包（音色/韵律来源）
    ├── conditioning.json
    ├── 平静/{ref_seq.i64, ref_bert.f32, ssl_content.f32, ref_audio.f32}
    ├── 开心/  温柔/  调侃/
    └── g2p/{symbols.json, char2phone.json}
```

> 旧的 `emotions/*.bin`（256×float32=1024 B）与 `tokens.txt`/`lexicon.txt` 只属于
> Bert-VITS2 快路径；GPT-SoVITS Pro 包**不含**这些文件，靠 `conditioning/` + `bert/`。

## 7. 插件方法契约（对齐 `ARCHITECTURE.md` §2）

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

> 注意：`pullModelFromModelScope` 失败时是 **`resolve({success:false, message})`**，不是
> `reject`。前端必须判 `res.success === true` 再 `installed = true`（见 `PROGRESS.md` §7 P0-1）。

## 8. 后端调度行为

| `backend` | 首选 | 实际降级路径 |
| :--- | :--- | :--- |
| `cpu` | `setIntraOpNumThreads(4)` + XNNPACK | XNNPACK 不存在 → 纯 ARM NEON 4 线程 |
| `gpu` | Vulkan EP（反射探测） | 官方 ORT Android 包不含 Vulkan → CPU/XNNPACK，`notes` 说明 |
| `npu` | `addNnapi()` | 某算子不支持导致建会话失败 → 自动剔除 NNAPI 用 CPU 重建 |

`accelerationActive` 返回**声码器实际生效**的加速器；T2S 与声码器后端不同时在 `notes` 注明。

## 9. 已知限制（真机验证前必读）

1. **Android 端 BERT 推理尚未实现**：`text_bert` 需要 WordPiece 分词 + `bert/model.int8.onnx`
   前向 + 与音素的 `word2ph` 对齐。当前 `GptSoVitsOnnxEngine` 会尝试从 `conditioning/` 读
   `ref_bert`，但 `text_bert` 还没有真实实现 → Pro 模型出声前必须先补这块（`PROGRESS.md` §3.7 硬约束 2）。
2. **未在真机跑过推理**：本机无 Android 设备，只验证了编译与打包；首次真机请先
   `adb logcat` 看 `mountModel` 的 `notes` 是否报缺张量。
3. **Vulkan EP**：`onnxruntime-android:1.18.0` 无 Vulkan，`backend='gpu'` 会安全回退 CPU。
4. **Lite 权重是占位**：`models/aemeath-vits/model.int8.onnx` 仅 65,589 B，Sherpa 路径出不了声。
5. **Release 未签名**：`assembleDebug` 通过；AAB 发布需自备 keystore 并配 `signingConfigs`。

## 10. 图片资源约定（头像 / 壁纸 / 看板娘）

App 内出现的三张图，文件名是**硬编码**在 `index.html` 内联脚本里的，必须保持原名进包：

| 文件 | 用途 | 体积 |
| :--- | :--- | :--- |
| `1794568840819.jpg` | 主头像（正圆形） | 2,913,783 B |
| `1794568844420.jpg` | 预备头像（正圆形） | 2,363,468 B |
| `kanban_aemeath.png` | 桌面看板娘立绘 | 402,130 B |

它们必须躺在 **`assets/public/` 根目录**（即 `dist/` 根目录），不能只留 Vite 的
`assets/<name>-<hash>.jpg` 哈希副本——因为内联 JS 用的是原始文件名，哈希副本它找不到，
表现为「首屏头像正常，一切换头像 / 换壁纸 / 看板娘就破图」。

这条约束由 `vite.config.ts` 的 `aemeath-raw-images` 插件保证（复制原图 + 改回原始名 +
删哈希副本 + 缺图直接构建失败），**不需要手工往 `android/app/src/main/assets/public/` 拷图**，
`npx cap sync android` 会自动同步。

排查破图时按顺序看：

```powershell
# 1) 产物里到底有什么
Get-ChildItem dist -Recurse -File | Select-Object Name,Length

# 2) 页面引用了什么、是否存在（任一 MISSING 就是破图根因）
Select-String -Path dist\index.html -Pattern "1794568840819|1794568844420|kanban_aemeath"
```

