package com.aemeath.terminal

import android.util.Base64
import android.util.Log
import com.getcapacitor.JSArray
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * AemeathTtsPlugin.kt — Capacitor 原生推理桥接插件（`window.Capacitor.Plugins.AemeathTtsPlugin`）
 *
 * 严格实现 `ARCHITECTURE.md` §2 的方法契约：
 *
 * | 方法 | 作用 |
 * | :--- | :--- |
 * | `pullModelFromModelScope` | 从魔搭分片下载模型至 `filesDir/models/aemeath/`，实时广播进度 |
 * | `mountModel` | 按 `arch` 动态路由 Sherpa-ONNX / ONNX Runtime，并激活 `backend` 加速 |
 * | `synthesizeSpeech` | 离线推理 + AudioTrack 零延迟播放，或回传 WAV Base64 |
 * | `stopPlayback` | 立即打断当前播放 |
 * | `unmountModel` | 释放全部原生会话与内存 |
 * | `getEngineStatus` | 供前端诊断当前实际加速器与装载状态 |
 * | `diagnose` | 端侧自检：装载/合成失败原因 + 模型目录清单 + BERT 是否就绪 |
 *
 * 所有耗时操作都在 `Dispatchers.Default` 协程中执行，绝不阻塞 Capacitor
 * 主线程（否则前端 120fps 弹簧透镜动画会掉帧）。
 */
@CapacitorPlugin(name = "AemeathTtsPlugin")
class AemeathTtsPlugin : Plugin() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val audioPlayer = AemeathAudioPlayer()
    private val stateLock = Any()

    /**
     * 引擎互斥锁：ORT 的 Session 不是线程安全的，前端「预合成下一句」与
     * 「当前句合成」一旦并发，会在同一份 KV cache 上反复分配 8MB 级张量，
     * 直接把 256MB 的 Java 堆打爆（真机实测 OOM: 8847379 byte allocation）。
     * 所有 mount / synthesize 都必须串行通过这把锁。
     */
    private val engineMutex = Mutex()

    @Volatile
    private var lastBackend: String? = null

    @Volatile
    private var engine: AemeathTtsEngine? = null

    @Volatile
    private var mountedModelId: String? = null

    @Volatile
    private var mountedArch: String? = null

    @Volatile
    private var mountedEngineId: String? = null

    @Volatile
    private var activeAccelerator: String = "未装载"

    @Volatile
    private var sampleRate: Int = 0

    @Volatile
    /** 最近一次装载使用的后端原始值（npu/gpu/cpu），同时用于「装载复用」判据与合成缓存键。 */
    private var samplingConfig: SamplingConfig = SamplingConfig()

    /**
     * 合成结果缓存（m07626）：同一句 + 同情绪 + 同模型 + 同采样参数只算一次，
     * 第二次点击播放直接吐 PCM。落盘在 `filesDir/cache/tts`，默认上限 256 MB。
     */
    private val ttsCache by lazy { AemeathTtsCache(context) }

    @Volatile
    private var lastModelDir: String = ""

    @Volatile
    private var lastMountError: String? = null

    @Volatile
    private var lastMountNotes: String = ""

    @Volatile
    private var lastSynthError: String? = null

    @Volatile
    private var lastSynthInfo: String = ""

    // ------------------------------------------------------------------ //
    // 1. 模型下载（魔搭分片）
    // ------------------------------------------------------------------ //
    @PluginMethod
    fun pullModelFromModelScope(call: PluginCall) {
        val repoId = call.getString("repoId")
        val modelId = call.getString("modelId")
        if (repoId.isNullOrBlank() || modelId.isNullOrBlank()) {
            call.reject("repoId 与 modelId 不能为空")
            return
        }
        val arch = call.getString("arch", "Bert-VITS2") ?: "Bert-VITS2"
        val revision = call.getString("revision", "master") ?: "master"
        val token = call.getString("token")
        val localDir = call.getString("localDir", "models/aemeath/$modelId")
            ?: "models/aemeath/$modelId"

        val targetDir = ModelPaths.resolve(context, localDir)
        Log.i(TAG, "开始拉取魔搭模型：$repoId -> ${targetDir.absolutePath}（arch=$arch）")

        val downloader = AemeathModelScopeDownloader(
            token = token?.takeIf { it.isNotBlank() },
            concurrency = DOWNLOAD_CONCURRENCY,
            timeoutSeconds = DOWNLOAD_TIMEOUT_SECONDS,
        )

        scope.launch {
            val result = downloader.download(
                repoId = repoId,
                revision = revision,
                destDir = targetDir,
                modelId = modelId,
            ) { progress -> emitDownloadProgress(progress) }

            val payload = JSObject().apply {
                put("success", result.success)
                put("modelId", result.modelId)
                put("savedDir", result.savedDir)
                put("totalBytes", result.totalBytes)
                put("hashVerified", result.hashVerified)
                put("message", result.message)
                put("arch", arch)
            }
            resolveOnUiThread(call, payload)
        }
    }

    private fun emitDownloadProgress(progress: DownloadProgress) {
        val payload = JSObject().apply {
            put("modelId", progress.modelId)
            put("progress", progress.progress)
            put("stage", progress.stage)
            put("downloadedBytes", progress.downloadedBytes)
            put("totalBytes", progress.totalBytes)
            put("speed", progress.speed)
        }
        val activity = activity
        if (activity != null) {
            activity.runOnUiThread { notifyListeners("modelDownloadProgress", payload) }
        } else {
            notifyListeners("modelDownloadProgress", payload)
        }
    }

    // ------------------------------------------------------------------ //
    // 2. 模型挂载（双引擎动态路由 + 硬件后端 + CPU 自动回退）
    // ------------------------------------------------------------------ //
    @PluginMethod
    fun mountModel(call: PluginCall) {
        val modelId = call.getString("modelId")
        val arch = call.getString("arch")
        if (modelId.isNullOrBlank() || arch.isNullOrBlank()) {
            call.reject("modelId 与 arch 不能为空")
            return
        }
        val engineName = call.getString("engine", "ONNX Runtime") ?: "ONNX Runtime"
        val backend = ComputeBackend.fromRaw(call.getString("backend", "cpu"))
        val modelDir = ModelPaths.resolve(context, call.getString("modelDir", "models/$modelId"))
        val emotionFiles = parseEmotionFiles(call)
        val config = EngineMountConfig(
            modelId = modelId,
            arch = arch,
            engineName = engineName,
            backend = backend,
            modelDir = modelDir,
            emotionFiles = emotionFiles,
            sampling = parseSamplingConfig(call),
        )

        Log.i(
            TAG,
            "mountModel: id=$modelId arch=$arch engine=$engineName backend=${backend.raw} " +
                "dir=${modelDir.absolutePath}"
        )
        lastModelDir = modelDir.absolutePath

        scope.launch {
            engineMutex.withLock {
                try {
                    // 同一模型 / 同一后端已经在内存里：直接复用，绝不重复建会话。
                    // 每次重建都会再握一份 1.17GB 权重并新开四个 Session，堆很快见底。
                    if (engine != null && mountedModelId == modelId && mountedArch == arch &&
                        lastBackend == backend.raw && modelDir.absolutePath == lastModelDir
                    ) {
                        val reuse = JSObject().apply {
                            put("success", true)
                            put("modelId", modelId)
                            put("engine", mountedEngineId ?: "")
                            put("sampleRate", sampleRate)
                            put("channels", 1)
                            put("accelerationActive", activeAccelerator)
                            put("memoryFootprintMb", 0)
                            put("backend", backend.raw)
                            put("notes", "复用已装载引擎（未重复建会话）")
                        }
                        resolveOnUiThread(call, reuse)
                        return@withLock
                    }

                    val selected = selectEngine(arch, engineName)
                    synchronized(stateLock) {
                        engine?.close()
                        engine = selected
                    }

                    // 加速器不可用时自动回退 CPU：NNAPI/Vulkan 在某些机型上建会话会直接抛错，
                    // 这里让「能出声」优先于「用上 NPU」，并把回退原因写进 notes 回传前端。
                    var fallbackNote: String? = null
                    var result = try {
                        selected.mount(config)
                    } catch (acceleratorError: Throwable) {
                        if (config.backend == ComputeBackend.CPU) throw acceleratorError
                        Log.w(TAG, "加速器 ${config.backend.raw} 失败，回退 CPU：${acceleratorError.message}")
                        try {
                            val cpuResult = selected.mount(config.copy(backend = ComputeBackend.CPU))
                            fallbackNote = "已自动回退 CPU（${config.backend.raw} 不可用：${acceleratorError.message}）"
                            cpuResult
                        } catch (cpuError: Throwable) {
                            throw AemeathEngineException(
                                "${acceleratorError.message}；CPU 回退同样失败：${cpuError.message}"
                            )
                        }
                    }

                    // NNAPI/Vulkan 常见陷阱：会话能建、算子却不支持（真机 8 Elite 上报
                    // ANEURALNETWORKS_OP_FAILED）。建完会话先做一次极短合成验证，
                    // 失败就静默重建 CPU 会话，保证「一定能出声」，并把原因写进 notes。
                    if (config.backend != ComputeBackend.CPU) {
                        val warmup = runCatching {
                            selected.synthesize("验证", Emotion.CALM, 1.0f, 1.0f)
                        }
                        if (warmup.isFailure) {
                            val reason = warmup.exceptionOrNull()?.message ?: "未知错误"
                            Log.w(TAG, "加速器 ${config.backend.raw} 推理不支持该模型算子，回退 CPU：$reason")
                            runCatching { selected.close() }
                            result = selected.mount(config.copy(backend = ComputeBackend.CPU))
                            fallbackNote = "已自动回退 CPU（${config.backend.raw} 推理不支持该模型算子：" +
                                reason.substringBefore('\n').take(120) + "）"
                        }
                    }

                    synchronized(stateLock) {
                        mountedModelId = result.modelId
                        mountedArch = arch
                        mountedEngineId = result.engine
                        activeAccelerator = result.accelerationActive
                        sampleRate = result.sampleRate
                        samplingConfig = config.sampling
                        lastBackend = backend.raw
                        lastMountError = null
                        lastMountNotes = (result.notes + listOfNotNull(fallbackNote)).joinToString("；")
                    }

                    val payload = JSObject().apply {
                        put("success", result.success)
                        put("modelId", result.modelId)
                        put("engine", result.engine)
                        put("sampleRate", result.sampleRate)
                        put("channels", result.channels)
                        put("accelerationActive", result.accelerationActive)
                        put("memoryFootprintMb", result.memoryFootprintMb)
                        put("backend", backend.raw)
                        put("notes", result.notes.joinToString("；"))
                    }
                    resolveOnUiThread(call, payload)
                } catch (error: Throwable) {
                    Log.e(TAG, "mountModel 失败：${error.message}", error)
                    synchronized(stateLock) {
                        lastMountError = error.message ?: error.javaClass.simpleName
                        runCatching { engine?.close() }
                        engine = null
                        mountedModelId = null
                        mountedArch = null
                        mountedEngineId = null
                        activeAccelerator = "未装载"
                    }
                    rejectOnUiThread(call, "模型装载失败：${error.message}")
                }
            }
        }
    }

    /** 双引擎动态路由：`Bert-VITS2` 走 Sherpa-ONNX，`GPT-SoVITS v2` / IndexTTS2 走 ONNX Runtime。 */
    private fun selectEngine(arch: String, engineName: String): AemeathTtsEngine {
        val normalizedArch = arch.trim().lowercase()
        val normalizedEngine = engineName.trim().lowercase()
        return when {
            normalizedArch == "bert-vits2" ||
                normalizedArch.contains("bert_vits") ||
                normalizedEngine.contains("sherpa") -> SherpaVitsEngine()

            normalizedArch.contains("gpt-sovits") ||
                normalizedArch.contains("index-tts") ||
                normalizedArch.contains("indextts") ||
                normalizedEngine.contains("onnx runtime") -> GptSoVitsOnnxEngine()

            else -> {
                Log.w(TAG, "未知架构 $arch，默认使用 ONNX Runtime 引擎")
                GptSoVitsOnnxEngine()
            }
        }
    }

    private fun parseEmotionFiles(call: PluginCall): Map<Emotion, String> {
        val result = HashMap<Emotion, String>()
        val obj = call.getObject("emotionFiles") ?: return result
        for (emotion in Emotion.ALL) {
            val value = obj.optString(emotion.key, "")
            if (value.isNotBlank()) result[emotion] = value
        }
        return result
    }

    private fun parseSamplingConfig(call: PluginCall): SamplingConfig {
        val obj = call.getObject("samplingConfig") ?: return SamplingConfig()
        return try {
            SamplingConfig(
                temperature = obj.optDouble("temperature", 0.85).toFloat(),
                topK = obj.optInt("topK", DEFAULT_TOP_K),
                topP = obj.optDouble("topP", 0.90).toFloat(),
                repetitionPenalty = obj.optDouble("repetitionPenalty", 1.35).toFloat(),
                stopToken = obj.optInt("stopToken", 1024),
                maxSemanticLength = obj.optInt("maxSemanticLength", 512),
            ).sanitized()
        } catch (error: Throwable) {
            Log.w(TAG, "samplingConfig 解析失败，使用默认值：${error.message}")
            SamplingConfig()
        }
    }

    /**
     * 采样参数签名（m07626）：缓存的音频依赖 temperature/topK/topP/惩罚/步数上限，
     * 任何一项改动都应该让旧缓存失效，所以把它们拼进缓存键。
     */
    private fun samplingSignature(): String = buildString {
        append("T").append("%.3f".format(samplingConfig.temperature))
        append(",K").append(samplingConfig.topK)
        append(",P").append("%.3f".format(samplingConfig.topP))
        append(",R").append("%.3f".format(samplingConfig.repetitionPenalty))
        append(",E").append(samplingConfig.stopToken)
        append(",M").append(samplingConfig.maxSemanticLength)
        append(",B").append(lastBackend)
    }

    // ------------------------------------------------------------------ //
    // 3. 语音合成
    // ------------------------------------------------------------------ //
    @PluginMethod
    fun synthesizeSpeech(call: PluginCall) {
        val text = call.getString("text")
        if (text.isNullOrBlank()) {
            call.reject("text 不能为空")
            return
        }
        val taskId = call.getString("id") ?: "tts_${System.currentTimeMillis()}"
        val emotion = Emotion.fromKey(call.getString("emotion", "温柔"))
        val speed = call.getDouble("speed", 1.0)?.toFloat() ?: 1.0f
        val volume = call.getDouble("volume", 1.0)?.toFloat() ?: 1.0f
        val playImmediately = call.getBoolean("playImmediately", true) == true

        val currentEngine = engine
        if (currentEngine == null) {
            call.reject("尚未 mountModel，无法合成")
            return
        }

        scope.launch {
            try {
                // ① 先看缓存：同文本 + 同情绪 + 同模型 + 同后端 + 同采样参数 → 直接复用 PCM。
                //    自回归一句要 15~25 s，而读一次缓存只要几毫秒，重复播放完全没必要重算。
                val cache = ttsCache
                val cacheKey = runCatching {
                    cache.keyOf(text, emotion.key, mountedModelId, mountedArch, samplingSignature())
                }.getOrNull()
                if (cache != null && cacheKey != null) {
                    val hit = withContext(Dispatchers.IO) { cache.load(cacheKey) }
                    if (hit != null && hit.samples.isNotEmpty()) {
                        val cachedPayload = JSObject().apply {
                            put("success", true)
                            put("id", taskId)
                            put("sampleRate", hit.sampleRate)
                            put("samplesCount", hit.samples.size)
                            put("durationMs", hit.durationMs)
                            put("engine", mountedEngineId ?: currentEngine.engineId)
                            put("accelerationActive", "cache")
                            put("cached", true)
                        }
                        if (playImmediately) {
                            val duration = audioPlayer.playAsync(hit.samples, hit.sampleRate, 1.0f)
                            cachedPayload.put("durationMs", duration)
                            cachedPayload.put("played", true)
                            cachedPayload.put("queued", audioPlayer.pendingCount())
                        } else {
                            cachedPayload.put("audioBase64", encodeWavBase64(hit.samples, hit.sampleRate))
                            cachedPayload.put("played", false)
                        }
                        lastSynthError = null
                        lastSynthInfo = "cache-hit text=${text.take(24)} emotion=${emotion.key} " +
                            "samples=${hit.samples.size} rate=${hit.sampleRate}"
                        Log.i(TAG, "合成缓存命中：${hit.durationMs} ms 音频（跳过自回归）")
                        resolveOnUiThread(call, cachedPayload)
                        return@launch
                    }
                }

                // 单句内只跑一次声码器；「边播边算」交给播放队列在句与句之间做（m05844 第 1 条）。
                // 与 mountModel 共享同一把锁：合成期间绝不并发合成、绝不并发重建会话
                val output = engineMutex.withLock {
                    val active = engine ?: throw AemeathEngineException("尚未 mountModel，无法合成")
                    withContext(Dispatchers.Default) {
                        active.synthesize(text, emotion, speed, volume)
                    }
                }
                if (output.samples.isEmpty()) {
                    lastSynthError = "合成结果为空"
                    rejectOnUiThread(call, "合成结果为空")
                    return@launch
                }
                // ② 落盘缓存（失败只记日志，绝不影响本次播放）
                if (cache != null && cacheKey != null) {
                    val stored = withContext(Dispatchers.IO) {
                        cache.store(cacheKey, output.samples, output.sampleRate)
                    }
                    if (!stored) Log.w(TAG, "合成缓存写入失败（忽略）")
                }

                val payload = JSObject().apply {
                    put("success", true)
                    put("id", taskId)
                    put("sampleRate", output.sampleRate)
                    put("samplesCount", output.samples.size)
                    put("durationMs", output.durationMs)
                    put("engine", mountedEngineId ?: currentEngine.engineId)
                    put("accelerationActive", activeAccelerator)
                    put("cached", false)
                }

                if (playImmediately) {
                    // 异步播放：PCM 入队后立刻 resolve，前端可以马上发起下一句合成，
                    // CPU 算下一句与 AudioTrack 播上一句并行（m05844 第 1 条）。
                    val duration = audioPlayer.playAsync(output.samples, output.sampleRate, 1.0f)
                    payload.put("durationMs", duration)
                    payload.put("played", true)
                    payload.put("queued", audioPlayer.pendingCount())
                } else {
                    payload.put("audioBase64", encodeWavBase64(output.samples, output.sampleRate))
                    payload.put("played", false)
                }
                lastSynthError = null
                lastSynthInfo = "text=${text.take(24)} emotion=${emotion.key} " +
                    "samples=${output.samples.size} rate=${output.sampleRate} " +
                    "durationMs=${output.durationMs}"
                notifyListeners("speechPlaybackStart", JSObject().apply {
                    put("id", taskId)
                    put("emotion", emotion.key)
                })
                resolveOnUiThread(call, payload)
            } catch (error: Throwable) {
                Log.e(TAG, "synthesizeSpeech 失败：${error.message}", error)
                lastSynthError = error.message ?: error.javaClass.simpleName
                rejectOnUiThread(call, "语音合成失败：${error.message}")
            }
        }
    }

    @PluginMethod
    fun stopPlayback(call: PluginCall) {
        val stopped = audioPlayer.stop()
        call.resolve(JSObject().apply { put("stopped", stopped) })
    }

    /**
     * 查询合成缓存占用（m07626）。前端设置页可展示「已缓存 N 句 / X MB」。
     */
    @PluginMethod
    fun getTtsCacheInfo(call: PluginCall) {
        scope.launch {
            val stats = withContext(Dispatchers.IO) { ttsCache.stats() }
            resolveOnUiThread(call, JSObject().apply {
                put("entries", stats[0])
                put("bytes", stats[1])
                put("limitBytes", AemeathTtsCache.DEFAULT_LIMIT_BYTES)
                put("version", AemeathTtsCache.CACHE_VERSION)
            })
        }
    }

    /**
     * 清空合成缓存（m07626）。返回删掉的条目数与释放字节数。
     */
    @PluginMethod
    fun clearTtsCache(call: PluginCall) {
        scope.launch {
            val result = withContext(Dispatchers.IO) { ttsCache.clear() }
            Log.i(TAG, "合成缓存已清空：${result.first} 条，释放 ${result.second / 1024} KB")
            resolveOnUiThread(call, JSObject().apply {
                put("removed", result.first)
                put("bytesFreed", result.second)
            })
        }
    }

    // ------------------------------------------------------------------ //
    // 4. 卸载与诊断
    // ------------------------------------------------------------------ //
    @PluginMethod
    fun unmountModel(call: PluginCall) {
        scope.launch {
            audioPlayer.stop()
            synchronized(stateLock) {
                runCatching { engine?.close() }
                engine = null
                mountedModelId = null
                mountedArch = null
                mountedEngineId = null
                activeAccelerator = "未装载"
                sampleRate = 0
            }
            resolveOnUiThread(call, JSObject().apply { put("success", true) })
        }
    }

    @PluginMethod
    fun getEngineStatus(call: PluginCall) {
        val payload = JSObject().apply {
            put("mounted", engine != null)
            put("modelId", mountedModelId ?: "")
            put("arch", mountedArch ?: "")
            put("engine", mountedEngineId ?: "")
            put("accelerationActive", activeAccelerator)
            put("sampleRate", sampleRate)
            put("isPlaying", audioPlayer.isPlaying())
            put("temperature", samplingConfig.temperature.toDouble())
            put("topK", samplingConfig.topK)
            put("topP", samplingConfig.topP.toDouble())
            put("repetitionPenalty", samplingConfig.repetitionPenalty.toDouble())
        }
        call.resolve(payload)
    }

    /**
     * 端侧自检：把「装载状态 / 最近失败原因 / 模型目录清单 / BERT 是否就绪」一次性回传前端，
     * 用于排查「已拉取但不出声」「挂载失败」这类只能上真机复现的问题。
     */
    @PluginMethod
    fun diagnose(call: PluginCall) {
        scope.launch {
            try {
                val payload = JSObject()
                // 未挂载时也要能自检：扫描沙盒里已有的模型目录（冷启动自动点亮的依据）
                val scanDir = if (lastModelDir.isNotBlank()) File(lastModelDir) else detectInstalledModelDir()
                payload.put("engine", mountedEngineId ?: "未装载")
                payload.put("modelId", mountedModelId ?: scanDir?.name ?: "")
                payload.put("arch", mountedArch ?: "")
                payload.put("accelerator", activeAccelerator)
                payload.put("sampleRate", sampleRate)
                payload.put("modelDir", scanDir?.absolutePath ?: "")
                payload.put("mountError", lastMountError ?: "")
                payload.put("mountNotes", lastMountNotes)
                payload.put("synthError", lastSynthError ?: "")
                payload.put("synthInfo", lastSynthInfo)
                payload.put(
                    "sampling",
                    "topK=" + samplingConfig.topK + " T=" + samplingConfig.temperature +
                        " topP=" + samplingConfig.topP + " penalty=" + samplingConfig.repetitionPenalty
                )

                val inventory = JSObject()
                val dir = scanDir ?: File("")
                val rootFiles = JSArray()
                val emotions = JSArray()
                var bertReady = false
                var conditioningReady = false
                if (dir.isDirectory) {
                    dir.listFiles()?.sortedBy { it.name }?.take(40)?.forEach { file ->
                        rootFiles.put(
                            JSObject().apply {
                                put("name", file.name)
                                put("size", if (file.isDirectory) -1 else file.length())
                            }
                        )
                        if (file.isDirectory && file.name == "conditioning") {
                            conditioningReady = file.listFiles()?.any { it.name == "conditioning.json" } == true
                            // 情绪可用性以「条件包目录里有参考特征」为准，前端据此只展示/保存真实存在的语调
                            file.listFiles()
                                ?.filter { it.isDirectory && conditioningBlobReady(it) }
                                ?.map { it.name }
                                ?.sortedBy { EMOTION_ORDER.indexOf(it).let { i -> if (i >= 0) i else EMOTION_ORDER.size } }
                                ?.forEach { emotions.put(it) }
                        }
                        if (file.isDirectory && file.name == "bert") {
                            bertReady = file.listFiles()?.any { it.name.endsWith(".onnx") } == true
                        }
                    }
                }
                inventory.put("files", rootFiles)
                inventory.put("exists", dir.isDirectory)
                inventory.put("bertReady", bertReady)
                inventory.put("conditioningReady", conditioningReady)
                inventory.put("emotions", emotions)
                payload.put("inventory", inventory)
                payload.put("emotions", emotions)

                resolveOnUiThread(call, payload)
            } catch (error: Throwable) {
                rejectOnUiThread(call, "自检失败：${error.message}")
            }
        }
    }

    /** 在 filesDir/models 下找已经落地的语音包（含 conditioning.json 与 encoder 权重）。 */
    private fun detectInstalledModelDir(): File? {
        val modelsRoot = File(context.filesDir, "models")
        val firstLevel = modelsRoot.listFiles()?.filter { it.isDirectory } ?: emptyList()
        val secondLevel = firstLevel.flatMap {
            it.listFiles()?.filter { sub -> sub.isDirectory } ?: emptyList()
        }
        return (firstLevel + secondLevel).firstOrNull { dir ->
            File(dir, "conditioning/conditioning.json").isFile &&
                (File(dir, "t2s_encoder.onnx").isFile || File(dir, "t2s_encoder.int8.onnx").isFile)
        }
    }

    // ------------------------------------------------------------------ //
    // 生命周期与线程工具
    // ------------------------------------------------------------------ //
    override fun handleOnDestroy() {
        scope.cancel()
        runCatching { engine?.close() }
        engine = null
        audioPlayer.release()
        super.handleOnDestroy()
    }

    private fun resolveOnUiThread(call: PluginCall, payload: JSObject) {
        val activity = activity
        if (activity != null) {
            activity.runOnUiThread { call.resolve(payload) }
        } else {
            call.resolve(payload)
        }
    }

    private fun rejectOnUiThread(call: PluginCall, message: String) {
        val activity = activity
        if (activity != null) {
            activity.runOnUiThread { call.reject(message) }
        } else {
            call.reject(message)
        }
    }

    /** 把 Float32 PCM 封装成 16bit WAV 并 Base64（`playImmediately=false` 时使用）。 */
    private fun encodeWavBase64(samples: FloatArray, rate: Int): String {
        val dataBytes = samples.size * 2
        val buffer = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        val byteRate = rate * 2

        buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
        buffer.putInt(36 + dataBytes)
        buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
        buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
        buffer.putInt(16)
        buffer.putShort(1.toShort())          // PCM
        buffer.putShort(1.toShort())          // mono
        buffer.putInt(rate)
        buffer.putInt(byteRate)
        buffer.putShort(2.toShort())          // block align
        buffer.putShort(16.toShort())         // bits per sample
        buffer.put("data".toByteArray(Charsets.US_ASCII))
        buffer.putInt(dataBytes)

        for (value in samples) {
            val clamped = value.coerceIn(-1f, 1f)
            buffer.putShort((clamped * 32767f).toInt().toShort())
        }
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    companion object {
        private const val TAG = "AemeathTtsPlugin"

        /** 出厂默认采样约束：top_k=15 / temperature=0.85（前端可传入 samplingConfig 覆盖）。 */
        private const val DEFAULT_TOP_K = 15
        private const val DOWNLOAD_CONCURRENCY = 3
        private const val DOWNLOAD_TIMEOUT_SECONDS = 60L

        /** 情绪展示顺序；条件包里真实存在的才会出现在该顺序里。 */
        private val EMOTION_ORDER = listOf("平静", "开心", "温柔", "调侃")
    }

    /** 判断某个情绪目录是否真的带着可用的参考条件（ref_seq 或 ssl_content 任一即可）。 */
    private fun conditioningBlobReady(dir: File): Boolean {
        return File(dir, "ref_seq.i64").isFile || File(dir, "ssl_content.f32").isFile
    }
}
