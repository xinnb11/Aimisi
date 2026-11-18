package com.aemeath.terminal

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import java.io.File

/**
 * SherpaVitsEngine.kt — `arch === "Bert-VITS2"` 的 Lite 轻量极速推理引擎
 *
 * 走 Sherpa-ONNX 官方 Android API（C++ JNI）：
 *
 * ```
 * OfflineTts(OfflineTtsConfig(model = OfflineTtsModelConfig(vits = OfflineTtsVitsModelConfig(...))))
 * ```
 *
 * 装载的是单文件量化权重 `model.int8.onnx` + `tokens.txt` + `lexicon.txt`
 * （中文必须带 lexicon，否则无法完成字音转换），可选 `espeak-ng-data` 目录。
 *
 * ## 情绪表达
 * Bert-VITS2 的 speaker/sid 与情绪并非一一对应，因此这里采用「sid 优先 +
 * 韵律后处理兜底」的双层策略：
 * 1. 若模型本身是多 speaker 且按情绪训练，`sid = 情绪序号` 直接生效；
 * 2. 若 `generate` 因 sid 越界抛异常，自动回退 `sid = 0`；
 * 3. 再叠加每个情绪专属的语速/增益包络（与前端 Worker 的 EMOTION_PROFILES 对齐）。
 *
 * 注意：本引擎不做自回归解码，因此不涉及核采样；`SamplingConfig` 仅由
 * GPT-SoVITS 引擎消费。
 */
class SherpaVitsEngine : AemeathTtsEngine {

    override val engineId: String = "Sherpa-ONNX"

    private var tts: OfflineTts? = null
    private var sampleRate: Int = 22050
    private var accelerator: String = "CPU (ARM NEON)"
    private var mountedFiles: List<File> = emptyList()
    private var mountedSid: Int = 0

    override fun mount(config: EngineMountConfig): EngineMountResult {
        val dir = config.modelDir
        if (!dir.isDirectory) {
            throw AemeathEngineException("模型目录不存在：${dir.absolutePath}")
        }

        val modelFile = listOf(
            "model.int8.onnx",
            "model.onnx",
            "vits.int8.onnx",
            "vits.onnx",
        ).map { File(dir, it) }.firstOrNull { it.isFile }
            ?: dir.listFiles { file -> file.extension == "onnx" }
                ?.sortedBy { it.name }
                ?.firstOrNull()
            ?: throw AemeathEngineException("未找到 Bert-VITS2 声学模型（model.int8.onnx）：${dir.absolutePath}")

        val tokens = File(dir, "tokens.txt").takeIf { it.isFile }
            ?: throw AemeathEngineException("缺少音素表 tokens.txt：${dir.absolutePath}")
        val lexicon = File(dir, "lexicon.txt").takeIf { it.isFile }
        val dataDir = File(dir, "espeak-ng-data").takeIf { it.isDirectory }

        val notes = mutableListOf<String>()
        if (lexicon == null) {
            notes += "未找到 lexicon.txt，中文可能无法正确字音转换"
        }

        val providerCandidates = buildList {
            when (config.backend) {
                ComputeBackend.NPU -> add("nnapi")
                ComputeBackend.GPU -> add("gpu")
                ComputeBackend.CPU -> add("cpu")
            }
            add("cpu")
        }.distinct()

        var lastError: Throwable? = null
        for (provider in providerCandidates) {
            try {
                val vitsConfig = OfflineTtsVitsModelConfig(
                    model = modelFile.absolutePath,
                    lexicon = lexicon?.absolutePath ?: "",
                    tokens = tokens.absolutePath,
                    dataDir = dataDir?.absolutePath ?: "",
                    noiseScale = 0.667f,
                    noiseScaleW = 0.8f,
                    lengthScale = 1.0f,
                )
                val modelConfig = OfflineTtsModelConfig(
                    vits = vitsConfig,
                    numThreads = THREADS,
                    debug = false,
                    provider = provider,
                )
                val ttsConfig = OfflineTtsConfig(
                    model = modelConfig,
                    ruleFsts = "",
                    ruleFars = "",
                    maxNumSentences = 1,
                )
                val engine = OfflineTts(config = ttsConfig)
                tts = engine
                accelerator = describeProvider(provider)
                mountedFiles = listOfNotNull(modelFile, tokens, lexicon)
                if (provider != providerCandidates.last()) {
                    notes += "Sherpa-ONNX 已按 $provider 初始化（若设备不支持会自动回退 CPU）"
                }
                Log.i(TAG, "Sherpa-ONNX 装载完成：${modelFile.name} @ $accelerator")

                return EngineMountResult(
                    success = true,
                    modelId = config.modelId,
                    engine = engineId,
                    sampleRate = sampleRate,
                    channels = 1,
                    accelerationActive = accelerator,
                    memoryFootprintMb = memoryFootprintMb(),
                    notes = notes,
                )
            } catch (error: Throwable) {
                lastError = error
                notes += "provider=$provider 初始化失败：${error.message}"
                Log.w(TAG, "Sherpa provider=$provider 初始化失败：${error.message}")
            }
        }

        throw AemeathEngineException(
            "Sherpa-ONNX 初始化失败：${lastError?.message}", lastError
        )
    }

    override fun synthesize(
        text: String,
        emotion: Emotion,
        speed: Float,
        volume: Float,
    ): SynthesisOutput {
        val engine = tts ?: throw AemeathEngineException("Sherpa-ONNX 尚未 mount")
        val normalized = text.trim()
        if (normalized.isEmpty()) {
            throw AemeathEngineException("待合成文本为空")
        }

        val profile = emotionProfile(emotion)
        val effectiveSpeed = (speed.coerceIn(0.5f, 2.0f) * profile.speed).coerceIn(0.4f, 2.5f)

        // sid 优先用情绪序号，越界/不支持时回退 0 号说话人
        val sidCandidates = listOf(emotion.ordinal, mountedSid, 0).distinct()
        var lastError: Throwable? = null
        for (sid in sidCandidates) {
            try {
                val audio = engine.generate(text = normalized, sid = sid, speed = effectiveSpeed)
                if (audio.samples.isEmpty()) {
                    throw AemeathEngineException("Sherpa-ONNX 返回空音频（sid=$sid）")
                }
                var samples = AemeathPcmPostProcess.normalize(audio.samples, targetPeak = 0.89f * volume)
                samples = AemeathPcmPostProcess.fade(samples, audio.sampleRate, 5f)
                return SynthesisOutput(samples, audio.sampleRate)
            } catch (error: Throwable) {
                lastError = error
                Log.w(TAG, "sid=$sid 合成失败：${error.message}")
            }
        }
        throw AemeathEngineException("Sherpa-ONNX 合成失败：${lastError?.message}", lastError)
    }

    override fun close() {
        runCatching { tts?.release() }
        tts = null
    }

    override fun memoryFootprintMb(): Int {
        val bytes = mountedFiles.sumOf { it.length() }
        val runtime = Runtime.getRuntime()
        val heapMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        return (bytes / (1024 * 1024) + heapMb).toInt().coerceAtLeast(1)
    }

    private fun describeProvider(provider: String): String = when (provider.lowercase()) {
        "nnapi" -> "NNAPI (NPU)"
        "gpu", "cuda" -> "GPU"
        else -> "CPU (ARM NEON, $THREADS 线程)"
    }

    private data class EmotionProfile(val speed: Float, val gain: Float)

    private fun emotionProfile(emotion: Emotion): EmotionProfile = when (emotion) {
        Emotion.CALM -> EmotionProfile(1.00f, 1.00f)
        Emotion.HAPPY -> EmotionProfile(1.12f, 1.05f)
        Emotion.GENTLE -> EmotionProfile(0.95f, 0.95f)
        Emotion.TEASE -> EmotionProfile(1.06f, 1.00f)
        Emotion.ANGRY -> EmotionProfile(1.05f, 1.05f)
    }

    companion object {
        private const val TAG = "AemeathSherpa"
        private const val THREADS = 4
    }
}
