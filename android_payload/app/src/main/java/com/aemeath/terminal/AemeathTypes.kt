package com.aemeath.terminal

import android.content.Context
import java.io.File

/**
 * AemeathTypes.kt — 原生推理层共享的数据契约
 *
 * 这里的每个类型都严格对齐 `ARCHITECTURE.md` §2 的 TypeScript 接口，
 * 前端 `LocalOnnxTtsBridge` 传入的 JSON 字段名与这里的字段一一对应。
 */

/** 情绪标签：必须与前端 `[平静] [开心] [温柔] [调侃] [愤怒生气]` 芯片完全一致。 */
enum class Emotion(val key: String) {
    CALM("平静"),
    HAPPY("开心"),
    GENTLE("温柔"),
    TEASE("调侃"),
    ANGRY("愤怒生气");

    companion object {
        val ALL: List<Emotion> = values().toList()

        fun fromKey(key: String?): Emotion = when (key?.trim()) {
            "平静" -> CALM
            "开心" -> HAPPY
            "温柔" -> GENTLE
            "调侃" -> TEASE
            "愤怒生气", "愤怒", "生气" -> ANGRY
            else -> GENTLE
        }
    }
}

/** 硬件推理后端：映射前端 `backend` 参数，见 ARCHITECTURE.md §2.2。 */
enum class ComputeBackend(val raw: String) {
    NPU("npu"),
    GPU("gpu"),
    CPU("cpu");

    companion object {
        fun fromRaw(raw: String?): ComputeBackend =
            values().firstOrNull { it.raw.equals(raw?.trim(), ignoreCase = true) } ?: NPU
    }
}

/**
 * 概率核采样参数。
 *
 * 默认 `topK = 15`、`temperature = 0.85`（端侧出厂口径）；
 * 前端若在 `mountModel.samplingConfig` 中显式传入（例如 25），以传入值为准。
 * **严禁**把 topK 设为 1 或把 temperature 设为 0 —— 等价于 argmax 贪婪解码，
 * 会让爱弥斯彻底失去语气灵动感（AGENTS.md 红线 3）。
 */
data class SamplingConfig(
    val temperature: Float = 0.85f,
    val topK: Int = 15,
    val topP: Float = 0.90f,
    val repetitionPenalty: Float = 1.35f,
    val stopToken: Int = 1024,
    val maxSemanticLength: Int = 512,
) {
    /**
     * 归一化到安全区间：任何「想退化成 argmax」的入参（topK=1 / temperature=0）
     * 都会在这里被纠正，而不是抛异常让前端拿不到声音。
     */
    fun sanitized(): SamplingConfig = copy(
        temperature = temperature.coerceIn(0.05f, 2.0f),
        topK = topK.coerceIn(2, 512),
        topP = topP.coerceIn(0.05f, 1.0f),
        repetitionPenalty = repetitionPenalty.coerceIn(1.0f, 3.0f),
        maxSemanticLength = maxSemanticLength.coerceIn(16, 2048),
    )
}

/** `mountModel` 的原生入参（已从 JSObject 解析完毕）。 */
data class EngineMountConfig(
    val modelId: String,
    val arch: String,
    val engineName: String,
    val backend: ComputeBackend,
    val modelDir: File,
    val emotionFiles: Map<Emotion, String>,
    val sampling: SamplingConfig,
)

/** `mountModel` 的返回结构，字段名与 TS 契约一致。 */
data class EngineMountResult(
    val success: Boolean,
    val modelId: String,
    val engine: String,
    val sampleRate: Int,
    val channels: Int,
    val accelerationActive: String,
    val memoryFootprintMb: Int,
    val notes: List<String> = emptyList(),
)

/** 一次合成的纯音频结果（未编码）。 */
class SynthesisOutput(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    val durationMs: Long get() = if (sampleRate <= 0) 0L else samples.size * 1000L / sampleRate
}

/** 下载进度事件：字段名对齐 `ModelDownloadProgressEvent`。 */
data class DownloadProgress(
    val modelId: String,
    val progress: Int,
    val stage: String,
    val downloadedBytes: Long,
    val totalBytes: Long,
    val speed: String,
)

/** 下载最终结果：字段名对齐 `PullModelResult`。 */
data class DownloadResult(
    val success: Boolean,
    val modelId: String,
    val savedDir: String,
    val totalBytes: Long,
    val hashVerified: Boolean,
    val failedFiles: List<String> = emptyList(),
    val message: String = "",
)

/**
 * 沙盒路径解析：前端传来的是相对路径（如 `models/aemeath-vits-lite`），
 * 统一挂载到 `context.filesDir` 之下；绝对路径仅允许落在应用私有目录内，
 * 杜绝通过 `../` 或绝对路径读取应用外部文件。
 */
object ModelPaths {
    private const val DEFAULT_ROOT = "models/aemeath"

    fun sandboxRoot(context: Context, modelsRoot: String? = null): File {
        val root = if (modelsRoot.isNullOrBlank()) DEFAULT_ROOT else modelsRoot
        return File(context.filesDir, root).apply { mkdirs() }
    }

    fun resolve(context: Context, rawPath: String?, modelsRoot: String? = null): File {
        val raw = (rawPath ?: "").replace('\\', '/').trim()
        if (raw.isEmpty()) {
            return sandboxRoot(context, modelsRoot)
        }

        val filesRoot = context.filesDir.canonicalFile

        // 绝对路径（前端会直接传 diagnose() 返回的沙盒真实路径）：
        // 必须先判绝对，绝不能 trimStart('/') 之后当相对路径拼接，
        // 否则会拼出 files/data/data/... 这种不存在的嵌套目录，挂载静默失败。
        if (raw.startsWith("/") || raw.startsWith("file://", ignoreCase = true)) {
            val candidate = if (raw.startsWith("file://", ignoreCase = true)) {
                runCatching { File(java.net.URI(raw)) }.getOrNull()
            } else {
                File(raw)
            }
            val target = candidate?.let { runCatching { it.canonicalFile }.getOrNull() }
            if (target != null && target.path.startsWith(filesRoot.path)) {
                return target
            }
            return File(sandboxRoot(context, modelsRoot), target?.name ?: raw.substringAfterLast('/'))
        }

        val cleaned = raw.trimStart('/')
        val underFiles = File(context.filesDir, cleaned).canonicalFile
        return if (underFiles.path.startsWith(filesRoot.path)) {
            underFiles
        } else {
            File(sandboxRoot(context, modelsRoot), underFiles.name)
        }
    }

    /** 沙盒内的相对路径（用于写回前端可读的绝对路径与 manifest 校验）。 */
    fun safeRelativePath(raw: String): String? {
        val normalized = raw.replace('\\', '/').trim().trimStart('/')
        if (normalized.isEmpty()) return null
        val segments = normalized.split('/')
        if (segments.any { it == ".." || it.isEmpty() }) return null
        return segments.joinToString("/")
    }
}

/** 引擎统一接口：`AemeathTtsPlugin` 只面向该接口，不关心底层是 Sherpa 还是 ORT。 */
interface AemeathTtsEngine {
    val engineId: String

    /** 装载权重、词典与情绪向量，返回采样率与实际启用的加速器。 */
    fun mount(config: EngineMountConfig): EngineMountResult

    /**
     * 离线合成，返回 Float32 PCM（单声道）。
     *
     * 单句内**只跑一次声码器**：句内反复跑 vits 会把总耗时翻倍、听感变成「说一下卡很久」
     * （m05844 第 1 条）。「边播边算」改由 [AemeathAudioPlayer] 的播放队列在句与句之间实现：
     * 上一句还在播时，CPU 已经在算下一句。
     */
    fun synthesize(
        text: String,
        emotion: Emotion,
        speed: Float,
        volume: Float,
    ): SynthesisOutput

    /** 释放 ORT / Sherpa 会话与本地句柄。 */
    fun close()

    /** 估算常驻内存（MB），用于 mountModel 回传。 */
    fun memoryFootprintMb(): Int
}

/** 原生层可读的运行时错误，便于回传给前端时给出可操作的提示。 */
class AemeathEngineException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
