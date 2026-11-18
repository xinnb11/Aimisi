package com.aemeath.terminal

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import java.io.File
import java.util.EnumSet

/**
 * AemeathOrtBackend.kt — ONNX Runtime Mobile 硬件后端调度（天玑 9500 异构分工）
 *
 * 目标芯片是**联发科天玑 9500**（MediaTek Dimensity，APU 890 + Immortalis-G925），
 * 没有高通 Hexagon，因此这里**不存在任何 QNN 代码路径**（m07045 第 2 条）。
 *
 * 分工策略（m07045 第 3 条）：
 * | 阶段 | 后端 | 原因 |
 * | :--- | :--- | :--- |
 * | `vits` 声码器 | **NNAPI (USE_FP16) → XNNPACK → CPU** | 静态大矩阵、一次前向，适合天玑 APU |
 * | `t2s_sdec` | **只准 CPU (ARM NEON · 4 线程)** | 逐 step 动态自回归，丢进 NNAPI 会让 APU HAL 驱动层死锁；XNNPACK 在该图上数值错误 |
 * | `t2s_encoder` / `t2s_fsdec` / `bert` | XNNPACK → CPU | 小图、单次前向，XNNPACK 已是实测最优 |
 *
 * 每个 Session 独立探测，任何一个失败只降级它自己，绝不整包回退。
 */
object AemeathOrtBackend {

    private const val TAG = "AemeathOrt"

    /** 单个模型可尝试的加速后端，按优先级从高到低。 */
    enum class Ep {
        NNAPI_FP16,
        XNNPACK,
        CPU_NEON,
    }

    data class SessionBundle(
        val session: OrtSession,
        val ep: Ep,
        val accelerator: String,
        val notes: List<String>,
    )

    /** 每个阶段探测失败的历史原因，便于 diagnose 输出。 */
    private val stageNotes = LinkedHashMap<String, MutableList<String>>()

    fun clearNotes() = stageNotes.clear()

    fun notesSnapshot(): Map<String, List<String>> = stageNotes.mapValues { it.value.toList() }

    /**
     * 为某个模型挑选「能用且数值可信」的最强后端。
     *
     * @param stage        阶段名（bert/enc/fsdec/sdec/vits），只用于日志与诊断
     * @param backend      前端偏好后端（npu/gpu/cpu）；`cpu` 时仍会给 vits 之外的阶段用 XNNPACK
     * @param allowNnapi   是否允许该阶段走 NNAPI（只有 `vits` 允许）
     * @param allowXnnpack 该模型在 XNNPACK 上是否数值可信（`t2s_sdec` 传 false）
     * @param probe        可选数值校验：返回 null 表示该 EP 结果可用，返回字符串表示不可用原因
     */
    fun openStage(
        env: OrtEnvironment,
        modelFile: File,
        stage: String,
        backend: ComputeBackend,
        threads: Int = 0,
        allowNnapi: Boolean = false,
        allowXnnpack: Boolean = true,
        probe: ((OrtSession) -> String?)? = null,
    ): SessionBundle {
        require(modelFile.isFile) { "模型文件不存在：${modelFile.absolutePath}" }
        val resolvedThreads = if (threads > 0) threads else autoThreads()
        val notes = stageNotes.getOrPut(stage) { mutableListOf() }
        val candidates = candidatesFor(backend, allowNnapi, allowXnnpack)
        var lastError: Throwable? = null

        for (ep in candidates) {
            var options: OrtSession.SessionOptions? = null
            try {
                options = buildOptions(ep, resolvedThreads)
                val session = env.createSession(modelFile.absolutePath, options)
                val failure = probe?.invoke(session)
                if (failure != null) {
                    notes += "${ep.name} 数值校验未通过：$failure"
                    Log.w(TAG, "[$stage] ${ep.name} 数值校验未通过：$failure")
                    runCatching { session.close() }
                    continue
                }
                val label = labelOf(ep, resolvedThreads)
                Log.i(TAG, "[$stage] 会话建立：${modelFile.name} @ $label")
                notes += "最终 EP：$label"
                return SessionBundle(session, ep, label, notes.toList())
            } catch (error: Throwable) {
                lastError = error
                val reason = rootMessage(error)
                notes += "${ep.name} 不可用：$reason"
                Log.w(TAG, "[$stage] ${ep.name} 不可用：$reason")
            } finally {
                options?.close()
            }
        }

        throw AemeathEngineException(
            "加载 $stage 失败（已尝试 ${candidates.joinToString(" → ") { it.name }}）：" +
                (lastError?.message ?: "无可用后端"),
            lastError,
        )
    }

    /** 兼容旧调用：单一模型、允许 XNNPACK、无探针。 */
    fun createSession(
        env: OrtEnvironment,
        modelFile: File,
        backend: ComputeBackend,
        threads: Int = 0,
    ): SessionBundle = openStage(env, modelFile, modelFile.name, backend, threads)

    /**
     * 后端尝试顺序。
     *
     * - `allowNnapi` 只在声码器上为 true（静态图）；`npu`/`gpu` 偏好只影响是否优先试 NNAPI。
     * - `allowXnnpack = false` 时直接只走 CPU NEON（sdec 专用）。
     */
    fun candidatesFor(backend: ComputeBackend, allowNnapi: Boolean, allowXnnpack: Boolean): List<Ep> {
        val list = mutableListOf<Ep>()
        val prefersHardware = backend != ComputeBackend.CPU
        if (allowNnapi && prefersHardware) list += Ep.NNAPI_FP16
        if (allowNnapi && !prefersHardware) list += Ep.NNAPI_FP16 // 声码器即使 cpu 偏好也允许 APU 试一次
        if (allowXnnpack) list += Ep.XNNPACK
        list += Ep.CPU_NEON
        return list.distinct()
    }

    private fun labelOf(ep: Ep, threads: Int): String = when (ep) {
        Ep.NNAPI_FP16 -> "NNAPI (FP16, APU) · $threads 线程"
        Ep.XNNPACK -> "XNNPACK · $threads 线程"
        Ep.CPU_NEON -> "CPU (ARM NEON) · $threads 线程"
    }

    private fun buildOptions(ep: Ep, threads: Int): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        val intra = if (ep == Ep.NNAPI_FP16) 1 else threads.coerceIn(1, 8)
        options.setIntraOpNumThreads(intra)
        options.setInterOpNumThreads(1)

        when (ep) {
            // 天玑 APU 走 NNAPI；必须带 USE_FP16，否则以 FP32 整图跑既慢又易在动态 shape 上崩。
            Ep.NNAPI_FP16 -> options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))

            Ep.XNNPACK -> options.addXnnpack(
                mapOf(
                    "intra_op_num_threads" to threads.toString(),
                    "inter_op_num_threads" to "1",
                    "enable_cpu_mem_arena" to "1",
                )
            )

            Ep.CPU_NEON -> Unit
        }
        return options
    }

    private fun rootMessage(error: Throwable): String {
        var current: Throwable? = error
        var message = error.message ?: error.javaClass.simpleName
        while (current?.cause != null && current.cause !== current) {
            current = current.cause
            message = current?.message ?: message
        }
        return message.lineSequence().firstOrNull()?.trim().orEmpty()
    }

    /**
     * 线程数：天玑 9500 为 1+3+4 三丛集，超大核承担自回归主线程最划算；
     * 实测 4 线程（含 1 个 X4 超大核 + 3 大核）比 8 线程更快（屏障同步开销更低）。
     */
    fun autoThreads(): Int = 4
}
