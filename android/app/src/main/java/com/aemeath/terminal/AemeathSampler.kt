package com.aemeath.terminal

import java.util.Random
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

/**
 * AemeathSampler.kt — 端侧概率核采样器（Top-K / Top-P / Temperature / Repetition Penalty）
 *
 * ## 为什么必须存在
 * GPT-SoVITS v2 的 T2S 自回归解码一旦退化为 `argmax(logits)`，输出会极度死板、
 * 并且极易陷入复读死循环。AGENTS.md 红线 3 明确禁止任何贪婪解码。
 *
 * 本采样器严格复刻前端 `SamplingStrategy.ts` / Web Worker 的采样语义：
 *
 * 1. **Repetition Penalty**：对最近 64 个已生成 token 施加滑动惩罚
 *    （正值除以 penalty，负值乘以 penalty）；
 * 2. **Temperature**：整体 logits 乘以 `1/T`；
 * 3. **Top-K**：保留概率最高的 K 个候选（默认 15）；
 * 4. **Top-P (Nucleus)**：按累积概率截断，只保留核内候选并重新归一化；
 * 5. **Multinomial**：以核内概率做轮盘赌随机抽样。
 *
 * 实现上使用「K 路插入选择 + 局部 softmax」，无需对整个词表排序，
 * 1024 词表 / K=15 下单步耗时稳定在微秒级，适合 4 线程 CPU 自回归循环。
 */
class AemeathSampler(
    private val vocabSize: Int,
    cfg: SamplingConfig,
    seed: Long = System.nanoTime(),
) {
    private val config: SamplingConfig = cfg.sanitized()
    private val random = Random(seed)

    private val topK: Int = min(config.topK, vocabSize).coerceAtLeast(2)
    private val topIndices = IntArray(topK)
    private val topValues = FloatArray(topK)
    private val adjusted = FloatArray(vocabSize)

    /** 最近生成 token 的滑动窗口，用于重复惩罚（与前端一致：64 步）。 */
    private val historyWindow = IntArray(HISTORY_SIZE)
    private var historyCount = 0

    fun reset() {
        historyCount = 0
        java.util.Arrays.fill(historyWindow, -1)
    }

    /**
     * 依据 logits 采样下一个 token。
     *
     * @param logits 当前步的原始 logits（长度必须为 vocabSize）
     * @return 采样得到的 token id（绝不会是 `argmax` 的直接结果）
     */
    fun sample(logits: FloatArray): Int {
        require(logits.size == vocabSize) {
            "logits 维度 ${logits.size} 与词表 $vocabSize 不匹配"
        }

        // 1) 拷贝 + 重复惩罚
        System.arraycopy(logits, 0, adjusted, 0, vocabSize)
        applyRepetitionPenalty()

        // 2) 温度缩放
        val invTemperature = 1.0f / config.temperature
        if (invTemperature != 1.0f) {
            for (i in 0 until vocabSize) {
                adjusted[i] *= invTemperature
            }
        }

        // 3) Top-K 候选（K 路插入选择，避免全词表排序）
        selectTopK()

        // 4) 局部 softmax
        val maxLogit = topValues[0]
        if (maxLogit == Float.NEGATIVE_INFINITY) {
            return random.nextInt(vocabSize)
        }
        var sum = 0.0f
        for (i in 0 until topK) {
            val value = exp((topValues[i] - maxLogit).toDouble()).toFloat()
            topValues[i] = value
            sum += value
        }
        if (!sum.isFinite() || sum <= 0f) {
            return random.nextInt(vocabSize)
        }

        // 5) Top-P 核截断（候选已按 logit 降序，累积概率单调递增）
        var cumulative = 0f
        var cut = topK
        for (i in 0 until topK) {
            cumulative += topValues[i] / sum
            if (cumulative >= config.topP) {
                cut = i + 1
                break
            }
        }
        if (cut < 1) {
            cut = 1
        }

        var nucleusSum = 0f
        for (i in 0 until cut) nucleusSum += topValues[i]
        if (nucleusSum <= 0f || !nucleusSum.isFinite()) {
            return uniformAmongNucleus(cut)
        }

        // 6) 多项式轮盘赌抽样
        val target = random.nextFloat() * nucleusSum
        var accumulator = 0f
        for (i in 0 until cut) {
            accumulator += topValues[i]
            if (target <= accumulator) {
                val token = topIndices[i]
                remember(token)
                return token
            }
        }

        // 浮点误差兜底：在核内均匀抽样（仍然不是 argmax）
        val fallback = uniformAmongNucleus(cut)
        remember(fallback)
        return fallback
    }

    /** 记录已生成 token，供下一步重复惩罚使用。 */
    fun remember(token: Int) {
        historyWindow[historyCount % HISTORY_SIZE] = token
        historyCount += 1
    }

    /** 当前分布的信息熵（nats），仅用于诊断日志与单测断言。 */
    fun entropyOf(logits: FloatArray): Float {
        var maxValue = Float.NEGATIVE_INFINITY
        for (value in logits) if (value > maxValue) maxValue = value
        var sum = 0.0f
        for (value in logits) sum += exp((value - maxValue).toDouble()).toFloat()
        if (sum <= 0f) return 0f
        var entropy = 0.0f
        for (value in logits) {
            val p = exp((value - maxValue).toDouble()).toFloat() / sum
            if (p > 1e-9f) entropy -= p * ln(p).toFloat()
        }
        return entropy
    }

    private fun applyRepetitionPenalty() {
        val penalty = config.repetitionPenalty
        if (penalty == 1.0f || historyCount == 0) return
        val window = min(historyCount, HISTORY_SIZE)
        for (i in 0 until window) {
            val token = historyWindow[i]
            if (token !in 0 until vocabSize) continue
            val value = adjusted[token]
            adjusted[token] = if (value > 0f) value / penalty else value * penalty
        }
    }

    private fun selectTopK() {
        for (i in 0 until topK) {
            topIndices[i] = -1
            topValues[i] = Float.NEGATIVE_INFINITY
        }
        for (token in 0 until vocabSize) {
            val value = adjusted[token]
            if (!value.isFinite() || value <= topValues[topK - 1]) continue
            var position = topK - 1
            while (position > 0 && topValues[position - 1] < value) {
                topValues[position] = topValues[position - 1]
                topIndices[position] = topIndices[position - 1]
                position -= 1
            }
            topValues[position] = value
            topIndices[position] = token
        }
        // 防御：极端情况下（全是 NaN/-Inf）至少留下一个合法 token
        if (topIndices[0] < 0) {
            topIndices[0] = 0
            topValues[0] = 0f
        }
    }

    private fun uniformAmongNucleus(cut: Int): Int {
        val index = random.nextInt(cut)
        return topIndices[index].coerceIn(0, vocabSize - 1)
    }

    companion object {
        private const val HISTORY_SIZE = 64
    }
}
