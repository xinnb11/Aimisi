package com.aemeath.terminal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 端侧 BERT 文本前端（`bert/model.int8.onnx` + `bert/vocab.txt`）。
 *
 * 作用只有一个：把「待合成文本」变成 GPT-SoVITS T2S encoder 需要的 `text_bert` [T,1024]。
 *
 * 关键规则（与 `scripts/verify_onnx_tts.py` 逐项等价，改一端必须改另一端）：
 * 1. 音素序列按「字」分组，`char2phone.json` 给出每个字对应几个音素；
 * 2. BERT 取 `last_hidden_state`，去掉 `[CLS]`/`[SEP]`，按字平均成 1024 维向量；
 * 3. 每个字的向量按它自己的音素个数重复，拼成音素级特征（行数必须等于音素数）。
 *
 * 内存约束：词表 21128 行、会话约 311MB（int8）。因此只在真正开始合成前装载，
 * 命名走显式匹配，缺名直接报错而不是静默喂零。
 */
internal class AemeathBertFrontend(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val vocab: Map<String, Int>,
    private val inputIdsName: String,
    private val attentionMaskName: String,
    private val tokenTypeIdsName: String,
    private val hiddenName: String,
    private val dim: Int = 1024,
) : AutoCloseable {

    /**
     * @param text   与 `counts` 一一对应的字符序列（不含空白字符）
     * @param counts 每个字的音素个数；0 表示这个字没有音素（跳过）
     * @return 音素级特征，长度 = `counts.sum() * dim`
     */
    fun encode(text: String, counts: IntArray): FloatArray {
        require(counts.size == text.length) {
            "字符与音素计数长度不一致：${text.length} vs ${counts.size}"
        }
        val totalPhones = counts.sum()
        if (totalPhones <= 0) return FloatArray(0)

        // ---- 1) WordPiece ----
        val tokenIds = ArrayList<Int>(text.length + 2)
        val tokenChar = ArrayList<Int>(text.length + 2)
        for (index in text.indices) {
            val ch = text[index]
            val pieces = piecesOf(ch)
            for (piece in pieces) {
                val id = vocab[piece] ?: vocab["[UNK]"] ?: UNK_ID
                tokenIds += id
                tokenChar += index
            }
        }
        val sequenceLength = tokenIds.size + 2
        val ids = LongArray(sequenceLength)
        val mask = LongArray(sequenceLength)
        val types = LongArray(sequenceLength)
        ids[0] = CLS_ID
        ids[sequenceLength - 1] = SEP_ID
        for (i in tokenIds.indices) {
            ids[i + 1] = tokenIds[i].toLong()
            mask[i + 1] = 1L
        }
        mask[0] = 1L
        mask[sequenceLength - 1] = 1L

        // ---- 2) 前向 ----
        val hidden = runForward(ids, mask, types, sequenceLength)

        // ---- 3) 按字聚合后再按音素数重复 ----
        val sums = Array(text.length) { FloatArray(dim) }
        val hits = IntArray(text.length)
        for (token in tokenIds.indices) {
            val owner = tokenChar[token]
            val base = (token + 1) * dim
            val target = sums[owner]
            for (d in 0 until dim) {
                target[d] += hidden[base + d]
            }
            hits[owner] += 1
        }

        val out = FloatArray(totalPhones * dim)
        var row = 0
        for (index in text.indices) {
            val count = counts[index]
            if (count <= 0) continue
            val vector = sums[index]
            val divisor = hits[index].toFloat()
            if (divisor > 0f) {
                for (d in 0 until dim) vector[d] /= divisor
            }
            for (repeat in 0 until count) {
                System.arraycopy(vector, 0, out, row * dim, dim)
                row += 1
            }
        }
        return out
    }

    private fun runForward(ids: LongArray, mask: LongArray, types: LongArray, sequenceLength: Int): FloatArray {
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            tensors[inputIdsName] = longTensor(ids)
            tensors[attentionMaskName] = longTensor(mask)
            tensors[tokenTypeIdsName] = longTensor(types)
            val result = session.run(tensors)
            try {
                // ORT 1.22 起 Result.get(String) 返回 java.util.Optional<OnnxValue>；
                // 1.18 直接返回 OnnxValue。这里兼容两种签名，避免「没有输出」的假报错。
                val value = result.get(hiddenName).orElse(null)
                if (value is OnnxTensor) {
                    val info = value.info as? TensorInfo
                    val shape = info?.shape ?: LongArray(0)
                    if (shape.isNotEmpty() && shape.last() != dim.toLong()) {
                        throw AemeathEngineException(
                            "BERT 输出维度异常：${shape.toList()}（期望最后一维 $dim）"
                        )
                    }
                    val buffer = value.floatBuffer.duplicate()
                    buffer.rewind()
                    val data = FloatArray(buffer.remaining())
                    buffer.get(data)
                    return data
                }
                throw AemeathEngineException("BERT 没有输出 $hiddenName")
            } finally {
                runCatching { result.close() }
            }
        } catch (error: AemeathEngineException) {
            throw error
        } catch (error: Throwable) {
            throw AemeathEngineException(
                "BERT 前向失败（seq=$sequenceLength）：${error.message}",
                error,
            )
        } finally {
            tensors.values.forEach { tensor -> runCatching { tensor.close() } }
            tensors.clear()
        }
    }

    private fun longTensor(values: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size * 8)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()
        buffer.put(values)
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, values.size.toLong()))
    }

    /** 单字 -> WordPiece 片段。汉字与标点整字成词，ASCII 走贪心最长匹配。 */
    private fun piecesOf(ch: Char): List<String> {
        if (ch.isWhitespace()) return emptyList()
        val text = ch.toString()
        if (isCjk(ch) || !ch.isLetterOrDigit()) return listOf(text)
        val lower = text.lowercase()
        val pieces = ArrayList<String>(2)
        var start = 0
        while (start < lower.length) {
            var end = lower.length
            var matched: String? = null
            while (start < end) {
                var candidate = lower.substring(start, end)
                if (start > 0) candidate = "##$candidate"
                if (vocab.containsKey(candidate)) {
                    matched = candidate
                    break
                }
                end -= 1
            }
            if (matched == null) {
                pieces += "[UNK]"
                break
            }
            pieces += matched
            start = end
        }
        return pieces
    }

    /** 诊断用：会话真实输入/输出名与本端选择的输出名（排查 ONNX 命名不一致）。 */
    fun ioSummary(): String = buildString {
        append("in=")
        append(session.inputInfo.keys.joinToString(","))
        append(" out=")
        append(session.outputInfo.keys.joinToString(","))
        append(" use=")
        append(hiddenName)
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        private const val CLS_ID = 101L
        private const val SEP_ID = 102L
        private const val UNK_ID = 100

        private fun isCjk(ch: Char): Boolean {
            val code = ch.code
            return (code in 0x4E00..0x9FFF) ||
                (code in 0x3400..0x4DBF) ||
                (code in 0xF900..0xFAFF)
        }

        /** 从 `vocab.txt` 读出「词 -> id」。 */
        fun loadVocab(file: File): Map<String, Int> {
            val text = file.readText(Charsets.UTF_8)
            val vocab = HashMap<String, Int>(24000)
            var index = 0
            var start = 0
            var cursor = 0
            while (cursor <= text.length) {
                if (cursor == text.length || text[cursor] == '\n') {
                    var end = cursor
                    if (end > start && text[end - 1] == '\r') end -= 1
                    if (end > start) vocab[text.substring(start, end)] = index
                    index += 1
                    start = cursor + 1
                }
                cursor += 1
            }
            return vocab
        }

        /** 打开 BERT 会话并解析张量名；缺任一必需项返回 null（调用方退化为喂零）。 */
        fun open(
            env: OrtEnvironment,
            modelFile: File,
            vocabFile: File,
            backend: ComputeBackend,
            threads: Int,
            onEp: ((String) -> Unit)? = null,
        ): AemeathBertFrontend? {
            val inputs = modelFile.let {
                val bundle = AemeathOrtBackend.openStage(
                    env, it, "bert", backend, threads, allowXnnpack = true,
                )
                onEp?.invoke(bundle.accelerator)
                bundle.session
            }
            val inputNames = inputs.inputInfo.keys.toList()
            val outputNames = inputs.outputInfo.keys.toList()
            val idName = pick(inputNames, "input_ids", "input_ids:0") ?: return null
            val maskName = pick(inputNames, "attention_mask", "input_mask") ?: return null
            val typeName = pick(inputNames, "token_type_ids", "segment_ids") ?: return null
            // GPT-SoVITS 官方取 hidden_states[-3]（倒数第三层），不是 last_hidden_state。
            // `bert/model.layer21.int8.onnx` 已把该层直接做成唯一输出，名字通常是
            // `/model/encoder/layer.21/output/LayerNorm/LayerNormalization_output_0`；
            // 这里优先按名字找，找不到就取唯一输出（不猜层，猜错就是刺耳音色）。
            val hiddenName = pick(outputNames, "hidden_states_21", "layer21", "last_hidden_state") 
                ?: outputNames.singleOrNull() 
                ?: return null
            val vocab = loadVocab(vocabFile)
            if (vocab.isEmpty()) return null
            return AemeathBertFrontend(env, inputs, vocab, idName, maskName, typeName, hiddenName)
        }

        private fun pick(candidates: List<String>, vararg priority: String): String? {
            for (want in priority) {
                candidates.firstOrNull { it.equals(want, ignoreCase = true) }?.let { return it }
            }
            for (want in priority) {
                candidates.firstOrNull { it.lowercase().contains(want.lowercase()) }?.let { return it }
            }
            return null
        }
    }
}

/** 模型目录下的相对路径解析（拒绝 `..` 越界）。 */
internal fun resolveModelFile(root: File, relative: String): File? {
    if (relative.isBlank()) return null
    val normalized = relative.replace('\\', '/').trimStart('/')
    if (normalized.split('/').any { it == ".." }) return null
    val file = File(root, normalized)
    return if (file.isFile) file else null
}

