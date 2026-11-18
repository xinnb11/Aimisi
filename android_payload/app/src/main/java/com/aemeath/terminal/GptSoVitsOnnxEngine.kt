package com.aemeath.terminal

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.ByteOrder

/**
 * GptSoVitsOnnxEngine.kt — `arch === "GPT-SoVITS v2"` 的端侧推理引擎
 *
 * 本实现对齐 **`Aimisi_v2Pro_e8_FullPack.tar` 的真实张量签名**（已逐图核对）：
 *
 * | 阶段 | 输入 | 输出 |
 * | :--- | :--- | :--- |
 * | `t2s_encoder` | `ref_seq`(i64[1,N]) `text_seq`(i64[1,T]) `ref_bert`(f32[N,1024]) `text_bert`(f32[T,1024]) `ssl_content`(f32[1,768,S]) | `x`(f32[1,X,512]) `prompts`(i64[1,P]) |
 * | `t2s_fsdec` | `x` `prompts` | `y`(i64) `k` `v` `y_emb` `x_example` |
 * | `t2s_sdec` | `iy`(i64) `ik` `iv` `iy_emb` `ix_example` | `logits`(f32[·,1025]) `k` `v` `y_emb` **`samples`** |
 * | `vits` | `text_seq`(i64[1,T]) `pred_semantic`(i64[1,1,S]) `ref_audio`(f32[1,L]) | `audio`(f32) |
 *
 * ## 关键工程决策
 *
 * 1. **`samples` 必须丢弃**。实测该输出链路为
 *    `MatMul → Gather → ScatterElements → Div → Less/Where → Softmax → Div → Softmax → ArgMax → Cast → Unsqueeze`，
 *    即导出时已在图内做了「temperature + top-k + argmax」的贪婪解码。
 *    本引擎只取 `logits`，交由 [AemeathSampler] 做完整概率核采样后再喂回 `iy`，
 *    这样 AGENTS.md 红线 3 才真正生效（不改图也能拿到随机性）。
 * 2. **参考侧条件全部预计算**。`ref_seq` / `ssl_content` / `ref_bert` / `ref_audio`
 *    只与参考音有关，由 `scripts/prepare_ref_conditioning.py` 在云端算一次并固化成
 *    `conditioning/<情绪>*.f32|i64`；手机端因此**不跑 CNHubert / BERT**。
 * 3. `text_bert` 依赖逐句文本，端侧无 BERT 时按约定填零
 *    （`conditioning.json` 的 `bert_source` 会标注），音质略有折损但可正常发声。
 * 4. 文本 → 音素 id 走 `conditioning/g2p/char2phone.json`（离线字表），
 *    端侧不需要 jieba / g2pw。
 *
 * ## 与官方 Python 推理脚本的对齐约定（可调，无需改代码）
 *
 * `conditioning.json` 支持两个可选字段：
 * * `prompt_prefix_tokens`（默认 `-1` = 自动）：`fsdec.y` 前多少个 token 属于
 *   参考音 prompt。自动策略为「若 `y.size > ref_seq.size` 则丢弃前 `ref_seq.size` 个」。
 * * `eos_token` / `vocab_size`：语义码表大小与结束符。
 */
class GptSoVitsOnnxEngine : AemeathTtsEngine {

    override val engineId: String = "ONNX Runtime"

    private var ortEnv: OrtEnvironment? = null
    private var encoderSession: OrtSession? = null
    private var firstDecoderSession: OrtSession? = null
    private var secondDecoderSession: OrtSession? = null
    private var vocoderSession: OrtSession? = null

    private val mountedFiles = mutableListOf<File>()
    private val mountNotes = mutableListOf<String>()

    private var g2p: G2pTable? = null
    private val conditioning = HashMap<String, Conditioning>()
    private var samplingConfig: SamplingConfig = SamplingConfig()
    private var sampleRate: Int = 32000
    private var vocabSize: Int = 1025
    private var eosToken: Int = 1024
    private var accelerator: String = ""
    /** 每个阶段实际生效的 EP（m06856 第 4 条：分段计时必须能对上是哪个后端）。 */
    private var encoderEp: String = ""
    private var fsdecEp: String = ""
    private var sdecEp: String = ""
    private var vitsEp: String = ""
    private var bertEp: String = ""
    private var bertSource: String = "unknown"
    private var bertFrontend: AemeathBertFrontend? = null

    /** 图内 `ar_audio_embedding.word_embeddings.weight`（[vocab, 512]，条件包 `audio_embedding.f32`）。
     *  外部采样后必须用它算出与采样 token 一致的 `iy_emb`，否则 token 与向量错位 → 乱码。 */
    private var audioEmbedding: FloatArray? = null
    private var audioEmbeddingRows: Int = 0
    /** 是否用采样 token 覆盖图内 `y_emb`（A/B 开关，默认关闭＝与桌面参考实现一致）。 */
    private var syncYEmb: Boolean = false
    /** 是否启用「EOS 早停护栏」（默认关闭，护栏会改变采样分布）。 */
    private var eosGuardEnabled: Boolean = false
    /** 是否启用三声连读/一不变调。 */
    private var sandhiEnabled: Boolean = false
    /** 是否启用多音字词组覆盖。 */
    private var lexiconEnabled: Boolean = false
    /** 音素 symbol 的声调号（0=非带调韵母），用于三声连读 / 「一、不」变调。 */
    private var toneOf: IntArray? = null
    /** "symbolId:目标调" -> 同韵母另一调的 symbolId。 */
    private var toneVariants: Map<String, Int> = emptyMap()
    /** 高频多音字词组覆盖：词 -> 整词音素 id 串（按 char2phone 逐字切分）。 */
    private var lexicon: Map<String, IntArray> = emptyMap()

    private var encoderNames: StageNames? = null
    private var firstDecoderNames: StageNames? = null
    private var secondDecoderNames: StageNames? = null
    private var vocoderNames: StageNames? = null

    private var sharedSampler: AemeathSampler? = null
    private var sharedSamplerWidth: Int = 0
    private var lastSeed: Long = System.nanoTime()

    // ------------------------------------------------------------------ //
    // 装载
    // ------------------------------------------------------------------ //
    override fun mount(config: EngineMountConfig): EngineMountResult {
        close()
        mountNotes.clear()
        mountedFiles.clear()
        samplingConfig = config.sampling.sanitized()
        bertSource = "unknown"

        val dir = config.modelDir
        if (!dir.isDirectory) throw AemeathEngineException("模型目录不存在：${dir.absolutePath}")

        val env = try {
            OrtEnvironment.getEnvironment()
        } catch (error: Throwable) {
            throw AemeathEngineException("ONNX Runtime 初始化失败：${error.message}", error)
        }
        ortEnv = env

        // ---- 参考条件包 ----
        val condDir = listOf(File(dir, "conditioning"), dir)
            .firstOrNull { File(it, CONDITIONING_FILE).isFile }
            ?: throw AemeathEngineException(
                "缺少 conditioning/conditioning.json：请先运行 " +
                    "scripts/prepare_ref_conditioning.py 生成参考条件包"
            )
        val condManifest = JSONObject(File(condDir, CONDITIONING_FILE).readText())
        vocabSize = condManifest.optInt("vocab_size", 1025)
        eosToken = condManifest.optInt("eos_token", 1024)
        sampleRate = condManifest.optInt("sample_rate", 32000)
        bertSource = condManifest.optString("bert_source", "unknown")
        // 两个实验开关：默认关（与桌面参考实现 scripts/verify_onnx_tts.py 完全一致）。
        syncYEmb = condManifest.optBoolean("sync_y_emb", false)
        eosGuardEnabled = condManifest.optBoolean("eos_guard", false)
        // G2P 增强（变调 / 多音字词表）：默认关，逐个打开做 A/B，确认不会引入错音。
        sandhiEnabled = condManifest.optBoolean("tone_sandhi", false)
        lexiconEnabled = condManifest.optBoolean("lexicon_override", false)
        mountedFiles += File(condDir, CONDITIONING_FILE)

        loadConditioning(condDir, condManifest)
        loadG2p(condDir, condManifest)

        // ---- 四个 ONNX 会话 ----
        val encoderFile = findStageFile(dir, "t2s_encoder", preferQuantized = true)
            ?: throw AemeathEngineException("缺少 t2s_encoder.onnx / t2s_encoder.int8.onnx")
        val firstDecoderFile = findStageFile(dir, "t2s_fsdec", preferQuantized = true)
            ?: throw AemeathEngineException("缺少 t2s_fsdec.onnx / t2s_fsdec.int8.onnx")
        val secondDecoderFile = findStageFile(dir, "t2s_sdec", preferQuantized = true)
            ?: throw AemeathEngineException("缺少 t2s_sdec.onnx / t2s_sdec.int8.onnx")
        val vocoderFile = findStageFile(dir, "vits", preferQuantized = false)
            ?: findStageFile(dir, "vocoder", preferQuantized = false)
            ?: throw AemeathEngineException("缺少 vits.onnx / vits.fp16.onnx（声码器）")

        // ---- 五个 ONNX 会话：逐模型独立选后端（m06856 / m07045 天玑异构分工）----
        // 芯片是联发科天玑 9500（无 Hexagon，QNN 代码已全部移除）：
        //   · vits：静态大矩阵、一次前向 → 允许 NNAPI(FP16) 上天玑 APU；失败退 XNNPACK/CPU。
        //   · sdec：逐 step 动态自回归 → **只准 CPU NEON**（进 NNAPI 会让 APU HAL 死锁，
        //     且 XNNPACK 在该图上数值错误，§14.3 A/B 已证）。
        //   · encoder / fsdec / bert：小图单次前向 → XNNPACK → CPU。
        val encoderBundle = AemeathOrtBackend.openStage(
            env, encoderFile, "t2s_encoder", config.backend, THREADS,
            allowNnapi = false, allowXnnpack = true
        )
        encoderSession = encoderBundle.session
        val firstBundle = AemeathOrtBackend.openStage(
            env, firstDecoderFile, "t2s_fsdec", config.backend, THREADS,
            allowNnapi = false, allowXnnpack = true
        )
        firstDecoderSession = firstBundle.session
        val secondBundle = AemeathOrtBackend.openStage(
            env, secondDecoderFile, "t2s_sdec", config.backend, THREADS,
            allowNnapi = false, allowXnnpack = false
        )
        secondDecoderSession = secondBundle.session
        val vocoderBundle = AemeathOrtBackend.openStage(
            env, vocoderFile, "vits", config.backend, THREADS,
            allowNnapi = true, allowXnnpack = true,
            probe = { session -> probeVocoder(env, session, condDir) },
        )
        vocoderSession = vocoderBundle.session

        accelerator = vocoderBundle.accelerator
        encoderEp = encoderBundle.accelerator
        fsdecEp = firstBundle.accelerator
        sdecEp = secondBundle.accelerator
        vitsEp = vocoderBundle.accelerator
        mountNotes += "后端（逐模型独立）：enc=${encoderBundle.accelerator}；" +
            "fsdec=${firstBundle.accelerator}；sdec=${secondBundle.accelerator}；vits=${vocoderBundle.accelerator}"
        mountNotes += encoderBundle.notes + firstBundle.notes + secondBundle.notes + vocoderBundle.notes
        mountedFiles += listOf(encoderFile, firstDecoderFile, secondDecoderFile, vocoderFile)

        // ---- 端侧 BERT 文本前端（决定 text_bert，缺它会立刻 EOS 近乎无声）----
        loadBert(dir, condManifest, config.backend)
        if (bertFrontend != null) {
            mountNotes += "端侧 BERT 已装载：text_bert 逐句实时计算（char2phone 分组 × 1024 维均值展开）"
        } else {
            mountNotes += "缺少 bert/model.int8.onnx 或 bert/vocab.txt：text_bert 退化为零向量，" +
                "模型会立刻 EOS（几乎无声），请重新拉取完整模型包"
        }

        // ---- 张量名解析（缺名即报错，不猜）----
        encoderNames = StageNames.resolve(
            "t2s_encoder", encoderBundle.session,
            listOf("ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"),
            listOf("x", "prompts"),
        )
        firstDecoderNames = StageNames.resolve(
            "t2s_fsdec", firstBundle.session,
            listOf("x", "prompts"),
            listOf("y", "k", "v", "y_emb", "x_example"),
        )
        secondDecoderNames = StageNames.resolve(
            "t2s_sdec", secondBundle.session,
            listOf("iy", "ik", "iv", "iy_emb", "ix_example"),
            listOf("logits", "k", "v", "y_emb", "y"),
        )
        vocoderNames = StageNames.resolve(
            "vits", vocoderBundle.session,
            listOf("text_seq", "pred_semantic", "ref_audio"),
            listOf("audio"),
        )

        if (bertFrontend == null) {
            mountNotes += "ref_bert 取自条件包（${conditioning.size} 个情绪），text_bert 不可用"
        }
        mountNotes += "已启用端侧概率核采样（top_k=${samplingConfig.topK}, T=${samplingConfig.temperature}）；" +
            "图内 argmax 的 samples 输出已弃用"

        Log.i(
            TAG,
            "GPT-SoVITS 装载完成：enc=${encoderFile.name} fsdec=${firstDecoderFile.name} " +
                "sdec=${secondDecoderFile.name} vits=${vocoderFile.name} @ $accelerator"
        )

        return EngineMountResult(
            success = true,
            modelId = config.modelId,
            engine = engineId,
            sampleRate = sampleRate,
            channels = 1,
            accelerationActive = accelerator,
            memoryFootprintMb = memoryFootprintMb(),
            notes = mountNotes.toList(),
        )
    }

    // ------------------------------------------------------------------ //
    // 合成
    // ------------------------------------------------------------------ //
    override fun synthesize(
        text: String,
        emotion: Emotion,
        speed: Float,
        volume: Float,
    ): SynthesisOutput {
        val env = ortEnv ?: throw AemeathEngineException("ONNX Runtime 尚未初始化")
        val encoder = encoderSession ?: throw AemeathEngineException("t2s_encoder 尚未装载")
        val fsdec = firstDecoderSession ?: throw AemeathEngineException("t2s_fsdec 尚未装载")
        val sdec = secondDecoderSession ?: throw AemeathEngineException("t2s_sdec 尚未装载")
        val vocoder = vocoderSession ?: throw AemeathEngineException("vits 尚未装载")
        val encNames = encoderNames ?: throw AemeathEngineException("encoder 张量名未解析")
        val fsNames = firstDecoderNames ?: throw AemeathEngineException("fsdec 张量名未解析")
        val sdNames = secondDecoderNames ?: throw AemeathEngineException("sdec 张量名未解析")
        val voNames = vocoderNames ?: throw AemeathEngineException("vits 张量名未解析")

        val normalized = text.trim()
        if (normalized.isEmpty()) throw AemeathEngineException("待合成文本为空")

        val condition = conditioning[emotion.key] ?: conditioning[Emotion.GENTLE.key]
            ?: throw AemeathEngineException("缺少「${emotion.key}」参考条件包")
        val sslContent = condition.sslContent
            ?: throw AemeathEngineException(
                "「${emotion.key}」条件包缺少 ssl_content（logs 模式常见），" +
                    "请用 prepare_ref_conditioning.py --mode repo 重建"
            )
        val sslShape = condition.sslShape ?: longArrayOf(1, 768, 1)

        val tBert0 = android.os.SystemClock.elapsedRealtime()
        val (textIds, textBert) = buildTextInputs(normalized)
        val bertMs = android.os.SystemClock.elapsedRealtime() - tBert0
        if (textIds.isEmpty()) throw AemeathEngineException("文本转音素结果为空：$normalized")
        val textLength = textIds.size
        if (textBert.size != textLength * BERT_DIM) {
            throw AemeathEngineException(
                "text_bert 尺寸异常：${textBert.size}（期望 ${textLength * BERT_DIM}，音素 $textLength 个）"
            )
        }

        var tbNorm = 0.0
        for (v in textBert) tbNorm += v.toDouble() * v
        Log.d(TAG, "text_bert: T=" + textLength + " norm=" + "%.3f".format(kotlin.math.sqrt(tbNorm)))

        // 1) 语义编码
        val tEnc0 = android.os.SystemClock.elapsedRealtime()
        val encoderOut = FeedBuilder().use { feed ->
            feed.putLong(
                env, encNames.input("ref_seq"), condition.refSeq,
                longArrayOf(1, condition.refSeq.size.toLong())
            )
            feed.putLong(env, encNames.input("text_seq"), textIds, longArrayOf(1, textLength.toLong()))
            feed.putFloat(env, encNames.input("ref_bert"), condition.refBertOrZeros(), condition.refBertShapeOrZeros())
            // text_bert 由端侧 BERT 逐句实时计算（缺权重时为零向量，模型会立刻 EOS）
            feed.putFloat(
                env, encNames.input("text_bert"),
                textBert,
                longArrayOf(textLength.toLong(), BERT_DIM.toLong()),
            )
            feed.putFloat(env, encNames.input("ssl_content"), sslContent, sslShape)
            feed.runTensors(encoder)
        }

        val hiddenX = encoderOut.floats(encNames.output("x"))
            ?: throw AemeathEngineException("encoder 未输出 x")
        val encMs = android.os.SystemClock.elapsedRealtime() - tEnc0
        val prompts = encoderOut.longs(encNames.output("prompts"))
            ?: throw AemeathEngineException("encoder 未输出 prompts")
        val hiddenShape = encoderOut.shapeOf(encNames.output("x"))
        val promptShape = encoderOut.shapeOf(encNames.output("prompts"))
        Log.i(TAG, "encoder: x=${hiddenShape.toList()} prompts=${promptShape.toList()}")
        encoderOut.close()

        // 2) 一阶段解码（非自回归前缀预测；官方管线同样直接使用其输出）
        // 一阶段输出要一直开到自回归循环结束：`k` / `v` / `y_emb` / `x_example` 都以
        // OnnxTensor 对象引用的形式直接喂给下一步，只有关闭 RunOutputs 才会释放它们的 native 内存。
        val firstOut = FeedBuilder().use { feed ->
            feed.putFloat(env, fsNames.input("x"), hiddenX, hiddenShape)
            feed.putLong(env, fsNames.input("prompts"), prompts, promptShape)
            feed.runTensors(fsdec)
        }
        val fsdecMs = android.os.SystemClock.elapsedRealtime() - tEnc0 - encMs

        val rawPrefix = firstOut.longs(fsNames.output("y"))
            ?: throw AemeathEngineException("fsdec 未输出 y")
        var cacheK = firstOut.tensor(fsNames.output("k"))
            ?: throw AemeathEngineException("fsdec 未输出 k")
        var cacheV = firstOut.tensor(fsNames.output("v"))
            ?: throw AemeathEngineException("fsdec 未输出 v")
        val xExampleTensor = firstOut.tensor(fsNames.output("x_example"))
            ?: throw AemeathEngineException("fsdec 未输出 x_example")
        // 默认沿用图输出的 y_emb；只有打开 sync_y_emb 时才新建一份并替换末行（必须拷贝）。
        var yEmbTensor = firstOut.tensor(fsNames.output("y_emb"))
            ?: throw AemeathEngineException("fsdec 未输出 y_emb")
        var yEmbOwned: OnnxTensor? = null


        val tSdec0 = android.os.SystemClock.elapsedRealtime()
        // 3) 自回归采样循环：只吃 logits，丢弃图内 argmax 的 samples
        //
        // 张量对齐约定（已用真实权重逐项验证，写死不得随意改）：
        //   · 第 N 步：iy = 上一轮完整 y（长度 = tokens+1），iy_emb = 上一轮完整 y_emb（长度 = tokens）
        //   · ik/iv 缓存随步增长，ix_example 恒为 fsdec 输出的 x_example
        //   · y 的前 prompts.size 个 token 是参考音语义码，不计入 pred_semantic
        val generated = ArrayList<Int>(minOf(samplingConfig.maxSemanticLength, 512))
        val promptCount = prompts.size
        // 参考音语义码集合：短句时模型容易「复读参考音」（真机听到开头的「也有一些危险刺激」），
        // 因此在开头若干步内对这些 token 做软惩罚，并在生成后剥离与 prompts 重叠的前缀。
        val promptTokenIds = HashSet<Int>()
        for (i in 0 until promptCount) {
            val id = prompts[i].toInt()
            if (id in 0 until vocabSize && id != eosToken) promptTokenIds.add(id)
        }
        val echoPenaltySteps = 40
        val echoPenalty = 2.5f
        // 「禁止复读参考音」的整词级约束（m07150）：
        // 只压「prompt 里出现过的 token」还不够——模型会换一种排列把参考音重排出来。
        // 这里照搬 GPT 系常见的 no-repeat-ngram 做法：把 prompts 里所有 5-gram 的
        // 「前 4 码 → 第 5 码」记下来，生成时只要当前尾部 4 码命中某个 5-gram 前缀，
        // 就把那个第 5 码重罚（-12），从根上断开「照抄参考音」这条路径。
        val promptBanMap = HashMap<Long, HashSet<Int>>()
        if (promptCount >= ECHO_GRAM) {
            for (i in 0..promptCount - ECHO_GRAM) {
                val key = nGramKeyOf(prompts, i, ECHO_GRAM - 1)
                var set = promptBanMap[key]
                if (set == null) {
                    set = HashSet(2)
                    promptBanMap[key] = set
                }
                set.add(prompts[i + ECHO_GRAM - 1].toInt())
            }
        }
        val echoBanPenalty = 12.0f
        var echoBanHits = 0
        rawPrefix.drop(promptCount).forEach { token ->
            val value = token.toInt()
            if (value in 0 until vocabSize && value != eosToken) {
                generated += value
                rememberToken(value)
            }
        }

        // 分句流水线（m05844 第 1 条）：单句内**只跑一次** vits。
        // 实测「每 32 码重跑一次 vits」会把 28 字句子的总耗时从 31 s 拉到 44.7 s，
        // 且听感是「说 0.6 s、死寂 9 s」，远不如「每句一次 vits + 原生异步播放」。
        // 句与句之间的并发靠 AemeathAudioPlayer 的播放队列实现（播上一句时 CPU 已在算下一句）。
        var currentY = rawPrefix
        var steps = 0
        var stopped = false

        var prevStepOut: RunOutputs? = null
        try {
        while (steps < samplingConfig.maxSemanticLength) {
            steps += 1
            val stepOut = FeedBuilder().use { feed ->
                feed.putLong(
                    env, sdNames.input("iy"), currentY,
                    longArrayOf(1, currentY.size.toLong())
                )
                // KV cache / y_emb / x_example 一律传上一轮 run() 返回的 OnnxTensor 引用，
                // 绝不拷回 JVM FloatArray 再重建张量（140 步 × ~15 MB 的 JNI 拷贝，m05844 第 2 条）。
                feed.putTensor(sdNames.input("ik"), cacheK)
                feed.putTensor(sdNames.input("iv"), cacheV)
                feed.putTensor(sdNames.input("iy_emb"), yEmbTensor)
                feed.putTensor(sdNames.input("ix_example"), xExampleTensor)
                feed.runTensors(sdec)
            }
            // 上一轮输出张量已在上面这次 run 里被当作输入消费完，可以安全释放。
            prevStepOut?.close()
            prevStepOut = stepOut

            val logits = stepOut.floats(sdNames.output("logits"))
                ?: throw AemeathEngineException("sdec 未输出 logits")
            val logitsShape = stepOut.shapeOf(sdNames.output("logits"))
            val width = logitsShape.lastOrNull()?.toInt() ?: logits.size
            if (logits.size != width) {
                throw AemeathEngineException(
                    "sdec logits 形状异常：${logitsShape.toList()}（元素 ${logits.size}）"
                )
            }

            // 早停护栏（默认关闭）：桌面参考实现（scripts/verify_onnx_tts.py）不加护栏也能
            // 生成完整句子，护栏会改变采样分布，只有在确认「真机系统性提前 EOS」时才启用。
            if (eosGuardEnabled) {
                val minCodesBeforeEos = (textLength * 2.2f).toInt().coerceAtLeast(24)
                if (generated.size < minCodesBeforeEos) {
                    logits[eosToken] = -1.0e9f
                }
            }
            if (steps == 1) {
                val top = logits.indices.sortedByDescending { logits[it] }.take(5)
                val cacheShape = (cacheK.info as? TensorInfo)?.shape?.toList()
                Log.d(
                    TAG,
                    "step=" + steps + " top5=" + top.joinToString(",") { it.toString() + ":" + "%.3f".format(logits[it]) } +
                        " iy=" + currentY.size + " k=" + cacheShape
                )
            }
            // 复读参考音抑制（只在前 24 码）：把「参考音语义码集合」里的 token 软压低，
            // 避免模型一开口就照抄参考音内容（真机实测开头常出现参考音最后几个字）。
            if (steps <= echoPenaltySteps && promptTokenIds.isNotEmpty()) {
                var hits = 0
                for (id in promptTokenIds) {
                    if (id < logits.size) {
                        logits[id] -= echoPenalty
                        hits += 1
                    }
                }
                if (steps == 1) Log.d(TAG, "prompt-echo 软惩罚启用：候选 $hits 个 token，-${echoPenalty}")
            }
            // 5-gram 复读硬约束（不必只在开头，任意位置命中都罚）：只要当前尾部 4 码
            // 和 prompts 里某个 5-gram 的前 4 码一致，就把该 5-gram 的第 5 码压掉。
            if (promptBanMap.isNotEmpty() && generated.size >= ECHO_GRAM - 1) {
                val key = nGramKeyOf(generated, generated.size - (ECHO_GRAM - 1), ECHO_GRAM - 1)
                val banned = promptBanMap[key]
                if (banned != null) {
                    var applied = 0
                    for (id in banned) {
                        if (id in 0 until logits.size) {
                            logits[id] -= echoBanPenalty
                            applied += 1
                        }
                    }
                    echoBanHits += applied
                    if (steps == 1) {
                        Log.d(
                            TAG,
                            "prompt-echo 5-gram 约束启用：${promptBanMap.size} 条前缀，本步压制 $applied 个候选（-$echoBanPenalty）"
                        )
                    }
                }
            }
            val sampledToken = sampleSmart(logits, width)
            cacheK = stepOut.tensor(sdNames.output("k")) ?: cacheK
            cacheV = stepOut.tensor(sdNames.output("v")) ?: cacheV
            // 关键：图内 y_emb 是「图自己 argmax 的 token」的嵌入。桌面参考实现沿用图输出即可，
            // 因此默认不覆盖；只有显式打开 sync_y_emb 时才把末行换成采样 token 的嵌入做 A/B。
            stepOut.tensor(sdNames.output("y_emb"))?.let { fresh ->
                val table = audioEmbedding
                val shape = (fresh.info as? TensorInfo)?.shape
                val widthEmb = shape?.lastOrNull()?.toInt() ?: AUDIO_EMB_DIM
                val shouldPatch = syncYEmb && table != null && widthEmb == AUDIO_EMB_DIM &&
                    sampledToken >= 0 && (sampledToken + 1) * AUDIO_EMB_DIM <= table.size
                if (shouldPatch) {
                    val buffer = fresh.floatBuffer.duplicate()
                    buffer.rewind()
                    val data = FloatArray(buffer.remaining())
                    buffer.get(data)
                    if (data.size >= AUDIO_EMB_DIM) {
                        System.arraycopy(
                            table, sampledToken * AUDIO_EMB_DIM,
                            data, data.size - AUDIO_EMB_DIM, AUDIO_EMB_DIM
                        )
                        yEmbOwned?.close()
                        yEmbOwned = OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(data),
                            shape ?: longArrayOf(1, (data.size / AUDIO_EMB_DIM).toLong(), AUDIO_EMB_DIM.toLong())
                        )
                        yEmbTensor = yEmbOwned!!
                    } else {
                        yEmbTensor = fresh
                    }
                } else {
                    yEmbTensor = fresh
                }
            }

            if (sampledToken == eosToken) {
                Log.i(TAG, "sdec 命中 EOS($eosToken)，步数=$steps")
                stopped = true
                break
            }
            generated += sampledToken
            rememberToken(sampledToken)
            // 与桌面参考实现（scripts/verify_onnx_tts.py）完全对齐：
            // 下一步的 iy 用「图输出的完整 y」，并把末位替换成运行时采样到的 token。
            // 不要自己往输入序列尾部追加——图输出的 y 才是与 KV cache 同步的那份。
            val graphY = stepOut.longs(sdNames.output("y"))
                ?: throw AemeathEngineException("sdec 未输出 y")
            graphY[graphY.size - 1] = sampledToken.toLong()
            currentY = graphY
        }
        } finally {
            prevStepOut?.close()
            yEmbOwned?.close()
            firstOut.close()
        }
        if (!stopped && generated.size > 1) {
            Log.w(TAG, "sdec 达到步数上限 ${samplingConfig.maxSemanticLength}，按已生成序列收尾")
        }

        if (generated.size <= 1) {
            throw AemeathEngineException(
                "自回归未生成有效语义码（prefix=${rawPrefix.size}, prompts=$promptCount, steps=$steps）。" +
                    "请检查参考条件包的 ref_seq 是否为参考音真实台词音素（ref_seq 过短会导致提前 EOS）"
            )
        }

        // 4) 声码器
        //
        // 严禁对 vits 输出做「线性插值变速」：它会同时改变基频（Pitch）与共振峰，
        // 直接把爱弥斯的音色变成花栗鼠或粗嗓（m04444 第 4 条）。情绪在没有独立参考音频前
        // 统一 1.0x 原速原采样率直出，只做统一的峰值归一（0.89）与 5ms 淡入淡出。
        // 剥离「复读参考音」：先按前缀连续命中剥，再扫任意位置的 5-gram 窗口，
        // 命中参考音就整窗删掉（真机表现是句子中间插进「也有一些危险刺激」这种参考音片段）。
        if (promptCount >= ECHO_GRAM && generated.size > ECHO_GRAM + 8) {
            val promptGrams = HashSet<Long>()
            for (i in 0..promptCount - ECHO_GRAM) {
                promptGrams.add(gramHash(prompts, i))
            }
            var stripped = 0
            while (stripped < 40 && generated.size - stripped > ECHO_GRAM + 8) {
                if (!promptGrams.contains(gramHash(generated, stripped))) break
                stripped += 1
            }
            if (stripped > 0) {
                repeat(stripped) { generated.removeAt(0) }
                Log.i(TAG, "prompt-echo 已剥离前缀 $stripped 码（剩余 ${generated.size}）")
            }
            // 任意位置窗口清理：从尾往头删，避免索引漂移；至少保留 9 码。
            var removedWindow = 0
            var index = generated.size - ECHO_GRAM
            while (index >= 0) {
                if (generated.size - ECHO_GRAM < 9) break
                if (promptGrams.contains(gramHash(generated, index))) {
                    repeat(ECHO_GRAM) { generated.removeAt(index) }
                    removedWindow += ECHO_GRAM
                }
                index -= 1
            }
            if (removedWindow > 0) {
                Log.i(TAG, "prompt-echo 已清除参考音片段 $removedWindow 码（剩余 ${generated.size}）")
            }
            if (echoBanHits > 0) {
                Log.i(TAG, "prompt-echo 运行期累计压制 $echoBanHits 个复读候选")
            }
        }

        if (condition.stripPrefixCodes > 0 && generated.size > condition.stripPrefixCodes + 8) {
            repeat(condition.stripPrefixCodes) { generated.removeAt(0) }
            Log.i(TAG, "已按条件包配置剥离参考音前缀 ${condition.stripPrefixCodes} 码（剩余 ${generated.size}）")
        }

        val codes = LongArray(generated.size) { generated[it].toLong() }
        Log.d(
            TAG,
            "codes=" + generated.size + " first8=" + generated.take(8).joinToString(",") +
                " last4=" + generated.takeLast(4).joinToString(",")
        )
        val sdecMs = android.os.SystemClock.elapsedRealtime() - tSdec0
        val tVits0 = android.os.SystemClock.elapsedRealtime()
        val audio = runVocoder(env, vocoder, voNames, textIds, textLength, codes, condition)
        val vitsMs = android.os.SystemClock.elapsedRealtime() - tVits0

        // 该模型图内没有情绪条件输入：在没有独立参考音频前，情绪不做任何波形级加工
        // （变速会改基频/共振峰，就是「音色不像」的直接原因）。
        var samples = AemeathPcmPostProcess.normalize(audio, targetPeak = 0.89f)
        samples = AemeathPcmPostProcess.fade(samples, sampleRate, 5f)
        if (volume < 0.999f) {
            val gain = volume.coerceIn(0f, 1f)
            samples = FloatArray(samples.size) { (samples[it] * gain).coerceIn(-1f, 1f) }
        }
        val perStep = if (steps > 0) sdecMs.toDouble() / steps else 0.0
        Log.i(
            TAG,
            "分段计时：bert=${bertMs}ms enc=${encMs}ms fsdec=${fsdecMs}ms " +
                "sdec=${sdecMs}ms($steps 步, " + "%.1f".format(perStep) + "ms/步) vits=${vitsMs}ms " +
                "| EP enc=$encoderEp fsdec=$fsdecEp sdec=$sdecEp vits=$vitsEp bert=$bertEp"
        )
        Log.i(TAG, "合成完成：${normalized.length} 字 -> ${codes.size} codes，${samples.size} 采样点，总耗时 ${sdecMs + vitsMs + encMs + fsdecMs + bertMs}ms")
        return SynthesisOutput(samples, sampleRate)
    }

    private data class EmotionProfile(val speed: Float, val gain: Float)

    private fun emotionProfile(emotion: Emotion): EmotionProfile = when (emotion) {
        Emotion.CALM -> EmotionProfile(1.00f, 1.00f)
        Emotion.HAPPY -> EmotionProfile(1.12f, 1.05f)
        Emotion.GENTLE -> EmotionProfile(0.95f, 0.95f)
        Emotion.TEASE -> EmotionProfile(1.06f, 1.00f)
        Emotion.ANGRY -> EmotionProfile(1.05f, 1.05f)
    }

    // ------------------------------------------------------------------ //
    // 采样与分词
    // ------------------------------------------------------------------ //
    private fun rememberToken(token: Int) {
        if (sharedSampler == null) {
            sharedSampler = AemeathSampler(vocabSize, samplingConfig, lastSeed)
            sharedSamplerWidth = vocabSize
            sharedSampler?.reset()
        }
        sharedSampler?.remember(token)
    }

    /** 复用同一采样器实例，让重复惩罚窗口跨步生效；logits 宽度变化时才重建。 */
    /** 语义码 n-gram 哈希（LongArray 版，用于参考音 prompts）。 */
    private fun gramHash(data: LongArray, start: Int): Long {
        var h = 0L
        for (i in 0 until ECHO_GRAM) h = h * 1000003L + (data[start + i] + 1L)
        return h
    }

    /** 语义码 n-gram 哈希（List<Int> 版，用于本次生成结果）。 */
    private fun gramHash(data: List<Int>, start: Int): Long {
        var h = 0L
        for (i in 0 until ECHO_GRAM) h = h * 1000003L + (data[start + i] + 1L)
        return h
    }

    /** 任意长度 n-gram 的键（用于「前 4 码 → 第 5 码」的复读约束表）。 */
    private fun nGramKeyOf(data: LongArray, start: Int, length: Int): Long {
        var h = 0L
        for (i in 0 until length) h = h * 1000003L + (data[start + i] + 1L)
        return h
    }

    private fun nGramKeyOf(data: List<Int>, start: Int, length: Int): Long {
        var h = 0L
        for (i in 0 until length) h = h * 1000003L + (data[start + i] + 1L)
        return h
    }

    private fun sampleSmart(logits: FloatArray, width: Int): Int {
        if (sharedSampler == null || sharedSamplerWidth != width) {
            sharedSampler = AemeathSampler(width, samplingConfig, lastSeed)
            sharedSamplerWidth = width
            sharedSampler?.reset()
            lastSeed = lastSeed xor (System.nanoTime() shl 1)
        }
        return sharedSampler!!.sample(logits)
    }

    private fun loadBert(dir: File, manifest: JSONObject, backend: ComputeBackend) {
        val env = ortEnv ?: return
        // GPT-SoVITS 官方用 hidden_states[-3]（第 21 层），不是 last_hidden_state。
        // 发布包里两个文件都在：优先用「第 21 层单独导出」的那张，缺失才退回整模型。
        val layerRel = manifest.optString("bert_model_layer21", "bert/model.layer21.int8.onnx")
        val modelRel = manifest.optString("bert_model", "bert/model.int8.onnx")
        val modelFile = resolveModelFile(dir, layerRel)
            ?: resolveModelFile(dir, modelRel)
            ?: run {
                Log.w(TAG, "BERT 权重不存在：$layerRel / $modelRel")
                return
            }
        val usingLayer21 = modelFile.name.contains("layer21")
        bertSource = if (usingLayer21) "layer21" else "model"
        val vocabRel = manifest.optJSONObject("bert")?.optString("vocab") ?: "bert/vocab.txt"
        val vocabFile = resolveModelFile(dir, vocabRel) ?: run {
            Log.w(TAG, "BERT 词表不存在：$vocabRel")
            return
        }
        // BERT 也走独立路由（m06856）：QNN/NNAPI → XNNPACK → CPU 逐个试，失败了只影响 BERT 自己。
        val frontend = try {
            AemeathBertFrontend.open(
                env, modelFile, vocabFile, backend, THREADS,
                onEp = { bertEp = it },
            )
        } catch (error: Throwable) {
            Log.w(TAG, "BERT 会话创建失败（按缺省处理）", error)
            null
        }
        if (frontend == null) {
            Log.w(TAG, "BERT 张量名不匹配或词表为空，text_bert 退化为零向量")
            return
        }
        bertFrontend = frontend
        mountedFiles += modelFile
        mountedFiles += vocabFile
        // 装载即自检：把张量名与一次极短前向的结果写进 notes，出错时前端直接能看到原因。
        val smoke = runCatching { frontend.encode("你", intArrayOf(1)) }
        mountNotes += if (smoke.isSuccess) {
            "BERT 自检通过（${smoke.getOrThrow().size} 维，来源=${if (usingLayer21) "第21层 hidden_states[-3]" else "整模型输出"}）；${frontend.ioSummary()}"
        } else {
            "BERT 自检失败：${smoke.exceptionOrNull()?.message}；${frontend.ioSummary()}"
        }
    }

    /**
     * 文本 -> (`text_seq` 音素 id, `text_bert` [T,1024])。
     *
     * `text_bert` 必须逐行对齐音素：汉字在 BERT 里通常 1 个 token，在音素表里 2 个
     * （声母/韵母+声调），所以要把每个字的 1024 维向量按它的音素个数重复。
     */
    private fun buildTextInputs(text: String): Pair<LongArray, FloatArray> {
        val table = g2p ?: return Pair(LongArray(0), FloatArray(0))
        val chars = StringBuilder(text.length)
        val perChar = ArrayList<IntArray>(text.length)
        var index = 0
        while (index < text.length) {
            val char = text[index]
            if (char.isWhitespace()) {
                index += 1
                continue
            }
            // 多音字词组优先（最长 4 字匹配）：漂泊者 / 爱弥斯 / 什么 / 快乐 / 银行 …
            var matched = false
            if (lexiconEnabled && lexicon.isNotEmpty()) {
                val limit = minOf(MAX_LEXICON_LEN, text.length - index)
                var length = limit
                while (length >= 2) {
                    val word = text.substring(index, index + length)
                    val ids = lexicon[word]
                    if (ids != null) {
                        // 词表存的是整词的音素 id 串；每字切几个由 char2phone 决定，
                        // 只有「切分数正好等于总长度」才算可用（防长度不一致时错位）。
                        val counts = IntArray(length) { table.charToPhones[word[it]]?.size ?: 0 }
                        if (counts.all { it > 0 } && counts.sum() == ids.size) {
                            var offset = 0
                            for (offsetChar in 0 until length) {
                                perChar += ids.copyOfRange(offset, offset + counts[offsetChar])
                                offset += counts[offsetChar]
                                chars.append(word[offsetChar])
                            }
                            index += length
                            matched = true
                            break
                        }
                    }
                    length -= 1
                }
            }
            if (matched) continue
            val phones = table.charToPhones[char]
                ?: if (char.isLetterOrDigit()) intArrayOf(table.unknownId) else IntArray(0)
            chars.append(char)
            perChar += phones
            index += 1
            if (perChar.sumOf { it.size } >= MAX_TEXT_PHONES) break
        }

        if (sandhiEnabled) applyToneSandhi(chars, perChar)

        val phoneIds = ArrayList<Long>(perChar.sumOf { it.size })
        val counts = IntArray(perChar.size)
        for (i in perChar.indices) {
            counts[i] = perChar[i].size
            perChar[i].forEach { phoneIds += it.toLong() }
        }
        val features = bertFrontend?.let { frontend ->
            if (phoneIds.isEmpty()) FloatArray(0) else frontend.encode(chars.toString(), counts)
        } ?: FloatArray(phoneIds.size * BERT_DIM)
        Log.d(
            TAG,
            "g2p: chars=${chars} phones=${phoneIds.take(24)} counts=${counts.take(24)} len=${phoneIds.size}"
        )
        return Pair(phoneIds.toLongArray(), features)
    }

    /**
     * 中文连读变调（端侧最小可用版）：
     * - 三声 + 三声 → 前一个字变二声（「你好」n i3 h ao3 → n i2 h ao3）；
     * - 「一」在四声前变二声、其余变四声；「不」在四声前变二声。
     * 声调号来自 `g2p/tone_table.json`，同韵母换调的 symbol id 也在那张表里。
     */
    private fun applyToneSandhi(chars: CharSequence, phones: MutableList<IntArray>) {
        val tones = toneOf ?: return
        if (toneVariants.isEmpty() || phones.size < 2) return
        fun finalId(ids: IntArray): Int = if (ids.isEmpty()) -1 else ids[ids.size - 1]
        fun toneOfId(id: Int): Int = if (id in tones.indices) tones[id] else 0

        for (i in phones.indices) {
            val ids = phones[i]
            val last = finalId(ids)
            if (last < 0) continue
            val tone = toneOfId(last)
            if (tone == 0) continue
            val nextTone = if (i + 1 < phones.size) toneOfId(finalId(phones[i + 1])) else 0
            val target = when (chars[i]) {
                '一' -> if (nextTone == 4) 2 else 4
                '不' -> if (nextTone == 4) 2 else 0
                else -> if (tone == 3 && nextTone == 3) 2 else 0
            }
            if (target == 0 || target == tone) continue
            val alt = toneVariants["$last:$target"] ?: continue
            val replaced = ids.copyOf()
            replaced[replaced.size - 1] = alt
            phones[i] = replaced
        }
    }

    // ------------------------------------------------------------------ //
    // 条件包 / G2P 装载
    // ------------------------------------------------------------------ //
    private fun loadConditioning(condDir: File, manifest: JSONObject) {
        val emotions = manifest.optJSONObject("emotions")
            ?: throw AemeathEngineException("conditioning.json 缺少 emotions 字段")
        for (emotion in Emotion.ALL) {
            val entry = emotions.optJSONObject(emotion.key) ?: continue
            val refSeq = Blob.tryLoad(condDir, emotion.key, entry, "ref_seq") ?: continue
            val refAudio = Blob.tryLoad(condDir, emotion.key, entry, "ref_audio") ?: continue
            val ssl = Blob.tryLoad(condDir, emotion.key, entry, "ssl_content")
            val refBert = Blob.tryLoad(condDir, emotion.key, entry, "ref_bert")
            val stripPrefix = entry.optInt("strip_prefix_codes", 0)
            conditioning[emotion.key] = Conditioning(
                refSeq = refSeq.longs ?: LongArray(0),
                refAudio = refAudio.floats ?: FloatArray(0),
                refAudioShape = refAudio.shape,
                sslContent = ssl?.floats,
                sslShape = ssl?.shape,
                refBert = refBert?.floats,
                refBertShape = refBert?.shape,
                stripPrefixCodes = stripPrefix,
            )
            if (ssl == null) mountNotes += "「${emotion.key}」缺少 ssl_content"
        }
        if (conditioning.isEmpty()) {
            throw AemeathEngineException("conditioning.json 中没有任何可用情绪条目")
        }

        // 图内音频嵌入表（外部采样后必须用它同步 iy_emb）
        val embRel = manifest.optJSONObject("audio_embedding")?.optString("file") ?: "audio_embedding.f32"
        val embFile = File(condDir, embRel)
        if (embFile.isFile) {
            val floats = Blob.tryLoadBare(embFile)
            if (floats != null && floats.isNotEmpty() && floats.size % 512 == 0) {
                audioEmbedding = floats
                audioEmbeddingRows = floats.size / 512
                mountedFiles += embFile
                mountNotes += "音频嵌入表已装载：${audioEmbeddingRows}×512（外部采样与 iy_emb 同步）"
            } else {
                mountNotes += "音频嵌入表尺寸异常（${floats?.size ?: 0}），外部采样将退回图内 argmax 对齐"
            }
        } else {
            mountNotes += "缺少 conditioning/$embRel：采样 token 与 iy_emb 无法同步（可能出现乱码）"
        }
    }

    private fun loadG2p(condDir: File, manifest: JSONObject) {
        val g2pEntry = manifest.optJSONObject("g2p")
            ?: throw AemeathEngineException("conditioning.json 缺少 g2p 字段：无法把文本转成音素 id")
        val charRel = g2pEntry.optJSONObject("char2phone")?.optString("file")
            ?: throw AemeathEngineException("conditioning.json 缺少 g2p.char2phone.file")
        val charFile = File(condDir, charRel)
        if (!charFile.isFile) throw AemeathEngineException("G2P 字表不存在：${charFile.absolutePath}")
        g2p = G2pTable.load(charFile)
        mountedFiles += charFile

        g2pEntry.optJSONObject("symbols")?.optString("file")?.let { rel ->
            File(condDir, rel).takeIf { it.isFile }?.let {
                mountedFiles += it
                mountNotes += "音素表：${it.name}"
            }
        }

        // 变调表：symbol -> 声调号，以及「同韵母换调」的备用 symbol id。
        val toneRel = g2pEntry.optString("tone_table", "")
        val toneFile = if (toneRel.isNotEmpty()) File(condDir, toneRel) else File(condDir, "g2p/tone_table.json")
        if (toneFile.isFile) {
            runCatching {
                val json = JSONObject(toneFile.readText())
                val tonesJson = json.optJSONArray("tone_of")
                if (tonesJson != null) {
                    val tones = IntArray(tonesJson.length())
                    for (i in 0 until tonesJson.length()) tones[i] = tonesJson.optInt(i, 0)
                    toneOf = tones
                }
                val variantsJson = json.optJSONObject("tone_variants")
                if (variantsJson != null) {
                    val map = HashMap<String, Int>(variantsJson.length() * 2)
                    val keys = variantsJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        map[key] = variantsJson.optInt(key, -1)
                    }
                    toneVariants = map
                }
                mountedFiles += toneFile
                mountNotes += "变调表已装载：${toneVariants.size} 条换调映射"
            }.onFailure { Log.w(TAG, "tone_table.json 解析失败（退化为不变调）", it) }
        }

        // 多音字词组表
        val lexiconRel = g2pEntry.optString("lexicon", "")
        val lexiconFile = if (lexiconRel.isNotEmpty()) File(condDir, lexiconRel) else File(condDir, "g2p/lexicon.json")
        if (lexiconFile.isFile) {
            runCatching {
                val root = JSONObject(lexiconFile.readText())
                val table = root.optJSONObject("table") ?: root
                val map = HashMap<String, IntArray>(table.length() * 2)
                val keys = table.keys()
                while (keys.hasNext()) {
                    val word = keys.next()
                    val idsJson = table.optJSONArray(word) ?: continue
                    val ids = IntArray(idsJson.length()) { idsJson.optInt(it, 0) }
                    if (ids.isNotEmpty()) map[word] = ids
                }
                lexicon = map
                mountedFiles += lexiconFile
                mountNotes += "多音字词表已装载：${map.size} 条"
            }.onFailure { Log.w(TAG, "lexicon.json 解析失败（退化为逐字拼音）", it) }
        }
    }

    // ------------------------------------------------------------------ //
    // 文件发现与资源释放
    // ------------------------------------------------------------------ //
    private fun findStageFile(dir: File, prefix: String, preferQuantized: Boolean): File? {
        val candidates = if (preferQuantized) {
            listOf("$prefix.int8.onnx", "$prefix.onnx", "$prefix.fp16.onnx")
        } else {
            listOf("$prefix.fp16.onnx", "$prefix.onnx", "$prefix.int8.onnx")
        }
        for (name in candidates) {
            val file = File(dir, name)
            if (file.isFile) return file
        }
        return dir.listFiles { file ->
            file.isFile && file.name.lowercase().startsWith(prefix.lowercase()) &&
                file.extension.equals("onnx", ignoreCase = true)
        }?.minByOrNull { it.name.length }
    }

    override fun close() {
        runCatching { encoderSession?.close() }
        runCatching { firstDecoderSession?.close() }
        runCatching { secondDecoderSession?.close() }
        runCatching { vocoderSession?.close() }
        runCatching { bertFrontend?.close() }
        encoderSession = null
        firstDecoderSession = null
        secondDecoderSession = null
        vocoderSession = null
        bertFrontend = null
        conditioning.clear()
        g2p = null
        sharedSampler = null
        sharedSamplerWidth = 0
        encoderNames = null
        firstDecoderNames = null
        secondDecoderNames = null
        vocoderNames = null
    }

    override fun memoryFootprintMb(): Int {
        val bytes = mountedFiles.sumOf { it.length() }
        val runtime = Runtime.getRuntime()
        val heapMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        return (bytes / (1024 * 1024) + heapMb).toInt().coerceAtLeast(1)
    }

    // ------------------------------------------------------------------ //
    // 声码器（vits）与前缀切片的波形整形
    // ------------------------------------------------------------------ //

    /**
     * 声码器数值探针（m06856 第 1 条）：
     * 用真实 `ref_audio` + 极短语义码跑一次前向，确认该 EP 在这个图上「能跑且结果不是垃圾」。
     * 只要出现非有限值、全零或空输出，就让 `openStage` 换下一个后端，而不是整包退回 CPU。
     */
    private fun probeVocoder(env: OrtEnvironment, session: OrtSession, condDir: File): String? {
        val names = StageNames.resolve(
            "vits(probe)", session,
            listOf("text_seq", "pred_semantic", "ref_audio"),
            listOf("audio"),
        )
        val condRoot = runCatching { readFirstConditioning(condDir) }.getOrNull()
            ?: return "没有可用的 conditioning 目录，跳过数值探针"
        val probeCodes = longArrayOf(1L, 2L, 3L)
        val textIds = LongArray(8) { (it % 5).toLong() }
        val out = FeedBuilder().use { feed ->
            feed.putLong(env, names.input("text_seq"), textIds, longArrayOf(1, textIds.size.toLong()))
            feed.putLong(
                env, names.input("pred_semantic"), probeCodes,
                longArrayOf(1, 1, probeCodes.size.toLong())
            )
            feed.putFloat(env, names.input("ref_audio"), condRoot.refAudio, condRoot.refAudioShape)
            feed.runTensors(session)
        }
        return out.use { result ->
            val audio = result.floats(names.output("audio")) ?: return "vits 未输出 audio"
            if (audio.isEmpty()) return "vits 输出为空"
            var energy = 0.0
            for (value in audio) {
                if (!value.isFinite()) return "vits 输出含 NaN/Inf"
                energy += value.toDouble() * value
            }
            if (energy <= 1e-6) return "vits 输出近乎全零（EP 数值异常）"
            null
        }
    }

    /** 取第一个可用情绪的条件包（探针只需要真实 ref_audio 的形状与数值）。 */
    private fun readFirstConditioning(condDir: File): Conditioning? {
        val manifestFile = File(condDir, CONDITIONING_FILE)
        if (!manifestFile.isFile) return null
        val manifest = JSONObject(manifestFile.readText())
        val emotions = manifest.optJSONObject("emotions") ?: return null
        for (key in emotions.keys()) {
            val entry = emotions.optJSONObject(key) ?: continue
            val refAudio = Blob.tryLoad(condDir, key, entry, "ref_audio") ?: continue
            val floats = refAudio.floats ?: continue
            if (floats.isEmpty()) continue
            return Conditioning(LongArray(0), floats, refAudio.shape, null, null, null, null)
        }
        return null
    }

    /** 用一个语义码前缀跑声码器，返回该前缀对应的完整 PCM。 */
    private fun runVocoder(
        env: OrtEnvironment,
        vocoder: OrtSession,
        voNames: StageNames,
        textIds: LongArray,
        textLength: Int,
        codes: LongArray,
        condition: Conditioning,
    ): FloatArray {
        val audioOut = FeedBuilder().use { feed ->
            feed.putLong(env, voNames.input("text_seq"), textIds, longArrayOf(1, textLength.toLong()))
            feed.putLong(
                env, voNames.input("pred_semantic"), codes,
                longArrayOf(1, 1, codes.size.toLong())
            )
            feed.putFloat(env, voNames.input("ref_audio"), condition.refAudio, condition.refAudioShape)
            feed.runTensors(vocoder)
        }
        return audioOut.use { out ->
            out.floats(voNames.output("audio"))
                ?: throw AemeathEngineException("vits 未输出 audio")
        }
    }

    // ------------------------------------------------------------------ //
    // 数据载体
    // ------------------------------------------------------------------ //
    private class Conditioning(
        val refSeq: LongArray,
        val refAudio: FloatArray,
        val refAudioShape: LongArray,
        val sslContent: FloatArray?,
        val sslShape: LongArray?,
        val refBert: FloatArray?,
        val refBertShape: LongArray?,
        val stripPrefixCodes: Int = 0,
    ) {
        fun refBertOrZeros(): FloatArray = refBert ?: FloatArray(refSeq.size * BERT_DIM)
        fun refBertShapeOrZeros(): LongArray =
            refBertShape ?: longArrayOf(refSeq.size.toLong(), BERT_DIM.toLong())
    }

    private class Blob(
        val floats: FloatArray?,
        val longs: LongArray?,
        val shape: LongArray,
    ) {
        companion object {
            fun tryLoad(root: File, emotion: String, entry: JSONObject, key: String): Blob? {
                val info = entry.optJSONObject(key) ?: return null
                val rel = info.optString("file", "")
                if (rel.isEmpty()) return null
                val file = File(File(root, emotion), rel)
                if (!file.isFile) return null
                val shapeJson = info.optJSONArray("shape")
                val shape = LongArray(shapeJson?.length() ?: 0)
                if (shapeJson != null) {
                    for (i in 0 until shapeJson.length()) shape[i] = shapeJson.optLong(i, 1L)
                }
                val bytes = file.readBytes()
                val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                return when (info.optString("dtype", "float32")) {
                    "int64" -> {
                        val longs = LongArray(bytes.size / 8)
                        buffer.asLongBuffer().get(longs)
                        Blob(null, longs, shape)
                    }

                    else -> {
                        val floats = FloatArray(bytes.size / 4)
                        buffer.asFloatBuffer().get(floats)
                        Blob(floats, null, shape)
                    }
                }
            }

            /** 读一个裸 float32 小端文件（无 JSON 元信息，如 `audio_embedding.f32`）。 */
            fun tryLoadBare(file: File): FloatArray? {
                if (!file.isFile) return null
                val bytes = file.readBytes()
                if (bytes.size % 4 != 0) return null
                val floats = FloatArray(bytes.size / 4)
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
                return floats
            }
        }
    }

    /**
     * 一次 `session.run()` 的输出集合。
     *
     * 关键设计（m05844 第 2 条）：**不把输出拷成 JVM 数组**，而是持有
     * [OrtSession.Result]，需要时再用 [tensor] 取出 `OnnxTensor` 引用直接喂给下一步。
     * 自回归循环里 `k` / `v` / `y_emb` 都是 [24, L, 1, 512] / [1, L, 512] 的大张量，
     * 每步拷回 FloatArray 再重建张量等于 140 步 × ~15 MB 的 JNI 往返。
     *
     * 生命周期：只要还被下一步当输入引用，就必须保持打开；用完记得 `close()`。
     */
    private class RunOutputs(private val result: OrtSession.Result) : AutoCloseable {
        fun tensor(name: String): OnnxTensor? = result.get(name).orElse(null) as? OnnxTensor

        fun floats(name: String): FloatArray? {
            val tensor = tensor(name) ?: return null
            val buffer = tensor.floatBuffer.duplicate()
            buffer.rewind()
            val data = FloatArray(buffer.remaining())
            buffer.get(data)
            return data
        }

        fun longs(name: String): LongArray? {
            val tensor = tensor(name) ?: return null
            val buffer = tensor.longBuffer.duplicate()
            buffer.rewind()
            val data = LongArray(buffer.remaining())
            buffer.get(data)
            return data
        }

        fun shapeOf(name: String): LongArray =
            (tensor(name)?.info as? TensorInfo)?.shape ?: longArrayOf()

        override fun close() {
            runCatching { result.close() }
        }
    }

    /** 单个阶段的输入输出张量名（必填项缺失时立刻报错，绝不静默猜名）。 */
    private class StageNames(
        val stage: String,
        private val inputs: Map<String, String>,
        private val outputs: Map<String, String>,
    ) {
        fun input(role: String): String =
            inputs[role] ?: throw AemeathEngineException("$stage 缺少输入 $role")

        fun output(role: String): String =
            outputs[role] ?: throw AemeathEngineException("$stage 缺少输出 $role")

        companion object {
            fun resolve(
                stage: String,
                session: OrtSession,
                requiredInputs: List<String>,
                requiredOutputs: List<String>,
            ): StageNames = StageNames(
                stage,
                match(stage, "输入", session.inputInfo.keys.toList(), requiredInputs),
                match(stage, "输出", session.outputInfo.keys.toList(), requiredOutputs),
            )

            private fun match(
                stage: String,
                kind: String,
                actual: List<String>,
                required: List<String>,
            ): Map<String, String> {
                val result = HashMap<String, String>()
                val missing = mutableListOf<String>()
                for (role in required) {
                    val hit = actual.firstOrNull { it.equals(role, ignoreCase = true) }
                        ?: actual.firstOrNull { it.lowercase().contains(role.lowercase()) }
                    if (hit != null) result[role] = hit else missing += role
                }
                if (missing.isNotEmpty()) {
                    throw AemeathEngineException(
                        "$stage 的${kind}缺少 [${missing.joinToString(", ")}]；实际可用：$actual。" +
                            "请核对模型导出签名"
                    )
                }
                return result
            }
        }
    }

    /** 张量装配器：`use {}` 保证异常路径也释放本类自己创建的 native 内存。 */
    private class FeedBuilder : AutoCloseable {
        private val tensors = LinkedHashMap<String, OnnxTensor>()
        private val owned = LinkedHashSet<String>()

        fun putFloat(env: OrtEnvironment, name: String, data: FloatArray, shape: LongArray) {
            val buffer = ByteBuffer.allocateDirect(data.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            buffer.put(data)
            buffer.rewind()
            tensors[name] = OnnxTensor.createTensor(env, buffer, shape)
            owned += name
        }

        fun putLong(env: OrtEnvironment, name: String, data: LongArray, shape: LongArray) {
            val buffer = ByteBuffer.allocateDirect(data.size * 8)
                .order(ByteOrder.nativeOrder())
                .asLongBuffer()
            buffer.put(data)
            buffer.rewind()
            tensors[name] = OnnxTensor.createTensor(env, buffer, shape)
            owned += name
        }

        /**
         * 直接借入别处持有的张量（例如上一轮的 `k` / `v` / `y_emb`、恒定的 `x_example`）。
         * 借入的张量**不由本类负责释放**，调用方必须保证它在 `run` 期间有效。
         */
        fun putTensor(name: String, tensor: OnnxTensor) {
            tensors[name] = tensor
            owned -= name
        }

        /** 只返回 [RunOutputs]，输出张量的 native 内存由调用方 `close()` 释放。 */
        fun runTensors(session: OrtSession): RunOutputs = RunOutputs(session.run(tensors))

        override fun close() {
            for (name in owned) {
                runCatching { tensors[name]?.close() }
            }
            tensors.clear()
            owned.clear()
        }
    }

    companion object {
        private const val TAG = "AemeathSoVits"
        private const val THREADS = 0 // 0 = AemeathOrtBackend.autoThreads()（按核数 4..8）
        private const val BERT_DIM = 1024
        /** ar_audio_embedding 每个 token 的嵌入宽度。 */
        private const val AUDIO_EMB_DIM = 512
        /** 多音字词表最长匹配字数。 */
        private const val MAX_LEXICON_LEN = 4
        private const val CONDITIONING_FILE = "conditioning.json"
        private const val MAX_TEXT_PHONES = 200
        /** 复读参考音检测的 n-gram 长度（码数）。 */
        private const val ECHO_GRAM = 5

    }
}

/** 离线构建的汉字 → 音素 id 表（`g2p/char2phone.json`）。 */
class G2pTable private constructor(
    val charToPhones: Map<Char, IntArray>,
    val unknownId: Int,
) {
    companion object {
        fun load(file: File): G2pTable {
            val root = JSONObject(file.readText())
            val mapping = HashMap<Char, IntArray>(root.length() * 2)
            for (key in root.keys()) {
                if (key.isEmpty()) continue
                val array = root.optJSONArray(key) ?: continue
                val ids = IntArray(array.length())
                for (i in 0 until array.length()) ids[i] = array.optInt(i, 0)
                if (ids.isNotEmpty()) mapping[key[0]] = ids
            }
            if (mapping.isEmpty()) throw AemeathEngineException("G2P 字表为空：${file.absolutePath}")
            return G2pTable(mapping, 0)
        }
    }
}

