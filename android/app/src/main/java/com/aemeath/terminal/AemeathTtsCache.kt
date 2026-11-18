package com.aemeath.terminal

import android.content.Context
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * 端侧语音合成结果缓存（m07626）。
 *
 * 动机：一句 20~30 字的中文在 CPU 上要跑 15~25 s 的自回归，而用户重复点"播放"、
 * 或者同一句在列表里被再次触发时，重算完全是浪费。缓存命中后直接吐 PCM，
 * 端到端从几十秒降到毫秒级。
 *
 * 设计要点：
 *  - 键 = 文本 + 情绪 + 模型 id + 架构 + 采样参数签名 + 缓存版本，任何一项变了就不命中
 *    （避免模型/参数更新后播旧音频）；
 *  - 落盘格式极为简单：`[int32 sampleRate][float32 PCM...]`，小端，便于零依赖读回；
 *  - 目录固定在 `filesDir/cache/tts`，不进模型目录，卸载即清；
 *  - 按「最后修改时间」做 LRU 淘汰，默认上限 256 MB（可配置）；
 *  - 所有方法都吞异常：缓存永远只是加速手段，坏掉不能影响合成主流程。
 */
internal class AemeathTtsCache(
    context: Context,
    private val limitBytes: Long = DEFAULT_LIMIT_BYTES,
) {
    private val dir: File = File(File(context.filesDir, "cache"), "tts").apply {
        if (!exists()) mkdirs()
    }

    /** 缓存条目：PCM 浮点采样 + 采样率。 */
    class Entry(val samples: FloatArray, val sampleRate: Int) {
        val durationMs: Long
            get() = if (sampleRate <= 0) 0L else samples.size * 1000L / sampleRate
    }

    fun keyOf(
        text: String,
        emotionKey: String,
        modelId: String?,
        arch: String?,
        samplingSignature: String,
    ): String {
        val raw = buildString {
            append(CACHE_VERSION)
            append('|')
            append(modelId ?: "-")
            append('|')
            append(arch ?: "-")
            append('|')
            append(emotionKey)
            append('|')
            append(samplingSignature)
            append('|')
            append(text.trim())
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(raw.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }.substring(0, 32)
    }

    fun load(key: String): Entry? {
        val file = File(dir, "$key.bin")
        if (!file.isFile || file.length() <= 4) return null
        return try {
            RandomAccessFile(file, "r").use { raf ->
                val header = ByteArray(4)
                raf.readFully(header)
                val sampleRate = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN).int
                if (sampleRate !in 8000..96000) return null
                val pcmBytes = (raf.length() - 4L).toInt()
                if (pcmBytes <= 0 || pcmBytes % 4 != 0) return null
                val body = ByteArray(pcmBytes)
                raf.readFully(body)
                val buffer = ByteBuffer.wrap(body).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val samples = FloatArray(buffer.remaining())
                buffer.get(samples)
                // 触碰 mtime，让 LRU 淘汰时把它排在后面
                file.setLastModified(System.currentTimeMillis())
                Entry(samples, sampleRate)
            }
        } catch (error: Throwable) {
            Log.w(TAG, "读取缓存失败（忽略）：${error.message}")
            runCatching { file.delete() }
            null
        }
    }

    fun store(key: String, samples: FloatArray, sampleRate: Int): Boolean {
        if (samples.isEmpty() || sampleRate <= 0) return false
        return try {
            val payload = ByteArray(samples.size * 4)
            val buffer = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            buffer.put(samples)
            val tmp = File(dir, "$key.tmp")
            tmp.outputStream().use { out ->
                out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(sampleRate).array())
                out.write(payload)
            }
            val target = File(dir, "$key.bin")
            if (!tmp.renameTo(target)) {
                // 某些文件系统不允许覆盖重命名，退化为「删掉再改名」
                target.delete()
                if (!tmp.renameTo(target)) {
                    tmp.delete()
                    return false
                }
            }
            evictIfNeeded()
            true
        } catch (error: Throwable) {
            Log.w(TAG, "写入缓存失败（忽略）：${error.message}")
            false
        }
    }

    /** 当前条目数与占用字节数（前端设置页可展示）。 */
    fun stats(): LongArray {
        val files = dir.listFiles { f -> f.isFile && f.name.endsWith(".bin") } ?: return longArrayOf(0, 0)
        var bytes = 0L
        for (file in files) bytes += file.length()
        return longArrayOf(files.size.toLong(), bytes)
    }

    fun clear(): Pair<Int, Long> {
        var removed = 0
        var bytes = 0L
        dir.listFiles()?.forEach { file ->
            if (file.isFile) {
                val size = file.length()
                if (file.delete()) {
                    removed += 1
                    bytes += size
                }
            }
        }
        return removed to bytes
    }

    private fun evictIfNeeded() {
        val files = (dir.listFiles { f -> f.isFile } ?: return).sortedBy { it.lastModified() }
        var total = files.sumOf { it.length() }
        if (total <= limitBytes) return
        for (file in files) {
            if (total <= limitBytes) break
            val size = file.length()
            if (file.delete()) {
                total -= size
                Log.i(TAG, "缓存淘汰：${file.name}（${size / 1024} KB）")
            }
        }
    }

    companion object {
        private const val TAG = "AemeathTtsCache"

        /** 缓存格式版本：改动 PCM 后处理或采样逻辑时 +1，旧缓存自动失效。 */
        const val CACHE_VERSION = 3

        const val DEFAULT_LIMIT_BYTES: Long = 256L * 1024 * 1024
    }
}
