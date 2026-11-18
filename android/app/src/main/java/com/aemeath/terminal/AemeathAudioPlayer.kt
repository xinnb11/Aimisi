package com.aemeath.terminal

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * AemeathAudioPlayer.kt — 原生 AudioTrack 播放队列
 *
 * 设计要点（m05844 第 1 条改造）：
 * 1. **异步播放队列**：`playAsync()` 只把 PCM 入队就返回，**不等播完**。
 *    上层 `synthesizeSpeech` 因此可以立刻 resolve，前端马上发起下一句合成，
 *    CPU 合成下一句与 AudioTrack 播放上一句真正并行——这就是「边播边算」。
 * 2. **句内绝不断音**：一句 PCM 只对应一条音轨，顺序写完后等 buffer 播干净再停。
 * 3. **可打断**：`stop()` 递增 generation，正在播的和队列里等待的立即作废。
 * 4. **走媒体流**：`USAGE_MEDIA` + `CONTENT_TYPE_SPEECH`，保证媒体音量生效。
 */
class AemeathAudioPlayer {

    private val generation = AtomicInteger(0)
    private val pending = AtomicInteger(0)

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "aemeath-audio").apply { isDaemon = true }
    }

    @Volatile
    private var track: AudioTrack? = null

    /** 常驻音轨（m07045：天玑 9500 上反复 create/release 会触发 HAL 爆音，故复用）。 */
    @Volatile
    private var idleTrack: AudioTrack? = null

    @Volatile
    private var idleRate: Int = 0

    @Volatile
    private var playing: Boolean = false

    @Volatile
    private var lastError: String? = null

    fun lastError(): String? = lastError

    /** 队列里还没播完的句数（含正在播的这一句）。 */
    fun pendingCount(): Int = pending.get()

    /**
     * 把一段 Float32 单声道 PCM 放进播放队列（**立即返回，不阻塞合成线程**）。
     *
     * @return 这段音频的时长（毫秒），供前端估算 UI 反馈。
     */
    fun playAsync(samples: FloatArray, sampleRate: Int, volume: Float = 1.0f): Long {
        val rate = sampleRate.coerceIn(8000, 96000)
        if (samples.isEmpty()) return 0L
        val durationMs = samples.size * 1000L / rate
        val token = generation.get()
        val scaled = if (volume == 1.0f) samples else FloatArray(samples.size) { samples[it] * volume }

        pending.incrementAndGet()
        executor.execute {
            try {
                if (generation.get() != token) {
                    Log.i(TAG, "跳过被 stop() 作废的分句：${samples.size} samples")
                    return@execute
                }
                playBlocking(scaled, rate, durationMs, token)
            } catch (error: Throwable) {
                lastError = "播放异常：${error.message}"
                Log.w(TAG, lastError!!, error)
            } finally {
                pending.decrementAndGet()
            }
        }
        return durationMs
    }

    private fun playBlocking(samples: FloatArray, rate: Int, durationMs: Long, token: Int) {
        val minBuffer = AudioTrack.getMinBufferSize(
            rate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        if (minBuffer <= 0) {
            lastError = "AudioTrack.getMinBufferSize 返回 $minBuffer（设备不支持 PCM_FLOAT/该采样率）"
            Log.e(TAG, lastError!!)
            return
        }
        val bufferSize = max(minBuffer, CHUNK_SAMPLES * 4 * 4)

        // 复用同一条 AudioTrack：每句都 create/play/stop/release 会让天玑音频 HAL 反复
        // 重新配置输出通路，听感上就是「每句开头/结尾各一声爆音」（m07045 第 3 条）。
        // 这里只在采样率变化或轨道失效时重建，正常语流期间只做 play/pause + flush。
        val audioTrack = ensureTrack(rate, bufferSize) ?: return

        lastError = null
        track = audioTrack
        playing = true
        try {
            // 清掉上一句可能残留的采样，避免开头叠上旧尾巴造成爆音
            runCatching { if (audioTrack.playState == AudioTrack.PLAYSTATE_PLAYING) audioTrack.pause() }
            runCatching { audioTrack.flush() }
            audioTrack.setVolume(1.0f)
            audioTrack.play()
            Log.i(
                TAG,
                "开始播放：${samples.size} samples @${rate}Hz ≈ ${durationMs}ms " +
                    "buffer=${bufferSize}B 队列剩余=${(pending.get() - 1).coerceAtLeast(0)}"
            )

            var offset = 0
            while (offset < samples.size) {
                if (generation.get() != token) break
                val count = minOf(CHUNK_SAMPLES, samples.size - offset)
                val written = audioTrack.write(samples, offset, count, AudioTrack.WRITE_BLOCKING)
                if (written < 0) {
                    lastError = "AudioTrack.write 返回 $written"
                    Log.w(TAG, lastError!!)
                    break
                }
                offset += written
            }

            // buffer 里还有数据没被硬件消费，先等它播完再 pause，否则尾音被截断
            if (generation.get() == token) {
                val bufferedMs = bufferSize * 1000L / 4L / rate
                Thread.sleep(bufferedMs.coerceIn(60L, 600L) + 40L)
            }
        } finally {
            playing = false
            // 只 pause + flush，不 stop/release：轨道常驻，下一句继续用同一条通路。
            runCatching {
                if (audioTrack.playState == AudioTrack.PLAYSTATE_PLAYING) audioTrack.pause()
                audioTrack.flush()
            }
            Log.i(TAG, "播放结束：${samples.size} samples")
        }
    }

    /** 按采样率复用同一条 AudioTrack（天玑 9500 上反复重建会触发 HAL 爆音）。 */
    private fun ensureTrack(rate: Int, bufferSize: Int): AudioTrack? {
        val existing = idleTrack
        if (existing != null && idleRate == rate && runCatching { existing.state == AudioTrack.STATE_INITIALIZED }.getOrDefault(false)) {
            return existing
        }
        runCatching { existing?.release() }
        idleTrack = null
        val created = try {
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(rate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        } catch (error: Throwable) {
            lastError = "AudioTrack 创建失败：${error.message}"
            Log.e(TAG, lastError!!, error)
            return null
        }
        idleTrack = created
        idleRate = rate
        return created
    }

    fun stop(): Boolean {
        generation.incrementAndGet()
        pending.set(0)
        val current = track ?: idleTrack
        playing = false
        if (current != null) {
            // 只 pause + flush，保留常驻音轨给下一句复用（stop()/release() 会造成 HAL 重配爆音）。
            runCatching {
                if (current.playState == AudioTrack.PLAYSTATE_PLAYING) current.pause()
                current.flush()
            }
        }
        return current != null
    }

    /** 正在播或队列里还有没播完的句子。 */
    fun isPlaying(): Boolean = playing || pending.get() > 0

    fun release() {
        stop()
        runCatching { idleTrack?.release() }
        idleTrack = null
        idleRate = 0
        track = null
        runCatching { executor.shutdownNow() }
    }

    companion object {
        private const val TAG = "AemeathAudio"
        private const val CHUNK_SAMPLES = 4096
    }
}

/**
 * 轻量的 PCM 后处理：试听音量归一 + 首尾 5ms 淡入淡出。
 *
 * 声码器输出偶尔会带轻微直流或首尾爆音，这里做最后一道保险，
 * 避免手机扬声器出现「啪」的破音。
 */
object AemeathPcmPostProcess {

    fun normalize(samples: FloatArray, targetPeak: Float = 0.89f): FloatArray {
        var peak = 0f
        for (value in samples) {
            val magnitude = if (value < 0f) -value else value
            if (magnitude > peak) peak = magnitude
        }
        if (peak <= 1e-6f) return samples
        if (peak <= targetPeak) return samples
        val gain = targetPeak / peak
        return FloatArray(samples.size) { (samples[it] * gain).coerceIn(-1f, 1f) }
    }

    fun fade(samples: FloatArray, sampleRate: Int, fadeMs: Float = 5f): FloatArray {
        val fadeSamples = (sampleRate * fadeMs / 1000f).roundToInt()
        if (fadeSamples <= 1 || samples.size < fadeSamples * 2) return samples
        val out = samples.copyOf()
        for (i in 0 until fadeSamples) {
            val gain = i.toFloat() / fadeSamples
            out[i] *= gain
            out[out.size - 1 - i] *= gain
        }
        return out
    }

    /**
     * 变速：**只在极端慢速时使用**。对波形直接重采样会同时改变基频与共振峰
     * （m04444 第 4 条），情绪在没有独立参考音频前一律 1.0x 直出，本函数默认不被调用。
     */
    fun changeSpeed(samples: FloatArray, speed: Float): FloatArray {
        if (speed <= 0f || kotlin.math.abs(speed - 1.0f) < 0.01f) return samples
        val outSize = (samples.size / speed).roundToInt().coerceAtLeast(1)
        val out = FloatArray(outSize)
        for (i in 0 until outSize) {
            val position = i * speed
            val index = position.toInt()
            val fraction = position - index
            val a = samples[index.coerceIn(0, samples.size - 1)]
            val b = samples[(index + 1).coerceIn(0, samples.size - 1)]
            out[i] = a + (b - a) * fraction
        }
        return out
    }
}
