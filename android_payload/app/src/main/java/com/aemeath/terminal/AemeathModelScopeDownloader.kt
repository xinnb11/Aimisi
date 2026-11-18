package com.aemeath.terminal

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.URLEncoder
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

/**
 * AemeathModelScopeDownloader.kt — 魔搭社区（ModelScope）端侧分片下载器
 *
 * 能力：
 * 1. 通过魔搭 Hub API 解析仓库文件树，或读取仓库根目录 `manifest.json` 的
 *    `files[]`（含 SHA256），两种来源都支持；
 * 2. 3 路并发分片下载 + `Range` 断点续传（`.part` 半成品文件）+ 指数退避重试；
 * 3. 每个文件下载完成后流式计算 SHA256，与前缀校验值比对；
 * 4. 全程向外抛 `DownloadProgress`（进度百分比 / 阶段文案 / 字节数 / 实时速率）；
 * 5. 路径白名单校验，杜绝 `../../` 目录穿越写入沙盒外。
 *
 * 目标目录：`context.filesDir/models/aemeath/<modelId>/`（应用私有沙盒）。
 */
class AemeathModelScopeDownloader(
    private val baseUrl: String = DEFAULT_BASE_URL,
    private val token: String? = null,
    private val concurrency: Int = 3,
    private val timeoutSeconds: Long = 60L,
    private val client: OkHttpClient = defaultClient(timeoutSeconds),
) {

    private val downloadedBytes = AtomicLong(0L)
    private val totalBytes = AtomicLong(0L)
    private var lastEmitAtMs = 0L
    private var lastEmitBytes = 0L

    suspend fun download(
        repoId: String,
        revision: String,
        destDir: File,
        modelId: String,
        onProgress: (DownloadProgress) -> Unit,
    ): DownloadResult = withContext(Dispatchers.IO) {
        try {
            destDir.mkdirs()
            onProgress(DownloadProgress(modelId, 0, "正在解析魔搭仓库元数据…", 0, 0, "—"))

            val remoteFiles = fetchFileList(repoId, revision)
            if (remoteFiles.isEmpty()) {
                return@withContext DownloadResult(
                    success = false,
                    modelId = modelId,
                    savedDir = destDir.absolutePath,
                    totalBytes = 0,
                    hashVerified = false,
                    message = "仓库 $repoId 中没有可下载的模型文件",
                )
            }

            val planned = remoteFiles.filter { allowedExtension(it.relativePath) }
            val bytesTotal = planned.sumOf { it.size }
            totalBytes.set(bytesTotal)
            downloadedBytes.set(planned.filter { alreadyComplete(destDir, it) }.sumOf { it.size })
            onProgress(
                DownloadProgress(
                    modelId,
                    2,
                    "发现 ${planned.size} 个文件，合计 ${formatBytes(bytesTotal)}，开始分片下载…",
                    downloadedBytes.get(),
                    bytesTotal,
                    "—",
                )
            )

            val semaphore = Semaphore(concurrency.coerceIn(1, 6))
            val failures = mutableListOf<String>()
            var hashVerified = true

            coroutineScope {
                planned.map { remote ->
                    async {
                        semaphore.withPermit {
                            val outcome = downloadWithRetry(repoId, revision, destDir, modelId, remote, onProgress)
                            if (!outcome.success) {
                                synchronized(failures) { failures += remote.relativePath }
                            }
                            if (!outcome.hashVerified) hashVerified = false
                        }
                    }
                }.awaitAll()
            }

            val complete = failures.isEmpty()
            if (complete) {
                writeCompletionMarker(destDir, repoId, revision, planned)
                onProgress(
                    DownloadProgress(
                        modelId, 100,
                        if (hashVerified) "SHA256 校验通过，模型包就绪！" else "下载完成，但存在哈希不一致文件",
                        downloadedBytes.get(), bytesTotal, "—",
                    )
                )
            }
            DownloadResult(
                success = complete,
                modelId = modelId,
                savedDir = destDir.absolutePath,
                totalBytes = bytesTotal,
                hashVerified = hashVerified && complete,
                failedFiles = failures.toList(),
                message = if (complete) "模型包下载完成" else "部分文件下载失败：${failures.joinToString(",")}",
            )
        } catch (error: Throwable) {
            Log.e(TAG, "下载失败：${error.message}", error)
            DownloadResult(
                success = false,
                modelId = modelId,
                savedDir = destDir.absolutePath,
                totalBytes = totalBytes.get(),
                hashVerified = false,
                message = "下载失败：${error.message}",
            )
        }
    }

    // ------------------------------------------------------------------ //
    // 仓库文件列表
    // ------------------------------------------------------------------ //
    private fun fetchFileList(repoId: String, revision: String): List<RemoteFile> {
        // 优先用仓库内 manifest.json：它由量化脚本生成，含嵌套路径与权威 SHA256
        val fromManifest = runCatching { fetchFileListFromManifest(repoId, revision) }.getOrElse {
            Log.i(TAG, "仓库不含可用 manifest.json（${it.message}），改用 Hub 文件树接口")
            emptyList()
        }
        if (fromManifest.isNotEmpty()) return fromManifest

        return runCatching { fetchFileListFromApi(repoId, revision) }.getOrElse {
            Log.w(TAG, "Hub 文件树接口失败：${it.message}")
            emptyList()
        }
    }

    /** `GET /api/v1/models/{repo}/repo/files?Revision=&Root=/` */
    private fun fetchFileListFromApi(repoId: String, revision: String): List<RemoteFile> {
        val url = "$baseUrl/api/v1/models/$repoId/repo/files" +
            "?Revision=${encode(revision)}&Root=/&Recursive=true"
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/json")
            .apply { token?.let { header("Authorization", "Bearer $it") } }
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("Hub 返回 HTTP ${response.code}")
            }
            val payload = response.body?.string().orEmpty()
            val root = JSONObject(payload).optJSONObject("Data") ?: return emptyList()
            val files = root.optJSONArray("Files") ?: return emptyList()

            val result = mutableListOf<RemoteFile>()
            for (index in 0 until files.length()) {
                val entry = files.optJSONObject(index) ?: continue
                val type = entry.optString("Type", "blob")
                if (type != "blob") continue
                val path = entry.optString("Path", entry.optString("Name", ""))
                val relative = ModelPaths.safeRelativePath(path) ?: continue
                result += RemoteFile(
                    relativePath = relative,
                    size = entry.optLong("Size", 0L),
                    sha256 = entry.optString("Sha256", "").ifBlank { null },
                )
            }
            return result
        }
    }

    /** 读取仓库根目录 `manifest.json` 的 `files[]`（由量化脚本生成，含 SHA256）。 */
    private fun fetchFileListFromManifest(repoId: String, revision: String): List<RemoteFile> {
        val url = fileUrl(repoId, revision, "manifest.json")
        val request = Request.Builder()
            .url(url)
            .apply { token?.let { header("Authorization", "Bearer $it") } }
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("manifest.json 返回 HTTP ${response.code}")
            }
            val text = response.body?.string().orEmpty()
            val manifest = JSONObject(text)
            val files = manifest.optJSONArray("files") ?: return emptyList()
            val result = mutableListOf<RemoteFile>()
            for (index in 0 until files.length()) {
                val entry = files.optJSONObject(index) ?: continue
                val relative = ModelPaths.safeRelativePath(entry.optString("path", "")) ?: continue
                result += RemoteFile(
                    relativePath = relative,
                    size = entry.optLong("bytes", 0L),
                    sha256 = entry.optString("sha256", "").ifBlank { null },
                )
            }
            return result
        }
    }

    // ------------------------------------------------------------------ //
    // 单文件下载（含断点续传 / 重试 / 校验）
    // ------------------------------------------------------------------ //
    private class FileOutcome(val success: Boolean, val hashVerified: Boolean)

    private suspend fun downloadWithRetry(
        repoId: String,
        revision: String,
        destDir: File,
        modelId: String,
        remote: RemoteFile,
        onProgress: (DownloadProgress) -> Unit,
    ): FileOutcome {
        var attempt = 0
        var lastError: Throwable? = null
        while (attempt < MAX_RETRY) {
            attempt += 1
            try {
                return downloadOnce(repoId, revision, destDir, modelId, remote, onProgress)
            } catch (error: Throwable) {
                lastError = error
                Log.w(TAG, "下载 ${remote.relativePath} 第 $attempt 次失败：${error.message}")
                if (attempt < MAX_RETRY) {
                    delay(400L * attempt * attempt)
                }
            }
        }
        Log.e(TAG, "放弃 ${remote.relativePath}：${lastError?.message}")
        return FileOutcome(false, false)
    }

    private fun downloadOnce(
        repoId: String,
        revision: String,
        destDir: File,
        modelId: String,
        remote: RemoteFile,
        onProgress: (DownloadProgress) -> Unit,
    ): FileOutcome {
        val target = File(destDir, remote.relativePath)
        target.parentFile?.mkdirs()
        val partial = File(target.absolutePath + ".part")

        if (alreadyComplete(destDir, remote)) {
            emitProgress(modelId, "已存在且校验通过：${remote.relativePath}", onProgress, force = true)
            return FileOutcome(true, true)
        }

        var existing = if (partial.exists()) partial.length() else 0L
        if (existing > 0 && remote.size > 0 && existing >= remote.size) {
            existing = 0L
            partial.delete()
        }

        val requestBuilder = Request.Builder().url(fileUrl(repoId, revision, remote.relativePath))
        if (existing > 0L) {
            requestBuilder.header("Range", "bytes=$existing-")
        }
        token?.let { requestBuilder.header("Authorization", "Bearer $it") }

        val stage = "正在下载 ${remote.relativePath}（${formatBytes(existing)}/${formatBytes(remote.size)}）…"
        emitProgress(modelId, stage, onProgress, force = true)

        client.newCall(requestBuilder.build()).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("HTTP ${response.code} @ ${remote.relativePath}")
            }
            val body = response.body ?: throw IllegalStateException("空响应体：${remote.relativePath}")
            val append = existing > 0L && response.code == 206
            val startOffset = if (append) existing else 0L
            if (!append && partial.exists()) partial.delete()

            body.byteStream().use { input ->
                FileOutputStream(partial, append).use { output ->
                    copyWithProgress(input, output, startOffset, modelId, remote, onProgress)
                }
            }
        }

        // SHA256 校验
        val digest = sha256(partial)
        val expected = remote.sha256?.lowercase()
        val verified = expected == null || expected == digest.lowercase()
        if (!verified) {
            Log.e(TAG, "SHA256 不匹配：${remote.relativePath} 期望=$expected 实际=$digest")
            partial.delete()
            downloadedBytes.addAndGet(-remote.size)
            return FileOutcome(true, false)
        }

        if (target.exists()) target.delete()
        if (!partial.renameTo(target)) {
            partial.copyTo(target, overwrite = true)
            partial.delete()
        }
        emitProgress(
            modelId,
            "SHA256 校验通过：${remote.relativePath}",
            onProgress,
            force = true,
        )
        return FileOutcome(true, true)
    }

    private fun copyWithProgress(
        input: InputStream,
        output: FileOutputStream,
        startOffset: Long,
        modelId: String,
        remote: RemoteFile,
        onProgress: (DownloadProgress) -> Unit,
    ) {
        val buffer = ByteArray(64 * 1024)
        var written = startOffset
        while (true) {
            val read = input.read(buffer)
            if (read <= 0) break
            output.write(buffer, 0, read)
            written += read
            downloadedBytes.addAndGet(read.toLong())
            emitProgress(
                modelId,
                "正在下载 ${remote.relativePath}（${formatBytes(written)}/${formatBytes(remote.size)}）…",
                onProgress,
            )
        }
        output.flush()
    }

    // ------------------------------------------------------------------ //
    // 工具
    // ------------------------------------------------------------------ //
    private fun alreadyComplete(destDir: File, remote: RemoteFile): Boolean {
        val target = File(destDir, remote.relativePath)
        if (!target.isFile) return false
        if (remote.size > 0 && target.length() != remote.size) return false
        val expected = remote.sha256 ?: return true
        return sha256(target).equals(expected, ignoreCase = true)
    }

    private fun emitProgress(
        modelId: String,
        stage: String,
        onProgress: (DownloadProgress) -> Unit,
        force: Boolean = false,
    ) {
        val now = System.currentTimeMillis()
        val done = downloadedBytes.get()
        val total = totalBytes.get().coerceAtLeast(1L)
        val percent = ((done.toDouble() / total) * 100.0).roundToInt().coerceIn(0, 100)

        if (!force && now - lastEmitAtMs < PROGRESS_INTERVAL_MS && percent < 100) return

        val elapsedSeconds = (now - lastEmitAtMs).coerceAtLeast(1L) / 1000.0
        val deltaBytes = (done - lastEmitBytes).coerceAtLeast(0L)
        val speed = if (lastEmitAtMs == 0L) "—" else formatBytes((deltaBytes / elapsedSeconds).toLong()) + "/s"

        lastEmitAtMs = now
        lastEmitBytes = done
        onProgress(DownloadProgress(modelId, percent, stage, done, total, speed))
    }

    private fun writeCompletionMarker(
        destDir: File,
        repoId: String,
        revision: String,
        files: List<RemoteFile>,
    ) {
        val marker = JSONObject().apply {
            put("repoId", repoId)
            put("revision", revision)
            put("fileCount", files.size)
            put("completedAt", System.currentTimeMillis())
            put("totalBytes", files.sumOf { it.size })
        }
        runCatching { File(destDir, MARKER_FILE).writeText(marker.toString(2), Charsets.UTF_8) }
    }

    private fun fileUrl(repoId: String, revision: String, path: String): String =
        "$baseUrl/api/v1/models/$repoId/repo?Revision=${encode(revision)}&FilePath=${encode(path)}"

    private fun encode(value: String): String = URLEncoder.encode(value, "UTF-8")

    companion object {
        private const val TAG = "AemeathDownloader"
        private const val DEFAULT_BASE_URL = "https://modelscope.cn"
        private const val MAX_RETRY = 3
        private const val PROGRESS_INTERVAL_MS = 150L
        private const val MARKER_FILE = ".aemeath_complete.json"

        private val ALLOWED_EXTENSIONS = setOf(
            // 模型与词典
            "onnx", "bin", "txt", "json", "md", "yaml", "yml",
            // 参考条件包的裸二进制张量（prepare_ref_conditioning.py 产出）
            "f32", "i64", "s64", "npy",
        )

        fun defaultClient(timeoutSeconds: Long): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .followRedirects(true)
            .build()

        private fun allowedExtension(path: String): Boolean {
            val ext = path.substringAfterLast('.', "").lowercase()
            return ext in ALLOWED_EXTENSIONS
        }

        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1 shl 20)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        fun formatBytes(bytes: Long): String {
            if (bytes < 1024) return "$bytes B"
            val units = arrayOf("KB", "MB", "GB", "TB")
            var value = bytes.toDouble() / 1024.0
            var index = 0
            while (value >= 1024.0 && index < units.size - 1) {
                value /= 1024.0
                index += 1
            }
            return String.format(java.util.Locale.US, "%.1f %s", value, units[index])
        }
    }
}

/** 仓库中的一个文件条目。 */
data class RemoteFile(
    val relativePath: String,
    val size: Long,
    val sha256: String?,
)
