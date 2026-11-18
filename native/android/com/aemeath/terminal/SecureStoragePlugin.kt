package com.aemeath.terminal

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin

/**
 * SecureStoragePlugin.kt - Android 原生安全存储插件
 * 使用 Jetpack Security (EncryptedSharedPreferences + AES256-GCM) 保存用户 BYOK API Key
 */
@CapacitorPlugin(name = "SecureStoragePlugin")
class SecureStoragePlugin : Plugin() {

    private lateinit var sharedPreferences: SharedPreferences

    override fun load() {
        super.load()
        try {
            val masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            sharedPreferences = EncryptedSharedPreferences.create(
                "aemeath_secure_prefs",
                masterKeyAlias,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
        } catch (e: Exception) {
            // 降级使用沙盒常规 SharedPreferences
            sharedPreferences = context.getSharedPreferences("aemeath_fallback_prefs", Context.MODE_PRIVATE)
        }
    }

    @PluginMethod
    fun setSecureKey(call: PluginCall) {
        val key = call.getString("key")
        val value = call.getString("value")

        if (key.isNullOrEmpty() || value == null) {
            call.reject("Key and value are required")
            return
        }

        sharedPreferences.edit().putString(key, value).apply()
        call.resolve()
    }

    @PluginMethod
    fun getSecureKey(call: PluginCall) {
        val key = call.getString("key")
        if (key.isNullOrEmpty()) {
            call.reject("Key is required")
            return
        }

        val value = sharedPreferences.getString(key, null)
        val res = JSObject()
        res.put("value", value)
        call.resolve(res)
    }

    @PluginMethod
    fun removeSecureKey(call: PluginCall) {
        val key = call.getString("key")
        if (key.isNullOrEmpty()) {
            call.reject("Key is required")
            return
        }

        sharedPreferences.edit().remove(key).apply()
        call.resolve()
    }
}
