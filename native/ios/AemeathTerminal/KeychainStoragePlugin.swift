import Foundation
import Capacitor
import Security

/**
 * KeychainStoragePlugin.swift - iOS Keychain 硬件级安全存储
 * 存储用户 BYOK API Key，具备设备级加密保护
 */
@objc(KeychainStoragePlugin)
public class KeychainStoragePlugin: CAPPlugin {
    
    private let serviceName = "com.aemeath.terminal.keychain"
    
    @objc func setSecureKey(_ call: CAPPluginCall) {
        guard let key = call.getString("key"),
              let value = call.getString("value"),
              let data = value.data(using: .utf8) else {
            call.reject("Key and value are required")
            return
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key
        ]
        
        SecItemDelete(query as CFDictionary)
        
        let attributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]
        
        let status = SecItemAdd(attributes as CFDictionary, nil)
        if status == errSecSuccess {
            call.resolve()
        } else {
            call.reject("Keychain write failed with status: \(status)")
        }
    }
    
    @objc func getSecureKey(_ call: CAPPluginCall) {
        guard let key = call.getString("key") else {
            call.reject("Key is required")
            return
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecSuccess, let data = dataTypeRef as? Data, let value = String(data: data, encoding: .utf8) {
            var res = JSObject()
            res["value"] = value
            call.resolve(res)
        } else {
            var res = JSObject()
            res["value"] = NSNull()
            call.resolve(res)
        }
    }
}
