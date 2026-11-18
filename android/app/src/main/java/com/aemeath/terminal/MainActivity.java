package com.aemeath.terminal;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        // 必须在 super.onCreate 之前注册，否则 window.Capacitor.Plugins.AemeathTtsPlugin 不存在，
        // 前端 LocalOnnxTtsBridge 会静默降级到 Web Worker（不报错，但不走原生推理）。
        registerPlugin(AemeathTtsPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
