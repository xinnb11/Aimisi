# GPT-SoVITS 2 INT8 模型部署目录说明 (ONNX Runtime)

本目录为微调后的 **GPT-SoVITS v2 (INT8 动态量化)** 离线模型包专属目录，运行在 **ONNX Runtime (ORT)** 推理引擎上。

## 1. 模型文件结构要求
模型训练及 INT8 量化导出完成后，请将生成的 `.onnx` 权重放置在此目录下：

```text
D:\语音模型项目\models\aemeath-gpt-sovits2\
├── manifest.json            # 语音模型元数据描述（已预设）
├── t2s.int8.onnx            # Text-to-Semantic 自回归模型 (INT8 量化)
├── vits.int8.onnx           # SoVITS 声学特征及 HiFi-GAN 声码器 (INT8 量化)
├── cnhubert.int8.onnx       # CNHuBERT SSL 音频特征提取器 (可选，若使用预提特征可免)
├── bert.int8.onnx           # Chinese RoBERTa 文本表征提取器 (INT8 量化)
└── emotions/                # 4 种情绪特征向量 (.bin，已就绪)
    ├── 平静.bin             # 256维 float32 特征
    ├── 开心.bin             # 256维 float32 特征
    ├── 温柔.bin             # 256维 float32 特征
    └── 调侃.bin             # 256维 float32 特征
```

## 2. 核心采样管线约束（拒绝 Argmax 退化）
系统已内置完整多项式核采样机制（Top-K + Top-P + Temperature + Repetition Penalty + Multinomial Sampling），绝非简单的贪婪 argmax：
- `temperature`: 0.85
- `top_k`: 25
- `top_p`: 0.90
- `repetition_penalty`: 1.35
- `stop_token`: 1024
