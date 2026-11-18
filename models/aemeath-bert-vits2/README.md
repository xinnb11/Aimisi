# BERT-VITS 2 INT8 模型部署目录说明 (Sherpa-ONNX)

本目录为微调后的 **BERT-VITS 2 (INT8 动态量化)** 离线模型包专属目录，运行在 **Sherpa-ONNX** 推理引擎上。

## 1. 模型文件结构要求
模型训练及 INT8 量化导出完成后，请将生成的权重放置在此目录下：

```text
D:\语音模型项目\models\aemeath-bert-vits2\
├── manifest.json            # 语音模型元数据描述（已预设）
├── model.int8.onnx          # BERT-VITS2 声学模型权重 (INT8 动态量化)
├── bert.int8.onnx           # Chinese RoBERTa 文本表征提取器 (INT8 量化)
├── tokens.txt               # 音素字典符号表 (已配置)
├── lexicon.txt              # 拼音音素映射词汇表 (已配置)
└── emotions/                # 4 种情绪特征向量 (.bin，已就绪)
    ├── 平静.bin             # 256维 float32 特征
    ├── 开心.bin             # 256维 float32 特征
    ├── 温柔.bin             # 256维 float32 特征
    └── 调侃.bin             # 256维 float32 特征
```

## 2. 推理配置
- `noise_scale`: 0.667
- `noise_scale_w`: 0.8
- `length_scale`: 1.0 (语速比)
- `sid`: 0 (爱弥斯专属说话人 ID)
