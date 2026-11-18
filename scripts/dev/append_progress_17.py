import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
## 17. 硬件加速路由改造（2026-11-17 20:20）

用户 m06856 要求「必须真正调用 NPU / GPU / XNNPACK，而不是一个节点报错就把 5 个模型一起退回 CPU」。本轮完成了第 1 条改造并在真机取得量化数据；第 2、3 条（裁剪 sdec 尾部采样链 / FP16）已做前置验证，尚未落地。

### 17.1 五阶段独立路由（已实现）
`AemeathOrtBackend.kt` 重写为按模型独立探测：

```
候选顺序：QNN_HTP → QNN_GPU → NNAPI(USE_FP16) → XNNPACK → CPU(NEON)
```
- `openStage(env, modelFile, stage, backend, threads, allowXnnpack, probe)`：每个 stage 独立建会话，任一后端失败**只降级它自己**；
- NNAPI 一律走 `addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))`（不再裸调无参版本）；
- QNN 走 `addQnn(mapOf("backend_path" to "libQnnHtp.so" | "libQnnGpu.so", "enable_htp_fp16_precision" to "1"))`；
- `probe` 数值探针：`vits` 已接（真实 `ref_audio` + 3 个语义码前向，检查 NaN/Inf/全零），其余 stage 以会话建立为闸门；
- **XNNPACK 解封**：`bert / t2s_encoder / t2s_fsdec / vits` 恢复 XNNPACK；只有 `t2s_sdec` 因 §14.3 实测数值错误保持 `allowXnnpack = false`。

### 17.2 真机路由结果（`JJFIE6AULVDIMR95`，骁龙 8 Elite，Android 16，ORT 1.22.0）
- `backend='npu'`：五个 stage 全部成功建立 **NNAPI (FP16)** 会话（日志逐条 `会话建立：… @ NNAPI (FP16) · 4 线程`），但**合成阶段挂死**（>300 s 未返回），只能中断 → NNAPI 在本模型（int8 权重 + 动态 KV cache + 逐 token 自回归）上不可用。
- `backend='cpu'`（XNNPACK）：`enc/fsdec/vits/bert` 实际生效 **XNNPACK · 4 线程**，`sdec` 生效 **CPU (ARM NEON) · 4 线程**。
- QNN：官方 `onnxruntime-android:1.22.0` 显式报 `QNN execution provider is not supported in this build`；已确认 Maven 存在 `onnxruntime-android-qnn:1.22.0`（6.8 MB，内含 18.1 MB `libonnxruntime.so` + `libonnxruntime4j_jni.so`，`p_align=0x4000` 满足 16 KB），但该包**不含** `libQnnHtp.so`，需要高通 QNN SDK 的 HTP 库随包投递才能驱动 Hexagon NPU。

### 17.3 分段计时（23 字「漂泊者，语音链路已经接通。现在可以正常对话啦！」，132 码）
| 阶段 | 耗时 | 实际 EP |
| --- | ---: | --- |
| BERT | 120 ms | XNNPACK · 4 线程 |
| encoder | 13 ms | XNNPACK · 4 线程 |
| fsdec | 231 ms | XNNPACK · 4 线程 |
| **sdec** | **15 412 ms（132 步，116.8 ms/步）** | CPU (ARM NEON) · 4 线程 |
| vits | 3 542 ms | XNNPACK · 4 线程 |
| 合计 | **19 318 ms** | — |

结论：**80% 时间在 sdec**；XNNPACK 对其它四段已生效但总收益有限（enc+fsdec+vits+bert 合计仅 3.9 s）。要真正提速必须让 sdec 走出 CPU。

### 17.4 sdec 提速的前置验证（已做）
- 写 `scripts/dev/probe_yemb_formula.py`：喂真实张量跑一步 sdec，比较图输出 `y_emb` 末行与 `audio_embedding[token]`。结果 `cos = 0.99902`、`max|diff| = 0.2386`，`corr(diff, sin/cos) < 0.1` —— 说明**图内 `y_emb` 末行基本就是 `audio_embedding[token]`（外加很小扰动）**，Kotlin 端完全可以自算，具备裁掉尾部采样链（`ScatterElements`/`Softmax`/`ArgMax`）的条件。
- 下一步：用 `onnx.utils.extract_model` 把 `t2s_sdec` 输出裁成 `["logits","k","v"]`，Kotlin 自维护 `y` 与 `y_emb` 历史，再分别测 `NNAPI(FP16)` / `XNNPACK` / FP16 权重的真机速度。

### 17.5 仍未完成
① 裁剪版 sdec 的导出与真机 A/B；② `vits.fp16.onnx` / `t2s_fsdec.fp16.onnx` 的真机对比；③ QNN SDK 库随包投递方案（体积 +200 MB 量级，需权衡）；④ 真机 ASR 复验（本轮 NNAPI 挂死、CPU 结果 19.3 s 未做 ASR）。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "## 17. 硬件加速路由改造" in text:
    print("SKIP: already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"
text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
