import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
### 16.18 真机三件验收（2026-11-17 19:56）

用户重新插上 USB 后安装最新包（`adb install -r` Success，设备 `JJFIE6AULVDIMR95`，model 2604FRK1EC），并用 CDP 逐项取证。

#### 16.18.1 首页无演示文案
```json
{"msgs":0,"streamKids":0,"stageAction":"","stageSpeech":""}
```
冷启动后聊天列表 `#chat-stream-container` 子节点数 = 0，立绘模式的示范动作/台词均为空。

#### 16.18.2 语调按条件包可用性
```json
{"engine":"ONNX Runtime",
 "emotionsNative":["平静","开心","温柔","调侃"],
 "availableEmotions":["平静","开心","温柔","调侃"],
 "emotion":"温柔","persisted":"温柔"}
```
当前包四个情绪都在，所以四个胶囊正常显示；随后做了两组真机验证：
1. **拦截不可用情绪**：`applyEmotionAvailability(['温柔'])` 后 `onEmotionChange(0)`（平静）→ `state.emotion` 仍为「温柔」，胶囊 `hidden` 状态为 `平静:true 开心:true 温柔:false 调侃:true`。
2. **持久化跨重启**：切到「调侃」→ `localStorage['aemeath_emotion_v1'] = "调侃"` → `am force-stop` + 重启 → 读回 `{"emotion":"调侃","persisted":"调侃"}`；最后切回「温柔」。

#### 16.18.3 语音包「未开放」
```json
[{"id":"aemeath-vits-lite","installed":false,"btn":"未开放","locked":true},
 {"id":"aemeath-sovits-pro","installed":true,"btn":null,"locked":null},
 {"id":"aemeath-indextts-ultra","installed":false,"btn":"未开放","locked":true}]
```
Pro 已装，因此不在「待拉取」列表里（渲染到已挂载列表）；Lite / Ultra 均为锁图标 + `cursor-not-allowed`。真机截图 `scripts/dev/fixtures/_acc_voice.png` 与实测一致。

#### 16.18.4 顺带复测 TTS 未回归
装机后 `synthesizeSpeech({text:'首页已经很干净了。', emotion:'温柔', playImmediately:true})` → `{"ms":4663,"success":true,"durationMs":1800,"queued":1,"played":true}`，冷启动自动挂载会把 `engine` 从「未装载」拉起到 `ONNX Runtime`。

#### 16.18.5 设备侧小坑
`adb start-server` 偶发「daemon not running / cannot connect」；处理：先 `adb start-server` 再 `adb devices`，`adb forward` 需重新 `--remove-all` 后按最新 `pidof` 重建（应用重启后 webview devtools socket 名会变）。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "### 16.18 真机三件验收" in text:
    print("SKIP: section already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"

text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
