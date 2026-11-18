import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
### 16.15 语调只保存真实可用的选项（2026-11-17 17:35）

用户要求：「把语音设置调一下，现在有哪些语调就保存哪些语音选项」。
此前的实现有四个问题：①`this.emotions` 写死四个情绪，条件包里没有的也会展示；②`state.emotion` 从不落盘，重启回到「温柔」；③语音页 `.bin` 试听胶囊恒显示 4 个，点了没有对应特征文件也只是空响；④底部凸透镜滑块四段全可点。

#### 16.15.1 原生：`diagnose()` 上报可用情绪
`android/app/src/main/java/com/aemeath/terminal/AemeathTtsPlugin.kt`

- 新增常量 `EMOTION_ORDER = listOf("平静", "开心", "温柔", "调侃")`。
- 新增 `private fun conditioningBlobReady(dir: File): Boolean`：目录里存在 `ref_seq.i64` **或** `ssl_content.f32` 即视为该情绪可用（兼容只有语义码或只有 SSL 特征的包）。
- `diagnose()` 扫描 `<modelDir>/conditioning/` 子目录，按 `EMOTION_ORDER` 排序后写入 `payload.emotions` 与 `inventory.emotions`（两个位置都给，前端任取其一）。
- 真机实测量到的顺序：`["平静","开心","温柔","调侃"]`（当前包四个都在）。

#### 16.15.2 前端：可用性 + 持久化
`index.html`（内联 JS）

| 位置 | 改动 |
|---|---|
| `AppController` 构造 | 新增 `this.availableEmotions = this.emotions.slice()`（未拿到原生信息前默认全开，浏览器降级也全开） |
| `applyEmotionAvailability(list)`（新增） | 过滤出真实可用项；语音页 `#btn-emo-*` 不可用的加 `hidden` + `disabled`；底部 `#slider-emotion` 不可用段 `opacity:0.32` + `pointer-events:none` + `aria-disabled`；若 `state.emotion` 不在列表则回落到「温柔」或首项 |
| `highlightEmotion(emo)`（新增） | 统一高亮：胶囊 className + 滑块 `data-active-index` + `is-active`，并调用滑块自带的 `__syncSnapLayout(idx,false)`；**不重建 DOM**，避免破坏凸透镜弹簧物理状态 |
| `persistEmotion()` / `loadEmotion()`（新增） | `localStorage['aemeath_emotion_v1']`；`init()` 里先 `loadEmotion()` 再 `highlightEmotion()` |
| `onEmotionChange(idx)` | 命中不可用情绪时 toast「当前语音包没有「X」情绪特征，已保留「Y」」并回滚高亮，不写 state |
| `auditionEmotionBin(emo)` | 同样前置守卫；改用 `highlightEmotion` + `persistEmotion` |
| `syncNativeModelState()` | 拿到 `info.emotions` / `info.inventory.emotions` 后调用 `applyEmotionAvailability()`（冷启动、切回语音页、点「重新检测」都会走到） |

#### 16.15.3 修复过程中发现的真 bug
`highlightEmotion()` 第一版用 `btn.className = classOf(...)` 重设整串类名，把 `applyEmotionAvailability()` 刚加的 `hidden` 又清掉了 —— 表现为「`disabled:true` 但 `hidden:false`，胶囊还能看见」。修法：设置 `className` 之后按 `availableEmotions` 重新 `classList.toggle('hidden', unavailable)` 并同步 `btn.disabled`。

#### 16.15.4 无头浏览器验证（`scripts/dev/probe_emotion_avail.js`）
注入 `applyEmotionAvailability(['温柔','调侃'])` 后：

```json
{"avail":["温柔","调侃"],
 "pills":[{"e":"平静","hidden":true,"disabled":true},{"e":"开心","hidden":true,"disabled":true},
          {"e":"温柔","hidden":false,"disabled":false},{"e":"调侃","hidden":false,"disabled":false}],
 "segs":[{"i":0,"opacity":"0.32","pe":"none"},{"i":1,"opacity":"0.32","pe":"none"},
         {"i":2,"opacity":"","pe":"","active":true},{"i":3,"opacity":"","pe":""}],
 "persisted":"温柔","afterBlocked":"温柔","emotionNow":"温柔"}
```

- 恢复全量 `['平静','开心','温柔','调侃']` 后四个胶囊全部 `hidden:false`，可逆。
- 点不可用项（索引 1 开心）时 `state.emotion` 保持「温柔」，未被改写。
- `_check.js` 加载自检 `{"errors":[],"hasController":true}`。

#### 16.15.5 产物
- `npx vite build` 177 ms → `npx cap sync android` → `gradlew assembleDebug`：APK **21,751,442 B**。
- 本轮 `adb devices` 为空（设备未连接），**真机安装与实机验证待用户插 USB 后补**；无头验证已覆盖 DOM 行为与持久化。

#### 16.15.6 设计取舍
- 底部滑块**不删段**：`ConvexLiquidLensSlider` 按 `data-segments="4"` 建立弹簧与光路，删段会让物理引擎索引错位；采用「置灰 + 不可点 + 回落」的方式满足「只保存真实可用的选项」。
- 若将来某包只有 1–2 个情绪，滑块仍显示 4 段槽位，但不可用段点不动、也不会被持久化选中。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "### 16.15 语调只保存真实可用的选项" in text:
    print("SKIP: section already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"

text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
