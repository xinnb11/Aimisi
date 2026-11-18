import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
### 16.16 语音包「未开放」闸门（2026-11-17 17:50）

用户要求：「再设置一下语音包，单用户想拉取其他语音包时显示未开放」。
现状是三个预置槽位都挂着「一键拉取」，但只有 Pro（`xinxin111x/Aimisi`）是真包：
- `aemeath-vits-lite`：`model.int8.onnx` 只有 65,589 B（占位），不可推理；
- `aemeath-indextts-ultra`：仓库 `aemeath/indextts2-aemeath-ultra-onnx` 从未发布。

#### 16.16.1 预置卡片
`index.html` 的 `defaultVoiceModels` 给两个不可用槽位加 `notOpen: true` 与 `notOpenReason`；`renderVoiceView()` 渲染时：

- 卡片整体加 `opacity-80`，标题转灰；
- 标题行多一个「未开放」徽章；
- 右侧按钮换成 `<i class="fa-solid fa-lock"></i> 未开放`（`text-slate-400` + `cursor-not-allowed`，无 hover/active 反馈），`onclick` 改指 `notifyModelNotOpen(id)`；
- 卡片底部补一行灰字说明原因（如「暂未开放：该轻量包权重仍是占位文件（65 KB），无法推理」）。

`pullPresetModel(modelId)` 首部新增守卫：`if (model.notOpen) { this.notifyModelNotOpen(modelId); return; }`——即使有人从控制台直接调这个函数也进不了下载流程。

`notifyModelNotOpen(modelId)` 统一 toast：`${模型名}：${原因}`。

#### 16.16.2 魔搭指令面板（自定义仓库）
同一面板的架构滑块（Bert-VITS2 / GPT-SoVITS v2 / IndexTTS2）也做了闸门：

- `onNewModelArchChange(idx)`：非索引 1 时徽章追加「· 暂未开放」并转玫红色，同时 toast「该架构语音包暂未开放：当前仅开放 GPT-SoVITS v2（Pro 高还原旗舰）」；
- 执行指令的入口（`btn-custom-pull`）在读取 `input-modelscope-cmd` 之后、置 `isDownloading` **之前**拦截：`requestedArch !== 1` 直接 `notifyModelNotOpen(...)` + `console.info('[Voice] 自定义拉取被拦截：arch=…（未开放）')` 并 return。

#### 16.16.3 无头浏览器验收（`scripts/dev/probe_voice_notopen.js`）

```json
{"cards":[
  {"id":"aemeath-vits-lite","btnText":"未开放","icon":"fa-solid fa-lock","cardOpacity":true,"disabledLooking":true},
  {"id":"aemeath-sovits-pro","btnText":"一键拉取","icon":"fa-solid fa-cloud-arrow-down accent-text","cardOpacity":false,"disabledLooking":false},
  {"id":"aemeath-indextts-ultra","btnText":"未开放","icon":"fa-solid fa-lock","cardOpacity":true,"disabledLooking":true}],
 "afterClickDownloading":false,
 "downloadingAfterCustom":false,
 "toastText":["暂未开放：八维情绪解耦版权重仍在整理，当前仅开放 Pro 高还原旗舰",
              "爱弥斯 · 星炬极速版：暂未开放：该轻量包权重仍是占位文件（65 KB），无法推理"]}
```

- 点「未开放」卡片不会进入 `isDownloading`；
- 自定义面板选 Bert-VITS2 再点执行，`isDownloading` 仍为 false，控制台出现拦截日志；
- 页面自检 `{"errors":[],"hasController":true}`。

#### 16.16.4 产物
`npx vite build` 178 ms → `npx cap sync android` → `gradlew assembleDebug`：APK **21,752,275 B**，SHA256 `6E36F9460E76491A8BA8C066847E1E5659047C15D74C65C6B8E11D14625AA0FB`。
设备当前未连接（`adb devices` 为空），真机安装待补。

#### 16.16.5 口径说明
- 已安装的模型不受影响：Pro 拉取成功后仍在「已挂载模型列表」里正常使用；
- 「未开放」不影响 `conditioning/` 情绪可用性判断（§16.15），两者独立；
- 将来某包真的要开放，只需删掉该槽位的 `notOpen` 字段即可恢复「一键拉取」。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "### 16.16 语音包「未开放」闸门" in text:
    print("SKIP: section already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"

text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
