import io, sys

PATH = r"D:\语音模型项目\PROGRESS.md"

section = """
### 16.17 首页去掉预置展示文案（2026-11-17 18:05）

用户要求：「把首页的展示文字去掉」。确认口径后明确指的是**对话页（聊天首页）**，不是启动 Portal 首屏。

#### 16.17.1 删掉的内容
| 位置 | 原内容 | 现状态 |
|---|---|---|
| `AppStore` 初始 `state.messages` | 三条演示对话：用户「爱弥斯，我已经把你的两张原图设为主头像和预备头像啦…」+ 两条爱弥斯回复（含 `[开心]`/`[温柔]` 标签与括号动作） | `messages: []`，附注释「首次进入不预置任何演示对话」 |
| `#stage-action-text`（立绘模式） | 「（单眼眨了眨 Wink，右手轻托脸颊，胸前水晶爱心泛起柔光）」 | 空字符串 |
| `#stage-speech-text`（立绘模式） | 「“漂泊者，我已经换上圆形头像框啦！…”」 | 空字符串 |
| 立绘头像下方说明 | 「点击爱弥斯圆形立绘触发本地语音 · 点击右下角相机切换预备头像」 | 空字符串（元素保留，避免布局跳动） |

没有动的：
- 语调条「平静/开心/温柔/调侃」、输入框占位「发送消息给爱弥斯…」——属于操作控件文案；
- `pokeCharacter()` 的三条随机回应——用户主动点头像才出现，属于交互反馈；
- 启动 Portal 的全部文字——用户明确说不是那里。

#### 16.17.2 无头浏览器验收（`scripts/dev/probe_chat_home_clean.js`）

```json
{"messagesLength":0,"streamChildren":0,"streamText":"",
 "stageActionText":"","stageSpeechText":"","pageErrors":[]}
```

截图 `scripts/dev/fixtures/_chat_home_clean.png`：对话页只剩顶部「对话 / 立绘」切换、底部语调条与输入框，中间完全空白，不再有假对话气泡。

#### 16.17.3 产物
`npx vite build` 190 ms → `npx cap sync android` → `gradlew assembleDebug`：APK **21,751,664 B**，SHA256 `AE11303D8704E1E37AA27D89358AE1FD8AF0DA295D4F95879CC8D723F3DF7B50`。
设备当前未连接，装机待补。

#### 16.17.4 说明
- 演示文案此前是「首次打开看起来有内容」的填充，删掉后首屏为空；真实对话从用户第一条输入开始渲染。
- `replayLastVoice()` 已有 `if (!lastAi) return;` 守卫，无历史消息时点喇叭不会报错。
"""

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

if "### 16.17 首页去掉预置展示文案" in text:
    print("SKIP: section already present")
    sys.exit(0)

if not text.endswith("\n"):
    text += "\n"

text += section

with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("OK lines:", text.count("\n") + 1)
