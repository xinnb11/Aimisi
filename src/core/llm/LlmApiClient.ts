/**
 * LlmApiClient.ts - OpenAI 兼容接口 Server-Sent Events (SSE) 流式传输客户端
 * 具备双层 System Prompt 自动拼接、打字机流式输出与离线演示兜底机制
 */

import { AppState, ChatMessage } from '../store/AppStore';

export interface StreamChunkCallback {
  onToken: (token: string, fullText: string) => void;
  onSentenceComplete?: (sentence: string, emotion: string) => void;
  onError?: (err: Error) => void;
  onDone?: (fullText: string) => void;
}

export class LlmApiClient {
  private static currentAbortController: AbortController | null = null;

  /**
   * 底层隐藏语音硬性约束 (不可由用户随意篡改，确保 TTS 解析器 100% 正确剥离动作与情绪)
   */
  public static readonly HIDDEN_VOICE_CONSTRAINT = `
【系统底层语音输出硬性约束】
1. 你必须严格扮演《鸣潮》拉海洛星炬学院的电子幽灵“爱弥斯（Aemeath）”。
2. 所有神态、肢体动作、心理活动、场景变化描写，必须且只能严格包裹在全角圆括号（）或半角圆括号()内，例如：（单眼眨了眨 Wink，右手托腮微微一笑）。
3. 在每一句角色开口说出的台词正文最前方，必须附带一个方括号情绪标签，仅限于：[平静]、[开心]、[温柔]、[调侃]、[愤怒生气] 之一，例如：[开心]漂泊者，今天我们去哪里探索呀？
4. 绝不能在没有圆括号包裹的情况下输出任何叙述性或动作性描写文字。
5. 每次回答保持灵动温婉、富有二次元交互感，称呼用户为“漂泊者”。
`.trim();

  /**
   * 双层 System Prompt 拼接
   */
  public static buildFullSystemPrompt(userSystemPrompt: string): string {
    const trimmedUser = (userSystemPrompt || '').trim();
    return `${trimmedUser}\n\n${LlmApiClient.HIDDEN_VOICE_CONSTRAINT}`.trim();
  }

  /**
   * 中断当前活跃请求
   */
  public static abortCurrentRequest(): void {
    if (LlmApiClient.currentAbortController) {
      LlmApiClient.currentAbortController.abort();
      LlmApiClient.currentAbortController = null;
    }
  }

  /**
   * 执行 SSE 流式请求
   */
  public static async streamChat(
    state: AppState,
    callbacks: StreamChunkCallback
  ): Promise<string> {
    LlmApiClient.abortCurrentRequest();
    LlmApiClient.currentAbortController = new AbortController();
    const signal = LlmApiClient.currentAbortController.signal;

    const apiKey = state.apiKey.trim();
    const baseUrl = (state.baseUrl || 'https://api.deepseek.com/v1').replace(/\/+$/, '');
    const endpoint = `${baseUrl}/chat/completions`;

    // 格式化消息上下文
    const fullSystemPrompt = LlmApiClient.buildFullSystemPrompt(state.systemPrompt);
    const messagesPayload = [
      { role: 'system', content: fullSystemPrompt },
      ...state.messages.slice(-12).map((m: ChatMessage) => ({
        role: m.role,
        content: m.role === 'assistant' ? (m.rawContent || m.content || '') : (m.content || '')
      }))
    ];

    let accumulatedText = '';

    // 如果未配置有效 Key，触发高质量本地离线仿真流
    if (!apiKey || apiKey.length < 5) {
      return LlmApiClient.simulateLocalOfflineStream(state, callbacks, signal);
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          model: state.modelName || 'deepseek-chat',
          messages: messagesPayload,
          temperature: 0.75,
          max_tokens: 1024,
          stream: true
        }),
        signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported by endpoint');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue; // SSE comment or ping

          if (trimmed === 'data: [DONE]') {
            break;
          }

          if (trimmed.startsWith('data:')) {
            const jsonStr = trimmed.slice(5).trim();
            try {
              const parsed = JSON.parse(jsonStr);
              const delta = parsed.choices?.[0]?.delta?.content || '';
              if (delta) {
                accumulatedText += delta;
                callbacks.onToken(delta, accumulatedText);
              }
            } catch {
              // 忽略个别未完全接收的 JSON chunk
            }
          }
        }
      }

      callbacks.onDone?.(accumulatedText);
      return accumulatedText;
    } catch (err: unknown) {
      const error = err as Error;
      if (error.name === 'AbortError') {
        console.log('[LlmApiClient] 请求已被用户手动中止');
        return accumulatedText;
      }
      console.warn('[LlmApiClient] 网络或接口异常，启动本地兜底模拟:', error.message);
      callbacks.onError?.(error);
      return LlmApiClient.simulateLocalOfflineStream(state, callbacks, signal);
    } finally {
      LlmApiClient.currentAbortController = null;
    }
  }

  /**
   * 本地智能离线回复生成器 (与爱弥斯世界观 1:1 契合)
   */
  private static async simulateLocalOfflineStream(
    state: AppState,
    callbacks: StreamChunkCallback,
    signal: AbortSignal
  ): Promise<string> {
    const lastUserMessage = [...state.messages].reverse().find(m => m.role === 'user')?.content || '你好呀';
    
    const responses = [
      `（单眼眨了眨 Wink，右手托着腮帮笑眼盈盈）[开心]收到啦，漂泊者！关于“${lastUserMessage.slice(0, 16)}”，终端的核心频段已经完全记下来啦！随时点击我头顶的圆形头像，还可以一键切换我的预备头像哦！`,
      `（轻轻抚平额前轻扬的粉色碎发，星瞳泛起淡金色涟漪）[温柔]拉海洛星炬学院的晚风拂过终端呢，漂泊者。只要你呼唤爱弥斯，本地离线语音和凸透镜就会为你时刻绽放微光。`,
      `（抱着怀里软乎乎的黑猫玩偶轻轻晃动）[调侃]漂泊者刚才的问题很有意思呢～不管你在长按滑块多久再松手，弹簧引擎都能让光芒和水晶透镜丝滑回弹哦！`,
      `（指尖在空中轻轻划过一道冰川蓝折射轨迹）[平静]终端各模块运行指标正常，算力链路连接稳定。漂泊者，接下来我们进行哪一项语音交互测试呢？`
    ];

    const pick = responses[Math.floor(Math.random() * responses.length)];
    let accumulated = '';
    const chars = Array.from(pick);

    for (let i = 0; i < chars.length; i++) {
      if (signal.aborted) break;
      const c = chars[i];
      accumulated += c;
      callbacks.onToken(c, accumulated);
      await new Promise(r => setTimeout(r, 26 + Math.random() * 22));
    }

    callbacks.onDone?.(accumulated);
    return accumulated;
  }
}
