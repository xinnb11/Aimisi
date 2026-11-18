/**
 * StreamRegexParser.ts - 流式动作剥离、情绪提取与短句切分管线
 * 
 * 双路分流：
 * 1. UI 渲染路：动作描写转为浅灰斜体，台词高亮，情绪标签转为光感胶囊
 * 2. TTS 合成路：彻底剥离括号动作描写，提取情绪标签，按标点切为即时可读短句
 */

export type SupportedEmotion = '平静' | '开心' | '温柔' | '调侃' | '愤怒生气';

export interface ParsedMessage {
  raw: string;
  actionText: string;
  emotion: SupportedEmotion;
  cleanSpeech: string;
  formattedHtml: string;
}

export interface SplitSentenceItem {
  sentence: string;
  emotion: SupportedEmotion;
  index: number;
}

export class StreamRegexParser {
  private static readonly ACTION_REGEX = /（[^）]*）|\([^)]*\)/g;
  private static readonly EMOTION_REGEX = /\[(平静|开心|温柔|调侃|愤怒生气|愤怒|生气)\]/g;
  private static readonly PUNCTUATION_SPLITTER = /([，。！？～；\n,.!?~;]+)/g;

  private static normalizeEmotion(rawEmo: string | undefined, fallback: SupportedEmotion = '温柔'): SupportedEmotion {
    if (!rawEmo) return fallback;
    if (rawEmo === '愤怒' || rawEmo === '生气' || rawEmo === '愤怒生气') return '愤怒生气';
    if (rawEmo === '平静' || rawEmo === '开心' || rawEmo === '温柔' || rawEmo === '调侃') return rawEmo;
    return fallback;
  }

  /**
   * 解析单条完整或累积文本
   */
  public static parseMessage(rawText: string, fallbackEmotion: SupportedEmotion = '温柔'): ParsedMessage {
    const raw = rawText || '';

    // 1. 提取所有圆括号内的神态动作
    const actionMatches = raw.match(StreamRegexParser.ACTION_REGEX) || [];
    const actionText = actionMatches.map(m => m.trim()).join(' ');

    // 2. 提取情绪标签（若未提取到则继承上下文 fallbackEmotion）
    const emoMatch = raw.match(/\[(平静|开心|温柔|调侃|愤怒生气|愤怒|生气)\]/);
    const emotion: SupportedEmotion = StreamRegexParser.normalizeEmotion(emoMatch ? emoMatch[1] : undefined, fallbackEmotion);

    // 3. 彻底剥离动作与情绪标签，得到 100% 纯净可读台词
    const cleanSpeech = raw
      .replace(StreamRegexParser.ACTION_REGEX, '')
      .replace(StreamRegexParser.EMOTION_REGEX, '')
      .replace(/\s+/g, ' ')
      .trim();

    // 4. 组装富文本渲染 HTML (UI 渲染路)
    const formattedHtml = StreamRegexParser.formatForUi(raw);

    return {
      raw,
      actionText,
      emotion,
      cleanSpeech,
      formattedHtml
    };
  }

  /**
   * 将含有动作与情绪的生文本转换为 iOS 26 液态玻璃样式 HTML
   */
  public static formatForUi(rawText: string): string {
    if (!rawText) return '';

    // 转义 HTML 基础字符防止 XSS
    let escaped = rawText
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // 将圆括号动作转换为精致浅灰色斜体
    escaped = escaped.replace(/（([^）]*)）/g, (_match, p1) => {
      return `<span class="text-xs text-slate-400 italic block my-1 font-normal select-text">（${p1}）</span>`;
    });
    escaped = escaped.replace(/\(([^)]*)\)/g, (_match, p1) => {
      return `<span class="text-xs text-slate-400 italic block my-1 font-normal select-text">（${p1}）</span>`;
    });

    // 将情绪标签转换为微光芯片
    escaped = escaped.replace(/\[(平静|开心|温柔|调侃|愤怒生气|愤怒|生气)\]/g, (_match, emo) => {
      return `<span class="inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-full accent-pill mr-1 align-middle select-none">${emo}</span>`;
    });

    return escaped;
  }

  /**
   * 将纯台词文本切分为自然发音短句，供本地 ONNX 引擎边推边播
   */
  public static splitSpeechSentences(cleanSpeech: string, emotion: SupportedEmotion): SplitSentenceItem[] {
    if (!cleanSpeech) return [];

    const tokens = cleanSpeech.split(StreamRegexParser.PUNCTUATION_SPLITTER);
    const results: SplitSentenceItem[] = [];
    let currentChunk = '';

    for (let i = 0; i < tokens.length; i++) {
      const part = tokens[i];
      if (!part) continue;

      if (StreamRegexParser.PUNCTUATION_SPLITTER.test(part)) {
        currentChunk += part;
        // 如果当前累积句子达到一定发音长度，或者遇到强句号/问号/感叹号
        if (currentChunk.trim().length >= 4 || /[。！？\n!?]/.test(part)) {
          const trimmed = currentChunk.trim();
          if (trimmed) {
            results.push({
              sentence: trimmed,
              emotion,
              index: results.length
            });
          }
          currentChunk = '';
        }
      } else {
        currentChunk += part;
      }
    }

    if (currentChunk.trim()) {
      results.push({
        sentence: currentChunk.trim(),
        emotion,
        index: results.length
      });
    }

    return results;
  }
}
