/**
 * ChatView.ts - 对话与立绘互动视图控制器
 */

import { AppStore, ChatMessage } from '../../core/store/AppStore';
import { LocalOnnxTtsBridge } from '../../core/tts/LocalOnnxTtsBridge';
import { StreamRegexParser } from '../../core/parser/StreamRegexParser';
import { AvatarCustomizationModal } from './AvatarCustomizationModal';

export class ChatView {
  private store: AppStore;
  private tts: LocalOnnxTtsBridge;
  private avatarModal: AvatarCustomizationModal;
  private chatContainer: HTMLElement | null = null;
  private stageContainer: HTMLElement | null = null;

  constructor(store: AppStore, tts: LocalOnnxTtsBridge, avatarModal: AvatarCustomizationModal) {
    this.store = store;
    this.tts = tts;
    this.avatarModal = avatarModal;
    this.chatContainer = document.getElementById('chat-stream-container');
    this.stageContainer = document.getElementById('character-stage-container');
  }

  public init(): void {
    this.renderMessages();
    this.tts.onPlaybackChange((isPlaying) => {
      const activeWaves = document.querySelectorAll('.wave-bar.playing-wave');
      if (!isPlaying) {
        activeWaves.forEach(b => b.classList.remove('playing-wave'));
      }
    });
  }

  public setDisplayMode(modeIdx: number): void {
    this.store.update(state => {
      state.displayMode = modeIdx === 0 ? 'terminal' : 'standee';
    });
    if (this.chatContainer) this.chatContainer.classList.toggle('hidden', modeIdx === 1);
    if (this.stageContainer) this.stageContainer.classList.toggle('hidden', modeIdx === 0);
  }

  public renderMessages(): void {
    if (!this.chatContainer) return;
    const avatarHtml = this.avatarModal.getAvatarHtml();

    this.chatContainer.innerHTML = this.store.state.messages.map((msg: ChatMessage, idx: number) => {
      if (msg.role === 'user') {
        return `
          <div class="flex justify-end animate-fadeIn">
            <div class="max-w-[80%] luminous-user-bubble rounded-3xl rounded-br-lg px-4 py-2.5 text-sm leading-relaxed">
              ${this.escape(msg.content || '')}
            </div>
          </div>
        `;
      } else {
        const parsed = StreamRegexParser.parseMessage(msg.rawContent || msg.content || '');
        return `
          <div class="flex items-start gap-2.5 animate-fadeIn">
            <!-- 标准正圆形 AI 头像框：点击可随时在主头像与预备头像之间切换 -->
            <div onclick="window.AppController.openAvatarModal()"
                 title="点击切换主头像 / 预备头像"
                 class="group relative mt-0.5 cursor-pointer shrink-0">
              <div class="circular-avatar-frame w-10 h-10">
                ${avatarHtml}
              </div>
              <span class="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-white shadow flex items-center justify-center text-[8px] accent-text">
                <i class="fa-solid fa-repeat"></i>
              </span>
            </div>
            <div class="max-w-[81%] clean-card px-4 py-3 space-y-2 rounded-3xl rounded-tl-lg">
              ${parsed.actionText ? `
                <div class="text-xs text-slate-400 italic leading-snug">
                  ${this.escape(parsed.actionText)}
                </div>
              ` : ''}
              <div class="text-sm text-slate-800 leading-relaxed">
                ${this.escape(parsed.cleanSpeech)}
              </div>
              <div onclick="window.AppController.playBubbleVoice(${idx}, this)"
                   class="inline-flex items-center gap-2.5 px-3 py-1 rounded-full accent-pill cursor-pointer transition active:scale-95">
                <i class="fa-solid fa-play text-[10px]"></i>
                <div class="flex items-center gap-0.5 h-3">
                  ${[45, 90, 60, 100, 70, 85, 50, 75].map(h => `
                    <span class="wave-bar w-0.5 rounded-full" style="height:${h}%; background: var(--accent-color);"></span>
                  `).join('')}
                </div>
                <span class="text-[11px] font-semibold">${parsed.emotion}</span>
              </div>
            </div>
          </div>
        `;
      }
    }).join('');

    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  public playBubbleVoice(msgIndex: number, btnEl?: HTMLElement): void {
    const msg = this.store.state.messages[msgIndex];
    if (!msg) return;

    const parsed = StreamRegexParser.parseMessage(msg.rawContent || msg.content || '');
    if (btnEl) {
      const bars = btnEl.querySelectorAll('.wave-bar');
      bars.forEach(b => b.classList.add('playing-wave'));
    }

    const textToSpeak = this.store.state.stripActions ? parsed.cleanSpeech : (msg.rawContent || msg.content || '');
    this.tts.speak(textToSpeak, parsed.emotion, () => {
      if (btnEl) {
        const bars = btnEl.querySelectorAll('.wave-bar');
        bars.forEach(b => b.classList.remove('playing-wave'));
      }
    });
  }

  public replayLastVoice(): void {
    const lastAi = [...this.store.state.messages].reverse().find(m => m.role === 'assistant');
    if (!lastAi) return;
    const parsed = StreamRegexParser.parseMessage(lastAi.rawContent || lastAi.content || '');
    this.tts.speak(parsed.cleanSpeech, parsed.emotion);
  }

  public pokeCharacter(): void {
    const replies = [
      '（单眼眨了眨 Wink，右手托着脸颊笑出声）[开心]漂泊者，我这张眨眼微笑的主头像和另一张比耶抱黑猫的预备头像，你更喜欢哪一张呀？',
      '（抱着怀里的黑猫玩偶轻轻晃了晃）[温柔]不管长按滑块多久再松手，弹簧引擎都会让光芒与水晶透镜一起丝滑回弹哦。',
      '（嘴角扬起一抹俏皮的弧度）[调侃]一直戳我，是想看我表演一秒换装还是想听我唱歌呢？'
    ];
    const pick = replies[Math.floor(Math.random() * replies.length)];
    const parsed = StreamRegexParser.parseMessage(pick);

    const actionEl = document.getElementById('stage-action-text');
    const speechEl = document.getElementById('stage-speech-text');
    if (actionEl) actionEl.textContent = parsed.actionText;
    if (speechEl) speechEl.textContent = `“${parsed.cleanSpeech}”`;

    this.tts.speak(parsed.cleanSpeech, parsed.emotion);
  }

  private escape(str: string): string {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}
