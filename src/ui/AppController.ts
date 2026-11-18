/**
 * AppController.ts - 主应用控制器
 * 协调全局状态管理、物理弹簧光学透镜、大模型 SSE 流式通信、实时切句与本地离线 TTS
 */

import { AppStore } from '../core/store/AppStore';
import { LocalOnnxTtsBridge } from '../core/tts/LocalOnnxTtsBridge';
import { LlmApiClient } from '../core/llm/LlmApiClient';
import { StreamRegexParser } from '../core/parser/StreamRegexParser';
import { SpringConvexLensController } from './physics/SpringConvexLensController';
import { AvatarCustomizationModal } from './components/AvatarCustomizationModal';
import { BackgroundSystem } from './components/BackgroundSystem';
import { ChatView } from './components/ChatView';
import { VoiceView } from './components/VoiceView';
import { SettingsView } from './components/SettingsView';

export class AppController {
  public store: AppStore;
  public tts: LocalOnnxTtsBridge;
  public avatarModal: AvatarCustomizationModal;
  public backgroundSystem: BackgroundSystem;
  public chatView: ChatView;
  public voiceView: VoiceView;
  public settingsView: SettingsView;

  private toastTimer: ReturnType<typeof setTimeout> | null = null;
  private isGenerating: boolean = false;

  constructor() {
    this.store = AppStore.getInstance();
    this.tts = LocalOnnxTtsBridge.getInstance();
    this.avatarModal = new AvatarCustomizationModal(this.store);
    this.backgroundSystem = new BackgroundSystem(this.store);
    this.chatView = new ChatView(this.store, this.tts, this.avatarModal);
    this.voiceView = new VoiceView(this.store, this.tts);
    this.settingsView = new SettingsView(this.store);
  }

  public init(): void {
    // 1. 初始化各子模块
    this.backgroundSystem.init();
    this.avatarModal.init();
    this.chatView.init();
    this.settingsView.init();
    this.applyLensPhysicsToDom();
    this.applyBoxGlassToDom();
    this.initCollapsibles();

    // 2. 挂载 1ms 阻尼弹簧与凸透镜物理引擎
    SpringConvexLensController.initAll(this as unknown as Record<string, unknown>);

    // 3. 挂载全局对象供 DOM 行内 onclick 调用
    (window as unknown as { AppController: AppController }).AppController = this;

    console.log('[Aemeath-Terminal] 鸣潮爱弥斯移动端终端启动完成，100% 开源 (AGPL-3.0)');
  }

  /**
   * ==================== 发送消息与流式切句播报 ====================
   */
  public async sendMessage(): Promise<void> {
    const input = document.getElementById('chat-input') as HTMLInputElement | null;
    if (!input) return;

    const text = input.value.trim();
    if (!text || this.isGenerating) return;
    input.value = '';

    // 1. 推入用户消息
    this.store.update(state => {
      state.messages.push({
        role: 'user',
        content: text,
        timestamp: Date.now()
      });
    });
    this.chatView.renderMessages();

    // 2. 创建助手等待消息气泡
    const aiMessageIndex = this.store.state.messages.length;
    this.store.update(state => {
      state.messages.push({
        role: 'assistant',
        rawContent: `（微微颔首，胸前星芒闪烁）[${state.emotion}]正在连接拉海洛星炬学院核心频段...`,
        timestamp: Date.now()
      });
    });
    this.chatView.renderMessages();

    this.isGenerating = true;
    let spokenSentenceIndex = 0;
    let sentenceBuffer = '';

    try {
      await LlmApiClient.streamChat(this.store.state, {
        onToken: (token, fullText) => {
          // 实时更新气泡文字
          this.store.state.messages[aiMessageIndex].rawContent = fullText;
          this.chatView.renderMessages();

          if (!this.store.state.autoPlayVoice) return;

          // 边流式生成边切句送入 TTS 队列
          sentenceBuffer += token;
          if (/[，。！？～；\n,.!?~;]/.test(token)) {
            const parsed = StreamRegexParser.parseMessage(sentenceBuffer, this.store.state.emotion);
            if (parsed.cleanSpeech.length >= 2) {
              this.tts.enqueueSpeech(parsed.cleanSpeech, parsed.emotion);
              sentenceBuffer = '';
              spokenSentenceIndex++;
            }
          }
        },
        onDone: (fullText) => {
          this.store.state.messages[aiMessageIndex].rawContent = fullText;
          this.chatView.renderMessages();
          this.isGenerating = false;

          // 若有尾句未送入 TTS
          if (this.store.state.autoPlayVoice && sentenceBuffer.trim()) {
            const parsed = StreamRegexParser.parseMessage(sentenceBuffer, this.store.state.emotion);
            if (parsed.cleanSpeech) {
              this.tts.enqueueSpeech(parsed.cleanSpeech, parsed.emotion);
            }
          }
        },
        onError: (err) => {
          this.isGenerating = false;
          this.showToast(`连接受限：${err.message.slice(0, 24)}`);
        }
      });
    } catch {
      this.isGenerating = false;
    }
  }

  // ==================== 底部导航与视图切换 ====================

  public onBottomTabChange(idx: number): void {
    const views = ['view-chat', 'view-voice', 'view-settings'];
    views.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('hidden', i !== idx);
    });
    this.store.update(state => {
      state.currentView = (['chat', 'voice', 'settings'][idx] as 'chat' | 'voice' | 'settings');
    });
  }

  public onChatModeChange(idx: number): void {
    this.chatView.setDisplayMode(idx);
  }

  public onEmotionChange(idx: number): void {
    const emotions: ('平静' | '开心' | '温柔' | '调侃' | '愤怒生气')[] = ['平静', '开心', '温柔', '调侃', '愤怒生气'];
    const emo = emotions[idx] || '温柔';
    this.store.update(state => {
      state.emotion = emo;
    });
    this.tts.speak(`已切换为${emo}声线`, emo);
  }

  // ==================== 语音包与引擎切换 ====================

  public onEngineTypeChange(idx: number): void {
    this.voiceView.setEngineType(idx);
    this.showToast(idx === 0 ? '已切换至 Bert-VITS2 (Sherpa-ONNX 极速引擎)' : '已切换至 GPT-SoVITS 引擎');
  }

  public onComputeBackendChange(idx: number): void {
    const text = this.voiceView.setComputeBackend(idx);
    this.showToast(`硬件加速后端已切换至：${text}`);
  }

  public auditionEmotionBin(emotion: '平静' | '开心' | '温柔' | '调侃' | '愤怒生气'): void {
    this.voiceView.auditionEmotionBin(emotion);
    this.showToast(`正在试听：${emotion}.bin 特征向量`);
  }

  public previewVoice(): void {
    this.voiceView.previewVoice();
    this.showToast('正在播放爱弥斯离线试听语音');
  }

  public async importVoicePack(event: Event): Promise<void> {
    const target = event.target as HTMLInputElement;
    const file = target?.files?.[0];
    if (!file) return;
    const name = await this.voiceView.importVoicePack(file);
    this.showToast(`已成功挂载语音模型 DLC：${name}`);
  }

  public onAutoPlayToggle(checked: boolean): void {
    this.voiceView.setAutoPlay(checked);
    this.showToast(checked ? '已开启流式语音自动播报' : '已关闭流式语音自动播报');
  }

  public onStripActionToggle(checked: boolean): void {
    this.voiceView.setStripActions(checked);
    this.showToast(checked ? '已开启括号动作自动剥离' : '已关闭括号动作剥离');
  }

  // ==================== 头像管理 ====================

  public openAvatarModal(): void {
    this.avatarModal.open();
  }

  public closeAvatarModal(): void {
    this.avatarModal.close();
  }

  public selectAvatarPreset(idx: number): void {
    this.avatarModal.selectPreset(idx);
    const item = this.store.avatarGallery[idx];
    const roleLabel = idx === 0 ? '默认主头像' : '预备头像';
    this.chatView.renderMessages();
    this.showToast(`已切换为${roleLabel}：${item.title}`);
  }

  public uploadAvatarFile(event: Event): void {
    const target = event.target as HTMLInputElement;
    const file = target?.files?.[0];
    if (!file) return;
    this.avatarModal.uploadCustomFile(file);
    this.chatView.renderMessages();
    this.showToast('已更新本地上传的圆形 AI 头像');
  }

  public applyAvatarUrl(): void {
    const input = document.getElementById('input-avatar-url') as HTMLInputElement | null;
    const url = input?.value || '';
    if (!url) {
      this.showToast('请先输入有效的图片链接');
      return;
    }
    const ok = this.avatarModal.applyCustomUrl(url);
    if (ok) {
      this.chatView.renderMessages();
      this.showToast('已应用自定义网络圆形头像');
    }
  }

  // ==================== 背景管理 (严格默认纯白 #FFFFFF) ====================

  public onBgModeChange(idx: number): void {
    this.backgroundSystem.setMode(idx);
    if (idx === 0) {
      this.showToast('已切换为默认纯白背景 (#FFFFFF)');
    }
  }

  public applySolidBackground(hexColor: string, labelText?: string): void {
    this.backgroundSystem.setSolidColor(hexColor, labelText);
    this.showToast(`软件背景已设为：${labelText || hexColor}`);
  }

  public applyImageBackground(src: string, labelText?: string): void {
    this.backgroundSystem.setImageWallpaper(src, labelText);
    this.showToast(`已切换背景壁纸：${labelText || '自定义壁纸'}`);
  }

  public uploadBackgroundFile(event: Event): void {
    const target = event.target as HTMLInputElement;
    const file = target?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const res = e.target?.result;
      if (typeof res === 'string') {
        this.applyImageBackground(res, `相册壁纸 · ${file.name}`);
      }
    };
    reader.readAsDataURL(file);
  }

  public applyBackgroundUrl(): void {
    const input = document.getElementById('input-bg-image-url') as HTMLInputElement | null;
    const url = input?.value.trim() || '';
    if (!url) {
      this.showToast('请先输入有效的背景图片链接');
      return;
    }
    this.applyImageBackground(url, '网络自定义壁纸');
  }

  public updateBackgroundVeil(val: string): void {
    const num = parseInt(val, 10);
    this.backgroundSystem.setVeilOpacity(isNaN(num) ? 25 : num);
  }

  public onUltraGlassToggle(checked: boolean): void {
    this.backgroundSystem.toggleUltraGlass(checked);
    this.showToast(checked ? '已开启方框与文字柔光玻璃化' : '已恢复标准玻璃通透度');
  }

  public updateUltraGlassTransparency(val: string): void {
    const num = parseInt(val, 10);
    this.backgroundSystem.setGlassTransparency(isNaN(num) ? 72 : num);
  }

  public updateUltraGlassBlur(val: string): void {
    const num = parseInt(val, 10);
    this.backgroundSystem.setGlassBlur(isNaN(num) ? 8 : num);
  }

  public resetBackgroundToWhite(): void {
    this.backgroundSystem.resetToPureWhite();
    this.showToast('已恢复默认纯白背景 (#FFFFFF)');
  }

  // ==================== 设置管理 ====================

  public onThemeGlowChange(idx: number): void {
    const theme = this.settingsView.applyTheme(idx);
    this.showToast(`发光主题已切换为：${theme.name}`);
  }

  public onPersonaTemplateChange(idx: number): void {
    this.settingsView.applyPersonaTemplate(idx);
    if (idx === 3) {
      this.showToast('已载入自定义 System 提示词');
    } else {
      this.showToast('已切换角色 System 提示词模板');
    }
  }

  public saveCustomSystemPrompt(): void {
    const input = document.getElementById('input-system-prompt') as HTMLTextAreaElement | null;
    const val = (input ? input.value : this.store.state.systemPrompt || '').trim();
    if (!val) {
      this.showToast('提示词不能为空');
      return;
    }
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('aemeath_custom_system_prompt', val);
    }
    this.updateSystemPrompt(val);
    const slider = document.getElementById('slider-persona-template') as any;
    if (slider && slider._lensController) {
      slider._lensController.moveTo(3);
    }
    this.showToast('已保存自定义 System 提示词');
  }

  public resetSystemPrompt(): void {
    this.settingsView.applyPersonaTemplate(0);
    const slider = document.getElementById('slider-persona-template') as any;
    if (slider && slider._lensController) {
      slider._lensController.moveTo(0);
    }
    this.showToast('已恢复默认 System 提示词');
  }

  public onApiProviderChange(idx: number): void {
    this.settingsView.applyApiPreset(idx);
    this.saveApiConfig();
    const names = ['DeepSeek', '硅基流动', '自定义'];
    this.showToast(`API 已切换至：${names[idx] || '默认'}`);
  }

  public onViewportChange(idx: number): void {
    this.settingsView.setViewport(idx);
  }

  public saveApiConfig(): void {
    const baseUrl = (document.getElementById('input-base-url') as HTMLInputElement)?.value || '';
    const apiKey = (document.getElementById('input-api-key') as HTMLInputElement)?.value || '';
    const modelName = (document.getElementById('input-model-name') as HTMLInputElement)?.value || '';
    this.settingsView.saveApiCredentials(baseUrl, apiKey, modelName);
  }

  public updateSystemPrompt(val: string): void {
    this.settingsView.updateSystemPrompt(val);
  }

  public playBubbleVoice(msgIndex: number, btnEl?: HTMLElement): void {
    this.chatView.playBubbleVoice(msgIndex, btnEl);
  }

  public replayLastVoice(): void {
    this.chatView.replayLastVoice();
  }

  public pokeCharacter(): void {
    this.chatView.pokeCharacter();
  }


  /**
   * ==================== 长按滑块凸透镜光学与折射物理调节 ====================
   */
  public applyLensPhysicsToDom(): void {
    const { refraction, magnification, bulge, specular } = this.store.state.lensPhysics || {
      refraction: 0.16,
      magnification: 1.38,
      bulge: 1.30,
      specular: 0.95
    };

    const disp = document.getElementById('convex-lens-disp');
    if (disp) {
      disp.setAttribute('scale', Number(refraction).toFixed(2));
    }

    document.documentElement.style.setProperty('--lens-mag', String(magnification));
    document.documentElement.style.setProperty('--lens-bulge', String(bulge));
    document.documentElement.style.setProperty('--lens-specular', String(specular));
    document.documentElement.style.setProperty('--lens-refraction', String(refraction));

    const rInput = document.getElementById('input-lens-refraction') as HTMLInputElement | null;
    const rLabel = document.getElementById('lens-refraction-label');
    if (rInput) rInput.value = String(Math.round(refraction * 100));
    if (rLabel) {
      const rText = refraction === 0 ? '纯平无畸变' : refraction < 0.12 ? '微曲弱折射' : refraction <= 0.22 ? '真实光学折射' : '极致水晶曲率';
      rLabel.textContent = `${Number(refraction).toFixed(2)} (${rText})`;
    }

    const mInput = document.getElementById('input-lens-magnification') as HTMLInputElement | null;
    const mLabel = document.getElementById('lens-magnification-label');
    if (mInput) mInput.value = String(Math.round(magnification * 100));
    if (mLabel) {
      const mText = magnification === 1.0 ? '原大不缩放' : magnification <= 1.25 ? '轻度放大' : magnification <= 1.45 ? '标准放大' : '沉浸特写';
      mLabel.textContent = `${Number(magnification).toFixed(2)}x (${mText})`;
    }

    const bInput = document.getElementById('input-lens-bulge') as HTMLInputElement | null;
    const bLabel = document.getElementById('lens-bulge-label');
    if (bInput) bInput.value = String(Math.round(bulge * 100));
    if (bLabel) {
      const bPercent = Math.round((bulge - 1) * 100);
      bLabel.textContent = `${Number(bulge).toFixed(2)}x (+${bPercent}% 鼓起)`;
    }

    const sInput = document.getElementById('input-lens-specular') as HTMLInputElement | null;
    const sLabel = document.getElementById('lens-specular-label');
    if (sInput) sInput.value = String(Math.round(specular * 100));
    if (sLabel) {
      const sText = specular < 0.5 ? '柔和暗哑' : specular <= 0.8 ? '通透微光' : '高光水晶';
      sLabel.textContent = `${Math.round(specular * 100)}% (${sText})`;
    }

    const fb = document.getElementById('lens-preview-feedback');
    if (fb) {
      fb.innerHTML = `当前物理：折射 <span class="accent-text font-semibold">${Number(refraction).toFixed(2)}</span> · 放大 <span class="accent-text font-semibold">${Number(magnification).toFixed(2)}x</span> · 鼓起 <span class="accent-text font-semibold">+${Math.round((bulge - 1) * 100)}%</span> · 高光 <span class="accent-text font-semibold">${Math.round(specular * 100)}%</span>`;
    }
  }

  public updateLensRefraction(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 0.16 : Math.max(0, Math.min(0.32, num));
    this.store.update(state => {
      state.lensPhysics.refraction = finalVal;
    });
    try {
      localStorage.setItem('aemeath_lens_physics_v1', JSON.stringify(this.store.state.lensPhysics));
    } catch (e) {}
    this.applyLensPhysicsToDom();
  }

  public updateLensMagnification(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 1.38 : Math.max(1.0, Math.min(1.65, num));
    this.store.update(state => {
      state.lensPhysics.magnification = finalVal;
    });
    try {
      localStorage.setItem('aemeath_lens_physics_v1', JSON.stringify(this.store.state.lensPhysics));
    } catch (e) {}
    this.applyLensPhysicsToDom();
  }

  public updateLensBulge(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 1.30 : Math.max(1.05, Math.min(1.55, num));
    this.store.update(state => {
      state.lensPhysics.bulge = finalVal;
    });
    try {
      localStorage.setItem('aemeath_lens_physics_v1', JSON.stringify(this.store.state.lensPhysics));
    } catch (e) {}
    this.applyLensPhysicsToDom();
  }

  public updateLensSpecular(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 0.95 : Math.max(0.30, Math.min(1.0, num));
    this.store.update(state => {
      state.lensPhysics.specular = finalVal;
    });
    try {
      localStorage.setItem('aemeath_lens_physics_v1', JSON.stringify(this.store.state.lensPhysics));
    } catch (e) {}
    this.applyLensPhysicsToDom();
  }

  public resetLensPhysics(): void {
    this.store.update(state => {
      state.lensPhysics = {
        refraction: 0.16,
        magnification: 1.38,
        bulge: 1.30,
        specular: 0.95
      };
    });
    try {
      localStorage.setItem('aemeath_lens_physics_v1', JSON.stringify(this.store.state.lensPhysics));
    } catch (e) {}
    this.applyLensPhysicsToDom();
    this.showToast('已重置凸透镜长按物理参数至默认值（折射 0.16 · 放大 1.38x · 鼓起 +30%）');
  }

  public onLensPreviewChange(idx: number): void {
    const statusEl = document.getElementById('lens-preview-status');
    if (statusEl) {
      statusEl.textContent = idx === 0 ? '体验中：水滴凸折射' : '体验中：水晶高光感';
    }
  }


  /**
   * ==================== 通用 iOS 26 可折叠面板控制器 ====================
   */

  /**
   * ==================== 方框描边与边缘液态玻璃调节 ====================
   */
  public loadBoxGlass(): void {
    try {
      const raw = localStorage.getItem('aemeath_box_glass_v1');
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          this.store.update(state => {
            state.boxGlass = { ...state.boxGlass, ...parsed };
          });
        }
      }
    } catch (e) {}
  }

  public persistBoxGlass(): void {
    try {
      localStorage.setItem('aemeath_box_glass_v1', JSON.stringify(this.store.state.boxGlass));
    } catch (e) {}
  }

  public applyBoxGlassToDom(): void {
    const box = this.store.state.boxGlass || {
      borderColor: '#FFFFFF',
      label: '纯白晶莹 #FFFFFF',
      borderWidth: 1.2,
      edgeGlowBlur: 16,
      edgeGlowOpacity: 35,
      edgeSpecular: 0.95,
      borderRadius: 24
    };

    let r = 255, g = 255, b = 255;
    if (box.borderColor && box.borderColor.startsWith('#')) {
      const hex = box.borderColor.replace('#', '');
      if (hex.length === 6) {
        r = parseInt(hex.substring(0, 2), 16) || 255;
        g = parseInt(hex.substring(2, 4), 16) || 255;
        b = parseInt(hex.substring(4, 6), 16) || 255;
      } else if (hex.length === 3) {
        r = parseInt(hex[0] + hex[0], 16) || 255;
        g = parseInt(hex[1] + hex[1], 16) || 255;
        b = parseInt(hex[2] + hex[2], 16) || 255;
      }
    }

    const root = document.documentElement;
    root.style.setProperty('--card-border-color', box.borderColor);
    root.style.setProperty('--card-border-hex', box.borderColor);
    root.style.setProperty('--card-border-rgb', `${r}, ${g}, ${b}`);
    root.style.setProperty('--card-border-width', `${box.borderWidth}px`);
    root.style.setProperty('--card-border-radius', `${box.borderRadius}px`);
    root.style.setProperty('--card-edge-glow-blur', `${box.edgeGlowBlur}px`);
    root.style.setProperty('--card-edge-glow-opacity', String((box.edgeGlowOpacity / 100).toFixed(2)));
    root.style.setProperty('--card-edge-specular', String(box.edgeSpecular));

    const dot = document.getElementById('box-border-color-dot');
    if (dot) dot.style.background = box.borderColor;

    const colorLabel = document.getElementById('box-border-color-label');
    if (colorLabel) colorLabel.textContent = box.label || box.borderColor;

    const picker = document.getElementById('input-box-border-picker') as HTMLInputElement | null;
    if (picker && box.borderColor.startsWith('#')) picker.value = box.borderColor;

    const wInput = document.getElementById('input-box-border-width') as HTMLInputElement | null;
    const wLabel = document.getElementById('box-border-width-label');
    if (wInput) wInput.value = String(Math.round(box.borderWidth * 10));
    if (wLabel) {
      const wText = box.borderWidth <= 0.8 ? '极细光丝' : box.borderWidth <= 1.6 ? '标准微晶' : '凝胶粗边';
      wLabel.textContent = `${Number(box.borderWidth).toFixed(1)}px (${wText})`;
    }

    const sInput = document.getElementById('input-box-edge-specular') as HTMLInputElement | null;
    const sLabel = document.getElementById('box-edge-specular-label');
    if (sInput) sInput.value = String(Math.round(box.edgeSpecular * 100));
    if (sLabel) {
      const sText = box.edgeSpecular < 0.4 ? '柔和暗哑' : box.edgeSpecular <= 0.8 ? '晶莹透亮' : '高光水晶';
      sLabel.textContent = `${Math.round(box.edgeSpecular * 100)}% (${sText})`;
    }

    const gInput = document.getElementById('input-box-edge-glow') as HTMLInputElement | null;
    const gLabel = document.getElementById('box-edge-glow-label');
    if (gInput) gInput.value = String(box.edgeGlowBlur);
    if (gLabel) {
      const gText = box.edgeGlowBlur === 0 ? '锐利无晕' : box.edgeGlowBlur <= 18 ? '柔和通透' : '梦幻光雾';
      gLabel.textContent = `${box.edgeGlowBlur}px (${gText})`;
    }

    const rInput = document.getElementById('input-box-border-radius') as HTMLInputElement | null;
    const rLabel = document.getElementById('box-border-radius-label');
    if (rInput) rInput.value = String(box.borderRadius);
    if (rLabel) {
      const rText = box.borderRadius <= 16 ? '紧凑圆角' : box.borderRadius <= 26 ? 'iOS 经典' : '水珠胶囊';
      rLabel.textContent = `${box.borderRadius}px (${rText})`;
    }
  }

  public updateBoxBorderColor(color: string, label = ''): void {
    this.store.update(state => {
      state.boxGlass.borderColor = color;
      state.boxGlass.label = label || color;
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
  }

  public updateBoxBorderWidth(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 1.2 : Math.max(0.5, Math.min(3.5, num));
    this.store.update(state => {
      state.boxGlass.borderWidth = finalVal;
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
  }

  public updateBoxEdgeSpecular(val: string | number): void {
    const num = parseFloat(String(val));
    const finalVal = isNaN(num) ? 0.95 : Math.max(0.1, Math.min(1.0, num));
    this.store.update(state => {
      state.boxGlass.edgeSpecular = finalVal;
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
  }

  public updateBoxEdgeGlow(val: string | number): void {
    const num = parseInt(String(val), 10);
    const finalVal = isNaN(num) ? 16 : Math.max(0, Math.min(32, num));
    this.store.update(state => {
      state.boxGlass.edgeGlowBlur = finalVal;
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
  }

  public updateBoxBorderRadius(val: string | number): void {
    const num = parseInt(String(val), 10);
    const finalVal = isNaN(num) ? 24 : Math.max(14, Math.min(32, num));
    this.store.update(state => {
      state.boxGlass.borderRadius = finalVal;
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
  }

  public resetBoxGlass(): void {
    this.store.update(state => {
      state.boxGlass = {
        borderColor: '#FFFFFF',
        label: '纯白晶莹 #FFFFFF',
        borderWidth: 1.2,
        edgeGlowBlur: 16,
        edgeGlowOpacity: 35,
        edgeSpecular: 0.95,
        borderRadius: 24
      };
    });
    this.persistBoxGlass();
    this.applyBoxGlassToDom();
    this.showToast('已恢复方框描边与边缘液态玻璃为纯白默认');
  }

  /**
   * ==================== iOS 26 无字极简折叠控制器 (高性能 120fps 流体阻尼与绝对图层隔离) ====================
   */
  public toggleCollapse(bodyId: string, chevronId?: string): void {
    const body = document.getElementById(bodyId);
    const chevron = chevronId ? document.getElementById(chevronId) : document.querySelector(`[data-chevron-for="${bodyId}"]`) as HTMLElement | null;
    if (!body) return;

    // 清理正在运行中的计时器，防止快速双击产生动画冲突
    const animBody = body as HTMLElement & { __animTimer?: ReturnType<typeof setTimeout> | null };
    if (animBody.__animTimer) {
      clearTimeout(animBody.__animTimer);
      animBody.__animTimer = null;
    }

    const isCurrentlyCollapsed = body.classList.contains('is-collapsed') || body.style.display === 'none' || body.offsetHeight === 0;
    const willCollapse = !isCurrentlyCollapsed;

    // 1. 箭头平滑旋转 (纯硬件加速 transform)
    if (chevron) {
      chevron.classList.toggle('is-open', !willCollapse);
    }

    // 2. 持久化偏好设置
    try {
      const raw = localStorage.getItem('aemeath_collapsed_panels_v1') || '{}';
      const map = JSON.parse(raw);
      map[bodyId] = willCollapse;
      localStorage.setItem('aemeath_collapsed_panels_v1', JSON.stringify(map));
    } catch (e) {}

    if (willCollapse) {
      // ==================== 极简 120fps 纯净折叠 ====================
      // 锁定当前精确像素高度，准备线性收缩
      const startHeight = body.scrollHeight;
      body.style.display = 'block';
      body.style.overflow = 'hidden';
      body.style.height = `${startHeight}px`;
      body.style.opacity = '1';
      body.style.transition = 'none';

      // 单次快速重排锁定起始尺寸
      void body.offsetHeight;

      // 启动 GPU 硬件加速阻尼收缩动画 (300ms)
      requestAnimationFrame(() => {
        body.style.transition = 'height 0.30s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.22s cubic-bezier(0.16, 1, 0.3, 1)';
        body.style.height = '0px';
        body.style.opacity = '0';
      });

      // 动画结束后应用 display: none，彻底从渲染树和合成树剔除，释放 GPU 显存与重绘负担
      animBody.__animTimer = setTimeout(() => {
        body.classList.add('is-collapsed');
        body.style.display = 'none';
        body.style.height = '0px';
        body.style.opacity = '0';
        body.style.transition = '';
        animBody.__animTimer = null;
      }, 310);

    } else {
      // ==================== 极简 120fps 纯净展开 ====================
      body.classList.remove('is-collapsed');
      body.style.display = 'block';
      body.style.overflow = 'hidden';
      body.style.transition = 'none';
      body.style.height = 'auto';
      body.style.opacity = '0';

      // 探测展开目标自然像素高度
      const targetHeight = body.scrollHeight;
      body.style.height = '0px';

      // 单次快速重排锁定起始尺寸
      void body.offsetHeight;

      // 启动 GPU 阻尼平滑展开动画 (340ms)
      requestAnimationFrame(() => {
        body.style.transition = 'height 0.34s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.28s cubic-bezier(0.16, 1, 0.3, 1)';
        body.style.height = `${targetHeight}px`;
        body.style.opacity = '1';
      });

      // 动画结束后解除固定高度限制，并【仅局部刷新本折叠框内的滑块几何】，杜绝触发全页面 resize 暴风
      animBody.__animTimer = setTimeout(() => {
        body.style.height = 'auto';
        body.style.overflow = 'visible';
        body.style.transition = '';
        animBody.__animTimer = null;

        // 仅对当前展开容器内的滑块进行精准几何刷新
        const innerSliders = body.querySelectorAll('[data-agent-component="ConvexLiquidLensSlider"]') as NodeListOf<HTMLElement & { __syncSnapLayout?: (idx: number, immediate: boolean) => void }>;
        innerSliders.forEach(slider => {
          if (typeof slider.__syncSnapLayout === 'function') {
            const activeIdx = parseInt(slider.getAttribute('data-active-index') || '0', 10);
            slider.__syncSnapLayout(activeIdx, true);
          }
        });
      }, 350);
    }
  }

  public initCollapsibles(): void {
    try {
      const raw = localStorage.getItem('aemeath_collapsed_panels_v1');
      if (raw) {
        const map = JSON.parse(raw);
        Object.keys(map).forEach(bodyId => {
          const isCollapsed = Boolean(map[bodyId]);
          const body = document.getElementById(bodyId);
          if (body) {
            if (isCollapsed) {
              body.classList.add('is-collapsed');
              body.style.display = 'none';
              body.style.height = '0px';
              body.style.opacity = '0';
              body.style.overflow = 'hidden';
            } else {
              body.classList.remove('is-collapsed');
              body.style.display = 'block';
              body.style.height = 'auto';
              body.style.opacity = '1';
              body.style.overflow = 'visible';
            }
            const chevron = document.querySelector(`[data-chevron-for="${bodyId}"]`) as HTMLElement | null || document.getElementById(`${bodyId.replace('-collapse-box', '')}-chevron`);
            if (chevron) {
              chevron.classList.toggle('is-open', !isCollapsed);
            }
          }
        });
      }
    } catch (e) {}
  }

  public showToast(text: string): void {
    const toast = document.getElementById('ios-toast');
    const toastText = document.getElementById('ios-toast-text');
    if (!toast || !toastText) return;

    toastText.textContent = text;
    toast.classList.remove('opacity-0');
    toast.classList.add('opacity-100');

    if (this.toastTimer) clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => {
      toast.classList.remove('opacity-100');
      toast.classList.add('opacity-0');
    }, 1900);
  }
}
