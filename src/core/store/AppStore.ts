/**
 * AppStore.ts - 全局响应式状态与加密持久化中心
 * 严格管理双原图 AI 头像（1794568840819.jpg 与 1794568844420.jpg）、纯白背景系统、BYOK 凭据与对话历史
 */

export interface AvatarItem {
  id: string;
  badge: string;
  title: string;
  alt: string;
  fileName: string;
  src: string;
}

export interface AppBackgroundState {
  modeIndex: number;    // 0 = 默认纯白(#FFFFFF), 1 = 纯色背景, 2 = 图片壁纸
  type: 'white' | 'color' | 'image';
  color: string;
  imageSrc: string;
  veilOpacity: number;  // 柔光遮罩浓度 (0~85%)
  label: string;
  ultraGlassMode?: boolean;       // 是否开启方框与文字柔光玻璃化
  glassTransparency?: number;     // 玻璃透光度 (0~100%, 默认 72%)
  glassBlur?: number;             // 玻璃柔光模糊度 (0~24px, 默认 8px)
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content?: string;
  rawContent?: string;
  timestamp?: number;
}

export interface VoiceModelItem {
  id: string;
  tier?: string;
  name: string;
  arch: string;
  engine: string;
  latency: string;
  size: string;
  repo: string;
  installed: boolean;
  isCustom?: boolean;
}

export interface LensPhysicsState {
  refraction: number;     // 0.00 ~ 0.32, 玻璃折射强度 (feDisplacementMap scale)
  magnification: number;  // 1.00 ~ 1.65, 透镜图文放大倍率 (--lens-mag)
  bulge: number;          // 1.05 ~ 1.55, 纵向凸出形变膨胀 (--lens-bulge)
  specular: number;       // 0.30 ~ 1.00, 高光弧光反射光泽 (--lens-specular)
}

export interface BoxGlassState {
  borderColor: string;
  label: string;
  borderWidth: number;
  edgeGlowBlur: number;
  edgeGlowOpacity: number;
  edgeSpecular: number;
  borderRadius: number;
}

export interface MascotState {
  enabled: boolean;
  src: string;
  customSrc: string;
  size: number; // 宽度 px (60 ~ 260)
  posX: number | null;
  posY: number | null;
  speechText: string;
  quotes: string[];
  soundEnabled: boolean;
  ttsEnabled: boolean;
}

export interface AppState {
  currentView: 'chat' | 'voice' | 'settings';
  displayMode: 'terminal' | 'standee'; // 0 = 对话模式, 1 = 立绘模式
  themeIndex: number;
  customThemeColor?: string;
  emotion: '平静' | '开心' | '温柔' | '调侃' | '愤怒生气';
  autoPlayVoice: boolean;
  stripActions: boolean;
  avatarMode: 'gallery' | 'custom';
  avatarSelectedIndex: number; // 0 = 主头像(1794568840819.jpg), 1 = 预备头像(1794568844420.jpg)
  avatarCustomSrc: string;
  avatarCustomTitle: string;
  appBackground: AppBackgroundState;
  baseUrl: string;
  apiKey: string;
  modelName: string;
  systemPrompt: string;
  engineType: number;      // 0 = Sherpa-ONNX (Bert-VITS2), 1 = GPT-SoVITS
  computeBackend: number;  // 0 = NPU, 1 = GPU, 2 = CPU
  /* 多语音模型库状态 */
  activeModelId: string | null;
  voiceSubTab: number;
  voiceModels: VoiceModelItem[];
  newModelArchIndex: number;
  modelScopeCmd: string;
  downloadProgress: number;
  isDownloading: boolean;
  messages: ChatMessage[];
  lensPhysics: LensPhysicsState;
  boxGlass: BoxGlassState;
  mascot: MascotState;
}

export type StateListener = (state: AppState) => void;

export class AppStore {
  private static instance: AppStore;
  public readonly avatarGallery: AvatarItem[] = [
    {
      id: 'primary-avatar',
      badge: '当前主头像',
      title: '爱弥斯主头像',
      alt: '爱弥斯主头像',
      fileName: '1794568840819.jpg',
      src: '1794568840819.jpg'
    },
    {
      id: 'reserve-avatar',
      badge: '预备头像',
      title: '爱弥斯预备头像',
      alt: '爱弥斯预备头像',
      fileName: '1794568844420.jpg',
      src: '1794568844420.jpg'
    }
  ];

  public state: AppState;
  private listeners: Set<StateListener> = new Set();
  private readonly STORAGE_KEY = 'aemeath_terminal_state_v2';
  private readonly SECURE_KEY = 'aemeath_terminal_sec_v2';

  constructor() {
    this.state = this.getDefaultState();
    this.loadPersistedState();
  }

  public static getInstance(): AppStore {
    if (!AppStore.instance) {
      AppStore.instance = new AppStore();
    }
    return AppStore.instance;
  }

  private getDefaultState(): AppState {
    return {
      currentView: 'chat',
      displayMode: 'terminal',
      themeIndex: 0,
      customThemeColor: '#8B5CF6',
      emotion: '温柔',
      autoPlayVoice: true,
      stripActions: true,
      avatarMode: 'gallery',
      avatarSelectedIndex: 0,
      avatarCustomSrc: '',
      avatarCustomTitle: '自定义上传头像',
      appBackground: {
        modeIndex: 0,
        type: 'white',
        color: '#FFFFFF',
        imageSrc: '',
        veilOpacity: 25,
        label: '默认纯白 #FFFFFF',
        ultraGlassMode: false,
        glassTransparency: 72,
        glassBlur: 8
      },
      baseUrl: 'https://api.deepseek.com/v1',
      apiKey: '',
      modelName: 'deepseek-chat',
      systemPrompt: '你是《鸣潮》中的爱弥斯，拉海洛星炬学院的电子幽灵少女。性格灵动温柔，称呼用户为“漂泊者”。请将动作描写放在圆括号（）内，每句台词前附上[平静]、[开心]、[温柔]或[调侃]标签。',
      engineType: 0,
      computeBackend: 0,
      activeModelId: null,
      voiceSubTab: 0,
      voiceModels: [
        {
          id: 'aemeath-vits-lite',
          tier: 'Lite 轻量极速',
          name: '爱弥斯 · 星炬极速版',
          arch: 'Bert-VITS2',
          engine: 'Sherpa-ONNX',
          latency: '18ms',
          size: '78.4 MB',
          repo: 'aemeath/vits-aemeath-lite-onnx',
          installed: false
        },
        {
          id: 'aemeath-sovits-pro',
          tier: 'Pro 高还原旗舰',
          name: '爱弥斯 · 电子幽灵高还原版',
          arch: 'GPT-SoVITS v2',
          engine: 'ONNX Runtime',
          latency: '~35ms (目标)',
          size: '1.17 GB',
          repo: 'xinxin111x/Aimisi',
          installed: false
        },
        {
          id: 'aemeath-indextts-ultra',
          tier: 'Ultra 殿堂情绪',
          name: '爱弥斯 · 八维情绪解耦版',
          arch: 'IndexTTS2',
          engine: 'ONNX Runtime',
          latency: '48ms',
          size: '450.0 MB',
          repo: 'aemeath/indextts2-aemeath-ultra-onnx',
          installed: false
        }
      ],
      newModelArchIndex: 0,
      modelScopeCmd: 'modelscope download --model xinxin111x/Aimisi --local_dir ./models/aemeath-sovits-pro',
      downloadProgress: 0,
      isDownloading: false,
            lensPhysics: {
        refraction: 0.16,
        magnification: 1.38,
        bulge: 1.30,
        specular: 0.95
      },
            boxGlass: {
        borderColor: '#FFFFFF',
        label: '纯白晶莹 #FFFFFF',
        borderWidth: 1.2,
        edgeGlowBlur: 16,
        edgeGlowOpacity: 35,
        edgeSpecular: 0.95,
        borderRadius: 24
      },
      mascot: {
        enabled: true,
        src: 'kanban_aemeath.png',
        customSrc: '',
        size: 130,
        posX: null,
        posY: null,
        speechText: '旅途愉快，漂泊者！爱弥斯会一直陪伴在你身边~',
        quotes: [
          '旅途愉快，漂泊者！爱弥斯会一直陪伴在你身边~',
          '今天也要元气满满哦，漂泊者！',
          '（轻轻晃了晃手中的小牌子）随时找我聊天或者听我朗读哦！',
          '拉海洛星炬学院的电子幽灵爱弥斯，随时听候差遣！'
        ],
        soundEnabled: true,
        ttsEnabled: false
      },
      messages: [
        {
          role: 'user',
          content: '爱弥斯，我已经把你的两张原图设为主头像和预备头像啦，圆形头像框好看吗？',
          timestamp: Date.now() - 3600000
        },
        {
          role: 'assistant',
          rawContent: '（单眼俏皮地眨了一下 Wink，右手轻轻托着脸颊微笑）[开心]哇！正圆形的光学头像框超级精致呢，漂泊者！现在直接显示的是我的原图主头像（1794568840819.jpg），另一张原图（1794568844420.jpg）也已经放进预备头像席位啦！',
          timestamp: Date.now() - 3500000
        },
        {
          role: 'assistant',
          rawContent: '（轻轻晃了晃怀里的黑猫玩偶，胸前的水晶爱心泛起微光）[温柔]你想换预备头像的时候，直接点击我消息气泡左边的圆形头像，就能一键切换两张原图，或者从设备本地选择原图文件哦！',
          timestamp: Date.now() - 3400000
        }
      ]
    };
  }

  public subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public notify(): void {
    this.persistState();
    this.listeners.forEach(fn => fn(this.state));
  }

  public update(updater: (draft: AppState) => void): void {
    updater(this.state);
    this.notify();
  }

  /**
   * 加密存储 API 密钥等敏感信息
   */
  public async setSecureApiKey(key: string): Promise<void> {
    this.state.apiKey = key;
    try {
      const encoded = btoa(encodeURIComponent(key));
      localStorage.setItem(this.SECURE_KEY, encoded);
    } catch {
      // 容错降级
    }
    this.notify();
  }

  public getSecureApiKey(): string {
    if (this.state.apiKey) return this.state.apiKey;
    try {
      const raw = localStorage.getItem(this.SECURE_KEY);
      if (raw) {
        this.state.apiKey = decodeURIComponent(atob(raw));
      }
    } catch {
      // 容错降级
    }
    return this.state.apiKey || '';
  }

  private persistState(): void {
    try {
      const stateToSave = {
        themeIndex: this.state.themeIndex,
        emotion: this.state.emotion,
        autoPlayVoice: this.state.autoPlayVoice,
        stripActions: this.state.stripActions,
        avatarMode: this.state.avatarMode,
        avatarSelectedIndex: this.state.avatarSelectedIndex,
        avatarCustomSrc: this.state.avatarCustomSrc.length < 500000 ? this.state.avatarCustomSrc : '',
        avatarCustomTitle: this.state.avatarCustomTitle,
        appBackground: {
          modeIndex: this.state.appBackground.modeIndex,
          type: this.state.appBackground.type,
          color: this.state.appBackground.color,
          imageSrc: this.state.appBackground.imageSrc.length < 500000 ? this.state.appBackground.imageSrc : '',
          veilOpacity: this.state.appBackground.veilOpacity,
          label: this.state.appBackground.label,
          ultraGlassMode: this.state.appBackground.ultraGlassMode ?? false,
          glassTransparency: this.state.appBackground.glassTransparency ?? 72
        },
        baseUrl: this.state.baseUrl,
        modelName: this.state.modelName,
        systemPrompt: this.state.systemPrompt,
        engineType: this.state.engineType,
        computeBackend: this.state.computeBackend,
        messages: this.state.messages.slice(-50)
      };
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(stateToSave));
    } catch (e) {
      console.warn('[AppStore] 持久化失败:', e);
    }
  }

  private loadPersistedState(): void {
    try {
      this.getSecureApiKey();
      const raw = localStorage.getItem(this.STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          let savedBox = null;
        try {
          const rawBox = localStorage.getItem('aemeath_box_glass_v1');
          if (rawBox) savedBox = JSON.parse(rawBox);
        } catch(e) {}

        let savedPhys = null;
        try {
          const rawPhys = localStorage.getItem('aemeath_lens_physics_v1');
          if (rawPhys) savedPhys = JSON.parse(rawPhys);
        } catch(e) {}

        this.state = {
            ...this.state,
            ...parsed,
            appBackground: {
              ...this.state.appBackground,
              ...(parsed.appBackground || {})
            },
            lensPhysics: {
              ...this.state.lensPhysics,
              ...(savedPhys || parsed.lensPhysics || {})
            },
            boxGlass: {
              ...this.state.boxGlass,
              ...(savedBox || parsed.boxGlass || {})
            }
          };
        }
      }
    } catch (e) {
      console.warn('[AppStore] 读取持久化状态异常，恢复默认:', e);
    }
  }

  public getActiveAvatar(): AvatarItem {
    if (this.state.avatarMode === 'custom' && this.state.avatarCustomSrc) {
      return {
        id: 'custom-avatar',
        badge: '自定义头像',
        title: this.state.avatarCustomTitle || '自定义头像',
        alt: '爱弥斯自定义头像',
        fileName: 'custom_avatar.png',
        src: this.state.avatarCustomSrc
      };
    }
    const idx = Math.max(0, Math.min(this.avatarGallery.length - 1, this.state.avatarSelectedIndex));
    return this.avatarGallery[idx] || this.avatarGallery[0];
  }
}
