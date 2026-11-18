/**
 * SettingsView.ts - 设置视图控制器
 * 管理主题发光色、BYOK API 接口、双层 System 提示词与视口模式
 */

import { AppStore } from '../../core/store/AppStore';

export interface ThemeConfig {
  name: string;
  color: string;
  rgb: string;
  glow: string;
  strong: string;
  soft: string;
  surface: string;
}

export class SettingsView {
  private store: AppStore;

  public static readonly THEMES: ThemeConfig[] = [
    {
      name: '冰川蓝 #007AFF',
      color: '#007AFF',
      rgb: '0, 122, 255',
      glow: 'rgba(0, 122, 255, 0.52)',
      strong: 'rgba(0, 122, 255, 0.68)',
      soft: 'rgba(56, 189, 248, 0.36)',
      surface: 'rgba(0, 122, 255, 0.09)'
    },
    {
      name: '星炬粉 #FF2D78',
      color: '#FF2D78',
      rgb: '255, 45, 120',
      glow: 'rgba(255, 45, 120, 0.50)',
      strong: 'rgba(255, 45, 120, 0.68)',
      soft: 'rgba(255, 117, 168, 0.36)',
      surface: 'rgba(255, 45, 120, 0.09)'
    },
    {
      name: '极光青 #00C7BE',
      color: '#00C7BE',
      rgb: '0, 199, 190',
      glow: 'rgba(0, 199, 190, 0.50)',
      strong: 'rgba(0, 199, 190, 0.68)',
      soft: 'rgba(45, 212, 191, 0.36)',
      surface: 'rgba(0, 199, 190, 0.09)'
    }
  ];

  public static readonly PERSONA_TEMPLATES = [
    '你是《鸣潮》中的爱弥斯，拉海洛星炬学院的电子幽灵少女。性格灵动温柔，称呼用户为“漂泊者”。请将动作描写放在圆括号（）内，每句台词前附上[平静]、[开心]、[温柔]、[调侃]或[愤怒生气]标签。',
    '你是《鸣潮》中的爱弥斯，星炬学院的俏皮学姐，喜欢在终端里和漂泊者开玩笑、分享学院趣闻。动作放在（）内，台词前附上情绪标签。',
    '你是《鸣潮》中的爱弥斯，在安静的夜晚通过终端陪伴漂泊者，声音轻柔治愈。动作放在（）内，台词前附上情绪标签。',
    '你是《鸣潮》中的爱弥斯，请自由设定你的角色性格与扮演细节。请将动作描写放在圆括号（）内，每句台词前附上情绪标签。'
  ];

  constructor(store: AppStore) {
    this.store = store;
  }

  public init(): void {
    const inputPrompt = document.getElementById('input-system-prompt') as HTMLTextAreaElement | null;
    const inputBaseUrl = document.getElementById('input-base-url') as HTMLInputElement | null;
    const inputApiKey = document.getElementById('input-api-key') as HTMLInputElement | null;
    const inputModel = document.getElementById('input-model-name') as HTMLInputElement | null;

    if (inputPrompt) {
      inputPrompt.value = this.store.state.systemPrompt;
      this.updatePromptCharCount(this.store.state.systemPrompt);
    }
    if (inputBaseUrl) inputBaseUrl.value = this.store.state.baseUrl;
    if (inputApiKey) inputApiKey.value = this.store.state.apiKey;
    if (inputModel) inputModel.value = this.store.state.modelName;

    this.applyTheme(this.store.state.themeIndex);
  }

  public static hexToRgb(hex: string): { r: number; g: number; b: number } {
    hex = (hex || '#007AFF').replace('#', '').trim();
    if (hex.length === 3) {
      hex = hex.split('').map(c => c + c).join('');
    }
    const num = parseInt(hex, 16);
    if (isNaN(num)) return { r: 0, g: 122, b: 255 };
    return {
      r: (num >> 16) & 255,
      g: (num >> 8) & 255,
      b: num & 255
    };
  }

  public applyCustomColor(hex: string, customName?: string): ThemeConfig {
    const { r, g, b } = SettingsView.hexToRgb(hex);
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const isBrightWhite = luminance >= 175;

    const theme: ThemeConfig = {
      name: customName || (hex.toUpperCase() === '#FFFFFF' ? '纯白月芒 #FFFFFF' : `自选色彩 ${hex.toUpperCase()}`),
      color: hex,
      rgb: `${r}, ${g}, ${b}`,
      glow: isBrightWhite ? 'rgba(255, 255, 255, 0.88)' : `rgba(${r}, ${g}, ${b}, 0.52)`,
      strong: isBrightWhite ? 'rgba(255, 255, 255, 0.98)' : `rgba(${r}, ${g}, ${b}, 0.68)`,
      soft: isBrightWhite ? 'rgba(255, 255, 255, 0.65)' : `rgba(${Math.min(255, r + 45)}, ${Math.min(255, g + 45)}, ${Math.min(255, b + 45)}, 0.38)`,
      surface: isBrightWhite ? 'rgba(15, 23, 42, 0.06)' : `rgba(${r}, ${g}, ${b}, 0.09)`
    };

    const root = document.documentElement.style;
    root.setProperty('--ios-blue', theme.color);
    root.setProperty('--accent-color', theme.color);
    root.setProperty('--accent-rgb', theme.rgb);
    root.setProperty('--accent-glow', theme.glow);
    root.setProperty('--accent-glow-strong', theme.strong);
    root.setProperty('--accent-glow-soft', theme.soft);
    root.setProperty('--accent-surface', theme.surface);

    if (isBrightWhite) {
      root.setProperty('--accent-contrast-text', '#0F172A');
      root.setProperty('--accent-contrast-shadow', '0 0 10px rgba(255, 255, 255, 0.95), 0 0 20px rgba(255, 255, 255, 0.80)');
      root.setProperty('--accent-btn-text', '#0F172A');
      root.setProperty('--accent-btn-border', 'rgba(15, 23, 42, 0.16)');
      root.setProperty('--accent-btn-shadow', 'inset 0 1.5px 2px rgba(255, 255, 255, 1), 0 4px 14px rgba(15, 23, 42, 0.08), 0 0 20px rgba(255, 255, 255, 0.95)');
      root.setProperty('--accent-pill-border', 'rgba(15, 23, 42, 0.14)');
      root.setProperty('--accent-toggle-bg', '#0F172A');
    } else {
      root.setProperty('--accent-contrast-text', theme.color);
      root.setProperty('--accent-contrast-shadow', `0 0 12px ${theme.soft}`);
      root.setProperty('--accent-btn-text', '#FFFFFF');
      root.setProperty('--accent-btn-border', 'rgba(255, 255, 255, 0.42)');
      root.setProperty('--accent-btn-shadow', `inset 0 1.5px 2px rgba(255, 255, 255, 0.65), 0 6px 20px -3px ${theme.glow}, 0 0 14px -2px ${theme.soft}`);
      root.setProperty('--accent-pill-border', 'rgba(255, 255, 255, 0.60)');
      root.setProperty('--accent-toggle-bg', theme.color);
    }

    const label = document.getElementById('theme-glow-label');
    if (label) label.textContent = theme.name;

    const dot = document.getElementById('theme-color-dot');
    if (dot) {
      dot.style.background = hex;
      dot.style.borderColor = isBrightWhite ? 'rgba(15, 23, 42, 0.25)' : '#FFFFFF';
    }

    const picker = document.getElementById('input-theme-color-picker') as HTMLInputElement | null;
    if (picker && picker.value.toLowerCase() !== hex.toLowerCase()) {
      picker.value = hex;
    }

    this.store.update(state => {
      state.themeIndex = 3;
      state.customThemeColor = hex;
    });

    return theme;
  }

  public applyTheme(idx: number): ThemeConfig {
    if (idx === 3) {
      const customHex = this.store.state.customThemeColor || '#8B5CF6';
      return this.applyCustomColor(customHex);
    }
    const theme = SettingsView.THEMES[idx] || SettingsView.THEMES[0];
    const root = document.documentElement.style;

    root.setProperty('--ios-blue', theme.color);
    root.setProperty('--accent-color', theme.color);
    root.setProperty('--accent-rgb', theme.rgb);
    root.setProperty('--accent-glow', theme.glow);
    root.setProperty('--accent-glow-strong', theme.strong);
    root.setProperty('--accent-glow-soft', theme.soft);
    root.setProperty('--accent-surface', theme.surface);
    root.setProperty('--accent-contrast-text', theme.color);
    root.setProperty('--accent-contrast-shadow', `0 0 12px ${theme.soft}`);
    root.setProperty('--accent-btn-text', '#FFFFFF');
    root.setProperty('--accent-btn-border', 'rgba(255, 255, 255, 0.42)');
    root.setProperty('--accent-btn-shadow', `inset 0 1.5px 2px rgba(255, 255, 255, 0.65), 0 6px 20px -3px ${theme.glow}, 0 0 14px -2px ${theme.soft}`);
    root.setProperty('--accent-pill-border', 'rgba(255, 255, 255, 0.60)');
    root.setProperty('--accent-toggle-bg', theme.color);

    const label = document.getElementById('theme-glow-label');
    if (label) label.textContent = theme.name;

    const dot = document.getElementById('theme-color-dot');
    if (dot) dot.style.background = theme.color;

    const picker = document.getElementById('input-theme-color-picker') as HTMLInputElement | null;
    if (picker) picker.value = theme.color;

    this.store.update(state => {
      state.themeIndex = idx;
    });

    return theme;
  }

  public applyApiPreset(idx: number): void {
    const presets = [
      { url: 'https://api.deepseek.com/v1', model: 'deepseek-chat', name: 'DeepSeek' },
      { url: 'https://api.siliconflow.cn/v1', model: 'deepseek-ai/DeepSeek-V3', name: '硅基流动' },
      { url: 'https://api.openai.com/v1', model: 'gpt-4o-mini', name: '自定义' }
    ];
    const p = presets[idx] || presets[0];

    const inputBaseUrl = document.getElementById('input-base-url') as HTMLInputElement | null;
    const inputModel = document.getElementById('input-model-name') as HTMLInputElement | null;
    if (inputBaseUrl) inputBaseUrl.value = p.url;
    if (inputModel) inputModel.value = p.model;

    this.store.update(state => {
      state.baseUrl = p.url;
      state.modelName = p.model;
    });
  }

  public applyPersonaTemplate(idx: number): string {
    let prompt: string;
    if (idx === 3) {
      const custom = (typeof localStorage !== 'undefined') ? (localStorage.getItem('aemeath_custom_system_prompt') || '') : '';
      prompt = custom.trim() || SettingsView.PERSONA_TEMPLATES[3];
    } else {
      prompt = SettingsView.PERSONA_TEMPLATES[idx] || SettingsView.PERSONA_TEMPLATES[0];
    }
    const inputPrompt = document.getElementById('input-system-prompt') as HTMLTextAreaElement | null;
    if (inputPrompt) inputPrompt.value = prompt;

    this.store.update(state => {
      state.systemPrompt = prompt;
    });
    this.updatePromptCharCount(prompt);
    return prompt;
  }

  public saveApiCredentials(baseUrl: string, apiKey: string, modelName: string): void {
    this.store.update(state => {
      state.baseUrl = baseUrl.trim();
      state.modelName = modelName.trim();
    });
    this.store.setSecureApiKey(apiKey.trim());
  }

  public updateSystemPrompt(val: string): void {
    this.store.update(state => {
      state.systemPrompt = val;
    });
    this.updatePromptCharCount(val);
  }

  private updatePromptCharCount(text: string): void {
    const charCountEl = document.getElementById('persona-char-count');
    if (charCountEl) {
      charCountEl.textContent = `${text.length} 字 · 双层生效`;
    }
  }

  public setViewport(idx: number): void {
    const shell = document.getElementById('app-shell');
    if (!shell) return;
    shell.classList.toggle('phone-viewport', idx === 0);
    shell.classList.toggle('fullscreen-viewport', idx === 1);
  }
}
