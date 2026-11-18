/**
 * BackgroundSystem.ts - 全局背景系统 (严格默认纯白 #FFFFFF)
 * 支持纯白、纯色色卡/自由取色器、爱弥斯原图壁纸/相册上传，0%~85% 柔光遮罩调节，
 * 以及方框与文字柔光玻璃化 (超透沉浸透光视效，透出完整自定义背景壁纸细节)
 */

import { AppStore, AppBackgroundState } from '../../core/store/AppStore';

export class BackgroundSystem {
  private store: AppStore;
  private layerEl: HTMLElement | null = null;
  private veilEl: HTMLElement | null = null;
  private shellEl: HTMLElement | null = null;

  constructor(store: AppStore) {
    this.store = store;
    this.layerEl = document.getElementById('app-background-layer');
    this.veilEl = document.getElementById('app-background-veil');
    this.shellEl = document.getElementById('app-shell');
  }

  public init(): void {
    this.applyBackground(this.store.state.appBackground);
  }

  public setMode(modeIndex: number): void {
    this.store.update(state => {
      state.appBackground.modeIndex = modeIndex;
      if (modeIndex === 0) {
        state.appBackground.type = 'white';
        state.appBackground.color = '#FFFFFF';
        state.appBackground.label = '默认纯白 #FFFFFF';
      } else if (modeIndex === 1) {
        if (state.appBackground.type !== 'color') {
          state.appBackground.type = 'color';
          state.appBackground.color = '#FAF8F5';
          state.appBackground.label = '暖玉白 #FAF8F5';
        }
      } else if (modeIndex === 2) {
        if (state.appBackground.type !== 'image' || !state.appBackground.imageSrc) {
          state.appBackground.type = 'image';
          state.appBackground.imageSrc = '1794568840819.jpg';
          state.appBackground.label = '爱弥斯原图壁纸 1';
        }
      }
    });
    this.applyBackground(this.store.state.appBackground);
  }

  public setSolidColor(hex: string, label?: string): void {
    this.store.update(state => {
      state.appBackground.modeIndex = 1;
      state.appBackground.type = 'color';
      state.appBackground.color = hex;
      state.appBackground.label = label || `纯色 ${hex}`;
    });
    this.applyBackground(this.store.state.appBackground);
  }

  public setImageWallpaper(src: string, label?: string): void {
    this.store.update(state => {
      state.appBackground.modeIndex = 2;
      state.appBackground.type = 'image';
      state.appBackground.imageSrc = src;
      state.appBackground.label = label || '图片壁纸';
    });
    this.applyBackground(this.store.state.appBackground);
  }

  public setVeilOpacity(val: number): void {
    const clamped = Math.max(0, Math.min(85, val));
    this.store.update(state => {
      state.appBackground.veilOpacity = clamped;
    });
    this.applyBackground(this.store.state.appBackground);
  }

  /**
   * 切换方框与文字柔光玻璃化视效
   */
  public toggleUltraGlass(checked: boolean): void {
    this.store.update(state => {
      state.appBackground.ultraGlassMode = checked;
    });
    this.applyBackground(this.store.state.appBackground);
  }

  /**
   * 调节玻璃透光度 (0% ~ 100%)
   */
  public setGlassTransparency(val: number): void {
    const clamped = Math.max(0, Math.min(100, val));
    this.store.update(state => {
      state.appBackground.glassTransparency = clamped;
    });
    this.applyBackground(this.store.state.appBackground);
  }

  /**
   * 调节玻璃柔光模糊度 (0px ~ 24px)
   */
  public setGlassBlur(val: number): void {
    const clamped = Math.max(0, Math.min(24, val));
    this.store.update(state => {
      state.appBackground.glassBlur = clamped;
    });
    this.applyBackground(this.store.state.appBackground);
  }

  public resetToPureWhite(): void {
    this.store.update(state => {
      state.appBackground = {
        modeIndex: 0,
        type: 'white',
        color: '#FFFFFF',
        imageSrc: '',
        veilOpacity: 25,
        label: '默认纯白 #FFFFFF',
        ultraGlassMode: false,
        glassTransparency: 72,
        glassBlur: 8
      };
    });
    this.applyBackground(this.store.state.appBackground);

    const slider = document.getElementById('slider-bg-mode');
    if (slider) {
      slider.setAttribute('data-active-index', '0');
      window.dispatchEvent(new Event('resize'));
    }
  }

  public applyBackground(bg: AppBackgroundState): void {
    this.syncPanels(bg.modeIndex);

    const statusEl = document.getElementById('settings-bg-status');
    const veilInput = document.getElementById('input-bg-veil-opacity') as HTMLInputElement | null;
    const veilLabel = document.getElementById('bg-veil-label');

    if (veilInput) veilInput.value = String(bg.veilOpacity ?? 25);
    if (veilLabel) veilLabel.textContent = `${bg.veilOpacity ?? 25}%`;
    if (statusEl) statusEl.textContent = `当前：${bg.label || '默认纯白 #FFFFFF'}`;

    // 处理超透柔光玻璃视效 (.ultra-glass-active)
    const isUltraGlass = Boolean(bg.ultraGlassMode);
    const transparency = bg.glassTransparency ?? 72;
    const blur = bg.glassBlur ?? 8;
    const shell = this.shellEl || document.getElementById('app-shell');

    if (shell) {
      shell.classList.toggle('ultra-glass-active', isUltraGlass);

      const glassOpacity = Math.max(0, Math.min(1, 1 - transparency / 100)).toFixed(2);
      shell.style.setProperty('--ultra-glass-opacity', glassOpacity);
      shell.style.setProperty('--ultra-glass-blur', `${blur}px`);
    }

    // 同步设置面板中的开关与滑杆
    const toggleUltraGlass = document.getElementById('toggle-ultra-glass');
    if (toggleUltraGlass) {
      toggleUltraGlass.setAttribute('data-checked', isUltraGlass ? 'true' : 'false');
      const orb = toggleUltraGlass.querySelector('.liquid-lens-toggle-orb') as HTMLElement | null;
      if (orb) {
        orb.style.transform = isUltraGlass ? 'translate3d(26px, 0, 0)' : 'translate3d(0, 0, 0)';
      }
    }

    const statusLabel = document.getElementById('ultra-glass-status-label');
    if (statusLabel) {
      statusLabel.textContent = isUltraGlass
        ? `已激活 · 实时极透柔光液态玻璃 (${transparency}% 透光 · ${blur}px 柔光模糊)`
        : '未开启 · 处于标准白底磨砂态（点击右侧开启）';
    }

    const transInput = document.getElementById('input-ultra-glass-transparency') as HTMLInputElement | null;
    const transLabel = document.getElementById('ultra-glass-transparency-label');
    if (transInput) transInput.value = String(transparency);
    if (transLabel) transLabel.textContent = `${transparency}%`;

    const blurInput = document.getElementById('input-ultra-glass-blur') as HTMLInputElement | null;
    const blurLabel = document.getElementById('ultra-glass-blur-label');
    if (blurInput) blurInput.value = String(blur);
    if (blurLabel) blurLabel.textContent = `${blur}px`;

    if (!this.layerEl) return;

    if (bg.type === 'image' && bg.imageSrc) {
      this.layerEl.style.backgroundColor = '#FFFFFF';
      this.layerEl.style.backgroundImage = `url("${bg.imageSrc.replace(/"/g, '\\"')}")`;
      if (this.veilEl) {
        this.veilEl.classList.remove('hidden');
        const alpha = Math.max(0, Math.min(0.85, (bg.veilOpacity ?? 25) / 100));
        this.veilEl.style.background = `rgba(255, 255, 255, ${alpha.toFixed(2)})`;
      }
    } else {
      const solidColor = bg.type === 'color' ? (bg.color || '#FFFFFF') : '#FFFFFF';
      this.layerEl.style.backgroundImage = 'none';
      this.layerEl.style.backgroundColor = solidColor;
      document.documentElement.style.setProperty('--app-bg', solidColor);
      if (this.veilEl) {
        this.veilEl.classList.add('hidden');
      }
    }
  }

  private syncPanels(idx: number): void {
    const panels = ['bg-panel-white', 'bg-panel-color', 'bg-panel-image'];
    panels.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('hidden', i !== idx);
    });
  }
}
