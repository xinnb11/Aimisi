/**
 * AvatarCustomizationModal.ts - 正圆形 AI 头像定制组件
 * 
 * 严格规范：
 * 1. 严禁使用任何 SVG / Canvas 代码绘制头像！
 * 2. 槽位 0（默认主头像）：100% 直接加载原图 1794568840819.jpg
 * 3. 槽位 1（预备头像）：100% 直接加载原图 1794568844420.jpg
 * 4. 支持相册上传自定义原图与输入图片 URL，全局实时响应式同步
 */

import { AppStore, AvatarItem } from '../../core/store/AppStore';

export class AvatarCustomizationModal {
  private store: AppStore;
  private modalEl: HTMLElement | null = null;

  constructor(store: AppStore) {
    this.store = store;
    this.modalEl = document.getElementById('avatar-modal');
  }

  public init(): void {
    this.renderPresets();
    this.syncAllAvatarViews();
  }

  public open(): void {
    this.syncAllAvatarViews();
    if (this.modalEl) {
      this.modalEl.classList.remove('hidden');
    }
  }

  public close(): void {
    if (this.modalEl) {
      this.modalEl.classList.add('hidden');
    }
  }

  public selectPreset(index: number): void {
    this.store.update(state => {
      state.avatarMode = 'gallery';
      state.avatarSelectedIndex = index;
    });
    this.syncAllAvatarViews();
    this.renderPresets();
  }

  public uploadCustomFile(file: File): void {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result;
      if (typeof result === 'string') {
        this.store.update(state => {
          state.avatarMode = 'custom';
          state.avatarCustomSrc = result;
          state.avatarCustomTitle = `本地上传 · ${file.name}`;
        });
        this.syncAllAvatarViews();
        this.renderPresets();
        this.close();
      }
    };
    reader.readAsDataURL(file);
  }

  public applyCustomUrl(url: string): boolean {
    const trimmed = (url || '').trim();
    if (!trimmed) return false;

    this.store.update(state => {
      state.avatarMode = 'custom';
      state.avatarCustomSrc = trimmed;
      state.avatarCustomTitle = '自定义网络原图';
    });
    this.syncAllAvatarViews();
    this.renderPresets();
    this.close();
    return true;
  }

  /**
   * 生成标准正圆形 <img> 标签 HTML
   */
  public getAvatarHtml(extraClass: string = ''): string {
    const active = this.store.getActiveAvatar();
    return `<img src="${this.escape(active.src)}" alt="${this.escape(active.alt)}" class="circular-avatar-img ${extraClass}" />`;
  }

  /**
   * 渲染弹窗内的主头像与预备头像双卡片
   */
  public renderPresets(): void {
    const grid = document.getElementById('avatar-preset-grid');
    if (!grid) return;

    const currentMode = this.store.state.avatarMode;
    const currentIdx = this.store.state.avatarSelectedIndex;

    grid.innerHTML = this.store.avatarGallery.map((item: AvatarItem, idx: number) => {
      const isSelected = currentMode === 'gallery' && currentIdx === idx;
      const badgeText = idx === 0 ? '默认主头像' : '预备头像';
      return `
        <div onclick="window.AppController.selectAvatarPreset(${idx})"
             class="p-2.5 rounded-2xl border-2 transition cursor-pointer flex flex-col items-center gap-1.5 active:scale-95 ${
               isSelected
                 ? 'border-blue-500 bg-blue-50/50 shadow-sm'
                 : 'border-slate-100 bg-white/70 hover:bg-white'
             }">
          <span class="text-[10px] font-bold px-2 py-0.5 rounded-full ${
            isSelected ? 'accent-pill' : 'bg-slate-100 text-slate-500'
          }">
            ${badgeText}
          </span>
          <div class="circular-avatar-frame w-16 h-16">
            <img src="${this.escape(item.src)}"
                 alt="${this.escape(item.alt)}"
                 class="circular-avatar-img" />
          </div>
          <div class="text-xs font-bold text-slate-800">${this.escape(item.title)}</div>
          <div class="text-[10px] text-slate-400 truncate max-w-full">${this.escape(item.fileName)}</div>
        </div>
      `;
    }).join('');
  }

  /**
   * 同步应用内所有 AI 头像容器 (气泡旁、立绘区、设置卡片、模态框)
   */
  public syncAllAvatarViews(): void {
    const avatarHtml = this.getAvatarHtml();
    const active = this.store.getActiveAvatar();

    const modalPreview = document.getElementById('modal-avatar-preview');
    const settingsPreview = document.getElementById('settings-avatar-preview');
    const stagePreview = document.getElementById('stage-avatar-container');

    if (modalPreview) modalPreview.innerHTML = avatarHtml;
    if (settingsPreview) settingsPreview.innerHTML = avatarHtml;
    if (stagePreview) stagePreview.innerHTML = avatarHtml;

    const modalCaption = document.getElementById('modal-avatar-caption');
    const settingsName = document.getElementById('settings-avatar-name');

    const desc = `${active.badge} · ${active.fileName}`;
    if (modalCaption) modalCaption.textContent = desc;
    if (settingsName) settingsName.textContent = `当前：${desc}`;
  }

  private escape(str: string): string {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}
