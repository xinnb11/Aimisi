/**
 * VoiceView.ts - 本地离线多语音模型库与推理引擎配置视图
 * 支持多模型管理库、魔搭社区 (ModelScope) 自定义指令拉取、本地文件建库导入、4 情绪试听及硬件加速
 */

import { AppStore, VoiceModelItem } from '../../core/store/AppStore';
import { LocalOnnxTtsBridge } from '../../core/tts/LocalOnnxTtsBridge';

export class VoiceView {
  private store: AppStore;
  private tts: LocalOnnxTtsBridge;

  constructor(store: AppStore, tts: LocalOnnxTtsBridge) {
    this.store = store;
    this.tts = tts;
  }

  public getActiveModel(): VoiceModelItem | null {
    const id = this.store.state.activeModelId;
    if (!id) return null;
    return this.store.state.voiceModels.find(m => m.id === id && m.installed) || null;
  }

  public previewVoice(): void {
    const active = this.getActiveModel();
    if (!active) {
      console.warn('[VoiceView] 尚未挂载模型，使用降级模拟音色播报');
      this.tts.speak('漂泊者你好，我是爱弥斯，当前未挂载本地模型，正使用降级模拟音色为您播报。', this.store.state.emotion);
      return;
    }
    const text = `漂泊者你好，我是爱弥斯，${active.name}已成功激活！`;
    this.tts.speak(text, this.store.state.emotion);
  }

  public auditionEmotionBin(emotion: '平静' | '开心' | '温柔' | '调侃' | '愤怒生气'): void {
    this.store.update(state => {
      state.emotion = emotion;
    });
    const active = this.getActiveModel();
    const prefix = active ? active.name : '降级音色';
    this.tts.speak(`正在调用爱弥斯${prefix}的${emotion}情绪特征向量进行端侧推理`, emotion);
  }

  public setActiveModel(modelId: string): void {
    const target = this.store.state.voiceModels.find(m => m.id === modelId && m.installed);
    if (!target) return;
    this.store.update(state => {
      state.activeModelId = target.id;
    });
    this.previewVoice();
  }

  public uninstallModel(modelId: string): void {
    this.store.update(state => {
      const model = state.voiceModels.find(m => m.id === modelId);
      if (!model) return;
      if (model.isCustom) {
        state.voiceModels = state.voiceModels.filter(m => m.id !== modelId);
      } else {
        model.installed = false;
      }
      if (state.activeModelId === modelId) {
        const nextInstalled = state.voiceModels.find(m => m.installed);
        state.activeModelId = nextInstalled ? nextInstalled.id : null;
      }
    });
  }

  public async importVoicePack(file: File): Promise<string> {
    if (!file) return '';
    const manifest = await this.tts.mountVoicePack(file);
    const mb = Math.max(12, Math.round(file.size / (1024 * 1024)) || 78);
    const fileName = file.name.replace(/\.[^.]+$/, '');
    const newId = 'local-' + Date.now();

    const newModel: VoiceModelItem = {
      id: newId,
      name: `爱弥斯 · ${manifest.name || fileName}`,
      arch: 'Bert-VITS2',
      engine: 'Sherpa-ONNX',
      latency: '18ms',
      size: `${mb} MB`,
      repo: `local/${file.name}`,
      installed: true,
      isCustom: true
    };

    this.store.update(state => {
      state.voiceModels.push(newModel);
      state.activeModelId = newId;
    });

    this.previewVoice();
    return newModel.name;
  }

  public setSubTab(idx: number): void {
    this.store.update(state => {
      state.voiceSubTab = idx;
    });
  }

  public setComputeBackend(idx: number): string {
    this.store.update(state => {
      state.computeBackend = idx;
    });
    const labels = ['NNAPI / NPU 加速', 'Vulkan / GPU 加速', 'ARM NEON / CPU 线程'];
    const text = labels[idx] || labels[0];
    const labelEl = document.getElementById('compute-backend-label');
    if (labelEl) labelEl.textContent = text;
    return text;
  }

  public setAutoPlay(val: boolean): void {
    this.store.update(state => {
      state.autoPlayVoice = val;
    });
  }

  public setStripActions(val: boolean): void {
    this.store.update(state => {
      state.stripActions = val;
    });
  }
}
