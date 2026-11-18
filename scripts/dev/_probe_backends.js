(async () => {
  const ac = window.AppController;
  const tts = ac.tts;
  const model = {
    id: 'aemeath-sovits-pro',
    name: '爱弥斯 · 电子幽灵高还原版',
    arch: 'GPT-SoVITS v2',
    engine: 'ONNX Runtime',
    repo: 'xinxin111x/Aimisi'
  };
  const results = [];
  for (const backend of ['npu', 'gpu', 'cpu']) {
    const entry = { backend };
    try {
      try { await tts.unmountModel(); } catch (e) {}
      const mounted = await tts.mountModel(model, backend);
      entry.mounted = !!(mounted && mounted.success);
      const diag = await tts.diagnose();
      entry.accelerationActive = diag.accelerationActive;
      const plugin = tts.getNativePlugin();
      entry.hasPlugin = !!plugin;
      if (plugin) {
        const t0 = Date.now();
        const s = await plugin.synthesizeSpeech({
          text: '模型验证，爱弥斯语音合成正常。',
          emotion: '温柔',
          speed: 1.0,
          volume: 1.0,
          playImmediately: false
        });
        entry.synthMs = Date.now() - t0;
        entry.synth = {
          success: s && s.success,
          durationMs: s && s.durationMs,
          samplesCount: s && s.samplesCount,
          sampleRate: s && s.sampleRate,
          audioBytes: s && s.audioBase64 ? Math.floor(s.audioBase64.length * 3 / 4) : 0,
          message: s && s.message
        };
      }
    } catch (e) {
      entry.error = String((e && e.message) || e);
    }
    results.push(entry);
  }
  return JSON.stringify(results);
})()
