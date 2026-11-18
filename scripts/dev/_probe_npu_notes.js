(async () => {
  const ac = window.AppController;
  const tts = ac.tts;
  const model = { id: 'aemeath-sovits-pro', name: '爱弥斯 · 电子幽灵高还原版', arch: 'GPT-SoVITS v2', engine: 'ONNX Runtime' };
  try { await tts.unmountModel(); } catch (e) {}
  const mounted = await tts.mountModel(model, 'npu');
  const diag = await tts.diagnose();
  const plugin = tts.getNativePlugin();
  let spoken = null;
  if (plugin) {
    spoken = await plugin.synthesizeSpeech({
      text: '爱弥斯语音已经恢复正常，可以正常对话了。',
      emotion: '温柔',
      speed: 1.0,
      volume: 1.0,
      playImmediately: true
    });
  }
  return JSON.stringify({
    mounted: !!mounted.success,
    accelerationActive: diag.accelerationActive,
    notes: diag.mountNotes,
    spoken: spoken && {
      success: spoken.success,
      durationMs: spoken.durationMs,
      played: spoken.played,
      samplesCount: spoken.samplesCount
    }
  });
})()
