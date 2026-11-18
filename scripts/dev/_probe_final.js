(async () => {
  const results = {};
  const t = window.AppController.tts;
  try {
    const d0 = await t.diagnose();
    results.diagnoseBeforeMount = {
      modelId: d0.modelId,
      modelDir: d0.modelDir,
      exists: d0.inventory && d0.inventory.exists,
      bertReady: d0.inventory && d0.inventory.bertReady,
      conditioningReady: d0.inventory && d0.inventory.conditioningReady,
    };
  } catch (e) {
    results.diagnoseBeforeMount = { error: String((e && e.message) || e) };
  }

  try {
    const mount = await t.mountModel(
      { id: 'aemeath-sovits-pro', arch: 'GPT-SoVITS v2', engine: 'ONNX Runtime' },
      'cpu'
    );
    results.mount = { success: mount.success, notes: mount.notes, accelerator: mount.accelerationActive };
  } catch (e) {
    results.mount = { error: String((e && e.message) || e) };
  }

  try {
    const r = await Capacitor.Plugins.AemeathTtsPlugin.synthesizeSpeech({
      text: '漂泊者，语音合成已经恢复正常了。',
      emotion: '温柔',
      playImmediately: true,
    });
    results.synth = { durationMs: r.durationMs, samples: r.samplesCount, played: r.played };
  } catch (e) {
    results.synth = { error: String((e && e.message) || e) };
  }

  await new Promise((res) => setTimeout(res, 600));
  try {
    const d1 = await t.diagnose();
    results.diagnoseAfter = { synthError: d1.synthError, synthInfo: d1.synthInfo, engine: d1.engine };
  } catch (e) {
    results.diagnoseAfter = { error: String((e && e.message) || e) };
  }
  return JSON.stringify(results);
})();
