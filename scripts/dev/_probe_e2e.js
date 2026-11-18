(async () => {
  const t = window.AppController.tts;
  const mount = await t.mountModel({ id: 'aemeath-sovits-pro', arch: 'GPT-SoVITS v2', engine: 'ONNX Runtime' }, 'cpu');
  const out = [];
  for (const [text, emotion] of [
    ['漂泊者，语音合成已经恢复正常了。', '温柔'],
    ['叮，模型加载完毕，现在你可以去体验语音模型了。', '开心'],
  ]) {
    const r = await Capacitor.Plugins.AemeathTtsPlugin.synthesizeSpeech({ text, emotion, playImmediately: true });
    out.push({ emotion, ms: r.durationMs, peak: r.peak, played: r.played });
  }
  let llmError = '';
  try {
    await window.AppController.llm.streamChat(
      [{ role: 'user', content: '你好' }],
      { onDelta() {}, onDone() {}, onError(e) { llmError = String(e && e.message || e); } },
      { apiKey: '', baseUrl: 'https://api.deepseek.com/v1', modelName: 'deepseek-chat' }
    );
  } catch (e) {
    llmError = llmError || String(e && e.message || e);
  }
  return JSON.stringify({ mount: mount.success, synth: out, llmError });
})();
