(async () => {
  const t = window.AppController.tts;
  const chunks = [];
  const r = await Capacitor.Plugins.AemeathTtsPlugin.synthesizeSpeech({
    text: '漂泊者，语音合成已经恢复正常了。',
    emotion: '温柔',
    playImmediately: false,
  });
  const b64 = r.audioBase64 || '';
  return JSON.stringify({ len: b64.length, durationMs: r.durationMs, samples: r.samplesCount, rate: r.sampleRate, b64 });
})();
