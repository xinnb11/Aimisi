(async () => {
  const ac = window.AppController;
  const out = {};
  try {
    ac.previewVoice();
    out.previewVoice = 'invoked';
  } catch (e) {
    out.previewVoice = 'error: ' + String((e && e.message) || e);
  }
  await new Promise((r) => setTimeout(r, 5000));
  try {
    const d = await ac.tts.diagnose();
    out.synthError = d.synthError;
    out.synthInfo = d.synthInfo;
  } catch (e) {
    out.diagnoseError = String((e && e.message) || e);
  }
  return JSON.stringify(out);
})();
