(async () => {
  const ac = window.AppController;
  ac.state.apiKey = '';
  ac.state.baseUrl = 'https://api.deepseek.com/v1';
  ac.state.modelName = 'deepseek-chat';
  let chatError = '';
  try {
    await LlmApiClient.streamChat(ac.state, {
      onDelta() {},
      onDone() {},
      onError(e) { chatError = String((e && e.message) || e); },
    });
  } catch (e) {
    chatError = chatError || String((e && e.message) || e);
  }
  const saved = {};
  ac.state.apiKey = 'sk-test-invalid-000';
  try {
    const r = await fetch(ac.state.baseUrl + '/models', {
      headers: { Authorization: 'Bearer sk-test-invalid-000' },
    });
    saved.httpStatus = r.status;
    saved.body = (await r.text()).slice(0, 120);
  } catch (e) {
    saved.netError = String((e && e.message) || e);
  }
  ac.saveApiConfig && ac.saveApiConfig();
  return JSON.stringify({
    chatError,
    storedKey: (localStorage.getItem('aemeath_llm_config_v1') || '').slice(0, 80),
    fetch: saved,
  });
})();
