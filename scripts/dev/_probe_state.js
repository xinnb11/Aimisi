(async () => {
  const ac = window.AppController;
  const out = {};
  out.models = (ac.state.voiceModels || []).map((m) => ({
    id: m.id,
    installed: !!m.installed,
    active: ac.state.activeModelId === m.id,
    size: m.size,
  }));
  out.llm = {
    baseUrl: ac.state.llmConfig && ac.state.llmConfig.baseUrl,
    model: ac.state.llmConfig && ac.state.llmConfig.modelName,
    keyLen: ((ac.state.llmConfig && ac.state.llmConfig.apiKey) || '').length,
  };
  try {
    ac.onBottomTabChange(1);
    out.tab = 'voice';
  } catch (e) {
    out.tabError = String((e && e.message) || e);
  }
  return JSON.stringify(out);
})();
