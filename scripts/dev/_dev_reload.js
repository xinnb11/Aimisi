// 重载页面并捕获所有异常/控制台错误，用于定位 WebView 内的启动失败。
const WebSocket = require('ws');

const wsUrl = process.argv[2];
const ws = new WebSocket(wsUrl, { perMessageDeflate: false, maxPayload: 64 * 1024 * 1024 });
let id = 0;
const pending = new Map();

function send(method, params) {
  return new Promise((resolve, reject) => {
    const msgId = ++id;
    pending.set(msgId, { resolve, reject });
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });
}

ws.on('message', (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) reject(new Error(JSON.stringify(msg.error)));
    else resolve(msg.result);
    return;
  }
  if (msg.method === 'Runtime.exceptionThrown') {
    const d = msg.params.exceptionDetails;
    console.log('[exception]', d.text, '|', d.exception?.description?.slice(0, 300) || '');
  }
  if (msg.method === 'Runtime.consoleAPICalled') {
    const args = (msg.params.args || []).map((a) => a.value ?? a.description).join(' ');
    if (msg.params.type === 'error') console.log('[console.error]', args.slice(0, 400));
  }
  if (msg.method === 'Log.entryAdded') {
    console.log('[log:' + msg.params.entry.level + ']', msg.params.entry.text.slice(0, 300));
  }
});

ws.on('open', async () => {
  await send('Runtime.enable', {});
  await send('Log.enable', {});
  await send('Page.enable', {});
  await send('Page.reload', { ignoreCache: true });
  await new Promise((r) => setTimeout(r, 12000));
  const res = await send('Runtime.evaluate', {
    expression: `JSON.stringify({ac: typeof AppController, init: !!window.__aemeathAppInitialized, tts: !!(window.AppController && window.AppController.tts), ready: document.readyState})`,
    returnByValue: true,
  });
  console.log('STATE:', res.result?.value);
  ws.close();
  process.exit(0);
});

ws.on('error', (e) => {
  console.log('WS ERROR:', e.message);
  process.exit(1);
});
