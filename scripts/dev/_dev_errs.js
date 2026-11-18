// 在页面加载前注入错误捕获，再 reload，最后读回 window.__errs（定位 WebView 内启动异常）。
const WebSocket = require('ws');
const wsUrl = process.argv[2];
const ws = new WebSocket(wsUrl, { perMessageDeflate: false, maxPayload: 64 * 1024 * 1024 });
let id = 0;
const pending = new Map();
const send = (method, params) =>
  new Promise((resolve, reject) => {
    const msgId = ++id;
    pending.set(msgId, { resolve, reject });
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });

ws.on('message', (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
  }
});

ws.on('open', async () => {
  await send('Runtime.enable', {});
  await send('Page.enable', {});
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: `window.__errs=[];window.addEventListener('error',e=>{try{window.__errs.push((e.message||'')+' @'+e.filename+':'+e.lineno+':'+e.colno)}catch(_){}});window.addEventListener('unhandledrejection',e=>{try{window.__errs.push('rejection: '+(e.reason&&(e.reason.stack||e.reason.message)||e.reason))}catch(_){}});`,
  });
  await send('Page.reload', { ignoreCache: true });
  await new Promise((r) => setTimeout(r, 10000));
  const res = await send('Runtime.evaluate', {
    expression: `JSON.stringify({errs: window.__errs, init: !!window.__aemeathAppInitialized, tts: !!(window.AppController && window.AppController.tts)})`,
    returnByValue: true,
  });
  console.log(res.result?.value);
  ws.close();
  process.exit(0);
});

ws.on('error', (e) => {
  console.log('WS ERROR:', e.message);
  process.exit(1);
});
