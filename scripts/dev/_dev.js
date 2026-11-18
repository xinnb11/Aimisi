// 通过 WebView DevTools 在 App 内执行 JS（调试用，不进产物）。
// 用法: node _dev.js "<ws-url>" "<js expression>"
const WebSocket = require('ws');

const fs = require('fs');
const wsUrl = process.argv[2];
const rawExpr = process.argv[3];
const expr = rawExpr && rawExpr.startsWith('@') ? fs.readFileSync(rawExpr.slice(1), 'utf8') : rawExpr;
const timeoutMs = Number(process.argv[4] || 240) * 1000;

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
  } else if (msg.method === 'Runtime.consoleAPICalled') {
    const args = (msg.params.args || []).map((a) => a.value ?? a.description).join(' ');
    console.log('[console]', args);
  } else if (msg.method === 'Log.entryAdded') {
    console.log('[log:' + msg.params.entry.level + ']', msg.params.entry.text);
  }
});

ws.on('open', async () => {
  try {
    await send('Runtime.enable', {});
    await send('Log.enable', {});
    const res = await send('Runtime.evaluate', {
      expression: expr,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    });
    if (res.exceptionDetails) {
      console.log('EXCEPTION:', JSON.stringify(res.exceptionDetails, null, 2));
      process.exitCode = 2;
    } else {
      const v = res.result?.value;
      console.log(typeof v === 'string' ? v : JSON.stringify(v, null, 2));
    }
  } catch (e) {
    console.log('ERROR:', e.message);
    process.exitCode = 1;
  } finally {
    ws.close();
    setTimeout(() => process.exit(process.exitCode || 0), 100);
  }
});

ws.on('error', (e) => {
  console.log('WS ERROR:', e.message);
  process.exit(1);
});

const timer = setTimeout(() => {
  console.log('TIMEOUT after', timeoutMs, 'ms');
  process.exit(3);
}, timeoutMs);
timer.unref();
