// 从 App 的 WebView 里取回 window.__wavB64 并落盘成 WAV（调试用）。
// 用法: node pull_device_wav.js "<ws-url>" "<out-path>"
const WebSocket = require('ws');
const fs = require('fs');

const wsUrl = process.argv[2];
const outPath = process.argv[3];

const ws = new WebSocket(wsUrl, { perMessageDeflate: false, maxPayload: 256 * 1024 * 1024 });
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
  }
});

async function evaluate(expression) {
  const res = await send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (res.exceptionDetails) throw new Error(JSON.stringify(res.exceptionDetails));
  return res.result?.value;
}

ws.on('open', async () => {
  try {
    await send('Runtime.enable', {});
    const total = await evaluate('(window.__wavB64 || "").length');
    if (!total) throw new Error('window.__wavB64 为空，先在页面里合成一次 playImmediately:false');
    const chunk = 200000;
    let b64 = '';
    for (let offset = 0; offset < total; offset += chunk) {
      const part = await evaluate(
        `window.__wavB64.slice(${offset}, ${Math.min(offset + chunk, total)})`
      );
      b64 += part;
    }
    fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
    console.log('saved', outPath, fs.statSync(outPath).size, 'bytes (b64', total, ')');
  } catch (error) {
    console.log('ERROR:', error.message);
    process.exitCode = 1;
  } finally {
    ws.close();
    setTimeout(() => process.exit(process.exitCode || 0), 100);
  }
});

ws.on('error', (error) => {
  console.log('WS ERROR:', error.message);
  process.exit(1);
});
