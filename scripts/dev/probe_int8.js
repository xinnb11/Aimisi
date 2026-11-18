// 真机 Int8 计时 + 出声取证：挂载（含装载耗时）→ 合成一句 → 落盘 WAV。
const WebSocket = require('ws');
const fs = require('fs');

const [, , wsUrl, outPath, text, emotion] = process.argv;
const TARGET = { id: 'aemeath-sovits-pro', arch: 'GPT-SoVITS v2', engine: 'ONNX Runtime' };

async function main() {
  const ws = new WebSocket(wsUrl, { perMessageDeflate: false, maxPayload: 256 * 1024 * 1024 });
  await new Promise((resolve, reject) => {
    ws.once('open', resolve);
    ws.once('error', reject);
  });
  let id = 0;
  const pending = new Map();
  ws.on('message', (raw) => {
    const msg = JSON.parse(raw.toString());
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });
  const send = (method, params) =>
    new Promise((resolve) => {
      const myId = ++id;
      pending.set(myId, resolve);
      ws.send(JSON.stringify({ id: myId, method, params }));
    });
  const evaluate = async (expression) => {
    const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    return res.result?.result?.value;
  };

  const ready = await evaluate(
    `(async()=>{
       const wait=(ms)=>new Promise(r=>setTimeout(r,ms));
       for (let i=0;i<40;i++) {
         const modal=document.getElementById('welcome-notice-modal');
         if (modal) modal.classList.add('hidden','opacity-0','pointer-events-none');
         if (window.AppController && window.AppController.tts) return 'ready';
         const btn=document.querySelector('#btn-portal-launch');
         if (btn) btn.click();
         await wait(500);
       }
       return 'timeout';
     })()`
  );
  if (ready !== 'ready') {
    console.log('NOT_READY', ready);
    ws.close();
    return;
  }

  const metaJson = await evaluate(
    `(async()=>{
       const t=window.AppController.tts;
       const t0=performance.now();
       const mount=await t.mountModel(${JSON.stringify(TARGET)},'cpu');
       const mountMs=Math.round(performance.now()-t0);
       if (!mount.success) return JSON.stringify({error:'mount failed: '+mount.message, mountMs});
       const t1=performance.now();
       const r=await Capacitor.Plugins.AemeathTtsPlugin.synthesizeSpeech({text:${JSON.stringify(text)},emotion:${JSON.stringify(emotion)},playImmediately:false});
       const synthMs=Math.round(performance.now()-t1);
       if (!r || !r.audioBase64) return JSON.stringify({error:'no audio', keys:Object.keys(r||{}), mountMs, synthMs});
       window.__aemeathWav = r.audioBase64;
       return JSON.stringify({
         mountMs, synthMs,
         engine: r.engine, acceleration: r.accelerationActive,
         durationMs: r.durationMs, samplesCount: r.samplesCount,
         played: r.played, notes: mount.notes,
         len: r.audioBase64.length
       });
     })()`
  );
  const meta = JSON.parse(metaJson);
  if (meta.error) {
    console.log('FAILED', metaJson);
    ws.close();
    return;
  }
  const chunks = [];
  for (let offset = 0; offset < meta.len; offset += 20000) {
    chunks.push(await evaluate(`window.__aemeathWav.slice(${offset}, ${offset + 20000})`));
  }
  const buf = Buffer.from(chunks.join(''), 'base64');
  fs.writeFileSync(outPath, buf);
  delete meta.len;
  console.log(JSON.stringify({ ...meta, bytes: buf.length, out: outPath }));
  ws.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

