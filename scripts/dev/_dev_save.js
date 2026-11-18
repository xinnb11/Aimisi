// 临时：连真机 WebView，等主界面就绪 -> 挂载模型 -> 合成一句 -> 把 WAV 落盘，便于本地分析音质。
const WebSocket = require('ws');
const fs = require('fs');

const [, , wsUrl, outPath, text, emotion] = process.argv;

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
       if (window.AppController && window.AppController.tts) return 'ready';
       const btn=document.querySelector('#btn-portal-launch');
       if (btn) btn.click();
       for (let i=0;i<30;i++) {
         await wait(1000);
         const modal=document.getElementById('welcome-notice-modal');
         if (modal) modal.classList.add('hidden','opacity-0','pointer-events-none');
         if (window.AppController && window.AppController.tts) return 'ready';
       }
       return 'timeout';
     })()`
  );
  if (ready !== 'ready') {
    console.log('NOT_READY', ready);
    ws.close();
    return;
  }

  const json = await evaluate(
    `(async()=>{
       const t=window.AppController.tts;
       const mount=await t.mountModel({id:'aemeath-sovits-pro',arch:'GPT-SoVITS v2',engine:'ONNX Runtime'},'cpu');
       if (!mount.success) return JSON.stringify({error:'mount failed: '+mount.message});
       const r=await Capacitor.Plugins.AemeathTtsPlugin.synthesizeSpeech({text:${JSON.stringify(text)},emotion:${JSON.stringify(emotion)},playImmediately:false});
       if (!r || !r.audioBase64) return JSON.stringify({error:'no audio', res:Object.keys(r||{})});
       window.__aemeathWav = r.audioBase64;
       return JSON.stringify({durationMs:r.durationMs,samplesCount:r.samplesCount,len:r.audioBase64.length});
     })()`
  );
  const meta = JSON.parse(json);
  if (meta.error) {
    console.log('FAILED', json);
    ws.close();
    return;
  }

  const chunks = [];
  for (let offset = 0; offset < meta.len; offset += 20000) {
    const piece = await evaluate(`window.__aemeathWav.slice(${offset}, ${offset + 20000})`);
    chunks.push(piece);
  }
  const buf = Buffer.from(chunks.join(''), 'base64');
  fs.writeFileSync(outPath, buf);
  console.log(JSON.stringify({ ...meta, bytes: buf.length, out: outPath }));
  ws.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
