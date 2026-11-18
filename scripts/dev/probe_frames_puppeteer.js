// 本机无头 Chrome 测切页帧率（相对比较用，绝对值与真机不同）。
// 用法: node scripts/dev/probe_frames_puppeteer.js
const path = require('path');
const puppeteer = require('puppeteer-core');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ROOT = 'D:\\语音模型项目';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu-vsync', '--hide-scrollbars'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  await page.goto('file:///' + path.join(ROOT, 'index.html').replace(/\\/g, '/'), { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#btn-portal-launch');
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3800));
  await page.evaluate(() => {
    const m = document.getElementById('welcome-notice-modal');
    if (m) m.classList.add('opacity-0', 'pointer-events-none', 'hidden');
  });

  const result = await page.evaluate(async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const ac = window.AppController;
    const samples = [];
    const measure = async (label, fn) => {
      const deltas = [];
      let last = performance.now();
      let stop = false;
      const tick = () => {
        const now = performance.now();
        deltas.push(now - last);
        last = now;
        if (!stop) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      await sleep(80);
      const t0 = performance.now();
      fn();
      const jsMs = performance.now() - t0;
      await sleep(700);
      stop = true;
      const sorted = deltas.slice(1).sort((a, b) => a - b);
      const p = (q) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))] || 0;
      samples.push({
        label,
        jsMs: Math.round(jsMs * 10) / 10,
        frames: sorted.length,
        median: Math.round(p(0.5) * 10) / 10,
        p90: Math.round(p(0.9) * 10) / 10,
        max: Math.round(sorted[sorted.length - 1] * 10) / 10,
        over32: sorted.filter((d) => d > 32).length,
      });
      await sleep(250);
    };
    await measure('chat->voice', () => ac.onBottomTabChange(1));
    await measure('voice->settings', () => ac.onBottomTabChange(2));
    await measure('settings->chat', () => ac.onBottomTabChange(0));
    await measure('chat->voice#2', () => ac.onBottomTabChange(1));
    return samples;
  });

  console.log(JSON.stringify(result, null, 1));
  await browser.close();
})();
