const puppeteer = require('puppeteer-core');
const path = require('path');
const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ROOT = 'D:\\语音模型项目';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push('CONSOLE: ' + m.text().slice(0, 200));
  });
  await page.goto('file:///' + path.join(ROOT, 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 20000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 2500));
  const state = await page.evaluate(async () => {
    const ac = window.AppController;
    const out = { hasController: !!ac };
    try {
      const d = await ac.tts.diagnose();
      out.diag = d;
    } catch (e) { out.diagErr = String(e); }
    try {
      const ok = await ac.testApiConnection();
      out.apiTest = ok;
    } catch (e) { out.apiErr = String(e); }
    return out;
  });
  console.log(JSON.stringify({ errors, state }, null, 2));
  await browser.close();
})();
