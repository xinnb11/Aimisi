// 临时探针：真实点击「进入」走完首屏，再切到语音页截图。
const puppeteer = require('puppeteer-core');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ROOT = 'D:\\语音模型项目';

async function run(browser, name, setup) {
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));
  await page.goto('file:///' + path.join(ROOT, 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded',
    timeout: 60000,
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 30000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3500));
  await page.evaluate((code) => {
    const welcome = document.getElementById('welcome-notice-modal');
    if (welcome) welcome.classList.add('opacity-0', 'pointer-events-none', 'hidden');
    try { window.AppController.onBottomTabChange(1); } catch (e) { console.error('tab', e); }
    if (code) { (new Function(code))(); }
  }, setup || '');
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: path.join(ROOT, name) });
  console.log('saved', name);
  await page.close();
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=2'],
  });
  await run(browser, '_probe_a.png', '');
  await run(
    browser,
    '_probe_b.png',
    `const ac = window.AppController;
     const m = ac.state.voiceModels.find(x => x.id === 'aemeath-sovits-pro');
     m.installed = true; m.size = '1.17 GB'; ac.state.activeModelId = m.id;
     ac.renderVoiceView();`
  );
  await browser.close();
})();
