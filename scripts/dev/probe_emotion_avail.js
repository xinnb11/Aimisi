// 语调可用性验证：applyEmotionAvailability 后检查胶囊/滑块/持久化。
const puppeteer = require('puppeteer-core');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ROOT = 'D:\\语音模型项目';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto('file:///' + path.join(ROOT, 'dist', 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded', timeout: 60000,
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 30000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3000));

  const report = await page.evaluate(() => {
    const close = document.getElementById('welcome-notice-modal');
    if (close) close.classList.add('opacity-0', 'pointer-events-none', 'hidden');
    const app = window.AppController;
    const before = { emotion: app.state.emotion, available: app.availableEmotions.slice() };

    const avail = app.applyEmotionAvailability(['温柔', '调侃']);
    const pills = ['平静', '开心', '温柔', '调侃'].map((e) => {
      const b = document.getElementById(`btn-emo-${e}`);
      return { e, hidden: b ? b.classList.contains('hidden') : null, disabled: b ? !!b.disabled : null };
    });
    const segs = Array.from(document.querySelectorAll('#slider-emotion .lens-base-item')).map((s, i) => ({
      i, opacity: s.style.opacity, pe: s.style.pointerEvents, active: s.classList.contains('is-active'),
    }));
    const persisted = localStorage.getItem('aemeath_emotion_v1');

    // 尝试点不可用项（开心）：应被拦截且不改 state
    app.onEmotionChange(1);
    const afterBlocked = app.state.emotion;

    // 恢复全量，检查复原
    const restored = app.applyEmotionAvailability(['平静', '开心', '温柔', '调侃']);
    const pillsRestored = ['平静', '开心', '温柔', '调侃'].map((e) => {
      const b = document.getElementById(`btn-emo-${e}`);
      return { e, hidden: b ? b.classList.contains('hidden') : null };
    });
    return { before, avail, pills, segs, persisted, afterBlocked, restored, pillsRestored, emotionNow: app.state.emotion };
  });
  console.log(JSON.stringify(report, null, 1));
  await browser.close();
})();
