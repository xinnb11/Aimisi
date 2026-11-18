// 语音包「未开放」状态验证：卡片按钮文案 / 点击提示 / 自定义拉取拦截。
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
  const toasts = [];
  page.on('console', (m) => { if (m.text().includes('[Voice]')) toasts.push(m.text()); });
  await page.goto('file:///' + path.join(ROOT, 'dist', 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded', timeout: 60000,
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 30000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3000));

  const report = await page.evaluate(() => {
    const modal = document.getElementById('welcome-notice-modal');
    if (modal) modal.classList.add('opacity-0', 'pointer-events-none', 'hidden');
    const app = window.AppController;
    app.onBottomTabChange(1);
    app.renderVoiceView();

    const cards = ['aemeath-vits-lite', 'aemeath-sovits-pro', 'aemeath-indextts-ultra'].map((id) => {
      const btn = document.getElementById(`btn-pull-preset-${id}`);
      const card = document.getElementById(`preset-card-${id}`);
      return {
        id,
        found: !!btn,
        btnText: btn ? btn.textContent.trim() : null,
        icon: btn ? (btn.querySelector('i')?.className || '') : '',
        textColor: btn ? btn.className.includes('text-slate-400') : null,
        cardOpacity: card ? card.className.includes('opacity-80') : null,
        disabledLooking: btn ? btn.className.includes('cursor-not-allowed') : null,
      };
    });

    // 点击「未开放」卡片：应弹 toast，且不进入下载态
    document.getElementById('btn-pull-preset-aemeath-indextts-ultra').click();
    const afterClickDownloading = app.state.isDownloading;

    // 自定义拉取选 Bert-VITS2 架构：应被拦截
    app.onNewModelArchChange(0);
    const cmdInput = document.getElementById('input-modelscope-cmd');
    const beforeCmd = cmdInput ? cmdInput.value : '';
    // 直接调内部执行入口（按钮 onclick）
    const customBtn = document.getElementById('btn-custom-pull');
    if (customBtn) customBtn.click();
    return { cards, afterClickDownloading, beforeCmd, downloadingAfterCustom: app.state.isDownloading };
  });

  await new Promise((r) => setTimeout(r, 600));
  const toastText = await page.evaluate(() => {
    const t = document.querySelector('.toast, [id*=toast]');
    const all = Array.from(document.querySelectorAll('div')).map((d) => d.textContent.trim()).filter((x) => x.includes('未开放'));
    return all.slice(-2);
  });
  console.log(JSON.stringify({ ...report, toastText, consoleHits: toasts }, null, 1));
  await page.screenshot({ path: path.join(ROOT, 'scripts', 'dev', 'fixtures', '_voice_notopen.png') });
  await browser.close();
})();
