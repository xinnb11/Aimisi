// 首页清洁度验证：聊天首屏不应有任何预置演示文案，立绘模式的示范台词也应为空。
const puppeteer = require('puppeteer-core');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ROOT = 'D:\\语音模型项目';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  const errs = [];
  page.on('pageerror', (e) => errs.push(e.message));
  await page.goto('file:///' + path.join(ROOT, 'dist', 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded', timeout: 60000,
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 30000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3200));

  const report = await page.evaluate(() => {
    const modal = document.getElementById('welcome-notice-modal');
    if (modal) modal.classList.add('opacity-0', 'pointer-events-none', 'hidden');
    const app = window.AppController;
    app.onBottomTabChange(0);
    app.renderMessages();
    const stream = document.getElementById('chat-stream-container');
    const stageAction = document.getElementById('stage-action-text');
    const stageSpeech = document.getElementById('stage-speech-text');
    return {
      messagesLength: app.state.messages.length,
      streamChildren: stream ? stream.children.length : -1,
      streamText: stream ? stream.textContent.trim().slice(0, 60) : null,
      stageActionText: stageAction ? stageAction.textContent.trim() : null,
      stageSpeechText: stageSpeech ? stageSpeech.textContent.trim() : null,
    };
  });
  await page.screenshot({ path: path.join(ROOT, 'scripts', 'dev', 'fixtures', '_chat_home_clean.png') });
  console.log(JSON.stringify({ ...report, pageErrors: errs }, null, 1));
  await browser.close();
})();
