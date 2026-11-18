// 校验 splitForSpeech 的切分结果（无头 Chrome，不依赖真机）。
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
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('file:///' + path.join(ROOT, 'index.html').replace(/\\/g, '/'), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForSelector('#btn-portal-launch', { timeout: 30000 });
  await page.click('#btn-portal-launch');
  await new Promise((r) => setTimeout(r, 3500));

  const out = await page.evaluate(() => {
    const tts = window.AppController && window.AppController.tts;
    if (!tts || typeof tts.splitForSpeech !== 'function') return { error: 'no splitForSpeech' };
    const sentence = '今天的任务就做到这里吧，我在地图右侧发现了一个隐藏房间。';
    const pieces = tts.splitForSpeech(sentence, 12);
    return {
      pieces,
      lengths: pieces.map((p) => p.length),
      total: pieces.join('').length,
      original: sentence.length,
      empty: typeof tts.splitForSpeech === 'function' ? tts.splitForSpeech('', 12) : null,
      short: tts.splitForSpeech('你好呀', 12),
    };
  });
  console.log(JSON.stringify({ out, errors }, null, 2));
  await browser.close();
})();
