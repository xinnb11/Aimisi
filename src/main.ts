/**
 * main.ts - Aemeath Terminal 应用程序主入口
 */

import { AppController } from './ui/AppController';

document.addEventListener('DOMContentLoaded', () => {
  const app = new AppController();
  app.init();
});
