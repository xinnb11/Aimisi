import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

/**
 * index.html 里的头像 / 壁纸 / 看板娘是「原始文件名」硬编码引用：
 *   - 静态标签：<img src="1794568840819.jpg">
 *   - 内联脚本：state.avatar.src = '1794568840819.jpg'、applyImageBackground('1794568840819.jpg')
 *
 * Vite 构建时只重写 HTML 里的静态标签（改写成 assets/<name>-<hash>.jpg），
 * 内联 JS 字符串不会被重写。于是构建产物里出现「有哈希的文件 + JS 仍找原始名」的错位，
 * 结果就是：首屏静态头像能显示，但切换头像 / 换壁纸 / 看板娘全部破图（404）。
 *
 * 这个插件在产物阶段把原图复制到 dist 根目录，并把静态标签改回原始名，
 * 最后删掉没人再引用的哈希副本——只保留一份，APK 体积不增加。
 */
const RAW_IMAGES = ['1794568840819.jpg', '1794568844420.jpg', 'kanban_aemeath.png'];

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function aemeathRawImages(): Plugin {
  return {
    name: 'aemeath-raw-images',
    apply: 'build',
    writeBundle(options) {
      const outDir = path.resolve(process.cwd(), options.dir ?? 'dist');
      const htmlPath = path.join(outDir, 'index.html');
      if (!fs.existsSync(htmlPath)) return;

      let html = fs.readFileSync(htmlPath, 'utf8');
      let copied = 0;

      for (const name of RAW_IMAGES) {
        const src = path.join(process.cwd(), name);
        if (!fs.existsSync(src)) continue;

        const ext = path.extname(name);
        const stem = name.slice(0, -ext.length);
        const pattern = new RegExp(
          `assets/${escapeRegExp(stem)}-[A-Za-z0-9_-]{6,}${escapeRegExp(ext)}`,
          'g',
        );
        const hashedFiles = [...new Set((html.match(pattern) ?? []).map((m) => m.split('/')[1]))];

        // 静态标签改回原始文件名，保证与内联 JS 的引用一致
        html = html.replace(pattern, name);

        // 原图放到 dist 根目录（相对路径才能同时满足 file:// 与 Capacitor assets 服务）
        fs.copyFileSync(src, path.join(outDir, name));
        copied += 1;

        // 删掉已无人引用的哈希副本
        for (const f of hashedFiles) {
          const abs = path.join(outDir, 'assets', f);
          if (fs.existsSync(abs)) fs.rmSync(abs, { force: true });
        }
      }

      fs.writeFileSync(htmlPath, html, 'utf8');
      this.info(`[aemeath-raw-images] 已复制 ${copied} 张原图到产物根目录，并统一引用为原始文件名`);

      // 兜底自检：产物里任何图片引用都必须真实存在，否则直接让构建失败，
      // 避免「首屏能看、切换头像/换壁纸就破图」这种只在运行时才暴露的问题。
      const refs = new Set<string>();
      for (const m of html.matchAll(/(?:src|href)="([^"]+\.(?:jpg|jpeg|png|webp|gif))"/gi)) {
        refs.add(m[1]);
      }
      for (const m of html.matchAll(/['"]([^'"<>\n]{1,120}?\.(?:jpg|jpeg|png|webp|gif))['"]/gi)) {
        refs.add(m[1]);
      }
      const missing = [...refs].filter(
        (r) => !/^(?:data:|https?:|blob:)/i.test(r) && !fs.existsSync(path.join(outDir, r.replace(/^\.?\//, ''))),
      );
      if (missing.length > 0) {
        throw new Error(`[aemeath-raw-images] 产物缺少图片引用：${missing.join(', ')}`);
      }
    },
  };
}

export default defineConfig({
  root: '.',
  base: './',
  plugins: [aemeathRawImages()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    target: 'es2022',
  },
  server: {
    port: 3000,
    open: true,
  },
});
