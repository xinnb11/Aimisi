/**
 * ConvexLensMapGenerator.ts - 2D 球面凸透镜桶形法线位移贴图生成器
 * 严禁添加彩虹色散，生成纯净无色差的高精度凸透镜法线贴图并注入 SVG DisplacementMap 滤镜
 */

export class ConvexLensMapGenerator {
  public static injectBarrelMap(size: number = 256): string {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    const imgData = ctx.createImageData(size, size);
    const data = imgData.data;

    // 严谨计算 2D 球面凸透镜法线折射向量 (Barrel Distortion Map)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        // 归一化坐标 (-1.0 ~ +1.0)
        const nx = (x / (size - 1)) * 2 - 1;
        const ny = (y / (size - 1)) * 2 - 1;
        const r2 = nx * nx + ny * ny;
        const r = Math.sqrt(r2);

        let dx = 0;
        let dy = 0;

        if (r < 1.0) {
          // 球面内部非线性指数膨胀，使透镜边缘文字产生自然曲率
          const edgeFactor = Math.pow(r, 2.2);
          dx = -nx * edgeFactor * 0.92;
          dy = -ny * edgeFactor * 0.78;
        } else {
          // 透镜外缘柔和衰减，杜绝突兀硬切边
          const falloff = Math.max(0, 1 - (r - 1) * 2.5);
          dx = -nx * falloff * 0.65;
          dy = -ny * falloff * 0.55;
        }

        const idx = (y * size + x) * 4;
        // R 通道映射 X 轴水平位移 (128 为中心零偏移)
        data[idx]     = Math.max(0, Math.min(255, Math.round(128 + dx * 127)));
        // G 通道映射 Y 轴垂直位移
        data[idx + 1] = Math.max(0, Math.min(255, Math.round(128 + dy * 127)));
        // B 通道保持 128 (无彩虹色散边缘)
        data[idx + 2] = 128;
        // A 通道不透明
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    const dataUrl = canvas.toDataURL('image/png');

    // 动态注入 SVG #convex-lens-filter 滤镜中的 <feImage>
    const feImg = document.getElementById('convex-lens-map-img');
    if (feImg) {
      feImg.setAttribute('href', dataUrl);
      feImg.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', dataUrl);
    }

    return dataUrl;
  }
}
