/**
 * SpringConvexLensController.ts - iOS 26 水晶凸透镜与 1ms 子步长双耦合阻尼弹簧物理控制器
 * 
 * 物理特性：
 * 1. 1ms 固定子步长微分求解器 (欧拉-半隐式阻尼弹簧)：无论 60Hz 还是 120Hz 刷新率，物理行为绝对一致
 * 2. 动量继承与投射投影 (Velocity Projection)：长按数秒或极速飞拂均能平滑滑入目标卡槽
 * 3. 双态光学表现：静止为磨砂灰胶囊，按压/拖动激活态膨胀为纯净白水晶透镜 (1.38x 凸透镜放大 + 多层微光晕染，无彩虹边)
 */

import { ConvexLensMapGenerator } from './ConvexLensMapGenerator';

export interface LensHostContext {
  [key: string]: unknown;
}

export class SpringConvexLensController {
  public static initAll(ctx: LensHostContext): void {
    ConvexLensMapGenerator.injectBarrelMap();

    // 挂载所有多段式凸透镜滑动器
    document.querySelectorAll('[data-agent-component="ConvexLiquidLensSlider"]').forEach(track => {
      SpringConvexLensController.mountSlider(track as HTMLElement, ctx);
    });

    // 挂载所有液态凸透镜发光胶囊开关
    document.querySelectorAll('[data-agent-component="ConvexToggleSwitch"], [data-agent-component="ConvexLiquidToggle"]').forEach(toggle => {
      SpringConvexLensController.mountToggle(toggle as HTMLElement, ctx);
    });
  }

  public static mountSlider(track: HTMLElement, ctx: LensHostContext): void {
    const segments = parseInt(track.getAttribute('data-segments') || '2', 10);
    const isHorizontal = track.getAttribute('data-horizontal') === 'true';
    const callbackName = track.getAttribute('data-on-change');
    const baseItems = Array.from(track.querySelectorAll('.lens-base-item')) as HTMLElement[];

    // 1. 构建轨道内部动态追光层 (.track-follow-glow)
    const trackGlowClip = document.createElement('div');
    trackGlowClip.className = 'track-glow-clip';
    const trackFollowGlow = document.createElement('div');
    trackFollowGlow.className = 'track-follow-glow';
    trackGlowClip.appendChild(trackFollowGlow);
    track.insertBefore(trackGlowClip, track.firstChild);

    // 2. 构建纯净白水晶凸透镜发光滑块 (.convex-lens-thumb)
    const thumb = document.createElement('div');
    thumb.className = 'convex-lens-thumb liquid-lens-thumb';
    thumb.style.width = `calc((100% - 10px) / ${segments})`;

    const ambientGlow = document.createElement('div');
    ambientGlow.className = 'lens-ambient-glow';

    const crystalSurface = document.createElement('div');
    crystalSurface.className = 'lens-crystal-surface';

    const specularArc = document.createElement('div');
    specularArc.className = 'lens-specular-arc';

    const innerViewport = document.createElement('div');
    innerViewport.className = 'lens-inner-viewport';

    const innerStrip = document.createElement('div');
    innerStrip.className = 'lens-inner-track lens-inner-strip';
    innerStrip.style.gridTemplateColumns = `repeat(${segments}, minmax(0, 1fr))`;

    const magnifiedItems = baseItems.map(item => {
      const clone = document.createElement('div');
      clone.className = `lens-magnified-item ${isHorizontal ? 'is-horizontal' : ''}`;
      clone.innerHTML = item.innerHTML;
      innerStrip.appendChild(clone);
      return clone;
    });

    innerViewport.appendChild(innerStrip);

    const touchDot = document.createElement('div');
    touchDot.className = 'lens-touch-dot';

    thumb.appendChild(ambientGlow);
    thumb.appendChild(crystalSurface);
    thumb.appendChild(innerViewport);
    thumb.appendChild(specularArc);
    thumb.appendChild(touchDot);
    track.insertBefore(thumb, trackGlowClip.nextSibling);

    // 3. 几何与弹簧动力学状态
    const pad = 5;
    let cachedRect = { left: 0, top: 0, width: 0, height: 0 };
    let cachedUsableWidth = 100;
    let cachedThumbWidth = 50;
    let cachedMaxMag = 1.38;
    let cachedBulge = 1.30;
    let cachedSpecular = 0.95;

    let activeIndex = parseInt(track.getAttribute('data-active-index') || '0', 10);
    let x = 0;
    let targetX = 0;
    let vx = 0;

    let p = 0; // 0 = 哑光灰胶囊, 1 = 纯透明水晶凸透镜
    let targetP = 0;
    let vp = 0;

    let dotX = 0;
    let dotY = 0;
    let targetDotX = 0;
    let targetDotY = 0;

    let isPointerDown = false;
    let downTime = 0;
    let startPointerX = 0;
    let lastPointerX = 0;
    let lastPointerTime = 0;
    let fingerVx = 0;
    let maxDragDistance = 0;

    let rafId: number | null = null;
    let lastFrameTime = 0;
    let tapReleaseHoldTimer: ReturnType<typeof setTimeout> | null = null;

    const refreshMetricsCache = () => {
      const r = track.getBoundingClientRect();
      if (r.width <= 0) return;
      cachedRect = { left: r.left, top: r.top, width: r.width, height: r.height };
      cachedUsableWidth = Math.max(1, r.width - pad * 2);
      cachedThumbWidth = cachedUsableWidth / segments;
      cachedMaxMag = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lens-mag')) || 1.38;
      cachedBulge = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lens-bulge')) || 1.30;
      cachedSpecular = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lens-specular')) || 0.95;
      innerStrip.style.width = `${cachedUsableWidth}px`;
    };

    const renderFrame = () => {
      const clampedP = Math.max(0, Math.min(1, p));
      const smoothP = clampedP * clampedP * (3 - 2 * clampedP);
      const visualP = smoothP * 0.42 + clampedP * 0.58;

      const speedStretch = Math.min(0.085, Math.abs(vx) * 0.00011);
      const landingCushion = Math.max(-0.042, Math.min(0.045, -vp * 0.011));
      const bulgeDelta = cachedBulge - 1.0;
      const scaleX = 1 + (bulgeDelta * 0.60) * visualP + speedStretch + landingCushion;
      const scaleY = 1 + bulgeDelta * visualP - speedStretch * 0.42 - landingCushion * 0.65;

      thumb.style.transform = `translate3d(${x.toFixed(2)}px, 0, 0) scale3d(${scaleX.toFixed(4)}, ${scaleY.toFixed(4)}, 1)`;
      innerStrip.style.transform = `translate3d(${(-x).toFixed(2)}px, 0, 0)`;

      const centerX = pad + x + cachedThumbWidth * 0.5;
      const glowScale = 0.85 + 0.55 * visualP;
      trackFollowGlow.style.transform = `translate3d(${centerX.toFixed(2)}px, -50%, 0) translate3d(-50%, 0, 0) scale3d(${glowScale.toFixed(3)}, ${glowScale.toFixed(3)}, 1)`;
      trackFollowGlow.style.opacity = (0.32 + 0.64 * visualP).toFixed(3);

      ambientGlow.style.opacity = (0.22 + 0.78 * visualP).toFixed(3);
      crystalSurface.style.opacity = visualP.toFixed(3);
      specularArc.style.opacity = (visualP * cachedSpecular).toFixed(3);
      innerViewport.style.opacity = Math.min(1, visualP * 1.15).toFixed(3);

      const currentItemMag = 1 + (cachedMaxMag - 1) * Math.max(0, visualP);
      const glowRadius = (8 * visualP).toFixed(2);
      const whiteRadius = (2.5 * visualP).toFixed(2);
      const filterStr = visualP > 0.015
        ? `drop-shadow(0 0 ${glowRadius}px var(--accent-glow)) drop-shadow(0 0 ${whiteRadius}px rgba(255,255,255,0.95))`
        : 'none';

      for (let i = 0; i < magnifiedItems.length; i++) {
        magnifiedItems[i].style.transform = `scale3d(${currentItemMag.toFixed(4)}, ${currentItemMag.toFixed(4)}, 1)`;
        magnifiedItems[i].style.filter = filterStr;
      }

      const dotScale = 0.45 + 0.55 * visualP;
      touchDot.style.transform = `translate3d(${dotX.toFixed(1)}px, ${dotY.toFixed(1)}px, 0) translate3d(-50%, -50%, 0) scale3d(${dotScale.toFixed(3)}, ${dotScale.toFixed(3)}, 1)`;
      touchDot.style.opacity = (visualP * 0.92).toFixed(3);
    };

    // 1ms 固定子步长双耦合阻尼弹簧物理求解器
    const tickSpring = (now: number) => {
      const rawDt = (now - lastFrameTime) / 1000;
      lastFrameTime = now;
      const frameDt = Math.max(0.001, Math.min(0.032, rawDt || 0.016));

      const stepSize = 0.001;
      let remaining = frameDt;

      while (remaining > 0.00001) {
        const dt = Math.min(stepSize, remaining);
        remaining -= dt;

        // X 位移阻尼弹簧 (松开后柔和入位)
        const omegaX = isPointerDown ? 28.0 : 18.5;
        const zetaX = isPointerDown ? 0.90 : 0.96;
        const stiffnessX = omegaX * omegaX;
        const dampingX = 2 * zetaX * omegaX;

        const ax = (targetX - x) * stiffnessX - vx * dampingX;
        vx += ax * dt;
        x += vx * dt;

        // 透镜激活度 P 弹簧：长按数秒松开时，确保伴随位移完成平滑尺寸回落
        const distRemaining = Math.abs(targetX - x);
        const distRatio = Math.min(1, distRemaining / Math.max(1, cachedThumbWidth * 0.85));
        const flightBuoyancy = (!isPointerDown && targetP === 0) ? (distRatio * distRatio * 0.66) : 0;
        const effectiveTargetP = isPointerDown ? targetP : Math.max(targetP, flightBuoyancy);

        const isInflating = effectiveTargetP > p;
        const omegaP = isPointerDown ? 23.0 : (isInflating ? 20.0 : 14.2);
        const zetaP = isPointerDown ? 0.86 : 0.98;
        const stiffnessP = omegaP * omegaP;
        const dampingP = 2 * zetaP * omegaP;

        const ap = (effectiveTargetP - p) * stiffnessP - vp * dampingP;
        vp += ap * dt;
        p += vp * dt;

        const dotLerp = Math.min(1, dt * 26);
        dotX += (targetDotX - dotX) * dotLerp;
        dotY += (targetDotY - dotY) * dotLerp;
      }

      renderFrame();

      const isSettled =
        !isPointerDown &&
        Math.abs(targetX - x) < 0.08 &&
        Math.abs(vx) < 0.6 &&
        Math.abs(targetP - p) < 0.002 &&
        Math.abs(vp) < 0.01;

      if (!isSettled) {
        rafId = requestAnimationFrame(tickSpring);
      } else {
        x = targetX;
        vx = 0;
        p = targetP;
        vp = 0;
        renderFrame();
        rafId = null;
      }
    };

    const ensureLoop = () => {
      if (!rafId) {
        lastFrameTime = performance.now();
        rafId = requestAnimationFrame(tickSpring);
      }
    };

    const syncSnapLayout = (idx: number, immediate: boolean = false) => {
      refreshMetricsCache();
      activeIndex = Math.max(0, Math.min(segments - 1, idx));
      track.setAttribute('data-active-index', String(activeIndex));
      baseItems.forEach((item, i) => item.classList.toggle('is-active', i === activeIndex));

      targetX = activeIndex * cachedThumbWidth;
      if (immediate) {
        x = targetX;
        vx = 0;
        dotX = cachedThumbWidth * 0.5;
        dotY = cachedRect.height * 0.5;
        targetDotX = dotX;
        targetDotY = dotY;
        renderFrame();
      } else {
        ensureLoop();
      }
    };

    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => syncSnapLayout(activeIndex, !isPointerDown && !rafId));
      ro.observe(track);
    }
    requestAnimationFrame(() => syncSnapLayout(activeIndex, true));
    window.addEventListener('resize', () => syncSnapLayout(activeIndex, true));

    const updateTargetsFromPointer = (e: PointerEvent, isInitialDown: boolean = false) => {
      const pointerXInTrack = e.clientX - cachedRect.left - pad;
      const rawX = pointerXInTrack - cachedThumbWidth * 0.5;
      const maxX = cachedUsableWidth - cachedThumbWidth;

      if (rawX < 0) {
        targetX = rawX * 0.22;
      } else if (rawX > maxX) {
        targetX = maxX + (rawX - maxX) * 0.22;
      } else {
        targetX = rawX;
      }

      const clampedThumbX = Math.max(0, Math.min(maxX, rawX));
      targetDotX = Math.max(14, Math.min(cachedThumbWidth - 14, pointerXInTrack - clampedThumbX));
      targetDotY = Math.max(12, Math.min(cachedRect.height - 12, e.clientY - cachedRect.top));

      if (isInitialDown) {
        dotX = targetDotX;
        dotY = targetDotY;
      }
    };

    track.addEventListener('pointerdown', (e: PointerEvent) => {
      if (tapReleaseHoldTimer) clearTimeout(tapReleaseHoldTimer);
      refreshMetricsCache();
      isPointerDown = true;
      downTime = performance.now();
      startPointerX = e.clientX;
      lastPointerX = e.clientX;
      lastPointerTime = downTime;
      fingerVx = 0;
      maxDragDistance = 0;

      targetP = 1.0;
      track.setPointerCapture(e.pointerId);
      updateTargetsFromPointer(e, true);
      ensureLoop();
    });

    track.addEventListener('pointermove', (e: PointerEvent) => {
      if (!isPointerDown) return;
      const now = performance.now();
      const dtMove = Math.max(1, now - lastPointerTime) / 1000;
      const dxMove = e.clientX - lastPointerX;

      const instVx = dxMove / dtMove;
      fingerVx = fingerVx * 0.42 + instVx * 0.58;
      lastPointerX = e.clientX;
      lastPointerTime = now;
      maxDragDistance = Math.max(maxDragDistance, Math.abs(e.clientX - startPointerX));

      updateTargetsFromPointer(e, false);
      ensureLoop();
    });

    const finishPointer = (e: PointerEvent) => {
      if (!isPointerDown) return;
      isPointerDown = false;
      const now = performance.now();

      if (now - lastPointerTime > 65) {
        fingerVx *= 0.12;
      }
      const clampedFingerVx = Math.max(-2200, Math.min(2200, fingerVx));

      if (maxDragDistance > 5) {
        vx = vx * 0.35 + clampedFingerVx * 0.65;
      }

      let nextIndex = activeIndex;
      if (maxDragDistance > 6 && cachedThumbWidth > 0) {
        // 轻拂投射
        const projectedX = x + clampedFingerVx * 0.11;
        nextIndex = Math.round(projectedX / cachedThumbWidth);
      } else if (cachedRect.width > 0) {
        // 点击切换
        const relX = Math.max(0, Math.min(cachedRect.width - 1, e.clientX - cachedRect.left));
        nextIndex = Math.floor((relX / cachedRect.width) * segments);
      }
      nextIndex = Math.max(0, Math.min(segments - 1, nextIndex));

      const prevIndex = activeIndex;
      activeIndex = nextIndex;
      track.setAttribute('data-active-index', String(activeIndex));
      baseItems.forEach((item, i) => item.classList.toggle('is-active', i === activeIndex));
      targetX = activeIndex * cachedThumbWidth;

      const pressDuration = now - downTime;
      if (pressDuration < 160 && p < 0.68) {
        targetP = 1.0;
        tapReleaseHoldTimer = setTimeout(() => {
          targetP = 0.0;
          ensureLoop();
        }, 135);
      } else {
        targetP = 0.0;
      }

      ensureLoop();

      if (prevIndex !== activeIndex || callbackName) {
        if (callbackName && typeof ctx[callbackName] === 'function') {
          (ctx[callbackName] as (idx: number) => void)(activeIndex);
        }
      }
    };

    track.addEventListener('pointerup', finishPointer);
    track.addEventListener('pointercancel', finishPointer);
  }

  public static mountToggle(toggle: HTMLElement, ctx: LensHostContext): void {
    const orb = toggle.querySelector('.liquid-lens-toggle-orb') as HTMLElement;
    const callbackName = toggle.getAttribute('data-on-toggle');

    const maxTravel = 26;
    let cachedRect = { left: 0, width: 58 };
    let checked = toggle.getAttribute('data-checked') === 'true';
    let x = checked ? maxTravel : 0;
    let targetX = x;
    let vx = 0;

    let p = 0;
    let targetP = 0;
    let vp = 0;

    let isPointerDown = false;
    let startClientX = 0;
    let lastClientX = 0;
    let lastMoveTime = 0;
    let fingerVx = 0;
    let moved = false;
    let downTime = 0;
    let tapHoldTimer: ReturnType<typeof setTimeout> | null = null;
    let rafId: number | null = null;
    let lastFrameTime = 0;

    const refreshToggleCache = () => {
      const r = toggle.getBoundingClientRect();
      if (r.width > 0) cachedRect = { left: r.left, width: r.width };
    };

    const renderOrb = () => {
      const clampedP = Math.max(0, Math.min(1, p));
      const smoothP = clampedP * clampedP * (3 - 2 * clampedP);
      const visualP = smoothP * 0.45 + clampedP * 0.55;

      const speedStretch = Math.min(0.09, Math.abs(vx) * 0.00018);
      const landingCushion = Math.max(-0.04, Math.min(0.04, -vp * 0.012));
      const scaleX = 1 + 0.30 * visualP + speedStretch + landingCushion;
      const scaleY = 1 + 0.26 * visualP - speedStretch * 0.4 - landingCushion * 0.6;

      if (orb) {
        orb.style.transform = `translate3d(${x.toFixed(2)}px, 0, 0) scale3d(${scaleX.toFixed(4)}, ${scaleY.toFixed(4)}, 1)`;
        orb.style.background = `rgba(255, 255, 255, ${(1 - 0.06 * visualP).toFixed(3)})`;
        orb.style.boxShadow = `
          inset 0 0 ${(10 * visualP).toFixed(1)}px rgba(255, 255, 255, 0.98),
          inset 0 2px 4px rgba(255, 255, 255, 1),
          0 0 ${(18 * visualP).toFixed(1)}px rgba(255, 255, 255, 0.95),
          0 ${(3 + 8 * visualP).toFixed(1)}px ${(10 + 18 * visualP).toFixed(1)}px var(--accent-glow-strong)
        `;
      }
    };

    const tick = (now: number) => {
      const rawDt = (now - lastFrameTime) / 1000;
      lastFrameTime = now;
      let remaining = Math.max(0.001, Math.min(0.032, rawDt || 0.016));
      const stepSize = 0.001;

      while (remaining > 0.00001) {
        const dt = Math.min(stepSize, remaining);
        remaining -= dt;

        const omegaX = isPointerDown ? 28.0 : 19.5;
        const zetaX = isPointerDown ? 0.90 : 0.96;
        const ax = (targetX - x) * (omegaX * omegaX) - vx * (2 * zetaX * omegaX);
        vx += ax * dt;
        x += vx * dt;

        const distRatio = Math.min(1, Math.abs(targetX - x) / maxTravel);
        const flightBuoyancy = (!isPointerDown && targetP === 0) ? (distRatio * distRatio * 0.62) : 0;
        const effectiveTargetP = isPointerDown ? targetP : Math.max(targetP, flightBuoyancy);

        const omegaP = isPointerDown ? 23.0 : 15.0;
        const zetaP = isPointerDown ? 0.86 : 0.98;
        const ap = (effectiveTargetP - p) * (omegaP * omegaP) - vp * (2 * zetaP * omegaP);
        vp += ap * dt;
        p += vp * dt;
      }

      renderOrb();

      const settled =
        !isPointerDown &&
        Math.abs(targetX - x) < 0.06 &&
        Math.abs(vx) < 0.5 &&
        Math.abs(targetP - p) < 0.002 &&
        Math.abs(vp) < 0.01;

      if (!settled) {
        rafId = requestAnimationFrame(tick);
      } else {
        x = targetX;
        vx = 0;
        p = targetP;
        vp = 0;
        renderOrb();
        rafId = null;
      }
    };

    const ensureLoop = () => {
      if (!rafId) {
        lastFrameTime = performance.now();
        rafId = requestAnimationFrame(tick);
      }
    };

    renderOrb();

    toggle.addEventListener('pointerdown', (e: PointerEvent) => {
      if (tapHoldTimer) clearTimeout(tapHoldTimer);
      refreshToggleCache();
      isPointerDown = true;
      moved = false;
      downTime = performance.now();
      startClientX = e.clientX;
      lastClientX = e.clientX;
      lastMoveTime = downTime;
      fingerVx = 0;
      targetP = 1.0;
      toggle.setPointerCapture(e.pointerId);
      ensureLoop();
    });

    toggle.addEventListener('pointermove', (e: PointerEvent) => {
      if (!isPointerDown) return;
      const now = performance.now();
      const dtMove = Math.max(1, now - lastMoveTime) / 1000;
      fingerVx = fingerVx * 0.4 + ((e.clientX - lastClientX) / dtMove) * 0.6;
      lastClientX = e.clientX;
      lastMoveTime = now;

      if (Math.abs(e.clientX - startClientX) > 3) moved = true;
      const rawX = e.clientX - cachedRect.left - 15;
      if (rawX < 0) {
        targetX = rawX * 0.2;
      } else if (rawX > maxTravel) {
        targetX = maxTravel + (rawX - maxTravel) * 0.2;
      } else {
        targetX = rawX;
      }
      ensureLoop();
    });

    const endToggle = (e: PointerEvent) => {
      if (!isPointerDown) return;
      isPointerDown = false;
      const now = performance.now();
      if (now - lastMoveTime > 65) fingerVx *= 0.12;

      if (!moved) {
        checked = !checked;
      } else {
        vx = vx * 0.35 + Math.max(-1200, Math.min(1200, fingerVx)) * 0.65;
        const projectedX = x + fingerVx * 0.08;
        checked = projectedX >= maxTravel * 0.5;
      }

      toggle.setAttribute('data-checked', checked ? 'true' : 'false');
      targetX = checked ? maxTravel : 0;

      const pressDur = now - downTime;
      if (pressDur < 160 && p < 0.68) {
        targetP = 1.0;
        tapHoldTimer = setTimeout(() => {
          targetP = 0.0;
          ensureLoop();
        }, 135);
      } else {
        targetP = 0.0;
      }

      ensureLoop();

      if (callbackName && typeof ctx[callbackName] === 'function') {
        (ctx[callbackName] as (val: boolean) => void)(checked);
      }
    };

    toggle.addEventListener('pointerup', endToggle);
    toggle.addEventListener('pointercancel', endToggle);
  }
}
