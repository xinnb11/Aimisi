/**
 * SamplingStrategy.ts - GPT-SoVITS 2 自回归语义 Token 高精度概率采样器
 * 
 * 严格遵循真实大模型生成解码规范：
 * 严禁退化为简单贪婪 argmax！通过 Temperature 缩放、上下文惩罚 (Repetition Penalty)、
 * Top-K 截断、Top-P (Nucleus 核采样) 与多项式随机分布抽样，还原真实灵动的情绪韵律。
 */

export interface SamplingOptions {
  temperature?: number;         // 采样温度 (默认 0.85，值越低越保守，值越高越丰富)
  topK?: number;                // Top-K 截断 (默认 25，保留概率最高的前 K 个候选)
  topP?: number;                // Top-P 核采样截断 (默认 0.90，累积概率阈值)
  repetitionPenalty?: number;   // 重复惩罚系数 (默认 1.35，抑制自回归循环死循环)
  repetitionWindow?: number;    // 重复惩罚窗口 (默认 64，关注最近生成的 token)
  stopToken?: number;           // 终止 EOS 标记 (默认 1024)
}

export class SamplingStrategy {
  /**
   * 执行完整的采样流水线 (禁止 argmax 贪婪选择，采用真实多项式核分布采样)
   */
  public static sample(
    rawLogits: Float32Array,
    generatedTokens: number[],
    options: SamplingOptions = {}
  ): number {
    const temperature = options.temperature ?? 0.85;
    const topK = options.topK ?? 25;
    const topP = options.topP ?? 0.90;
    const repetitionPenalty = options.repetitionPenalty ?? 1.35;
    const repetitionWindow = options.repetitionWindow ?? 64;

    const vocabSize = rawLogits.length;
    // 克隆 logits 避免污染底层推理张量
    const logits = new Float32Array(rawLogits);

    // 1. 应用重复惩罚 (Repetition Penalty)
    if (repetitionPenalty !== 1.0 && generatedTokens.length > 0) {
      const startIdx = Math.max(0, generatedTokens.length - repetitionWindow);
      const recentTokens = new Set(generatedTokens.slice(startIdx));
      for (const tok of recentTokens) {
        if (tok >= 0 && tok < vocabSize) {
          if (logits[tok] > 0) {
            logits[tok] /= repetitionPenalty;
          } else {
            logits[tok] *= repetitionPenalty;
          }
        }
      }
    }

    // 2. 应用温度缩放 (Temperature Scaling)
    if (temperature > 0 && temperature !== 1.0) {
      const invTemp = 1.0 / Math.max(temperature, 1e-4);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] *= invTemp;
      }
    }

    // 3. 构建索引数组用于 Top-K 与 Top-P 排序筛选
    const candidates: Array<{ idx: number; logit: number }> = [];
    for (let i = 0; i < vocabSize; i++) {
      candidates.push({ idx: i, logit: logits[i] });
    }

    // 按 logit 从大到小降序排列
    candidates.sort((a, b) => b.logit - a.logit);

    // 4. 应用 Top-K 截断
    let effectiveCandidates = candidates;
    if (topK > 0 && topK < candidates.length) {
      effectiveCandidates = candidates.slice(0, topK);
    }

    // 5. 应用数值稳定的 Softmax 计算概率分布
    let maxLogit = -Infinity;
    for (const c of effectiveCandidates) {
      if (c.logit > maxLogit) maxLogit = c.logit;
    }

    let sumExp = 0.0;
    const probs: Array<{ idx: number; prob: number }> = [];
    for (const c of effectiveCandidates) {
      const expVal = Math.exp(c.logit - maxLogit);
      probs.push({ idx: c.idx, prob: expVal });
      sumExp += expVal;
    }

    if (sumExp > 0) {
      for (const p of probs) {
        p.prob /= sumExp;
      }
    }

    // 6. 应用 Top-P (Nucleus) 核采样截断
    let nucleusCandidates = probs;
    if (topP > 0.0 && topP < 1.0) {
      let cumProb = 0.0;
      const filtered: Array<{ idx: number; prob: number }> = [];
      for (const p of probs) {
        filtered.push(p);
        cumProb += p.prob;
        if (cumProb >= topP) break;
      }
      nucleusCandidates = filtered;

      // 重新归一化
      let nucleusSum = 0.0;
      for (const p of nucleusCandidates) {
        nucleusSum += p.prob;
      }
      if (nucleusSum > 0) {
        for (const p of nucleusCandidates) {
          p.prob /= nucleusSum;
        }
      }
    }

    // 7. 多项式分布随机抽样 (Multinomial / Categorical Sampling)
    // 绝不使用 argmax，通过轮盘赌随机采样生成富有生命力的二次元语气波动
    const rand = Math.random();
    let cumulative = 0.0;
    for (const item of nucleusCandidates) {
      cumulative += item.prob;
      if (rand <= cumulative) {
        return item.idx;
      }
    }

    // 兜底返回核中候选首项
    return nucleusCandidates.length > 0 ? nucleusCandidates[0].idx : 0;
  }
}
