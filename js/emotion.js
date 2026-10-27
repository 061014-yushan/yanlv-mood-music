/**
 * ============================================================================
 *  情绪识别模块 (Emotion Engine)
 *  ---------------------------------------------------------------------------
 *  本地推理，三层算法流水线，全部在浏览器端完成（无需上传图片到服务器）：
 *
 *    L1  人脸检测  TinyFaceDetector (SSD, MobileNetV1 深度可分离卷积骨架)
 *        · 输入：归一化后的 RGB 张量
 *        · 输出：人脸 bounding box + 5 个关键点 (左右眼、鼻尖、左右嘴角)
 *        · 置信度阈值 0.4，NMS IoU 0.5
 *
 *    L2  关键点对齐  FaceLandmark68Net (68 点回归)
 *        · 作用：用上一步的 5 点做人脸仿射对齐，消除头部姿态/尺度差异，
 *          使表情特征落在统一的归一化坐标系里
 *
 *    L3  表情分类  FaceExpressionNet
 *        · 骨架：类 MobileNet 的深度可分离卷积 + 全局平均池化 + FC
 *        · 输出：7 类 softmax 概率
 *          neutral / happy / sad / angry / fearful / disgusted / surprised
 *
 *  ---- 上层的"情绪融合算法"（本项目自己实现，不只是拿最大概率）----
 *
 *  步骤 1  时序平滑：指数移动平均 EMA
 *          p_t = α · p̂_t + (1-α) · p_{t-1}     α = 0.35
 *          作用：抑制单帧抖动，避免表情一闪就切歌
 *
 *  步骤 2  Valence-Arousal 二维映射（情感计算经典模型）
 *          V = Σ p_i · v_i     效价：愉悦(+1) ↔ 不悦(-1)
 *          A = Σ p_i · a_i     唤醒：激动(+1) ↔ 平静(-1)
 *          7 类情绪在 VA 空间中的锚点见 EMOTION_VA
 *
 *  步骤 3  主导情绪判定
 *          主导 = argmax(p)，但需满足：
 *            · 置信度 p_max ≥ 0.45
 *            · 与次高类别间隔 p_max - p_second ≥ 0.12
 *          否则判为 uncertain（不确定），不强行推荐，提示用户重试
 *
 *  步骤 4  情绪强度
 *          intensity = p_max × (1 + 0.5·|V|)
 *          高唤醒的愤怒/惊讶比中性的"平静"更"强烈"
 *
 *  步骤 5  多脸融合
 *          多张脸时按人脸框面积加权：
 *          p̄ = Σ (a_j · p_j) / Σ a_j       a_j = 人脸框面积
 *          画面中占主导的人对结果影响更大
 * ============================================================================
 */

/* ---------- 7 类基础情绪在 Valence-Arousal 空间的锚点 ---------- */
const EMOTION_VA = {
  neutral:    { v:  0.00, a: -0.25 },  // 中性：平和、低唤醒
  happy:      { v:  0.85, a:  0.60 },  // 开心：正向、高唤醒
  sad:        { v: -0.70, a: -0.55 },  // 悲伤：负向、低唤醒
  angry:      { v: -0.80, a:  0.75 },  // 愤怒：负向、高唤醒
  fearful:    { v: -0.65, a:  0.55 },  // 恐惧：负向、高唤醒
  disgusted:  { v: -0.75, a:  0.20 },  // 厌恶：负向、中唤醒
  surprised:  { v:  0.15, a:  0.80 }   // 惊讶：中性偏正、超高唤醒
};

/* ---------- 中文标签与配色 ---------- */
const EMOTION_META = {
  neutral:   { zh: '平静', en: 'Neutral',   color: '#8fa6c4', emoji: '😐', desc: '情绪平稳，呼吸均匀，肌肉放松' },
  happy:     { zh: '开心', en: 'Happy',     color: '#ffc93c', emoji: '😄', desc: '嘴角上扬、眼角收缩，状态积极' },
  sad:       { zh: '悲伤', en: 'Sad',       color: '#5b8dd6', emoji: '😢', desc: '眉头内侧上提、嘴角下垂' },
  angry:     { zh: '愤怒', en: 'Angry',     color: '#ff5c5c', emoji: '😠', desc: '眉头下压、眼睑紧张、嘴唇紧抿' },
  fearful:   { zh: '恐惧', en: 'Fearful',   color: '#a86bff', emoji: '😨', desc: '双眼睁大、眉毛上抬并聚拢' },
  disgusted: { zh: '厌恶', en: 'Disgusted', color: '#5fbf7f', emoji: '🤢', desc: '鼻梁皱起、上唇上提' },
  surprised: { zh: '惊讶', en: 'Surprised', color: '#ff9f43', emoji: '😲', desc: '眉毛高抬、嘴巴张开' },
  uncertain: { zh: '待确认', en: 'Uncertain', color: '#7a8699', emoji: '🤔', desc: '表情特征不够显著，请调整光线或正对镜头' }
};

/* ---------- 算法超参数 ---------- */
const EMOTION_CFG = {
  EMA_ALPHA: 0.35,          // 时序平滑系数
  MIN_CONFIDENCE: 0.45,     // 主导情绪最低置信度
  MIN_MARGIN: 0.12,         // 与次高类的最小间隔
  DETECT_INPUT_SIZE: 416,   // 检测网络输入边长（越大越准、越慢）
  SCORE_THRESHOLD: 0.4,     // 人脸检测置信度阈值
  MAX_FACES: 5              // 最多同时分析的人脸数
};

class EmotionEngine {
  constructor() {
    this.ready = false;
    this.loadedModels = [];
    this.prevProbs = null;       // EMA 历史
    this.frameCount = 0;
    this.lastBoxes = [];
  }

  /** 加载三阶段神经网络权重（本地 models/ 目录） */
  async loadModels(baseUrl = './models', onProgress = () => {}) {
    const uri = baseUrl.replace(/\/$/, '');
    // 后端选择顺序：WebGL（GPU 加速）→ WASM（本机 SIMD）→ CPU
    // 逐级探测，记录实际使用的后端；没有 WebGL 的无头/老设备也能跑
    await this.initBackend();
    onProgress(`计算后端：${String(this.backend).toUpperCase()}`, 8);

    onProgress('正在加载人脸检测网络 TinyFaceDetector…', 15);
    await faceapi.nets.tinyFaceDetector.loadFromUri(uri);
    this.loadedModels.push('tinyFaceDetector');

    onProgress('正在加载 68 点关键点回归网络…', 45);
    await faceapi.nets.faceLandmark68Net.loadFromUri(uri);
    this.loadedModels.push('faceLandmark68Net');

    onProgress('正在加载表情分类网络…', 75);
    await faceapi.nets.faceExpressionNet.loadFromUri(uri);
    this.loadedModels.push('faceExpressionNet');

    onProgress('模型加载完成', 100);
    this.ready = true;
    return this.loadedModels;
  }

  /**
   * 计算后端选择：WebGL(GPU) → WASM(SIMD) → CPU
   * 逐个探测：先看 canvas 能否真正拿到 WebGL 上下文，再 setBackend，
   * 最后用一次张量运算做"冒烟测试"，失败就降级，保证任何设备都能跑。
   */
  async initBackend() {
    // 注意：这个打包版本只暴露 faceapi，tf 命名空间挂在 faceapi.tf 上
    const TF = (window.tf) || (window.faceapi && window.faceapi.tf) || null;
    this.tf = TF;
    if (!TF || !TF.setBackend) { this.backend = 'unknown'; return; }

    // 1) 探测 WebGL 与 WASM-SIMD 能力
    const canWebGL = (() => {
      try {
        const cv = document.createElement('canvas');
        const gl = cv.getContext('webgl2') || cv.getContext('webgl');
        return !!(gl && gl.getParameter(gl.VERSION));
      } catch (e) { return false; }
    })();
    let simd = false;
    try {
      simd = WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
        10, 10, 1, 8, 0, 65, 0, 253, 15, 26, 11
      ]));
    } catch (e) { simd = false; }
    try { TF.env() && TF.env().set && TF.env().set('WASM_HAS_SIMD_SUPPORT', simd); } catch (e) {}

    // 2) 依次尝试后端，每个都做一次真实张量冒烟测试
    const order = canWebGL ? ['webgl', 'wasm', 'cpu'] : ['wasm', 'cpu'];
    for (const b of order) {
      try {
        const ok = await TF.setBackend(b);
        await TF.ready();
        if (!ok || TF.getBackend() !== b) continue;
        const t = TF.tensor1d([1, 2, 3]);
        const r = t.add(1).dataSync();
        t.dispose();
        if (r && r.length === 3) { this.backend = b; return; }
      } catch (e) {
        // wasm 初始化失败时移除该后端，避免它继续占据最高优先级
        try { if (typeof TF.removeBackend === 'function') TF.removeBackend(b); } catch (_) {}
      }
    }
    try { await TF.setBackend('cpu'); await TF.ready(); } catch (e) {}
    this.backend = 'cpu';
  }

  /** 构造检测参数 */
  options() {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: EMOTION_CFG.DETECT_INPUT_SIZE,
      scoreThreshold: EMOTION_CFG.SCORE_THRESHOLD
    });
  }

  /**
   * 对任意输入源（video / img / canvas）做一次完整推理
   * @returns {Promise<{faces:Array, boxes:Array}>}
   */
  async detect(source) {
    if (!this.ready) throw new Error('模型尚未加载完成');
    const results = await faceapi
      .detectAllFaces(source, this.options())
      .withFaceLandmarks()
      .withFaceExpressions();

    this.frameCount++;
    return {
      faces: results.map(r => ({
        box: r.detection.box,
        score: r.detection.score,
        landmarks: r.landmarks.positions,
        expressions: r.expressions   // {neutral, happy, sad, ...}
      })),
      boxes: results.map(r => r.detection.box)
    };
  }

  /** 把 face-api 的每张脸结果转换为原始概率向量 */
  _toVector(expressions) {
    const out = {};
    for (const k of Object.keys(EMOTION_VA)) out[k] = expressions[k] || 0;
    return out;
  }

  /** 多脸加权融合：按人脸框面积加权 */
  _fuseMultiFace(faces) {
    if (!faces.length) return null;
    let wsum = 0;
    const acc = {};
    for (const k of Object.keys(EMOTION_VA)) acc[k] = 0;
    for (const f of faces) {
      const w = Math.max(1, f.box.width * f.box.height);
      const v = this._toVector(f.expressions);
      for (const k of Object.keys(acc)) acc[k] += v[k] * w;
      wsum += w;
    }
    for (const k of Object.keys(acc)) acc[k] /= wsum;
    return acc;
  }

  /** EMA 时序平滑 */
  _smooth(raw, reset = false) {
    if (!raw) return null;
    if (reset || !this.prevProbs) {
      this.prevProbs = { ...raw };
      return { ...raw };
    }
    const a = EMOTION_CFG.EMA_ALPHA;
    const out = {};
    for (const k of Object.keys(raw)) {
      out[k] = a * raw[k] + (1 - a) * (this.prevProbs[k] || 0);
    }
    // 重新归一化
    const s = Object.values(out).reduce((x, y) => x + y, 0) || 1;
    for (const k of Object.keys(out)) out[k] /= s;
    this.prevProbs = { ...out };
    return out;
  }

  /**
   * 核心：从检测结果得到最终情绪结论
   * @param {Array} faces detect() 返回的人脸数组
   * @param {boolean} resetSmooth 是否重置时序平滑（切图/换人时用）
   */
  infer(faces, resetSmooth = false) {
    if (!faces || !faces.length) {
      return { ok: false, reason: 'no-face', label: 'uncertain', ...EMOTION_META.uncertain };
    }

    const fused = this._fuseMultiFace(faces);
    const probs = this._smooth(fused, resetSmooth);

    // 排序
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const [topLabel, topP] = sorted[0];
    const secondP = sorted[1][1];
    const margin = topP - secondP;

    // Valence / Arousal
    let V = 0, A = 0;
    for (const [k, p] of Object.entries(probs)) {
      V += p * EMOTION_VA[k].v;
      A += p * EMOTION_VA[k].a;
    }

    // 判定是否可信
    const confident = topP >= EMOTION_CFG.MIN_CONFIDENCE && margin >= EMOTION_CFG.MIN_MARGIN;
    const label = confident ? topLabel : 'uncertain';

    // 强度
    const intensity = Math.min(1, topP * (1 + 0.5 * Math.abs(V)));

    return {
      ok: true,
      label,
      rawLabel: topLabel,
      confidence: topP,
      margin,
      confident,
      probs,
      sorted,
      valence: V,
      arousal: A,
      intensity,
      faceCount: faces.length,
      ...EMOTION_META[label]
    };
  }

  resetSmoothing() {
    this.prevProbs = null;
  }
}

window.EmotionEngine = EmotionEngine;
window.EMOTION_VA = EMOTION_VA;
window.EMOTION_META = EMOTION_META;
window.EMOTION_CFG = EMOTION_CFG;
