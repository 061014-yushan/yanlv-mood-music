/**
 * ============================================================================
 *  歌曲库 & 推荐引擎
 *  ---------------------------------------------------------------------------
 *  一、曲库组成（双通道）
 *     · 本地曲目   audio/local-*.mp3  （14 首离线可播，随项目分发）
 *     · 在线曲目   iTunes Search API 实时检索 + 30s 试听直链
 *                   https://itunes.apple.com/search?term=..&media=music&country=us
 *                   走 JSONP 绕过 CORS，无需 API Key
 *
 *  二、每首歌携带"音频特征向量"
 *       v : valence  效价 -1..1      （越正越愉悦）
 *       a : arousal  唤醒度 -1..1    （越高越激烈）
 *       bpm / 调式 / 流派 / 语言 / 能量
 *      标签 tags：用于语义匹配（如 "治愈"、"解压"、"共情"）
 *
 *  三、推荐算法（多目标加权打分 + MMR 去重）
 *      对候选歌曲 s 计算：
 *
 *      Score(s) = w1·Sim_VA(s)     效价-唤醒二维欧氏距离相似度
 *               + w2·Match_Tag(s)  情绪标签命中率（Jaccard）
 *               + w3·Sim_BPM(s)    节奏接近度（心情唤醒度映射到目标 BPM 区间）
 *               + w4·Novelty(s)    新鲜度（避免重复推同一批）
 *               + w5·Diversity(s)  MMR 多样性惩罚（与已选歌曲的相似度）
 *               + w6·Quality(s)    曲库质量/热度先验
 *
 *      Sim_VA  = 1 - (Δv² + Δa²)^0.5 / (2√2)      归一化到 0..1
 *      Sim_BPM = exp( -((bpm - bpm*)²) / (2σ²) )   σ = 22
 *      Match_Tag = |T_emotion ∩ T_song| / |T_emotion ∪ T_song|
 *      MMR     = max_{j∈Selected} cos(feat_s, feat_j)   （特征向量 6 维）
 *
 *  四、推荐策略（strategy）
 *      · 共情 empathy      ：推同情绪的歌，让用户"被理解"  —— 悲伤→悲伤
 *      · 调节 regulate     ：推能"修正情绪"的歌             —— 悲伤→平静/明亮
 *      · 反差 contrast     ：推完全相反的歌                 —— 平淡→高能
 *      · 混合 hybrid       ：70% 共情 + 30% 调节（默认）
 *      策略通过"目标 VA 锚点"驱动：target = α·VA(当前) + (1-α)·VA(目标)
 * ============================================================================
 */

/* ---------------- 本地曲库（14 首，全部由项目内合成器生成，可离线播放） ----------------
   feat 为人工标注 + 合成参数推导出的音频特征 */
const LOCAL_TRACKS = [
  { id:'L01', title:'Sunlit Steps',    artist:'颜律工坊', file:'audio/local-happy-1.mp3',
    cover:'#ffd166', v: 0.82, a: 0.62, bpm:126, key:'C 大调', genre:'流行', lang:'器乐', energy:0.80,
    dur:'0:16', tags:['开心','明亮','阳光','跃动','元气','治愈'],
    mood:'happy', desc:'明亮的切分钢琴与轻快鼓组，像午后阳光洒在台阶上。' },
  { id:'L02', title:'Neon Bounce',     artist:'颜律工坊', file:'audio/local-happy-2.mp3',
    cover:'#ffb03a', v: 0.75, a: 0.70, bpm:118, key:'F 大调', genre:'电子流行', lang:'器乐', energy:0.85,
    dur:'0:17', tags:['开心','跳跃','霓虹','活力','派对'],
    mood:'happy', desc:'合成器铺底加上行旋律，适合把好心情放大。' },
  { id:'L03', title:'Quiet Window',    artist:'颜律工坊', file:'audio/local-calm-1.mp3',
    cover:'#9ad0c2', v: 0.15, a:-0.55, bpm: 68, key:'C 大调', genre:'钢琴', lang:'器乐', energy:0.20,
    dur:'0:29', tags:['平静','舒缓','钢琴','安静','治愈','助眠'],
    mood:'neutral', desc:'留白很多的舒缓钢琴，呼吸会不自觉跟着慢下来。' },
  { id:'L04', title:'Drifting Light',  artist:'颜律工坊', file:'audio/local-calm-2.mp3',
    cover:'#8ecae6', v: 0.10, a:-0.68, bpm: 56, key:'D 小调', genre:'氛围', lang:'器乐', energy:0.14,
    dur:'0:27', tags:['平静','氛围','冥想','漂浮','放松'],
    mood:'neutral', desc:'长音铺底与零星琴音，适合放空和冥想。' },
  { id:'L05', title:'Rain on Glass',   artist:'颜律工坊', file:'audio/local-sad-1.mp3',
    cover:'#5b8dd6', v:-0.62, a:-0.50, bpm: 64, key:'A 小调', genre:'弦乐', lang:'器乐', energy:0.26,
    dur:'0:31', tags:['悲伤','弦乐','雨天','共情','孤独'],
    mood:'sad', desc:'小调弦乐缓慢推进，像雨水沿着玻璃往下流。' },
  { id:'L06', title:'Faded Music Box', artist:'颜律工坊', file:'audio/local-sad-2.mp3',
    cover:'#7fa6d9', v:-0.58, a:-0.42, bpm: 76, key:'D 小调', genre:'八音盒', lang:'器乐', energy:0.22,
    dur:'0:26', tags:['悲伤','怀旧','八音盒','回忆','共情'],
    mood:'sad', desc:'褪色的八音盒旋律，适合安静地想一想旧事。' },
  { id:'L07', title:'Red Riot',        artist:'颜律工坊', file:'audio/local-angry-1.mp3',
    cover:'#ff5c5c', v:-0.55, a: 0.82, bpm:150, key:'E 小调', genre:'摇滚', lang:'器乐', energy:0.96,
    dur:'0:14', tags:['愤怒','释放','摇滚','解压','强节奏'],
    mood:'angry', desc:'强力和弦与重拍，把憋着的情绪安全地砸出来。' },
  { id:'L08', title:'Midnight Engine', artist:'颜律工坊', file:'audio/local-angry-2.mp3',
    cover:'#e04a4a', v:-0.48, a: 0.75, bpm:140, key:'D 小调', genre:'暗黑电子', lang:'器乐', energy:0.90,
    dur:'0:15', tags:['愤怒','宣泄','暗色','推进','解压'],
    mood:'angry', desc:'暗色合成器与持续低频推进，情绪有出口。' },
  { id:'L09', title:'Hollow Corridor', artist:'颜律工坊', file:'audio/local-fear-1.mp3',
    cover:'#a86bff', v:-0.50, a: 0.55, bpm: 90, key:'小二度', genre:'悬疑', lang:'器乐', energy:0.55,
    dur:'0:22', tags:['不安','悬疑','紧张','提心','小二度'],
    mood:'fearful', desc:'半音纠缠的动机制造不安感，紧张情绪的写照。' },
  { id:'L10', title:'Heartbeat Room',  artist:'颜律工坊', file:'audio/local-fear-2.mp3',
    cover:'#8e5cf0', v:-0.55, a: 0.60, bpm:100, key:'C 小调', genre:'氛围电子', lang:'器乐', energy:0.60,
    dur:'0:20', tags:['不安','心跳','压迫','焦虑'],
    mood:'fearful', desc:'心跳般的双踩底鼓，适合需要被理解的紧张时刻。' },
  { id:'L11', title:'Upside Ascent',   artist:'颜律工坊', file:'audio/local-surprised-1.mp3',
    cover:'#ff9f43', v: 0.30, a: 0.78, bpm:132, key:'C 大调', genre:'电子', lang:'器乐', energy:0.88,
    dur:'0:16', tags:['惊讶','上扬','琶音','惊喜','好奇'],
    mood:'surprised', desc:'一路向上的琶音，把"哇"的那一下延长成音乐。' },
  { id:'L12', title:'Static Spark',    artist:'颜律工坊', file:'audio/local-surprised-2.mp3',
    cover:'#ffa94d', v: 0.20, a: 0.85, bpm:144, key:'混合利底亚', genre:'电子放克', lang:'器乐', energy:0.92,
    dur:'0:14', tags:['惊讶','跳跃','断奏','俏皮','好奇'],
    mood:'surprised', desc:'断奏和弦像一串火花，适合突然来了灵感。' },
  { id:'L13', title:'Coffee Loop',     artist:'颜律工坊', file:'audio/local-neutral-1.mp3',
    cover:'#b8a58a', v: 0.08, a: 0.05, bpm: 84, key:'C 大调', genre:'Lo-Fi', lang:'器乐', energy:0.40,
    dur:'0:24', tags:['平静','Lo-Fi','日常','专注','工作'],
    mood:'neutral', desc:'低饱和的 Lo-Fi 循环，写东西时的背景音。' },
  { id:'L14', title:'Wooden Strings',  artist:'颜律工坊', file:'audio/local-neutral-2.mp3',
    cover:'#c8b39a', v: 0.12, a:-0.10, bpm: 88, key:'G 大调', genre:'民谣器乐', lang:'器乐', energy:0.38,
    dur:'0:23', tags:['平静','木吉他','民谣','温暖','日常'],
    mood:'neutral', desc:'木吉他分解和弦，朴素、踏实、不打扰。' }
];

/* ---------------- 在线检索用的情绪关键词库（英文更易命中 iTunes） ---------------- */
const ONLINE_QUERY_BANK = {
  happy:     [{q:'happy upbeat pop', tag:'开心'}, {q:'feel good sunshine music', tag:'开心'},
              {q:'dance party hits', tag:'跃动'}, {q:'pharrell happy', tag:'元气'}],
  sad:       [{q:'sad piano ballads', tag:'悲伤'}, {q:'melancholy acoustic', tag:'共情'},
              {q:'lonely night songs', tag:'孤独'}, {q:'sad violin', tag:'悲伤'}],
  angry:     [{q:'rock motivation', tag:'释放'}, {q:'aggressive rock', tag:'愤怒'},
              {q:'metal workout', tag:'宣泄'}, {q:'hard rock riffs', tag:'解压'}],
  fearful:   [{q:'calm relaxing piano', tag:'安抚'}, {q:'ambient meditation', tag:'冥想'},
              {q:'soft acoustic guitar', tag:'治愈'}, {q:'peaceful nature sounds', tag:'放松'}],
  disgusted: [{q:'clean fresh acoustic', tag:'清新'}, {q:'light indie pop', tag:'轻松'},
              {q:'bossa nova', tag:'松弛'}],
  surprised: [{q:'energetic electronic', tag:'兴奋'}, {q:'edm drop', tag:'高能'},
              {q:'fun quirky pop', tag:'俏皮'}],
  neutral:   [{q:'lofi study beats', tag:'平静'}, {q:'jazz cafe', tag:'日常'},
              {q:'instrumental focus music', tag:'专注'}, {q:'acoustic morning', tag:'温暖'}],
  uncertain: [{q:'chill mix', tag:'放松'}, {q:'popular instrumental', tag:'日常'}]
};

/* ---------------- 推荐策略：目标 VA 锚点 ---------------- */
const STRATEGY_CFG = {
  empathy:  { name:'共情',   alpha:0.85, desc:'理解你的此刻，推同情绪的歌' },
  regulate: { name:'调节',   alpha:0.20, desc:'轻轻把你带向更好的状态' },
  contrast: { name:'反差',   alpha:0.00, desc:'用完全相反的歌打破循环' },
  hybrid:   { name:'混合',   alpha:0.55, desc:'七分共情、三分调节（默认）' }
};

/* 每种情绪"调节"时该去往的目标 VA */
const REGULATE_TARGET = {
  sad:       { v: 0.45, a:-0.20 },   // 悲伤 → 温和明亮
  angry:     { v: 0.25, a:-0.55 },   // 愤怒 → 平静降温
  fearful:   { v: 0.40, a:-0.55 },   // 恐惧 → 安全安定
  disgusted: { v: 0.35, a: 0.10 },   // 厌恶 → 清爽
  happy:     { v: 0.85, a: 0.55 },   // 开心 → 保持高峰
  surprised: { v: 0.40, a:-0.30 },   // 惊讶 → 稳住情绪
  neutral:   { v: 0.50, a: 0.30 },   // 平静 → 提一点能量
  uncertain: { v: 0.40, a:-0.10 }
};

const REC_WEIGHTS = { va:0.34, tag:0.26, bpm:0.14, quality:0.06, novelty:0.10, mmr:0.26 };

/* 效价极性惩罚：当目标效价为正时，负效价歌曲会被明显降权（反之亦然），
   避免"开心时被推荐愤怒摇滚"。极性冲突按 0.55 的系数扣分。 */
const POLARITY_PENALTY = 0.55;

/* 情绪 → 不相容标签：命中即强降权（比极性惩罚更硬） */
const INCOMPATIBLE_TAGS = {
  happy:     ['悲伤','愤怒','不安','宣泄','孤独'],
  sad:       ['开心','兴奋','高能','派对'],
  angry:     ['开心','助眠','冥想','治愈','轻松'],
  fearful:   ['愤怒','宣泄','强节奏','派对'],
  surprised: ['悲伤','助眠','冥想'],
  neutral:   ['愤怒','宣泄'],
  disgusted: ['愤怒','悲伤'],
  uncertain: []
};

/* ============================ 工具函数 ============================ */
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function simVA(song, tv, ta) {
  const d = Math.hypot(song.v - tv, song.a - ta);
  return 1 - d / (2 * Math.SQRT2);          // 最大距离 2√2
}
function simBPM(song, targetBpm, sigma = 22) {
  return Math.exp(-Math.pow(song.bpm - targetBpm, 2) / (2 * sigma * sigma));
}
function jaccard(a, b) {
  const A = new Set(a), B = new Set(b);
  let inter = 0;
  A.forEach(x => { if (B.has(x)) inter++; });
  const uni = A.size + B.size - inter;
  return uni === 0 ? 0 : inter / uni;
}
/* 6 维特征向量，用于 MMR 多样性 */
function featVec(s) {
  return [s.v, s.a, s.bpm / 160, s.energy, Math.abs(s.v) * Math.sign(s.v), s.a * s.a];
}
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

/* ============================ 推荐引擎主类 ============================ */
class Recommender {
  constructor(tracks) {
    this.tracks = tracks.slice();
    this.history = [];        // 播放过的 id
    this.shownCount = {};     // 曝光次数（用于新鲜度衰减）
  }

  /** 计算目标 VA 锚点 */
  computeTarget(emotionResult, strategy = 'hybrid') {
    const cur = {
      v: emotionResult.valence ?? 0,
      a: emotionResult.arousal ?? 0
    };
    const cfg = STRATEGY_CFG[strategy] || STRATEGY_CFG.hybrid;
    let goal;
    if (strategy === 'empathy')       goal = cur;                             // 直接等于当前情绪
    else if (strategy === 'contrast') {
      goal = { v: -cur.v * 0.9, a: -cur.a * 0.8 + 0.15 };
      // 边界情况：当前情绪接近中性时，反向目标会落回原点附近，
      // 「反差」就名存实亡（四种策略会给出同一个第一名）。
      // 此时显式推向负效价-高唤醒象限，保证确实拉开距离。
      if (Math.hypot(goal.v, goal.a) < 0.45) goal = { v: -0.6, a: 0.6 };
    }
    else                              goal = REGULATE_TARGET[emotionResult.label] || REGULATE_TARGET.neutral;
    const target = {
      v: clamp(cfg.alpha * cur.v + (1 - cfg.alpha) * goal.v, -1, 1),
      a: clamp(cfg.alpha * cur.a + (1 - cfg.alpha) * goal.a, -1, 1)
    };
    // 唤醒度 → 目标 BPM（A=-1 → 60，A=1 → 150）
    target.bpm = 60 + (target.a + 1) / 2 * 90;
    return target;
  }

  /**
   * 主入口：推荐
   * @param {object} emotion  EmotionEngine.infer() 的结果
   * @param {object} opt { strategy, limit, pool }
   */
  recommend(emotion, opt = {}) {
    const { strategy = 'hybrid', limit = 8, pool = null } = opt;
    const target = this.computeTarget(emotion, strategy);
    const emotionTags = this._emotionTags(emotion);
    const candidates = pool || this.tracks;

    const scored = candidates.map(s => {
      const va = simVA(s, target.v, target.a);
      const tag = jaccard(emotionTags, s.tags || []);
      const bpmS = simBPM(s, target.bpm);
      const quality = clamp((s.energy * 0.4 + 0.6), 0, 1);       // 简化质量先验
      const shown = this.shownCount[s.id] || 0;
      const novelty = 1 / (1 + shown * 0.6);

      // ① 效价极性冲突惩罚：目标愉悦而歌曲负效价（或反之）
      const polarity = Math.sign(s.v) !== 0 && Math.sign(target.v) !== 0 && Math.sign(s.v) !== Math.sign(target.v)
        ? Math.min(1, Math.abs(s.v - target.v) / 2) * POLARITY_PENALTY : 0;
      // ② 不相容标签惩罚
      const bad = (INCOMPATIBLE_TAGS[emotion.label] || []);
      const hitBad = (s.tags || []).some(t => bad.includes(t));
      const tagPenalty = hitBad ? 0.30 : 0;

      const base = REC_WEIGHTS.va * va + REC_WEIGHTS.tag * tag +
                   REC_WEIGHTS.bpm * bpmS + REC_WEIGHTS.quality * quality +
                   REC_WEIGHTS.novelty * novelty - polarity - tagPenalty;
      return {
        song: s, base,
        parts: { va, tag, bpm: bpmS, quality, novelty, polarity, tagPenalty }
      };
    });

    // MMR 贪心选择：兼顾得分与多样性
    const selected = [];
    const poolLeft = scored.slice();
    while (selected.length < limit && poolLeft.length) {
      let best = -1, bestScore = -Infinity;
      poolLeft.forEach((c, i) => {
        let maxSim = 0;
        for (const sel of selected) maxSim = Math.max(maxSim, Math.max(0, cosine(featVec(c.song), featVec(sel.song))));
        const mmrScore = c.base - REC_WEIGHTS.mmr * maxSim;
        if (mmrScore > bestScore) { bestScore = mmrScore; best = i; }
      });
      const pick = poolLeft.splice(best, 1)[0];
      pick.finalScore = bestScore;
      pick.maxSim = 0;
      for (const sel of selected) pick.maxSim = Math.max(pick.maxSim, Math.max(0, cosine(featVec(pick.song), featVec(sel.song))));
      selected.push(pick);
      this.shownCount[pick.song.id] = (this.shownCount[pick.song.id] || 0) + 1;
    }

    // 归一到 0-100 便于展示
    const maxS = Math.max(...selected.map(s => s.finalScore), 0.0001);
    return {
      target,
      strategy,
      strategyName: STRATEGY_CFG[strategy]?.name || strategy,
      emotionTags,
      list: selected.map(s => ({
        ...s.song,
        score: s.finalScore,
        // 用绝对分（理论满分 = 五项权重之和 0.90）换算百分比，
        // 而不是按本批最大值归一化——否则每次第一名都恒为 100%，
        // 会掩盖"其实只匹配了 70%"这种真实情况。
        scorePct: Math.round(clamp(s.finalScore / 0.90, 0, 1) * 1000) / 10,
        parts: s.parts,
        reason: this._reason(s, emotion, strategy, target)
      }))
    };
  }

  /** 生成"推荐理由"文案 */
  _reason(item, emotion, strategy, target) {
    const p = item.parts;
    const bits = [];
    if (p.va > 0.78) bits.push('情绪曲线高度贴合你此刻的状态');
    else if (p.va > 0.6) bits.push('情绪基调与你当前氛围接近');
    else if (strategy === 'contrast') bits.push('刻意选择反差曲风来切换心境');
    if (p.tag > 0.1) bits.push('标签命中「' + (item.song.tags.find(t => this._emotionTags(emotion).includes(t)) || item.song.tags[0]) + '」');
    const d = Math.abs(item.song.bpm - target.bpm);
    if (d < 12) bits.push(`节奏 ${item.song.bpm} BPM 与目标节奏几乎一致`);
    else if (d < 26) bits.push(`节奏接近目标 ${Math.round(target.bpm)} BPM`);
    if (!bits.length) bits.push('综合相似度较高的备选');
    return bits.slice(0, 2).join('，') + '。';
  }

  /** 情绪 → 期望的音乐标签集合 */
  _emotionTags(emotion) {
    const map = {
      happy:     ['开心','明亮','阳光','跃动','元气','兴奋','俏皮','高能'],
      sad:       ['悲伤','共情','孤独','怀旧','治愈','雨天','弦乐'],
      angry:     ['愤怒','释放','宣泄','解压','强节奏','摇滚'],
      fearful:   ['不安','安抚','冥想','放松','治愈','紧张'],
      disgusted: ['清新','轻松','松弛','日常'],
      surprised: ['惊讶','上扬','好奇','跳跃','高能','兴奋'],
      neutral:   ['平静','日常','专注','舒缓','温暖','Lo-Fi'],
      uncertain: ['放松','日常','平静','舒缓']
    };
    return map[emotion.label] || map.neutral;
  }

  markPlayed(id) { if (!this.history.includes(id)) this.history.push(id); }
}

/* ---------------- iTunes 在线曲库（JSONP） ---------------- */
const OnlineLibrary = {
  _cache: new Map(),
  _seq: 0,

  jsonp(url, timeout = 9000) {
    return new Promise((resolve, reject) => {
      const cb = '__itunes_cb_' + (++this._seq);
      const s = document.createElement('script');
      let done = false;
      const cleanup = () => { delete window[cb]; s.remove(); };
      const timer = setTimeout(() => { if (!done) { done = true; cleanup(); reject(new Error('请求超时')); } }, timeout);
      window[cb] = data => {
        if (done) return; done = true; clearTimeout(timer); cleanup(); resolve(data);
      };
      s.onerror = () => { if (done) return; done = true; clearTimeout(timer); cleanup(); reject(new Error('网络错误')); };
      s.src = url + (url.includes('?') ? '&' : '?') + 'callback=' + cb;
      document.head.appendChild(s);
    });
  },

  /** 按情绪关键词检索 */
  async searchByEmotion(emotionLabel, limit = 12) {
    const bank = ONLINE_QUERY_BANK[emotionLabel] || ONLINE_QUERY_BANK.neutral;
    const picked = bank[Math.floor(Math.random() * bank.length)];
    const key = emotionLabel + '|' + picked.q;
    if (this._cache.has(key)) {
      const c = this._cache.get(key);
      if (Date.now() - c.t < 5 * 60000) return c.v;
    }
    const url = `https://itunes.apple.com/search?term=${encodeURIComponent(picked.q)}` +
                `&media=music&entity=song&limit=${limit}&country=us`;
    const data = await this.jsonp(url);
    const list = (data.results || [])
      .filter(r => r.previewUrl && r.kind === 'song')
      .map((r, i) => ({
        id: 'itunes-' + r.trackId,
        title: r.trackName,
        artist: r.artistName,
        album: r.collectionName,
        file: r.previewUrl,
        cover: (r.artworkUrl100 || '').replace('100x100', '200x200'),
        online: true,
        v: 0.3 * Math.sin(i + picked.q.length),      // 无真实音频分析时给弱先验，主要由得分排序
        a: 0.2,
        bpm: 0, key:'—', genre: r.primaryGenreName || '在线', lang:'', energy: 0.5,
        dur: '0:30',
        tags: [picked.tag, emotionLabel === 'happy' ? '开心' : '在线'],
        desc: `来自 iTunes 的 30 秒试听 · ${r.primaryGenreName || ''}`
      }));
    this._cache.set(key, { t: Date.now(), v: list });
    return list;
  }
};

window.LOCAL_TRACKS = LOCAL_TRACKS;
window.Recommender = Recommender;
window.OnlineLibrary = OnlineLibrary;
window.STRATEGY_CFG = STRATEGY_CFG;
window.REC_WEIGHTS = REC_WEIGHTS;
window.ONLINE_QUERY_BANK = ONLINE_QUERY_BANK;
window.recUtils = { simVA, simBPM, jaccard, cosine, featVec };
