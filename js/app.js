/**
 * MoodTone · 主应用（Vue 3 Composition API）
 * 结构：状态 → 模型加载 → 四种采集源 → 实时循环 → 识别 → 推荐 → 播放器
 */
const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;

createApp({
  setup() {
    /* ================== 全局状态 ================== */
    const tab = ref('main');
    const tabs = [
      { key: 'main', name: '识别与推荐', icon: '🎯' },
      { key: 'algo', name: '算法说明', icon: '📐' }
    ];

    const modelReady = ref(false);
    const modelError = ref('');
    const loadMsg = ref('准备加载模型…');
    const modelList = ref([]);

    const sources = [
      { key: 'image',     name: '图片',   icon: '🖼',  needStream: false },
      { key: 'videoFile', name: '视频',   icon: '🎬',  needStream: false },
      { key: 'camera',    name: '摄像头', icon: '📷',  needStream: true  }
    ];
    const source = ref('image');
    const hasImage = ref(false);
    const imageUrl = ref('');
    const loadingFile = ref(false);
    const videoName = ref('');

    const video = ref(null);
    const videoFile = ref(null);
    const img = ref(null);
    const canvas = ref(null);
    const audio = ref(null);

    const streaming = ref(false);
    const videoPlaying = ref(false);
    const analyzing = ref(false);
    const finalResult = ref(null);
    const liveResult = ref(null);
    const fps = ref(0);
    const inferMs = ref(0);
    const engineBackend = ref('');

    /* 内置示例人像（点缩略图即可体验，无需自己找图） */
    const SAMPLE_FACES = [
      { name: '示例 1', thumb: './samples/thumb-415829.jpg', src: './samples/face-415829.jpg' },
      { name: '示例 2', thumb: './samples/thumb-614810.jpg', src: './samples/face-614810.jpg' },
      { name: '示例 3', thumb: './samples/thumb-774909.jpg', src: './samples/face-774909.jpg' },
      { name: '示例 4', thumb: './samples/thumb-733872.jpg', src: './samples/face-733872.jpg' },
      { name: '示例 5', thumb: './samples/thumb-1239291.jpg', src: './samples/face-1239291.jpg' }
    ];

    const strategy = ref('hybrid');
    const recommend = ref(null);
    const pool = ref([]);            // 当前候选池（本地 + 在线）
    const onlineLoading = ref(false);
    const onlineMsg = ref('');
    const shownLimit = ref(8);

    const current = ref(null);
    const playing = ref(false);
    const progress = ref(0);
    const curTime = ref(0);
    const duration = ref(0);
    const volume = ref(70);
    const muted = ref(false);
    const queue = ref([]);

    const detail = ref(null);

    const EMOTION_META = window.EMOTION_META;
    const EMOTION_VA = window.EMOTION_VA;
    const STRATEGY_CFG = window.STRATEGY_CFG;
    const REC_WEIGHTS = window.REC_WEIGHTS;

    const engine = new window.EmotionEngine();
    const recommender = new window.Recommender(window.LOCAL_TRACKS);

    /* ================== 模型加载 ================== */
    (async () => {
      try {
        if (typeof faceapi === 'undefined') throw new Error('face-api.js 未加载');
        await engine.loadModels('./models', (msg, pct) => { loadMsg.value = `${msg} ${pct}%`; });
        modelReady.value = true;
        modelList.value = engine.loadedModels;
        engineBackend.value = String(engine.backend || '').toUpperCase();
      } catch (e) {
        modelError.value = e.message || String(e);
        loadMsg.value = '模型加载失败：' + modelError.value;
        console.error(e);
      }
    })();

    /* ================== 工具 ================== */
    const canAnalyze = computed(() => {
      if (!modelReady.value) return false;
      if (source.value === 'image') return hasImage.value;
      if (source.value === 'videoFile') return videoPlaying.value || !!videoName.value;
      return streaming.value;
    });

    const displayList = computed(() => recommend.value ? recommend.value.list.slice(0, shownLimit.value) : []);

    /* ---------- 仪表盘可视化：环形进度 / VA 散点坐标 ---------- */
    const GAUGE_C = 2 * Math.PI * 54;
    const gaugeDash = computed(() => {
      const p = finalResult.value ? Math.max(0, Math.min(1, finalResult.value.confidence)) : 0;
      return `${(p * GAUGE_C).toFixed(1)} ${GAUGE_C.toFixed(1)}`;
    });
    // VA 空间坐标映射：viewBox 340×176，原点 (170, 82)
    const vaX = v => +(170 + v * 130).toFixed(1);
    const vaY = a => +(82 - a * 70).toFixed(1);
    const vaAnchors = computed(() =>
      Object.entries(EMOTION_VA).map(([k, a]) => {
        const y = vaY(a.a);
        // 标签默认在点上方；低唤醒（a<0）的放到下方，避免和当前点/目标点重叠
        return { k, zh: EMOTION_META[k].zh, c: EMOTION_META[k].color, x: vaX(a.v), y, labelY: a.a < 0 ? y + 13 : y - 7 };
      })
    );
    // 画面左上角标签：摄像头看实时结果，图片/视频优先展示最终识别结果，避免和仪表数字不一致
    const tagResult = computed(() =>
      source.value === 'camera' ? (liveResult.value || finalResult.value) : (finalResult.value || liveResult.value)
    );
    const vaNow = computed(() => finalResult.value
      ? { x: vaX(finalResult.value.valence), y: vaY(finalResult.value.arousal) }
      : { x: 170, y: 82 });
    const vaTgt = computed(() => recommend.value && recommend.value.target
      ? { x: vaX(recommend.value.target.v), y: vaY(recommend.value.target.a) }
      : { x: 170, y: 82 });

    function scoreLabel(p) {
      if (!p) return '—';
      if (p >= 75) return '极贴切';
      if (p >= 60) return '很贴近';
      if (p >= 45) return '较接近';
      if (p >= 30) return '一般';
      return '偏弱';
    }

    const detailMetrics = computed(() => {
      if (!detail.value || !detail.value.parts) return [];
      const p = detail.value.parts;
      return [
        { k: 'va',  name: 'VA 效价-唤醒相似度 (0.34)', v: p.va,  c: '#7c5cff' },
        { k: 'tag', name: '情绪标签命中率 (0.26)',     v: p.tag, c: '#22a3c4' },
        { k: 'bpm', name: '节奏接近度 (0.14)',         v: p.bpm, c: '#f0a020' },
        { k: 'qua', name: '曲库质量先验 (0.06)',       v: p.quality, c: '#12b76a' },
        { k: 'nov', name: '新鲜度 (0.10)',             v: p.novelty, c: '#e0567f' }
      ];
    });

    function fmtTime(t) {
      if (!isFinite(t) || t < 0) t = 0;
      const m = Math.floor(t / 60), s = Math.floor(t % 60);
      return `${m}:${String(s).padStart(2, '0')}`;
    }
    // 匹配分是绝对口径（满分 0.90），不是批内归一化，
    // 所以阈值要按真实分布标定：14 首小曲库里"能听"的一首通常 70+，
    // 40 左右只算沾边，20 以下基本是硬凑。
    function scoreColor(p) {
      if (p >= 70) return '#12b76a';
      if (p >= 45) return '#22a3c4';
      if (p >= 30) return '#f0a020';
      return '#c3cad6';
    }
    function coverStyle(s) {
      const c = s.cover || '#334155';
      if (c.startsWith('http')) return { background: '#0b1220' };
      return { background: `linear-gradient(135deg, ${c}, ${shade(c, -35)})` };
    }
    function shade(hex, pct) {
      const n = parseInt(hex.replace('#', ''), 16);
      let r = (n >> 16) + pct, g = ((n >> 8) & 0xff) + pct, b = (n & 0xff) + pct;
      r = Math.max(0, Math.min(255, r)); g = Math.max(0, Math.min(255, g)); b = Math.max(0, Math.min(255, b));
      return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0');
    }

    /* ================== 画布尺寸同步 ================== */
    function syncCanvas() {
      const c = canvas.value;
      if (!c) return;
      let w = 640, h = 400;
      if (source.value === 'camera' && video.value && video.value.videoWidth) {
        w = video.value.videoWidth; h = video.value.videoHeight;
      } else if (source.value === 'videoFile' && videoFile.value && videoFile.value.videoWidth) {
        w = videoFile.value.videoWidth; h = videoFile.value.videoHeight;
      } else if (source.value === 'image' && img.value && img.value.naturalWidth) {
        w = img.value.naturalWidth; h = img.value.naturalHeight;
      }
      c.width = w; c.height = h;
      return { w, h };
    }

    function currentSourceEl() {
      if (source.value === 'camera') return streaming.value ? video.value : null;
      if (source.value === 'videoFile') return videoFile.value && videoFile.value.readyState >= 2 ? videoFile.value : null;
      if (source.value === 'image') return img.value && img.value.complete && img.value.naturalWidth ? img.value : null;
      return null;
    }

    /** 把检测框画到 canvas 上（图片/视频模式可见） */
    function drawOverlay(faces, srcEl, dims) {
      const c = canvas.value;
      if (!c) return;
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, c.width, c.height);
      if (!faces || !faces.length) return;
      // 保证 canvas 与来源同尺寸
      if (srcEl && (c.width !== srcEl.videoWidth && c.width !== srcEl.naturalWidth)) {
        c.width = srcEl.videoWidth || srcEl.naturalWidth || c.width;
        c.height = srcEl.videoHeight || srcEl.naturalHeight || c.height;
      }
      for (const f of faces) {
        const b = f.box;
        const top = Object.entries(f.expressions).sort((x, y) => y[1] - x[1])[0];
        const meta = EMOTION_META[top[0]] || EMOTION_META.neutral;
        // 圆角框
        ctx.lineWidth = Math.max(2, c.width / 320);
        ctx.strokeStyle = meta.color;
        ctx.shadowColor = meta.color; ctx.shadowBlur = 12;
        const r = 10;
        ctx.beginPath();
        ctx.roundRect(b.x, b.y, b.width, b.height, r);
        ctx.stroke();
        ctx.shadowBlur = 0;
        // 标签
        const label = `${meta.emoji} ${meta.zh} ${(top[1] * 100).toFixed(0)}%`;
        const fs = Math.max(13, c.width / 40);
        ctx.font = `600 ${fs}px system-ui, sans-serif`;
        const tw = ctx.measureText(label).width + 16;
        const th = fs + 12;
        ctx.fillStyle = meta.color;
        ctx.beginPath();
        ctx.roundRect(b.x, Math.max(0, b.y - th - 4), tw, th, 7);
        ctx.fill();
        ctx.fillStyle = '#0b1220';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, b.x + 8, Math.max(0, b.y - th - 4) + th / 2);
        // 关键点
        if (f.landmarks) {
          ctx.fillStyle = 'rgba(255,255,255,.55)';
          for (const p of f.landmarks) {
            ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(1, c.width / 400), 0, Math.PI * 2); ctx.fill();
          }
        }
      }
    }

    /* ================== 采集源切换 ================== */
    async function switchSource(key) {
      if (key === source.value) return;
      stopRealtime();
      if (source.value === 'camera' && streaming.value) await closeCamera();
      videoPlaying.value = false;
      source.value = key;
      engine.resetSmoothing();
      liveResult.value = null;
      await nextTick();
      syncCanvas();
      // 清空旧叠加层
      const c = canvas.value;
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
      hasImage.value = (key === 'image' && !!imageUrl.value);
    }

    /* ---- 图片 ---- */
    async function onPickImage(e) {
      const f = e.target.files && e.target.files[0];
      e.target.value = '';
      if (!f) return;
      loadingFile.value = true;
      try {
        const url = URL.createObjectURL(f);
        if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
        imageUrl.value = url;
        hasImage.value = true;
        if (source.value !== 'image') await switchSource('image');
        engine.resetSmoothing();
        finalResult.value = null;
        await nextTick();
        // 等图片解码
        await new Promise(res => {
          if (img.value.complete) return res();
          img.value.onload = res;
        });
        syncCanvas();
        drawOverlay(null);
        await analyze();
      } catch (err) {
        console.error(err);
        alert('图片读取失败：' + err.message);
      } finally {
        loadingFile.value = false;
      }
    }

    /* ---- 示例图一键体验 ---- */
    async function loadSample(src) {
      if (!modelReady.value) { alert('模型还在加载，请稍等…'); return; }
      if (source.value !== 'image') await switchSource('image');
      imageUrl.value = src;
      hasImage.value = true;
      engine.resetSmoothing();
      finalResult.value = null;
      await nextTick();
      await new Promise(res => {
        if (img.value && img.value.complete && img.value.naturalWidth) return res();
        if (img.value) img.value.onload = res;
        setTimeout(res, 4000);
      });
      syncCanvas();
      drawOverlay(null);
      await analyze();
    }

    /* ---- 视频文件 ---- */
    async function onPickVideo(e) {
      const f = e.target.files && e.target.files[0];
      e.target.value = '';
      if (!f) return;
      if (source.value !== 'videoFile') await switchSource('videoFile');
      await nextTick();
      const v = videoFile.value;
      if (!v) return;
      if (videoFileUrl) URL.revokeObjectURL(videoFileUrl);
      videoFileUrl = URL.createObjectURL(f);
      v.src = videoFileUrl;
      videoName.value = f.name;
      engine.resetSmoothing();
      finalResult.value = null;

      // 等元数据就绪，确保拿到视频分辨率后再同步 canvas
      await new Promise(res => {
        const done = () => { v.removeEventListener('loadedmetadata', done); res(); };
        if (v.readyState >= 1 && v.videoWidth) return res();
        v.addEventListener('loadedmetadata', done);
        setTimeout(res, 3000);
      });

      // 静音才能绕过浏览器自动播放策略（识别只用画面，不需要声音）
      v.muted = true;
      v.loop = true;
      try {
        await v.play();
        videoPlaying.value = true;
      } catch (err) {
        console.warn('视频自动播放失败，可手动点击播放', err);
        videoPlaying.value = true;
      }
      await nextTick();
      syncCanvas();
    }

    let videoFileUrl = '';
    let lastVideoTime = 0;
    async function onVideoTime() {
      const v = videoFile.value;
      if (!v || !modelReady.value || !videoPlaying.value) return;
      // 限制到 6fps，降低负载
      if (v.currentTime - lastVideoTime < 1 / 6) return;
      lastVideoTime = v.currentTime;
      try {
        const r = await engine.detect(v);
        const res = engine.infer(r.faces);
        liveResult.value = res.ok ? res : null;
        syncCanvas();
        drawOverlay(r.faces, v);
      } catch (err) { /* 忽略单帧错误 */ }
    }

    /* ---- 摄像头 ---- */
    let stream = null;
    let rafId = null;
    let lastTick = performance.now();
    let frames = 0;
    let busy = false;

    async function openCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('当前浏览器不支持摄像头，或页面不在安全上下文（需 https 或 localhost）。');
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });
        source.value = 'camera';
        await nextTick();
        video.value.srcObject = stream;
        await video.value.play();
        streaming.value = true;
        engine.resetSmoothing();
        await nextTick();
        syncCanvas();
        startRealtime();
      } catch (err) {
        alert('无法访问摄像头：' + err.message + '\n请检查浏览器权限设置。');
      }
    }

    async function closeCamera() {
      stopRealtime();
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      if (video.value) video.value.srcObject = null;
      streaming.value = false;
      liveResult.value = null;
    }

    async function toggleCamera() {
      if (streaming.value) await closeCamera();
      else await openCamera();
    }

    function startRealtime() {
      cancelAnimationFrame(rafId);
      frames = 0; lastTick = performance.now();
      const loop = async () => {
        if (!streaming.value) return;
        if (!busy && video.value && video.value.readyState >= 2) {
          busy = true;
          try {
            const r = await engine.detect(video.value);
            const res = engine.infer(r.faces);
            liveResult.value = res.ok ? res : { ...EMOTION_META.uncertain, ok: false, confidence: 0, emoji: '🙈', zh: '未检测到人脸' };
            frames++;
            const now = performance.now();
            if (now - lastTick > 1000) {
              fps.value = Math.round(frames * 1000 / (now - lastTick));
              frames = 0; lastTick = now;
            }
          } catch (e) { /* 忽略 */ }
          busy = false;
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    function stopRealtime() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    }

    let snapUrl = '';
    async function snapPhoto() {
      if (!streaming.value || !video.value) return;
      const v = video.value;
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      // 关闭摄像头，切到图片模式
      await closeCamera();
      if (snapUrl) URL.revokeObjectURL(snapUrl);
      const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
      snapUrl = URL.createObjectURL(blob);
      source.value = 'image';
      if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
      imageUrl.value = snapUrl;
      hasImage.value = true;
      await nextTick();
      await new Promise(res => { if (img.value.complete) return res(); img.value.onload = res; });
      syncCanvas();
      await analyze();
    }

    /* ================== 识别 ================== */
    async function analyze() {
      const el = currentSourceEl();
      if (!el) { alert('请先提供人脸来源（图片/视频/摄像头）'); return; }
      analyzing.value = true;
      try {
        const t0 = performance.now();
        const r = await engine.detect(el);
        const res = engine.infer(r.faces, false);
        inferMs.value = Math.round(performance.now() - t0);
        if (!res.ok) {
          finalResult.value = { ...EMOTION_META.uncertain, ok: false, faceCount: 0,
            valence: 0, arousal: 0, confidence: 0, margin: 0, intensity: 0,
            sorted: Object.entries(EMOTION_META).filter(([k]) => EMOTION_VA[k]).map(([k]) => [k, 0]) };
          alert('没有检测到人脸，请换一张清晰、正对镜头的图片。');
          return;
        }
        finalResult.value = res;
        syncCanvas();
        drawOverlay(r.faces, el);
        doRecommend();
        if (source.value === 'image') tab.value = 'main';
      } catch (err) {
        console.error(err);
        alert('识别失败：' + err.message);
      } finally {
        analyzing.value = false;
      }
    }

    /* ================== 推荐 ================== */
    function doRecommend() {
      if (!finalResult.value) return;
      // 候选池：本地曲库 + 已加载的在线曲目（在线曲目用其标签参与排序）
      const poolNow = window.LOCAL_TRACKS.concat(pool.value);
      const rec = recommender.recommend(finalResult.value, {
        strategy: strategy.value,
        limit: 14,
        pool: poolNow
      });
      recommend.value = rec;
      shownLimit.value = 8;
    }

    function setStrategy(k) {
      strategy.value = k;
      if (finalResult.value) doRecommend();
    }

    function refreshRec() {
      if (finalResult.value) doRecommend();
    }

    async function loadOnline() {
      if (onlineLoading.value) return;
      const emo = finalResult.value ? finalResult.value.label : 'neutral';
      onlineLoading.value = true;
      onlineMsg.value = '';
      try {
        const list = await window.OnlineLibrary.searchByEmotion(emo, 12);
        if (!list.length) { onlineMsg.value = '在线曲库暂无结果，请稍后重试'; return; }
        // 在线曲目没有音频特征，用情绪标签 + 流派给一个近似 VA 先验
        const priors = { happy: [0.7, 0.55], sad: [-0.5, -0.3], angry: [-0.4, 0.6],
                         fearful: [0.3, -0.4], surprised: [0.3, 0.65], neutral: [0.15, -0.15], disgusted: [0.2, 0.1] };
        const [pv, pa] = priors[emo] || [0.1, 0];
        list.forEach((t, i) => {
          t.v = pv + (Math.random() - 0.5) * 0.18;
          t.a = pa + (Math.random() - 0.5) * 0.18;
          t.energy = 0.4 + Math.random() * 0.4;
        });
        pool.value = list;
        onlineMsg.value = `已从 iTunes 获取 ${list.length} 首在线试听`;
        doRecommend();
      } catch (e) {
        onlineMsg.value = '在线检索失败：' + e.message + '（可继续使用本地曲库）';
      } finally {
        onlineLoading.value = false;
      }
    }

    function browseAll() {
      recommend.value = {
        strategyName: '浏览',
        target: { v: 0, a: 0, bpm: 90 },
        list: window.LOCAL_TRACKS.map(s => ({ ...s, scorePct: 0, reason: s.desc, parts: null, score: 0 }))
      };
      shownLimit.value = 14;
    }

    function openDetail(s) {
      if (!s.parts) return;
      detail.value = s;
    }

    /* ================== 播放器 ================== */
    let playUrlCache = {};

    function playSong(s) {
      if (current.value && current.value.id === s.id) {
        togglePlay();
        return;
      }
      // 记录播放队列
      const list = displayList.value;
      queue.value = list.map(x => x.id);
      current.value = s;
      recommender.markPlayed(s.id);
      progress.value = 0; curTime.value = 0;
      nextTick(() => {
        const a = audio.value;
        if (!a) return;
        a.src = s.file;
        a.volume = muted.value ? 0 : volume.value / 100;
        a.play().catch(err => {
          console.warn(err);
          if (s.online) alert('在线试听请求失败，可能是网络或版权限制，可换下一首。');
        });
      });
    }

    function togglePlay() {
      const a = audio.value;
      if (!a || !current.value) return;
      if (a.paused) a.play().catch(() => {}); else a.pause();
    }

    function nextSong() {
      if (!queue.value.length || !current.value) return;
      const i = queue.value.indexOf(current.value.id);
      const nxt = displayList.value[(i + 1) % displayList.value.length];
      if (nxt) playSong(nxt);
    }
    function prevSong() {
      if (!queue.value.length || !current.value) return;
      const i = queue.value.indexOf(current.value.id);
      const len = displayList.value.length;
      const prv = displayList.value[(i - 1 + len) % len];
      if (prv) playSong(prv);
    }

    function onTime() {
      const a = audio.value;
      if (!a) return;
      curTime.value = a.currentTime;
      progress.value = a.duration ? (a.currentTime / a.duration * 100) : 0;
    }
    function onMeta() {
      const a = audio.value;
      duration.value = a ? a.duration : 0;
    }
    function onEnded() {
      playing.value = false;
      nextSong();
    }
    function onAudioError() {
      if (current.value) console.warn('音频加载失败：', current.value.file);
    }
    function seek(e) {
      const a = audio.value;
      if (!a || !a.duration) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const p = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      a.currentTime = p * a.duration;
    }
    function setVolume() {
      const a = audio.value;
      if (!a) return;
      muted.value = volume.value === 0;
      a.volume = volume.value / 100;
    }
    function muteToggle() {
      const a = audio.value;
      if (!a) return;
      muted.value = !muted.value;
      a.volume = muted.value ? 0 : volume.value / 100;
    }

    /* ================== 生命周期 ================== */
    onMounted(() => {
      syncCanvas();
      // 未做识别时也展示曲库，避免右侧一片空白
      browseAll();
      window.addEventListener('resize', () => { /* canvas 自适应由 CSS 控制 */ });
    });

    // 切换 tab 时暂停实时循环外的重型操作（保持摄像头画面）
    watch(tab, v => { if (v === 'algo' && streaming.value) { /* 不停摄像头 */ } });

    return {
      tab, tabs, modelReady, modelError, loadMsg, modelList,
      sources, source, hasImage, imageUrl, loadingFile, videoName,
      video, videoFile, img, canvas, audio,
      streaming, videoPlaying, analyzing, finalResult, liveResult, fps,
      strategy, recommend, displayList, onlineLoading, onlineMsg, detail, detailMetrics,
      current, playing, progress, curTime, duration, volume, muted,
      EMOTION_META, EMOTION_VA, STRATEGY_CFG, canAnalyze,
      inferMs, engineBackend, SAMPLE_FACES, tagResult, gaugeDash, vaAnchors, vaNow, vaTgt, scoreLabel,
      switchSource, loadSample, onPickImage, onPickVideo, onVideoTime, toggleCamera, snapPhoto, analyze,
      setStrategy, refreshRec, loadOnline, browseAll, openDetail,
      playSong, togglePlay, nextSong, prevSong, onTime, onMeta, onEnded, onAudioError,
      seek, setVolume, muteToggle, fmtTime, scoreColor, coverStyle
    };
  }
}).mount('#app');
