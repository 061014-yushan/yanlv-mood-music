# 颜律 YanLü · 基于人脸情绪识别与场景匹配的智能音乐推荐

纯前端实现（HTML5 + CSS3 + JavaScript ES6+ + **Vue 3 Composition API**），
**情绪识别算法完全在浏览器本地运行**，不上传任何图片。

```
mood-music/
├── index.html              # 页面结构（Vue 模板）
├── css/style.css           # 样式（浅色仪表盘 + CSS 变量）
├── js/
│   ├── vue.global.prod.js  # Vue 3.4.38 运行时（本地引入，不依赖网络）
│   ├── face-api.min.js     # face-api.js 1.7.15（内置 TensorFlow.js）
│   ├── tfjs-backend-wasm*.wasm  # TF.js WASM 后端（CPU 回退时使用）
│   ├── emotion.js          # ★ 情绪识别 + 情绪融合算法
│   ├── recommend.js        # ★ 歌曲库 + 推荐算法 + iTunes API
│   └── app.js              # Vue 应用逻辑
├── models/                 # 三个预训练模型权重（约 900KB，断网可用）
├── audio/                  # 14 首本地原创器乐片段（MP3，可离线播放）
├── samples/                # 5 张示例人像（点缩略图即可体验）+ 测试视频
└── server/
    ├── serve.py            # 本地静态服务器（正确 MIME + COOP/COEP）
    ├── gen_audio.py        # 本地曲目合成器（生成 audio/*.mp3）
    └── test_recommend.js   # 推荐算法离线回归测试
```

## 运行

```bash
cd mood-music
python3 server/serve.py          # 默认 http://localhost:8099
```

> **必须用 HTTP 服务器打开**，不能直接双击 `index.html`。
> 原因：`getUserMedia`（摄像头）与模型权重 `fetch` 在 `file://` 下受限。
> 手机测试：用终端输出的局域网地址访问；摄像头在非 localhost 的 http 下需要
> 浏览器允许不安全源（Chrome: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`）。

## 功能

| 能力 | 实现 |
|---|---|
| 输入图片 | `<input type=file accept=image/*>` → Blob URL → canvas 叠加检测框 |
| 打开视频 | 视频文件逐帧识别（限流 6fps），实时框选 + HUD |
| 实时摄像头 | `getUserMedia` + `requestAnimationFrame` 循环，带 FPS 统计 |
| 拍照 | 截取当前视频帧到 canvas → 转为图片模式分析 |
| 情绪识别 | 三阶段 CNN 级联 + 5 步融合算法（见下） |
| 歌曲推荐 | 6 项加权打分 + MMR 多样性重排（见下） |
| 播放 | 本地 MP3 + iTunes 30s 试听，含进度/音量/上下一首 |
| 在线曲库 | iTunes Search API（JSONP，免密钥） |
| 界面 | 浅色仪表盘：① 获取人脸 ② 情绪识别（环形置信度 + 7 类概率 + VA 散点）③ 推荐表格 + 底部播放条 |

界面按「采集 → 识别 → 推荐」三段式排布：左侧采集卡内嵌 5 张示例人像缩略图，
右侧识别卡用环形仪表显示置信度、6 项指标条、7 类 softmax 概率，以及
Valence–Arousal 二维散点（实心点＝当前情绪，空心圈＝推荐目标，虚线连接两者）；
底部推荐表按「序号 / 歌曲 / 匹配分 / 推荐依据 / 匹配度」列展示，可原地播放与查看打分细节。

---

## 一、情绪识别算法（明确说明）

### 1.1 推理流水线（三层级联 CNN，本地 TensorFlow.js）

| 阶段 | 模型 | 结构 | 输入 → 输出 |
|---|---|---|---|
| L1 人脸检测 | **TinyFaceDetector** | SSD + MobileNetV1（深度可分离卷积） | 416×416 RGB → 人脸框 + 5 关键点 + 置信度 |
| L2 关键点对齐 | **FaceLandmark68Net** | 68 点坐标回归 | 对齐人脸 → 68 个关键点 |
| L3 表情分类 | **FaceExpressionNet** | 深度可分离卷积 + 全局平均池化 + FC | 对齐人脸 → 7 类 softmax |

超参：`scoreThreshold = 0.4`，`inputSize = 416`，`NMS IoU = 0.5`，最多 5 张脸。

7 类输出：`neutral / happy / sad / angry / fearful / disgusted / surprised`

### 1.2 情绪融合算法（自行实现，5 步）

模型只给 7 个概率，直接 `argmax` 会抖得厉害。本项目在其上叠加 5 步后处理：

**① 多脸面积加权融合**
```
p̄ = Σ(a_j · p_j) / Σ a_j      a_j = 人脸框宽×高
```
画面中占主要位置的人对结果影响更大。

**② EMA 时序平滑**
```
p_t = α · p̂_t + (1−α) · p_{t−1}      α = 0.35
```
抑制单帧抖动，避免表情一闪就切歌。

**③ Valence–Arousal 二维映射**（情感计算经典模型）

每个情绪在 VA 空间有锚点：

| | neutral | happy | sad | angry | fearful | disgusted | surprised |
|---|---|---|---|---|---|---|---|
| V 效价 | 0.00 | 0.85 | −0.70 | −0.80 | −0.65 | −0.75 | 0.15 |
| A 唤醒 | −0.25 | 0.60 | −0.55 | 0.75 | 0.55 | 0.20 | 0.80 |

```
V = Σ p_i · v_i        A = Σ p_i · a_i
```

**④ 可信度判定**

主导情绪需同时满足：
- `p_max ≥ 0.45`
- `p_max − p_second ≥ 0.12`

否则输出「待确认 / uncertain」，**不强行推荐**，提示用户调整光线或正对镜头。

**⑤ 情绪强度**
```
intensity = p_max × (1 + 0.5·|V|)
```
高唤醒的愤怒比中性的平静更"强烈"。

### 1.3 计算后端自适应

`WebGL → WASM(SIMD) → CPU` 逐级探测，每个后端都做一次真实张量冒烟测试，
失败即降级。没有 GPU 的无头/老设备同样能跑（本项目已在无 WebGL 的
headless Chromium 下验证通过，走 CPU 后端）。

---

## 二、歌曲推荐算法（明确说明）

### 2.1 曲库双通道

| 通道 | 内容 |
|---|---|
| **本地** | 14 首原创器乐（`audio/*.mp3`，由 `gen_audio.py` 用 numpy 加法合成 + FFT 混响生成），覆盖 7 种情绪各 2 首，离线可播 |
| **在线** | iTunes Search API 实时检索，按情绪关键词取回 12 首，带 30 秒试听直链与封面 |

本地曲目均带**音频特征标注**：`v`(效价) `a`(唤醒) `bpm` `调式` `流派` `energy` `tags[]`。

### 2.2 第一步：目标 VA 锚点

```
target = α · VA(当前情绪) + (1 − α) · VA(目标情绪)
BPM*   = 60 + (A + 1)/2 × 90
```

| 策略 | α | 目标 | 效果 |
|---|---|---|---|
| 共情 empathy | 0.85 | = 当前情绪 | 悲伤时放悲伤的歌，被理解 |
| 调节 regulate | 0.20 | 修正方向（悲伤→温和明亮） | 轻轻把情绪带出来 |
| 反差 contrast | 0.00 | = 相反 VA | 用相反曲风打破情绪循环 |
| 混合 hybrid（默认） | 0.55 | 折中 | 七分共情三分调节 |

> 反差策略的边界处理：当前情绪接近中性（`|VA| < 0.45`）时，"取反"会落回原点附近，
> 反差就没有意义了（四种策略会给出同一个第一名）。此时显式把目标推向负效价-高唤醒象限
> （`V=-0.6, A=0.6`），实测中性情绪下反差首位从 Coffee Loop 变成 Heartbeat Room，四种策略才真正区分开。

### 2.3 第二步：六项加权打分

```
Score = 0.34·Sim_VA + 0.26·Match_tag + 0.14·Sim_BPM + 0.06·Quality
      + 0.10·Novelty − 0.26·MMR − Polarity − TagPenalty
```

| 项 | 公式 | 作用 |
|---|---|---|
| `Sim_VA` | `1 − √((Δv)²+(Δa)²)/(2√2)` | 情绪曲线上是否贴合 |
| `Match_tag` | `Jaccard(情绪期望标签, 歌曲标签)` | 语义命中（治愈/解压/共情…） |
| `Sim_BPM` | `exp(−(bpm−bpm*)²/(2σ²))`，σ=22 | 节奏是否匹配心情唤醒度 |
| `Quality` | 能量先验 | 曲库质量 |
| `Novelty` | `1/(1+曝光次数×0.6)` | 避免总推同样几首 |
| `MMR` | `max cos(feat_候选, feat_已选)` | 惩罚同质化（6 维特征） |
| `Polarity` | 效价符号冲突 ≥ 0.55 | **开心时不会推愤怒摇滚** |
| `TagPenalty` | 命中不相容标签 −0.30 | 硬约束（如 happy 排除「悲伤」「宣泄」） |

### 2.4 第三步：MMR 贪心重排

每轮选 `argmax(Score − 0.30·maxSim)`，既保证相关性，又保证歌单有变化——
不会八首歌都是一个味道。

### 2.5 匹配分的口径

`匹配分 = finalScore / 0.90 × 100`（0.90 为五项正权重之和，即理论满分）。

这里**不按本批最大值归一化**——如果按批归一化，每次第一名的分数都恒等于 100%，
会掩盖「其实只匹配了 70%」这一事实。改为绝对口径后，同一首歌换策略会得到不同分数，
例如「悲伤 + 共情」下 Rain on Glass 85.1，切到「调节」降到 26.4，排序变化是真实可解释的。

> 注意：反差策略会刻意挑选与你此刻情绪相反的歌曲，匹配分天然偏低（常见 20–60），
> 这不是算法失效，界面上也做了对应提示。

### 2.6 输出解释性

每首歌返回 `parts` 明细（各项得分）与自然语言 `reason`，
UI 里的 ⓘ 按钮可查看完整打分拆解，做到**推荐结果可解释**。

### 2.7 离线验证

```bash
node server/test_recommend.js
```

实际输出（节选）：

```
[sad|empathy]  Rain on Glass(85.1) / Faded Music Box(48.1) / Hollow Corridor(37.6) ...
[sad|regulate] Coffee Loop(63.5) / Quiet Window(43.9) / Wooden Strings(34.1) ...
[sad|contrast] Upside Ascent(66.1) / Static Spark(33.0) / Drifting Light(27.2) ...
[happy|hybrid] Sunlit Steps(79.5) / Neon Bounce(33.7) / Upside Ascent(29.0) ...
开心首位: Sunlit Steps | 含Red Riot? false     ← 极性惩罚生效
返回条数: 14 唯一: 14                          ← MMR 无重复
```

---

## 三、技术栈

- **HTML5**：`<video>` `<canvas>` `getUserMedia` `Blob/URL` `Web Audio`
- **CSS3**：Grid、Flex、CSS 自定义属性、`backdrop-filter`、`conic-gradient`、关键帧动画
- **JavaScript ES6+**：class、async/await、Proxy 响应式
- **Vue 3.4.38**：Composition API（`ref` / `computed` / `watch` / `onMounted` / `nextTick`），**本地引入**
- **face-api.js 1.7.15 + TensorFlow.js**
- **iTunes Search API**（JSONP 免密钥）

## 四、已验证项

在 headless Chromium 中完成端到端验证：

- 模型加载 3 个网络成功（WebGL / WASM / CPU 自适应，无 WebGL 环境自动降到 CPU）
- 图片识别：检测框 + 68 关键点 + 7 类概率 + VA 平面
- 视频识别：逐帧推理、画面标签实时更新
- 摄像头：`getUserMedia` 拿到 640×480 流，8 FPS 实时推理
- 拍照分析：视频帧截图 → 转图片模式 → 识别
- 播放：本地 MP3 播放、进度、时长正确
- 在线：iTunes 取回 12 首带封面与试听直链
- 策略切换：共情/调节/反差/混合均正确改变目标 VA 与排序
- 推荐打分详情弹窗、算法说明页渲染正常
- 界面：1440×960 桌面与 390×844 移动端两套布局均无溢出、无控制台报错

## 五、免责声明

- 本地 14 首曲目为程序合成音频（`gen_audio.py` 生成），可自由使用。
- 在线试听来自 iTunes Search API，仅作演示，版权归原权利人所有。
- 情绪识别结果为算法推断，不构成任何医学或心理诊断依据。
