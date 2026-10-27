// 推荐算法离线回归测试（Node 环境）
global.window = {};
const fs = require('fs');
const path = require('path');
eval(fs.readFileSync(path.join(__dirname, '../js/recommend.js'), 'utf8'));

const R = new window.Recommender(window.LOCAL_TRACKS);

function show(label, v, a) {
  const emo = { label, valence: v, arousal: a };
  for (const st of ['empathy', 'regulate', 'hybrid', 'contrast']) {
    const r = R.recommend(emo, { strategy: st, limit: 5 });
    const names = r.list.map(x => `${x.title}(${x.scorePct})`);
    console.log(`[${label}|${st}] ` + names.join(' / '));
  }
  console.log('');
}

show('sad', -0.62, -0.50);
show('happy', 0.85, 0.60);
show('angry', -0.80, 0.75);
show('neutral', 0.00, -0.25);

// 断言：开心不应混入愤怒摇滚；悲伤共情应首位是悲伤曲
const hap = R.recommend({ label: 'happy', valence: 0.85, arousal: 0.6 }, { strategy: 'hybrid', limit: 5 });
const sad = R.recommend({ label: 'sad', valence: -0.62, arousal: -0.5 }, { strategy: 'empathy', limit: 5 });
console.log('开心首位:', hap.list[0].title, '| 含Red Riot?', hap.list.some(x => x.id === 'L07'));
console.log('悲伤共情首位:', sad.list[0].title, sad.list[0].mood);
const all = R.recommend({ label: 'sad', valence: -0.6, arousal: -0.5 }, { strategy: 'empathy', limit: 14 });
console.log('返回条数:', all.list.length, '唯一:', new Set(all.list.map(x => x.id)).size);

// 打印一条推荐理由
console.log('理由示例:', sad.list[0].reason);
