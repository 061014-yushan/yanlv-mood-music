#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地示例曲目合成器
生成 7 种情绪 × 2 首 = 14 段原创器乐片段（每段约 22~28 秒），
用 numpy 做加法合成 + 简单混响，再由 ffmpeg 编码为 96kbps 单声道 MP3。
这些音频与歌曲库中 local[*].mp3 一一对应，离线可播放。
"""
import numpy as np, os, subprocess, tempfile

SR = 22050
rng = np.random.default_rng(20240501)

NOTE = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}

def hz(name):
    """'A4' -> 440.0"""
    n = name[:-1]; octv = int(name[-1])
    return 440.0 * (2 ** ((NOTE[n] + (octv - 4) * 12 - 9) / 12))

def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    a, d, r = int(a*SR), int(d*SR), int(r*SR)
    s_len = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, max(a,1)),
        np.linspace(1, s, max(d,1)),
        np.full(s_len, s),
        np.linspace(s, 0, max(r,1))
    ])
    if len(env) < n: env = np.pad(env, (0, n-len(env)))
    return env[:n]

def pluck(f, dur, vol=0.5, bright=0.6):
    """拨弦/钢琴式音色：多个泛音 + 快速衰减"""
    n = int(dur*SR); t = np.arange(n)/SR
    sig = np.zeros(n)
    for k, amp in enumerate([1, .5, .3, .18, .1, .06], start=1):
        if f*k > SR/2: break
        sig += amp * bright**(k-1) * np.sin(2*np.pi*f*k*t)
    env = np.exp(-t*4.5) * (1 - np.exp(-t*220))
    return vol * sig * env / 2.2

def pad(f, dur, vol=0.3):
    """柔和铺底：两个微失谐的三角波叠加"""
    n = int(dur*SR); t = np.arange(n)/SR
    sig = np.zeros(n)
    for det in (-0.12, 0.0, 0.12):
        ff = f * (2 ** (det/12))
        for k in (1,2,3,4):
            if ff*k > SR/2: break
            sig += (1/k**1.6) * np.sin(2*np.pi*ff*k*t + rng.uniform(0,6.28))
    return vol * sig * adsr(n, .35, .2, .8, .6) / 2.0

def bass(f, dur, vol=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    sig = np.sin(2*np.pi*f*t) + 0.35*np.sin(2*np.pi*f*2*t) + 0.12*np.sin(2*np.pi*f*3*t)
    return vol * sig * adsr(n, .006, .12, .55, .25) / 1.4

def kick(vol=0.9):
    n = int(0.28*SR); t = np.arange(n)/SR
    f = 120*np.exp(-t*28) + 45
    sig = np.sin(2*np.pi*np.cumsum(f)/SR)
    return vol * sig * np.exp(-t*13)

def snare(vol=0.35):
    n = int(0.16*SR); t = np.arange(n)/SR
    noise = rng.normal(0, 1, n)
    sig = 0.7*noise + 0.5*np.sin(2*np.pi*190*t)
    return vol * sig * np.exp(-t*26)

def hat(vol=0.12, dur=0.05):
    n = int(dur*SR)
    noise = rng.normal(0, 1, n)
    noise = np.diff(np.concatenate([[0], noise]))  # 高通化
    return vol * noise * np.exp(-np.arange(n)/SR*70)

def click(vol=0.06):
    n = int(0.03*SR)
    noise = rng.normal(0,1,n)
    return vol*noise*np.exp(-np.arange(n)/SR*150)

def reverb(sig, mix=0.22, decay=0.28):
    """简易卷积混响：指数衰减噪声当 IR（FFT 快速卷积）"""
    ir_len = int(0.85*SR)
    ir = rng.normal(0,1,ir_len) * np.exp(-np.arange(ir_len)/SR/decay)
    ir /= np.abs(ir).sum()**0.5
    n = len(sig)
    nfft = 1 << int(np.ceil(np.log2(n + ir_len)))
    wet = np.fft.irfft(np.fft.rfft(sig, nfft) * np.fft.rfft(ir, nfft), nfft)[:n]
    return (1-mix)*sig + mix*wet*1.6

def mix_dur(buf, dur):
    n = int(dur*SR)
    if len(buf) < n: return np.pad(buf, (0, n-len(buf)))
    return buf[:n]

class Track:
    def __init__(self, bpm, bars, beats_per_bar=4):
        self.bpm = bpm
        self.beat = 60.0/bpm
        self.bars = bars
        self.dur = bars*beats_per_bar*self.beat
        self.buf = np.zeros(int(self.dur*SR)+SR)

    def add(self, sig, at, gain=1.0):
        i = int(at*SR)
        if i >= len(self.buf): return
        j = min(len(self.buf), i+len(sig))
        self.buf[i:j] += sig[:j-i]*gain

    def render(self, rev=0.22, normalize=0.9):
        s = reverb(self.buf, rev)
        # 首尾淡入淡出
        f = int(0.04*SR)
        s[:f] *= np.linspace(0,1,f)
        e = int(0.7*SR)
        if len(s) > e: s[-e:] *= np.linspace(1,0,e)
        peak = np.abs(s).max() or 1
        return s/peak*normalize

def save(sig, name, bitrate='96k'):
    wav = tempfile.mktemp(suffix='.wav')
    pcm = (np.clip(sig, -1, 1)*32767).astype('<i2')
    import wave
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    out = os.path.join(OUT, name)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',wav,'-b:a',bitrate,'-ac','1',out], check=True)
    os.remove(wav)
    print(f'  ✓ {name}  {os.path.getsize(out)/1024:.0f} KB  {len(sig)/SR:.1f}s')

# ============================ 各情绪曲目 ============================

def happy_upbeat():                      # 明亮大调，♩=126
    T = Track(126, 8)
    prog = [['C4','E4','G4'], ['G3','B3','D4'], ['A3','C4','E4'], ['F3','A3','C4']]
    roots = ['C2','G1','A1','F1']
    for b in range(8):
        t0 = b*4*T.beat
        ch = prog[b % 4]; rt = roots[b % 4]
        # 切分钢琴伴奏
        for i, off in enumerate([0, 1.5, 2, 3.5]):
            T.add(pluck(hz(ch[i % 3]), 0.55, .40, .8), t0+off*T.beat)
            T.add(pluck(hz(ch[i % 3])*2, 0.35, .16, .9), t0+off*T.beat)
        for beat in range(4):
            T.add(bass(hz(rt), T.beat*.9, .45), t0+beat*T.beat)
            if beat in (0,2): T.add(kick(.85), t0+beat*T.beat)
            if beat in (1,3): T.add(snare(.30), t0+beat*T.beat)
            T.add(hat(.11), t0+beat*T.beat)
            T.add(hat(.07), t0+(beat+.5)*T.beat)
    # 主旋律
    mel = [('E5',1),('G5',1),('A5',1.5),('G5',.5),('E5',1),
           ('D5',1),('E5',1),('C5',2),
           ('A4',1),('C5',1),('D5',1.5),('E5',.5),('G5',1),
           ('F5',1),('E5',1),('C5',2)]
    t = 8*T.beat
    for nm, d in mel:
        T.add(pluck(hz(nm), d*T.beat*1.05, .34, .75), t); t += d*T.beat
    return T.render(.20)

def happy_synth():                       # 合成器流行，♩=118
    T = Track(118, 8)
    prog = [['F3','A3','C4'], ['C3','E3','G3'], ['D3','F3','A3'], ['A#2','D3','F3']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        for k in ch:
            T.add(pad(hz(k), 4*T.beat, .17), t0)
        for beat in range(4):
            T.add(kick(.8), t0+beat*T.beat)
            T.add(hat(.10), t0+(beat+.5)*T.beat)
            if beat % 2 == 1: T.add(snare(.26), t0+beat*T.beat)
        for off in [0,.75,1.5,2.25,3,3.75]:
            T.add(pluck(hz(ch[0])*2, .3, .14, .95), t0+off*T.beat)
    lead = ['C5','E5','G5','E5','A5','G5','E5','D5']*2
    t = 4*T.beat
    for i, nm in enumerate(lead):
        T.add(pluck(hz(nm), T.beat*.95, .26, .9), t); t += T.beat
    return T.render(.25)

def calm_piano():                        # 舒缓钢琴，♩=68
    T = Track(68, 8)
    prog = [['C3','G3','E4'], ['A2','E3','C4'], ['F2','C3','A3'], ['G2','D3','B3']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        T.add(pluck(hz(ch[0]), 3.5*T.beat, .34, .45), t0)
        T.add(pluck(hz(ch[1]), 3*T.beat, .22, .40), t0+.05)
        T.add(pluck(hz(ch[2]), 2.5*T.beat, .18, .38), t0+.1)
        for off in [1, 2.5, 3.5]:
            T.add(pluck(hz(ch[2])*2, 1.2*T.beat, .13, .4), t0+off*T.beat)
    mel = [('E5',2),('D5',1),('C5',1),('A4',3),('G4',1),
           ('A4',2),('C5',1),('D5',1),('E5',3),('D5',1)]
    t = 4*T.beat
    for nm, d in mel:
        T.add(pluck(hz(nm), d*T.beat, .30, .42), t); t += d*T.beat
    return T.render(.34, .85)

def calm_ambient():                      # 氛围铺底，♩=56
    T = Track(56, 6, 4)
    pads = [['D3','A3','F4'], ['A#2','F3','D4'], ['F2','C3','A3'], ['C3','G3','E4']]
    for b in range(6):
        t0 = b*4*T.beat; ch = pads[b % 4]
        for k in ch: T.add(pad(hz(k), 4.2*T.beat, .20), t0)
        T.add(bass(hz(ch[0])/2, 4*T.beat, .22), t0)
        T.add(click(.05), t0)
    bells = ['D5','F5','A5','F5','C5','E5','G5','A5','F5','D5','C5','A4']
    t = 2*T.beat
    for i, nm in enumerate(bells):
        T.add(pluck(hz(nm), 2.4*T.beat, .16, .5), t); t += 1.6*T.beat
    return T.render(.45, .8)

def sad_strings():                       # 小调弦乐，♩=64
    T = Track(64, 8)
    prog = [['A2','E3','C4'], ['F2','C3','A3'], ['C3','G3','E4'], ['G2','D3','B3']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        for k in ch: T.add(pad(hz(k), 4.1*T.beat, .21), t0)
        T.add(bass(hz(ch[0])/2, 4*T.beat, .26), t0)
    mel = [('E5',3),('C5',1),('B4',2),('A4',2),('C5',3),('E5',1),('D5',4),('B4',4)]
    t = 2*T.beat
    for nm, d in mel:
        T.add(pad(hz(nm), d*T.beat, .13), t)
        T.add(pluck(hz(nm), d*T.beat, .16, .3), t)
        t += d*T.beat
    return T.render(.40, .85)

def sad_musicbox():                      # 八音盒，♩=76
    T = Track(76, 8)
    prog = [['D3','A3','F4'], ['A#2','F3','D4'], ['G2','D3','A#3'], ['A2','E3','C#4']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        T.add(pluck(hz(ch[0])*2, 4*T.beat, .18, .35), t0)
        T.add(pluck(hz(ch[2]), 4*T.beat, .14, .35), t0+.02)
        for off in [0,2]:
            T.add(bass(hz(ch[0])/2, T.beat*1.6, .30), t0+off*T.beat)
    mel = [('A5',1),('F5',1),('D5',1),('F5',1),('A#5',1),('A5',1),('F5',2),
           ('G5',1),('D5',1),('A#4',2),('A4',1),('C#5',1),('E5',2)]
    t = T.beat
    for nm, d in mel:
        T.add(pluck(hz(nm), d*T.beat*1.3, .22, .55), t); t += d*T.beat
    return T.render(.42, .8)

def angry_rock():                        # 强力和弦，♩=150
    T = Track(150, 8)
    roots = ['E2','G2','A2','C3']
    for b in range(8):
        t0 = b*4*T.beat; rt = roots[b % 4]
        for beat in range(4):
            for off in (0, .5):
                tt = t0+(beat+off)*T.beat
                T.add(bass(hz(rt), T.beat*.45, .55), tt)
                T.add(pluck(hz(rt)*2, T.beat*.4, .30, 1.0), tt)
                T.add(pluck(hz(rt)*3, T.beat*.4, .20, 1.0), tt)
            T.add(kick(.9), t0+beat*T.beat)
            if beat in (1,3): T.add(snare(.42), t0+beat*T.beat)
            T.add(hat(.10), t0+(beat+.5)*T.beat)
    lead = ['E5','E5','G5','E5','A5','G5','E5','D5']
    t = 4*T.beat
    for i, nm in enumerate(lead*2):
        T.add(pluck(hz(nm), T.beat*.7, .26, 1.0), t); t += T.beat
    return T.render(.16, .95)

def angry_dark():                        # 暗色合成器，♩=140
    T = Track(140, 8)
    roots = ['D2','D2','F2','C2']
    for b in range(8):
        t0 = b*4*T.beat; rt = roots[b % 4]
        T.add(pad(hz(rt)*2, 4*T.beat, .18), t0)
        T.add(pad(hz(rt)*3, 4*T.beat, .10), t0)
        for beat in range(4):
            T.add(bass(hz(rt), T.beat*.85, .55), t0+beat*T.beat)
            T.add(kick(.85), t0+beat*T.beat)
            T.add(hat(.09), t0+(beat+.5)*T.beat)
            if beat == 2: T.add(snare(.34), t0+beat*T.beat)
    t = 2*T.beat
    for i in range(12):
        nm = ['D5','F5','D5','C5','A#4','D5'][i % 6]
        T.add(pluck(hz(nm), T.beat*.8, .20, .95), t); t += T.beat
    return T.render(.22, .92)

def fear_tense():                        # 悬疑，♩=90，小二度
    T = Track(90, 8)
    base = ['D3','D#3','A2','A#2']
    for b in range(8):
        t0 = b*4*T.beat
        T.add(pad(hz(base[b % 4]), 4*T.beat, .20), t0)
        T.add(pad(hz(base[b % 4])*2, 4*T.beat, .11), t0)
        T.add(bass(hz(base[b % 4])/2, 4*T.beat, .32), t0)
        T.add(click(.14), t0); T.add(click(.10), t0+2*T.beat)
    motif = ['A5','A#5','A5','G#5','A5','A#5','D6','C#6']
    t = 2*T.beat
    for i, nm in enumerate(motif):
        T.add(pluck(hz(nm), T.beat*1.6, .18+0.01*i, .35), t); t += 1.4*T.beat
    return T.render(.50, .85)

def fear_pulse():                        # 心跳脉冲，♩=100
    T = Track(100, 8)
    root = 'C3'
    for b in range(8):
        t0 = b*4*T.beat
        T.add(pad(hz(root), 4*T.beat, .16), t0)
        T.add(pad(hz('C#4'), 4*T.beat, .08), t0)
        for beat in range(4):
            T.add(kick(.55 if beat % 2 else .95), t0+beat*T.beat)
            T.add(kick(.35), t0+(beat+.28)*T.beat)
        if b % 2: T.add(pluck(hz('G5'), T.beat*2, .15, .9), t0)
    return T.render(.44, .85)

def surprise_arp():                      # 上行琶音，♩=132
    T = Track(132, 8)
    scale = ['C4','D4','E4','G4','A4','C5','D5','E5']
    for b in range(8):
        t0 = b*4*T.beat
        for i in range(8):
            nm = scale[(i+b) % len(scale)]
            T.add(pluck(hz(nm), T.beat*.55, .24, .95), t0+i*T.beat*.5)
        T.add(kick(.7), t0)
        T.add(pluck(hz('C3'), T.beat*2, .30, .5), t0)
        T.add(snare(.24), t0+2*T.beat)
    return T.render(.28, .9)

def surprise_stab():                     # 断奏重音，♩=144
    T = Track(144, 8)
    chords = [['C4','E4','G4'],['D4','F4','A4'],['E4','G4','B4'],['C4','E4','G4']]
    for b in range(8):
        t0 = b*4*T.beat; ch = chords[b % 4]
        for beat, d in [(0,1.5),(2,.5),(3,.5)]:
            for k in ch:
                T.add(pluck(hz(k), d*T.beat, .26, .95), t0+beat*T.beat)
            T.add(bass(hz(ch[0])/2, d*T.beat*.8, .40), t0+beat*T.beat)
        T.add(kick(.8), t0); T.add(snare(.3), t0+2*T.beat)
    t = 2*T.beat
    for nm in ['C5','G5','E5','C6','G5','C5']:
        T.add(pluck(hz(nm), T.beat*.9, .28, .95), t); t += T.beat
    return T.render(.28, .9)

def neutral_lofi():                      # Lo-Fi 循环，♩=84
    T = Track(84, 8)
    prog = [['C3','E3','B3'], ['A2','C3','G3'], ['D3','F3','A3'], ['G2','B2','F3']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        for k in ch: T.add(pluck(hz(k), 4*T.beat, .20, .30), t0)
        for beat in range(4):
            T.add(bass(hz(ch[0])/2, T.beat*.9, .34), t0+beat*T.beat)
            if beat in (0,2): T.add(kick(.66), t0+beat*T.beat)
            if beat == 2: T.add(snare(.20), t0+beat*T.beat)
            T.add(hat(.06), t0+(beat+.5)*T.beat)
    return T.render(.32, .85)

def neutral_guitar():                    # 分解和弦，♩=88
    T = Track(88, 8)
    prog = [['G2','D3','G3','B3'], ['E2','B2','E3','G3'], ['C3','G3','C4','E4'], ['D3','A3','D4','F#4']]
    for b in range(8):
        t0 = b*4*T.beat; ch = prog[b % 4]
        for i in range(8):
            nm = ch[i % 4]
            T.add(pluck(hz(nm), T.beat*1.1, .24, .58), t0+i*T.beat*.5)
        T.add(bass(hz(ch[0])/2, 4*T.beat, .28), t0)
    return T.render(.30, .85)

TASKS = [
    ('local-happy-1.mp3', 'Sunlit Steps', happy_upbeat),
    ('local-happy-2.mp3', 'Neon Bounce', happy_synth),
    ('local-calm-1.mp3',  'Quiet Window', calm_piano),
    ('local-calm-2.mp3',  'Drifting Light', calm_ambient),
    ('local-sad-1.mp3',   'Rain on Glass', sad_strings),
    ('local-sad-2.mp3',   'Faded Music Box', sad_musicbox),
    ('local-angry-1.mp3', 'Red Riot', angry_rock),
    ('local-angry-2.mp3', 'Midnight Engine', angry_dark),
    ('local-fear-1.mp3',  'Hollow Corridor', fear_tense),
    ('local-fear-2.mp3',  'Heartbeat Room', fear_pulse),
    ('local-surprised-1.mp3', 'Upside Ascent', surprise_arp),
    ('local-surprised-2.mp3', 'Static Spark', surprise_stab),
    ('local-neutral-1.mp3', 'Coffee Loop', neutral_lofi),
    ('local-neutral-2.mp3', 'Wooden Strings', neutral_guitar),
]

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(__file__), '..', 'audio'))
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
print('输出目录:', OUT)
for name, title, fn in TASKS:
    print(f'合成 {title} → {name}')
    save(fn(), name)
print('全部完成')
