"""合并 素材指标 + 抽帧 + OCR + 逐条画面标注 → data/report.json 与分析 Excel。"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
XLSX = BASE / '箱包鞋靴_10.10_ECO创意分析.xlsx'
FRAMEWORKS = ['剧情型', '人设口播型', '产品展示型', '街拍穿搭型', '测评实验型', '工厂溯源型', '情感送礼型', '直播切片型', '图文型']
PF_BUCKETS = ['0–3秒', '3–10秒', '10–30秒', '30秒内未出现']
SKIP_BINS = [(0, 40, '<40%'), (40, 50, '40–50%'), (50, 60, '50–60%'), (60, 70, '60–70%'), (70, 80, '70–80%'), (80, 101, '≥80%')]
PLAY_BINS = [(0, 4, '<4s'), (4, 6, '4–6s'), (6, 8, '6–8s'), (8, 12, '8–12s'), (12, 20, '12–20s'), (20, 1e9, '≥20s')]


def load(name, default=None):
    p = DATA / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def annotations():
    out = {}
    for p in sorted((DATA / 'annotations').glob('*.json')):
        for a in json.loads(p.read_text(encoding='utf-8')):
            out[a['key']] = a
    return out


def pf_bucket(v):
    if v is None:
        return '30秒内未出现'
    return '0–3秒' if v <= 3 else '3–10秒' if v <= 10 else '10–30秒'


def frame_table():
    mats = load('materials.json')
    media = load('media.json', {})
    ocr = load('ocr.json', {})
    ann = annotations()
    rows = []
    for m in mats:
        me = media.get(m['key'], {})
        a = ann.get(m['key'], {})
        frames = me.get('frames', [])
        texts = [ocr.get(f['path'], '') for f in frames]
        nf = len(frames)
        r = {**m, 'kind': me.get('kind'), 'media_ok': bool(me.get('ok')), 'media_error': me.get('error'),
             'duration': me.get('duration'), 'cuts15': me.get('cuts15'), 'cuts30': me.get('cuts30'),
             'frames': frames, 'ocr': texts, 'ocr_chars': sum(len(t.replace(' / ', '')) for t in texts) / nf if nf else None,
             'reviewed': bool(a) and a.get('framework') in FRAMEWORKS}
        for k in ['framework', 'type', 'person', 'face3', 'product_first', 'product_frames', 'hook', 'hook_text', 'price', 'story', 'confidence']:
            r[k] = a.get(k)
        r['product_share'] = (a.get('product_frames') / nf) if r['reviewed'] and nf and a.get('product_frames') is not None else None
        r['pf_bucket'] = pf_bucket(a.get('product_first')) if r['reviewed'] else None
        rows.append(r)
    d = pd.DataFrame(rows)
    d['ok'] = d[['eco', 'skip', 'play']].notna().all(axis=1)
    for cat in ['鞋靴', '箱包']:
        s = d[(d.category == cat) & d.ok]
        em, cm, tm = s.eco.median(), s.cost.median(), s.ctr.median()
        idx = s.index
        d.loc[idx, 'eco_hi'] = s.eco >= em
        d.loc[idx, 'quad_cost'] = np.where(s.eco >= em, '高eco', '低eco') + np.where(s.cost >= cm, '·高消耗', '·低消耗')
        d.loc[idx, 'quad_ctr'] = np.where(s.eco >= em, '高eco', '低eco') + np.where(s.ctr >= tm, '·高CTR', '·低CTR')
    return d


def ols(s):
    x = np.column_stack([np.ones(len(s)), s.skip, s.play])
    y = s.eco.to_numpy(float)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    pred = np.zeros(len(y))
    for fold in np.array_split(np.random.default_rng(42).permutation(len(y)), 5):
        tr = np.ones(len(y), bool)
        tr[fold] = False
        pred[fold] = x[fold] @ np.linalg.lstsq(x[tr], y[tr], rcond=None)[0]
    zs = (s[['skip', 'play']] - s[['skip', 'play']].mean()) / s[['skip', 'play']].std()
    bz = np.linalg.lstsq(np.column_stack([np.ones(len(s)), zs]), (y - y.mean()) / y.std(), rcond=None)[0]
    return {'n': len(s), 'intercept': b[0], 'b_skip': b[1], 'b_play': b[2],
            'r2': 1 - ((y - x @ b) ** 2).sum() / sst, 'cv_r2': 1 - ((y - pred) ** 2).sum() / sst,
            'beta_skip': bz[1], 'beta_play': bz[2]}


def metric_agg(g, total_cost):
    return {'n': len(g), 'cost_sum': g.cost.sum(), 'cost_share': g.cost.sum() / total_cost if total_cost else None,
            'cost_mean': g.cost.mean(), 'eco': g.eco.mean(), 'eco_w': np.average(g.eco, weights=g.cost) if g.cost.sum() else None,
            'skip': g.skip.mean(), 'play': g.play.mean(), 'ctr': g.ctr.mean(),
            'face3': g.face3.astype(float).mean() if g.face3.notna().any() else None,
            'pf03': (g.pf_bucket == '0–3秒').mean() if g.pf_bucket.notna().any() else None,
            'pf_none': (g.pf_bucket == '30秒内未出现').mean() if g.pf_bucket.notna().any() else None,
            'product_share': g.product_share.mean(), 'cuts30': g.cuts30.mean(), 'price': g.price.astype(float).mean() if g.price.notna().any() else None,
            'duration_med': g.duration.median(), 'keys': list(g.sort_values('cost', ascending=False).key)}


def group_table(s, col, order=None, min_n=1):
    total = s.cost.sum()
    out = []
    keys = order or [k for k, _ in Counter(s[col].dropna()).most_common()]
    for k in keys:
        g = s[s[col] == k]
        if len(g) >= min_n:
            out.append({'name': k, **metric_agg(g, total)})
    return out


def profile(s, mask):
    g, rest = s[mask], s[~mask]
    def share(frame, col, val):
        return (frame[col] == val).mean() if len(frame) else None
    feats = []
    for fw in FRAMEWORKS:
        feats.append({'feature': '框架=' + fw, 'group': share(g, 'framework', fw), 'rest': share(rest, 'framework', fw)})
    for b in PF_BUCKETS:
        feats.append({'feature': '产品首现=' + b, 'group': share(g, 'pf_bucket', b), 'rest': share(rest, 'pf_bucket', b)})
    for col, label in [('face3', '前3秒真人露脸'), ('price', '30秒内出现价格/优惠')]:
        feats.append({'feature': label, 'group': g[col].astype(float).mean(), 'rest': rest[col].astype(float).mean()})
    for h, _ in Counter(s.hook.dropna()).most_common():
        feats.append({'feature': '开场钩子=' + h, 'group': share(g, 'hook', h), 'rest': share(rest, 'hook', h)})
    for f in feats:
        f['diff'] = (f['group'] or 0) - (f['rest'] or 0)
    return {'n': len(g), 'metrics': {k: v for k, v in metric_agg(g, s.cost.sum()).items() if k != 'keys'} if len(g) else None,
            'rest_metrics': {k: v for k, v in metric_agg(rest, s.cost.sum()).items() if k != 'keys'} if len(rest) else None,
            'features': sorted(feats, key=lambda f: -abs(f['diff'])), 'keys': list(g.sort_values('cost', ascending=False).key)}


def spearman(a, b):
    return a.rank().corr(b.rank())


def scope_stats(d, cat):
    s = d[d.ok & d.reviewed] if cat == '全部' else d[d.ok & d.reviewed & (d.category == cat)]
    allv = d[d.ok] if cat == '全部' else d[d.ok & (d.category == cat)]
    st = {'n_rows': int(len(d) if cat == '全部' else (d.category == cat).sum()), 'n_valid': len(allv), 'n_reviewed': len(s),
          'cost_sum': allv.cost.sum(), 'eco': allv.eco.mean(), 'eco_w': np.average(allv.eco, weights=allv.cost),
          'skip': allv.skip.mean(), 'play': allv.play.mean(), 'ctr': allv.ctr.mean(),
          'eco_median': allv.eco.median(), 'cost_median': allv.cost.median(), 'ctr_median': allv.ctr.median()}
    st['corr'] = {m: {'pearson': allv.eco.corr(allv[m]), 'spearman': spearman(allv.eco, allv[m])} for m in ['skip', 'play', 'cost', 'ctr', 'cvr']}
    st['corr_skip_play'] = spearman(allv.skip, allv.play)
    st['ols'] = ols(allv)
    st['skip_bins'] = [{'label': l, 'n': int(((allv.skip >= a) & (allv.skip < b)).sum()), 'eco': allv.eco[(allv.skip >= a) & (allv.skip < b)].mean()} for a, b, l in SKIP_BINS]
    st['play_bins'] = [{'label': l, 'n': int(((allv.play >= a) & (allv.play < b)).sum()), 'eco': allv.eco[(allv.play >= a) & (allv.play < b)].mean()} for a, b, l in PLAY_BINS]
    st['frameworks'] = group_table(s, 'framework', [f for f in FRAMEWORKS if (s.framework == f).any()])
    st['types'] = group_table(s, 'type', min_n=3)
    st['persons'] = group_table(s, 'person')
    st['hooks'] = group_table(s, 'hook')
    st['pf'] = group_table(s, 'pf_bucket', [b for b in PF_BUCKETS if (s.pf_bucket == b).any()])
    ps = s.copy()
    ps['ps_bucket'] = pd.cut(ps.product_share, [-0.01, 0.3, 0.6, 0.8, 1.01], labels=['≤30%', '30–60%', '60–80%', '>80%']).astype(str)
    st['product_share'] = group_table(ps, 'ps_bucket', ['≤30%', '30–60%', '60–80%', '>80%'])
    st['quad_cost'] = {q: profile(s, s.quad_cost == q) for q in ['高eco·高消耗', '高eco·低消耗', '低eco·高消耗', '低eco·低消耗']}
    st['quad_ctr'] = {q: profile(s, s.quad_ctr == q) for q in ['高eco·高CTR', '高eco·低CTR', '低eco·高CTR', '低eco·低CTR']}
    st['eco_hi'] = profile(s, s.eco_hi == True)
    return st


CONCL_TITLES = [
    ('headline', '01 赛道判断 · 一句话判断'), ('fw', '01 赛道判断 · 创意大类结构'), ('persons', '01 赛道判断 · 人物出镜方式'),
    ('pf', '01 赛道判断 · 产品首次成为主体的时间'), ('product_share', '01 赛道判断 · 讲品帧占比'), ('hooks', '01 赛道判断 · 开场钩子'),
    ('driver', '02 ECO 驱动验证 · 相关与回归'), ('skip_bins', '02 ECO 驱动验证 · 按 3s 快滑率分档'), ('play_bins', '02 ECO 驱动验证 · 按平均播放时长分档'),
    ('quad_cost', '03 四象限 · ECO × 消耗'), ('cost_traits', '03 四象限 · 高 ECO + 高消耗特征'),
    ('quad_ctr', '03 四象限 · ECO × CTR'), ('ctr_traits', '03 四象限 · 高 ECO + 高 CTR 特征'),
    ('quadrant', '04 素材类型 · 四象限图'), ('types_hi', '04 素材类型 · 高 ECO 类型表'), ('types_lo', '04 素材类型 · 低 ECO 类型表')]


def fe(v, n=0):
    return f'{v:,.{n}f}'


def sg(v, n=1):
    return f'{v:+.{n}f}'


def corr_word(r):
    a = abs(r)
    if a < .2:
        return '几乎无相关'
    return ('较弱' if a < .4 else '中等' if a < .6 else '较强' if a < .8 else '很强') + ('正相关' if r > 0 else '负相关')


def _ok(groups, min_n=5):
    return [g for g in groups if g['n'] >= min_n and pd.notna(g.get('eco'))]


def _pool(gs, k):
    return sum(g[k] * g['n'] for g in gs) / sum(g['n'] for g in gs)


def group_concl(groups, s, label, trend=None, min_n=5):
    avg, tn = s['eco'], s['n_reviewed']
    ok = _ok(groups, min_n)
    if len(ok) < 2:
        return [f'≥{min_n} 条样本的{label}不足 2 类，样本偏少，暂不下结论。']

    def desc(g):
        return (f"「{g['name']}」ECO {fe(g['eco'])}（较均值 {sg((g['eco'] / avg - 1) * 100)}%），3s 快滑 {g['skip']:.0f}%、均播 {g['play']:.1f}s，"
                f"占素材 {g['n'] / tn * 100:.0f}%、消耗 {(g['cost_share'] or 0) * 100:.0f}%")
    best, worst = max(ok, key=lambda g: g['eco']), min(ok, key=lambda g: g['eco'])
    big = max(groups, key=lambda g: g['n'])
    out = [f'ECO 最高：{desc(best)}。', f'ECO 最低：{desc(worst)}。',
           f"{label}中数量最多的是「{big['name']}」（{big['n']} 条），ECO {fe(big['eco'])}，{'高于' if big['eco'] >= avg else '低于'}均值 {fe(avg)}；最高与最低相差 {fe(best['eco'] - worst['eco'])} 分（≥{min_n} 条的分组之间）。"]
    hi, lo = [g for g in ok if g['eco'] >= avg], [g for g in ok if g['eco'] < avg]
    if hi and lo:
        hs, hp, ls, lp = _pool(hi, 'skip'), _pool(hi, 'play'), _pool(lo, 'skip'), _pool(lo, 'play')
        tail = 'ECO 高低与留存表现（快滑低、播放长）一致。' if hs < ls and hp > lp else '留存表现与 ECO 高低并不完全同步。'
        out.append(f'高于均值的 {len(hi)} 类平均快滑 {hs:.0f}%、均播 {hp:.1f}s；低于均值的 {len(lo)} 类快滑 {ls:.0f}%、均播 {lp:.1f}s，{tail}')
    if trend:
        a, z = ok[0], ok[-1]
        out.append(f"沿「{trend}」从「{a['name']}」到「{z['name']}」，ECO 由 {fe(a['eco'])} 变为 {fe(z['eco'])}（{sg(z['eco'] - a['eco'], 0)} 分），"
                   f"{'整体越往后越低' if z['eco'] < a['eco'] else '整体越往后越高'}。")
    return out


def bin_concl(bins, coef, name, per, expect):
    ok = [b for b in bins if b['n'] >= 5 and pd.notna(b['eco'])]
    if len(ok) < 2:
        return ['有效档位不足，暂不下结论。']
    a, z = ok[0], ok[-1]
    steps = list(zip(ok, ok[1:]))
    good = sum(1 for x, y in steps if (y['eco'] - x['eco']) * expect > 0)
    word = '下降' if expect < 0 else '上升'
    trend = f'{len(steps)} 个相邻档中 {good} 个呈{word}方向，{"单调" if good == len(steps) else "大体"}{word}' if good * 2 > len(steps) else f'{len(steps)} 个相邻档中仅 {good} 个呈{word}方向，趋势不明显'
    out = [f"{name}最低档「{a['label']}」（{a['n']} 条）平均 ECO {fe(a['eco'])}，最高档「{z['label']}」（{z['n']} 条）{fe(z['eco'])}，相差 {fe(abs(a['eco'] - z['eco']))} 分；{trend}。",
           f"回归估算：{name}每{'升高' if per[1] == 'pt' else '增加'} {per[0]} {per[1]}，ECO 约{'下降' if coef < 0 else '上升'} {fe(abs(coef * per[0]))} 分。"]
    if any(b['n'] < 10 for b in bins if pd.notna(b['eco'])):
        out.append('n<10 的档位（图中浅色柱）波动大，仅作参考。')
    return out


def driver_concl(s):
    c, o = s['corr'], s['ols']
    sk, pl = c['skip']['spearman'], c['play']['spearman']
    holds = sk < -.3 and pl > .3
    out = [f"快滑率与 ECO：{corr_word(sk)}（Spearman {sk:.2f}）；均播与 ECO：{corr_word(pl)}（{pl:.2f}）。“快滑越低、播放越长，ECO 越高”在 {s['n_valid']} 条素材上{'成立' if holds else '只部分成立'}。",
           f"两项合计可解释 ECO 差异的 {o['r2'] * 100:.0f}%（5 折验证 {o['cv_r2'] * 100:.0f}%，{'二者接近，没有明显过拟合' if abs(o['r2'] - o['cv_r2']) < .05 else '差距较大，存在过拟合'}），其余约 {100 - o['r2'] * 100:.0f}% 来自表内没有的因素（互动、负反馈等）。",
           f"标准化影响力：快滑 {abs(o['beta_skip']):.2f} vs 均播 {abs(o['beta_play']):.2f}，{'快滑率的权重更大，优先压低前 3 秒流失' if abs(o['beta_skip']) > abs(o['beta_play']) else '均播的权重更大，优先拉长后续停留'}；两者本身相关（ρ={s['corr_skip_play']:.2f}），不是完全独立。"]
    cc, ct = c['cost']['spearman'], c['ctr']['spearman']
    tail = '提 ECO 不会天然带来放量，需要同时盯消耗和 CTR。' if abs(cc) < .3 else ('ECO 高的素材更容易放量。' if cc > 0 else 'ECO 高的素材反而消耗偏低，注意量级取舍。')
    out.append(f'ECO 与消耗：{corr_word(cc)}（ρ={cc:.2f}）；ECO 与 CTR：{corr_word(ct)}（ρ={ct:.2f}）。{tail}')
    return out


def quad_concl(q, keys, dim, n_total, rho):
    hh, hl, lh, ll = keys
    m = {k: q[k]['metrics'] for k in keys}
    out = [f"已复核素材分布：{hh} {q[hh]['n']} 条、{hl} {q[hl]['n']} 条、{lh} {q[lh]['n']} 条、{ll} {q[ll]['n']} 条（共 {n_total} 条，各品类按中位数切分）。"]
    if m[hh] and m[lh]:
        a, b = m[hh], m[lh]
        out.append(f"「{hh}」与「{lh}」同样{dim}偏高，但前者快滑 {a['skip']:.0f}% vs {b['skip']:.0f}%、均播 {a['play']:.1f}s vs {b['play']:.1f}s、"
                   f"前 3 秒露脸 {(a['face3'] or 0) * 100:.0f}% vs {(b['face3'] or 0) * 100:.0f}%——差距来自留存，而不是{dim}。")
    if dim == '消耗' and m[lh]:
        out.append(f"「{lh}」{q[lh]['n']} 条合计消耗 ¥{m[lh]['cost_sum'] / 1e4:.1f} 万，占已复核素材消耗 {(m[lh]['cost_share'] or 0) * 100:.0f}%，"
                   f"平均 ECO 仅 {fe(m[lh]['eco'])}，是“保量提 ECO”最直接的改造池。")
        if m[hh]:
            out.append(f"「{hh}」{q[hh]['n']} 条占已复核素材消耗 {(m[hh]['cost_share'] or 0) * 100:.0f}%（ECO×消耗：{corr_word(rho)}，ρ={rho:.2f}），"
                       + ('高 ECO 与高消耗可以兼得。' if rho > -.15 else '整体上 ECO 高并不意味着更放量，放量还取决于出价、预算等。'))
    if dim == 'CTR' and m[hl]:
        out.append(f"「{hl}」{q[hl]['n']} 条 ECO 高（{fe(m[hl]['eco'])}）但 CTR 仅 {m[hl]['ctr']:.2f}%（对比「{hh}」{m[hh]['ctr']:.2f}%），"
                   f"属于“留得住、点不动”，在不降低留存的前提下提升点击是主要空间（ECO×CTR：{corr_word(rho)}，ρ={rho:.2f}）。" if m[hh] else f"「{hl}」{q[hl]['n']} 条 ECO 高但 CTR 偏低。")
    return out


def trait_concl(pr, name):
    fs = [x for x in pr['features'] if abs(x['diff']) >= .04]
    if not pr['n'] or not fs:
        return [f'「{name}」样本不足或与其余素材无明显差异（差值均 <4pt）。']
    up, dn = [x for x in fs if x['diff'] > 0][:4], [x for x in fs if x['diff'] < 0][:3]
    t = lambda x: f"{x['feature']}（{x['diff'] * 100:+.0f}pt）"
    out = [f"「{name}」共 {pr['n']} 条，相对其余素材"
           + (f"更集中于：{'、'.join(t(x) for x in up)}" if up else '没有明显更集中的特征')
           + (f"；更少出现：{'、'.join(t(x) for x in dn)}。" if dn else '。')]
    return out


def type_concl(types, avg, s, hi):
    sel = sorted([t for t in types if (t['eco'] >= avg) == hi], key=lambda t: -t['eco'] if hi else t['eco'])
    tag = '高于' if hi else '低于'
    if not sel:
        return [f'没有{tag}品类均值的类型（≥3 条）。']
    n = sum(t['n'] for t in sel)
    cs = sum(t['cost_sum'] for t in sel) / sum(t['cost_sum'] for t in types) * 100
    top = '、'.join(f"{t['name']}（{fe(t['eco'])}）" for t in sel[:3])
    out = [f"{tag}品类平均 ECO（{fe(avg)}）的类型共 {len(sel)} 个、{n} 条素材，占类型合计消耗 {cs:.0f}%；{'ECO 最高' if hi else 'ECO 最低'}的三类：{top}。",
           f"这些类型平均播放 {min(t['play'] for t in sel):.1f}–{max(t['play'] for t in sel):.1f}s、3s 快滑 {min(t['skip'] for t in sel):.0f}–{max(t['skip'] for t in sel):.0f}%"
           f"（品类均值 快滑 {s['skip']:.0f}% / 均播 {s['play']:.1f}s）。"]
    stable = [t for t in sel if t['n'] >= 10]
    if stable:
        out.append('样本 ≥10 条、结论更稳的类型：' + '、'.join(f"{t['name']}（{t['n']} 条，ECO {fe(t['eco'])}）" for t in sorted(stable, key=lambda t: -t['n'])[:4]) + '；n<5 的类型波动大，仅作参考。')
    big = max(sel, key=lambda t: t['cost_sum'])
    out.append(f"消耗最大的是「{big['name']}」（¥{big['cost_sum'] / 1e4:.1f} 万，ECO {fe(big['eco'])}），" + ('值得继续放大。' if hi else '是降低整体 ECO 拖累的优先改造对象。'))
    return out


def quadrant_data(st, d, cat):
    sub = d[d.reviewed] if cat == '全部' else d[d.reviewed & (d.category == cat)]
    fwmap = sub.groupby('type').framework.agg(lambda x: x.mode().iat[0]).to_dict()
    pts = [{'name': t['name'], 'framework': fwmap.get(t['name'], ''), 'n': t['n'], 'cost_sum': t['cost_sum'], 'eco': t['eco'],
            'skip': t['skip'], 'play': t['play'], 'ctr': t['ctr']} for t in st['types']]
    k = round(len(pts) / 3)
    for i, p in enumerate(sorted(pts, key=lambda p: -p['eco'])):
        p['tier'] = '高' if i < k else '低' if i >= len(pts) - k else '中'
    sr, pr = st['skip'], st['play']
    return {'points': pts, 'skip_ref': sr, 'play_ref': pr}


def quadrant_concl(qd):
    pts, sr, pr = qd['points'], qd['skip_ref'], qd['play_ref']
    if len(pts) < 3:
        return ['素材类型不足 3 个，暂不作图。']
    lab = lambda p: f"{p['framework']}-{p['name']}" if p['framework'] else p['name']
    tiers = {t: [p for p in pts if p['tier'] == t] for t in ['高', '中', '低']}
    tl = sorted([p for p in pts if p['skip'] < sr and p['play'] > pr], key=lambda p: -p['eco'])
    br = sorted([p for p in pts if p['skip'] >= sr and p['play'] <= pr], key=lambda p: p['eco'])
    mean = lambda ps, k: sum(p[k] for p in ps) / len(ps)
    out = [f"{len(pts)} 个素材类型（≥3 条）按 ECO 分三档：高 ECO {len(tiers['高'])} 种（ECO ≥ {fe(min(p['eco'] for p in tiers['高']))}）、中 {len(tiers['中'])} 种、低 {len(tiers['低'])} 种（ECO ≤ {fe(max(p['eco'] for p in tiers['低']))}）；虚线为品类均值（快滑 {sr:.1f}%、均播 {pr:.1f}s）。",
           f"左上“高 ECO 象限”（快滑低 + 播放长）有 {len(tl)} 种" + (f"：{'、'.join(lab(p) for p in tl[:5])}" if tl else '') + f"，其中高 ECO 档 {sum(p['tier'] == '高' for p in tl)} 种；"
           f"右下“低 ECO 象限”（快滑高 + 播放短）有 {len(br)} 种" + (f"：{'、'.join(lab(p) for p in br[:5])}" if br else '') + f"，其中低 ECO 档 {sum(p['tier'] == '低' for p in br)} 种。",
           f"高 ECO 档类型平均快滑 {mean(tiers['高'], 'skip'):.0f}% / 均播 {mean(tiers['高'], 'play'):.1f}s，低 ECO 档 {mean(tiers['低'], 'skip'):.0f}% / {mean(tiers['低'], 'play'):.1f}s，"
           f"{'三档之间留存差距清晰，快滑与均播能有效区分 ECO 档位。' if mean(tiers['高'], 'skip') < mean(tiers['低'], 'skip') and mean(tiers['高'], 'play') > mean(tiers['低'], 'play') else '三档之间留存差距不明显。'}"]
    big = max(pts, key=lambda p: p['cost_sum'])
    zone = '左上高 ECO 象限' if big['skip'] < sr and big['play'] > pr else '右下低 ECO 象限' if big['skip'] >= sr and big['play'] <= pr else '快滑低但播放短的左下区' if big['skip'] < sr else '快滑高但播放长的右上区'
    out.append(f"气泡最大（消耗最高）的是「{lab(big)}」：¥{big['cost_sum'] / 1e4:.1f} 万，占类型合计 {big['cost_sum'] / sum(p['cost_sum'] for p in pts) * 100:.0f}%，位于{zone}，ECO {fe(big['eco'])}（{big['tier']}档）。")
    return out


def conclusions(st, cat, bench, d):
    out = {}
    frs = _ok(st['frameworks'])
    avg = st['eco']
    hi = sorted([g for g in frs if g['eco'] >= avg], key=lambda g: -g['eco'])[:3]
    lo = sorted([g for g in frs if g['eco'] < avg], key=lambda g: g['eco'])[:3]
    nm = lambda gs: '、'.join(f"{g['name']}（{fe(g['eco'])}）" for g in gs) or '无'
    out['headline'] = [f"{cat}平均 ECO {fe(avg)}，较原表整体基准 {fe(bench['eco'])} {sg((avg / bench['eco'] - 1) * 100)}%（消耗加权 {fe(st['eco_w'])}）。"
                       f"创意大类里高于均值：{nm(hi)}；低于均值：{nm(lo)}。"
                       f"快滑率与 ECO 的 Spearman 为 {st['corr']['skip']['spearman']:.2f}、均播为 {st['corr']['play']['spearman']:.2f}。"]
    out['fw'] = group_concl(st['frameworks'], st, '创意大类')
    out['persons'] = group_concl(st['persons'], st, '人物出镜方式')
    out['pf'] = group_concl(st['pf'], st, '产品首现时间档', trend='产品首次成为画面主体的时间')
    out['product_share'] = group_concl(st['product_share'], st, '讲品帧占比档', trend='讲品帧占比')
    out['hooks'] = group_concl(st['hooks'], st, '开场钩子')
    out['driver'] = driver_concl(st)
    out['skip_bins'] = bin_concl(st['skip_bins'], st['ols']['b_skip'], '快滑率', (10, 'pt'), -1)
    out['play_bins'] = bin_concl(st['play_bins'], st['ols']['b_play'], '平均播放时长', (1, '秒'), 1)
    qc, qt, n = st['quad_cost'], st['quad_ctr'], st['n_reviewed']
    out['quad_cost'] = quad_concl(qc, ['高eco·高消耗', '高eco·低消耗', '低eco·高消耗', '低eco·低消耗'], '消耗', n, st['corr']['cost']['spearman'])
    out['quad_ctr'] = quad_concl(qt, ['高eco·高CTR', '高eco·低CTR', '低eco·高CTR', '低eco·低CTR'], 'CTR', n, st['corr']['ctr']['spearman'])
    out['cost_traits'] = trait_concl(qc['高eco·高消耗'], '高 ECO + 高消耗')
    out['ctr_traits'] = trait_concl(qt['高eco·高CTR'], '高 ECO + 高 CTR')
    types = st['types']
    out['types_hi'] = type_concl(types, avg, st, True)
    out['types_lo'] = type_concl(types, avg, st, False)
    st['quadrant'] = quadrant_data(st, d, cat)
    out['quadrant'] = quadrant_concl(st['quadrant'])
    st['type_names'] = {'hi': '、'.join(t['name'] for t in sorted([t for t in types if t['eco'] >= avg], key=lambda t: -t['eco'])[:3]),
                        'lo': '、'.join(t['name'] for t in sorted([t for t in types if t['eco'] < avg], key=lambda t: t['eco'])[:3])}
    return out


def build():
    d = frame_table()
    audit = load('source_audit.json')
    stats = {c: scope_stats(d, c) for c in ['全部', '鞋靴', '箱包']}
    bench = next((e for e in audit['excluded'] if e.get('category') == '整体'), None)
    for c in stats:
        stats[c]['concl'] = conclusions(stats[c], c, bench, d)
    cols = ['key', 'source_row', 'id', 'category', 'url', 'cost', 'eco', 'mix', 'skip', 'play', 'ctr', 'cvr', 'kind', 'media_ok', 'media_error',
            'duration', 'cuts15', 'cuts30', 'frames', 'ocr', 'reviewed', 'framework', 'type', 'person', 'face3', 'product_first', 'product_frames',
            'product_share', 'pf_bucket', 'hook', 'hook_text', 'price', 'story', 'confidence', 'quad_cost', 'quad_ctr']
    mats = [{k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()} for r in d[cols].to_dict('records')]
    payload = clean({'stats': stats, 'materials': mats, 'audit': {k: v for k, v in audit.items()}, 'benchmark': bench,
                     'insights': load('insights.json', {}), 'frameworks': FRAMEWORKS})
    (DATA / 'report.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    excel(d, stats, audit, bench)
    print(json.dumps({'materials': len(d), 'media_ok': int(d.media_ok.sum()), 'reviewed': int(d.reviewed.sum()), 'xlsx': str(XLSX)}, ensure_ascii=False))


def excel(d, stats, audit, bench):
    wb = xlsxwriter.Workbook(XLSX)
    wb.set_properties({'title': '箱包鞋靴 10.10 素材 ECO 创意分析', 'comments': '来源：通用分析数据集 - 2026-10-10T114915.478.xlsx / Sheet1'})
    base = {'font_name': 'Arial', 'font_size': 10, 'valign': 'top'}
    F = {k: wb.add_format({**base, **v}) for k, v in {
        'n': {}, 'w': {'text_wrap': True}, 'h': {'bold': True, 'bg_color': '#166653', 'font_color': '#FFFFFF', 'text_wrap': True, 'valign': 'vcenter'},
        't': {'bold': True, 'font_size': 16, 'font_color': '#FFFFFF', 'bg_color': '#142B2B', 'valign': 'vcenter'},
        's': {'bold': True, 'font_size': 12, 'bg_color': '#E7EFD9'}, 'note': {'font_color': '#64736F', 'text_wrap': True},
        'i': {'num_format': '#,##0;(#,##0);"-"'}, 'f1': {'num_format': '#,##0.0;(#,##0.0);"-"'}, 'p': {'num_format': '0.0%;(0.0%);"-"'},
        'link': {'font_color': '#166653', 'underline': True}}.items()}
    ws = wb.add_worksheet('阅读说明')
    ws.set_column('A:A', 20)
    ws.set_column('B:B', 110)
    ws.merge_range('A1:B2', '箱包鞋靴 · 10.10 · 500 条素材 ECO 创意分析', F['t'])
    notes = [
        ('源文件', '通用分析数据集 - 2026-10-10T114915.478.xlsx / Sheet1。共 500 条带素材链接记录（鞋靴 385 / 箱包 115）；第 2 行“整体”为汇总基准，不计入素材。'),
        ('金额口径', '“消耗(元)”为原表统计周期内消耗，不是日均值；表内无日期列，周期以导出口径为准。'),
        ('ECO口径', '使用“视频号混排eco总分均值(有效曝光口径)”；不是“混排分均值”。百分比字段在本工作簿以 0–1 存储、百分比显示。'),
        ('抽帧', '每条视频取前 30 秒（不足 30 秒取全片）0.3/1.5/3/5/8/12/16/20/25/29.5 秒共 10 帧；同时按 5 帧/秒检测镜头切换。静态图片素材保留 1 张原图。'),
        ('字幕识别', '关键帧文字由 macOS Vision 识别（中英文），用于辅助判断钩子与讲品内容；未做语音转写，口播内容只以画面字幕为准。'),
        ('画面标注', '逐条查看 10 张关键帧 + 识别字幕后人工标注：分类大类、素材类型、人物出镜方式、前3秒露脸、产品首现时间、产品为主体的帧数、开场钩子、是否出现价格。'),
        ('四象限', '在各品类内部按 ECO 中位数、消耗中位数（或 CTR 中位数）切分；“全部”视图沿用各自品类的切分结果，避免两类水位不同造成偏差。'),
        ('统计', '均值为逐素材等权，另附消耗加权 ECO。相关为 Pearson/Spearman；回归为 OLS + 5 折交叉验证。相关不等于因果，消耗还受出价、预算、账户影响。'),
        ('缺失', f"{len(audit['missing_metrics'])} 条素材缺少 ECO/快滑/均播，保留在全量表但不进入指标统计，不补 0。"),
    ]
    for i, (a, b) in enumerate(notes, 3):
        ws.write(i, 0, a, F['s'])
        ws.write(i, 1, b, F['w'])
        ws.set_row(i, 34)
    cw = wb.add_worksheet('结论汇总')
    cw.merge_range('A1:C1', '各表格总结性结论（随数据自动生成，与网页一致）', F['t'])
    cw.write_row(1, 0, ['范围', '表格', '结论'], F['h'])
    cr = 2
    for cat in ['全部', '鞋靴', '箱包']:
        for key, title in CONCL_TITLES:
            if cat == '全部' and key in ('quadrant', 'types_hi', 'types_lo'):
                continue
            cw.write(cr, 0, cat, F['n'])
            cw.write(cr, 1, title, F['n'])
            txt = '\n'.join(f'{j}. {t}' for j, t in enumerate(stats[cat]['concl'][key], 1))
            cw.write(cr, 2, txt, F['w'])
            cw.set_row(cr, max(30, 15 * (len(txt) // 70 + txt.count('\n') + 1)))
            cr += 1
    cw.set_column('A:A', 8)
    cw.set_column('B:B', 34)
    cw.set_column('C:C', 120)
    cw.freeze_panes(2, 0)
    # 全量
    raw = wb.add_worksheet('全量素材')
    hdr = ['原表行号', '素材标识', '品类', '消耗(元)', 'ECO总分', '3s快滑率', '平均播放(s)', 'CTR', '浅层CVR', '媒体', '原片时长(s)', '前15s切镜', '前30s切镜',
           '分类大类', '素材类型', '人物出镜', '前3秒露脸', '产品首现(s)', '产品主体帧/总帧', '开场钩子', '开场字幕', '出现价格', '前30秒叙事', '消耗象限', 'CTR象限', '复核', '原素材']
    raw.write_row(0, 0, hdr, F['h'])
    raw.set_row(0, 30)
    yn = {True: '是', False: '否'}
    for i, r in enumerate(d.to_dict('records'), 1):
        nf = len(r['frames'])
        vals = [r['source_row'], r['id'], r['category'], r['cost'], r['eco'], None if pd.isna(r['skip']) else r['skip'] / 100, r['play'],
                None if pd.isna(r['ctr']) else r['ctr'] / 100, None if pd.isna(r['cvr']) else r['cvr'] / 100,
                {'video': '视频', 'image': '静态图'}.get(r['kind'], '未获取'), r['duration'], r['cuts15'], r['cuts30'], r['framework'], r['type'], r['person'],
                yn.get(r['face3'], ''), r['product_first'], f"{r['product_frames']}/{nf}" if r['product_frames'] is not None else '', r['hook'], r['hook_text'],
                yn.get(r['price'], ''), r['story'], r.get('quad_cost'), r.get('quad_ctr'), '已复核' if r['reviewed'] else '未复核']
        fm = {3: F['i'], 4: F['i'], 5: F['p'], 6: F['f1'], 7: F['p'], 8: F['p'], 10: F['f1']}
        for j, v in enumerate(vals):
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                raw.write_blank(i, j, None, F['n'])
            else:
                raw.write(i, j, v, fm.get(j, F['n']))
        raw.write_url(i, len(vals), r['url'], F['link'], '查看')
    raw.autofilter(0, 0, len(d), len(hdr) - 1)
    raw.freeze_panes(1, 3)
    raw.set_column('A:A', 8)
    raw.set_column('B:B', 14)
    raw.set_column('C:M', 10)
    raw.set_column('N:P', 13)
    raw.set_column('Q:V', 11)
    raw.set_column('U:U', 24)
    raw.set_column('W:W', 44)
    raw.set_column('X:Z', 13)
    end = len(d) + 1

    def R(col):
        return f"'全量素材'!${col}$2:${col}${end}"
    crit_base = f'{R("Z")},"已复核"'
    # 分类汇总（公式）
    sm = wb.add_worksheet('分类汇总')
    sm.merge_range('A1:J1', '分类大类 × 品类（公式联动“全量素材”；仅已复核且指标齐全）', F['t'])
    sm.write_row(2, 0, ['品类', '分类大类', '素材数', '消耗合计(元)', '消耗占比', '平均ECO', '平均3s快滑率', '平均播放(s)', '平均CTR', '前3秒露脸占比'], F['h'])
    r0 = 3
    for cat in ['鞋靴', '箱包']:
        tot = stats[cat]['cost_sum']
        for fw in stats[cat]['frameworks']:
            er = r0 + 1
            c = f'{R("C")},A{er},{R("N")},B{er},{crit_base},{R("E")},"<>"'
            sm.write(r0, 0, cat, F['n'])
            sm.write(r0, 1, fw['name'], F['n'])
            sm.write_formula(r0, 2, f'=COUNTIFS({c})', F['i'], fw['n'])
            sm.write_formula(r0, 3, f'=SUMIFS({R("D")},{c})', F['i'], fw['cost_sum'])
            sm.write_formula(r0, 4, f'=IFERROR(D{er}/SUMIFS({R("D")},{R("C")},A{er},{crit_base},{R("E")},"<>"),0)', F['p'], fw['cost_sum'] / sum(x['cost_sum'] for x in stats[cat]['frameworks']))
            for col, L, key, fmt, sc in [(5, 'E', 'eco', F['i'], 1), (6, 'F', 'skip', F['p'], 100), (7, 'G', 'play', F['f1'], 1), (8, 'H', 'ctr', F['p'], 100)]:
                sm.write_formula(r0, col, f'=IFERROR(AVERAGEIFS({R(L)},{c}),0)', fmt, fw[key] / sc)
            sm.write_formula(r0, 9, f'=IFERROR(COUNTIFS({c},{R("Q")},"是")/C{er},0)', F['p'], fw['face3'] or 0)
            r0 += 1
        r0 += 1
    sm.set_column('A:B', 13)
    sm.set_column('C:J', 14)
    sm.freeze_panes(3, 2)
    # 截图样式：高低eco素材类型
    ty = wb.add_worksheet('高低ECO素材类型')
    ty.merge_range('A1:H1', '高 / 低 ECO 素材类型（素材类型 ≥3 条；按品类平均 ECO 划分高低）', F['t'])
    ty.merge_range('A2:H2', '消耗=该类型素材消耗合计（元）；得分=平均 ECO；3s快滑率、平均播放时长为逐素材均值。', F['note'])
    rr = 3
    for cat in ['鞋靴', '箱包']:
        avg = stats[cat]['eco']
        types = stats[cat]['types']
        fwmap = d[d.reviewed].groupby('type').framework.agg(lambda x: x.mode().iat[0]).to_dict()
        for label, sel in [('高eco', sorted([t for t in types if t['eco'] >= avg], key=lambda t: -t['eco'])), ('低eco', sorted([t for t in types if t['eco'] < avg], key=lambda t: t['eco']))]:
            ty.merge_range(rr, 0, rr, 7, f'{cat} · {label}（品类平均 ECO {avg:,.0f}）', F['s'])
            rr += 1
            ty.write_row(rr, 0, ['分类', '素材类型', '素材数', '消耗(元)', '得分', '3s快滑率', '平均播放时长(s)', '平均CTR'], F['h'])
            rr += 1
            for t in sel:
                er = rr + 1
                c = f'{R("C")},"{cat}",{R("O")},B{er},{crit_base},{R("E")},"<>"'
                ty.write(rr, 0, fwmap.get(t['name'], ''), F['n'])
                ty.write(rr, 1, t['name'], F['n'])
                ty.write_formula(rr, 2, f'=COUNTIFS({c})', F['i'], t['n'])
                ty.write_formula(rr, 3, f'=SUMIFS({R("D")},{c})', F['i'], t['cost_sum'])
                for col, L, key, fmt, sc in [(4, 'E', 'eco', F['i'], 1), (5, 'F', 'skip', F['p'], 100), (6, 'G', 'play', F['f1'], 1), (7, 'H', 'ctr', F['p'], 100)]:
                    ty.write_formula(rr, col, f'=IFERROR(AVERAGEIFS({R(L)},{c}),0)', fmt, t[key] / sc)
                rr += 1
            rr += 1
    ty.set_column('A:A', 13)
    ty.set_column('B:B', 22)
    ty.set_column('C:H', 14)
    # 四象限
    q = wb.add_worksheet('四象限特征')
    q.set_column('A:A', 10)
    q.set_column('B:B', 30)
    q.set_column('C:E', 14)
    q.write_row(0, 0, ['品类', '象限 / 特征', '该组占比', '其余占比', '差值'], F['h'])
    rr = 1
    for cat in ['鞋靴', '箱包']:
        for qk in ['quad_cost', 'quad_ctr']:
            name = '高eco·高消耗' if qk == 'quad_cost' else '高eco·高CTR'
            pr = stats[cat][qk][name]
            q.merge_range(rr, 0, rr, 4, f"{cat} · {name}（n={pr['n']}）", F['s'])
            rr += 1
            for f in pr['features'][:14]:
                q.write_row(rr, 0, [cat, f['feature']], F['n'])
                q.write(rr, 2, f['group'] or 0, F['p'])
                q.write(rr, 3, f['rest'] or 0, F['p'])
                q.write(rr, 4, f['diff'], F['p'])
                rr += 1
            rr += 1
    # 驱动
    dv = wb.add_worksheet('ECO驱动验证')
    dv.set_column('A:H', 16)
    dv.write_row(0, 0, ['范围', '样本数', 'ECO~快滑 Spearman', 'ECO~均播 Spearman', 'ECO~消耗 Spearman', 'ECO~CTR Spearman', 'OLS R²', '5折 R²'], F['h'])
    for i, cat in enumerate(['全部', '鞋靴', '箱包'], 1):
        st = stats[cat]
        dv.write_row(i, 0, [cat, st['n_valid']], F['n'])
        for j, m in enumerate(['skip', 'play', 'cost', 'ctr'], 2):
            dv.write_number(i, j, st['corr'][m]['spearman'], wb.add_format({**base, 'num_format': '0.000'}))
        dv.write_number(i, 6, st['ols']['r2'], F['p'])
        dv.write_number(i, 7, st['ols']['cv_r2'], F['p'])
    rr = 5
    for cat in ['全部', '鞋靴', '箱包']:
        dv.write_row(rr, 0, [cat + ' · 快滑分档', '素材数', '平均ECO', '', cat + ' · 均播分档', '素材数', '平均ECO'], F['h'])
        rr += 1
        for a, b in zip(stats[cat]['skip_bins'], stats[cat]['play_bins']):
            dv.write_row(rr, 0, [a['label'], a['n']], F['n'])
            dv.write(rr, 2, a['eco'] if a['eco'] == a['eco'] and a['eco'] is not None else '', F['i'])
            dv.write_row(rr, 4, [b['label'], b['n']], F['n'])
            dv.write(rr, 6, b['eco'] if b['eco'] == b['eco'] and b['eco'] is not None else '', F['i'])
            rr += 1
        rr += 1
    ex = wb.add_worksheet('排除与缺失')
    ex.write_row(0, 0, ['原表行号', '品类', '原因'], F['h'])
    rows = [(e['source_row'], e['category'], e['reason']) for e in audit['excluded']] + \
           [(r['source_row'], r['category'], '缺少ECO/快滑/均播') for r in d[~d.ok].to_dict('records')] + \
           [(r['source_row'], r['category'], '媒体获取失败：' + str(r['media_error'])[:80]) for r in d[~d.media_ok].to_dict('records')]
    for i, r in enumerate(rows, 1):
        ex.write_row(i, 0, r, F['n'])
    ex.set_column('C:C', 70)
    for s in wb.worksheets():
        s.hide_gridlines(2)
    wb.close()


if __name__ == '__main__':
    build()
