"""Aggregate E1 / E2 / E3 results into CSV tables + deck-styled charts (transparent PNG, dark slide background).

usage: python3 experiments/analyze.py [--res experiments/results] [--fig ../presentation/_build/exp_img]
"""
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
SEQ = ['LR', 'LF', 'RF', 'RR', 'SR', 'SL', 'FR', 'BR', 'SE', 'BA']
COND_LABEL = dict(CL='Closed loop', NOF='No force feedback', NOP='No re-observation', OL='Open loop')
COND_SUB = dict(CL='perception + admittance + retry', NOF='position-only insertion', NOP='nominal poses only',
                OL='nominal poses, position-only')

# ---- style (deck: dark background, Segoe UI) ----
for f in ('segoeui', 'segoeuisl', 'seguisb', 'segoeuil', 'bahnschrift'):
    p = Path(f'/mnt/c/Windows/Fonts/{f}.ttf')
    if p.exists():
        fm.fontManager.addfont(str(p))
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 12, 'axes.unicode_minus': False})
INK, INK2, INK3, GRID, CARD = '#E8EAED', '#A9AFB7', '#717983', '#2A2F36', '#21262D'
# categorical (validated all-pairs on the dark surface): teal, orange-red, blue
TEAL, ORANGE, BLUE = '#2E9E8D', '#D95926', '#3987E5'
NEUTRAL = '#5B636E'
DESIGN = {'float': ('Locating + floating end', TEAL, '-', 'o'),
          'rigid_mid': ('Both ends rigid, rail centred', BLUE, (0, (5, 2.5)), 's'),
          'rigid_loc': ('Both ends rigid, aligned to one end', ORANGE, (0, (1.5, 2)), 'D')}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def style_ax(ax, ygrid=True):
    ax.set_facecolor('none')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    if ygrid:
        ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.patch.set_alpha(0)
    fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)
    print('wrote', path)


# ====================================================================== E1
def e1(res, fig_dir, out):
    rows = list(csv.DictReader(open(res / 'E1/e1_trials.csv')))
    for r in rows:
        r['e_mm'] = float(r['e_mm']); r['success'] = int(r['success'])
    errs = sorted({r['e_mm'] for r in rows})
    table = []
    for dz in DESIGN:
        for e in errs:
            sel = [r for r in rows if r['design'] == dz and r['e_mm'] == e]
            k = sum(r['success'] for r in sel)
            p, lo, hi = wilson(k, len(sel))
            pk = [float(r['peak_force_N']) for r in sel]
            table.append(dict(design=dz, e_mm=e, n=len(sel), success=k, rate=round(p, 3), ci_lo=round(lo, 3),
                              ci_hi=round(hi, 3), mean_peak_N=round(float(np.mean(pk)), 2),
                              jams=Counter(r['result'] for r in sel if not r['success']).most_common(1)[0][0]
                              if k < len(sel) else ''))
    with open(out / 'e1_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    # tolerance windows: largest error with >= 95 % success (all smaller errors too) and the 50 % crossing
    tol = {}
    for dz in DESIGN:
        rate = {t['e_mm']: t['rate'] for t in table if t['design'] == dz}
        w95 = 0.0
        for e in errs:
            if rate[e] >= 0.95:
                w95 = e
            else:
                break
        e50 = next((errs[i - 1] + (errs[i] - errs[i - 1]) * (rate[errs[i - 1]] - 0.5) / (rate[errs[i - 1]] - rate[errs[i]])
                    for i in range(1, len(errs)) if rate[errs[i]] < 0.5 <= rate[errs[i - 1]]), None)
        tol[dz] = dict(w95=w95, e50=None if e50 is None else round(e50, 2))
    json.dump(tol, open(out / 'e1_tolerance.json', 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    style_ax(ax)
    ax.axvspan(0, 2.0, color='#FFFFFF', alpha=0.035, lw=0)
    ax.text(1.0, 1.035, 'spacing error range in E2/E3\n(each leg nest ±1 mm)', color=INK3, fontsize=10.5,
            ha='center', va='bottom', linespacing=1.25, transform=ax.get_xaxis_transform())
    for dz, (lab, c, ls, mk) in DESIGN.items():
        xs = [t['e_mm'] for t in table if t['design'] == dz]
        ys = [100 * t['rate'] for t in table if t['design'] == dz]
        ax.plot(xs, ys, color=c, lw=2.4, ls=ls, marker=mk, ms=5.5, mec='#16191E', mew=1.0, label=lab, zorder=3,
                solid_capstyle='round')
    # direct labels where each curve falls
    lab_pos = {'float': (3.1, 60, 'floating end', 'left'), 'rigid_mid': (1.98, 36, 'rigid, centred', 'left'),
               'rigid_loc': (1.27, 46, 'rigid, one end', 'right')}
    for dz, (x, y, t, ha) in lab_pos.items():
        ax.text(x, y, t, color=INK, fontsize=12, ha=ha, va='center')
    ax.set_xlim(-0.08, 4.08); ax.set_ylim(-4, 104)
    ax.set_xticks(np.arange(0, 4.01, 0.5)); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100 %'])
    ax.set_xlabel('leg spacing error along the rail (mm)', color=INK2, fontsize=12)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=10.5, labelcolor=INK2, handlelength=2.6,
                    bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / 'e1_success.png')
    return table, tol


# ====================================================================== E2 / E3
def load_e23(res, exp='E23'):
    trials = []
    for p in sorted((res / exp).glob('*_summary.json')):
        s = json.load(open(p))
        steps = {st['part']: st for st in s.get('steps', [])}
        ok = bool(s['success']) and s.get('final_disp_mm') is not None and s['final_disp_mm'] < 2.0
        trials.append(dict(cond=s['cond'], config=s['config'], seed=s['seed'], success=ok,
                           n_parts_ok=s.get('n_parts_ok', 0), fail_part=s.get('fail_part'),
                           fail_cause=s.get('fail_cause') or ('UNSTABLE' if s['success'] and not ok else None),
                           sim_time=s.get('sim_time'), wall_time=s.get('wall_time'), final_disp_mm=s.get('final_disp_mm'),
                           retries=sum(max(0, st.get('attempts', 1) - 1) for st in steps.values()),
                           n_jams=len(s.get('jams', [])), fallbacks=len(s.get('fallbacks', [])),
                           max_peak_N=max([st.get('peak_force', 0.0) for st in steps.values()] or [0.0]),
                           exp=exp, nest=s.get('nest_offsets_mm', {}), steps=steps))
    return trials


def e23_tables(trials, out, name='e23'):
    flat = [{k: v for k, v in t.items() if k not in ('steps', 'nest')} for t in trials]
    if not flat:
        return []
    with open(out / f'{name}_trials.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    summ = []
    for cond in COND_LABEL:
        for cfg in ('A', 'B'):
            sel = [t for t in trials if t['cond'] == cond and t['config'] == cfg]
            if not sel:
                continue
            k = sum(t['success'] for t in sel)
            p, lo, hi = wilson(k, len(sel))
            okt = [t for t in sel if t['success']]
            summ.append(dict(cond=cond, config=cfg, n=len(sel), success=k, rate=round(p, 3), ci_lo=round(lo, 3),
                             ci_hi=round(hi, 3), parts_ok_mean=round(float(np.mean([t['n_parts_ok'] for t in sel])), 2),
                             sim_time_mean=round(float(np.mean([t['sim_time'] for t in okt])), 1) if okt else None,
                             sim_time_sd=round(float(np.std([t['sim_time'] for t in okt])), 1) if okt else None,
                             wall_time_mean=round(float(np.mean([t['wall_time'] for t in okt])), 1) if okt else None,
                             disp_mean=round(float(np.mean([t['final_disp_mm'] for t in okt])), 2) if okt else None,
                             disp_max=round(float(np.max([t['final_disp_mm'] for t in okt])), 2) if okt else None,
                             trials_with_retry=sum(t['retries'] > 0 for t in sel),
                             retries_total=sum(t['retries'] for t in sel),
                             recovered_jams=sum(t['retries'] for t in okt),
                             fallbacks_total=sum(t['fallbacks'] for t in sel),
                             causes=dict(Counter(f"{t['fail_part']}:{t['fail_cause']}" for t in sel if not t['success']))))
    with open(out / f'{name}_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    json.dump(summ, open(out / f'{name}_summary.json', 'w'), indent=1)
    return summ


def cell_status(t, part):
    st = t['steps'].get(part)
    if st is None:
        return 'none'
    if not st['ok']:
        return 'fail'
    return 'retry' if st.get('attempts', 1) > 1 else 'ok'


STATUS = dict(ok=(TEAL, ''), retry=(BLUE, '↻'), fail=(ORANGE, '✕'), none=(CARD, ''))


def e2_grid(trials, fig_dir, cond='CL', fname='e2_grid.png'):
    """One column per seeded build, one row per part (assembly order), chair A above chair B, legend below."""
    cfgs = [c for c in ('A', 'B') if any(t['cond'] == cond and t['config'] == c for t in trials)]
    if not cfgs:
        return
    seeds = sorted({t['seed'] for t in trials if t['cond'] == cond})
    ns = len(seeds)
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    ax.set_facecolor('none'); ax.axis('off')
    g, y0 = 0.14, 0.0

    def cell(x, y, stt):
        c, glyph = STATUS[stt]
        ax.add_patch(FancyBboxPatch((x + g / 2, y + g / 2), 1 - g, 1 - g, boxstyle='round,pad=0,rounding_size=0.12',
                                    fc=c, ec='none'))
        if glyph:
            ax.text(x + 0.5, y + 0.55, glyph, color='#FFFFFF', fontsize=6.5, ha='center', va='center', fontweight='bold')

    for cfg in cfgs:
        by_seed = {t['seed']: t for t in trials if t['cond'] == cond and t['config'] == cfg}
        k = sum(t['success'] for t in by_seed.values())
        ax.text(0, y0 - 0.45, f'Chair {cfg}', color=INK, fontsize=13, ha='left', va='bottom', fontweight='semibold')
        ax.text(ns, y0 - 0.45, f'{k} / {len(by_seed)} complete', color=INK2, fontsize=11.5, ha='right', va='bottom')
        for j, part in enumerate(SEQ):
            ax.text(-0.5, y0 + j + 0.5, part, color=INK2, fontsize=9.5, ha='right', va='center')
            for i, sd in enumerate(seeds):
                t = by_seed.get(sd)
                cell(i, y0 + j, cell_status(t, part) if t else 'none')
        y0 += len(SEQ) + 1.9
    for i, sd in enumerate(seeds):
        if sd % 5 == 0:
            ax.text(i + 0.5, y0 - 1.45, str(sd), color=INK3, fontsize=8.5, ha='center', va='center')
    ax.text(-0.5, y0 - 1.45, 'seed', color=INK3, fontsize=8.5, ha='right', va='center')
    lx = 0.0
    for stt, lab in (('ok', 'placed first time'), ('retry', 'placed after jam + retry'), ('fail', 'failed, build stops'),
                     ('none', 'not reached')):
        cell(lx, y0 - 0.5, stt)
        if stt == 'none':          # the empty cell is near the background colour: outline it in the legend
            ax.add_patch(FancyBboxPatch((lx + g / 2, y0 - 0.5 + g / 2), 1 - g, 1 - g, boxstyle='round,pad=0,rounding_size=0.12',
                                        fc='none', ec=INK3, lw=0.8))
        ax.text(lx + 1.3, y0, lab, color=INK2, fontsize=10, ha='left', va='center')
        lx += 1.3 + 0.36 * len(lab) + 1.6
    ax.set_xlim(-2.6, ns + 0.1); ax.set_ylim(y0 + 0.8, -1.4)
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.1)
    save(fig, fig_dir / fname)


def e3_success(summ, fig_dir):
    rows = [s for s in summ if s['config'] == 'A']
    if not rows:
        return
    rows.sort(key=lambda s: list(COND_LABEL).index(s['cond']))
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    style_ax(ax, ygrid=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    for i, s in enumerate(rows):
        c = TEAL if s['cond'] == 'CL' else NEUTRAL
        ax.barh(i, 100 * s['rate'], height=0.56, color=c, zorder=2)
        ax.plot([100 * s['ci_lo'], 100 * s['ci_hi']], [i, i], color=INK, lw=1.2, zorder=3)
        for x in (s['ci_lo'], s['ci_hi']):
            ax.plot([100 * x] * 2, [i - 0.09, i + 0.09], color=INK, lw=1.2, zorder=3)
        ax.text(101.5 if s['ci_hi'] < 0.86 else 1.5 + 100 * s['ci_hi'], i, f"{s['success']}/{s['n']}", color=INK,
                fontsize=12.5, va='center', ha='left')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([COND_LABEL[s['cond']] for s in rows], color=INK, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 116); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(['0', '25', '50', '75', '100 %'])
    ax.set_xlabel('complete chairs (chair A, 95 % Wilson interval)', color=INK2, fontsize=11)
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / 'e3_success.png')


def e3_progress(trials, fig_dir):
    conds = [c for c in COND_LABEL if any(t['cond'] == c and t['config'] == 'A' for t in trials)]
    if not conds:
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    style_ax(ax, ygrid=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    for i, c in enumerate(conds):
        vals = sorted(t['n_parts_ok'] for t in trials if t['cond'] == c and t['config'] == 'A')
        cnt = Counter(vals)
        for v, k in cnt.items():
            ys = i + (np.arange(k) - (k - 1) / 2) * 0.075
            ax.scatter([v] * k, ys, s=34, color=TEAL if c == 'CL' else INK2, edgecolor='#16191E', lw=0.8, zorder=3)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([COND_LABEL[c] for c in conds], color=INK, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 10.5); ax.set_xticks(range(0, 11))
    ax.set_xticklabels([str(i) for i in range(10)] + ['10\nall'])
    ax.set_xlabel('parts placed before the run stopped (one dot per trial)', color=INK2, fontsize=11)
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / 'e3_progress.png')


def e3_heat(groups, fig_dir):
    """Where failed builds stopped: rows = condition (grouped, e.g. nominal / 2x perturbation), columns = part."""
    rows = []                        # (group title or None, cond, cfg, trials)
    for title, trials in groups:
        sub = [(c, cfg) for c in COND_LABEL for cfg in ('A', 'B') if any(t['cond'] == c and t['config'] == cfg for t in trials)]
        if not sub:
            continue
        rows.append((title, None, None, None))
        rows += [(None, c, cfg, trials) for c, cfg in sub]
    if not rows:
        return
    M = np.zeros((len(rows), len(SEQ)), int)
    N = np.zeros(len(rows), int)
    causes = defaultdict(Counter)
    for i, (title, c, cfg, trials) in enumerate(rows):
        if c is None:
            continue
        sel = [t for t in trials if t['cond'] == c and t['config'] == cfg]
        N[i] = len(sel)
        for t in sel:
            if not t['success'] and t['fail_part'] in SEQ:
                j = SEQ.index(t['fail_part']); M[i, j] += 1; causes[(i, j)][t['fail_cause']] += 1
    fig, ax = plt.subplots(figsize=(7.9, 0.5 * len(rows) + 0.7))
    ax.set_facecolor('none'); ax.axis('off')
    vmax = max(1, M.max())
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('o', ['#3A2A20', '#F0A23B'])
    short = dict(JAM='JAM', JAM_POSITION='JAM', PLAN_FAIL='PLAN', UNREACHABLE='REACH', GRASP_MISS='GRASP', SLIP='SLIP',
                 RETENTION='RET', UNSTABLE='UNST', EXCEPTION='ERR', WALL_TIMEOUT='TIME')
    for i, (title, c, cfg, trials) in enumerate(rows):
        if c is None:
            ax.text(-3.7, i + 0.62, title, color=ORANGE, fontsize=10, ha='left', va='center', fontweight='semibold')
            continue
        ax.text(-0.25, i + 0.5, f'{COND_LABEL[c]}' + (f'  ·  {cfg}' if c == 'CL' else ''), color=INK, fontsize=11,
                ha='right', va='center')
        ax.text(len(SEQ) + 0.2, i + 0.5, f'{int(M[i].sum())}/{N[i]}', color=INK2, fontsize=10, ha='left', va='center')
        for j in range(len(SEQ)):
            v = M[i, j]
            fc = CARD if v == 0 else cmap(0.25 + 0.75 * v / vmax)
            ax.add_patch(FancyBboxPatch((j + 0.05, i + 0.06), 0.9, 0.88, boxstyle='round,pad=0,rounding_size=0.1',
                                        fc=fc, ec='none'))
            if v:
                cause = causes[(i, j)].most_common(1)[0][0]
                dark = v / vmax > 0.45
                ax.text(j + 0.5, i + 0.42, str(v), color='#16191E' if dark else INK, fontsize=11, ha='center', va='center',
                        fontweight='semibold')
                ax.text(j + 0.5, i + 0.77, short.get(cause, cause[:5]), color='#16191E' if dark else INK2, fontsize=7,
                        ha='center', va='center')
    for j, p in enumerate(SEQ):
        ax.text(j + 0.5, -0.12, p, color=INK2, fontsize=10.5, ha='center', va='bottom')
    ax.text(len(SEQ) + 0.2, -0.12, 'failed', color=INK3, fontsize=9.5, ha='left', va='bottom')
    ax.set_xlim(-3.7, len(SEQ) + 1.3); ax.set_ylim(len(rows) + 0.05, -0.7)
    fig.tight_layout(pad=0.2)
    save(fig, fig_dir / 'e3_heat.png')


def e3_force(trials, fig_dir):
    conds = [c for c in COND_LABEL if any(t['cond'] == c and t['config'] == 'A' for t in trials)]
    if not conds:
        return {}
    data = {c: [st['peak_force'] for t in trials if t['cond'] == c and t['config'] == 'A'
                for st in t['steps'].values() if st.get('attempts', 0) > 0] for c in conds}
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    style_ax(ax, ygrid=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        v = np.maximum(np.array(data[c]), 0.1)
        if not len(v):
            continue
        ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=10, color=TEAL if c == 'CL' else INK2, alpha=0.55, lw=0, zorder=2)
        med = np.median(v)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2.2, zorder=3, solid_capstyle='round')
        ax.text(med, i - 0.36, f'{med:.1f} N', color=INK, fontsize=10.5, ha='center', va='bottom')
    ax.axvline(90, color=ORANGE, lw=1.1, ls=(0, (4, 3)))
    ax.text(97, -0.62, 'jam limit 90 N', color=INK2, fontsize=10, ha='left', va='center')
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xlim(0.1, 2000); ax.set_ylim(len(conds) - 0.5, -0.8)
    ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(['≤0.1', '1', '10', '100', '1000 N'])
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([COND_LABEL[c] for c in conds], color=INK, fontsize=12.5)
    ax.set_xlabel('peak axial force per insertion (log scale, all parts)', color=INK2, fontsize=11)
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / 'e3_force.png')
    return {c: dict(n=len(v), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                    max=round(float(np.max(v)), 2)) for c, v in data.items() if len(v)}


def paired_bars(rows, series, fig_dir, fname, xlabel, figsize=(6.3, 4.5)):
    """Horizontal grouped bars of success rate with Wilson whiskers. rows: [(label, key)], series: [(name, colour, {key: s})]."""
    fig, ax = plt.subplots(figsize=figsize)
    style_ax(ax, ygrid=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    hgt = 0.8 / len(series)
    for i, (lab, key) in enumerate(rows):
        for j, (name, colour, data) in enumerate(series):
            s = data.get(key)
            if s is None:
                continue
            y = i - 0.4 + hgt * (j + 0.5)
            ax.barh(y, 100 * s['rate'], height=hgt * 0.9, color=colour, zorder=2, label=name if i == 0 else None)
            ax.plot([100 * s['ci_lo'], 100 * s['ci_hi']], [y, y], color=INK, lw=1.0, zorder=3)
            ax.text(max(100 * s['ci_hi'], 100 * s['rate']) + 1.5, y, f"{s['success']}/{s['n']}", color=INK, fontsize=10.5,
                    va='center', ha='left')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 122); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(['0', '25', '50', '75', '100 %'])
    ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.13), ncol=len(series), frameon=False, fontsize=10.5, labelcolor=INK2,
              handlelength=1.2)
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / fname)


def e4_stress(summ23, summ4, fig_dir):
    rows = [((COND_LABEL[c] + (f'  ·  {cfg}' if c == 'CL' else '')), (c, cfg)) for c in COND_LABEL for cfg in ('A', 'B')
            if (c, cfg) in summ4]
    if rows:
        paired_bars(rows, [('nominal', TEAL, summ23), ('2× perturbation', ORANGE, summ4)], fig_dir, 'e4_stress.png',
                    'complete chairs (95 % Wilson interval)')


RAIL_HOSTS = dict(FR=('LF', 'RF', 0), BR=('LR', 'RR', 0), SL=('LF', 'LR', 1), SR=('RF', 'RR', 1))


def rail_spacing(trials, e1_table, fig_dir, out, bins=(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)):
    """Every closed-loop rail insertion (E2 + E4, both chairs): leg spacing error along that rail (from the fixture
    offsets) vs outcome, against the E1 coupon curve of the same joint."""
    pts = []
    for t in trials:
        if t['cond'] != 'CL' or not t['nest']:
            continue
        for r, (h1, h2, k) in RAIL_HOSTS.items():
            st = t['steps'].get(r)
            if st is None:
                continue
            e = abs(t['nest'][h2][k] - t['nest'][h1][k])
            pts.append(dict(exp=t['exp'], config=t['config'], seed=t['seed'], rail=r, e_mm=round(e, 3), ok=bool(st['ok']),
                            first=bool(st['ok'] and st.get('attempts', 1) == 1), attempts=st.get('attempts', 0)))
    if not pts:
        return None
    with open(out / 'rail_spacing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [p for p in pts if lo <= p['e_mm'] < hi]
        if sel:
            rows.append(dict(lo=lo, hi=hi, mid=(lo + hi) / 2, n=len(sel), first=sum(p['first'] for p in sel) / len(sel),
                             ok=sum(p['ok'] for p in sel) / len(sel)))
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    style_ax(ax)
    xs = [t['e_mm'] for t in e1_table if t['design'] == 'float']
    ys = [100 * t['rate'] for t in e1_table if t['design'] == 'float']
    ax.plot(xs, ys, color=TEAL, lw=2.2, ls=(0, (5, 2.5)), label='E1 coupon (one attempt)', zorder=2)
    mx = [r['mid'] for r in rows]
    ax.plot(mx, [100 * r['first'] for r in rows], color=BLUE, lw=2.2, marker='s', ms=6, mec='#16191E',
            label='robot, first attempt', zorder=3)
    ax.plot(mx, [100 * r['ok'] for r in rows], color=ORANGE, lw=2.2, marker='o', ms=6, mec='#16191E',
            label='robot, after re-observe + retry', zorder=4)
    for r in rows:
        ax.text(r['mid'], -11, f"{r['n']}", color=INK3, fontsize=8.5, ha='center', va='center')
    ax.text(-0.02, -11, 'n =', color=INK3, fontsize=8.5, ha='right', va='center')
    ax.set_xlim(-0.3, 4.05); ax.set_ylim(-16, 106)
    ax.set_xticks(np.arange(0, 4.01, 0.5)); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100 %'])
    ax.set_xlabel('leg spacing error along the rail (mm, bins of 0.5)', color=INK2, fontsize=11)
    ax.legend(loc='lower left', frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.0, 0.1))
    fig.tight_layout(pad=0.4)
    save(fig, fig_dir / 'rail_spacing.png')
    json.dump(rows, open(out / 'rail_spacing.json', 'w'), indent=1)
    return rows


def e3_case(res, fig_dir, seed, part, conds=('CL', 'NOF', 'NOP')):
    """Insertion force of one part at one seed (same perturbations) under several controllers, small multiples."""
    traces = {}
    for c in conds:
        f = res / f'E23/{c}_A_seed{seed:02d}_hud.json'
        if not f.exists():
            continue
        hud = json.load(open(f))['hud']
        pts = [(x['t'], x['force'], x.get('retry', 0)) for x in hud if x['part'] == part and x['phase'].startswith('insert')]
        if pts:
            traces[c] = pts
    if not traces:
        return None
    fig, axes = plt.subplots(len(traces), 1, figsize=(7.4, 1.32 * len(traces) + 0.55), sharex=True, squeeze=False)
    tmax = max(p[-1][0] - p[0][0] for p in traces.values())
    info = {}
    for ax, (c, pts) in zip(axes[:, 0], traces.items()):
        style_ax(ax)
        t0 = pts[0][0]
        segs, cur = [], [pts[0]]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            if b_[0] - a_[0] > 0.1:               # a gap = retreat / re-align between attempts
                segs.append(cur); cur = []
            cur.append(b_)
        segs.append(cur)
        for sg in segs:
            ax.plot([q[0] - t0 for q in sg], [max(q[1], 0) for q in sg], color=TEAL if c == 'CL' else INK2, lw=2.0,
                    solid_capstyle='round')
        ax.axhline(25, color=INK3, lw=0.9, ls=(0, (4, 3)))
        peak = max(q[1] for q in pts)
        ax.set_ylim(-3, max(32, peak * 1.15))
        ax.set_yticks([0, 25] + ([int(round(peak, -1))] if peak > 45 else []))
        ax.text(0.0, 1.02, COND_LABEL[c], transform=ax.transAxes, color=INK, fontsize=12, ha='left', va='bottom')
        info[c] = dict(attempts=len(segs), peak=round(peak, 1), duration=round(pts[-1][0] - t0, 2))
        ax.text(1.0, 1.02, f"{len(segs)} attempt{'s' if len(segs) > 1 else ''}  ·  peak {peak:.1f} N", transform=ax.transAxes,
                color=INK2, fontsize=11, ha='right', va='bottom')
    axes[-1, 0].set_xlim(-0.1, tmax + 0.2)
    axes[-1, 0].set_xlabel(f'time since {part} insertion started (s)   ·   dashed: F_ref 25 N', color=INK2, fontsize=11)
    fig.tight_layout(pad=0.4, h_pad=1.2)
    save(fig, fig_dir / 'e3_case.png')
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default=str(HERE / 'results'))
    ap.add_argument('--fig', default=str(HERE.parents[1] / 'presentation/_build/exp_img'))
    ap.add_argument('--case', default='0:FR', help='seed:part for the force-trace case study')
    a = ap.parse_args()
    res, fig_dir = Path(a.res), Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = res / 'tables'; out.mkdir(exist_ok=True)
    table, tol = e1(res, fig_dir, out)
    print('E1 tolerance', tol)
    trials = load_e23(res)
    summ = e23_tables(trials, out)
    for s in summ:
        print(s)
    key = lambda ss: {(x['cond'], x['config']): x for x in ss}
    trials4 = load_e23(res, 'E4') if (res / 'E4').exists() else []
    summ4 = e23_tables(trials4, out, 'e4') if trials4 else []
    e4_stress(key(summ), key(summ4), fig_dir)
    e3_heat([('NOMINAL PERTURBATION', trials), ('2× PERTURBATION (E4)', trials4)], fig_dir)
    print('E4', [(s['cond'], s['config'], s['success'], s['n']) for s in summ4])
    print('rail spacing', rail_spacing(trials + trials4, table, fig_dir, out))
    e2_grid(trials, fig_dir)
    e3_success(summ, fig_dir)
    e3_progress(trials, fig_dir)
    forces = e3_force(trials, fig_dir)
    json.dump(forces, open(out / 'e3_force.json', 'w'), indent=1)
    print('forces', forces)
    sd, pt = a.case.split(':')
    case = e3_case(res, fig_dir, int(sd), pt)
    json.dump(dict(seed=int(sd), part=pt, traces=case), open(out / 'e3_case.json', 'w'), indent=1)
    print('case', case)
