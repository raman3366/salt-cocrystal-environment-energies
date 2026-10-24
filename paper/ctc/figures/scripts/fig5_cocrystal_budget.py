"""Fig. 5: the error budget of the four cocrystals at P4r, every check as a point (forest plot, DOUBLE).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig5_cocrystal_budget.py

summary.json crystals[].budget at P4r: budget.margin for dapsone and pcplut (an other-side minimum exists at
PBE0+MP2: the classification margin), budget.P4r for mppcp and nbaetpy (no other-side minimum at PBE0+MP2: the
quantity is P4r - P0, not a margin; their resolution rests on R5 at P4, resolution.quantity).  For cocrystals the
stored sign (other - observed) equals dE_PT (asserted through _ctc.sign_pt = +1).
Points: the central value (PBE0+MP2), central + each check shift (components), the computed combinations
(combined, absolute values), and, not part of the envelope, the definition alternatives and the frozen-frame
zero-point lower bound (bounds).  Band: the budget envelope (asserted = min/max of central, central + shifts and
combinations).  Zero marked.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _ctc as S  # noqa: E402
from figkit import style  # noqa: E402

STEM = 'fig5_cocrystal_budget'
FS = style.SMALL_PT
KEYS = ['dapsone', 'pcplut', 'mppcp', 'nbaetpy']
# (component, sub-key prefix) -> row label; order = row order
ROWS = [
    (('central', None), 'central value, PBE0 + MP2 pair'),
    (('level_spread', 'PBE0+SCS_MP2'), 'pair correction SCS-MP2'),
    (('level_spread', 'PBE0+wB97X_D3BJ'), 'pair correction ωB97X-D3(BJ)'),
    (('delta_T', '[CCSD(T)'), 'δ(T): CCSD(T) − MP2, def2-SVP'),
    (('anchor_PBE', 'PBE +'), 'anchor PBE + [MP2 − PBE]'),
    (('anchor_PBE50', 'PBE50'), 'anchor PBE50 + [MP2 − PBE50]'),
    (('in_field_vs_vacuum', 'X-ray'), 'pair correction in field, X-ray cell'),
    (('in_field_vs_vacuum', 'relaxed'), 'pair correction in field, relaxed cells'),
    (('basis_QZVP', '['), 'pair correction def2-QZVP'),
    (('basis_TZVPD', '['), 'pair correction def2-TZVPD'),
    (('cell', 'relaxed'), 'relaxed cells'),
    (('k_mesh', 'PBE'), 'dense k mesh'),
    (('q_mesh', 'PBE0'), 'EXX q mesh = k mesh'),
    (('ZPE', 'frozen'), 'zero point, frozen frame (H)'),
    (('ZPE', 'harmonic partial Hessian, H D A'), 'zero point, harmonic (H, D, A)'),
    (('ZPE', 'harmonic partial Hessian, H only'), 'zero point, harmonic (H only)'),
    (('combined', 'relaxed cells, in-field'), 'combined: relaxed cells + in field'),
    (('combined', 'relaxed cells, vacuum +'), 'combined: relaxed + k + q (+ δ(T))'),
    (('combined', 'relaxed cells, in-field +'), 'combined: relaxed + in field + k + q (+ δ(T))'),
    (('definition_alternatives', 'lowest'), 'alt. definition: lowest other-side point'),
    (('definition_alternatives', 'observed-side'), 'alt. definition: obs.-side point below P0'),
    (('bounds', 'frozen'), 'bound: frozen-frame zero point (≥)'),
]


def points(k):
    """(row index, value, kind, summary key) for every check of crystal k."""
    q = 'margin' if 'margin' in S.CRY[k]['budget'] else 'P4r'
    assert S.sign_pt(k) == 1
    b = S.V(k, 'budget', q)
    c = b['central']
    out, inenv = [], [c]
    for i, ((comp, pre), _) in enumerate(ROWS):
        if comp == 'central':
            out.append((i, c, 'central', ('budget', q, 'central')))
            continue
        d = b.get(comp) if comp in ('combined', 'definition_alternatives', 'bounds') else b['components'].get(comp)
        for sk, v in (d or {}).items():
            if not isinstance(v, (int, float)):
                continue
            # 'relaxed cells, in-field' must not also catch 'relaxed cells, in-field + ...'
            if not sk.startswith(pre) or (pre == 'relaxed cells, in-field' and sk != pre):
                continue
            val = v if comp == 'combined' else c + v
            kind = {'combined': 'combined', 'definition_alternatives': 'alt', 'bounds': 'alt'}.get(comp, 'check')
            key = ('budget', q, comp, sk) if kind != 'check' else ('budget', q, 'components', comp, sk)
            out.append((i, val, kind, key))
            if kind != 'alt':
                inenv.append(val)
    env = b['envelope']
    assert abs(min(inenv) - env[0]) < 2e-3 and abs(max(inenv) - env[1]) < 2e-3, (k, min(inenv), max(inenv), env)
    n_src = sum(len([x for x in (v or {}).values() if isinstance(x, (int, float))]) for v in b['components'].values())
    n_src += len(b['combined'] or {}) + len(b['definition_alternatives'] or {}) + \
        len([x for x in (b['bounds'] or {}).values() if isinstance(x, (int, float))])
    assert len(out) - 1 == n_src, (k, len(out) - 1, n_src)              # every check placed in exactly one row
    return q, c, env, out


def main():
    plt = S.setup()
    n = len(ROWS)
    pitch = 3.05
    AH = n * pitch
    L, PW, gx, B = 58.0, 26.0, 3.2, 10.0
    W, H = 175.0, B + AH + 25.5
    fig = S.figure_mm(W, H)
    col = S.CLASS_COLOURS['cocrystal']
    for j, k in enumerate(KEYS):
        q, c, env, pts = points(k)
        ax = S.axes_mm(fig, L + j * (PW + gx), B, PW, AH)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlim(-2.5, 11.0)
        ax.set_xticks([0, 5, 10])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for _, lab in ROWS] if j == 0 else [])
        ax.tick_params(axis='y', length=0, labelsize=FS)
        for i in range(n):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color='0.965', lw=0, zorder=0)
        ax.axvspan(env[0], env[1], color=col, alpha=0.22, lw=0, zorder=1)
        ax.axvline(0, color='#B4202A', lw=0.8, zorder=2)
        ax.axvline(c, color='0.25', lw=0.5, ls=(0, (2, 1.5)), zorder=2)
        S.V(k, 'budget', q, 'envelope')
        for i, v, kind, key in pts:
            sty = {'central': dict(marker='D', ms=4.0, mfc='black', mec='black'),
                   'check': dict(marker='o', ms=3.0, mfc='0.25', mec='0.25'),
                   'combined': dict(marker='s', ms=3.0, mfc=S.CLASS_DARK['cocrystal'], mec=S.CLASS_DARK['cocrystal']),
                   'alt': dict(marker='^', ms=3.2, mfc='white', mec='0.35', mew=0.7)}[kind]
            ax.plot(v, i, ls='none', zorder=5, **sty)
            S.V(k, *key)
            S.mark(k, *key)
            ax.plot([c, v], [i, i], color='0.55', lw=0.5, zorder=3) if kind == 'check' else None
        ax.set_xlabel('kJ mol$^{-1}$', fontsize=FS)
        ax.tick_params(axis='x', labelsize=FS)
        res = S.V(k, 'resolution')
        ok = res['verdict'].startswith('resolved')
        S.mark(k, 'resolution', printed=res['verdict'])
        what = 'margin at P4r' if q == 'margin' else 'P4r − P0, no other-\nside minimum'
        ttl = f"{S.NAME[k]}\n{what}\n{S.num(c, 1, sign=True)} [{S.num(env[0], 1)}, {S.num(env[1], 1)}]"
        S.mark(k, 'budget', q, 'central', printed=S.num(c, 1, sign=True))
        S.mark(k, 'budget', q, 'envelope', printed=f'[{S.num(env[0], 1)}, {S.num(env[1], 1)}]')
        ax.set_title(ttl, fontsize=FS, loc='left', pad=2.0, linespacing=1.1)
        ax.text(0.0, 1.0 + 10.2 / AH, ('resolved (on R5 at P4)' if q != 'margin' else 'resolved') if ok else
                'unresolved', transform=ax.transAxes, ha='left', va='bottom', fontsize=FS, linespacing=1.1,
                color='0.35' if ok else '#B4202A', fontweight='normal' if ok else 'bold')
    hd = [Line2D([], [], ls='none', marker='D', ms=4.0, mfc='black', mec='black', label='central value'),
          Line2D([], [], ls='none', marker='o', ms=3.0, mfc='0.25', mec='0.25', label='central + one check'),
          Line2D([], [], ls='none', marker='s', ms=3.0, mfc=S.CLASS_DARK['cocrystal'], mec=S.CLASS_DARK['cocrystal'],
                 label='computed combination'),
          Line2D([], [], ls='none', marker='^', ms=3.2, mfc='white', mec='0.35', mew=0.7,
                 label='not in the envelope'),
          Patch(facecolor=col, alpha=0.22, label='budget envelope'),
          Line2D([], [], color='#B4202A', lw=0.8, label='0 (states equal)')]
    fig.legend(handles=hd, loc='lower left', bbox_to_anchor=(1.0 / W, (B + AH + 17.0) / H), ncol=6, fontsize=FS,
               handlelength=1.4, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    fig.text(1.0 / W, (B + AH + 2.0) / H, 'E(ion pair) − E(neutral pair)\nat P4r, kJ mol$^{-1}$ per pair',
             fontsize=FS, ha='left', va='bottom', color='0.3', linespacing=1.1)
    S.save(fig, STEM, W)


if __name__ == '__main__':
    main()
