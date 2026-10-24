"""Fig. 2 (main): the isolated pair, the environment term and the crystal per crystal, R5 = R1f + dE_env (DOUBLE).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig2_decomposition.py

Per crystal three horizontal bars on one linear axis, all as dE_PT = E(ion pair) - E(neutral pair) (kJ/mol per
transferring pair; > 0: the neutral pair is lower):
    R1f      the pair in vacuum at the crystal geometry, at the pair level of R5 (R1f.MP2 = PBE0 + [MP2 - PBE0] of the
             pair); the PBE0 value (fixed_ion_minus_neutral.R1f) as a black tick
    dE_env   R5 - R1f at PBE0 (fixed_ion_minus_neutral.dE_env_PBE0); the pair correction enters R5 and R1f alike and
             cancels (budget.dE_env.none_reason.level_spread), so R1f(MP2) + dE_env = R5(PBE0+MP2) exactly (asserted)
    R5       periodic PBE0 + DF-MP2 pair correction at P4 (fixed_ion_minus_neutral.R5_PBE0+MP2), coloured by the
             observed class
Whiskers: budget envelopes (budget.dE_env, budget.R5_P4; stored other - observed, drawn as dE_PT).  R1f has no budget
in summary.json (no whisker).  dapsone and pcplut, whose resolution is the margin at P4r (resolution.quantity), also
show that margin (budget.margin) as an open diamond with its envelope.  Salts first, then cocrystals, each by dpKa.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _ctc as S  # noqa: E402
from figkit import style  # noqa: E402

STEM = 'fig2_decomposition'
W_MM = 175.0
FS = style.SMALL_PT
GREY = '#8A8A8A'


def rows():
    out = []
    for k in S.by_dpka(S.SALTS) + S.by_dpka(S.COCRYSTALS):
        r1 = S.PT(k, 'R1f', 'MP2')
        r1_pbe0 = S.V(k, 'fixed_ion_minus_neutral', 'R1f')
        de = S.V(k, 'fixed_ion_minus_neutral', 'dE_env_PBE0')
        r5 = S.V(k, 'fixed_ion_minus_neutral', 'R5_PBE0+MP2')
        assert abs(r1 + de - r5) < 2e-3, (k, r1 + de - r5)
        assert abs(S.PT(k, 'budget', 'R5_P4', 'central') - r5) < 1e-9
        assert abs(S.PT(k, 'budget', 'dE_env', 'central') - de) < 1e-9
        res = S.V(k, 'resolution')
        r = dict(k=k, cls=S.CRY[k]['observed'], set=S.CRY[k]['set'], dpka=S.V(k, 'dpka'), r1=r1, r1_pbe0=r1_pbe0,
                 de=de, r5=r5, env_r5=sorted(S.PT(k, 'budget', 'R5_P4', 'envelope')),
                 env_de=sorted(S.PT(k, 'budget', 'dE_env', 'envelope')), verdict=res['verdict'], margin=None)
        if res['quantity'] == 'margin':
            m = S.PT(k, 'budget', 'margin', 'central')
            assert abs(m - S.PT(k, 'resolution', 'central')) < 1e-9
            r['margin'] = (m, sorted(S.PT(k, 'budget', 'margin', 'envelope')))
        out.append(r)
    return out


def main():
    plt = S.setup()
    R = rows()
    n_s = len(S.SALTS)
    pitch, gap = 7.2, 5.0                      # mm per crystal, extra gap between the groups
    L, AW, top = 37.0, 98.0, 12.0
    ys = [i * pitch + (gap if i >= n_s else 0) for i in range(len(R))]
    AH = ys[-1] + pitch
    H = AH + top + 13.0
    fig = S.figure_mm(W_MM, H)
    ax = S.axes_mm(fig, L, 11.0, AW, AH)
    ax.set_xlim(-180, 135)
    ax.set_ylim(AH, 0)
    ax.set_yticks([])
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    ax.set_xticks(range(-160, 121, 40))
    ax.set_xticks(range(-180, 131, 10), minor=True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
    ax.set_xlabel('$\\Delta E_\\mathrm{PT}$ = E(ion pair) − E(neutral pair)  (kJ mol$^{-1}$ per transferring pair)')
    ax.grid(axis='x', which='major', color='0.90', lw=0.5, zorder=0)
    ax.axhline(n_s * pitch + gap / 2, color='0.55', lw=0.5, ls=(0, (3, 2)))
    bh = 1.55
    off = {'r1': 1.35, 'de': 3.15, 'r5': 4.95}
    ek = dict(color='black', lw=0.6, capsize=1.6, capthick=0.6, zorder=6)
    for r, y0 in zip(R, ys):
        k, col = r['k'], S.CLASS_COLOURS[r['cls']]
        ax.barh(y0 + off['r1'], r['r1'], height=bh, color='white', edgecolor='0.25', lw=0.6, zorder=3)
        ax.plot([r['r1_pbe0']] * 2, [y0 + off['r1'] - bh / 2 - 0.25, y0 + off['r1'] + bh / 2 + 0.25],
                color='black', lw=0.9, zorder=5, solid_capstyle='butt')
        ax.barh(y0 + off['de'], r['de'], height=bh, color='0.62', edgecolor='0.25', lw=0.6, zorder=3)
        ax.barh(y0 + off['r5'], r['r5'], height=bh, color=col, edgecolor='0.15', lw=0.6, zorder=3)
        for q, e in (('de', r['env_de']), ('r5', r['env_r5'])):
            v = r[q]
            ax.errorbar([v], [y0 + off[q]], xerr=[[v - e[0]], [e[1] - v]], fmt='none', **ek)
        for p in (('R1f', 'MP2'), ('fixed_ion_minus_neutral', 'R1f'), ('fixed_ion_minus_neutral', 'dE_env_PBE0'),
                  ('fixed_ion_minus_neutral', 'R5_PBE0+MP2'), ('budget', 'R5_P4', 'envelope'),
                  ('budget', 'dE_env', 'envelope')):
            S.mark(k, *p)
        # left labels: name, set, dpKa
        ax.text(-0.335, y0 + 3.15, S.NAME[k], transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=style.TICK_PT,
                color=S.CLASS_DARK[r['cls']], fontweight='bold')
        ax.text(-0.155, y0 + 3.15, S.SET_TAG[r['set']], transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=FS, color='0.35', style='italic')
        dp = S.num(r['dpka'], 2, sign=True)
        ax.text(-0.012, y0 + 3.15, dp, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS)
        S.mark(k, 'dpka', printed=dp)
        # value of R5 at the bar end (outside the whisker), and the margin of the unresolved crystals
        e = r['env_r5']
        s5 = S.num(r['r5'], 1, sign=True)
        if r['r5'] < 0:
            ax.text(e[0] - 2.5, y0 + off['r5'], s5, ha='right', va='center', fontsize=FS, color=S.CLASS_DARK[r['cls']])
        else:
            xr = max(e[1], r['margin'][1][1] if r['margin'] else e[1])
            ax.text(xr + 2.5, y0 + off['r5'], s5, ha='left', va='center', fontsize=FS, color=S.CLASS_DARK[r['cls']])
        S.mark(k, 'fixed_ion_minus_neutral', 'R5_PBE0+MP2', printed=s5)
        if r['margin']:
            m, me = r['margin']
            ym = y0 + off['r5'] + 0.0
            ax.errorbar([m], [ym], xerr=[[m - me[0]], [me[1] - m]], fmt='D', ms=3.0, mfc='white', mec='black',
                        mew=0.6, ecolor='black', elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=7)
            S.mark(k, 'budget', 'margin', 'central')
            S.mark(k, 'budget', 'margin', 'envelope')
        # right column: R1f and dE_env values, resolution
        s1, sd = S.num(r['r1'], 1, sign=True), S.num(r['de'], 1, sign=True)
        ax.text(1.085, y0 + 3.15, s1, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS)
        ax.text(1.19, y0 + 3.15, sd, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS,
                color='0.30')
        S.mark(k, 'R1f', 'MP2', printed=s1)
        S.mark(k, 'fixed_ion_minus_neutral', 'dE_env_PBE0', printed=sd)
        ok = r['verdict'].startswith('resolved')
        S.mark(k, 'resolution', printed=r['verdict'])
        ax.text(1.215, y0 + 3.15, 'resolved' if ok else 'unresolved', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=FS, color='0.35' if ok else '#B4202A',
                fontweight='normal' if ok else 'bold')
    # group labels inside the plot, on the empty side
    ax.text(-176, 0.25, 'salts (observed: ion pair)', ha='left', va='top', fontsize=style.TICK_PT,
            color=S.CLASS_DARK['salt'], fontweight='bold')
    ax.text(-176, n_s * pitch + gap - 0.1, 'cocrystals (observed: neutral pair)', ha='left', va='top',
            fontsize=style.TICK_PT, color=S.CLASS_DARK['cocrystal'], fontweight='bold')
    ax.text(3, -1.4, 'neutral pair lower →', ha='left', va='bottom', fontsize=FS, color='0.35')
    ax.text(-3, -1.4, '← ion pair lower', ha='right', va='bottom', fontsize=FS, color='0.35')
    # column headers
    yh = -1.4
    for x, s, ha in ((-0.335, 'crystal', 'left'), (-0.155, 'set', 'left'), (-0.012, 'ΔpK$_a$', 'right'),
                     (1.085, '$R1f$', 'right'), (1.19, '$\\Delta E_\\mathrm{env}$', 'right'),
                     (1.215, 'R5 budget', 'left')):
        ax.text(x, yh, s, transform=ax.get_yaxis_transform(), ha=ha, va='bottom', fontsize=FS, color='0.25')
    # legend above
    hd = [Patch(facecolor='white', edgecolor='0.25', lw=0.6, label='$R1f$: isolated pair (PBE0 + MP2 pair)'),
          Line2D([], [], color='black', lw=0.9, marker='|', ms=5, ls='none', label='$R1f$ at PBE0'),
          Patch(facecolor='0.62', edgecolor='0.25', lw=0.6, label='$\\Delta E_\\mathrm{env}$ = R5 − R1f (PBE0)'),
          Patch(facecolor=S.CLASS_COLOURS['salt'], edgecolor='0.15', lw=0.6, label='$R5$ salt'),
          Patch(facecolor=S.CLASS_COLOURS['cocrystal'], edgecolor='0.15', lw=0.6, label='$R5$ cocrystal'),
          Line2D([], [], color='black', lw=0.6, marker='|', ms=3, label='budget envelope'),
          Line2D([], [], ls='none', marker='D', ms=3.0, mfc='white', mec='black', mew=0.6,
                 label='margin at P4r (resolution quantity)')]
    leg = fig.legend(handles=hd, loc='lower left', bbox_to_anchor=(L / W_MM - 0.2 + 0.0, (11.0 + AH + 2.2) / H),
                     ncol=4, fontsize=FS, handlelength=1.6, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.0)
    leg.set_bbox_to_anchor((1.0 / W_MM, (11.0 + AH + 2.6) / H), transform=fig.transFigure)
    S.save(fig, STEM, W_MM, extra={'derived': 'R1f(MP2) + dE_env(PBE0) = R5(PBE0+MP2) asserted to 2e-3 per crystal'})


if __name__ == '__main__':
    main()
