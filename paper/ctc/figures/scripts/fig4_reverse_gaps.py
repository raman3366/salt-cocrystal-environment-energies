"""Fig. 4: embedded-pair verdicts depend on whose environment is used; the periodic closure cycle (DOUBLE).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig4_reverse_gaps.py

(a) Vertical gaps of the pair in a point-charge field (summary.json crystals[].reverse): forward = in the observed
    state's environment (embedded_single_site_kJ), reverse = in the other state's environment (reverse_kJ), and their
    linear-response mean (lr_mean_kJ), for the three charge models (REPEAT, self-consistent in-crystal CHELPG 'sc',
    gas-phase CHELPG) and, for the 5 salts where it was run, the embedded site cluster (site_repeat, REPEAT).
    This panel is a classification diagnostic and keeps the stored sign: other state - observed state
    (> 0: the observed state is lower in that environment).  Counts printed are recomputed here from the plotted
    values (reverse < 0, LR mean < 0).
(b) The periodic closure cycle R1f -> R4 -> T1 -> N_vert -> R5 at PBE0 (closure.cycle_pbe0; each step a difference,
    sum = R5 asserted), drawn cumulatively as dE_PT = E(ion pair) - E(neutral pair) (stored other - observed,
    converted with the summary.json factor of _ctc.sign_pt).  R4: embedded cluster (single site); T1: periodic single
    site (one pair of the Z moved); N_vert: every pair moved, vertical; R5: relaxed periodic crystal.  Line ends are
    named in one column right of the panel (end_column), every name joined to its point by a leader, so no leader
    crosses a name (figkit plots.end_labels starts undisplaced names closer to the axis than the leaders end).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.lines import Line2D  # noqa: E402

import _ctc as S  # noqa: E402
import numpy as np  # noqa: E402
from figkit import style  # noqa: E402
from figkit.labels import freeze_layout  # noqa: E402

STEM = 'fig4_reverse_gaps'
FS = style.SMALL_PT
MODELS = [('pair_repeat', 'REPEAT', '#0B4F8A'), ('pair_sc', 'in-crystal CHELPG', '#009E73'),
          ('pair_gas', 'gas-phase CHELPG', '#D55E00'), ('site_repeat', 'site cluster (REPEAT)', 'black')]
STEPS = [('R1f', 'R1f'), ('R4_minus_R1f', 'R4'), ('T1_minus_R4', 'T1'), ('Nv_minus_T1', 'N$_\\mathrm{vert}$'),
         ('R5_minus_Nv', 'R5')]


def end_column(ax, items, x_pt=9.0, gap_pt=0.6, fontsize=None):
    """Names of the line ends in one column x_pt right of the last point, spread vertically so that none overlaps
    (the 1-D repulsion of figkit plots.end_labels), each joined to its point by a straight leader that stops 1.5 pt
    short of the column.  items: (text, x_last, y_last, colour)."""
    fig = ax.figure
    freeze_layout(fig)
    r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    arts = [ax.annotate(t, (x, y), xytext=(x_pt, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=fontsize or FS, color=c, annotation_clip=False) for t, x, y, c in items]
    want = np.array([ax.transData.transform((x, y))[1] for _, x, y, _ in items])
    h = np.array([a.get_window_extent(r).height for a in arts]) + gap_pt * ppt
    order = np.argsort(want)
    y, hh = want[order].astype(float), h[order]
    lo, hi = ax.bbox.y0 + hh[0] / 2, ax.bbox.y1 - hh[-1] / 2
    for _ in range(500):
        moved = False
        for k in range(1, len(y)):
            need = (hh[k] + hh[k - 1]) / 2
            if y[k] - y[k - 1] < need - 1e-6:
                d = (need - (y[k] - y[k - 1])) / 2
                y[k] += d
                y[k - 1] -= d
                moved = True
        y += max(0.0, lo - y[0]) - max(0.0, y[-1] - hi)
        if not moved:
            break
    for k, i in enumerate(order):
        dy = (y[k] - want[i]) / ppt
        arts[i].xyann = (x_pt, dy)
        p0 = ax.transData.transform(arts[i].xy)
        seg = [p0 + np.array([1.5 * ppt, 0.0]), p0 + np.array([(x_pt - 1.5) * ppt, dy * ppt])]
        ax.plot(*ax.transData.inverted().transform(seg).T, color=arts[i].get_color(), lw=0.4, clip_on=False,
                zorder=1)
    return arts


def main():
    plt = S.setup()
    keys = S.by_dpka(S.KEYS)
    W, H = 175.0, 112.0
    fig = S.figure_mm(W, H)
    pitch = 6.4
    AH = pitch * len(keys)
    B = 11.0
    ax = S.axes_mm(fig, 22.0, B, 80.0, AH)
    ax.set_ylim(AH, 0)
    ax.set_yticks([])
    ax.set_xlim(-85, 195)
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    ax.grid(axis='x', color='0.9', lw=0.5, zorder=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
    ax.set_xlabel('vertical gap, other state − observed  (kJ mol$^{-1}$)')
    wins = {m: [0, 0] for m, _, _ in MODELS}
    lrneg = {m: [] for m, _, _ in MODELS}
    for i, k in enumerate(keys):
        y0 = i * pitch
        cls = S.CRY[k]['observed']
        ax.text(-0.02, y0 + pitch / 2, S.NAME[k], transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=style.TICK_PT, color=S.CLASS_DARK[cls], fontweight='bold')
        if i:
            ax.axhline(y0, color='0.88', lw=0.4, zorder=0)
        rv = S.V(k, 'reverse')
        for j, (m, _, col) in enumerate(MODELS):
            if m not in rv:
                continue
            y = y0 + 1.1 + j * 1.4
            f = S.V(k, 'reverse', m, 'embedded_single_site_kJ')
            r = S.V(k, 'reverse', m, 'reverse_kJ')
            lr = S.V(k, 'reverse', m, 'lr_mean_kJ')
            assert abs((f + r) / 2 - lr) < 2e-3, (k, m)
            for q in ('embedded_single_site_kJ', 'reverse_kJ', 'lr_mean_kJ'):
                S.mark(k, 'reverse', m, q)
            ax.plot([r, f], [y, y], color=col, lw=0.7, zorder=2)
            ax.plot(f, y, 'o', ms=2.8, mfc=col, mec=col, mew=0.5, zorder=4)
            ax.plot(r, y, 'o', ms=2.8, mfc='white', mec=col, mew=0.7, zorder=4)
            ax.plot(lr, y, '|', ms=4.2, mew=1.1, color=col, zorder=5)
            wins[m][0] += r < 0
            wins[m][1] += 1
            if lr < 0:
                lrneg[m].append(k)
    # counts (recomputed from the plotted values) go into the legend labels
    hd = [Line2D([], [], color=c, lw=0.7, marker='o', ms=2.8,
                 label=f'{l}: reverse < 0 in {wins[m][0]}/{wins[m][1]}') for m, l, c in MODELS]
    hd += [Line2D([], [], ls='none', marker='o', ms=2.8, mfc='0.3', mec='0.3', label='forward (observed env.)'),
           Line2D([], [], ls='none', marker='o', ms=2.8, mfc='white', mec='0.3', label='reverse (other env.)'),
           Line2D([], [], ls='none', marker='|', ms=4.2, mew=1.1, color='0.3', label='mean of the two (LR)')]
    fig.legend(handles=hd, loc='lower left', bbox_to_anchor=(9.0 / W, (B + AH + 2.5) / H), ncol=2, fontsize=FS,
               handlelength=1.5, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    S.panel_letter(fig, 1.0, B + AH + 14.0, 'a')

    # (b) closure cycle, salts and cocrystals
    bx, bw = 116.0, 38.0
    groups = [(S.by_dpka(S.SALTS), 'salts'), (S.by_dpka(S.COCRYSTALS), 'cocrystals')]
    hb = [(AH + 3.0) * 0.62, (AH + 3.0) * 0.38 - 10.0]
    yb = [B + hb[1] + 10.0, B]
    for (ks, name), h, y in zip(groups, hb, yb):
        a2 = S.axes_mm(fig, bx, y, bw, h)
        a2.axhline(0, color='black', lw=0.6, zorder=1)
        a2.set_xlim(-0.25, 4.25)
        a2.set_xticks(range(5))
        a2.set_xticklabels([lab for _, lab in STEPS] if name == 'cocrystals' else [])
        a2.set_ylabel('$\\Delta E_\\mathrm{PT}$ (kJ mol$^{-1}$)')
        a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
        a2.grid(axis='x', color='0.9', lw=0.5, zorder=0)
        items = []
        for k in ks:
            cyc = S.V(k, 'closure', 'cycle_pbe0')
            assert cyc['sum_equals_R5'] and abs(sum(cyc[s] for s, _ in STEPS) - cyc['R5']) < 2e-3, k
            s = S.sign_pt(k)
            cum, ys = 0.0, []
            for st, _ in STEPS:
                cum += cyc[st]
                ys.append(s * cum)
                S.USED.setdefault(S._key(k, ('closure', 'cycle_pbe0', st)),
                                  {'value': cyc[st], 'printed': ['plotted (cumulative)'],
                                   'note': f'stored other - observed; drawn x{s:+d} as E(ion) - E(neutral)'})
            assert abs(ys[-1] - S.V(k, 'fixed_ion_minus_neutral', 'R5_PBE0')) < 2e-3, k
            col = S.CLASS_COLOURS[S.CRY[k]['observed']]
            a2.plot(range(5), ys, color=col, lw=0.7, marker='o', ms=1.9, zorder=3, alpha=0.9)
            items.append((S.NAME[k], 4, ys[-1], S.CLASS_DARK[S.CRY[k]['observed']]))
        a2.text(0.03 if name == 'salts' else 0.97, 0.03 if name == 'salts' else 0.96, name, transform=a2.transAxes,
                ha='left' if name == 'salts' else 'right', va='bottom' if name == 'salts' else 'top', fontsize=FS, color=S.CLASS_DARK[ks[0] and
                                                                                           S.CRY[ks[0]]['observed']],
                fontweight='bold')
        a2._items = items
        if name == 'cocrystals':
            a2.set_ylim(-8, 100)
            a2.set_yticks([0, 25, 50, 75, 100])
        if name == 'salts':
            S.panel_letter(fig, bx - 13.0, y + h + 3.0, 'b', title='closure cycle (PBE0)')
        a2.tick_params(axis='x', labelsize=FS)
        groups_axes = getattr(fig, '_b', [])
        groups_axes.append(a2)
        fig._b = groups_axes
    for a2 in fig._b:
        end_column(a2, a2._items)
    S.save(fig, STEM, W, extra={'derived': {'counts': {lab: f'{wins[m][0]}/{wins[m][1]}' for m, lab, _ in MODELS},
                                            'lr_mean_below_0': lrneg}})


if __name__ == '__main__':
    main()
