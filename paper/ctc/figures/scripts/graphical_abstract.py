"""Graphical abstract (2.5:1, at least 1328 x 531 px): the isolated pair's cost R1f against the lattice's lowering of the
ion pair -dE_env, both PBE0; the diagonal R5 = 0 separates the salts (above) from the cocrystals (below).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/graphical_abstract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import _ctc as S  # noqa: E402
from figkit import style  # noqa: E402

STEM = 'graphical_abstract'


def main():
    plt = S.setup()
    W, H = 135.0, 54.0
    fig = S.figure_mm(W, H)
    ax = S.axes_mm(fig, 14.0, 9.5, 52.0, 41.0)
    for k in S.KEYS:
        x = S.V(k, 'fixed_ion_minus_neutral', 'R1f')
        y = -S.V(k, 'fixed_ion_minus_neutral', 'dE_env_PBE0')
        S.mark(k, 'fixed_ion_minus_neutral', 'R1f')
        S.mark(k, 'fixed_ion_minus_neutral', 'dE_env_PBE0')
        cls = S.CRY[k]['observed']
        ax.plot(x, y, 'o', ms=4.2, color=S.CLASS_COLOURS[cls], mec='white', mew=0.4, zorder=3)
    lim = (0, 175)
    xx = np.array(lim)
    ax.fill_between(xx, xx, 175, color=S.CLASS_COLOURS['salt'], alpha=0.07, lw=0)
    ax.fill_between(xx, 0, xx, color=S.CLASS_COLOURS['cocrystal'], alpha=0.07, lw=0)
    ax.plot(xx, xx, color='0.35', lw=0.7, ls='--')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel('cost of the ion pair, isolated (kJ mol$^{-1}$)', fontsize=style.TICK_PT)
    ax.set_ylabel('lowering by the crystal (kJ mol$^{-1}$)', fontsize=style.TICK_PT)
    ax.text(12, 160, 'salts', color=S.CLASS_DARK['salt'], fontsize=style.BASE_PT, fontweight='bold', va='top')
    ax.text(165, 12, 'cocrystals', color=S.CLASS_DARK['cocrystal'], fontsize=style.BASE_PT, fontweight='bold',
            ha='right')
    t = fig.text
    t(71.0 / W, 47.0 / H, 'The lattice favours the ion pair', fontsize=9.5, fontweight='bold', va='top')
    t(71.0 / W, 41.5 / H, 'in all 13 acid–base crystals', fontsize=9.5, fontweight='bold', va='top')
    lines = ['Isolated pair: neutral in every crystal',
             'Salts: the crystal outweighs the pair',
             'Cocrystals: it cancels 75–95% of the cost',
             '     and leaves only a few kJ mol$^{-1}$',
             'Classification near ΔpK$_a$ = 0 needs',
             '     an error budget per crystal']
    for i, s in enumerate(lines):
        t(71.0 / W, (33.5 - 4.6 * i) / H, s, fontsize=style.TICK_PT, va='top')
    S.save(fig, STEM, W)


if __name__ == '__main__':
    main()
