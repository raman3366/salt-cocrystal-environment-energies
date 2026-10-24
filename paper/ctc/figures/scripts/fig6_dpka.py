"""Fig. 6: the isolated pair, the crystal and the environment term against dpKa (DOUBLE, three panels).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig6_dpka.py

y: dE_PT = E(ion pair) - E(neutral pair) from summary.json fixed_ion_minus_neutral: (a) R1f (PBE0), (b) R5 PBE0+MP2
at P4, (c) dE_env (PBE0).  x: the pre-registered dpKa (crystals[].dpka).  Markers by acid family, filled by observed
class colour; sulfamate open (dpka_off_axis in crystals.json).  Line: least-squares fit of the phosphate family
(5 salts; np.polyfit as analyse.py, slope asserted against secondary.within_family.H3PO4), with the
2.303RT reference slope (secondary.RT2303_kJ_per_pKa, sign: E(ion) - E(neutral) falls as dpKa rises) drawn through
the family's centroid.  perchlorate (dpKa +12.0) is off scale: an arrow marker at the right edge with its value.
Printed rho: Spearman of the phosphate family and of all 13 (secondary.pooled_added_22oct, added 22 Oct).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _ctc as S  # noqa: E402
from figkit import style  # noqa: E402

STEM = 'fig6_dpka'
FS = style.SMALL_PT
FAM_MARKERS = {'H3PO4': ('o', 'dihydrogen phosphate'), 'carboxylic acid': ('s', 'carboxylic acid'),
               'phenol': ('^', 'phenol'), 'sulfanilamide': ('D', 'sulfanilamide (strong acid)')}
QS = [('R1f', 'R1f PBE0', '$R1f$: isolated pair (PBE0)'),
      ('R5_PBE0+MP2', 'R5 PBE0+MP2', '$R5$: crystal (PBE0 + MP2 pair)'),
      ('dE_env_PBE0', 'dE_env PBE0', '$\\Delta E_\\mathrm{env}$ = R5 − R1f (PBE0)')]
XMAX = 3.0


def main():
    plt = S.setup()
    rt = S.G('secondary', 'RT2303_kJ_per_pKa')
    fam = S.G('secondary', 'within_family', 'H3PO4')
    W, H = 175.0, 70.0
    fig = S.figure_mm(W, H)
    L, AW, AH, gx, B = 14.0, 38.0, 42.0, 17.5, 10.5
    for j, (q, sk, title) in enumerate(QS):
        ax = S.axes_mm(fig, L + j * (AW + gx), B, AW, AH)
        ax.set_xlim(-1.8, XMAX)
        ax.axhline(0, color='0.6', lw=0.5, zorder=1)
        ys = []
        for k in S.KEYS:
            c = S.CRY[k]
            x, y = S.V(k, 'dpka'), S.V(k, 'fixed_ion_minus_neutral', q)
            S.mark(k, 'fixed_ion_minus_neutral', q)
            col = S.CLASS_COLOURS[c['observed']]
            m = FAM_MARKERS[c['family']][0]
            off = S.META[k].get('dpka_off_axis', False)
            if x > XMAX:                                          # perchlorate: off scale
                ax.plot(XMAX, y, marker='>', ms=4.6, mfc=col, mec='black', mew=0.5, clip_on=False, zorder=6)
                ax.annotate(S.num(y, 1, sign=True), (XMAX, y), xytext=(4.5, 0), textcoords='offset points', ha='left',
                            va='center', fontsize=FS, color=S.CLASS_DARK['salt'], annotation_clip=False)
                S.mark(k, 'fixed_ion_minus_neutral', q, printed=S.num(y, 1, sign=True))
                S.mark(k, 'dpka', printed=S.num(x, 1, sign=True) + ' (legend)')
                ys.append(y)
                continue
            ax.plot(x, y, marker=m, ms=4.0, mfc='white' if off else col, mec=col if off else 'black',
                    mew=0.9 if off else 0.4, ls='none', zorder=5)
            ys.append(y)
        # phosphate family fit and 2.303RT reference
        g = fam['members']
        assert sorted(g) == sorted(k for k in S.KEYS if S.CRY[k]['family'] == 'H3PO4')
        gx_ = np.array([S.CRY[k]['dpka'] for k in g])
        gy_ = np.array([S.CRY[k]['fixed_ion_minus_neutral'][q] for k in g])
        b1, b0 = np.polyfit(gx_, gy_, 1)
        ref = S.G('secondary', 'within_family', 'H3PO4', sk, 'slope_kJ_per_pKa')
        assert abs(round(b1, 2) - ref) < 1e-9, (q, b1, ref)
        xx = np.array([gx_.min() - 0.15, gx_.max() + 0.15])
        ax.plot(xx, b0 + b1 * xx, color=S.CLASS_DARK['salt'], lw=0.9, zorder=3)
        xc, yc = gx_.mean(), gy_.mean()
        ax.plot(xx, yc - rt * (xx - xc), color='0.35', lw=0.8, ls=(0, (3, 2)), zorder=2)
        rho_f = S.G('secondary', 'within_family', 'H3PO4', sk, 'spearman')
        rho_a = S.G('secondary', 'pooled_added_22oct', sk, 'spearman')
        txt = (f'phosphate: {S.num(ref, 1, sign=True)} per unit, ρ {S.num(rho_f, 1, sign=True)}\n'
               f'all 13: ρ {S.num(rho_a, 2, sign=True)}')
        S.USED[f'secondary.within_family.H3PO4.{sk}.slope_kJ_per_pKa']['printed'].append(S.num(ref, 1, sign=True))
        S.USED[f'secondary.within_family.H3PO4.{sk}.spearman']['printed'].append(S.num(rho_f, 1, sign=True))
        S.USED[f'secondary.pooled_added_22oct.{sk}.spearman']['printed'].append(S.num(rho_a, 2, sign=True))
        lo, hi = min(ys), max(ys)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 3.2 * pad)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top', fontsize=FS, linespacing=1.15,
                zorder=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: S.num(v, 0, sign=v > 0)))
        ax.set_xlabel('ΔpK$_a$ = pK$_a$(BH$^+$) − pK$_a$(HA)')
        if j == 0:
            ax.set_ylabel('$\\Delta E_\\mathrm{PT}$ (kJ mol$^{-1}$)')
        S.panel_letter(fig, L + j * (AW + gx) - 11.0, B + AH + 2.2, 'abc'[j], title=title)
    S.USED['secondary.RT2303_kJ_per_pKa']['printed'].append('slope of the dashed line')
    hd = [Line2D([], [], ls='none', marker=m, ms=4.0, mfc='0.6', mec='black', mew=0.4, label=lab)
          for m, lab in FAM_MARKERS.values()]
    hd += [Line2D([], [], ls='none', marker='o', ms=4.0, mfc=S.CLASS_COLOURS['salt'], mec='black', mew=0.4,
                  label='salt'),
           Line2D([], [], ls='none', marker='o', ms=4.0, mfc=S.CLASS_COLOURS['cocrystal'], mec='black', mew=0.4,
                  label='cocrystal'),
           Line2D([], [], ls='none', marker='o', ms=4.0, mfc='white', mec=S.CLASS_COLOURS['salt'], mew=0.9,
                  label='sulfamate (ΔpK$_a$ off axis)'),
           Line2D([], [], ls='none', marker='>', ms=4.6, mfc=S.CLASS_COLOURS['salt'], mec='black', mew=0.5,
                  label='SAPC, ΔpK$_a$ +12.0 (off scale; value printed)'),
           Line2D([], [], color=S.CLASS_DARK['salt'], lw=0.9, label='phosphate-family fit'),
           Line2D([], [], color='0.35', lw=0.8, ls=(0, (3, 2)), label=f'2.303RT = {S.num(rt, 1)} per unit')]
    fig.legend(handles=hd, loc='lower left', bbox_to_anchor=(1.0 / W, (B + AH + 7.0) / H), ncol=4, fontsize=FS,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    S.save(fig, STEM, W)


if __name__ == '__main__':
    main()
