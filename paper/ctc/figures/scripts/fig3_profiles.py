"""Fig. 3: periodic proton-transfer profiles at every level (DOUBLE); SI: fig3_profiles_all (all 13 crystals).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig3_profiles.py

E(delta) - E(P0) per transferring pair from summary.json crystals[].profile, delta = d(D-H) - d(H...A) with D the
donor of the observed state (delta < 0: observed side).  Levels PBE, PBE0, PBE0+MP2, PBE0+SCS-MP2,
PBE0+wB97X-D3(BJ).  Hysteresis pairs (two relaxed geometries at one constrained d(D-H)) reduced to the lower energy
exactly as optc/src/analyse.py lowest_per_target (copied below).  Curves: PCHIP through the reduced points of the
constrained path (P0-P4 and X); P4r, a relaxed geometry (frame relaxed too), is drawn as a star and not joined, so the
curve does not jump between P3 and P4r where they lie 0.01 A apart (dapsone).  analyse.py classifies on the PCHIP
with P4r included; the class is not redrawn here.  Markers: pre-registered P0-P4 filled, extra points X (19-20 Oct) open, P4r (the relax from P4)
a star.  Grey band: the |delta| <= 0.1 A window of the classification rule.
Main figure: saph, dapsone, pcplut, perchlorate.  SI: all 13, ordered by dpKa.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

import _ctc as S  # noqa: E402
from figkit import style  # noqa: E402

FS = style.SMALL_PT
MAIN = ['saph', 'dapsone', 'pcplut', 'perchlorate']


def lowest_per_target(prof, lev):
    """optc/src/analyse.py lowest_per_target, verbatim logic."""
    best = {}
    for pt, p in prof.items():
        e, dl, t = p.get(lev), p.get('delta'), p.get('target_DH')
        if e is None or dl is None:
            continue
        k = ('t', round(t, 3)) if t else ('d', round(dl, 3))
        if k not in best or e < best[k][1]:
            best[k] = (dl, e, pt)
    return sorted(best.values())


def sides(k):
    """Labels of the two sides: delta < 0 is the observed state (asserted from the P0 point)."""
    c = S.CRY[k]
    assert S.V(k, 'profile', 'P0', 'delta') < -0.1
    return ('ion pair', 'neutral pair') if c['observed'] == 'salt' else ('neutral pair', 'ion pair')


def draw(ax, k, small=False):
    prof = S.V(k, 'profile')
    S.mark(k, 'profile')
    lo, hi = 0.0, 0.0
    ax.axvspan(-0.1, 0.1, color='0.92', lw=0, zorder=0)
    ax.axhline(0, color='0.6', lw=0.5, zorder=1)
    ms = 2.6 if small else 3.2
    for lev in S.LEVELS:
        pts = lowest_per_target(prof, lev)
        d = np.array([p[0] for p in pts])
        e = np.array([p[1] for p in pts])
        col = S.LEVEL_COLOURS[lev]
        on = np.array([p[2] != 'P4r' for p in pts])          # P4r (relaxed frame) is marked, not joined
        x = np.linspace(d[on][0], d[on][-1], 400)
        ax.plot(x, PchipInterpolator(d[on], e[on])(x), color=col, lw=0.8 if small else 0.9, zorder=3)
        for dl, en, pt in pts:
            extra = prof[pt].get('extra', False)
            if pt == 'P4r':
                ax.plot(dl, en, marker='*', ms=ms * 1.9, mfc=col, mec='black', mew=0.4, ls='none', zorder=5)
            else:
                ax.plot(dl, en, marker=S.LEVEL_MARKERS[lev], ms=ms * (0.85 if extra else 1.0),
                        mfc='white' if extra else col, mec=col, mew=0.6, ls='none', zorder=4)
        lo, hi = min(lo, e.min()), max(hi, e.max())
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.6 * pad)
    ax.set_xlim(-0.9, 0.85)
    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.xaxis.set_major_formatter(lambda v, p: S.num(v, 1))
    ax.yaxis.set_major_formatter(lambda v, p: S.num(v, 0))
    a, b = sides(k)
    ax.text(0.03, 0.97, a, transform=ax.transAxes, ha='left', va='top', fontsize=FS, color='0.3')
    ax.text(0.97, 0.97, b, transform=ax.transAxes, ha='right', va='top', fontsize=FS, color='0.3')
    c = S.CRY[k]
    dp = S.num(S.V(k, 'dpka'), 2, sign=True)
    t = (f"{S.NAME[k]} · {S.SET_TAG[c['set']]} · ΔpK$_a$ {dp}" if small else
         f"{S.NAME[k]}  ({c['observed']}, {S.SET_TAG[c['set']].rstrip('.')}, ΔpK$_a$ {dp})")
    S.mark(k, 'dpka', printed=S.num(c['dpka'], 2, sign=True))
    ax.set_title(t, fontsize=FS if small else style.BASE_PT, color=S.CLASS_DARK[c['observed']],
                 loc='left', pad=2.0)


def legend(fig, x, y, ncol):
    hd = [Line2D([], [], color=S.LEVEL_COLOURS[l], lw=0.9, marker=S.LEVEL_MARKERS[l], ms=3.2,
                 label=S.LEVEL_LABELS[l]) for l in S.LEVELS]
    hd += [Line2D([], [], ls='none', marker='o', ms=3.0, mfc='white', mec='0.3', mew=0.6, label='extra point X'),
           Line2D([], [], ls='none', marker='*', ms=6.0, mfc='0.6', mec='black', mew=0.4, label='P4r (relaxed from P4)')]
    W, H = fig._mm
    fig.legend(handles=hd, loc='lower left', bbox_to_anchor=(x / W, y / H), ncol=ncol, fontsize=FS,
               handlelength=1.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)


def main_fig():
    S.USED.clear()
    fig = S.figure_mm(175.0, 122.0)
    L, AW, AH, gx, gy, B = 13.0, 73.0, 40.0, 14.0, 15.0, 11.0
    for i, k in enumerate(MAIN):
        r, c = divmod(i, 2)
        ax = S.axes_mm(fig, L + c * (AW + gx), B + (1 - r) * (AH + gy), AW, AH)
        draw(ax, k)
        if r == 1:
            ax.set_xlabel('$\\delta$ = d(D–H) − d(H···A)  (Å)')
        if c == 0:
            ax.set_ylabel('E − E(P0)  (kJ mol$^{-1}$)')
        S.panel_letter(fig, L + c * (AW + gx) - 11.5, B + (1 - r) * (AH + gy) + AH + 2.0, 'abcd'[i])
    legend(fig, 2.0, B + 2 * AH + gy + 7.5, 4)
    S.save(fig, 'fig3_profiles', 175.0)


def all_fig():
    S.USED.clear()
    keys = S.by_dpka(S.KEYS)
    ncol, AW, AH, gx, gy, L, B = 4, 33.5, 30.0, 8.5, 13.5, 11.0, 10.0
    nrow = 4
    fig = S.figure_mm(175.0, B + nrow * AH + (nrow - 1) * gy + 9.0)
    for i, k in enumerate(keys):
        r, c = divmod(i, ncol)
        ax = S.axes_mm(fig, L + c * (AW + gx), B + (nrow - 1 - r) * (AH + gy), AW, AH)
        draw(ax, k, small=True)
        ax.tick_params(labelsize=FS)
        if r == nrow - 1 or i + ncol >= len(keys):
            ax.set_xlabel('$\\delta$ (Å)', fontsize=FS)
        if c == 0:
            ax.set_ylabel('E − E(P0) (kJ mol$^{-1}$)', fontsize=FS)
    # legend in the empty last slots
    legend(fig, L + 1 * (AW + gx) + 3.0, B + 4.0, 2)
    S.save(fig, 'fig3_profiles_all', 175.0)


if __name__ == '__main__':
    S.setup()
    main_fig() if "--si" not in sys.argv else None
    all_fig()
