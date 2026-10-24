"""Reduced density gradient (RDG) versus sign(λ2)ρ scatter plots.

Public API
    NCI_RANGE = (-0.035, 0.02)                 colour range (a.u.) shared with the 3D NCI isosurfaces
    nci_cmap()                                 blue–green–red map (VMD 'BGR': green at the middle of NCI_RANGE)
    load_rdg(path, xlim=(-0.05, 0.05), ylim=(0, 2), with_xyz=False) -> dict(x, y[, xyz])
    read_spikes(path)                          -> list[dict(value, rdg, n, type)]   (nci_spikes.csv)
    identify_spikes(data, spikes, symbols, xyz, labels, fragments=None, ...) -> list[dict(... label, atoms)]
    plot_rdg(ax, x, y, xlim=(-0.05, 0.05), ylim=(0, 2), spikes=None, key=True, ...) -> dict
    rdg_figure(panels, width=None, ...)        -> (fig, axes)   one or more side-by-side panels, shared y axis

rdg_scatter.txt (Multiwfn function 20 export) has five columns: X Y Z (Å), sign(λ2)ρ (a.u.), RDG. The companion
rdg_scatter.npz stores only the (x, y) pairs with -0.25 < x < 0.06 and y < 2.5, which covers the default window,
so it is used whenever no coordinates are needed and the requested window lies inside that filter.
"""
from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import transforms as mtrans

from . import style, geometry as geo, qa
from .labels import LabelPlacer, freeze_layout

NCI_RANGE = (-0.035, 0.02)
REGION_BOUNDS = (-0.015, 0.01)          # H-bond | vdW | steric, as used when nci_spikes.csv was written
NPZ_FILTER = (-0.25, 0.06, 2.5)         # x_min, x_max, y_max of rdg_scatter.npz
XLABEL = r'sign($\lambda_2$)$\rho$ (a.u.)'


def nci_cmap():
    """Blue (strong attraction) – green (weak, near zero) – red (repulsion), as VMD's BGR scale over NCI_RANGE."""
    return LinearSegmentedColormap.from_list('nci_bgr', [(0.0, '#0000FF'), (0.5, '#00D000'), (1.0, '#FF0000')])


@functools.lru_cache(maxsize=4)
def _read_txt(path):
    import pandas as pd
    a = pd.read_csv(path, sep=r'\s+', header=None, engine='c', dtype=np.float64).values
    return a


def _resolve(path):
    p = Path(path)
    if p.is_dir():
        p = p / 'rdg_scatter.txt'
    return p


def load_rdg(path, xlim=(-0.05, 0.05), ylim=(0.0, 2.0), with_xyz=False):
    """Scatter data inside the window. path: rdg_scatter.txt, .npz, or the results directory."""
    p = _resolve(path)
    npz = p.with_suffix('.npz')
    inside = xlim[0] >= NPZ_FILTER[0] and xlim[1] <= NPZ_FILTER[1] and ylim[1] <= NPZ_FILTER[2]
    if not with_xyz and npz.exists() and inside:
        z = np.load(npz)
        x, y = z['x'].astype(float), z['y'].astype(float)
        m = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        return dict(x=x[m], y=y[m], n_total=int(z['n_total']), source=str(npz))
    if not p.exists():
        raise FileNotFoundError(p)
    a = _read_txt(str(p))
    x, y = a[:, 3], a[:, 4]
    m = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    out = dict(x=x[m], y=y[m], n_total=len(a), source=str(p))
    if with_xyz:
        out['xyz'] = a[m, :3]
    return out


def read_spikes(path):
    """nci_spikes.csv -> list of dicts (value = sign(λ2)ρ of the spike, rdg = its minimum RDG)."""
    p = Path(path)
    if p.is_dir():
        p = p / 'nci_spikes.csv'
    import csv
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            out.append(dict(value=float(r['sign_l2_rho_au']), rdg=float(r['min_RDG']), n=int(r['n_points']),
                            type=r['type']))
    return out


def identify_spikes(path, spikes, symbols, xyz, labels, fragments=None, half_width=0.0006, rdg_slack=0.03,
                    unique=True):
    """Attach an interaction label to each spike from the grid points that form it.

    For each spike, the grid points with |sign(λ2)ρ − value| <= half_width and RDG <= min + rdg_slack are located
    (rdg_scatter.txt coordinates are in Å); the lowest-RDG point is assigned to the two nearest atoms. A pair
    that matches a detected hydrogen bond becomes 'N5–H5···O7'; a point within 0.6 Å of a ring centroid becomes
    '6-ring (picrate)' (fragment name from the case JSON); anything else 'A···B'. Returns copies of the spike
    dicts with 'label', 'atoms', 'point', 'kind' added. unique=True keeps, for spikes that resolve to the same
    interaction (neighbouring bins of one broad spike), only the one with the lowest RDG."""
    data = load_rdg(path, xlim=(-1, 1), ylim=(0, 100), with_xyz=True)
    x, y, P = data['x'], data['y'], data['xyz']
    xyz = np.asarray(xyz, float)
    bonds = geo.detect_bonds(symbols, xyz)
    hbs = {(h.h, h.a): h for h in geo.detect_hbonds(symbols, xyz, bonds)}
    rings = geo.find_rings(len(symbols), bonds)
    cents = [xyz[r].mean(0) for r in rings]
    frag_of = {}
    for name, idx in (fragments or {}).items():
        for i in idx:
            frag_of[i] = name.split()[0]
    out = []
    for s in spikes:
        sel = np.abs(x - s['value']) <= half_width
        if not sel.any():
            out.append(dict(s, label=None, atoms=None, point=None, kind=None))
            continue
        ymin = y[sel].min()
        pts = P[sel & (y <= ymin + rdg_slack)]
        k = np.argmin(y[sel & (y <= ymin + rdg_slack)])
        p = pts[k]
        d = np.linalg.norm(xyz - p, axis=1)
        i, j = np.argsort(d)[:2]
        lab, kind, atoms = None, 'contact', (int(i), int(j))
        for r, c in zip(rings, cents):
            if np.linalg.norm(c - p) < 0.6:
                fr = frag_of.get(r[0])
                lab = f'{len(r)}-ring' + (f' ({fr})' if fr else '')
                kind, atoms = 'ring', tuple(int(t) for t in r)
                break
        if lab is None:
            for a_, b_ in ((i, j), (j, i)):
                if (a_, b_) in hbs:
                    hb = hbs[(a_, b_)]
                    lab, kind, atoms = hb.label(labels), 'hbond', (hb.d, hb.h, hb.a)
                    break
        if lab is None:
            lab = f'{labels[i]}···{labels[j]}'
            if any(set((i, j)) == set(b) for b in bonds):
                kind = 'bond'
        out.append(dict(s, label=lab, atoms=atoms, point=p, kind=kind))
    if unique:
        best = {}
        for k, o in enumerate(out):
            if o['label'] is None:
                continue
            if o['label'] not in best or o['rdg'] < out[best[o['label']]]['rdg']:
                best[o['label']] = k
        keep = set(best.values())
        out = [o for k, o in enumerate(out) if o['label'] is None or k in keep]
    return out


def plot_rdg(ax, x, y, xlim=(-0.05, 0.05), ylim=(0.0, 2.0), vrange=NCI_RANGE, cmap=None, s=0.35, spikes=None,
             key=True, key_labels=('H-bond', 'vdW', 'steric'), ylabel='RDG', xlabel=XLABEL, fontsize=None,
             region_bounds=REGION_BOUNDS, spike_color='#222222', spike_style='auto', max_leader_mm=12.0,
             numbering=None, spike_key=True, spike_prefer=(0.3, -1), number_rings=5, obstacle_vlines=()):
    """RDG vs sign(λ2)ρ scatter (rasterized points in a vector figure) with optional spike labels and a compact
    colour key drawn directly above the x range it encodes.

    spikes: list of dicts with 'value', 'rdg' and 'label' (e.g. from identify_spikes); entries whose label is None
            or whose value is outside xlim are skipped. Hydrogen-bond spikes are labelled first.
    spike_style: 'direct' (text in empty areas, leaders at most max_leader_mm long), 'numbered' (small numbers
            at the spikes plus a key in the emptiest part of the plot), or 'auto' (direct, falling back to
            numbered when any label cannot be placed).
    numbering: dict label -> number, to keep the same numbers across panels; spike_key=False leaves out the
            in-plot key (rdg_figure then draws one shared key under the panels).
    spike_prefer: preferred direction (dx, dy; display space, y up) of the spike labels from their spikes
            (default (0.3, -1): below and slightly right, as before); a spike dict may carry its own 'prefer'.
    A spike dict with 'mark_only': True gets a marker (its 'marker', default an open square 's', and 'ms') but no
            label; it is an obstacle for the labels (e.g. features explained once in a key or caption).
    number_rings: candidate rings (distance steps from the spike) tried for the numbers of spike_style='numbered'
            (default 5, as before); more rings allow longer leaders where many spikes crowd together.
    obstacle_vlines: x values of vertical guide lines that the caller draws in the plot (e.g. the region bounds):
            spike labels and numbers keep clear of them like of the zero line (default (): none, as before).
    Returns dict(scatter, key_ax, placed, dropped)."""
    fontsize = style.SMALL_PT if fontsize is None else fontsize
    cmap = nci_cmap() if cmap is None else cmap
    order = np.argsort(-y)                            # low-RDG points (the spikes) drawn last, on top
    sc = ax.scatter(x[order], y[order], c=x[order], cmap=cmap, vmin=vrange[0], vmax=vrange[1], s=s, lw=0,
                    marker='o', rasterized=True, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.axvline(0, color='0.55', lw=0.4, ls=(0, (3, 2)), zorder=1)
    key_ax = None
    if key:
        f0 = (max(vrange[0], xlim[0]) - xlim[0]) / (xlim[1] - xlim[0])
        f1 = (min(vrange[1], xlim[1]) - xlim[0]) / (xlim[1] - xlim[0])
        fig = ax.figure
        hfrac = 1.8 * style.MM / (ax.get_position().height * fig.get_figheight())
        gfrac = 1.2 * style.MM / (ax.get_position().height * fig.get_figheight())
        key_ax = ax.inset_axes([f0, 1 + gfrac, f1 - f0, hfrac])
        grad = np.linspace(max(vrange[0], xlim[0]), min(vrange[1], xlim[1]), 256)[None]
        key_ax.imshow(grad, aspect='auto', cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                      extent=[grad[0, 0], grad[0, -1], 0, 1])
        key_ax.set_xlim(grad[0, 0], grad[0, -1])
        key_ax.set_xticks([])
        key_ax.set_yticks([])
        for sp in key_ax.spines.values():
            sp.set_linewidth(0.4)
        b = [grad[0, 0], region_bounds[0], region_bounds[1], grad[0, -1]]
        for k, t in enumerate(key_labels):
            key_ax.text((b[k] + b[k + 1]) / 2, 1.0, t, transform=key_ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=fontsize)
        for v in region_bounds:
            key_ax.axvline(v, color='white', lw=0.5)
    placed, dropped, mode = [], [], None
    todo = [sp for sp in (spikes or []) if sp.get('label') and not sp.get('mark_only')
            and xlim[0] <= sp['value'] <= xlim[1] and ylim[0] <= sp['rdg'] <= ylim[1]]
    marks = [sp for sp in (spikes or []) if sp.get('mark_only') and xlim[0] <= sp['value'] <= xlim[1]
             and ylim[0] <= sp['rdg'] <= ylim[1]]
    for sp in marks:                                  # marker only, no label (an obstacle for the labels)
        sp['_marker'] = ax.plot(sp['value'], sp['rdg'], sp.get('marker', 's'), ms=sp.get('ms', 2.8), mfc='none',
                                mec=sp.get('mec', '#111111'), mew=0.6, zorder=4)[0]
    if todo:
        if key_ax is not None:
            freeze_layout(ax.figure, ax)
        for sp in todo:
            sp['_marker'] = ax.plot(sp['value'], sp['rdg'], 'o', ms=3.0, mfc='none', mec='#111111', mew=0.6,
                                    zorder=4)[0]
        box_kw = dict(bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.9))
        ring_h_mm = fontsize * 1.15 * 25.4 / 72
        max_rings = max(2, int(max_leader_mm / ring_h_mm))

        def new_placer(name, n_rings):
            pl = LabelPlacer(ax, name=name, n_rings=n_rings, ring_step=1.0, leader_from=0, n_dirs=36, gap_pt=4.0)
            pl.add_density(x, y, bins=(200, 150), min_count=1, hard=0.04)
            pl.add_segment((0, ylim[0]), (0, ylim[1]), hw_pt=0.6)
            for v in obstacle_vlines:
                pl.add_segment((v, ylim[0]), (v, ylim[1]), hw_pt=0.6)
            for sp in todo + marks:
                pl.add_disc((sp['value'], sp['rdg']), r_pt=1.6)
            return pl

        if spike_style in ('auto', 'direct'):
            placer = new_placer('rdg spike labels', max_rings)
            for n, sp in enumerate(todo):
                placer.add_item(sp['label'], (sp['value'], sp['rdg']), radius_pt=1.6, prefer=sp.get('prefer', spike_prefer),
                                priority=0 if sp.get('kind') == 'hbond' else 1,
                                kw=dict(fontsize=fontsize, color=spike_color, leader_color='0.25', **box_kw),
                                key=('spike', n), alts=(sp['label'].replace(' (', '\n(', 1),) if ' (' in sp['label'] else ())
            placed, dropped = placer.place()
            mode = 'direct'
            if dropped and spike_style == 'auto':
                for it in placed:                     # start again with numbered markers and a key
                    it.artist.remove()
                    if it.leader_artist is not None:
                        it.leader_artist.remove()
                placer._placed = []
                placed, dropped = [], []
                mode = 'numbered'
        else:
            mode = 'numbered'
        if mode == 'numbered':
            order = sorted(range(len(todo)), key=lambda k: todo[k]['value'])
            if numbering:
                num = {k: numbering[todo[k]['label']] for k in range(len(todo))}
                order = sorted(order, key=lambda k: num[k])
            else:
                num = {k: i + 1 for i, k in enumerate(order)}
            placer = new_placer('rdg spike numbers', number_rings)
            for k, sp in enumerate(todo):
                placer.add_item(str(num[k]), (sp['value'], sp['rdg']), radius_pt=1.6, prefer=sp.get('prefer', spike_prefer),
                                kw=dict(fontsize=fontsize, color=spike_color, leader_color='0.25',
                                        fontweight='bold', **box_kw), key=('spike', k))
            placed, dropped = placer.place()
            lines = [f'{num[k]}  {todo[k]["label"]}' for k in order if not any(it.key[1] == k for it in dropped)]
            if spike_key:
                _place_key(ax, '\n'.join(lines), x, y, placed, fontsize, spike_color)
        for it in dropped:                            # never leave an unlabelled marker behind
            todo[it.key[1]]['_marker'].remove()
        if dropped:
            msg = 'LOUD WARNING plot_rdg: spike labels dropped: ' + ', '.join(it.text for it in dropped)
            print(msg)
            warnings.warn(msg, qa.FigureQAWarning)
    return dict(scatter=sc, key_ax=key_ax, placed=placed, dropped=dropped, spike_mode=mode)


def _place_key(ax, text, x, y, number_items, fontsize, colour):
    """Put a multi-line key where the scatter is emptiest (scan of the axes), clear of the number labels."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    t = ax.text(0, 0, text, transform=None, ha='left', va='bottom', fontsize=fontsize, color=colour,
                linespacing=1.25, zorder=60, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='0.6', lw=0.4,
                                                       alpha=0.95))
    bb = t.get_window_extent(r)
    pad = 0.3 * fontsize * fig.dpi / 72 + 1
    w, h = bb.width + 2 * pad, bb.height + 2 * pad
    A = ax.bbox
    P = ax.transData.transform(np.c_[x, y])
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=(200, 150), range=[[A.x0, A.x1], [A.y0, A.y1]])
    occ = H > 0
    boxes = []
    for it in number_items:
        b = it.artist.get_window_extent(r)
        boxes.append((b.x0, b.y0, b.x1, b.y1))
        if it.leader_artist is not None:
            xd, yd = it.leader_artist.get_data()
            q = ax.transData.transform(np.c_[xd, yd])
            boxes.append((q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
    zero = ax.transData.transform([(0, 0)])[0][0]
    best = None
    m = 3 * fig.dpi / 72
    for fx in np.linspace(0, 1, 25):
        for fy in np.linspace(0, 1, 21):
            x0 = A.x0 + m + fx * (A.width - w - 2 * m)
            y0 = A.y0 + m + fy * (A.height - h - 2 * m)
            x1, y1 = x0 + w, y0 + h
            i0, i1 = np.searchsorted(xe, [x0, x1])
            j0, j1 = np.searchsorted(ye, [y0, y1])
            frac = occ[max(i0 - 1, 0):i1, max(j0 - 1, 0):j1].mean()
            hit = any(not (x1 < b[0] or b[2] < x0 or y1 < b[1] or b[3] < y0) for b in boxes)
            crosses_zero = x0 < zero < x1
            corner = min(np.hypot(fx - cx, fy - cy) for cx in (0, 1) for cy in (0, 1))
            score = (hit, round(frac, 3) + 0.05 * crosses_zero, corner)
            if best is None or score < best[0]:
                best = (score, x0 + pad, y0 + pad)
    (hit, frac, _), X0, Y0 = best
    t.set_transform(ax.transAxes)
    t.set_position(ax.transAxes.inverted().transform((X0, Y0)))
    if hit or frac > 0.02:
        warnings.warn(f'plot_rdg: the spike key covers data (occupied fraction {frac:.2f})', qa.FigureQAWarning)
    return t


def rdg_figure(panels, width=None, height=None, xlim=(-0.05, 0.05), ylim=(0.0, 2.0), panel_letters=True, **kw):
    """Side-by-side RDG panels sharing the y axis.

    panels: list of dicts with keys x, y, and optionally spikes and title. Returns (fig, axes, results).
    With spike_style='auto' (default) every panel uses the same style. When numbered labels are needed in a
    multi-panel figure, one numbering is used for all panels and a single key is written under them."""
    first = dict(kw)
    if len(panels) > 1:
        first['spike_key'] = False                # a multi-panel numbered figure gets one shared key instead
    fig, axes, res = _rdg_figure(panels, width, height, xlim, ylim, panel_letters, **first)
    modes = {r.get('spike_mode') for r in res} - {None}
    if 'numbered' in modes and (len(panels) > 1 or len(modes) > 1):
        plt.close(fig)
        numbering, entries = {}, []
        for p in panels:
            sp = [q for q in (p.get('spikes') or []) if q.get('label') and xlim[0] <= q['value'] <= xlim[1]
                  and ylim[0] <= q['rdg'] <= ylim[1]]
            for q in sorted(sp, key=lambda q: q['value']):
                if q['label'] not in numbering:
                    numbering[q['label']] = len(numbering) + 1
                    entries.append(f'{numbering[q["label"]]}  {q["label"]}')
        kw = dict(kw, spike_style='numbered', numbering=numbering, spike_key=len(panels) == 1)
        fig, axes, res = _rdg_figure(panels, width, height, xlim, ylim, panel_letters,
                                     shared_key=entries if len(panels) > 1 else None, **kw)
    return fig, axes, res


def _key_rows(entries, fontsize, max_width_in, sep_in):
    """Greedy rows of key entries that fit max_width (inches); returns (rows, widths, row_height)."""
    tmp = plt.figure(figsize=(max_width_in, 1))
    r = tmp.canvas.get_renderer()
    w = []
    for e in entries:
        t = tmp.text(0, 0, e, fontsize=fontsize)
        b = t.get_window_extent(r)
        w.append(b.width / tmp.dpi)
        h = b.height / tmp.dpi
    plt.close(tmp)
    rows, cur, cw = [], [], 0.0
    for e, wi in zip(entries, w):
        need = wi if not cur else cw + sep_in + wi
        if cur and need > max_width_in:
            rows.append(cur)
            cur, cw = [e], wi
        else:
            cur.append(e)
            cw = need
    if cur:
        rows.append(cur)
    return rows, dict(zip(entries, w)), h


def _rdg_figure(panels, width, height, xlim, ylim, panel_letters, shared_key=None, **kw):
    style.apply()
    n = len(panels)
    width = (style.SINGLE if n == 1 else style.DOUBLE) if width is None else width
    MM = style.MM
    fs = kw.get('fontsize') or style.SMALL_PT
    left, right, bottom, gap = 11 * MM, 2.5 * MM, 9.5 * MM, 4 * MM
    top = 10.5 * MM                                   # colour key + its words + panel title
    rows, widths, row_h = [], {}, 0
    if shared_key:
        rows, widths, row_h = _key_rows(shared_key, fs, width - left - right, 4 * MM)
        bottom += len(rows) * row_h * 1.35 + 2.0 * MM
    pw = (width - left - right - gap * (n - 1)) / n
    ph = pw * 0.78 if height is None else height - top - bottom
    H = ph + top + bottom
    fig = plt.figure(figsize=(width, H))
    axes, res = [], []
    for k, p in enumerate(panels):
        ax = fig.add_axes([(left + k * (pw + gap)) / width, bottom / H, pw / width, ph / H])
        axes.append(ax)
    for k, (ax, p) in enumerate(zip(axes, panels)):
        r = plot_rdg(ax, p['x'], p['y'], xlim=xlim, ylim=ylim, spikes=p.get('spikes'),
                     ylabel='RDG' if k == 0 else None, **kw)
        if k:
            ax.tick_params(labelleft=False)
        title = p.get('title')
        if title or (panel_letters and n > 1):
            letter = f'({chr(97 + k)}) ' if (panel_letters and n > 1) else ''
            ax.text(0.0, 1 + 7.0 * MM / ph, f'{letter}{title or ""}', transform=ax.transAxes, ha='left',
                    va='bottom', fontsize=style.BASE_PT, fontweight='bold')
        res.append(r)
    if rows:
        y = 0.6 * MM + (len(rows) - 1) * row_h * 1.35
        for row in rows:
            tot = sum(widths[e] for e in row) + 4 * MM * (len(row) - 1)
            x = left + (width - left - right - tot) / 2
            for e in row:
                fig.text(x / width, y / H, e, ha='left', va='bottom', fontsize=fs,
                         color=kw.get('spike_color', '#222222'))
                x += widths[e] + 4 * MM
            y -= row_h * 1.35
    return fig, axes, res
