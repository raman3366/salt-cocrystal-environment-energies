"""Assemble 3D renders (figkit.render3d.Render) into print-size figures with matplotlib typography.

No VTK scalar bars and no VTK text: panel letters, titles, colour bars and atom labels are matplotlib artists
laid out at the FINAL print width (style.SINGLE / ONE_HALF / DOUBLE), so fonts are exactly style.* sizes.

    fig, axes = compose.grid([r_a, r_b, r_c, r_d], ncols=2, width=style.DOUBLE,
                             titles=['HOMO', 'LUMO', 'HOMO', 'LUMO'], row_titles=['Gas phase', 'IEF-PCM (water)'],
                             colourbar=None)
    compose.label_atoms(axes[0], r_a, ['O7', 'N5', 'N4', 'N6'])
    compose.save(fig, 'figures/fig5_fmo')           # 600 dpi PNG + PDF, QA-checked (figkit.qa)

Layout rules: panels keep a common scale (Angstrom per mm) when their renders share px_per_A, so two models
drawn side by side are directly comparable; the header of each panel carries its bold letter and title, so
letters never sit on top of the picture.
"""
from __future__ import annotations

import math
import string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patheffects, ticker

from . import qa, style
from .labels import LabelPlacer

MM = style.MM
OUTER_MM = 0.6            # outer margin; style.save adds 0.02 in when cropping, so the width stays <= nominal


# ----------------------------------------------------------------------------------------------------------
# colour bars
# ----------------------------------------------------------------------------------------------------------
def _formatter(fmt):
    def f(x, pos=None):
        s = fmt.format(x)
        if s.lstrip('-').strip('0.') == '':          # '-0.00' -> '0.00'
            s = s.lstrip('-')
        return s.replace('-', '−')
    return ticker.FuncFormatter(f)


def colourbar(fig, rect_in, scale, orientation='horizontal', label=None, linear=False):
    """A matplotlib colour bar for a render3d.ColourScale in a rectangle given in INCHES (x0, y0, w, h) from
    the lower-left corner of the figure.  Plain-decimal tick labels with a true minus sign.
    linear=True draws the long axis on a linear value scale even for a non-linear norm (e.g. the two-slope
    MEP norm with green at zero): ticks are then evenly spaced and the colours carry the non-linearity
    (default False = matplotlib's norm-spaced axis, as before)."""
    W, H = fig.get_size_inches()
    cax = fig.add_axes([rect_in[0] / W, rect_in[1] / H, rect_in[2] / W, rect_in[3] / H])
    sm = cm.ScalarMappable(norm=scale.norm, cmap=scale.cmap)
    cb = fig.colorbar(sm, cax=cax, orientation=orientation, extend=scale.extend)
    if linear:
        (cb.ax.set_xscale if orientation == 'horizontal' else cb.ax.set_yscale)('linear')
    if scale.ticks is not None:
        cb.set_ticks(scale.ticks)
    (cb.ax.xaxis if orientation == 'horizontal' else cb.ax.yaxis).set_major_formatter(_formatter(scale.fmt))
    cb.ax.minorticks_off()
    cb.set_label(label or scale.label, fontsize=style.BASE_PT, labelpad=1.5)
    cb.ax.tick_params(labelsize=style.TICK_PT, width=0.5, length=2.2, pad=1.2)
    cb.outline.set_linewidth(0.5)
    return cb


# ----------------------------------------------------------------------------------------------------------
# grid layout
# ----------------------------------------------------------------------------------------------------------
def _header_mm():
    return style.PANEL_PT / 72 * 25.4 * 1.25 + 0.6


def grid(renders, ncols=2, width=style.DOUBLE, titles=None, letters=True, col_titles=None, row_titles=None,
         gap_mm=(2.5, 1.5), same_scale=True, colourbar_scale=None, colourbar_len=0.5, colourbar_label=None,
         max_height=style.MAX_HEIGHT, first_letter='a', extra_bottom_mm=0.0, colourbar_linear=False):
    """Place renders on a grid at print width.  Returns (fig, axes) with axes in the order of `renders`
    (None entries leave an empty cell and give None).

    titles:        per-panel header text after the bold letter, e.g. 'HOMO, gas phase'
    col_titles / row_titles: optional column headings above the grid / rotated row headings on the left
    gap_mm:        (horizontal, vertical) gap between panels
    same_scale:    one Angstrom-per-mm scale for every panel (renders must share px_per_A); otherwise each
                   panel is scaled to fill its column
    colourbar_scale: a render3d.ColourScale for ONE shared horizontal colour bar under the grid
                   (colourbar_len = fraction of the width); None = no colour bar
    colourbar_linear: passed to colourbar(linear=...) (default False); the Colorbar is stored in
                   fig._figkit_grid['colourbar']
    extra_bottom_mm: empty band of this height under everything else (default 0 = no band), for ordinary
                   matplotlib panels (plots next to renders); its rectangle in inches (x0, y0, w, h, from the
                   lower-left corner) is fig._figkit_grid['bottom_band']
    The axes' data coordinates are image pixels (x right, y down), so Render.project() output can be drawn
    directly."""
    style.apply()
    rs = list(renders)
    n = len(rs)
    nrows = math.ceil(n / ncols)
    cells = [[rs[i * ncols + j] if i * ncols + j < n else None for j in range(ncols)] for i in range(nrows)]
    gx, gy = gap_mm[0] * MM, gap_mm[1] * MM
    outer = OUTER_MM * MM
    left = outer + (4.2 * MM if row_titles else 0)
    top_band = (4.0 * MM if col_titles else 0)
    head = _header_mm() * MM if (letters or titles) else 0.6 * MM
    cb_band = (2.2 + 11.0) * MM if colourbar_scale is not None else 0
    band = max(0.0, float(extra_bottom_mm)) * MM
    cb_band += band
    avail_w = width - left - outer - gx * (ncols - 1)

    def size_px(r):
        return (r.image.shape[1], r.image.shape[0]) if r is not None else (0, 0)

    if same_scale:
        pxA = {round(r.px_per_A, 6) for r in rs if r is not None}
        if len(pxA) > 1:
            raise ValueError(f'same_scale=True but renders have different px_per_A {sorted(pxA)}')
        colw_px = [max(size_px(cells[i][j])[0] for i in range(nrows)) for j in range(ncols)]
        rowh_px = [max(size_px(cells[i][j])[1] for j in range(ncols)) for i in range(nrows)]
        k = avail_w / sum(colw_px)                               # inch per pixel, common to all panels
        fixed_h = outer * 2 + top_band + nrows * head + gy * (nrows - 1) + cb_band
        if fixed_h + k * sum(rowh_px) > max_height:
            k = (max_height - fixed_h) / sum(rowh_px)
        colw = [c * k for c in colw_px]
        rowh = [h * k for h in rowh_px]
        scale_of = lambda r: k                                   # noqa: E731
    else:
        cw = avail_w / ncols
        colw = [cw] * ncols
        scale_of = lambda r: cw / r.image.shape[1]               # noqa: E731
        rowh = [max((size_px(cells[i][j])[1] * scale_of(cells[i][j]) if cells[i][j] is not None else 0)
                    for j in range(ncols)) for i in range(nrows)]
    content_w = sum(colw) + gx * (ncols - 1)
    left += max(0.0, (width - outer - left - content_w) / 2)     # centre if narrower than the width
    H = outer * 2 + top_band + nrows * head + sum(rowh) + gy * (nrows - 1) + cb_band
    fig = plt.figure(figsize=(width, H))
    axes = []
    letters_seq = iter(string.ascii_lowercase[string.ascii_lowercase.index(first_letter):])
    y = H - outer - top_band
    xs = [left + sum(colw[:j]) + gx * j for j in range(ncols)]
    if col_titles:
        for j, t in enumerate(col_titles):
            fig.text((xs[j] + colw[j] / 2) / width, (H - outer - 0.2 * MM) / H, t, ha='center', va='top',
                     fontsize=style.BASE_PT, fontweight='bold')
    for i in range(nrows):
        y_head_top = y
        y_img_top = y - head
        if row_titles and i < len(row_titles) and row_titles[i]:
            fig.text((outer + 1.6 * MM) / width, (y_img_top - rowh[i] / 2) / H, row_titles[i], rotation=90,
                     ha='center', va='center', fontsize=style.BASE_PT)
        for j in range(ncols):
            r = cells[i][j]
            idx = i * ncols + j
            if idx >= n:
                continue
            if r is None:
                axes.append(None)
                continue
            s = scale_of(r)
            w_in, h_in = r.image.shape[1] * s, r.image.shape[0] * s
            x0 = xs[j] + (colw[j] - w_in) / 2
            y0 = y_img_top - h_in - (rowh[i] - h_in) / 2
            ax = fig.add_axes([x0 / width, y0 / H, w_in / width, h_in / H])
            ax.imshow(r.image, extent=(0, r.image.shape[1], r.image.shape[0], 0), interpolation='antialiased')
            ax.set_xlim(0, r.image.shape[1])
            ax.set_ylim(r.image.shape[0], 0)
            ax.set_axis_off()
            ax._figkit_render = r
            if letters or titles:
                letter = f'({next(letters_seq)})' if letters else None
                title = titles[idx] if titles and idx < len(titles) else None
                _header(fig, xs[j] / width, (y_head_top - 0.1 * MM) / H, letter, title)
            axes.append(ax)
        y = y_img_top - rowh[i] - gy
    if colourbar_scale is not None:
        L = colourbar_len * content_w
        x0 = left + (content_w - L) / 2
        bar_h = 2.2 * MM
        y0 = outer + band + 8.6 * MM
        cbar = colourbar(fig, (x0, y0, L, bar_h), colourbar_scale, 'horizontal', colourbar_label,
                         linear=colourbar_linear)
    fig._figkit_grid = dict(width=width, height=H)
    if colourbar_scale is not None:
        fig._figkit_grid['colourbar'] = cbar
    if band > 0:
        fig._figkit_grid['bottom_band'] = (outer, outer, width - 2 * outer, band)
    return fig, axes


def _header(fig, x, y, letter=None, title=None):
    """Bold panel letter (style.PANEL_PT) followed by a plain title (style.BASE_PT) on one baseline, top-left
    anchored at figure fraction (x, y)."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    kids = []
    if letter:
        kids.append(TextArea(letter, textprops=dict(fontsize=style.PANEL_PT, fontweight='bold')))
    if title:
        kids.append(TextArea(title, textprops=dict(fontsize=style.BASE_PT)))
    if not kids:
        return None
    box = AnchoredOffsetbox(loc='upper left', child=HPacker(children=kids, align='baseline', pad=0, sep=3.0),
                            bbox_to_anchor=(x, y), bbox_transform=fig.transFigure, frameon=False, pad=0,
                            borderpad=0)
    fig.add_artist(box)
    return box


# ----------------------------------------------------------------------------------------------------------
# labels and markers drawn in the matplotlib layer
# ----------------------------------------------------------------------------------------------------------
_HALO = [patheffects.withStroke(linewidth=1.6, foreground='white')]


def _render_of(ax, render):
    return render if render is not None else getattr(ax, '_figkit_render')


def _open_direction(p, others, reach):
    """Screen direction (pixels, y down) pointing away from nearby obstacles around p (1/d^2 repulsion)."""
    d = p[None] - others
    r = np.hypot(d[:, 0], d[:, 1])
    m = (r > 1e-6) & (r < reach)
    if not m.any():
        return np.array([1.0, -1.0])
    v = (d[m] / r[m, None] ** 3).sum(0)
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, -1.0])


def label_atoms(ax, atoms, render=None, mol=None, fontsize=style.SMALL_PT, colour='black', halo=True,
                only_visible=True, hydrogens_droppable=True, surface='auto', leaders=True, prefer=None,
                name='3d atom labels', hbond_obstacles=False, extra_segments=None, placer_kw=None):
    """Crisp atom labels for a render, placed by figkit.labels.LabelPlacer around the atoms' projected screen
    positions (no overlaps with other labels, visible atom discs or bonds; registered with figkit.qa).

    atoms: labels ('O7', 'N5') or indices.  Atoms hidden behind an opaque surface are skipped when
    only_visible (z-buffer test; translucent orbital lobes do not hide atoms).
    surface: how isosurface patches count as obstacles: 'hard' (labels never cover them; default for NCI),
    'soft' (avoided when possible; default for orbitals) or None.
    Each label prefers the side of its atom that is most open (repulsion from every nearby atom and surface
    point, not only bonded ones), so a label never sits across a hydrogen bond from its atom.
    prefer: {label: (dx, dy)} overrides that preference for single atoms (screen direction, y up).
    hbond_obstacles: also keep labels (and leader lines) off the H...A segments of mol.hbonds (the dashed
    hydrogen bonds drawn by molecule_meshes); default False = previous behaviour.
    extra_segments: further obstacles as pairs of 3D points [(p0, p1), ...] (Angstrom), e.g. arrows.
    placer_kw: extra keyword arguments for the figkit.labels.LabelPlacer (e.g. n_rings=6 to let a label in a
    crowded spot search further out, with a leader); default None = previous behaviour.
    Returns (placed, dropped)."""
    r = _render_of(ax, render)
    mol = mol or r.mol
    idx = [a if isinstance(a, (int, np.integer)) else mol.index(a) for a in atoms]
    rad = np.array([style.BALL_RADII.get(s, 0.35) for s in mol.symbols])
    xy, _ = r.project(mol.coords)
    vis = r.visible(mol.coords, towards_camera=rad) if only_visible else np.ones(len(mol.symbols), bool)
    # leader_from=1: any label that cannot sit right next to its atom gets a leader line, so it can never be
    # read as belonging to a neighbouring atom (checked again by _ambiguity at save time)
    placer = LabelPlacer(ax, **{'name': name, 'clearance_pt': 1.5, 'leader_from': 1 if leaders else 2,
                                **(placer_kw or {})})
    for k in np.flatnonzero(vis):
        placer.add_disc(xy[k], r_data=rad[k] * r.px_per_A, owner=int(k))
    for i, j in mol.bonds:
        if vis[i] and vis[j]:
            placer.add_segment(xy[i], xy[j], hw_data=0.085 * r.px_per_A)
    if hbond_obstacles:
        for _, h, a in (mol.hbonds or []):
            if vis[h] and vis[a]:
                placer.add_segment(xy[h], xy[a], hw_data=0.06 * r.px_per_A)
    for p0, p1 in (extra_segments or []):
        q, _ = r.project(np.array([p0, p1], float))
        placer.add_segment(q[0], q[1], hw_data=0.06 * r.px_per_A)
    obstacles = [xy[vis]]
    sp = r.meta.get('surface_points')
    if surface == 'auto':
        surface = 'hard' if r.meta.get('kind') == 'nci' else 'soft'
    if surface and sp is not None and len(sp):
        spx, _ = r.project(sp)
        if surface == 'hard':
            # one obstacle disc per occupied 0.35 A screen cell: dense per-vertex discs would dominate the
            # placer's clearance cost and push every label one ring out, away from its own atom
            cell = 0.35 * r.px_per_A
            keys = np.unique(np.floor(spx / cell).astype(int), axis=0)
            keep = (keys + 0.5) * cell
            for q in keep:
                placer.add_disc(q, r_data=0.5 * cell * 1.1)
            obstacles.append(keep)
        else:
            placer.add_density(spx[:, 0], spx[:, 1])
    obstacles = np.vstack(obstacles)
    for k in idx:
        if only_visible and not vis[k]:
            continue
        away = _open_direction(xy[k], obstacles, 2.2 * r.px_per_A)
        if prefer and mol.labels[k] in prefer:
            dx, dy = prefer[mol.labels[k]]
            away = np.array([dx, -dy], float)
        kw = dict(fontsize=fontsize, color=colour)
        if halo:
            kw['path_effects'] = _HALO
        placer.add_item(mol.labels[k], xy[k], radius_data=rad[k] * r.px_per_A, prefer=(away[0], -away[1]),
                        droppable=hydrogens_droppable and mol.symbols[k] == 'H', leader=leaders, kw=kw,
                        key=int(k))
    placed, dropped = placer.place()
    discs = {int(k): (xy[k], rad[k] * r.px_per_A) for k in np.flatnonzero(vis)}
    qa.register_check(ax.figure, lambda fig, rend: _ambiguity(ax, placed, discs, rend), name + ' (ambiguity)')
    return placed, dropped


def _ambiguity(ax, placed, discs, renderer):
    """A label must be nearer to its own atom than to any other atom (edge-to-edge, display space)."""
    issues = []
    T = ax.transData
    for it in placed:
        if it.artist is None or it.leader_artist is not None:        # a leader line names its atom
            continue
        bb = it.artist.get_window_extent(renderer)

        def gap(k):
            (x, y), rr = discs[k]
            X, Y = T.transform((x, y))
            rr = rr * (T.transform((1, 0))[0] - T.transform((0, 0))[0])
            dx = max(bb.x0 - X, 0, X - bb.x1)
            dy = max(bb.y0 - Y, 0, Y - bb.y1)
            return np.hypot(dx, dy) - rr
        own = gap(it.key) if it.key in discs else None
        if own is None:
            continue
        others = [(gap(k), k) for k in discs if k != it.key]
        if others:
            g, k = min(others)
            if g < own - 0.5 * ax.figure.dpi / 72:
                issues.append(f'label "{it.text}" is closer to another atom (index {k}) than to its own')
    return issues


def annotate_markers(ax, render=None, which='global', fontsize=style.SMALL_PT, colour='black', fmt='{:.1f}',
                     only_visible=True, name='3d marker labels'):
    """Value labels (e.g. MEP extrema in kcal/mol) next to the markers stored in the render.
    which: 'global' (global min and max), 'all', or a callable(marker) -> bool."""
    r = _render_of(ax, render)
    ms = list(r.markers)
    if which == 'global':
        lo = min(ms, key=lambda m: m['value']) if ms else None
        hi = max(ms, key=lambda m: m['value']) if ms else None
        ms = [m for m in ms if m is lo or m is hi]
    elif callable(which):
        ms = [m for m in ms if which(m)]
    if not ms:
        return [], []
    P = np.array([m['pos'] for m in ms])
    xy, _ = r.project(P)
    rad = np.array([m['radius'] for m in ms])
    win = max(1, int(0.6 * rad.max() * r.px_per_A))            # a bead on the silhouette still counts as seen
    vis = r.visible(P, towards_camera=rad, window=win) if only_visible else np.ones(len(ms), bool)
    placer = LabelPlacer(ax, name=name, clearance_pt=1.0)
    for k, m in enumerate(ms):
        if not vis[k]:
            continue
        placer.add_disc(xy[k], r_data=m['radius'] * r.px_per_A)
    for k, m in enumerate(ms):
        if not vis[k]:
            continue
        kw = dict(fontsize=fontsize, color=colour, path_effects=_HALO)
        placer.add_item(fmt.format(m['value']).replace('-', '−'), xy[k], radius_data=m['radius'] * r.px_per_A,
                        kw=kw)
    return placer.place()


def save(fig, path, strict=False, formats=('png', 'pdf')):
    """figkit.qa-checked save (600 dpi PNG + vector PDF).  Returns the written paths."""
    w = fig._figkit_grid['width'] / MM if hasattr(fig, '_figkit_grid') else None
    return qa.save(fig, path, strict=strict, max_width_mm=w, formats=formats)
