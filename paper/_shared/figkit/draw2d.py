"""Ball-and-stick and overlay drawings of molecular clusters in matplotlib (vector output).

Public functions
    draw_molecule(ax, symbols, xyz, labels=None, ...)        -> MoleculeDrawing
    molecule_figure(symbols, xyz, labels=None, width=style.DOUBLE, orient='best', ...) -> (fig, ax, MoleculeDrawing)
    draw_overlay(ax, symbols, geoms, colours, ...)           -> MoleculeDrawing
    overlay_figure(symbols, ref_xyz, others, labels=None, width=style.SINGLE, ...) -> (fig, ax, MoleculeDrawing)
    orient_geometry(symbols, xyz, orient)                    -> (xyz_view, R, centre)
    crop_to_content(fig, ax, pad_mm=1.0)                     shrink a drawing figure to its content, scale kept
    add_top_legend(fig, ax, handles, max_width)              legend in a strip above the drawing
    shade_colour(colour, f) / lighten(colour, f) / darken(colour, f)

The *_figure helpers build the figure at print width, orient the cluster with geometry.best_view (least
overlap of atoms and bonds in projection), draw, place labels, crop to the content and, if labels pushed the
content past the requested width, redraw at a slightly smaller scale until it fits.

Coordinates passed to draw_* are the viewing frame: x to the right, y up, z towards the viewer (use
geometry.principal_orientation or geometry.plane_orientation first; the *_figure helpers do that for you).
Atoms and bonds are depth-sorted (painter's algorithm); labels are placed by figkit.labels.LabelPlacer so they
never overlap each other, atoms or bonds. When a label cannot be placed, hydrogen labels are dropped first and
every dropped heavy-atom label is reported loudly.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from . import style, geometry as geo, qa
from .labels import LabelPlacer, freeze_layout

HBOND_COLOUR = '#0E7C7B'          # teal: distinct from N blue, O red, Cl green and the two model colours
LIGHT = np.array([-0.55, 0.83])   # light direction for the highlights (upper left)


def lighten(c, f):
    c = np.array(to_rgb(c))
    return tuple(c + (1 - c) * f)


def darken(c, f):
    return tuple(np.array(to_rgb(c)) * (1 - f))


def shade_colour(c, t):
    """t in [-1, 1]: -1 dark edge, 0 base colour, +1 highlight."""
    return darken(c, -t * 0.45) if t < 0 else lighten(c, t * 0.75)


def element_colour(s):
    return style.ELEMENT_COLOURS.get(s, '#FF66CC')


@dataclass
class MoleculeDrawing:
    ax: object
    scale_pt_per_A: float = 0.0
    placed: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    hbonds: list = field(default_factory=list)
    placer: object = None

    def dropped_labels(self):
        return [it.text for it in self.dropped]


# ---------------------------------------------------------------------------------------------------------------
# limits and scale
# ---------------------------------------------------------------------------------------------------------------


def _fit_limits(ax, xy, radii, margin):
    """Equal-aspect limits covering the discs plus `margin` (data units), expanded to the axes' box shape."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_position()
    W = bb.width * fig.get_figwidth()
    H = bb.height * fig.get_figheight()
    lo = (xy - radii[:, None]).min(0) - margin
    hi = (xy + radii[:, None]).max(0) + margin
    ext = hi - lo
    s = min(W / ext[0], H / ext[1])
    cx, cy = (lo + hi) / 2
    hw, hh = W / s / 2, H / s / 2
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - hh, cy + hh)
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)
    return s * 72.0            # points per data unit


# ---------------------------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------------------------


def _sphere(ax, x, y, r, colour, z, shade=True, lw=0.35, alpha=1.0):
    """Shaded-sphere disc made of concentric circles offset towards the light; returns the patches."""
    arts = []
    edge = darken(colour, 0.55)
    base = Circle((x, y), r, facecolor=shade_colour(colour, -0.35) if shade else colour, edgecolor=edge, lw=lw,
                  zorder=z, alpha=alpha)
    arts.append(ax.add_patch(base))
    if shade:
        n = 12
        for k in range(1, n + 1):
            t = k / n
            rr = r * (1 - 0.78 * t)
            off = LIGHT * r * 0.34 * t
            col = shade_colour(colour, -0.35 + 1.15 * t ** 1.3)
            arts.append(ax.add_patch(Circle((x + off[0], y + off[1]), rr, facecolor=col, edgecolor='none',
                                            lw=0, zorder=z + 1e-6 * k, alpha=alpha)))
    return arts


def _bond(ax, a, m, b, hw, ci, cj, z, shade=True, lw=0.35):
    """Bond from a to b split at m into the two element colours; one dark outline for the whole bond."""
    d = b[:2] - a[:2]
    L = np.hypot(*d)
    if L < 1e-6:
        return []
    u = d / L
    n = np.array([-u[1], u[0]])
    if n @ LIGHT < 0:
        n = -n
    q = [a[:2] + n * hw, b[:2] + n * hw, b[:2] - n * hw, a[:2] - n * hw]
    arts = [ax.add_patch(Polygon(q, closed=True, facecolor='none', edgecolor=darken(ci, 0.55), lw=lw * 2,
                                 zorder=z, joinstyle='miter'))]
    for k, (p0, p1, c) in enumerate(((a, m, ci), (m, b, cj))):
        q = [p0[:2] + n * hw, p1[:2] + n * hw, p1[:2] - n * hw, p0[:2] - n * hw]
        arts.append(ax.add_patch(Polygon(q, closed=True, facecolor=shade_colour(c, -0.15) if shade else c,
                                         edgecolor='none', lw=0, zorder=z + (1 + k) * 1e-6)))
        if shade:
            cc = 0.38 * hw
            w2 = 0.32 * hw
            q2 = [p0[:2] + n * (cc + w2), p1[:2] + n * (cc + w2), p1[:2] + n * (cc - w2), p0[:2] + n * (cc - w2)]
            arts.append(ax.add_patch(Polygon(q2, closed=True, facecolor=shade_colour(c, 0.55), edgecolor='none',
                                             lw=0, zorder=z + (3 + k) * 1e-6)))
    return arts


def _atom_radius(s, atom_scale):
    return style.BALL_RADII.get(s, 0.35) * atom_scale


# ---------------------------------------------------------------------------------------------------------------
# ball and stick
# ---------------------------------------------------------------------------------------------------------------


def draw_molecule(ax, symbols, xyz, labels=None, bonds=None, hbonds='auto', *, label_atoms='all', label_h='auto',
                  hbond_distances=True, atom_scale=1.0, bond_radius=0.085, fontsize=None, margin=0.9,
                  shade=True, fit=True, hbond_colour=HBOND_COLOUR, label_colour='black', strict=False,
                  placer_kw=None, extra_items=(), hbond_fmt='{:.2f} Å', hbond_colours=None, hbond_lw=0.9,
                  extra_segments=(), extra_discs=(), label_kw=None, atom_colours=None, text_obstacles=()):
    """Depth-sorted ball-and-stick drawing with collision-free atom labels.

    symbols, xyz     element symbols and coordinates in the viewing frame (Å; z towards the viewer)
    labels           clean labels (geometry.clean_labels); None = no labels
    bonds            list of (i, j); detected with geometry.detect_bonds if None
    hbonds           'auto' (geometry.detect_hbonds), a list of (d, h, a) / HBond, or None
    label_atoms      'all', 'heavy', or an iterable of atom indices / label strings (label only these)
    label_h          True: H labels are mandatory; False: never label H; 'auto': label H, drop them first if crowded
    hbond_distances  write H···A distances (Å, 2 decimals) next to the dashed lines; True (all), False (none) or
                     an iterable of (d, h, a) triples (label only those hydrogen bonds)
    fit              set equal-aspect limits to the molecule plus `margin` Å (room for labels)
    extra_items      further dicts for LabelPlacer.add_item (placed together with the atom labels)
    hbond_fmt        format of the H···A distance labels (default '{:.2f} Å'; e.g. '{:.3f} Å')
    hbond_colours    optional dict (d, h, a) -> colour for individual hydrogen bonds (default hbond_colour)
    hbond_lw         line width of the dashed hydrogen bonds (pt)
    extra_segments   further label obstacles in data coordinates: (p0, p1, hw_pt) or (p0, p1, hw_pt, hw_data)
                     (e.g. arrows drawn on the molecule)
    extra_discs      further label obstacles: (xy, r_data) or (xy, r_data, r_pt)
    label_kw         extra text properties for the atom labels (e.g. dict(color='0.35'))
    atom_colours     optional per-atom colour override: dict index -> colour or a sequence of colours (None =
                     element colour) for the balls and the bond halves (e.g. context molecules lightened)
    text_obstacles   Text artists already on the axes (e.g. distance labels placed by the caller) that no label
                     may overlap (LabelPlacer.add_text_obstacle); default none
    Returns MoleculeDrawing (placed / dropped labels, scale).
    """
    fontsize = style.SMALL_PT if fontsize is None else fontsize
    xyz = np.asarray(xyz, float)
    n = len(symbols)
    if atom_colours is None:
        _acol = [element_colour(s) for s in symbols]
    else:
        _get = atom_colours.get if isinstance(atom_colours, dict) else (lambda k, _d=None: atom_colours[k])
        _acol = [(_get(k, None) or element_colour(s)) for k, s in enumerate(symbols)]
    radii = np.array([_atom_radius(s, atom_scale) for s in symbols])
    bonds = geo.detect_bonds(symbols, xyz) if bonds is None else bonds
    if isinstance(hbonds, str) and hbonds == 'auto':
        hbl = [(h.d, h.h, h.a) for h in geo.detect_hbonds(symbols, xyz, bonds)]
    elif hbonds is None:
        hbl = []
    else:
        hbl = [(h.d, h.h, h.a) if isinstance(h, geo.HBond) else tuple(h) for h in hbonds]
    ax.axis('off')
    scale = _fit_limits(ax, xyz[:, :2], radii, margin) if fit else 0.0
    freeze_layout(ax.figure, ax)
    prims = []                                   # (depth, kind, payload)
    for s_i, (i, j) in enumerate(bonds):
        Pi, Pj = xyz[i], xyz[j]
        v = Pj - Pi
        L = np.linalg.norm(v)
        u = v / L
        a = Pi + u * radii[i] * 0.85
        b = Pj - u * radii[j] * 0.85
        m = (a + b) / 2
        prims.append((m[2], 'bond', (a, m, b, _acol[i], _acol[j])))
    hb_segments = []
    for d, h, a in hbl:
        Ph, Pa = xyz[h], xyz[a]
        v = Pa - Ph
        L = np.linalg.norm(v)
        u = v / L
        p0 = Ph + u * radii[h] * 1.05
        p1 = Pa - u * radii[a] * 1.05
        prims.append(((p0[2] + p1[2]) / 2, 'hbond', (p0, p1, L, (d, h, a))))
        hb_segments.append((p0, p1, L, (d, h, a)))
    for i in range(n):
        prims.append((xyz[i, 2], 'atom', i))
    prims.sort(key=lambda p: (p[0], {'hbond': 0, 'bond': 1, 'atom': 2}[p[1]]))
    for rank, (depth, kind, pl) in enumerate(prims):
        z = 2 + rank * 1e-3
        if kind == 'bond':
            _bond(ax, *pl[:3], bond_radius, pl[3], pl[4], z, shade=shade)
        elif kind == 'hbond':
            p0, p1 = pl[0], pl[1]
            hc = (hbond_colours or {}).get(tuple(pl[3]), hbond_colour)
            ln = Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=hc, lw=hbond_lw, ls=(0, (2.2, 1.4)),
                        zorder=z, solid_capstyle='butt', dash_capstyle='butt')
            ax.add_line(ln)
        else:
            i = pl
            _sphere(ax, xyz[i, 0], xyz[i, 1], radii[i], _acol[i], z, shade=shade)
    drawing = MoleculeDrawing(ax=ax, scale_pt_per_A=scale, hbonds=hbl)
    if labels is None and not hbond_distances and not extra_items:
        return drawing
    pk = dict(name='molecule labels', n_rings=7, n_dirs=32)
    pk.update(placer_kw or {})
    placer = LabelPlacer(ax, **pk)
    for i in range(n):
        placer.add_disc(xyz[i, :2], r_data=radii[i], owner=i)
    for (i, j) in bonds:
        placer.add_segment(xyz[i, :2], xyz[j, :2], hw_data=bond_radius)
    for p0, p1, L, _ in hb_segments:
        placer.add_segment(p0[:2], p1[:2], hw_pt=0.45)
    for sg in extra_segments:
        placer.add_segment(sg[0], sg[1], hw_pt=sg[2] if len(sg) > 2 else 0.0,
                           hw_data=sg[3] if len(sg) > 3 else 0.0)
    for dc in extra_discs:
        placer.add_disc(dc[0], r_data=dc[1], r_pt=dc[2] if len(dc) > 2 else 0.0)
    for t in text_obstacles:
        placer.add_text_obstacle(t)
    if hbond_distances:
        _want = None if isinstance(hbond_distances, (bool, int, np.integer)) else {tuple(t) for t in hbond_distances}
        for p0, p1, L, (d, h, a) in hb_segments:
            if _want is not None and (d, h, a) not in _want:
                continue
            mid = (p0[:2] + p1[:2]) / 2
            dv = p1[:2] - p0[:2]
            perp = np.array([-dv[1], dv[0]])
            hc = (hbond_colours or {}).get((d, h, a), hbond_colour)
            txt = hbond_fmt.format(L)
            placer.add_item(txt, mid, radius_pt=0.6, prefer=perp if perp[1] >= 0 else -perp, priority=0,
                            kw=dict(fontsize=fontsize, color=hc, fontstyle='normal'),
                            key=('hbond', d, h, a), max_ring=6, alts=(txt.replace(' Å', ''),))
    if labels is not None:
        sel = _select_labels(symbols, labels, label_atoms, label_h)
        nbr = {i: [] for i in range(n)}
        for i, j in list(bonds) + [(h, a) for d, h, a in hbl]:
            nbr[i].append(j)
            nbr[j].append(i)
        for i in sel:
            vecs = [xyz[i, :2] - xyz[j, :2] for j in nbr[i]]
            vecs = [v / np.linalg.norm(v) for v in vecs if np.linalg.norm(v) > 1e-6]
            prefer = np.sum(vecs, 0) if vecs else None
            if prefer is not None and np.linalg.norm(prefer) < 0.3:
                prefer = None
            isH = symbols[i] == 'H'
            placer.add_item(labels[i], xyz[i, :2], radius_data=radii[i], prefer=prefer,
                            priority=2 if isH else 1, droppable=(isH and label_h == 'auto'),
                            kw=dict(dict(fontsize=fontsize, color=label_colour), **(label_kw or {})),
                            key=('atom', i), max_ring=3)
    for ex in extra_items:
        placer.add_item(**ex)
    placed, dropped = placer.place()
    drawing.placed, drawing.dropped, drawing.placer = placed, dropped, placer
    heavy_lost = [it.text for it in dropped if not (it.key and it.key[0] == 'atom' and symbols[it.key[1]] == 'H')]
    if dropped:
        msg = f'draw_molecule: {len(dropped)} label(s) dropped for lack of space: ' + ', '.join(it.text for it in dropped)
        if heavy_lost:
            msg = 'LOUD WARNING ' + msg + ' (includes non-hydrogen labels; enlarge the figure or pass label_atoms)'
            if strict:
                raise qa.FigureQAError(msg)
        print(msg, file=sys.stderr)
        warnings.warn(msg, qa.FigureQAWarning, stacklevel=2)
    return drawing


def _select_labels(symbols, labels, label_atoms, label_h):
    n = len(symbols)
    if label_atoms in ('all', None):
        sel = list(range(n))
    elif label_atoms == 'heavy':
        sel = [i for i in range(n) if symbols[i] != 'H']
    else:
        want = set(label_atoms)
        sel = [i for i in range(n) if i in want or labels[i] in want]
    if label_h is False:
        sel = [i for i in sel if symbols[i] != 'H']
    return sel


def _figure_for(xy_extent, width, max_height, margin_mm):
    ex, ey = xy_extent
    avail = width - 2 * margin_mm * style.MM
    s = avail / ex                                  # inches per Å
    h = ey * s + 2 * margin_mm * style.MM
    if h > max_height:
        h = max_height
        s = (h - 2 * margin_mm * style.MM) / ey
    fig = plt.figure(figsize=(width, h))
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


def orient_geometry(symbols, xyz, orient='principal', plane_atoms=None, weights='heavy'):
    """Rotate a geometry into a viewing frame. orient: 'best' (least atom overlap, geometry.best_view),
    'principal', 'plane' (needs plane_atoms), 'none' or a (R, centre) tuple. Returns (xyz_view, R, centre)."""
    xyz = np.asarray(xyz, float)
    if isinstance(orient, tuple):
        R, c = orient
        return geo.apply_orientation(xyz, R, c), R, c
    if orient == 'none':
        return xyz.copy(), np.eye(3), np.zeros(3)
    w = np.array([0.15 if s == 'H' else 1.0 for s in symbols]) if weights == 'heavy' else None
    if orient == 'best':
        return geo.best_view(symbols, xyz)
    if orient == 'plane':
        return geo.plane_orientation(xyz, plane_atoms)
    return geo.principal_orientation(xyz, w)


def molecule_figure(symbols, xyz, labels=None, width=None, orient='best', plane_atoms=None,
                    max_height=None, margin_mm=4.0, rotate_deg=0.0, crop=True, **kw):
    """Complete figure for one structure at print size. kw go to draw_molecule. Returns (fig, ax, drawing)."""
    style.apply()
    width = style.DOUBLE if width is None else width
    max_height = style.MAX_HEIGHT if max_height is None else max_height
    if crop:
        # labels may reach past the margin; shrink the drawing scale until the cropped figure fits the width
        target, W = width, width
        for _ in range(4):
            fig, ax, d = molecule_figure(symbols, xyz, labels, W, orient, plane_atoms, max_height, margin_mm,
                                         rotate_deg, crop=False, **dict(kw))
            crop_to_content(fig, ax)
            fw, fh = fig.get_size_inches()
            if fw <= target + 1e-3 and fh <= max_height + 1e-3:
                return fig, ax, d
            plt.close(fig)
            W = W * min(target / fw, max_height / fh) - 0.02
        return fig, ax, d
    xv, R, c = orient_geometry(symbols, xyz, orient, plane_atoms)
    if rotate_deg:
        a = np.radians(rotate_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        xv = xv @ Rz.T
    radii = np.array([_atom_radius(s, kw.get('atom_scale', 1.0)) for s in symbols])
    ext = (xv[:, :2] + radii[:, None]).max(0) - (xv[:, :2] - radii[:, None]).min(0)
    fig, ax = _figure_for(ext, width, max_height, margin_mm)
    # margin in Å equivalent to margin_mm at the final scale
    s_in_per_A = (width - 2 * margin_mm * style.MM) / ext[0]
    s_in_per_A = min(s_in_per_A, (fig.get_figheight() - 2 * margin_mm * style.MM) / ext[1])
    kw.setdefault('margin', margin_mm * style.MM / s_in_per_A)
    d = draw_molecule(ax, symbols, xv, labels, **kw)
    d.view = xv
    return fig, ax, d


def crop_to_content(fig, ax, pad_mm=1.0):
    """Shrink the figure around what is drawn in `ax` (atoms, bonds, labels, legend), keeping the print scale
    (inches per Å) unchanged. The axes then fills the figure. Returns the new figure size (inches)."""
    from matplotlib.transforms import Bbox
    from matplotlib.text import Text
    from matplotlib.patches import Patch
    from matplotlib.collections import Collection
    from matplotlib.legend import Legend
    from matplotlib.spines import Spine
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = []
    for a in ax.get_children():
        if not a.get_visible() or a is ax.patch or isinstance(a, Spine):
            continue
        if isinstance(a, Text) and not a.get_text().strip():
            continue
        if isinstance(a, (Patch, Line2D, Text, Legend, Collection)):
            try:
                bb = a.get_window_extent(r)
            except Exception:
                continue
            if np.all(np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1])) and (bb.width > 0 or bb.height > 0):
                bbs.append(bb)
    if not bbs:
        return fig.get_size_inches()
    u = Bbox.union(bbs)
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[u.x0, u.y0], [u.x1, u.y1]])
    xl = ax.get_xlim()
    ipd = (ax.bbox.width / fig.dpi) / (xl[1] - xl[0])            # inches per data unit
    pad = pad_mm * style.MM / ipd
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    fig.set_size_inches((x1 - x0) * ipd, (y1 - y0) * ipd)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.canvas.draw()
    return fig.get_size_inches()


def add_top_legend(fig, ax, handles, max_width=None, fontsize=None, gap_mm=1.0):
    """Add a legend in a strip above the axes: the figure grows upward (and sideways up to max_width if the
    legend needs it) while the axes keeps its physical size, so labels placed in it stay valid."""
    fontsize = style.TICK_PT if fontsize is None else fontsize
    W0, H0 = fig.get_size_inches()
    max_width = max(W0, max_width or W0)
    ncol = len(handles)
    while True:
        leg = fig.legend(handles=handles, loc='lower center', ncol=ncol, fontsize=fontsize, frameon=False,
                         handlelength=1.8, columnspacing=1.2, borderaxespad=0.0, borderpad=0.1)
        fig.canvas.draw()
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        w, h = bb.width / fig.dpi, bb.height / fig.dpi
        if w <= max_width - 0.04 or ncol == 1:
            break
        leg.remove()
        ncol -= 1
    W = max(W0, w + 0.04)
    H = H0 + h + gap_mm * style.MM
    fig.set_size_inches(W, H)
    ax.set_position([(W - W0) / 2 / W, 0, W0 / W, H0 / H])
    leg.set_bbox_to_anchor((0.5, (H0 + gap_mm * style.MM) / H), transform=fig.transFigure)
    fig.canvas.draw()
    return leg


# ---------------------------------------------------------------------------------------------------------------
# overlay of two or more geometries (same atom order)
# ---------------------------------------------------------------------------------------------------------------


def draw_overlay(ax, symbols, geoms, colours, bonds=None, include_h=False, lw=1.5, halo=True, labels=None,
                 label_atoms=(), fontsize=None, margin=0.9, legend=True, legend_labels=None, legend_loc='best',
                 atom_dots=True, fit=True, hbonds=None, strict=False, hbond_obstacles=False):
    """Stick overlay of several geometries already in one viewing frame.

    geoms          dict name -> xyz (N, 3), same atom order; the first entry is the reference (drawn with its bonds)
    colours        dict name -> colour
    include_h      draw hydrogen atoms (X-ray hydrogens are riding-model positions; default False)
    label_atoms    indices or labels to annotate (placed without overlaps against every stick)
    hbonds         list of (d, h, a) to draw as dashed lines for every geometry (needs include_h=True), or None;
                   or a dict name -> list of (d, h, a) when the geometries differ (e.g. a transferred proton)
    bonds          list of (i, j) used for every geometry (default: detected on the reference), or a dict
                   name -> list of (i, j) (geometries missing from the dict get their own detected bonds)
    hbond_obstacles  True: the dashed hydrogen bonds are also label obstacles (default False, as before)
    legend_labels  dict name -> legend text (e.g. with RMSD)
    lw             stick width (pt), or a dict name -> width (e.g. nested widths, widest at the bottom, so every
                   layer stays visible where the geometries coincide)
    """
    fontsize = style.SMALL_PT if fontsize is None else fontsize
    names = list(geoms)
    lw_of = (lambda n: float(lw[n])) if isinstance(lw, dict) else (lambda n: float(lw))
    ref = np.asarray(geoms[names[0]], float)
    keep = [i for i, s in enumerate(symbols) if include_h or s != 'H']
    kset = set(keep)
    if isinstance(bonds, dict):                  # per-geometry bond lists
        bmap = {n: [(i, j) for i, j in (bonds[n] if n in bonds else
                                          geo.detect_bonds(symbols, np.asarray(geoms[n], float)))
                    if i in kset and j in kset] for n in names}
        bonds = bmap[names[0]]
    else:
        bonds = geo.detect_bonds(symbols, ref) if bonds is None else bonds
        bonds = [(i, j) for i, j in bonds if i in kset and j in kset]
        bmap = {n: bonds for n in names}
    if isinstance(hbonds, dict):
        hmap = {n: [tuple(t) for t in hbonds.get(n, [])] for n in names}
    else:
        hmap = {n: [tuple(t) for t in (hbonds or [])] for n in names}
    ax.axis('off')
    allxy = np.vstack([np.asarray(g, float)[keep, :2] for g in geoms.values()])
    scale = _fit_limits(ax, allxy, np.full(len(allxy), 0.1), margin) if fit else 0.0
    freeze_layout(ax.figure, ax)
    # one layer per geometry (reference at the bottom); inside a layer: white halo under all bonds, then the
    # coloured bonds depth-sorted, then atom dots. The halo separates a layer from the layers underneath.
    for k, name in enumerate(names):
        X = np.asarray(geoms[name], float)
        c = colours[name]
        base = 2 + 10 * k
        bl = sorted(bmap[name], key=lambda b: (X[b[0], 2] + X[b[1], 2]) / 2)
        for rank, (i, j) in enumerate(bl):
            if halo:
                ax.add_line(Line2D([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='white', lw=lw_of(name) + 1.6, zorder=base,
                                   solid_capstyle='round'))
            ax.add_line(Line2D([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color=c, lw=lw_of(name),
                               zorder=base + 1 + rank * 1e-4, solid_capstyle='round'))
        for (d, h, a) in hmap[name]:
            ax.add_line(Line2D([X[h, 0], X[a, 0]], [X[h, 1], X[a, 1]], color=c, lw=0.7, ls=(0, (2, 1.4)),
                               zorder=base + 1))
        if atom_dots:
            for i in keep:
                r = 0.075 if symbols[i] != 'H' else 0.05
                ax.add_patch(Circle((X[i, 0], X[i, 1]), r, facecolor=c, edgecolor='none', zorder=base + 3))
    drawing = MoleculeDrawing(ax=ax, scale_pt_per_A=scale)
    handles = []
    if legend:
        for name in names:
            handles.append(Line2D([], [], color=colours[name], lw=lw_of(name),
                                  label=(legend_labels or {}).get(name, name)))
    leg = None
    if handles:
        leg = ax.legend(handles=handles, loc=legend_loc, fontsize=style.TICK_PT, handlelength=1.8, frameon=False,
                        borderaxespad=0.2)
    if labels is not None and label_atoms:
        placer = LabelPlacer(ax, name='overlay labels')
        for name in names:
            X = np.asarray(geoms[name], float)
            for i, j in bmap[name]:
                placer.add_segment(X[i, :2], X[j, :2], hw_pt=lw_of(name) / 2 + 0.8)
            if hbond_obstacles:
                for (d, h, a) in hmap[name]:
                    placer.add_segment(X[h, :2], X[a, :2], hw_pt=0.8)
            for i in keep:
                placer.add_disc(X[i, :2], r_data=0.09, r_pt=0.3)
        if leg is not None:
            freeze_layout(ax.figure, ax)
            for t in leg.get_texts():
                placer.add_text_obstacle(t)
        want = set(label_atoms)
        nbr = {i: [] for i in range(len(symbols))}
        for i, j in bonds:
            nbr[i].append(j)
            nbr[j].append(i)
        for i in range(len(symbols)):
            if (i in want or labels[i] in want) and i in kset:
                vecs = [ref[i, :2] - ref[j, :2] for j in nbr[i]]
                vecs = [v / np.linalg.norm(v) for v in vecs if np.linalg.norm(v) > 1e-6]
                prefer = np.sum(vecs, 0) if vecs else None
                if prefer is not None and np.linalg.norm(prefer) < 0.3:
                    prefer = None
                placer.add_item(labels[i], ref[i, :2], radius_data=0.12, radius_pt=0.3, priority=1, prefer=prefer,
                                kw=dict(fontsize=fontsize, color='black'), key=('atom', i), max_ring=3)
        placed, dropped = placer.place()
        drawing.placed, drawing.dropped, drawing.placer = placed, dropped, placer
        if dropped:
            msg = 'LOUD WARNING draw_overlay: labels dropped: ' + ', '.join(it.text for it in dropped)
            if strict:
                raise qa.FigureQAError(msg)
            print(msg, file=sys.stderr)
    drawing.legend = leg
    return drawing


def overlay_figure(symbols, ref_xyz, others, labels=None, ref_name='X-ray', colours=None, legend_labels=None,
                   align_on='heavy', include_h=False, width=None, max_height=None, margin_mm=4.0, orient='best',
                   plane_atoms=None, label_atoms=(), legend_loc='top', crop=True, **kw):
    """Kabsch-align each geometry in `others` (dict name -> xyz) onto `ref_xyz`, orient, and draw the overlay.

    align_on: 'heavy' (default), 'all' or a list of atom indices. The legend reports each RMSD over those atoms.
    legend_loc: 'top' (a strip above the drawing, never over the molecule) or any matplotlib loc inside the axes.
    Returns (fig, ax, drawing); drawing.rmsd holds the RMSD values (Å)."""
    style.apply()
    width = style.SINGLE if width is None else width
    max_height = style.MAX_HEIGHT if max_height is None else max_height
    if crop and not kw.pop('_inner', False):
        W = width
        for _ in range(4):
            fig, ax, d = overlay_figure(symbols, ref_xyz, others, labels, ref_name, colours, legend_labels,
                                        align_on, include_h, W, max_height, margin_mm, orient, plane_atoms,
                                        label_atoms, legend_loc, crop=True, _inner=True, **dict(kw))
            fw, fh = fig.get_size_inches()
            if fw <= width + 1e-3 and fh <= max_height + 1e-3:
                return fig, ax, d
            plt.close(fig)
            W = W * min(width / fw, max_height / fh) - 0.02
        return fig, ax, d
    ref = np.asarray(ref_xyz, float)
    idx = geo.heavy_atoms(symbols) if align_on == 'heavy' else (None if align_on == 'all' else list(align_on))
    aligned, rmsd = {}, {}
    for name, X in others.items():
        aligned[name], rmsd[name] = geo.align(np.asarray(X, float), ref, idx=idx)
    rv, R, c = orient_geometry(symbols, ref, orient, plane_atoms)
    geoms = {ref_name: rv}
    for name in aligned:
        geoms[name] = geo.apply_orientation(aligned[name], R, c)
    if colours is None:
        colours = {ref_name: style.MODEL_COLOURS['xray']}
        pal = [style.MODEL_COLOURS['pcm'], style.MODEL_COLOURS['gas'], '#6A3D9A', '#33A02C']
        for k, name in enumerate(aligned):
            colours[name] = pal[k % len(pal)]
    if legend_labels is None:
        legend_labels = {ref_name: ref_name}
        for name in aligned:
            legend_labels[name] = f'{name}, RMSD {rmsd[name]:.3f} Å'
    keep = [i for i, s in enumerate(symbols) if include_h or s != 'H']
    allxy = np.vstack([g[keep, :2] for g in geoms.values()])
    ext = allxy.max(0) - allxy.min(0) + 0.2
    fig, ax = _figure_for(ext, width, max_height, margin_mm)
    s_in_per_A = min((width - 2 * margin_mm * style.MM) / ext[0], (fig.get_figheight() - 2 * margin_mm * style.MM) / ext[1])
    kw.setdefault('margin', margin_mm * style.MM / s_in_per_A)
    top = legend_loc == 'top'
    d = draw_overlay(ax, symbols, geoms, colours, include_h=include_h, labels=labels, label_atoms=label_atoms,
                     legend_labels=legend_labels, legend_loc=legend_loc, legend=not top, **kw)
    d.rmsd = rmsd
    d.view = geoms
    if crop:
        crop_to_content(fig, ax)
    if top:
        lw = kw.get('lw', 1.5)
        handles = [Line2D([], [], color=colours[n], lw=lw, label=legend_labels.get(n, n)) for n in geoms]
        d.legend = add_top_legend(fig, ax, handles, max_width=width)
    return fig, ax, d


# ---------------------------------------------------------------------------------------------------------------
# normal-mode displacement arrows (added for the BTP vibration figures; see CHANGES.md)
# ---------------------------------------------------------------------------------------------------------------


def displacement_arrows(ax, xy, vec, atoms=None, scale=1.0, min_len=0.0, colour='black', lw=0.7, head_pt=3.0,
                        halo=True, halo_colour='white', halo_lw=1.1, zorder=40, draw=True):
    """Arrows from atom positions along (projected) displacement vectors, e.g. a normal mode.

    xy       (N, 2) atom positions in the viewing frame (data units, Å)
    vec      (N, 2) or (N, 3) displacement vectors in the same frame; only x, y are drawn
    atoms    indices to draw (default all); arrows shorter than min_len (data units, after scaling) are skipped
    scale    factor applied to vec (e.g. so that the largest displacement is drawn 1.0 Å long)
    halo     thin outline (halo_colour) around each arrow so it stays visible over atoms of any colour
    Returns a list of label-obstacle segments (p0, p1, hw_pt) for draw_molecule(extra_segments=...), one per
    arrow drawn. With draw=False nothing is drawn (use it to reserve the obstacles before the labels are placed,
    then call again with draw=True)."""
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe
    xy = np.asarray(xy, float)
    vec = np.asarray(vec, float)[:, :2] * scale
    idx = range(len(xy)) if atoms is None else atoms
    segs = []
    for i in idx:
        L = float(np.hypot(*vec[i]))
        if L < max(min_len, 1e-9):
            continue
        p0, p1 = xy[i], xy[i] + vec[i]
        segs.append((p0.copy(), p1.copy(), lw / 2 + (halo_lw if halo else 0.0) + 0.3))
        if draw:
            a = FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle=f'-|>,head_length={head_pt * 0.9 / lw:.3f},'
                                f'head_width={head_pt * 0.45 / lw:.3f}', mutation_scale=lw, lw=lw, color=colour,
                                shrinkA=0, shrinkB=0, zorder=zorder, joinstyle='miter', capstyle='butt')
            if halo:
                a.set_path_effects([pe.withStroke(linewidth=lw + 2 * halo_lw, foreground=halo_colour), pe.Normal()])
            ax.add_patch(a)
    return segs
