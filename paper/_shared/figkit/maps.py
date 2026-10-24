"""ELF / LOL colour-filled contour maps from Multiwfn plane exports.

Public functions
    read_plane(path)                                    -> PlaneGrid   (*_plane.txt preferred; *.npz fallback)
    plane_files(case_dir, plane_name)                   -> dict(elf=Path, lol=Path)
    plot_plane_map(ax, plane, symbols, xyz, labels=None, kind='ELF', cmap='turbo', ...) -> PlaneMap
    plane_map_figure(plane, symbols, xyz, labels, kind='ELF', width=style.SINGLE, ...) -> (fig, ax, PlaneMap)
    elf_lol_figure(elf, lol, symbols, xyz, labels, width=style.DOUBLE, ...)             -> (fig, axes, [PlaneMap, PlaneMap])

The Multiwfn *_plane.txt export has six columns: X Y Z (Å) of each grid point, the two in-plane graph coordinates
(Å) and the function value. read_plane recovers the plane frame (origin, in-plane unit vectors, normal) by an exact
linear fit of XYZ against the graph coordinates, so atoms from any geometry file can be projected onto the map.
Atoms within ±atom_tol (default 0.5 Å) of the plane are drawn and labelled (white text, thin dark outline),
bonded in-plane atoms are joined by thin lines; other atoms are hidden or drawn faintly (offplane='faint').
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from . import style, geometry as geo, qa
from .labels import LabelPlacer, freeze_layout


@dataclass
class PlaneGrid:
    u: np.ndarray               # (nx,) graph-x coordinates, Å
    v: np.ndarray               # (ny,) graph-y coordinates, Å
    F: np.ndarray               # (ny, nx) values
    origin: np.ndarray          # XYZ (Å) of graph point (0, 0)
    ex: np.ndarray              # unit vector of graph-x in space
    ey: np.ndarray              # unit vector of graph-y in space
    source: str = ''
    fit_residual: float = 0.0   # max deviation of the linear frame fit (Å)

    @property
    def normal(self):
        n = np.cross(self.ex, self.ey)
        return n / np.linalg.norm(n)

    def project(self, xyz):
        """Graph coordinates (N, 2) and signed distance from the plane (N,) for Cartesian points (Å)."""
        d = np.asarray(xyz, float) - self.origin
        return np.c_[d @ self.ex, d @ self.ey], d @ self.normal


def read_plane(path):
    """Read a Multiwfn plane export. `path` may be the *_plane.txt file, the matching .npz, or the stem."""
    path = Path(path)
    txt = path if path.name.endswith('_plane.txt') else path.with_name(path.stem.replace('.npz', '') + '_plane.txt')
    if path.suffix == '.npz':
        txt = path.with_name(path.stem + '_plane.txt')
    if txt.exists():
        a = np.loadtxt(txt)
        P, g, F = a[:, :3], a[:, 3:5], a[:, 5]
        u = np.unique(np.round(g[:, 0], 6))
        v = np.unique(np.round(g[:, 1], 6))
        nx, ny = len(u), len(v)
        if nx * ny != len(F):
            raise ValueError(f'{txt}: {len(F)} points do not form a {nx}x{ny} grid')
        # order: find which graph coordinate varies fastest
        if abs(g[1, 1] - g[0, 1]) > 0 and abs(g[1, 0] - g[0, 0]) < 1e-9:
            Fg = F.reshape(nx, ny).T                 # rows = graph-x blocks, inner loop graph-y
        else:
            Fg = F.reshape(ny, nx)
        # sort by coordinate (Multiwfn writes increasing values, but be safe)
        A = np.c_[np.ones(len(g)), g]
        C, *_ = np.linalg.lstsq(A, P, rcond=None)
        origin, ex, ey = C[0], C[1], C[2]
        res = float(np.abs(A @ C - P).max())
        if not (abs(np.linalg.norm(ex) - 1) < 1e-3 and abs(np.linalg.norm(ey) - 1) < 1e-3 and abs(ex @ ey) < 1e-3):
            warnings.warn(f'{txt}: graph axes are not orthonormal in space (|ex|={np.linalg.norm(ex):.4f}, '
                          f'|ey|={np.linalg.norm(ey):.4f}, ex.ey={ex @ ey:.2e})')
        ex, ey = ex / np.linalg.norm(ex), ey / np.linalg.norm(ey)
        return PlaneGrid(u, v, Fg, origin, ex, ey, str(txt), res)
    npz = path if path.suffix == '.npz' else path.with_suffix('.npz')
    if not npz.exists():
        raise FileNotFoundError(f'no plane data for {path}')
    warnings.warn(f'{npz}: *_plane.txt missing; using the .npz, whose X/Y are projections on a rotated frame')
    z = np.load(npz)
    X, Y, F = z['X'], z['Y'], z['F']
    c, n = z['center'], z['normal']
    # rebuild an orthonormal frame from the projected grid: X/Y are coordinates along e1/e2 about c
    e = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, e)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    u = np.linspace(X.min(), X.max(), X.shape[1])
    v = np.linspace(Y.min(), Y.max(), X.shape[0])
    from scipy.interpolate import griddata
    UU, VV = np.meshgrid(u, v)
    Fi = griddata(np.c_[X.ravel(), Y.ravel()], F.ravel(), (UU, VV), method='linear')
    return PlaneGrid(u, v, Fi, c, e1, e2, str(npz), np.nan)


def plane_files(case_dir, plane_name):
    """Paths of the ELF and LOL exports for one plane of a case results directory."""
    d = Path(case_dir)
    out = {}
    for k in ('elf', 'lol'):
        p = d / f'{k}_{plane_name}_plane.txt'
        out[k] = p if p.exists() else d / f'{k}_{plane_name}.npz'
    return out


@dataclass
class PlaneMap:
    ax: object
    mappable: object
    inplane: np.ndarray
    placed: list
    dropped: list
    extent: tuple
    rotation_deg: float


def _auto_rotation(uv):
    """Angle (deg) that puts the long axis of the in-plane atoms horizontal."""
    if len(uv) < 2:
        return 0.0
    X = uv - uv.mean(0)
    w, V = np.linalg.eigh(X.T @ X)
    a = V[:, -1]
    ang = -np.degrees(np.arctan2(a[1], a[0]))
    return (ang + 90) % 180 - 90


def _rot(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def map_geometry(plane, symbols, xyz, atom_tol=0.5, rotate='auto', crop='atoms', crop_margin=1.4, centre=None,
                 mirror=False):
    """Projected atom positions, rotation and crop window shared by every panel drawn for one plane.

    Only atoms within ±atom_tol of the plane AND inside the computed grid count as in-plane.
    rotate: 'auto' (0 or 90 degrees, whichever gives a landscape window; never loses grid data), or an angle in
            degrees (arbitrary angles shrink the window so it stays inside the computed grid).
    crop: 'atoms' (in-plane atoms + crop_margin Å, clipped to the grid), 'full', or (x0, x1, y0, y1) in the
          rotated, centred frame.
    centre: None (default: mean of the in-plane heavy atoms), graph coordinates (u, v), or a Cartesian point
            (x, y, z; Å) projected onto the plane. A fixed centre keeps a molecule at the same place in the maps of
            two geometries (e.g. gas phase and IEF-PCM) whose in-plane atom sets differ.
    mirror: False (default) or True: reflect the graph y axis before rotating, i.e. show the plane as seen from
            the other side (to match the handedness of a 3D camera). Both options leave the default output
            unchanged.
    Returns dict(xy, dist, inplane, rot, R, centre, window, mirror)."""
    uv, dist = plane.project(xyz)
    U0, U1, V0, V1 = plane.u[0], plane.u[-1], plane.v[0], plane.v[-1]
    ingrid = (uv[:, 0] >= U0 - 0.05) & (uv[:, 0] <= U1 + 0.05) & (uv[:, 1] >= V0 - 0.05) & (uv[:, 1] <= V1 + 0.05)
    inplane = (np.abs(dist) <= atom_tol) & ingrid
    if not inplane.any():
        warnings.warn('no atom within the plane tolerance inside the grid')
    heavy = inplane & np.array([s != 'H' for s in symbols])
    ref = uv[heavy] if heavy.sum() >= 2 else uv[inplane]
    if centre is None:
        centre = ref.mean(0) if len(ref) else np.array([(U0 + U1) / 2, (V0 + V1) / 2])
    else:
        centre = np.asarray(centre, float)
        if centre.shape == (3,):
            centre = plane.project(centre[None])[0][0]
    grid_box = np.array([U0 - centre[0], U1 - centre[0], V0 - centre[1], V1 - centre[1]])
    M = np.diag([1.0, -1.0]) if mirror else np.eye(2)

    def window_for(theta):
        R = _rot(theta) @ M
        xy = (uv - centre) @ R.T
        if isinstance(crop, (tuple, list)):
            win = np.array(crop, float)
        elif crop == 'full':
            corners = np.array([[grid_box[0], grid_box[2]], [grid_box[1], grid_box[2]],
                                [grid_box[0], grid_box[3]], [grid_box[1], grid_box[3]]]) @ R.T
            win = np.array([corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()])
        else:
            pts = xy[inplane] if inplane.any() else np.zeros((1, 2))
            win = np.array([pts[:, 0].min() - crop_margin, pts[:, 0].max() + crop_margin,
                            pts[:, 1].min() - crop_margin, pts[:, 1].max() + crop_margin])
        return R, xy, _fit_window(win, R, grid_box)

    if rotate == 'auto':
        cands = [window_for(t) + (t,) for t in (0.0, 90.0)]
        # landscape first, then larger area
        R, xy, win, theta = max(cands, key=lambda c: ((c[2][1] - c[2][0]) >= (c[2][3] - c[2][2]),
                                                     (c[2][1] - c[2][0]) * (c[2][3] - c[2][2]), -c[3]))
    else:
        theta = float(rotate)
        R, xy, win = window_for(theta)
    return dict(xy=xy, dist=dist, inplane=inplane, rot=theta, R=R, centre=centre, window=tuple(win),
                mirror=bool(mirror))


def _fit_window(win, R, grid_box):
    """Shrink an axis-aligned window (rotated frame) until it lies inside the rotated grid rectangle."""
    win = np.array(win, float)

    def outside(pts):
        q = pts @ R                                   # back to the unrotated, centred grid frame
        return (q[:, 0] < grid_box[0] - 1e-9) | (q[:, 0] > grid_box[1] + 1e-9) | \
               (q[:, 1] < grid_box[2] - 1e-9) | (q[:, 1] > grid_box[3] + 1e-9)

    for _ in range(400):
        xs = np.linspace(win[0], win[1], 41)
        ys = np.linspace(win[2], win[3], 41)
        sides = [np.c_[np.full(41, win[0]), ys], np.c_[np.full(41, win[1]), ys],
                 np.c_[xs, np.full(41, win[2])], np.c_[xs, np.full(41, win[3])]]
        bad = [outside(sd).any() for sd in sides]
        if not any(bad):
            break
        step = 0.02
        if bad[0] and win[1] - win[0] > 1.0:
            win[0] += step
        if bad[1] and win[1] - win[0] > 1.0:
            win[1] -= step
        if bad[2] and win[3] - win[2] > 1.0:
            win[2] += step
        if bad[3] and win[3] - win[2] > 1.0:
            win[3] -= step
    return win


def plot_plane_map(ax, plane, symbols, xyz, labels=None, kind='ELF', cmap='turbo', geom=None, levels=None,
                   line_levels=None, atom_tol=0.5, offplane='hide', rotate='auto', crop='atoms', crop_margin=1.4,
                   bonds=True, hbonds='auto', label_h=True, fontsize=None, colorbar=True, cax=None,
                   rasterize_fill=True, xlabel='x (Å)', ylabel='y (Å)', edge_clear=0.15, strict=False,
                   label_atoms=None, label_prefer=None, line_smooth=0.0, line_min_extent=0.0):
    """Colour-filled contour map (0–1) of ELF or LOL in one plane, with projected atoms and labels.

    geom        output of map_geometry (pass the same one to several panels so they share rotation and window)
    offplane    'hide' or 'faint' (grey open circles for atoms farther than atom_tol from the plane)
    hbonds      'auto' (geometry.detect_hbonds; drawn dashed when H and acceptor are both in the plane), list, None
    edge_clear  atoms closer than this (Å) to the window edge are not drawn (their markers would be cut)
    colorbar    True -> a colour bar next to ax (or in cax); False -> none
    label_atoms None (default: label every drawn atom, subject to label_h) or an iterable of atom indices: only
                these atoms get a label (e.g. the heavy atoms plus the two N–H hydrogens); every drawn atom still
                counts as a label obstacle
    label_prefer None (default: each label prefers the direction away from its bonded in-plane neighbours) or a
                dict atom index -> (dx, dy) preferred label direction in map coordinates for those atoms (e.g. away
                from a ring centre, so that ring-atom labels sit outside the ring); other atoms keep the default
    line_smooth Gaussian sigma (grid points) applied to the field for the black contour LINES only (the colour
                fill is unchanged): removes pixel-size contour islands where a line level runs along a plateau.
                0 (default) = no smoothing, exactly as before
    line_min_extent  contour-line pieces (sub-paths) whose extent is smaller than this (map units, Å), and thin
                slivers (enclosed area < 0.15 x extent^2) smaller than 3 x this, are not drawn: broken dashes of a line
                level running along a plateau. 0 (default) keeps everything, as before
    Returns PlaneMap.
    """
    fontsize = style.SMALL_PT if fontsize is None else fontsize
    label_atoms = None if label_atoms is None else {int(k) for k in label_atoms}
    geom = geom or map_geometry(plane, symbols, xyz, atom_tol, rotate, crop, crop_margin)
    R, centre, win = geom['R'], geom['centre'], geom['window']
    UU, VV = np.meshgrid(plane.u, plane.v)
    P = np.c_[UU.ravel() - centre[0], VV.ravel() - centre[1]] @ R.T
    X, Y = P[:, 0].reshape(UU.shape), P[:, 1].reshape(UU.shape)
    levels = np.linspace(0, 1, 41) if levels is None else levels
    line_levels = np.round(np.arange(0.1, 0.91, 0.1), 2) if line_levels is None else line_levels
    F = np.clip(plane.F, 0, 1)
    cs = ax.contourf(X, Y, F, levels=levels, cmap=cmap, vmin=0, vmax=1, extend='neither')
    if rasterize_fill:
        cs.set_rasterized(True)
    FL = F
    if line_smooth:
        from scipy.ndimage import gaussian_filter
        FL = gaussian_filter(F, line_smooth, mode='nearest')
    cl = ax.contour(X, Y, FL, levels=line_levels, colors='k', linewidths=0.25, alpha=0.45)
    if line_min_extent:
        from matplotlib.path import Path as _Path
        kept = []
        for path in cl.get_paths():
            def keep(v):
                ext = np.ptp(v, axis=0).max()
                if ext < line_min_extent:
                    return False
                # a thin sliver (enclosed area < 0.15 x extent^2) up to 3 x the limit: a broken plateau dash
                area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1)))
                return not (ext < 3 * line_min_extent and area < 0.15 * ext ** 2)
            polys = [v for v in path.to_polygons(closed_only=False) if len(v) > 1 and keep(v)]
            kept.append(_Path.make_compound_path(*[_Path(v) for v in polys]) if polys else _Path(np.zeros((0, 2))))
        cl.set_paths(kept)
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect('equal', adjustable='box')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(direction='out')
    xy, inplane = geom['xy'], geom['inplane']
    e = edge_clear
    vis = (xy[:, 0] >= win[0] + e) & (xy[:, 0] <= win[1] - e) & (xy[:, 1] >= win[2] + e) & (xy[:, 1] <= win[3] - e)
    show = inplane & vis
    stroke = [pe.withStroke(linewidth=1.1, foreground='#111111')]
    bl = geo.detect_bonds(symbols, xyz)
    seg_list = []
    if bonds:
        for i, j in bl:
            if show[i] and show[j]:
                ln = Line2D(xy[[i, j], 0], xy[[i, j], 1], color='white', lw=0.55, zorder=5, path_effects=stroke,
                            solid_capstyle='round')
                ax.add_line(ln)
                seg_list.append((xy[i], xy[j]))
    hb_pairs = []
    if hbonds:
        hbl = geo.detect_hbonds(symbols, xyz, bl) if hbonds == 'auto' else hbonds
        for hb in hbl:
            h, a = (hb.h, hb.a) if isinstance(hb, geo.HBond) else (hb[1], hb[2])
            if show[h] and show[a]:
                hb_pairs.append((h, a))
                ln = Line2D(xy[[h, a], 0], xy[[h, a], 1], color='white', lw=0.6, ls=(0, (2.0, 1.3)), zorder=5,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground='#111111')])
                ax.add_line(ln)
                seg_list.append((xy[h], xy[a]))
    if offplane == 'faint':
        off = vis & ~inplane
        ax.scatter(xy[off, 0], xy[off, 1], s=5, facecolors='none', edgecolors='0.85', linewidths=0.4, zorder=4,
                   alpha=0.8)
    sizes = np.array([2.0 if s == 'H' else 3.2 for s in symbols])        # marker diameters, pt
    for i in np.nonzero(show)[0]:
        ax.plot(xy[i, 0], xy[i, 1], 'o', ms=sizes[i], mfc=style.ELEMENT_COLOURS.get(symbols[i], '#FF66CC'),
                mec='white', mew=0.5, zorder=6)
    placed, dropped = [], []
    if colorbar:
        fig = ax.figure
        cb = fig.colorbar(cs, cax=cax, ax=None if cax is not None else ax, ticks=np.linspace(0, 1, 6))
        cb.set_label(kind)
        cb.ax.tick_params(labelsize=style.TICK_PT)
        cb.outline.set_linewidth(0.5)
    if labels is not None:
        placer = LabelPlacer(ax, name=f'{kind} map labels', n_rings=3, gap_pt=0.6)
        for i in np.nonzero(show)[0]:
            placer.add_disc(xy[i], r_pt=sizes[i] / 2 + 0.4, owner=int(i))
        for p0, p1 in seg_list:
            placer.add_segment(p0, p1, hw_pt=0.55)
        n = len(symbols)
        for i in np.nonzero(show)[0]:
            if symbols[i] == 'H' and not label_h:
                continue
            if label_atoms is not None and int(i) not in label_atoms:
                continue
            nb = [j for a, b in list(bl) + hb_pairs for j in ((b,) if a == i else (a,) if b == i else ()) if show[j]]
            vecs = [xy[i] - xy[j] for j in nb]
            vecs = [v / np.linalg.norm(v) for v in vecs if np.linalg.norm(v) > 1e-6]
            pref = np.sum(vecs, 0) if vecs else None
            if pref is not None and np.linalg.norm(pref) < 0.3:
                pref = None
            if label_prefer is not None and int(i) in label_prefer:
                pref = label_prefer[int(i)]
            placer.add_item(labels[i], xy[i], radius_pt=sizes[i] / 2 + 0.3, prefer=pref,
                            priority=2 if symbols[i] == 'H' else 1, droppable=symbols[i] == 'H' and label_h == 'auto',
                            kw=dict(fontsize=fontsize, color='white', path_effects=stroke, leader_color='white'),
                            key=('atom', int(i)))
        placed, dropped = placer.place()
        if dropped:
            msg = f'LOUD WARNING plot_plane_map: labels dropped: {", ".join(it.text for it in dropped)}'
            if strict:
                raise qa.FigureQAError(msg)
            print(msg)
    return PlaneMap(ax, cs, inplane, placed, dropped, win, geom['rot'])


def _layout(width, win, n_panels, left_mm=11.0, right_mm=15.0, bottom_mm=10.0, top_mm=2.0, gap_mm=3.0,
            cbar_mm=2.8, cbar_gap_mm=1.8, max_height=None):
    """Axes rectangles (figure fractions) for n equal map panels plus one colour bar, figure height from aspect."""
    MM = style.MM
    wx, wy = win[1] - win[0], win[3] - win[2]
    panel_w = (width - (left_mm + right_mm + gap_mm * (n_panels - 1)) * MM) / n_panels
    panel_h = panel_w * wy / wx
    max_height = style.MAX_HEIGHT if max_height is None else max_height
    H = panel_h + (bottom_mm + top_mm) * MM
    if H > max_height:
        panel_h = max_height - (bottom_mm + top_mm) * MM
        panel_w = panel_h * wx / wy
        H = max_height
    rects = []
    x = left_mm * MM
    for k in range(n_panels):
        rects.append([x / width, bottom_mm * MM / H, panel_w / width, panel_h / H])
        x += panel_w + gap_mm * MM
    x += cbar_gap_mm * MM - gap_mm * MM
    cb = [x / width, bottom_mm * MM / H, cbar_mm * MM / width, panel_h / H]
    return H, rects, cb


def plane_map_figure(plane, symbols, xyz, labels=None, kind='ELF', width=None, cmap='turbo', **kw):
    """One map with its colour bar at print width (default one column). Returns (fig, ax, PlaneMap)."""
    style.apply()
    width = style.SINGLE if width is None else width
    geom = map_geometry(plane, symbols, xyz, kw.pop('atom_tol', 0.5), kw.pop('rotate', 'auto'),
                        kw.pop('crop', 'atoms'), kw.pop('crop_margin', 1.4))
    H, rects, cbr = _layout(width, geom['window'], 1)
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes(rects[0])
    cax = fig.add_axes(cbr)
    m = plot_plane_map(ax, plane, symbols, xyz, labels, kind=kind, cmap=cmap, geom=geom, cax=cax, **kw)
    return fig, ax, m


def elf_lol_figure(elf, lol, symbols, xyz, labels=None, width=None, cmap='turbo', panel_letters=('a', 'b'),
                   titles=('ELF', 'LOL'), cbar_label='ELF, LOL', **kw):
    """ELF and LOL for the same plane side by side: same rotation, window and label positions, one colour bar.

    elf, lol: PlaneGrid objects or paths. Returns (fig, [ax_elf, ax_lol], [PlaneMap, PlaneMap])."""
    style.apply()
    width = style.DOUBLE if width is None else width
    elf = elf if isinstance(elf, PlaneGrid) else read_plane(elf)
    lol = lol if isinstance(lol, PlaneGrid) else read_plane(lol)
    if np.abs(elf.origin - lol.origin).max() > 1e-3 or np.abs(elf.normal - lol.normal).max() > 1e-3:
        warnings.warn('ELF and LOL planes differ; the LOL panel uses its own frame')
    geom = map_geometry(elf, symbols, xyz, kw.pop('atom_tol', 0.5), kw.pop('rotate', 'auto'),
                        kw.pop('crop', 'atoms'), kw.pop('crop_margin', 1.4))
    top = 5.0 if (panel_letters or titles) else 2.0
    H, rects, cbr = _layout(width, geom['window'], 2, top_mm=top)
    fig = plt.figure(figsize=(width, H))
    ax1 = fig.add_axes(rects[0])
    ax2 = fig.add_axes(rects[1])
    cax = fig.add_axes(cbr)
    m1 = plot_plane_map(ax1, elf, symbols, xyz, labels, kind='ELF', cmap=cmap, geom=geom, colorbar=False, **kw)
    geom2 = geom if np.abs(elf.origin - lol.origin).max() < 1e-3 else None
    m2 = plot_plane_map(ax2, lol, symbols, xyz, labels, kind='LOL', cmap=cmap, geom=geom2, colorbar=False,
                        ylabel=None, **kw)
    ax2.tick_params(labelleft=False)
    cb = fig.colorbar(m1.mappable, cax=cax, ticks=np.linspace(0, 1, 6))
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.5)
    for k, (ax, t) in enumerate(zip((ax1, ax2), titles)):
        letter = f'({panel_letters[k]}) ' if panel_letters else ''
        ax.text(0.0, 1.0 + 1.2 * style.MM / (ax.get_position().height * H), f'{letter}{t}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=style.BASE_PT, fontweight='bold')
    return fig, [ax1, ax2], [m1, m2]
