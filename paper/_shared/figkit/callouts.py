"""Callouts for 3D renders: labels placed on the white background AROUND a render (never on the picture) and
joined to their targets by short leader lines.  Made for MEP extrema (values outside the surface instead of a
number half on the surface edge), numbered NCI patches and sites buried under an opaque surface.

    from figkit import callouts, compose
    r = callouts.expand_canvas(r, left=90, top=40)                 # white room outside the picture (pixels)
    fig, axes = compose.grid([r, ...], ...)
    callouts.place(axes[0], [dict(pos=xyz_A, text='−52.6'), dict(target=(x_px, y_px), text='1')])

Placement rule: a label box must lie entirely on background pixels (every channel >= `thresh`), at least
`clearance_mm` away from rendered content, inside the axes, clear of every text already drawn in the figure and
of the other callouts and their leaders.  Candidates are 36 directions x leader lengths from `min_leader_mm` to
`max_leader_mm`; the shortest leader wins (ties: the preferred direction, then the leader that crosses the least
picture).  Leaders are dark lines with a thin white halo so they stay visible where they cross a coloured surface,
and start at a small dot on the target.  A qa check registered on the figure re-measures every callout at save
time (text on the picture, text outside the axes, leader longer than allowed).

The module only adds functions; nothing in render3d or compose changes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage
from matplotlib import patheffects
from matplotlib.lines import Line2D

from . import qa, style

__all__ = ['expand_canvas', 'content_mask', 'place', 'patch_target']

_LEADER_HALO = [patheffects.withStroke(linewidth=1.5, foreground='white')]


def expand_canvas(render, left=0, top=0, right=0, bottom=0):
    """The same Render on a larger white canvas (margins in image pixels), e.g. to make room for callouts
    outside the picture.  Render.project() and Render.visible() stay valid (only the crop offset changes)."""
    img = render.image
    h, w = img.shape[:2]
    L, T, R, B = (int(round(max(0, v))) for v in (left, top, right, bottom))
    out = np.full((h + T + B, w + L + R, 3), 255, np.uint8)
    out[T:T + h, L:L + w] = img
    ox, oy = render.offset
    return replace(render, image=out, offset=(ox - L, oy - T))


def content_mask(image, thresh=250):
    """True where the render shows something (any channel darker than `thresh`)."""
    return np.asarray(image)[..., :3].min(axis=2) < thresh


def patch_target(render, points, centre=None):
    """A good leader target on a surface patch: of the patch vertices `points` (Å) that are visible in the render
    (z-buffer), the one whose projection is nearest to the projected centroid of the visible ones (or to the
    projection of `centre`).  Returns (x, y) image pixels, or None when no vertex is visible."""
    P = np.atleast_2d(np.asarray(points, float))
    if not len(P):
        return None
    vis = render.visible(P, towards_camera=0.02)
    if not vis.any():
        return None
    xy, _ = render.project(P[vis])
    c = xy.mean(0) if centre is None else render.project(np.atleast_2d(centre))[0][0]
    return tuple(xy[np.argmin(np.hypot(*(xy - c).T))])


def _render_of(ax, render):
    return render if render is not None else getattr(ax, '_figkit_render')


def _segs_cross(a0, a1, B):
    """Does segment a0-a1 properly cross any of the segments B (N, 4)?"""
    if not len(B):
        return False
    B = np.asarray(B, float)

    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    b0, b1 = B[:, 0:2], B[:, 2:4]
    o1 = orient(a0[None], a1[None], b0)
    o2 = orient(a0[None], a1[None], b1)
    o3 = orient(b0, b1, a0[None])
    o4 = orient(b0, b1, a1[None])
    return bool(((o1 * o2 < 0) & (o3 * o4 < 0)).any())


def _seg_hits_box(a0, a1, box, n=24):
    t = np.linspace(0, 1, n)[:, None]
    P = a0[None] * (1 - t) + a1[None] * t
    return bool(((P[:, 0] > box[0]) & (P[:, 0] < box[2]) & (P[:, 1] > box[1]) & (P[:, 1] < box[3])).any())


def place(ax, items, render=None, fontsize=style.SMALL_PT, colour='black', min_leader_mm=1.0, max_leader_mm=9.0,
          clearance_mm=0.45, thresh=250, dot=True, dot_colour='#111111', leader_colour='#111111', leader_lw=0.5,
          halo=True, n_dirs=36, step_mm=0.2, pad_mm=0.25, avoid_texts=True, strict=True, name='3d callouts',
          text_kw=None, no_leader_below_mm=0.0, avoid_boxes=None, avoid_segments=None, patch_extent=False):
    """Place callouts on a render axes made by figkit.compose.grid (data coordinates = image pixels).

    items: dicts with 'text' and either 'target' ((x, y) image pixels, y down) or 'pos' ((3,) Å, projected with
           the render); optional 'prefer' (dx, dy) screen direction (y UP), 'fontsize' / 'kw' (text props),
           'radius' (image px: the leader starts on this circle round the target, e.g. an atom's ball; leader
           lengths are counted from it), 'footprint' ((N, 2) image px of the target's patch, e.g. the projected
           visible vertices of an NCI patch: the leader then stops where it reaches the patch instead of running
           over it to the target) and 'dot' (per-item override of `dot`); optional 'prefer_weight' (default 0.6:
           cost of a direction opposite to 'prefer', in mm of leader) and 'min_leader_mm' (per-item override of
           `min_leader_mm`, e.g. to force a leader).
    no_leader_below_mm: labels whose leader would be shorter than this sit directly next to the target, without
           a leader (atom labels on free atoms).
    avoid_boxes: optional extra rectangles (x0, y0, x1, y1) in image px (y down) that no label may overlap and no
           leader may pass through, e.g. other labels grown by a clearance.
    avoid_segments: optional (N, 4) segments (x0, y0, x1, y1) in image px that no leader may cross, e.g. the
           projected bonds of the molecule, so that a leader cannot be read as an extra bond.
    patch_extent: True measures a label by its bbox patch (e.g. a circled number) instead of its glyphs.
    The three defaults reproduce the previous behaviour exactly.
    The label never sits on the picture; the leader runs from the target (small dot) to the nearest point of the
    label box.  Items are placed in the given order.  Returns a list of dicts (text artist, leader, dot, box).
    strict=True raises when an item cannot be placed (otherwise it is skipped with a loud warning)."""
    r = _render_of(ax, render)
    fig = ax.figure
    img = r.image
    H, W = img.shape[:2]
    content = content_mask(img, thresh)
    dist = ndimage.distance_transform_edt(~content)           # image px to the nearest content pixel
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    T = ax.transData
    s_disp = float(abs(T.transform((1, 0))[0] - T.transform((0, 0))[0]))   # display px per image px
    mm = fig.dpi / 25.4 / s_disp                                            # image px per mm at print size
    clear = clearance_mm * mm
    pad = pad_mm * mm
    inv = T.inverted()

    def disp_box_to_img(bb):
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])

    others = []
    if avoid_texts:
        for t, bb in qa.rendered_texts(fig):
            b = disp_box_to_img(bb)
            if b[2] > 0 and b[0] < W and b[3] > 0 and b[1] < H:
                others.append(b)
    placed_boxes = list(others)
    if avoid_boxes is not None:
        placed_boxes += [np.asarray(b, float) for b in avoid_boxes]
    blocked = np.asarray(avoid_segments, float).reshape(-1, 4) if avoid_segments is not None else np.zeros((0, 4))
    placed_leaders = []
    out = []
    th = np.arange(n_dirs) * 2 * np.pi / n_dirs
    U = np.c_[np.cos(th), -np.sin(th)]                          # image coordinates: y down
    lengths = np.arange(min_leader_mm, max_leader_mm + 1e-9, step_mm) * mm
    for it in items:
        if 'target' in it and it['target'] is not None:
            tgt = np.asarray(it['target'], float)
        else:
            tgt = r.project(np.atleast_2d(it['pos']))[0][0]
        kw = dict(fontsize=it.get('fontsize', fontsize), color=colour, ha='center', va='center', zorder=60,
                  clip_on=False)
        kw.update(text_kw or {})
        kw.update(it.get('kw', {}))
        txt = ax.text(tgt[0], tgt[1], it['text'], **kw)
        if patch_extent and txt.get_bbox_patch() is not None:
            txt.update_bbox_position_size(renderer)
            bb = disp_box_to_img(txt.get_bbox_patch().get_window_extent(renderer))
        else:
            bb = disp_box_to_img(txt.get_window_extent(renderer))
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        pref = it.get('prefer')
        if pref is not None:
            pv = np.array([pref[0], -pref[1]], float)
            pv /= np.linalg.norm(pv)
        else:
            c = np.array([W / 2, H / 2])                          # default: away from the picture centre
            pv = tgt - c
            pv = pv / np.linalg.norm(pv) if np.linalg.norm(pv) > 0 else np.array([1.0, 0.0])
        rad = float(it.get('radius', 0.0))
        w_pref = float(it.get('prefer_weight', 0.6))
        lens = lengths if it.get('min_leader_mm') is None else lengths[lengths >= float(it['min_leader_mm']) * mm - 1e-9]
        best = None
        for u in U:
            ext = min((w / 2) / max(abs(u[0]), 1e-9), (h / 2) / max(abs(u[1]), 1e-9))
            for L in lens:
                cen = tgt + (rad + L + ext) * u
                box = np.array([cen[0] - w / 2, cen[1] - h / 2, cen[0] + w / 2, cen[1] + h / 2])
                if box[0] < 1 or box[1] < 1 or box[2] > W - 1 or box[3] > H - 1:
                    continue
                i0, i1 = int(np.floor(box[1])), int(np.ceil(box[3]))
                j0, j1 = int(np.floor(box[0])), int(np.ceil(box[2]))
                if dist[i0:i1, j0:j1].min() < clear:
                    continue
                if any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]) for b in placed_boxes):
                    continue
                end = np.clip(tgt, box[:2], box[2:])
                v = end - tgt
                start = tgt + v / max(np.hypot(*v), 1e-9) * min(rad, np.hypot(*v))
                if _segs_cross(start, end, placed_leaders):
                    continue
                if len(blocked) and _segs_cross(start, end, blocked):
                    continue
                if any(_seg_hits_box(start, end, b) for b in placed_boxes):
                    continue
                n = 16
                tt = np.linspace(0, 1, n)[:, None]
                P = (start[None] * (1 - tt) + end[None] * tt).astype(int)
                P[:, 0] = np.clip(P[:, 0], 0, W - 1)
                P[:, 1] = np.clip(P[:, 1], 0, H - 1)
                over = content[P[:, 1], P[:, 0]].mean()
                cost = L / mm + w_pref * (1 - float(u @ pv)) + 1.5 * over
                if best is None or cost < best[0]:
                    best = (cost, cen, box, end, start)
                break                                          # longer leaders in this direction only cost more
        if best is None:
            txt.remove()
            msg = f'callouts: no room outside the picture for "{it["text"]}" (expand the canvas or allow longer leaders)'
            if strict:
                raise qa.FigureQAError(msg)
            print('LOUD WARNING ' + msg)
            continue
        _, cen, box, end, start = best
        fp = it.get('footprint')
        if fp is not None and len(fp):
            start = _entry_point(np.asarray(fp, float), end, start, 0.25 * mm)
        txt.set_position((cen[0], cen[1]))
        L_mm = float(np.hypot(*(end - start)) / mm)
        ln = None
        if L_mm >= no_leader_below_mm:
            ln = Line2D([start[0], end[0]], [start[1], end[1]], lw=leader_lw, color=leader_colour, zorder=58,
                        solid_capstyle='round', path_effects=_LEADER_HALO if halo else None)
            ax.add_line(ln)
            placed_leaders.append([start[0], start[1], end[0], end[1]])
        dt = None
        if it.get('dot', dot):
            dt = ax.plot([tgt[0]], [tgt[1]], 'o', ms=2.2, mfc=dot_colour, mec='white', mew=0.5, zorder=59)[0]
        placed_boxes.append(box)
        out.append(dict(item=it, text=txt, leader=ln, dot=it.get('dot', dot), dot_artist=dt, box=box, target=tgt,
                        leader_mm=L_mm))
    qa.register_check(fig, lambda f, rend: _check(ax, out, dist, clear, max_leader_mm, rend), name)
    return out


def _entry_point(fp, end, start, tol):
    """First point, walking from the label end of a leader towards its start, within `tol` px of a footprint
    point (falls back to the original start)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(fp)
    t = np.linspace(0, 1, 400)[:, None]
    P = end[None] * (1 - t) + start[None] * t
    d, _ = tree.query(P)
    hit = np.flatnonzero(d <= tol)
    return P[hit[0]] if len(hit) else start


def _check(ax, out, dist, clear, max_leader_mm, renderer):
    """Save-time re-measurement: callout text off the picture and inside the axes; leaders within the limit."""
    issues = []
    fig = ax.figure
    T = ax.transData
    inv = T.inverted()
    s_disp = float(abs(T.transform((1, 0))[0] - T.transform((0, 0))[0]))
    mm = fig.dpi / 25.4 / s_disp
    H, W = dist.shape
    for o in out:
        t = o['text']
        if t.get_figure() is None:
            continue
        bb = t.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x0 < -0.5 or y0 < -0.5 or x1 > W + 0.5 or y1 > H + 0.5:
            issues.append(f'callout "{t.get_text()}" leaves its panel')
            continue
        i0, i1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), H)
        j0, j1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), W)
        if i1 > i0 and j1 > j0 and dist[i0:i1, j0:j1].min() < 0.5 * clear:
            issues.append(f'callout "{t.get_text()}" touches the picture')
        if o['leader'] is None:
            continue
        xd, yd = o['leader'].get_data()
        L = float(np.hypot(xd[1] - xd[0], yd[1] - yd[0])) / mm
        if L > max_leader_mm + 0.5:
            issues.append(f'callout "{t.get_text()}" has a {L:.1f} mm leader (> {max_leader_mm} mm)')
    return issues
