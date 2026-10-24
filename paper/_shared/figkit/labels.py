"""Deterministic, overlap-free label placement (used by draw2d, maps and rdg).

Obstacles (atom discs, bonds, lines, existing texts, dense scatter regions) are declared in data coordinates with
sizes in data units and/or points, so they can be re-evaluated after any later layout change. Each label gets
candidate positions around its anchor (n_dirs directions x several rings; outer rings draw a leader line);
the placer checks hard constraints (no overlap with other labels, discs, segments, the allowed region), chooses
greedily by a cost (preferred direction, ring, clearance "repulsion", scatter density), then runs refinement passes
in which each label is re-placed given all others. Labels that cannot be placed are dropped, droppable ones
(hydrogen labels) first. verify() re-measures the rendered text bounding boxes; the same check is registered
with qa so it runs again at save time.

    placer = LabelPlacer(ax)
    placer.add_disc(xy, r_data=0.3)                 # atom
    placer.add_segment(p0, p1, hw_data=0.05)        # bond
    placer.add_item('C1', xy, radius_data=0.3, prefer=(0, 1), droppable=False)
    placed, dropped = placer.place()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.lines import Line2D

from . import qa


@dataclass
class LabelItem:
    text: str
    anchor: tuple                     # data coordinates
    radius_data: float = 0.0          # exclusion radius around the anchor, data units (x scale)
    radius_pt: float = 0.0            # ... plus this many points
    prefer: tuple | None = None       # preferred direction in display space (need not be normalised)
    priority: int = 0                 # lower is placed first
    droppable: bool = False           # may be dropped when crowded (hydrogen labels)
    leader: bool = True               # outer rings allowed (with a leader line)
    max_ring: int | None = None       # override the placer's number of rings for this item
    alts: tuple = ()                  # shorter wordings tried in order when the label cannot be placed
    kw: dict = field(default_factory=dict)
    key: object = None
    artist: object = None
    leader_artist: object = None
    placed: bool = False
    ring: int = -1


def _box_disc_dist(bx, cx, cy):
    """Distance from boxes (N,4: x0,y0,x1,y1) to points (M) -> (N, M)."""
    dx = np.maximum(np.maximum(bx[:, None, 0] - cx[None], cx[None] - bx[:, None, 2]), 0)
    dy = np.maximum(np.maximum(bx[:, None, 1] - cy[None], cy[None] - bx[:, None, 3]), 0)
    return np.hypot(dx, dy)


def _box_box_gap(a, b):
    """Gap distance between boxes a (N,4) and b (M,4); 0 when overlapping -> (N, M), plus overlap mask."""
    dx = np.maximum(np.maximum(a[:, None, 0] - b[None, :, 2], b[None, :, 0] - a[:, None, 2]), 0)
    dy = np.maximum(np.maximum(a[:, None, 1] - b[None, :, 3], b[None, :, 1] - a[:, None, 3]), 0)
    ov = (a[:, None, 0] < b[None, :, 2]) & (b[None, :, 0] < a[:, None, 2]) & \
         (a[:, None, 1] < b[None, :, 3]) & (b[None, :, 1] < a[:, None, 3])
    return np.hypot(dx, dy), ov


def _seg_box_hit(bx, segs):
    """Liang–Barsky: does segment k (x0,y0,x1,y1,hw) intersect box n inflated by hw_k? -> (N, K) bool."""
    if len(segs) == 0 or len(bx) == 0:
        return np.zeros((len(bx), len(segs)), bool)
    x0, y0, x1, y1, hw = [segs[:, i][None] for i in range(5)]
    xmin, ymin = bx[:, 0:1] - hw, bx[:, 1:2] - hw
    xmax, ymax = bx[:, 2:3] + hw, bx[:, 3:4] + hw
    dx, dy = x1 - x0, y1 - y0
    t0 = np.zeros(np.broadcast_shapes(xmin.shape, x0.shape))
    t1 = np.ones_like(t0)
    ok = np.ones_like(t0, bool)
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        p = np.broadcast_to(p, t0.shape)
        q = np.broadcast_to(q, t0.shape)
        par = np.abs(p) < 1e-12
        ok &= ~(par & (q < 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(par, 0.0, q / np.where(par, 1.0, p))
        t0 = np.where(~par & (p < 0), np.maximum(t0, r), t0)
        t1 = np.where(~par & (p > 0), np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


class LabelPlacer:
    """Place text labels without overlaps. See module docstring."""

    def __init__(self, ax, gap_pt=0.9, pad_pt=0.35, n_dirs=24, n_rings=4, ring_step=0.95, leader_from=2,
                 bounds='axes', bounds_pad_pt=0.5, clearance_pt=2.0, name='labels', freeze=True):
        self.ax = ax
        self.fig = ax.figure
        self.gap_pt, self.pad_pt = gap_pt, pad_pt
        self.n_dirs, self.n_rings, self.ring_step, self.leader_from = n_dirs, n_rings, ring_step, leader_from
        self.bounds, self.bounds_pad_pt = bounds, bounds_pad_pt
        self.clearance_pt = clearance_pt
        self.name = name
        self.discs = []        # (x, y, r_data, r_pt, owner)
        self.segs = []         # (x0, y0, x1, y1, hw_data, hw_pt)
        self.text_obstacles = []
        self.density = None    # (H, xedges_px, yedges_px) occupancy fraction
        self.items: list[LabelItem] = []
        if freeze:
            freeze_layout(self.fig, ax)

    # ---------------- obstacles ----------------
    def add_disc(self, xy, r_data=0.0, r_pt=0.0, owner=None):
        self.discs.append((float(xy[0]), float(xy[1]), float(r_data), float(r_pt), owner))

    def add_segment(self, p0, p1, hw_data=0.0, hw_pt=0.0):
        self.segs.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), float(hw_data), float(hw_pt)))

    def add_text_obstacle(self, text_artist):
        self.text_obstacles.append(text_artist)

    def add_density(self, x, y, bins=120, min_count=1, hard=0.02):
        """Scatter points as an obstacle (data coordinates). A candidate label box whose fraction of occupied
        cells (cells with >= min_count points) exceeds `hard` is rejected; lower occupancy is penalised."""
        P = self.ax.transData.transform(np.c_[x, y])
        bb = self.ax.bbox
        H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=bins, range=[[bb.x0, bb.x1], [bb.y0, bb.y1]])
        H = (H >= min_count).astype(float)
        self.density = (H, xe, ye)
        self.density_hard = hard

    # ---------------- items ----------------
    def add_item(self, text, anchor, **kw):
        it = LabelItem(text=text, anchor=(float(anchor[0]), float(anchor[1])), **kw)
        self.items.append(it)
        return it

    # ---------------- geometry in display space ----------------
    def _scale(self):
        """Display pixels per data unit (x direction; the drawing axes use equal aspect)."""
        t = self.ax.transData
        a = t.transform([(0, 0), (1, 0)])
        return float(np.hypot(*(a[1] - a[0])))

    def _ppt(self):
        return self.fig.dpi / 72.0

    def _disc_array(self, exclude_owner=None):
        if not self.discs:
            return np.zeros((0, 3))
        s, ppt = self._scale(), self._ppt()
        D = [(x, y, rd * s + rp * ppt) for (x, y, rd, rp, o) in self.discs if exclude_owner is None or o != exclude_owner]
        if not D:
            return np.zeros((0, 3))
        D = np.array(D)
        D[:, :2] = self.ax.transData.transform(D[:, :2])
        return D

    def _seg_array(self):
        if not self.segs:
            return np.zeros((0, 5))
        s, ppt = self._scale(), self._ppt()
        S = np.array(self.segs)
        p0 = self.ax.transData.transform(S[:, 0:2])
        p1 = self.ax.transData.transform(S[:, 2:4])
        return np.c_[p0, p1, S[:, 4] * s + S[:, 5] * ppt]

    def _bounds_px(self):
        if self.bounds is None:
            return None
        if self.bounds == 'axes':
            bb = self.ax.bbox
            b = np.array([bb.x0, bb.y0, bb.x1, bb.y1])
        elif self.bounds == 'figure':
            bb = self.fig.bbox
            b = np.array([bb.x0, bb.y0, bb.x1, bb.y1])
        else:
            b = np.asarray(self.bounds, float)
        p = self.bounds_pad_pt * self._ppt()
        return b + np.array([p, p, -p, -p])

    def _static_text_boxes(self, renderer):
        out = []
        for t in self.text_obstacles:
            if t.get_visible() and t.get_text():
                bb = t.get_window_extent(renderer)
                out.append([bb.x0, bb.y0, bb.x1, bb.y1])
        return np.array(out) if out else np.zeros((0, 4))

    # ---------------- candidates ----------------
    def _candidates(self, it, w, h):
        ppt = self._ppt()
        r = it.radius_data * self._scale() + it.radius_pt * ppt
        a = self.ax.transData.transform([it.anchor])[0]
        th = np.arange(self.n_dirs) * 2 * np.pi / self.n_dirs
        u = np.c_[np.cos(th), np.sin(th)]
        nr = self.n_rings if it.max_ring is None else min(self.n_rings, it.max_ring + 1)
        if not it.leader:
            nr = min(nr, self.leader_from)
        rings = np.arange(nr)
        D = r + self.gap_pt * ppt + rings * self.ring_step * h
        U = np.repeat(u[None], len(rings), 0).reshape(-1, 2)
        Dm = np.repeat(D[:, None], len(u), 1).reshape(-1)
        ring_id = np.repeat(rings[:, None], len(u), 1).reshape(-1)
        lo = np.zeros(len(U))
        hi = Dm + w + h
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            dx = np.maximum(np.abs(mid * U[:, 0]) - w / 2, 0)
            dy = np.maximum(np.abs(mid * U[:, 1]) - h / 2, 0)
            far = np.hypot(dx, dy) >= Dm
            hi = np.where(far, mid, hi)
            lo = np.where(far, lo, mid)
        C = a[None] + hi[:, None] * U
        pad = self.pad_pt * ppt
        boxes = np.c_[C[:, 0] - w / 2 - pad, C[:, 1] - h / 2 - pad, C[:, 0] + w / 2 + pad, C[:, 1] + h / 2 + pad]
        return a, r, U, ring_id, C, boxes

    def _evaluate(self, it, w, h, placed_boxes, placed_leaders, discs, segs, bnd, static_boxes):
        a, r, U, ring_id, C, boxes = self._candidates(it, w, h)
        ok = np.ones(len(boxes), bool)
        if bnd is not None:
            ok &= (boxes[:, 0] >= bnd[0]) & (boxes[:, 1] >= bnd[1]) & (boxes[:, 2] <= bnd[2]) & (boxes[:, 3] <= bnd[3])
        clear = np.zeros(len(boxes))
        sig = self.clearance_pt * self._ppt()
        if len(discs):
            dd = _box_disc_dist(boxes, discs[:, 0], discs[:, 1]) - discs[None, :, 2]
            ok &= ~(dd < 0).any(1)
            clear += np.exp(-np.maximum(dd, 0) / sig).sum(1)
        if len(segs):
            ok &= ~_seg_box_hit(boxes, segs).any(1)
        allb = [b for b in (placed_boxes, static_boxes) if len(b)]
        if allb:
            B = np.vstack(allb)
            gap, ov = _box_box_gap(boxes, B)
            ok &= ~ov.any(1)
            clear += 1.5 * np.exp(-gap / sig).sum(1)
        lead = ring_id >= self.leader_from
        # leader lines must not cross other labels
        if lead.any() and allb:
            L = self._leaders(a, r, C, boxes)
            hit = _seg_box_hit(np.vstack(allb), np.c_[L, np.zeros(len(L))]).T   # (Ncand, Nbox)
            ok &= ~(lead & hit.any(1))
        cost = 0.9 * ring_id + 0.8 * lead + 0.35 * clear
        if lead.any() and (len(discs) or len(segs)):
            # leaders must not pass through other atoms (discs not centred on the anchor); crossing bonds costs
            L = self._leaders(a, r, C, boxes)
            if len(discs):
                other = np.hypot(discs[:, 0] - a[0], discs[:, 1] - a[1]) > max(r, 1.0)
                Dd = discs[other]
                if len(Dd):
                    P0, P1 = L[:, None, 0:2], L[:, None, 2:4]
                    dv = P1 - P0
                    l2 = np.maximum((dv ** 2).sum(-1), 1e-9)
                    t = np.clip(((Dd[None, :, :2] - P0) * dv).sum(-1) / l2, 0, 1)
                    q = P0 + t[..., None] * dv
                    dist = np.hypot(*(Dd[None, :, :2] - q).transpose(2, 0, 1))
                    ok &= ~(lead & (dist < Dd[None, :, 2]).any(1))
            if len(segs):
                cross = _segments_cross(L, segs[:, :4])
                cost += 1.0 * (lead * cross.sum(1))
        # ambiguity: a label that sits closer to another atom than to its own reads as that atom's label
        if len(discs):
            dc = _box_disc_dist(boxes, discs[:, 0], discs[:, 1]) - discs[None, :, 2]
            own = np.hypot(discs[:, 0] - a[0], discs[:, 1] - a[1]) < 1e-3
            if own.any():
                d_own = dc[:, own].min(1)
                d_oth = np.where(own[None, :], np.inf, dc).min(1)
                cost += 2.5 * (d_oth < d_own)
        if it.prefer is not None:
            p = np.asarray(it.prefer, float)
            nrm = np.linalg.norm(p)
            if nrm > 0:
                cost += 2.5 * (1 - U @ (p / nrm)) / 2
        cost += -0.04 * U[:, 1] - 0.02 * U[:, 0]            # deterministic tie-break: up and right
        if self.density is not None:
            Hd, xe, ye = self.density
            occ = self._density_cost(boxes, Hd, xe, ye)
            cost += 4.0 * occ
            if getattr(self, 'density_hard', None) is not None:
                ok &= occ <= self.density_hard
            if lead.any():                          # leaders running through dense data are penalised
                L = self._leaders(a, r, C, boxes)
                ts = np.linspace(0.15, 0.95, 9)
                px = L[:, None, 0] + ts[None] * (L[:, None, 2] - L[:, None, 0])
                py = L[:, None, 1] + ts[None] * (L[:, None, 3] - L[:, None, 1])
                ii = np.clip(np.searchsorted(xe, px) - 1, 0, Hd.shape[0] - 1)
                jj = np.clip(np.searchsorted(ye, py) - 1, 0, Hd.shape[1] - 1)
                cost += 3.0 * lead * Hd[ii, jj].mean(1)
        # leaders crossing existing leaders: penalise
        if lead.any() and len(placed_leaders):
            L = self._leaders(a, r, C, boxes)
            cross = _segments_cross(L, np.asarray(placed_leaders))
            cost += 2.0 * (lead & cross.any(1))
        return ok, cost, C, boxes, ring_id, a, r

    @staticmethod
    def _density_cost(boxes, Hd, xe, ye):
        out = np.zeros(len(boxes))
        for n, b in enumerate(boxes):
            i0, i1 = np.searchsorted(xe, [b[0], b[2]])
            j0, j1 = np.searchsorted(ye, [b[1], b[3]])
            i0, j0 = max(i0 - 1, 0), max(j0 - 1, 0)
            sub = Hd[i0:i1, j0:j1]
            out[n] = sub.mean() if sub.size else 0.0
        return out

    @staticmethod
    def _leaders(a, r, C, boxes):
        """Leader segment from the anchor's exclusion circle to the nearest point of each box."""
        nx = np.clip(a[0], boxes[:, 0], boxes[:, 2])
        ny = np.clip(a[1], boxes[:, 1], boxes[:, 3])
        v = np.c_[nx - a[0], ny - a[1]]
        d = np.linalg.norm(v, axis=1, keepdims=True)
        d[d == 0] = 1
        start = a[None] + v / d * r
        return np.c_[start, nx, ny]

    def _measure(self, it, text, renderer):
        t = it.artist
        t.set_text(text)
        t.set_position(it.anchor)
        bb = t.get_window_extent(renderer)
        a = self.ax.transData.transform([it.anchor])[0]
        it._w, it._h = bb.width, bb.height
        it._off = np.array([bb.x0 + bb.width / 2 - a[0], bb.y0 + bb.height / 2 - a[1]])

    # ---------------- main ----------------
    def place(self, passes=2):
        """Place all items. Returns (placed, dropped) lists of LabelItem."""
        ax, fig = self.ax, self.fig
        renderer = fig.canvas.get_renderer()
        for it in self.items:
            kw = dict(ha='center', va='center', zorder=50, clip_on=False)
            kw.update({k: v for k, v in it.kw.items() if k not in ('leader_color', 'leader_lw')})
            t = ax.text(it.anchor[0], it.anchor[1], it.text, **kw)
            t.set_in_layout(True)      # layout is frozen; True keeps the label inside a tight bounding box
            it.artist = t
            self._measure(it, it.text, renderer)
            pe = t.get_path_effects()
            it._pe_pad = 0.0
            if 'bbox' in it.kw:                       # background box: its padding counts as part of the label
                it._pe_pad = 0.2 * t.get_fontsize() * self._ppt()
            for e in pe:
                lw = getattr(e, '_gc', {}).get('linewidth', None) if hasattr(e, '_gc') else None
                if lw:
                    it._pe_pad = max(it._pe_pad, lw / 2 * self._ppt())
        discs, segs, bnd = self._disc_array(), self._seg_array(), self._bounds_px()
        static = self._static_text_boxes(renderer)
        # difficulty ordering: fewer feasible candidates first, within priority classes
        diff = []
        for n, it in enumerate(self.items):
            ok, *_ = self._evaluate(it, it._w + 2 * it._pe_pad, it._h + 2 * it._pe_pad, np.zeros((0, 4)), [],
                                    discs, segs, bnd, static)
            diff.append((it.priority, ok.sum(), n))
        order = [n for _, _, n in sorted(diff)]
        state = {}                                   # n -> (box, leader or None, centre, ring)
        dropped = set()

        def boxes_except(n):
            ks = [k for k in state if k != n]
            B = np.array([state[k][0] for k in ks]) if ks else np.zeros((0, 4))
            L = [state[k][1] for k in ks if state[k][1] is not None]
            return B, L, ks

        def try_place(n, ignore=()):
            it = self.items[n]
            ks = [k for k in state if k != n and k not in ignore]
            B = np.array([state[k][0] for k in ks]) if ks else np.zeros((0, 4))
            L = [state[k][1] for k in ks if state[k][1] is not None]
            w, h = it._w + 2 * it._pe_pad, it._h + 2 * it._pe_pad
            ok, cost, C, boxes, ring_id, a, r = self._evaluate(it, w, h, B, L, discs, segs, bnd, static)
            if not ok.any():
                return None
            k = int(np.argmin(np.where(ok, cost, np.inf)))
            lead = None
            if ring_id[k] >= self.leader_from:
                lead = self._leaders(a, r, C[k:k + 1], boxes[k:k + 1])[0]
            return boxes[k], lead, C[k], int(ring_id[k])

        for n in order:
            res = try_place(n)
            if res is None and not self.items[n].droppable:
                droppable_placed = [k for k in state if self.items[k].droppable]
                res2 = try_place(n, ignore=droppable_placed)
                if res2 is not None:
                    box = res2[0][None]
                    for k in droppable_placed:
                        _, ov = _box_box_gap(box, np.array([state[k][0]]))
                        lk = state[k][1]
                        hit = False
                        if lk is not None:
                            hit = _seg_box_hit(box, np.array([[*lk, 0.0]]))[0, 0]
                        if ov[0, 0] or hit:
                            del state[k]
                            dropped.add(k)
                    res = try_place(n)
            if res is None and self.items[n].alts:
                it = self.items[n]
                for alt in it.alts:                          # shorter wordings, tried in order
                    self._measure(it, alt, renderer)
                    res = try_place(n)
                    if res is not None:
                        it.text = alt
                        break
                if res is None:
                    self._measure(it, it.text, renderer)
            if res is None:
                dropped.add(n)
                continue
            state[n] = res
        for _ in range(passes):
            for n in order:
                if n not in state:
                    continue
                res = try_place(n)
                if res is not None:
                    state[n] = res
        # retry dropped droppable labels once more after refinement
        for n in order:
            if n in dropped and n not in state:
                res = try_place(n)
                if res is not None:
                    state[n] = res
                    dropped.discard(n)
        placed, lost = [], []
        for n, it in enumerate(self.items):
            if n in state:
                box, lead, c, ring = state[n]
                pos = c - it._off
                it.artist.set_position(ax.transData.inverted().transform([pos])[0])
                it.placed, it.ring = True, ring
                if lead is not None:
                    p = ax.transData.inverted().transform(lead.reshape(2, 2))
                    col = it.kw.get('leader_color', it.kw.get('color', '0.3'))
                    ln = Line2D(p[:, 0], p[:, 1], lw=it.kw.get('leader_lw', 0.4), color=col, zorder=49,
                                solid_capstyle='butt')
                    if 'path_effects' in it.kw and it.kw.get('leader_color') == 'white':
                        import matplotlib.patheffects as _pe
                        ln.set_path_effects([_pe.withStroke(linewidth=1.0, foreground='#111111')])
                    ax.add_line(ln)
                    it.leader_artist = ln
                placed.append(it)
            else:
                it.artist.remove()
                it.artist = None
                lost.append(it)
        self._placed = placed
        qa.register_check(fig, self._check, self.name)
        return placed, lost

    # ---------------- verification ----------------
    def _check(self, fig, renderer):
        """Re-measure rendered label boxes against every obstacle (runs inside qa.check_figure)."""
        issues = []
        items = [it for it in getattr(self, '_placed', []) if it.artist is not None and it.artist.get_figure() is not None]
        if not items:
            return issues
        B = []
        for it in items:
            bb = it.artist.get_window_extent(renderer)
            p = it._pe_pad
            B.append([bb.x0 - p, bb.y0 - p, bb.x1 + p, bb.y1 + p])
        B = np.array(B)
        tol = 0.25 * self._ppt()
        Bi = B + np.array([tol, tol, -tol, -tol])
        discs, segs = self._disc_array(), self._seg_array()
        if len(discs):
            dd = _box_disc_dist(Bi, discs[:, 0], discs[:, 1]) - discs[None, :, 2]
            for i, j in zip(*np.nonzero(dd < 0)):
                issues.append(f'label "{items[i].text}" overlaps a disc/atom at depth {-dd[i, j] / self._ppt():.2f} pt')
        if len(segs):
            hit = _seg_box_hit(Bi, segs)
            for i in np.nonzero(hit.any(1))[0]:
                issues.append(f'label "{items[i].text}" overlaps a bond/line')
        _, ov = _box_box_gap(Bi, Bi)
        np.fill_diagonal(ov, False)
        for i, j in zip(*np.nonzero(np.triu(ov))):
            issues.append(f'labels "{items[i].text}" and "{items[j].text}" overlap')
        bnd = self._bounds_px()
        if bnd is not None:
            for i, b in enumerate(B):
                if b[0] < bnd[0] - 2 * tol or b[1] < bnd[1] - 2 * tol or b[2] > bnd[2] + 2 * tol or b[3] > bnd[3] + 2 * tol:
                    issues.append(f'label "{items[i].text}" leaves its allowed region')
        return issues

    def verify(self):
        renderer = self.fig.canvas.get_renderer()
        self.fig.canvas.draw()
        return self._check(self.fig, renderer)


def _segments_cross(A, B):
    """Proper crossing test between segments A (N,4) and B (M,4) -> (N, M) bool."""
    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    a0, a1 = A[:, None, 0:2], A[:, None, 2:4]
    b0, b1 = B[None, :, 0:2], B[None, :, 2:4]
    o1, o2 = orient(a0, a1, b0), orient(a0, a1, b1)
    o3, o4 = orient(b0, b1, a0), orient(b0, b1, a1)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def freeze_layout(fig, ax=None):
    """Run the layout engine once, then fix every axes position and the given axes' limits, so labels placed in
    display space stay where they were placed."""
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine('none')
    axes = [ax] if ax is not None else fig.axes
    for a in axes:
        a.set_xlim(a.get_xlim())
        a.set_ylim(a.get_ylim())
        a.set_autoscale_on(False)
