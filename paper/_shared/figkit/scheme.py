"""2D chemical-structure schemes from RDKit, drawn as vector matplotlib (lines + Liberation Sans text).

Because every symbol is a matplotlib Text, qa.save checks the scheme like any other figure (font sizes, text
overlaps, clipping). Coordinates come from RDKit (CoordGen), bond length normalised to 1; the drawing scale is
given in data units per bond, so an axes set up in millimetres (xlim 0..W_mm, ylim 0..H_mm, equal aspect) gives
exact print sizes.

    from figkit import scheme
    s = scheme.Structure.from_smiles('c1ccc2[nH:6]n[nH+:4]c2c1', charge=+1)    # sanitised, charge asserted
    s.orient(s.idx(4), s.h_of(s.idx(4)), 180)     # rotate: vector N4 -> its H points left
    s.mirror_to(i, j, k, side=+1)                 # reflect so atom k lies left (+1) / right (-1) of i -> j
    ax = scheme.mm_axes(fig)                      # one axes over the whole figure, 1 data unit = 1 mm
    d = scheme.draw_structure(ax, s, origin=(x_mm, y_mm), bond=5.5, fontsize=8.5,
                              atom_colours={h: '#0E7C7B'})
    d.pos[i]                                      # atom positions (data units)
    scheme.add_obstacles(placer, d)               # atoms, bonds, symbols as LabelPlacer obstacles

Conventions (close to the ACS 1996 document settings): skeletal carbons; heteroatom symbols in black (or
coloured per atom); explicit hydrogens only on heteroatoms (explicit_h='hetero') so N-H / O-H protons can be
highlighted and labelled; ring double bonds drawn inside the ring and shortened; other double bonds centred;
formal charges as small + / − beside the atom in the free direction; nitro groups condensed to NO2 / O2N so that
the only charges shown are the ionic ones (condense_nitro=True).
Added 2026-09-23 for the BTPC scheme (see CHANGES.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from . import style

CAP = 0.688                      # Liberation Sans cap height / em
PT_MM = 25.4 / 72.0


# ---------------------------------------------------------------------------------------------------------------
# structure (RDKit molecule + 2D coordinates)
# ---------------------------------------------------------------------------------------------------------------
class Structure:
    def __init__(self, mol, xy, groups=None, hidden=None, maps=None):
        self.mol = mol
        self.xy = np.asarray(xy, float)
        self.groups = dict(groups or {})        # atom index -> (text when pointing right/up, text when left)
        self.hidden = set(hidden or ())
        self.maps = dict(maps or {})            # atom-map number in the SMILES -> atom index

    @classmethod
    def from_smiles(cls, smiles, charge=None, explicit_h='hetero', condense_nitro=True, coordgen=True):
        """Parse and sanitise `smiles` (atom-map numbers may tag atoms: s.idx(map)), assert the total formal
        charge, add explicit H on heteroatoms, compute CoordGen 2D coordinates and kekulise."""
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f'RDKit could not parse {smiles!r}')
        q = Chem.GetFormalCharge(m)
        if charge is not None and q != charge:
            raise AssertionError(f'{smiles}: total formal charge {q}, expected {charge}')
        maps = {}
        for a in m.GetAtoms():
            if a.GetAtomMapNum():
                maps[a.GetAtomMapNum()] = a.GetIdx()
                a.SetAtomMapNum(0)
        if explicit_h == 'hetero':
            on = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() not in ('C', 'H') and a.GetTotalNumHs() > 0]
        else:
            on = list(explicit_h or [])
        if on:
            m = Chem.AddHs(m, onlyOnAtoms=on)
        rdDepictor.SetPreferCoordGen(bool(coordgen))
        rdDepictor.Compute2DCoords(m)
        Chem.Kekulize(m, clearAromaticFlags=True)
        c = m.GetConformer()
        xy = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
        L = np.median([np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]) for b in m.GetBonds()])
        xy = (xy - xy.mean(0)) / L
        groups, hidden = {}, set()
        if condense_nitro:
            patt = Chem.MolFromSmarts('[N+](=O)[O-]')
            for n, o1, o2 in m.GetSubstructMatches(patt):
                groups[n] = ('NO$_2$', 'O$_2$N')
                hidden |= {o1, o2}
        return cls(m, xy, groups, hidden, maps)

    # ---- atom lookup ----
    def idx(self, mapnum):
        return self.maps[mapnum]

    def h_of(self, i):
        """Indices of the explicit hydrogens on atom i (first one if exactly one)."""
        hs = [n.GetIdx() for n in self.mol.GetAtomWithIdx(i).GetNeighbors() if n.GetSymbol() == 'H']
        return hs[0] if len(hs) == 1 else hs

    def neighbours(self, i):
        return [n.GetIdx() for n in self.mol.GetAtomWithIdx(i).GetNeighbors()]

    # ---- geometry ----
    def orient(self, i, j, angle_deg):
        """Rotate about the centroid so that the vector atom i -> atom j points at angle_deg."""
        v = self.xy[j] - self.xy[i]
        a = np.radians(angle_deg) - np.arctan2(v[1], v[0])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = self.xy.mean(0)
        self.xy = (self.xy - c) @ R.T + c
        return self

    def mirror_to(self, i, j, k, side=+1):
        """Reflect the whole structure across the line i-j if needed so atom k lies on `side` of i -> j
        (+1 left, -1 right)."""
        p, q = self.xy[i], self.xy[j]
        u = (q - p) / np.linalg.norm(q - p)
        cr = u[0] * (self.xy[k] - p)[1] - u[1] * (self.xy[k] - p)[0]
        if np.sign(cr) != np.sign(side):
            d = self.xy - p
            self.xy = p + 2 * np.outer(d @ u, u) - d
        return self

    def reflect_atom(self, k, i, j):
        """Reflect atom k (e.g. an O-H hydrogen) across the line through atoms i and j."""
        p, q = self.xy[i], self.xy[j]
        u = (q - p) / np.linalg.norm(q - p)
        d = self.xy[k] - p
        self.xy[k] = p + 2 * (d @ u) * u - d
        return self

    def rotate_atom(self, k, centre, angle_deg):
        """Set atom k at the same distance from atom `centre` but at direction angle_deg."""
        r = np.linalg.norm(self.xy[k] - self.xy[centre])
        a = np.radians(angle_deg)
        self.xy[k] = self.xy[centre] + r * np.array([np.cos(a), np.sin(a)])
        return self

    def visible(self):
        return [i for i in range(self.mol.GetNumAtoms()) if i not in self.hidden]

    def extent(self):
        v = self.xy[self.visible()]
        return np.array([v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()])


# ---------------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Drawn:
    pos: np.ndarray
    texts: list = field(default_factory=list)          # every Text drawn (symbols, groups, charges)
    symbol_text: dict = field(default_factory=dict)    # atom index -> Text
    segments: list = field(default_factory=list)       # (p0, p1) of every line drawn
    boxes: dict = field(default_factory=dict)          # atom index -> (cx, cy, hw, hh) clip box, data units
    lines: list = field(default_factory=list)
    upp: float = 1.0                                   # data units per point


def mm_axes(fig):
    """One axes over the whole figure, axis off, 1 data unit = 1 mm, origin at the lower-left corner."""
    W, H = fig.get_size_inches() * 25.4
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)
    ax.axis('off')
    return ax


def units_per_point(ax):
    fig = ax.figure
    xl = ax.get_xlim()
    w_pt = ax.get_position().width * fig.get_figwidth() * 72.0
    return (xl[1] - xl[0]) / w_pt


def _ink(s, fontsize, weight='normal'):
    """Ink extents (x0, y0, x1, y1) in points of string s at fontsize, baseline at y = 0."""
    fp = FontProperties(family=plt.rcParams['font.sans-serif'][0], weight=weight)
    tp = TextPath((0, 0), s, size=fontsize, prop=fp)
    e = tp.get_extents()
    return e.x0, e.y0, e.x1, e.y1


def _advance(s, fontsize, weight='normal'):
    """Advance width (points) of s, from the ink extents of 'x' + s + 'x' minus those of 'xx'."""
    a = _ink('x' + s + 'x', fontsize, weight)
    b = _ink('xx', fontsize, weight)
    return (a[2] - a[0]) - (b[2] - b[0])


def _clip_t(p, q, box):
    """Fraction t along p -> q at which the segment leaves the rectangle box = (cx, cy, hw, hh) centred on p."""
    d = q - p
    ts = []
    if abs(d[0]) > 1e-12:
        ts.append(box[2] / abs(d[0]))
    if abs(d[1]) > 1e-12:
        ts.append(box[3] / abs(d[1]))
    return min(ts) if ts else 0.0


def _seg_rect_dist(p0, p1, rect):
    """Distance from segment p0-p1 to the axis-aligned rectangle (x0, y0, x1, y1); 0 if they intersect."""
    x0, y0, x1, y1 = rect
    best = np.inf
    for t in np.linspace(0, 1, 41):
        p = p0 + t * (p1 - p0)
        dx = max(x0 - p[0], 0, p[0] - x1)
        dy = max(y0 - p[1], 0, p[1] - y1)
        best = min(best, np.hypot(dx, dy))
    return best


def draw_structure(ax, s, origin=(0.0, 0.0), bond=5.0, fontsize=8.5, lw=0.8, colour='black', atom_colours=None,
                   atom_weights=None, bond_colours=None, double_gap=0.18, double_shorten=0.13, pad_pt=0.9,
                   charge_scale=0.82, show_charges=True, zorder=3):
    """Draw Structure `s` with its coordinate origin at `origin` and `bond` data units per bond length.

    atom_colours / atom_weights  dict atom index -> colour / font weight for the atom symbol (e.g. a highlighted H)
    bond_colours                 dict (i, j) -> colour
    Returns Drawn (positions, texts, line segments, symbol clip boxes)."""
    atom_colours = atom_colours or {}
    atom_weights = atom_weights or {}
    bond_colours = {tuple(sorted(k)): v for k, v in (bond_colours or {}).items()}
    upp = units_per_point(ax)
    m = s.mol
    P = np.asarray(origin, float) + s.xy * bond
    vis = set(s.visible())
    d = Drawn(pos=P, upp=upp)
    cap = CAP * fontsize * upp
    pad = pad_pt * upp
    # ---- atom symbols ----
    for i in sorted(vis):
        a = m.GetAtomWithIdx(i)
        sym = a.GetSymbol()
        if sym == 'C' and i not in s.groups and a.GetDegree() > 0:
            continue
        col = atom_colours.get(i, colour)
        wt = atom_weights.get(i, 'normal')
        if i in s.groups:
            nb = [n for n in s.neighbours(i) if n in vis]
            v = P[i] - P[nb[0]] if nb else np.array([1.0, 0.0])
            left = v[0] < -0.35 * np.linalg.norm(v)
            txt = s.groups[i][1] if left else s.groups[i][0]
            wN = _advance('N', fontsize, wt) * upp
            if left:
                t = ax.text(P[i, 0] + wN / 2, P[i, 1] - cap / 2, txt, ha='right', va='baseline', fontsize=fontsize,
                            color=col, fontweight=wt, zorder=zorder + 1)
            else:
                t = ax.text(P[i, 0] - wN / 2, P[i, 1] - cap / 2, txt, ha='left', va='baseline', fontsize=fontsize,
                            color=col, fontweight=wt, zorder=zorder + 1)
            x0, _, x1, _ = _ink('N', fontsize, wt)
            hw = (x1 - x0) / 2 * upp + pad
        else:
            t = ax.text(P[i, 0], P[i, 1] - cap / 2, sym, ha='center', va='baseline', fontsize=fontsize, color=col,
                        fontweight=wt, zorder=zorder + 1)
            x0, _, x1, _ = _ink(sym, fontsize, wt)
            hw = (x1 - x0) / 2 * upp + pad
        d.texts.append(t)
        d.symbol_text[i] = t
        d.boxes[i] = (P[i, 0], P[i, 1], hw, cap / 2 + pad)
    # ---- bonds ----
    ri = m.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings()]

    def ring_centre(i, j):
        cands = [r for r in rings if i in r and j in r]
        if not cands:
            return None
        r = min(cands, key=len)
        return P[r].mean(0)

    def add_line(p0, p1, c):
        ln = Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=lw, solid_capstyle='round', zorder=zorder)
        ax.add_line(ln)
        d.lines.append(ln)
        d.segments.append((np.array(p0), np.array(p1)))

    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in vis or j not in vis:
            continue
        c = bond_colours.get(tuple(sorted((i, j))), colour)
        p, q = P[i], P[j]
        L = np.linalg.norm(q - p)
        u = (q - p) / L
        n = np.array([-u[1], u[0]])
        ti = _clip_t(p, q, d.boxes[i]) if i in d.boxes else 0.0
        tj = _clip_t(q, p, d.boxes[j]) if j in d.boxes else 0.0
        order = b.GetBondTypeAsDouble()
        if order == 2:
            rc = ring_centre(i, j)
            g = double_gap * bond
            if rc is not None:
                add_line(p + ti * (q - p), q - tj * (q - p), c)
                if n @ (rc - (p + q) / 2) < 0:
                    n = -n
                a0 = max(ti * L, double_shorten * L) if i not in d.boxes else ti * L + 0.4 * g
                a1 = max(tj * L, double_shorten * L) if j not in d.boxes else tj * L + 0.4 * g
                add_line(p + n * g + u * a0, q + n * g - u * a1, c)
            else:
                for sgn in (+0.5, -0.5):
                    add_line(p + n * g * sgn + ti * (q - p), q + n * g * sgn - tj * (q - p), c)
        elif order == 3:
            g = double_gap * bond
            for sgn in (-1, 0, 1):
                add_line(p + n * g * sgn + ti * (q - p), q + n * g * sgn - tj * (q - p), c)
        else:
            add_line(p + ti * (q - p), q - tj * (q - p), c)
    # ---- formal charges ----
    if show_charges:
        for i in sorted(vis):
            a = m.GetAtomWithIdx(i)
            fc = a.GetFormalCharge()
            if fc == 0 or i in s.groups:
                continue
            txt = ('' if abs(fc) == 1 else str(abs(fc))) + ('+' if fc > 0 else '−')
            cfs = fontsize * charge_scale
            w_sym = (d.boxes[i][2] - pad) if i in d.boxes else 0.8 * upp
            cw = _advance(txt, cfs) * upp
            best, best_cost = None, None
            for sx in (+1, -1):
                for dy in (0.42, -0.42, 0.62, -0.62, 0.85, -0.85):
                    cx = P[i, 0] + sx * (w_sym + 0.9 * upp + cw / 2)
                    cy = P[i, 1] + dy * fontsize * upp
                    rect = (cx - cw / 2 - 0.6 * upp, cy - 0.36 * cfs * upp, cx + cw / 2 + 0.6 * upp,
                            cy + 0.36 * cfs * upp)
                    clear = min([_seg_rect_dist(p0, p1, rect) for p0, p1 in d.segments] + [np.inf])
                    for k, bx in d.boxes.items():
                        if k == i:
                            continue
                        dx = max(bx[0] - bx[2] - rect[2], 0, rect[0] - (bx[0] + bx[2]))
                        dyy = max(bx[1] - bx[3] - rect[3], 0, rect[1] - (bx[1] + bx[3]))
                        clear = min(clear, np.hypot(dx, dyy))
                    # prefer the upper right (N+, O-), then any raised position; lowered positions last
                    cost = -min(clear, 3.0 * upp) + (0.0 if (sx > 0 and dy > 0) else 0.15 * upp) + \
                        0.1 * upp * (abs(dy) > 0.5) + 0.1 * upp * (abs(dy) > 0.7) + 0.6 * upp * (dy < 0)
                    if best_cost is None or cost < best_cost:
                        best, best_cost = (cx, cy), cost
            ct = ax.text(best[0], best[1] - 0.36 * cfs * upp, txt, ha='center', va='baseline', fontsize=cfs,
                         color=atom_colours.get(i, colour), zorder=zorder + 1)
            d.texts.append(ct)
    return d


def add_obstacles(placer, d, bond_hw_pt=0.8, carbon_r_pt=1.2):
    """Declare everything a Drawn structure occupies as LabelPlacer obstacles."""
    for p0, p1 in d.segments:
        placer.add_segment(p0, p1, hw_pt=bond_hw_pt)
    for t in d.texts:
        placer.add_text_obstacle(t)
    for i, p in enumerate(d.pos):
        if i not in d.boxes:
            placer.add_disc(p, r_pt=carbon_r_pt)
