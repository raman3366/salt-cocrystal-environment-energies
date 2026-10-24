"""RDKit 2D depictions drawn as vector matplotlib artists (for chemical schemes).

RDKit's MolDraw2DSVG writes every bond as a stroked path and, with a FreeType font file, every atom label as
filled glyph outlines. This module parses that SVG and re-draws it in a matplotlib axes, so a scheme can be
composed in matplotlib (arrows, hydrogen bonds, annotations placed with figkit.labels) and saved as true
vectors through figkit.qa.save, in the house font (Liberation Sans).

    Depiction(mol, bond_mm=5.0, font_pt=7.0, line_pt=0.6, ...)    one molecule, drawn by RDKit at print scale
        .atom_px(i)                    RDKit draw coordinates of atom i (px, y down)
        .extent_px                     (x0, y0, x1, y1) of everything drawn (px)
        .draw(ax, origin, recolour=None, zorder=3) -> Placed
    Placed
        .xy(i)                         atom position in data units (mm; axes must use mm, y up, equal aspect)
        .atom_box(i)                   bounding box of atom i's label glyphs in data units, or None (no label)
        .segments()                    bond strokes as (p0, p1, half_width_mm) for LabelPlacer obstacles
        .glyph_boxes()                 all label boxes (x0, y0, x1, y1) in data units
        .bbox                          (x0, y0, x1, y1) of the whole depiction in data units
    parse_svg(svg) -> list of dict(kind, verts, codes, colour, lw_px, dash, classes)

Units: the axes must be in millimetres at print size (e.g. an axes filling a figure of W x H mm with xlim (0, W),
ylim (0, H)); Depiction scales RDKit's pixels so that one bond is bond_mm long and labels are font_pt high.
Added 2026-09-23 for the BTP scheme (see CHANGES.md); nothing else in figkit depends on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

from . import style

PT_PER_MM = 72.0 / 25.4
_TOKEN = re.compile(r'[MLQCZmlqcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')
_ELEM = re.compile(r"<(path|ellipse|rect|text)\b([^>]*)/?>", re.S)
_ATTR = re.compile(r"([\w:-]+)='([^']*)'")
# ratio of the RDKit FreeType font size (px) to the em size of the font as drawn; measured with Liberation Sans
# ('O' height / em = 0.745) in RDKit 2026.03: glyphs come out at 1.0 x fixedFontSize em.
_EM_PER_FONTPX = 1.0


def _style_dict(s):
    out = {}
    for part in s.split(';'):
        if ':' in part:
            k, v = part.split(':', 1)
            out[k.strip()] = v.strip()
    return out


def _path_from_d(d):
    toks = _TOKEN.findall(d)
    verts, codes = [], []
    i, cmd = 0, None
    start = None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in 'Zz':
                if start is not None:
                    verts.append(start)
                    codes.append(MPath.CLOSEPOLY)
                continue
        if cmd is None:
            raise ValueError(f'SVG path without a command: {d[:40]}')
        if cmd in 'ml':
            raise ValueError('relative SVG path commands are not supported')
        n = {'M': 2, 'L': 2, 'Q': 4, 'C': 6}[cmd]
        vals = [float(x) for x in toks[i:i + n]]
        i += n
        pts = [(vals[k], vals[k + 1]) for k in range(0, n, 2)]
        if cmd == 'M':
            verts.append(pts[0])
            codes.append(MPath.MOVETO)
            start = pts[0]
            cmd = 'L'                       # implicit lineto after a moveto
        elif cmd == 'L':
            verts.append(pts[0])
            codes.append(MPath.LINETO)
        elif cmd == 'Q':
            verts += pts
            codes += [MPath.CURVE3, MPath.CURVE3]
        elif cmd == 'C':
            verts += pts
            codes += [MPath.CURVE4] * 3
    return np.array(verts, float).reshape(-1, 2), np.array(codes, np.uint8)


def parse_svg(svg):
    """Drawable items of an RDKit SVG: dict(kind='stroke'|'fill', verts (n, 2) px, codes, colour, lw_px, dash,
    classes). The white background rect is skipped; <text> elements (no FreeType) raise."""
    items = []
    for tag, attrs in _ELEM.findall(svg):
        a = dict(_ATTR.findall(attrs))
        st = _style_dict(a.get('style', ''))
        classes = a.get('class', '').split()
        if tag == 'rect':
            continue
        if tag == 'text':
            raise ValueError('RDKit wrote <text> elements: set a FreeType font file (Depiction does)')
        if tag == 'ellipse':
            cx, cy = float(a['cx']), float(a['cy'])
            rx, ry = float(a['rx']), float(a['ry'])
            t = np.linspace(0, 2 * np.pi, 49)
            verts = np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]
            codes = np.full(len(verts), MPath.LINETO, np.uint8)
            codes[0] = MPath.MOVETO
        else:
            verts, codes = _path_from_d(a['d'])
        fill = a.get('fill', st.get('fill', 'none'))
        stroke = a.get('stroke', st.get('stroke', 'none'))
        lw = st.get('stroke-width', a.get('stroke-width', '0'))
        lw = float(lw.replace('px', '')) if lw else 0.0
        dash = st.get('stroke-dasharray')
        dash = None if dash in (None, 'none') else [float(x) for x in re.split(r'[ ,]+', dash.strip()) if x]
        if fill not in ('none', ''):
            items.append(dict(kind='fill', verts=verts, codes=codes, colour=fill, lw_px=0.0, dash=None,
                              classes=classes))
        if stroke not in ('none', '') and lw > 0 and fill in ('none', ''):
            items.append(dict(kind='stroke', verts=verts, codes=codes, colour=stroke, lw_px=lw, dash=dash,
                              classes=classes))
    return items


def _atoms_of(classes):
    return [int(c.split('-')[1]) for c in classes if c.startswith('atom-')]


def _bond_of(classes):
    for c in classes:
        if c.startswith('bond-'):
            return int(c.split('-')[1])
    return None


class Depiction:
    """One molecule drawn by RDKit at print scale.

    mol          RDKit Mol with 2D coordinates (conformer 0); drawn as given (no re-layout)
    bond_mm      printed length of a bond of the median length in the conformer
    font_pt      printed label size (points; em size of the font)
    line_pt      bond line width (points)
    bw           black-and-white atom palette (journal style); otherwise RDKit's default colours
    atom_labels  dict atom index -> label text overriding RDKit's (e.g. to hide a formal charge)
    kekulize     False draws aromatic bonds of a non-kekulized molecule as RDKit's dashed aromatic bonds
    options      extra MolDrawOptions attributes, e.g. dict(multipleBondOffset=0.18)
    """

    def __init__(self, mol, bond_mm=5.0, font_pt=7.0, line_pt=0.6, bw=True, atom_labels=None, kekulize=True,
                 font_file=None, options=None, px_per_mm=20.0):
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit import Chem
        self.mol = mol
        self.px_per_mm = float(px_per_mm)
        conf = mol.GetConformer()
        P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
        bl = [np.linalg.norm(P[b.GetBeginAtomIdx()] - P[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
        self.coord_bond = float(np.median(bl)) if bl else 1.5
        bond_px = bond_mm * self.px_per_mm
        span = (P.max(0) - P.min(0)) / self.coord_bond * bond_px if len(P) > 1 else np.array([bond_px, bond_px])
        pad = 6 * font_pt / PT_PER_MM * self.px_per_mm
        W, H = int(span[0] + 2 * pad), int(span[1] + 2 * pad)
        d = rdMolDraw2D.MolDraw2DSVG(W, H)
        o = d.drawOptions()
        if font_file is None:
            ff = style.FONT_DIR / 'LiberationSans-Regular.ttf'
            font_file = str(ff) if ff.exists() else None
        if font_file:
            o.fontFile = font_file
        o.fixedBondLength = bond_px / self.coord_bond        # RDKit: pixels per coordinate unit
        font_px = font_pt / PT_PER_MM * self.px_per_mm / _EM_PER_FONTPX
        o.fixedFontSize = int(round(font_px))
        o.minFontSize = o.fixedFontSize
        o.maxFontSize = o.fixedFontSize
        o.bondLineWidth = max(1, int(round(line_pt / PT_PER_MM * self.px_per_mm)))
        o.scaleBondWidth = False
        o.padding = 0.0
        o.clearBackground = True
        if bw:
            o.useBWAtomPalette()
        for k, v in (atom_labels or {}).items():
            o.atomLabels[int(k)] = v
        for k, v in (options or {}).items():
            setattr(o, k, v)
        m = mol
        if kekulize:
            m = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, addChiralHs=False)
        else:
            o.prepareMolsBeforeDrawing = False
        d.DrawMolecule(m)
        d.FinishDrawing()
        self.svg = d.GetDrawingText()
        self._coords = np.array([[d.GetDrawCoords(i).x, d.GetDrawCoords(i).y] for i in range(m.GetNumAtoms())])
        self.items = parse_svg(self.svg)
        allv = np.vstack([it['verts'] for it in self.items])
        pads = max(it['lw_px'] for it in self.items) / 2
        self.extent_px = (allv[:, 0].min() - pads, allv[:, 1].min() - pads, allv[:, 0].max() + pads,
                          allv[:, 1].max() + pads)
        # measured scale: RDKit may shrink the drawing to fit the canvas; check the bond length it used
        bl_px = [np.linalg.norm(self._coords[b.GetBeginAtomIdx()] - self._coords[b.GetEndAtomIdx()])
                 for b in m.GetBonds()]
        self.bond_px = float(np.median(bl_px)) if bl_px else bond_px
        if abs(self.bond_px - bond_px) > 0.02 * bond_px:
            raise RuntimeError(f'RDKit drew bonds {self.bond_px:.1f} px long instead of {bond_px:.1f} px '
                               '(canvas too small?)')

    def atom_px(self, i):
        return self._coords[i].copy()

    @property
    def size_mm(self):
        x0, y0, x1, y1 = self.extent_px
        return (x1 - x0) / self.px_per_mm, (y1 - y0) / self.px_per_mm

    def draw(self, ax, origin, recolour=None, zorder=3, anchor='extent', true_minus=True):
        """Draw into a millimetre axes. origin = data position (mm) of the lower-left corner of the extent
        (anchor='extent') or of atom i (anchor=('atom', i) puts atom i at origin). recolour: dict atom index ->
        colour for that atom's label glyphs and its half of each bond. true_minus: RDKit writes negative charges
        with the hyphen glyph; redraw it as a typographic minus (width of the plus sign, on the math axis)."""
        s = 1.0 / self.px_per_mm
        if anchor == 'extent':
            x0, y0, x1, y1 = self.extent_px
            ox, oy = origin[0] - x0 * s, origin[1] + y1 * s
        else:
            i = anchor[1]
            ox, oy = origin[0] - self._coords[i, 0] * s, origin[1] + self._coords[i, 1] * s
        return Placed(self, ax, (ox, oy), s, recolour or {}, zorder, true_minus=true_minus)

    def offset_mm(self, i):
        """Extent of the drawing relative to atom i, in mm with y up: (x0, y0, x1, y1)."""
        s = 1.0 / self.px_per_mm
        x0, y0, x1, y1 = self.extent_px
        cx, cy = self._coords[i]
        return ((x0 - cx) * s, (cy - y1) * s, (x1 - cx) * s, (cy - y0) * s)


# Liberation Sans glyph boxes (font units): hyphen (91, 464, 591, 624); minus sign (101, 608, 1096, 754)
_MINUS_W, _MINUS_DY, _MINUS_T = 995 / 500, (681 - 544) / 500, 146 / 160


def _hyphen_to_minus(V, codes):
    """If V (px, y down) is the rectangle of a hyphen glyph, return the rectangle of a minus sign."""
    if len(V) > 6 or any(c not in (MPath.MOVETO, MPath.LINETO, MPath.CLOSEPOLY) for c in codes):
        return None
    w, h = np.ptp(V[:, 0]), np.ptp(V[:, 1])
    if h <= 0 or w < 1.8 * h:
        return None
    cx, cy = (V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2
    W, T = w * _MINUS_W, h * _MINUS_T
    cy -= _MINUS_DY * w
    R = np.array([[cx - W / 2, cy - T / 2], [cx + W / 2, cy - T / 2], [cx + W / 2, cy + T / 2],
                  [cx - W / 2, cy + T / 2], [cx - W / 2, cy - T / 2]])
    return R, np.array([MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY], np.uint8)


@dataclass
class _Tr:
    ox: float
    oy: float
    s: float

    def __call__(self, P):
        P = np.asarray(P, float)
        return np.c_[self.ox + P[..., 0] * self.s, self.oy - P[..., 1] * self.s] if P.ndim > 1 else \
            np.array([self.ox + P[0] * self.s, self.oy - P[1] * self.s])


class Placed:
    def __init__(self, dep, ax, o, s, recolour, zorder, true_minus=True):
        self.dep, self.ax = dep, ax
        neg = {a.GetIdx() for a in dep.mol.GetAtoms() if a.GetFormalCharge() < 0} if true_minus else set()
        self.tr = _Tr(o[0], o[1], s)
        self.artists = []
        self._boxes = {}
        self._segs = []
        lw_scale = s * PT_PER_MM
        for it in dep.items:
            atoms = _atoms_of(it['classes'])
            verts, codes = it['verts'], it['codes']
            if it['kind'] == 'fill' and len(atoms) == 1 and atoms[0] in neg and _bond_of(it['classes']) is None:
                mm = _hyphen_to_minus(verts, codes)
                if mm is not None:
                    verts, codes = mm
            V = self.tr(verts)
            col = it['colour']
            bond = _bond_of(it['classes'])
            if recolour:
                if bond is None and len(atoms) == 1 and atoms[0] in recolour:
                    col = recolour[atoms[0]]
                elif bond is not None and len(atoms) == 2:
                    # RDKit splits a bond into two halves drawn in the two atoms' colours: recolour the half
                    # nearer the recoloured atom
                    mid = V.mean(0)
                    near = min(atoms, key=lambda a: np.linalg.norm(self.xy(a) - mid))
                    if near in recolour:
                        col = recolour[near]
            path = MPath(V, codes)
            if it['kind'] == 'fill':
                p = PathPatch(path, facecolor=col, edgecolor='none', lw=0, zorder=zorder + 0.1)
                if bond is None and len(atoms) == 1:
                    b = self._boxes.get(atoms[0])
                    bb = (V[:, 0].min(), V[:, 1].min(), V[:, 0].max(), V[:, 1].max())
                    self._boxes[atoms[0]] = bb if b is None else (min(b[0], bb[0]), min(b[1], bb[1]),
                                                                   max(b[2], bb[2]), max(b[3], bb[3]))
            else:
                p = PathPatch(path, facecolor='none', edgecolor=col, lw=it['lw_px'] * lw_scale, zorder=zorder,
                              capstyle='butt', joinstyle='miter')
                if it['dash']:
                    p.set_linestyle((0, [x * lw_scale / max(it['lw_px'] * lw_scale, 1e-9) for x in it['dash']]))
                if len(V) >= 2:
                    for k in range(len(V) - 1):
                        self._segs.append((V[k], V[k + 1], it['lw_px'] * s / 2))
            ax.add_patch(p)
            self.artists.append(p)
        x0, y0, x1, y1 = dep.extent_px
        a = self.tr(np.array([[x0, y1], [x1, y0]]))
        self.bbox = (a[0, 0], a[0, 1], a[1, 0], a[1, 1])

    def xy(self, i):
        return self.tr(self.dep._coords[i])

    def atom_box(self, i):
        return self._boxes.get(i)

    def glyph_boxes(self):
        return list(self._boxes.values())

    def segments(self):
        return list(self._segs)
