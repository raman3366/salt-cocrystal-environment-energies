"""Geometry helpers shared by the 2D and 3D figure code.

Stable public API (other modules and agents import these names; do not rename):

    read_xyz(path)                      -> Frame                 first frame of an XYZ file
    read_xyz_frames(path)               -> list[Frame]           every frame (opt_traj.xyz), comments kept
    frame_energy(comment)               -> float | None          'E = -1681.2553 Ha' -> -1681.2553 (Hartree)
    write_xyz(path, symbols, xyz, comment='')
    find_case(case)                     -> CasePaths             where a case's JSON / tracked / raw files live
    read_case(case_or_path)             -> dict                  case JSON + 'labels' (clean), 'raw_labels', 'paths'
    clean_label(raw)                    -> str                   'C9#1_545' -> 'C9'
    clean_labels(raw_labels)            -> list[str]             strips suffixes, primes duplicates (C9, C9')
    detect_bonds(symbols, xyz, factor=1.2)            -> list[(i, j)]
    detect_hbonds(symbols, xyz, bonds=None, ...)      -> list[HBond]
    find_rings(n_atoms, bonds, sizes=(5, 6))          -> list[list[int]]
    principal_orientation(xyz, weights=None)          -> (xyz_rotated, R, centre)
    best_view(symbols, xyz, n_dirs=800)               -> (xyz_rotated, R, centre)   least atom overlap in projection
    apply_orientation(xyz, R, centre)                 -> xyz_rotated
    plane_orientation(xyz, idx)                       -> (xyz_rotated, R, centre)   view down a mean-plane normal
    kabsch(mobile, ref, weights=None)                 -> (R, t, rmsd)   with  aligned = mobile @ R.T + t
    align(mobile, ref, idx=None, weights=None)        -> (aligned_mobile, rmsd)
    distance(xyz, i, j), angle(xyz, i, j, k), dihedral(xyz, i, j, k, l)
    bond_label(labels, i, j, sep='–'), hbond_label(labels, d, h, a)

Coordinates are always numpy arrays of shape (N, 3) in Angstrom; atom indices are 0-based and follow the CIF
order used everywhere in the project (``cif_labels`` in the case JSON).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import style

BOHR = 0.529177210903          # Angstrom per Bohr
HARTREE_EV = 27.211386245988
HARTREE_KCAL = 627.509474

PRIMES = ('′', '″', '‴')     # ′ ″ ‴

# ----------------------------------------------------------------------------------------------------------------
# XYZ files
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Frame:
    """One XYZ frame: element symbols, coordinates (Å) and the untouched comment line."""
    symbols: list
    xyz: np.ndarray
    comment: str = ''

    @property
    def energy(self):
        """Energy in Hartree parsed from the comment line, or None."""
        return frame_energy(self.comment)

    def __len__(self):
        return len(self.symbols)


def _norm_symbol(s):
    s = re.sub(r'[^A-Za-z]', '', s)
    return s[:1].upper() + s[1:].lower() if s else s


def read_xyz_frames(path):
    """Read every frame of a (multi-frame) XYZ file. Comment lines are kept verbatim in Frame.comment."""
    lines = Path(path).read_text().splitlines()
    frames, k = [], 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        comment = lines[k + 1] if k + 1 < len(lines) else ''
        sym, xyz = [], []
        for ln in lines[k + 2:k + 2 + n]:
            t = ln.split()
            sym.append(_norm_symbol(t[0]))
            xyz.append([float(v) for v in t[1:4]])
        if len(sym) != n:
            raise ValueError(f'{path}: truncated frame at line {k + 1}')
        frames.append(Frame(sym, np.array(xyz, float), comment))
        k += 2 + n
    return frames


def read_xyz(path):
    """Read the first frame of an XYZ file."""
    return read_xyz_frames(path)[0]


_E_PATTERNS = [
    re.compile(r'\bE\s*=\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*(Ha|Eh|hartree|au|a\.u\.)?', re.I),
    re.compile(r'\benergy\s*[:=]\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)', re.I),
    re.compile(r'^\s*(-?\d+\.\d{5,})\s*$'),
]


def frame_energy(comment):
    """Parse an energy (Hartree) from an XYZ comment line such as 'step 3  E = -1681.25536693 Ha'."""
    for p in _E_PATTERNS:
        m = p.search(comment or '')
        if m:
            return float(m.group(1))
    return None


def write_xyz(path, symbols, xyz, comment=''):
    with open(path, 'w') as f:
        f.write(f'{len(symbols)}\n{comment}\n')
        for s, (x, y, z) in zip(symbols, np.asarray(xyz)):
            f.write(f'{s:<2s} {x:14.8f} {y:14.8f} {z:14.8f}\n')


# ----------------------------------------------------------------------------------------------------------------
# Cases and labels
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CasePaths:
    """Where the files of one case live. Any attribute may be None when the file does not exist."""
    case: str
    mol_id: str
    json: Path | None               # molecules/<id>/cases/<case>.json (or the closest parent case)
    tracked: Path | None            # molecules/<id>/results/<case>/
    raw: Path | None                # data/r2/molecules/<id>/results/<case>/
    xray_xyz: Path | None           # starting (X-ray derived) cluster, from the case JSON 'xyz'
    opt_xyz: Path | None            # optimised geometry
    traj: Path | None               # opt_traj.xyz (raw snapshot)
    extra: dict = field(default_factory=dict)

    def file(self, name):
        """Return the first existing of raw/<name>, tracked/<name>, else None."""
        for base in (self.raw, self.tracked):
            if base is not None and (base / name).exists():
                return base / name
        return None


def find_case(case):
    """Locate a case by name (e.g. 'btp_pcm', 'pzc_gas_saddle_1731').

    Cases without their own JSON (such as the PZC saddle point) fall back to the JSON whose name is the longest
    prefix of the case name ('pzc_gas'); CasePaths.extra['json_fallback'] records that.
    """
    repo = style.REPO
    mol_dirs = sorted((repo / 'molecules').glob('*'))
    mol_id, jpath, fallback = None, None, False
    for m in mol_dirs:
        p = m / 'cases' / f'{case}.json'
        if p.exists():
            mol_id, jpath = m.name, p
            break
    if mol_id is None:
        for m in mol_dirs:
            if (m / 'results' / case).exists() or (style.DATA / 'molecules' / m.name / 'results' / case).exists():
                mol_id = m.name
                cands = sorted((m / 'cases').glob('*.json'), key=lambda q: -len(q.stem))
                for q in cands:
                    if case.startswith(q.stem):
                        jpath, fallback = q, True
                        break
                break
    if mol_id is None:
        raise FileNotFoundError(f'case {case!r} not found under molecules/*/cases or */results')
    tracked = repo / 'molecules' / mol_id / 'results' / case
    raw = style.DATA / 'molecules' / mol_id / 'results' / case
    tracked = tracked if tracked.exists() else None
    raw = raw if raw.exists() else None
    xray = None
    if jpath is not None:
        x = json.loads(jpath.read_text()).get('xyz')
        if x and (repo / x).exists():
            xray = repo / x
    cp = CasePaths(case, mol_id, jpath, tracked, raw, xray, None, None, {'json_fallback': fallback})
    cp.opt_xyz = cp.file('opt.xyz')
    cp.traj = cp.file('opt_traj.xyz')
    return cp


def clean_label(raw):
    """Strip the internal symmetry suffix: 'C9#1_545' -> 'C9'."""
    return str(raw).split('#', 1)[0].strip()


def clean_labels(raw_labels):
    """Strip suffixes; if two labels in the list become identical, prime the symmetry-generated one(s).

    The copy without a '#' suffix (or the first one) keeps the plain label; later symmetry copies get ′, ″, ‴.
    """
    raw_labels = [str(r) for r in raw_labels]
    base = [clean_label(r) for r in raw_labels]
    out = list(base)
    groups = {}
    for i, b in enumerate(base):
        groups.setdefault(b, []).append(i)
    for b, idx in groups.items():
        if len(idx) < 2:
            continue
        # plain (asymmetric-unit) atom first, then symmetry copies in order of appearance
        idx = sorted(idx, key=lambda i: ('#' in raw_labels[i], i))
        for n, i in enumerate(idx[1:]):
            out[i] = b + (PRIMES[n] if n < len(PRIMES) else PRIMES[0] * (n + 1))
    return out


def read_case(case_or_path):
    """Load a case JSON. Adds 'labels' (clean, unique), 'raw_labels', 'paths' (CasePaths) and 'name'.

    Also converts 'hbonds' to a list of dicts {'d','h','a','label'} under 'hbond_list' with clean labels.
    """
    p = Path(str(case_or_path))
    if p.suffix == '.json' and p.exists():
        d = json.loads(p.read_text())
        paths = None
        name = p.stem
        try:
            paths = find_case(p.stem)
        except FileNotFoundError:
            pass
    else:
        paths = find_case(str(case_or_path))
        if paths.json is None:
            raise FileNotFoundError(f'no case JSON for {case_or_path}')
        d = json.loads(paths.json.read_text())
        name = str(case_or_path)
    raw = d.get('cif_labels') or []
    d['raw_labels'] = raw
    d['labels'] = clean_labels(raw)
    d['paths'] = paths
    d['name'] = name
    hl = []
    for key, trip in (d.get('hbonds') or {}).items():
        if len(trip) == 3:
            dd, hh, aa = trip
            hl.append({'d': dd, 'h': hh, 'a': aa, 'label': hbond_label(d['labels'], dd, hh, aa)})
    d['hbond_list'] = hl
    return d


def bond_label(labels, i, j, sep='–'):
    """'C1–C2' with an en dash."""
    return f'{labels[i]}{sep}{labels[j]}'


def hbond_label(labels, d, h, a):
    """'N4–H4···O11' (en dash, three middle dots)."""
    return f'{labels[d]}–{labels[h]}···{labels[a]}'


# ----------------------------------------------------------------------------------------------------------------
# Connectivity
# ----------------------------------------------------------------------------------------------------------------


def _radius(s):
    return style.COVALENT_RADII.get(s, 0.8)


def detect_bonds(symbols, xyz, factor=1.2):
    """Covalent bonds: d(i, j) < factor * (r_i + r_j) with Cordero radii; H–H pairs are never bonded."""
    xyz = np.asarray(xyz, float)
    r = np.array([_radius(s) for s in symbols])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    lim = factor * (r[:, None] + r[None])
    isH = np.array([s == 'H' for s in symbols])
    ok = (d < lim) & (d > 0.1) & ~(isH[:, None] & isH[None])
    i, j = np.nonzero(np.triu(ok, 1))
    return [(int(a), int(b)) for a, b in zip(i, j)]


@dataclass
class HBond:
    d: int          # donor heavy atom
    h: int          # hydrogen
    a: int          # acceptor
    r_ha: float     # H···A (Å)
    r_da: float     # D···A (Å)
    angle: float    # D–H···A (degrees)

    def label(self, labels):
        return hbond_label(labels, self.d, self.h, self.a)


def detect_hbonds(symbols, xyz, bonds=None, donors=('N', 'O', 'Cl'), acceptors=('N', 'O', 'Cl'),
                  max_ha=2.6, max_ha_cl=3.0, min_angle=120.0):
    """D–H···A hydrogen bonds with D, A in N/O/Cl; H···A < 2.6 Å (< 3.0 Å if A is Cl); angle D–H···A > 120°.

    The donor is the heavy atom the H is covalently bonded to (from `bonds`, detected if not given). The acceptor
    must not be bonded to that H. Returns HBond records sorted by H···A distance.
    """
    xyz = np.asarray(xyz, float)
    bonds = detect_bonds(symbols, xyz) if bonds is None else bonds
    nb = {i: set() for i in range(len(symbols))}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    out = []
    for h, s in enumerate(symbols):
        if s != 'H':
            continue
        ds = [k for k in nb[h] if symbols[k] in donors]
        if not ds:
            continue
        d = min(ds, key=lambda k: np.linalg.norm(xyz[k] - xyz[h]))
        for a, sa in enumerate(symbols):
            if sa not in acceptors or a == d or a in nb[h]:
                continue
            rha = float(np.linalg.norm(xyz[a] - xyz[h]))
            if rha >= (max_ha_cl if sa == 'Cl' else max_ha):
                continue
            ang = angle(xyz, d, h, a)
            if ang <= min_angle:
                continue
            out.append(HBond(d, h, a, rha, float(np.linalg.norm(xyz[a] - xyz[d])), ang))
    return sorted(out, key=lambda b: b.r_ha)


def find_rings(n_atoms, bonds, sizes=(5, 6)):
    """Smallest rings of the given sizes (simple cycle search; fine for molecular clusters)."""
    nb = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    rings = set()
    maxn = max(sizes)

    def dfs(start, cur, path):
        if len(path) > maxn:
            return
        for n in nb[cur]:
            if n == start and len(path) in sizes:
                rings.add(tuple(sorted(path)))
            elif n not in path and n > start:
                dfs(start, n, path + [n])

    for s in range(n_atoms):
        dfs(s, s, [s])
    out = []
    for r in sorted(rings, key=len):
        # drop envelopes of fused rings (a ring that contains a smaller ring's atoms plus a bridge)
        if not any(set(o) < set(r) for o in out):
            out.append(list(r))
    return out


def ring_order(ring, bonds):
    """Return the atoms of a ring in cyclic order."""
    rs = set(ring)
    nb = {i: [] for i in ring}
    for i, j in bonds:
        if i in rs and j in rs:
            nb[i].append(j)
            nb[j].append(i)
    order = [ring[0]]
    prev = None
    while len(order) < len(ring):
        cur = order[-1]
        nxt = [n for n in nb[cur] if n != prev and n not in order]
        if not nxt:
            break
        prev = cur
        order.append(nxt[0])
    return order


# ----------------------------------------------------------------------------------------------------------------
# Orientation and alignment
# ----------------------------------------------------------------------------------------------------------------


def _fix_signs(R, X):
    """Deterministic axis signs: positive third moment along x and y, then det(R) = +1 (no mirror images)."""
    Y = X @ R.T
    for k in range(2):
        m3 = np.sum(Y[:, k] ** 3)
        if m3 < 0:
            R[k] *= -1
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R


def principal_orientation(xyz, weights=None):
    """Rotate so the largest spread lies along x, the next along y, and the view (z, towards the viewer) is along
    the smallest one. Returns (xyz_rotated, R, centre) with xyz_rotated = (xyz - centre) @ R.T; det(R) = +1."""
    xyz = np.asarray(xyz, float)
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, float)
    c = (w[:, None] * xyz).sum(0) / w.sum()
    X = xyz - c
    C = (w[:, None] * X).T @ X / w.sum()
    evals, evecs = np.linalg.eigh(C)
    R = evecs[:, ::-1].T.copy()          # rows: axes by decreasing variance
    R = _fix_signs(R, X)
    return X @ R.T, R, c


def best_view(symbols, xyz, n_dirs=800, h_weight=0.35, spread_weight=0.02, radii=None, bonds=None,
              hbonds='auto', bond_radius=0.085, occlusion_weight=1.0, crossing_weight=0.05):
    """Viewing rotation that keeps atoms and bonds from hiding one another in projection.

    Each direction of a Fibonacci hemisphere is scored by
      * the squared projected overlap of every pair of ball-and-stick discs (pairs with a hydrogen weighted by
        h_weight, except hydrogens that take part in a hydrogen bond, which count fully),
      * the squared overlap of bonds passing over atoms they do not belong to (occlusion_weight),
      * the number of crossing bond pairs (crossing_weight each),
      * minus a small reward for projected spread.
    The best direction becomes the view axis (z towards the viewer) and the projected long axis is laid
    horizontally. Deterministic. Returns (xyz_rotated, R, centre) like principal_orientation."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    c = xyz.mean(0)
    X = xyz - c
    r = np.array([style.BALL_RADII.get(s, 0.35) for s in symbols]) if radii is None else np.asarray(radii)
    bonds = detect_bonds(symbols, xyz) if bonds is None else bonds
    hb = detect_hbonds(symbols, xyz, bonds) if hbonds == 'auto' else (hbonds or [])
    hb_h = {h.h if isinstance(h, HBond) else h[1] for h in hb}
    light = np.array([s == 'H' and i not in hb_h for i, s in enumerate(symbols)])
    iu, ju = np.triu_indices(n, 1)
    rsum = (r[iu] + r[ju]) * 1.15
    w = np.where(light[iu] | light[ju], h_weight, 1.0)
    w = np.where(light[iu] & light[ju], h_weight ** 2, w)
    B = np.array(bonds, int).reshape(-1, 2)
    nb = len(B)
    # atoms not belonging to each bond
    notmine = np.ones((nb, n), bool)
    if nb:
        notmine[np.arange(nb), B[:, 0]] = False
        notmine[np.arange(nb), B[:, 1]] = False
    occ_lim = (r[None, :] + bond_radius) * 1.05
    occ_w = np.where(light, h_weight, 1.0)[None, :]
    share = np.zeros((nb, nb), bool)
    if nb:
        share = (B[:, None, 0] == B[None, :, 0]) | (B[:, None, 0] == B[None, :, 1]) | \
                (B[:, None, 1] == B[None, :, 0]) | (B[:, None, 1] == B[None, :, 1])
    k = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - k / n_dirs)                 # hemisphere: z >= 0
    th = np.pi * (1 + 5 ** 0.5) * k
    V = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    best, best_v = None, None
    for v in V:
        a = np.eye(3)[np.argmin(np.abs(v))]
        e1 = np.cross(v, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        P = X @ np.c_[e1, e2]
        d = np.linalg.norm(P[iu] - P[ju], axis=1)
        pen = float((w * np.maximum(rsum - d, 0) ** 2).sum())
        if nb:
            A0, A1 = P[B[:, 0]], P[B[:, 1]]
            D = A1 - A0
            L2 = np.maximum((D ** 2).sum(1), 1e-12)
            t = np.clip(((P[None, :, :] - A0[:, None, :]) * D[:, None, :]).sum(-1) / L2[:, None], 0, 1)
            Q = A0[:, None, :] + t[..., None] * D[:, None, :]
            ds = np.linalg.norm(P[None, :, :] - Q, axis=-1)
            ov = np.where(notmine, np.maximum(occ_lim - ds, 0), 0.0)
            pen += occlusion_weight * float((occ_w * ov ** 2).sum())
            if crossing_weight:
                def orient(p, q, rr):
                    return np.sign((q[..., 0] - p[..., 0]) * (rr[..., 1] - p[..., 1]) -
                                   (q[..., 1] - p[..., 1]) * (rr[..., 0] - p[..., 0]))
                a0, a1 = A0[:, None], A1[:, None]
                b0, b1 = A0[None], A1[None]
                cross = (orient(a0, a1, b0) * orient(a0, a1, b1) < 0) & (orient(b0, b1, a0) * orient(b0, b1, a1) < 0)
                pen += crossing_weight * float(np.triu(cross & ~share, 1).sum())
        spread = float(np.sqrt((P ** 2).sum(1).mean()))
        score = pen - spread_weight * spread
        if best is None or score < best - 1e-12:
            best, best_v = score, v
    v = best_v / np.linalg.norm(best_v)
    Xp = X - np.outer(X @ v, v)
    evals, evecs = np.linalg.eigh(Xp.T @ Xp)
    e1 = evecs[:, -1]
    e2 = np.cross(v, e1)
    R = np.array([e1, e2, v])
    R = _fix_signs(R, X)
    return X @ R.T, R, c


def apply_orientation(xyz, R, centre):
    """Apply a rotation returned by principal_orientation / plane_orientation to another geometry."""
    return (np.asarray(xyz, float) - centre) @ np.asarray(R).T


def plane_orientation(xyz, idx, weights=None):
    """View along the normal of the least-squares plane through atoms `idx`; x = longest in-plane direction of the
    whole structure. Useful for flat aromatic clusters. Returns (xyz_rotated, R, centre)."""
    xyz = np.asarray(xyz, float)
    P = xyz[list(idx)]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    X = xyz - xyz.mean(0)
    Xp = X - np.outer(X @ n, n)
    C = Xp.T @ Xp
    evals, evecs = np.linalg.eigh(C)
    e1 = evecs[:, -1]
    e2 = np.cross(n, e1)
    R = np.array([e1, e2, n])
    R = _fix_signs(R, X)
    centre = xyz.mean(0)
    return (xyz - centre) @ R.T, R, centre


def kabsch(mobile, ref, weights=None):
    """Optimal rotation R and translation t so that mobile @ R.T + t best matches ref (same atom order).

    Returns (R, t, rmsd) with the rmsd evaluated over the same (weighted) atoms."""
    P = np.asarray(mobile, float)
    Q = np.asarray(ref, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    pc = (w[:, None] * P).sum(0)
    qc = (w[:, None] * Q).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, dd])
    R = Vt.T @ D @ U.T
    t = qc - pc @ R.T
    diff = P @ R.T + t - Q
    rmsd = float(np.sqrt((w * (diff ** 2).sum(1)).sum()))
    return R, t, rmsd


def align(mobile, ref, idx=None, weights=None):
    """Kabsch-align `mobile` onto `ref` using atoms `idx` (all if None). Returns (aligned_mobile, rmsd_over_idx)."""
    mobile = np.asarray(mobile, float)
    ref = np.asarray(ref, float)
    sel = np.arange(len(ref)) if idx is None else np.asarray(list(idx))
    w = None if weights is None else np.asarray(weights, float)[sel]
    R, t, rmsd = kabsch(mobile[sel], ref[sel], w)
    return mobile @ R.T + t, rmsd


def heavy_atoms(symbols):
    return [i for i, s in enumerate(symbols) if s != 'H']


# ----------------------------------------------------------------------------------------------------------------
# Internal coordinates
# ----------------------------------------------------------------------------------------------------------------


def distance(xyz, i, j):
    return float(np.linalg.norm(np.asarray(xyz)[i] - np.asarray(xyz)[j]))


def angle(xyz, i, j, k):
    """Angle i–j–k in degrees."""
    xyz = np.asarray(xyz, float)
    a, b = xyz[i] - xyz[j], xyz[k] - xyz[j]
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def dihedral(xyz, i, j, k, l):
    xyz = np.asarray(xyz, float)
    b0, b1, b2 = xyz[i] - xyz[j], xyz[k] - xyz[j], xyz[l] - xyz[k]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))
