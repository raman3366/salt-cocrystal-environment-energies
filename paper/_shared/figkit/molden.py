"""Molecular orbitals on arbitrary grids from a Molden file (numpy GTO evaluator, no PySCF needed).

Why: the orbital cubes written during the production runs use a 4.5 Bohr margin, which clips diffuse virtual
orbitals (LUMOs of ionic clusters with ++ basis sets).  This module re-evaluates any MO from final.molden on a
box of your choice and writes a Gaussian cube that figkit.render3d.read_cube reads.

    from figkit import molden
    mf   = molden.read_molden('final.molden')
    grid = molden.box_grid(mf.coords, margin=8.0, spacing=0.25)          # Bohr
    vals = mf.mo_on_grid([mf.homo, mf.lumo], grid)                        # (2, nx, ny, nz)
    molden.write_cube('lumo.cube', mf, grid, vals[1], title='LUMO')

Conventions (PySCF molden writer): [Atoms] in Bohr (AU) or Angstrom; spherical functions ([5d] [7f] [9g]) in
Molden order (d0, d+1, d-1, d+2, d-2; f0, f+1, f-1, ..., f+3, f-3; p = x, y, z); contraction coefficients
multiply NORMALISED primitives; each AO is R(r) * Y_lm with real spherical harmonics normalised on the unit
sphere (the libcint convention).  Cartesian d/f (no [5d]/[7f] flag) are supported for l <= 3 with Molden's
cartesian ordering and per-component normalisation.  Verified on PZC (420 functions, l <= 3) against the
PySCF cubes to < 1e-6 a.u.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ['MoldenFile', 'read_molden', 'box_grid', 'Grid', 'write_cube']

_L = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
_Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
      'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Zn': 30, 'Br': 35}
BOHR = 0.529177210903


@dataclass
class Shell:
    atom: int
    l: int
    exps: np.ndarray
    coefs: np.ndarray            # for normalised primitives


@dataclass
class MoldenFile:
    symbols: list
    numbers: np.ndarray
    coords: np.ndarray           # Bohr
    shells: list
    spherical: dict              # l -> bool
    mo_energy: np.ndarray        # Hartree
    mo_occ: np.ndarray
    mo_coeff: np.ndarray         # (nao, nmo), Molden AO order
    path: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def nao(self):
        return self.mo_coeff.shape[0]

    @property
    def homo(self):
        return int(np.flatnonzero(self.mo_occ > 0.5).max())

    @property
    def lumo(self):
        return self.homo + 1

    def ao_on_points(self, P, cutoff=1e-10):
        """(npts, nao) AO values at points P (Bohr)."""
        P = np.asarray(P, float)
        out = np.zeros((len(P), self.nao))
        col = 0
        for sh in self.shells:
            nf = _nfunc(sh.l, self.spherical.get(sh.l, False))
            d = P - self.coords[sh.atom]
            r2 = np.einsum('ij,ij->i', d, d)
            amin = sh.exps.min()
            # skip points where even the most diffuse primitive is negligible
            keep = r2 < (-math.log(cutoff) + 10.0) / amin
            if keep.any():
                dk, r2k = d[keep], r2[keep]
                rad = np.zeros(len(r2k))
                for a, c in zip(sh.exps, sh.coefs):
                    rad += c * _gto_norm(sh.l, a) * np.exp(-a * r2k)
                ang = _angular(sh.l, dk, self.spherical.get(sh.l, False))
                out[keep, col:col + nf] = rad[:, None] * ang
            col += nf
        if col != self.nao:
            raise ValueError(f'basis gives {col} functions, MO coefficients have {self.nao}')
        return out

    def mo_on_points(self, mos, P, chunk=20000):
        mos = np.atleast_1d(mos)
        C = self.mo_coeff[:, mos]
        out = np.empty((len(mos), len(P)))
        for i in range(0, len(P), chunk):
            out[:, i:i + chunk] = (self.ao_on_points(P[i:i + chunk]) @ C).T
        return out

    def mo_on_grid(self, mos, grid, chunk=20000):
        """(len(mos), nx, ny, nz) MO values on a Grid (a.u.)."""
        vals = self.mo_on_points(mos, grid.points(), chunk)
        return vals.reshape((len(np.atleast_1d(mos)),) + grid.shape)


def _nfunc(l, spherical):
    return 2 * l + 1 if spherical else (l + 1) * (l + 2) // 2


def _gto_norm(l, a):
    # 1 / sqrt( int_0^inf r^(2l+2) exp(-2 a r^2) dr )
    return 1.0 / math.sqrt(math.gamma(l + 1.5) / (2.0 * (2.0 * a) ** (l + 1.5)))


_PI = math.pi


def _angular(l, d, spherical):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    if l == 0:
        return np.full((len(d), 1), 1.0 / math.sqrt(4 * _PI))
    if l == 1:
        return math.sqrt(3 / (4 * _PI)) * d
    if spherical:
        if l == 2:          # Molden order d0, d+1, d-1, d+2, d-2
            return np.c_[math.sqrt(5 / (16 * _PI)) * (2 * z * z - x * x - y * y),
                         math.sqrt(15 / (4 * _PI)) * x * z,
                         math.sqrt(15 / (4 * _PI)) * y * z,
                         math.sqrt(15 / (16 * _PI)) * (x * x - y * y),
                         math.sqrt(15 / (4 * _PI)) * x * y]
        if l == 3:          # f0, f+1, f-1, f+2, f-2, f+3, f-3
            r2xy = x * x + y * y
            return np.c_[math.sqrt(7 / (16 * _PI)) * z * (2 * z * z - 3 * r2xy),
                         math.sqrt(21 / (32 * _PI)) * x * (4 * z * z - r2xy),
                         math.sqrt(21 / (32 * _PI)) * y * (4 * z * z - r2xy),
                         math.sqrt(105 / (16 * _PI)) * z * (x * x - y * y),
                         math.sqrt(105 / (4 * _PI)) * x * y * z,
                         math.sqrt(35 / (32 * _PI)) * x * (x * x - 3 * y * y),
                         math.sqrt(35 / (32 * _PI)) * y * (3 * x * x - y * y)]
        raise NotImplementedError(f'spherical l={l}')
    # cartesian, Molden order; each component normalised on its own (double factorials)
    if l == 2:
        comps = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    elif l == 3:
        comps = [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2),
                 (0, 2, 1), (1, 1, 1)]
    else:
        raise NotImplementedError(f'cartesian l={l}')

    def dfact(n):
        return 1 if n <= 0 else n * dfact(n - 2)
    cols = []
    for i, j, k in comps:
        # R(r) is normalised for r^l; the angular factor makes x^i y^j z^k e^{-ar^2} normalised
        f = math.sqrt(dfact(2 * l + 1) / (dfact(2 * i - 1) * dfact(2 * j - 1) * dfact(2 * k - 1)) / (4 * _PI))
        cols.append(f * x ** i * y ** j * z ** k)
    return np.column_stack(cols)


def read_molden(path):
    """Parse [Atoms], [GTO], [5d]/[7f]/[9g] and [MO] (restricted, one spin) of a Molden file."""
    path = Path(path)
    lines = path.read_text().splitlines()
    sec, symbols, numbers, coords, shells = None, [], [], [], []
    spherical = {2: False, 3: False, 4: False}
    unit = BOHR
    mo_e, mo_o, mo_c = [], [], []
    i = 0
    cur_atom = None
    cur_mo = None
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith('['):
            tag = ln.split(']')[0].lower() + ']'
            if tag == '[atoms]':
                sec = 'atoms'
                unit = 1.0 if '(au)' in ln.lower() else 1.0 / BOHR
            elif tag == '[gto]':
                sec = 'gto'
            elif tag == '[mo]':
                sec = 'mo'
            elif tag in ('[5d]', '[5d7f]', '[5d10f]', '[7f]', '[9g]'):
                if '5d' in tag:
                    spherical[2] = True
                if '7f' in tag:
                    spherical[3] = True
                if '9g' in tag:
                    spherical[4] = True
                sec = None
            else:
                sec = None
            i += 1
            continue
        if not ln:
            i += 1
            continue
        tok = ln.split()
        if sec == 'atoms':
            symbols.append(tok[0].capitalize() if len(tok[0]) > 1 else tok[0])
            numbers.append(int(tok[2]))
            coords.append([float(v) * unit for v in tok[3:6]])
        elif sec == 'gto':
            if len(tok) == 2 and tok[0].isdigit():
                cur_atom = int(tok[0]) - 1
            elif tok[0].lower() in _L:
                l, n = _L[tok[0].lower()], int(tok[1])
                ex, co = [], []
                for k in range(n):
                    a, c = lines[i + 1 + k].replace('D', 'E').split()[:2]
                    ex.append(float(a))
                    co.append(float(c))
                shells.append(Shell(cur_atom, l, np.array(ex), np.array(co)))
                i += n
        elif sec == 'mo':
            if '=' in ln:
                key, val = [s.strip() for s in ln.split('=', 1)]
                if key.lower() == 'ene':
                    cur_mo = []
                    mo_c.append(cur_mo)
                    mo_e.append(float(val))
                elif key.lower() == 'occup':
                    mo_o.append(float(val))
            else:
                cur_mo.append(float(tok[1]))
        i += 1
    nao = len(mo_c[0])
    C = np.array(mo_c).T
    if C.shape[0] != nao:
        raise ValueError('ragged MO block')
    return MoldenFile(symbols=symbols, numbers=np.array(numbers), coords=np.array(coords), shells=shells,
                      spherical=spherical, mo_energy=np.array(mo_e), mo_occ=np.array(mo_o), mo_coeff=C,
                      path=str(path))


@dataclass
class Grid:
    origin: np.ndarray           # Bohr
    axes: np.ndarray             # (3, 3) Bohr, row = step vector
    shape: tuple

    def points(self):
        n1, n2, n3 = self.shape
        i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing='ij')
        return (self.origin + i.reshape(-1, 1) * self.axes[0] + j.reshape(-1, 1) * self.axes[1]
                + k.reshape(-1, 1) * self.axes[2])


def box_grid(coords_bohr, margin=8.0, spacing=0.25):
    """Axis-aligned grid enclosing the atoms with `margin` Bohr on every side, step <= `spacing` Bohr."""
    X = np.asarray(coords_bohr, float)
    lo, hi = X.min(0) - margin, X.max(0) + margin
    n = np.ceil((hi - lo) / spacing).astype(int) + 1
    step = (hi - lo) / (n - 1)
    return Grid(origin=lo, axes=np.diag(step), shape=tuple(int(v) for v in n))


def write_cube(path, mf, grid, values, title='MO values', comment='figkit.molden'):
    """Gaussian cube in Bohr (read back by figkit.render3d.read_cube)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, float).reshape(grid.shape)
    with open(path, 'w') as f:
        f.write(f'{title}\n{comment}\n')
        f.write(f'{len(mf.numbers):5d} {grid.origin[0]:12.6f} {grid.origin[1]:12.6f} {grid.origin[2]:12.6f}\n')
        for n, ax in zip(grid.shape, grid.axes):
            f.write(f'{n:5d} {ax[0]:12.6f} {ax[1]:12.6f} {ax[2]:12.6f}\n')
        for z, c in zip(mf.numbers, mf.coords):
            f.write(f'{int(z):5d} {0.0:12.6f} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n')
        flat = v.reshape(-1, v.shape[2])
        for row in flat:
            for s in range(0, len(row), 6):
                f.write(' '.join(f'{x:13.5E}' for x in row[s:s + 6]) + '\n')
    return path
