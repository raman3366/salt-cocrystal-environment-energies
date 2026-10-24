"""Headless 3D renders for the manuscripts: orbitals, MEP surfaces, NCI isosurfaces, ball-and-stick models.

PyVista/VTK off-screen through EGL (llvmpipe on the Raspberry Pi 5). This module only produces *images*;
all typography (panel letters, colour bars, atom labels) is added in the matplotlib layer by figkit.compose,
from what each Render carries (projection matrix, z-buffer, colour scale, markers).

Public API
    read_cube(path) -> Cube            Gaussian cube, Bohr -> Angstrom, general (skewed) axes, MO-cube header,
                                       negative voxel counts (Angstrom files); Cube.interpolate(points, order),
                                       Cube.isosurface(level, refine), Cube.grid(), Cube.refined(factor)
    load_molecule(case, xyz=None) -> Molecule   opt.xyz + clean labels + fragments + case-JSON H-bonds
    auto_view(mol, mode='face'|'tilted'|'hbond', tilt, azimuth, elevation, roll, landscape) -> View
    matched_view(view, mol_ref, mol, atoms)     same orientation for another model (Kabsch)
    View.fit(*scenes_or_points, pad) / .turned(az, el, roll) / .flipped() / .as_dict()
    molecule_scene(mol) / orbital_scene(cube, mol, iso=0.02) / mep_scene(density, esp, mol, iso=0.001,
        vrange, extrema) / recolour_mep(scene, vrange) / nci_scene(rdg, sl2r, mol, iso=0.5,
        vrange=(-0.035, 0.02)) -> Scene
    render(scene, view, px_per_A=None, panel_width_mm=85, supersample=2) -> Render
    crop_together([renders]) -> [renders]      one crop box for renders made with one camera
    px_per_angstrom(views, panel_width_mm)     common pixel density (>= 600 dpi at that panel width)
    Render.project(points) -> (xy px, depth);  Render.visible(points, towards_camera) (z-buffer test)
    case_dir(case) / case_file(case, name)     R2 snapshot paths;  MEP_CMAP, NCI_CMAP, ORBITAL_COLOURS

Typical use::

    from figkit import render3d as r3
    mol   = r3.load_molecule('btp_pcm')
    homo  = r3.orbital_scene(r3.case_file('btp_pcm', 'homo.cube'), mol)
    lumo  = r3.orbital_scene(r3.case_file('btp_pcm', 'lumo.cube'), mol)
    view  = r3.auto_view(mol, mode='tilted', tilt=20).fit(homo, lumo)     # one camera for both
    rh, rl = r3.crop_together([r3.render(homo, view, crop=False), r3.render(lumo, view, crop=False)])

Units: Angstrom everywhere inside this module; ESP in kcal/mol on MEP surfaces, sign(lambda2)rho in a.u.
Timing on the Pi (BTP, 49 atoms, ~2050 px panels, 2x supersampling): orbital scene 4 s (spline refinement),
MEP scene 2 s, NCI scene 2 s, each render 3-7 s.
"""
from __future__ import annotations

import os

# Headless rendering on the Pi works only through EGL device 1 (device 0 fails, OSMesa is not installed).
os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkEGLRenderWindow')
os.environ.setdefault('VTK_DEFAULT_EGL_DEVICE_INDEX', '1')

import csv
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import ndimage
import pyvista as pv
from matplotlib import colors as mcolors

from . import geometry, style

pv.OFF_SCREEN = True

__all__ = [
    'BOHR', 'HARTREE_KCAL', 'Cube', 'read_cube', 'Molecule', 'load_molecule', 'case_file', 'case_dir',
    'View', 'auto_view', 'matched_view', 'shared_hbond_view', 'ColourScale', 'MEP_CMAP', 'NCI_CMAP', 'ORBITAL_COLOURS',
    'Scene', 'molecule_scene', 'orbital_scene', 'mep_scene', 'recolour_mep', 'nci_scene', 'read_mep_extrema',
    'mep_colour_scale', 'nci_colour_scale', 'clean_label',
    'Render', 'render', 'crop_together', 'px_per_angstrom',
]

BOHR = 0.529177210903            # Angstrom per Bohr (CODATA 2018)
HARTREE_KCAL = 627.5094740631    # kcal/mol per Hartree
REPO = style.REPO
DATA = style.DATA

_SYMBOLS = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
            'Br', 'Kr']


# ----------------------------------------------------------------------------------------------------------
# Gaussian cube files
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Cube:
    """A volumetric grid.  data[i, j, k] sits at origin + i*axes[0] + j*axes[1] + k*axes[2] (Angstrom)."""
    data: np.ndarray            # (n1, n2, n3) float64
    origin: np.ndarray          # (3,) Angstrom
    axes: np.ndarray            # (3, 3) Angstrom, row m = step vector along index m
    numbers: np.ndarray         # (natoms,) atomic numbers
    coords: np.ndarray          # (natoms, 3) Angstrom
    title: str = ''
    comment: str = ''
    path: str = ''

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_orthogonal(self):
        a = self.axes / np.linalg.norm(self.axes, axis=1, keepdims=True)
        g = a @ a.T
        return bool(np.allclose(g, np.eye(3), atol=1e-8))

    @property
    def is_axis_aligned(self):
        off = self.axes - np.diag(np.diag(self.axes))
        return bool(np.all(np.abs(off) < 1e-10) and np.all(np.diag(self.axes) > 0))

    @property
    def symbols(self):
        return [_SYMBOLS[int(z)] if int(z) < len(_SYMBOLS) else str(int(z)) for z in self.numbers]

    def same_grid(self, other, tol=1e-6):
        return (self.shape == other.shape and np.allclose(self.origin, other.origin, atol=tol)
                and np.allclose(self.axes, other.axes, atol=tol))

    def grid(self, name='values'):
        """pv.ImageData for axis-aligned grids (fast flying-edges contouring), pv.StructuredGrid otherwise."""
        n1, n2, n3 = self.shape
        if self.is_axis_aligned:
            g = pv.ImageData(dimensions=(n1, n2, n3), spacing=tuple(np.diag(self.axes)), origin=tuple(self.origin))
        else:
            i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing='ij')
            pts = (self.origin[None, None, None, :] + i[..., None] * self.axes[0] + j[..., None] * self.axes[1]
                   + k[..., None] * self.axes[2])
            g = pv.StructuredGrid(pts[..., 0], pts[..., 1], pts[..., 2])
        g.point_data[name] = self.data.ravel(order='F')     # VTK: first index fastest
        return g

    def fractional_index(self, points):
        """Continuous (i, j, k) grid indices of Cartesian points (Angstrom); works for any axes."""
        pts = np.atleast_2d(np.asarray(points, float))
        return np.linalg.solve(self.axes.T, (pts - self.origin).T)          # (3, N)

    def interpolate(self, points, order=1, fill=np.nan):
        """Value at arbitrary points: trilinear (order=1) or cubic-spline (order=3) interpolation."""
        idx = self.fractional_index(points)
        out = ndimage.map_coordinates(self.data, idx, order=order, mode='nearest', prefilter=order > 1)
        outside = np.any((idx < -1e-9) | (idx > (np.array(self.shape)[:, None] - 1 + 1e-9)), axis=0)
        out[outside] = fill
        return out

    def refined(self, factor=2):
        """The cube resampled on a `factor`-times finer grid by cubic-spline interpolation (smoother
        isosurfaces from coarse PySCF grids; no new information is created)."""
        if factor == 1:
            return self
        n = np.array(self.shape)
        m = (n - 1) * factor + 1
        coords = np.meshgrid(*[np.linspace(0, k - 1, j) for k, j in zip(n, m)], indexing='ij', sparse=True)
        fine = ndimage.map_coordinates(self.data, np.broadcast_arrays(*coords), order=3, mode='nearest')
        return replace(self, data=fine, axes=self.axes / factor)

    def isosurface(self, level, normals=True, refine=1):
        """Triangulated isosurface (pv.PolyData).  normals=True attaches point normals from the field gradient
        (smoother than facet normals); Scene items flag such meshes with `field_normals=True` so render() keeps
        them and switches the actor to Gouraud interpolation.  refine>1 contours a spline-refined grid."""
        c = self
        if refine > 1:
            # clip far from the isovalue first: the spline would otherwise ring around the nuclear cusps and
            # create spurious islands (seen as MEP 'surfaces' with V ~ 7000 kcal/mol around the nuclei)
            lim = 8 * abs(level) if level else np.abs(self.data).max()
            c = replace(self, data=np.clip(self.data, -lim, lim)).refined(refine)
        g = c.grid('v')
        method = 'flying_edges' if isinstance(g, pv.ImageData) else 'contour'
        surf = g.contour([level], scalars='v', method=method, compute_normals=normals)
        return surf


def read_cube(path):
    """Read a Gaussian cube file (PySCF, Multiwfn, Gaussian).  Bohr -> Angstrom; supports general axes,
    negative voxel counts (Angstrom units) and the MO-cube header variant (negative atom count).
    Multi-value cubes keep only the first value per voxel."""
    path = Path(path)
    with open(path) as f:
        title = f.readline().rstrip('\n')
        comment = f.readline().rstrip('\n')
        tok = f.readline().split()
        natoms = int(tok[0])
        origin = np.array(tok[1:4], float)
        nval = int(tok[4]) if len(tok) > 4 else 1
        shape, axes, angstrom = [], [], False
        for _ in range(3):
            tok = f.readline().split()
            n = int(tok[0])
            if n < 0:
                angstrom = True
            shape.append(abs(n))
            axes.append([float(x) for x in tok[1:4]])
        numbers, coords = [], []
        for _ in range(abs(natoms)):
            tok = f.readline().split()
            numbers.append(int(float(tok[0])))
            coords.append([float(x) for x in tok[2:5]])
        if natoms < 0:                           # Gaussian MO cube: a line with the number of MOs and their ids
            tok = f.readline().split()
            nval = int(tok[0])
        values = np.array(f.read().split(), dtype=float)
    shape = tuple(shape)
    expected = shape[0] * shape[1] * shape[2] * nval
    if values.size != expected:
        raise ValueError(f'{path}: {values.size} values, header promises {expected} ({shape} x {nval})')
    data = values.reshape(shape + (nval,))[..., 0] if nval > 1 else values.reshape(shape)
    scale = 1.0 if angstrom else BOHR
    return Cube(data=data, origin=origin * scale, axes=np.array(axes) * scale, numbers=np.array(numbers),
                coords=np.array(coords).reshape(-1, 3) * scale, title=title.strip(), comment=comment.strip(),
                path=str(path))


# ----------------------------------------------------------------------------------------------------------
# Paths and geometry (figkit.geometry does the reading, label cleaning, bond / H-bond detection and Kabsch)
# ----------------------------------------------------------------------------------------------------------
def case_dir(case, raw=True):
    """Results directory of a case: the R2 snapshot (cubes, grids) when raw=True, else the tracked copy."""
    cp = geometry.find_case(case)
    d = cp.raw if raw else cp.tracked
    if d is None:
        raise FileNotFoundError(f'no {"raw" if raw else "tracked"} results for {case}')
    return d


def case_file(case, name, raw=True):
    return case_dir(case, raw) / name


clean_label = geometry.clean_label


@dataclass
class Molecule:
    symbols: list
    coords: np.ndarray                       # (N, 3) Angstrom
    labels: list                             # clean display labels (no '#1_545'); primes for duplicates
    raw_labels: list
    bonds: list                              # [(i, j)]
    hbonds: list = field(default_factory=list)       # [(donor, H, acceptor)]
    fragments: dict = field(default_factory=dict)    # name -> [indices]
    name: str = ''

    @property
    def heavy(self):
        return np.array([s != 'H' for s in self.symbols])

    def index(self, label):
        """Index of an atom from a clean or raw label."""
        for lst in (self.labels, self.raw_labels):
            if label in lst:
                return lst.index(label)
        raise KeyError(label)

    def principal_axes(self, atoms=None):
        """(centre, axes) with axes[:, 0] the LARGEST-variance direction, axes[:, 2] the smallest."""
        idx = np.flatnonzero(self.heavy) if atoms is None else np.asarray(atoms)
        P = self.coords[idx]
        c = P.mean(0)
        w, v = np.linalg.eigh(np.cov((P - c).T) if len(P) > 1 else np.eye(3))
        return c, v[:, ::-1], w[::-1]


def load_molecule(case=None, xyz=None, detect_hbonds=False):
    """Molecule for a case: optimised geometry (opt.xyz), clean labels (primes for duplicates), fragments and
    the hydrogen bonds named in the case JSON.  `xyz` overrides the geometry (e.g. the X-ray cluster).
    detect_hbonds=True adds every further N/O/Cl-H...A contact found by figkit.geometry.detect_hbonds."""
    meta = geometry.read_case(case) if case is not None else {}
    if xyz is None:
        xyz = meta['paths'].opt_xyz
    fr = geometry.read_xyz(xyz)
    sym, X = list(fr.symbols), np.asarray(fr.xyz, float)
    raw = meta.get('raw_labels') or [f'{s}{i + 1}' for i, s in enumerate(sym)]
    labels = meta.get('labels') or geometry.clean_labels(raw)
    if len(raw) != len(sym):
        raise ValueError(f'{xyz}: {len(sym)} atoms but {len(raw)} labels')
    bonds = geometry.detect_bonds(sym, X)
    hb = [(h['d'], h['h'], h['a']) for h in meta.get('hbond_list', [])]
    if detect_hbonds:
        for t in geometry.detect_hbonds(sym, X, bonds):
            if (t.d, t.h, t.a) not in hb:
                hb.append((t.d, t.h, t.a))
    return Molecule(symbols=sym, coords=X, labels=list(labels), raw_labels=list(raw), bonds=bonds,
                    hbonds=hb, fragments=meta.get('fragments', {}), name=case or str(xyz))


# ----------------------------------------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot(axis, deg):
    a = _unit(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


@dataclass
class View:
    """A camera.  `direction` points from the focal point TOWARDS the camera; `width`/`height` (Angstrom) are
    the framed region (None = frame the scene automatically at render time)."""
    focal: np.ndarray
    direction: np.ndarray
    up: np.ndarray
    width: float | None = None
    height: float | None = None
    parallel: bool = True
    view_angle: float = 15.0

    def __post_init__(self):
        self.direction = _unit(self.direction)
        up = np.asarray(self.up, float)
        up = up - (up @ self.direction) * self.direction
        self.up = _unit(up)
        self.focal = np.asarray(self.focal, float)

    @property
    def right(self):
        return _unit(np.cross(-self.direction, self.up))

    # manual tweaks, all in degrees, in screen terms
    def azimuth(self, deg):
        """Turn the molecule about the screen vertical (camera orbits horizontally)."""
        R = _rot(self.up, deg)
        return replace(self, direction=R @ self.direction, up=self.up.copy())

    def elevation(self, deg):
        """Tilt about the screen horizontal: positive shows more of the top side."""
        R = _rot(self.right, -deg)
        return replace(self, direction=R @ self.direction, up=R @ self.up)

    def roll(self, deg):
        """Rotate the picture in the screen plane (counter-clockwise for positive angles)."""
        R = _rot(self.direction, -deg)
        return replace(self, up=R @ self.up)

    def turned(self, azimuth=0.0, elevation=0.0, roll=0.0):
        v = self
        if azimuth:
            v = v.azimuth(azimuth)
        if elevation:
            v = v.elevation(elevation)
        if roll:
            v = v.roll(roll)
        return v

    def flipped(self):
        """Look from the other side, keeping screen-up: the camera turned by 180 deg about the screen vertical (a true
        view of the same molecule with left and right exchanged, not a mirror image)."""
        return replace(self, direction=-self.direction, up=self.up.copy())

    def transformed(self, R, t):
        return replace(self, focal=R @ self.focal + t, direction=R @ self.direction, up=R @ self.up)

    def screen(self, points):
        """(u, v, depth) of points: u right, v up, depth towards the camera (Angstrom)."""
        P = np.atleast_2d(points) - self.focal
        return np.c_[P @ self.right, P @ self.up, P @ self.direction]

    def fit(self, *items, pad=0.25, aspect=None):
        """Frame the union of Scenes / point arrays / Molecules: returns a new View with focal point, width and
        height set so the content fills the picture with `pad` Angstrom all round.  `aspect` (w/h) forces a
        frame shape (content centred)."""
        pts = np.vstack([_points_of(x) for x in items])
        s = self.screen(pts)
        u0, u1 = s[:, 0].min() - pad, s[:, 0].max() + pad
        v0, v1 = s[:, 1].min() - pad, s[:, 1].max() + pad
        w, h = u1 - u0, v1 - v0
        if aspect is not None:
            if w / h < aspect:
                w = h * aspect
            else:
                h = w / aspect
        focal = self.focal + 0.5 * (u0 + u1) * self.right + 0.5 * (v0 + v1) * self.up + s[:, 2].mean() * self.direction
        return replace(self, focal=focal, width=float(w), height=float(h))

    def distance(self, depth_extent=30.0):
        if self.parallel:
            return 200.0
        return 0.5 * self.height / math.tan(math.radians(self.view_angle) / 2) + depth_extent

    def as_dict(self):
        return dict(focal=self.focal.tolist(), direction=self.direction.tolist(), up=self.up.tolist(),
                    width=self.width, height=self.height, parallel=self.parallel, view_angle=self.view_angle)


def _points_of(x):
    if isinstance(x, Scene):
        return x.points()
    if isinstance(x, Molecule):
        return x.coords
    if isinstance(x, pv.DataSet):
        return np.asarray(x.points)
    return np.atleast_2d(np.asarray(x, float))


def _occluded_count(view, mol, targets, exclude=(), margin=0.25):
    """Number of atoms drawn in front of each target point within its footprint (ball radius + margin)."""
    s = view.screen(mol.coords)
    t = view.screen(targets)
    rad = np.array([style.BALL_RADII.get(x, 0.35) for x in mol.symbols])
    counts = []
    for k, (u, v, d) in enumerate(t):
        dist = np.hypot(s[:, 0] - u, s[:, 1] - v)
        front = (s[:, 2] > d + 0.1) & (dist < rad + margin)
        front[list(exclude[k]) if exclude else []] = False
        counts.append(int(front.sum()))
    return np.array(counts)


def auto_view(mol, mode='face', atoms=None, tilt=None, azimuth=0.0, elevation=0.0, roll=0.0, landscape=True,
              parallel=True, hbonds=None, flip=False):
    """Automatic camera from the principal axes of the heavy atoms.

    mode='face'   look along the smallest-variance axis (face-on for near-planar clusters), longest in-plane
                  axis horizontal (landscape=True) or vertical.
    mode='hbond'  start from the face view and tilt it so the hydrogen-bond NCI discs (normal ~ H->A) are seen
                  obliquely instead of edge-on, with no atom in front of the disc centres.  `tilt` (deg) fixes
                  the tilt angle; otherwise it is searched in 30-65 deg.
    `mode='tilted'` is 'face' plus `tilt` degrees about the screen horizontal.
    azimuth/elevation/roll are applied afterwards (degrees, screen terms).  Returns an unframed View;
    call .fit(scene, ...) to frame it (render() does that automatically when needed).
    Orientation is deterministic: the first fragment (CIF order) is placed on the left / top."""
    c, ax, ev = mol.principal_axes(atoms)
    e1, e2, e3 = ax[:, 0], ax[:, 1], ax[:, 2]
    # deterministic signs: first-fragment centroid on the negative e1 side, right-handed frame
    first = list(mol.fragments.values())[0] if mol.fragments else [0]
    if (mol.coords[first].mean(0) - c) @ e1 > 0:
        e1 = -e1
    e3 = np.cross(e1, e2)
    d = e3 if not flip else -e3
    if mode in ('face', 'tilted', 'hbond'):
        v = _orient(View(c, d, e2), mol, atoms, landscape)
    else:
        raise ValueError(mode)
    if mode == 'tilted':
        v = v.elevation(tilt if tilt is not None else 35.0)
    elif mode == 'hbond':
        hb = mol.hbonds if hbonds is None else hbonds
        if hb:
            v = _hbond_view(v, mol, hb, tilt, landscape, atoms)
    return v.turned(azimuth, elevation, roll)


def _orient(view, mol, atoms, landscape):
    """Keep the viewing direction, choose screen-up so the long projected axis is horizontal (or vertical)."""
    idx = np.flatnonzero(mol.heavy) if atoms is None else np.asarray(atoms)
    s = view.screen(mol.coords[idx])[:, :2]
    s = s - s.mean(0)
    w, v = np.linalg.eigh(np.cov(s.T))
    major = v[:, 1]                                      # long 2D axis, in (right, up) components
    long3 = major[0] * view.right + major[1] * view.up
    first = list(mol.fragments.values())[0] if mol.fragments else [0]
    if (mol.coords[first].mean(0) - mol.coords[idx].mean(0)) @ long3 > 0:
        long3 = -long3                                   # first fragment on the left / top
    if landscape:
        up = np.cross(view.direction, long3)             # makes screen-right = long3
    else:
        up = -long3
    return replace(view, up=_unit(up))


def _hbond_view(view, mol, hbonds, tilt, landscape, atoms):
    H = np.array([mol.coords[h] for _, h, _ in hbonds])
    A = np.array([mol.coords[a] for _, _, a in hbonds])
    n = (A - H) / np.linalg.norm(A - H, axis=1, keepdims=True)
    centres = H + 0.45 * (A - H)
    excl = [(d, h, a) for d, h, a in hbonds]
    best = None
    tilts = [tilt] if tilt is not None else list(range(30, 66, 5))
    for t in tilts:
        for phi in range(0, 360, 15):                     # direction of the tilt in the screen plane
            v = view.roll(phi).elevation(t).roll(-phi)
            vis = np.abs(n @ v.direction)                 # 0 = disc edge-on, 1 = face-on
            occ = _occluded_count(v, mol, centres, excl)
            face = abs(v.direction @ view.direction)
            score = np.minimum(vis / 0.55, 1.0).sum() - 1.0 * occ.sum() + 0.4 * face
            if best is None or score > best[0] + 1e-9:
                best = (score, v)
    return _orient(best[1], mol, atoms, landscape)


def matched_view(view, mol_ref, mol, atoms=None):
    """The view of `mol_ref` carried over to `mol` (same atom order) by a Kabsch fit on `atoms` (default: all
    heavy atoms), so two models of the same cluster are shown in the same orientation."""
    idx = np.flatnonzero(mol_ref.heavy) if atoms is None else np.asarray(atoms)
    R, t, _ = geometry.kabsch(mol_ref.coords[idx], mol.coords[idx])     # mol ~ mol_ref @ R.T + t
    return view.transformed(R, t)


def _ring_normal(mol, idx):
    P = mol.coords[idx] - mol.coords[idx].mean(0)
    return np.linalg.svd(P)[2][2]


def _disc_hidden_fraction(view, mol, h, a, disc_radius=0.45, n_ring=8):
    """Fraction of sample points of an H-bond NCI disc (centre H + 0.45 (A - H), normal H->A, radius
    disc_radius) that lie behind an atom ball in the given view."""
    H, A = mol.coords[h], mol.coords[a]
    n = _unit(A - H)
    c = H + 0.45 * (A - H)
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 = _unit(e1)
    e2 = np.cross(n, e1)
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    S = np.vstack([c, c + disc_radius * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))])
    s, t = view.screen(mol.coords), view.screen(S)
    rad = np.array([style.BALL_RADII.get(x, 0.35) for x in mol.symbols])
    hidden = [bool(((s[:, 2] > d + 0.05) & (np.hypot(s[:, 0] - u, s[:, 1] - v) < rad)).any()) for u, v, d in t]
    return float(np.mean(hidden))


def shared_hbond_view(mols, ref=None, atoms=None, tilts=range(25, 41, 5), phis=range(0, 360, 15),
                      orient_atoms=None, landscape=True, edge_on_deg=60.0, disc_radius=0.45):
    """ONE camera for several models of the same cluster (same atom order, e.g. gas phase and IEF-PCM) in which
    the hydrogen-bond NCI discs of EVERY model are seen obliquely and unobstructed and no fragment ring of any
    model is seen edge-on.  (auto_view(mode='hbond') optimises one model only; carried to a second model whose
    ring has turned, its camera can put that model's disc behind the turned ring.)

    mols: dict name -> Molecule (or a list).  ref: key/index of the reference model (default: the first).
    atoms: fit atoms of the reference (face view and the Kabsch transfer to the other models, as matched_view;
    default all heavy atoms).  The search tilts the reference face view by t in `tilts` about a screen axis at
    angle phi in `phis` and scores, summed over models: min(|n.d| / 0.5, 1) per hydrogen bond (n = H->A; disc
    visibility) - 2 x the fraction of the disc hidden behind atom balls - max(0, (ring-normal angle -
    edge_on_deg) / 25) per fragment in mol.fragments, + 0.3 |d.d_face|.  The winner is oriented landscape (or
    portrait) on `orient_atoms` of the reference (default: all its heavy atoms, i.e. the whole cluster).
    Returns (views, info): views with the same keys / order as `mols` (unframed), info = dict(tilt, phi, score,
    details) with per-model disc elevations (deg from edge-on), hidden fractions and ring angles (deg)."""
    keyed = isinstance(mols, dict)
    names = list(mols) if keyed else list(range(len(mols)))
    M = dict(mols) if keyed else dict(enumerate(mols))
    ref = names[0] if ref is None else ref
    mref = M[ref]
    fit = np.flatnonzero(mref.heavy) if atoms is None else np.asarray(atoms)
    face = auto_view(mref, mode='face', atoms=fit)
    kab = {k: geometry.kabsch(mref.coords[fit], M[k].coords[fit])[:2] for k in names}
    frags = {k: {f: [i for i in idx if M[k].symbols[i] != 'H'] for f, idx in M[k].fragments.items()}
             for k in names}
    best = None
    for t in tilts:
        for phi in (phis if t else [0]):
            v = face.roll(phi).elevation(t).roll(-phi) if t else face
            score = 0.3 * abs(v.direction @ face.direction)
            det = {}
            for k in names:
                m = M[k]
                vv = v.transformed(*kab[k])
                for _, h, a in m.hbonds:
                    vis = abs(_unit(m.coords[a] - m.coords[h]) @ vv.direction)
                    hid = _disc_hidden_fraction(vv, m, h, a, disc_radius)
                    score += min(vis / 0.5, 1.0) - 2.0 * hid
                    det[f'{k}: {m.labels[h]}...{m.labels[a]}'] = dict(
                        elevation_deg=round(math.degrees(math.asin(min(vis, 1.0))), 1), hidden=round(hid, 2))
                for f, idx in frags[k].items():
                    if len(idx) >= 3:
                        ang = math.degrees(math.acos(min(1.0, abs(_ring_normal(m, idx) @ vv.direction))))
                        score -= max(0.0, (ang - edge_on_deg) / 25.0)
                        det[f'{k}: {f} ring normal to view (deg)'] = round(ang, 1)
            if best is None or score > best[0] + 1e-9:
                best = (score, v, t, phi, det)
    oa = np.flatnonzero(mref.heavy) if orient_atoms is None else np.asarray(orient_atoms)
    v_ref = _orient(best[1], mref, oa, landscape)
    views = {k: (v_ref if k == ref else matched_view(v_ref, mref, M[k], atoms=fit)) for k in names}
    info = dict(tilt=best[2], phi=best[3], score=round(best[0], 3), details=best[4])
    return (views if keyed else [views[k] for k in names]), info


# ----------------------------------------------------------------------------------------------------------
# Colour scales (shared with compose.py, which draws the matplotlib colour bars)
# ----------------------------------------------------------------------------------------------------------
MEP_CMAP = mcolors.LinearSegmentedColormap.from_list('mep_rgb', [
    (0.00, '#B2182B'), (0.17, '#E4572E'), (0.33, '#F3C03F'), (0.50, '#5BB450'),
    (0.70, '#2A9FC0'), (1.00, '#1F3A93')])
NCI_CMAP = mcolors.LinearSegmentedColormap.from_list('nci_bgr', [
    (0.0, '#1E3CD2'), (0.5, '#23B43A'), (1.0, '#D7261E')])
ORBITAL_COLOURS = ('#E0842A', '#6A5CC4')        # positive, negative phase: orange / blue-violet


@dataclass
class ColourScale:
    """Everything the colour bar needs: cmap, norm (maps data -> [0, 1]), label, ticks, tick format."""
    cmap: mcolors.Colormap
    norm: mcolors.Normalize
    label: str = ''
    ticks: list | None = None
    fmt: str = '{:g}'
    extend: str = 'neither'

    def rgba(self, values):
        return self.cmap(self.norm(np.asarray(values, float)))


def _nice_step(span, n=5):
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def mep_colour_scale(values_kcal=None, vrange='symmetric', centre_zero=True, nice=5.0):
    """Colour scale for MEP maps in kcal/mol.  vrange: 'symmetric' (+-max|V|), 'data' (min, max) or an explicit
    (vmin, vmax).  centre_zero puts green exactly at 0 kcal/mol for asymmetric ranges (two-slope norm).
    `nice` rounds data-driven limits outwards to that multiple."""
    if isinstance(vrange, str):
        v = np.asarray(values_kcal, float)
        v = v[np.isfinite(v)]
        lo, hi = float(v.min()), float(v.max())
        if vrange == 'symmetric':
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        if nice:
            lo, hi = nice * math.floor(lo / nice), nice * math.ceil(hi / nice)
    else:
        lo, hi = map(float, vrange)
    if centre_zero and lo < 0 < hi and abs(lo + hi) > 1e-9:
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)
    else:
        norm = mcolors.Normalize(lo, hi)
    step = _nice_step(hi - lo, 6)
    ticks = [t for t in np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)]
    if 0.0 not in ticks and lo < 0 < hi:
        ticks = sorted(set(ticks + [0.0]))
    return ColourScale(MEP_CMAP, norm, 'Electrostatic potential (kcal mol$^{-1}$)', ticks, '{:.0f}')


def nci_colour_scale(vmin=-0.035, vmax=0.02):
    step = 0.01
    ticks = [round(t, 3) for t in np.arange(math.ceil(vmin / step) * step, vmax + 1e-9, step)]
    return ColourScale(NCI_CMAP, mcolors.Normalize(vmin, vmax), r'sign($\lambda_2$)$\rho$ (a.u.)', ticks,
                       '{:.2f}', extend='neither')


# ----------------------------------------------------------------------------------------------------------
# Scenes
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Scene:
    """Meshes to draw + what the 2D layer needs to know."""
    mol: Molecule | None
    items: list = field(default_factory=list)        # [(pv.PolyData, add_mesh kwargs)]
    colour: ColourScale | None = None
    markers: list = field(default_factory=list)      # [dict(pos=(3,), text=str, kind=str, radius=float)]
    meta: dict = field(default_factory=dict)
    translucent: bool = False

    def points(self):
        pts = [np.asarray(m.points) for m, _ in self.items if m.n_points]
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def add(self, mesh, **kw):
        self.items.append((mesh, kw))
        return self


def _sphere_mesh(centres, radii, colours, res=36):
    meshes = []
    for c, r, col in zip(centres, radii, colours):
        s = pv.Sphere(radius=float(r), center=tuple(c), theta_resolution=res, phi_resolution=res)
        s.point_data['rgb'] = np.tile((np.array(mcolors.to_rgb(col)) * 255).astype(np.uint8), (s.n_points, 1))
        meshes.append(s)
    return pv.merge(meshes) if meshes else pv.PolyData()


def _cylinder(a, b, r, res):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    return pv.Cylinder(center=tuple((a + b) / 2), direction=tuple(b - a), radius=float(r), height=float(L),
                       resolution=res, capping=False)


def molecule_meshes(mol, atom_scale=1.0, bond_radius=0.085, hbonds=True, hbond_colour='#3A3A3A',
                    hbond_radius=0.035, dash=(0.16, 0.11), atoms=None, res=36):
    """[(mesh, kwargs)] for spheres (per-element colour and radius), half-coloured bond cylinders and dashed
    hydrogen bonds.  `atoms` restricts drawing to a subset of indices."""
    keep = np.ones(len(mol.symbols), bool) if atoms is None else np.isin(np.arange(len(mol.symbols)), atoms)
    rad = np.array([style.BALL_RADII.get(s, 0.35) * atom_scale for s in mol.symbols])
    col = [style.ELEMENT_COLOURS.get(s, '#FF69B4') for s in mol.symbols]
    X = mol.coords
    spheres = _sphere_mesh(X[keep], rad[keep], [c for c, k in zip(col, keep) if k], res)
    cyl = []
    for i, j in mol.bonds:
        if not (keep[i] and keep[j]):
            continue
        u = _unit(X[j] - X[i])
        L = np.linalg.norm(X[j] - X[i])
        m = X[i] + u * (rad[i] + 0.5 * (L - rad[i] - rad[j]))     # middle of the visible stick
        for a, b, c in ((X[i] + u * 0.5 * rad[i], m, col[i]), (m, X[j] - u * 0.5 * rad[j], col[j])):
            cy = _cylinder(a, b, bond_radius, 24)
            cy.point_data['rgb'] = np.tile((np.array(mcolors.to_rgb(c)) * 255).astype(np.uint8), (cy.n_points, 1))
            cyl.append(cy)
    out = [(spheres, dict(scalars='rgb', rgb=True, smooth_shading=True, specular=0.35, specular_power=30,
                          diffuse=0.8, ambient=0.18))]
    if cyl:
        out.append((pv.merge(cyl), dict(scalars='rgb', rgb=True, smooth_shading=True, specular=0.25,
                                        specular_power=20, diffuse=0.8, ambient=0.2)))
    if hbonds and mol.hbonds:
        dashes = []
        for d, h, a in mol.hbonds:
            if not (keep[h] and keep[a]):
                continue
            u = _unit(X[a] - X[h])
            start, end = X[h] + u * (rad[h] + 0.04), X[a] - u * (rad[a] + 0.04)
            L = np.linalg.norm(end - start)
            period = dash[0] + dash[1]
            n = max(1, int((L + dash[1]) // period))
            off = 0.5 * (L - (n * period - dash[1]))
            for k in range(n):
                s0 = start + u * (off + k * period)
                dashes.append(_cylinder(s0, s0 + u * dash[0], hbond_radius, 16))
        if dashes:
            out.append((pv.merge(dashes), dict(color=hbond_colour, smooth_shading=True, specular=0.1)))
    return out


def molecule_scene(mol, **kw):
    sc = Scene(mol)
    for m, k in molecule_meshes(mol, **kw):
        sc.add(m, **k)
    return sc


def _subsample(P, n=6000, seed=0):
    P = np.asarray(P)
    if len(P) <= n:
        return P.copy()
    return P[np.random.default_rng(seed).choice(len(P), n, replace=False)]


def _as_cube(x):
    return x if isinstance(x, Cube) else read_cube(x)


def orbital_scene(cube, mol=None, iso=0.02, colours=ORBITAL_COLOURS, opacity=0.6, molecule=True,
                  molecule_kw=None, refine=2):
    """+-iso isosurfaces of an orbital cube (a.u.) with the ball-and-stick framework inside.
    If mol is None the atoms are taken from the cube (generic labels)."""
    t0 = time.time()
    cube = _as_cube(cube)
    if mol is None:
        mol = _mol_from_cube(cube)
    _check_frame(cube, mol)
    sc = Scene(mol, translucent=opacity < 1)
    if molecule:
        for m, k in molecule_meshes(mol, **(molecule_kw or {})):
            sc.add(m, **k)
    for lvl, colr in ((iso, colours[0]), (-iso, colours[1])):
        s = cube.isosurface(lvl, refine=refine)
        if s.n_points:
            sc.add(s, color=colr, opacity=opacity, field_normals=True, specular=0.35, specular_power=25,
                   diffuse=0.85, ambient=0.15)
    lobes = [m for m, k in sc.items if k.get('field_normals')]
    sc.meta.update(kind='orbital', iso=iso, cube=cube.path, t_build=time.time() - t0,
                   n_tri=int(sum(m.n_cells for m in lobes)),
                   surface_points=_subsample(np.vstack([m.points for m in lobes])) if lobes else None)
    return sc


def _mol_from_cube(cube):
    sym = cube.symbols
    return Molecule(symbols=sym, coords=cube.coords, labels=[f'{s}{i + 1}' for i, s in enumerate(sym)],
                    raw_labels=[f'{s}{i + 1}' for i, s in enumerate(sym)],
                    bonds=geometry.detect_bonds(sym, cube.coords))


def _check_frame(cube, mol, tol=2e-3):
    """The cube atoms must coincide with the molecule (same frame, same order)."""
    if len(cube.coords) == len(mol.coords):
        dev = np.abs(cube.coords - mol.coords).max()
        if dev > tol:
            raise ValueError(f'{cube.path}: atoms deviate from the molecule by {dev:.4f} A (wrong geometry?)')


def read_mep_extrema(path):
    """Rows of mep_extrema.csv as dicts: type ('minimum'/'maximum'), global (bool), kcal, pos (A), near."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(type=r['type'], index=int(r['index']), is_global=r['global'].strip() == 'yes',
                             au=float(r['ESP_au']), kcal=float(r['ESP_kcal_mol']),
                             pos=np.array([float(r['x_A']), float(r['y_A']), float(r['z_A'])]),
                             near=r.get('nearest_atoms', '')))
    return rows


def mep_scene(density, esp, mol=None, iso=0.001, vrange='symmetric', centre_zero=True, nice=5.0,
              opacity=1.0, molecule=True, extrema=None, extrema_min_kcal=0.0, marker_radius=0.19,
              marker_colours=('#00B7C7', '#F2A200'), interp_order=3, molecule_kw=None, refine=2):
    """MEP: iso (a.u.) density isosurface coloured by the ESP cube interpolated onto its vertices (kcal/mol).
    vrange/centre_zero/nice: see mep_colour_scale.  extrema: None, 'global', 'all' or a list of rows from
    read_mep_extrema (a path is read); extrema_min_kcal drops extrema with |V| below it (globals are kept).
    Markers: minima cyan, maxima amber (small spheres on the surface), also stored in scene.markers for the
    2D layer."""
    t0 = time.time()
    dens, pot = _as_cube(density), _as_cube(esp)
    if mol is None:
        mol = _mol_from_cube(dens)
    _check_frame(dens, mol)
    surf = dens.isosurface(iso, refine=refine)
    v = pot.interpolate(surf.points, order=interp_order) * HARTREE_KCAL
    if np.isnan(v).any():
        raise ValueError('density isosurface leaves the ESP grid')
    scale = mep_colour_scale(v, vrange, centre_zero, nice)
    surf.point_data['t'] = np.clip(scale.norm(v), 0, 1)
    surf.point_data['esp_kcal'] = v
    sc = Scene(mol, colour=scale, translucent=opacity < 1)
    if molecule and opacity < 1:
        for m, k in molecule_meshes(mol, **(molecule_kw or {})):
            sc.add(m, **k)
    sc.add(surf, scalars='t', cmap=scale.cmap, clim=(0, 1), opacity=opacity, field_normals=True,
           specular=0.12, specular_power=20, diffuse=0.7, ambient=0.32, interpolate_before_map=True,
           show_scalar_bar=False, n_colors=512)
    rows = []
    if extrema is not None:
        if isinstance(extrema, (str, Path)) and str(extrema) not in ('global', 'all'):
            rows = read_mep_extrema(extrema)
        elif isinstance(extrema, str):
            src = Path(dens.path).with_name('mep_extrema.csv')
            rows = read_mep_extrema(src)
            if extrema == 'global':
                rows = [r for r in rows if r['is_global'] or r['kcal'] == max(x['kcal'] for x in rows)]
        else:
            rows = list(extrema)
        rows = [r for r in rows if abs(r['kcal']) >= extrema_min_kcal or r['is_global']]
        # Multiwfn puts the extrema ON the surface; lift each marker by 0.6 radius along the outward normal
        # (-grad rho) so it reads as a bead sitting on the surface instead of a half-buried dot
        for r in rows:
            r.setdefault('pos_marker', _lift(dens, r['pos'], 0.6 * marker_radius))
        for kind, colr in (('minimum', marker_colours[0]), ('maximum', marker_colours[1])):
            pts = [r['pos_marker'] for r in rows if r['type'] == kind]
            if pts:
                sc.add(_sphere_mesh(pts, [marker_radius] * len(pts), [colr] * len(pts), 24),
                       scalars='rgb', rgb=True, smooth_shading=True, specular=0.4)
        for r in rows:
            sc.markers.append(dict(pos=r['pos_marker'], text=f"{r['kcal']:.1f}".replace('-', '−'), kind=r['type'],
                                   radius=marker_radius, value=r['kcal'], is_global=r['is_global']))
    sc.meta.update(kind='mep', iso=iso, vmin_surface=float(v.min()), vmax_surface=float(v.max()),
                   n_vertices=int(surf.n_points), t_build=time.time() - t0)
    return sc


def _lift(cube, p, dist, h=0.05):
    p = np.asarray(p, float)
    g = np.array([(cube.interpolate([p + h * e], order=1)[0] - cube.interpolate([p - h * e], order=1)[0]) / (2 * h)
                  for e in np.eye(3)])
    if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0:
        return p
    return p - dist * g / np.linalg.norm(g)


def recolour_mep(scene, vrange='symmetric', centre_zero=True, nice=5.0):
    """Re-map an MEP scene's colours to a new range without rebuilding it (e.g. a common range for several
    models: vrange=(min(lo), max(hi)) over their scene.meta['vmin_surface'/'vmax_surface'])."""
    surf = next(m for m, k in scene.items if 'esp_kcal' in m.point_data)
    scale = mep_colour_scale(surf.point_data['esp_kcal'], vrange, centre_zero, nice)
    surf.point_data['t'] = np.clip(scale.norm(np.asarray(surf.point_data['esp_kcal'])), 0, 1)
    for m, k in scene.items:
        if m is surf:
            k['cmap'] = scale.cmap
    scene.colour = scale
    return scene


def nci_scene(rdg, sl2r, mol=None, iso=0.5, vrange=(-0.035, 0.02), opacity=1.0, molecule=True,
              min_points=0, molecule_kw=None, refine=1):
    """NCI: RDG = iso isosurface coloured by sign(lambda2)rho (a.u.) over vrange, blue-green-red.
    All patches are kept (ring-centre steric patches included); min_points drops tiny fragments."""
    t0 = time.time()
    r, s = _as_cube(rdg), _as_cube(sl2r)
    if not r.same_grid(s):
        raise ValueError('rdg and sl2r cubes are on different grids')
    if mol is None:
        mol = _mol_from_cube(r)
    _check_frame(r, mol)
    surf = r.isosurface(iso, refine=refine)
    if min_points and surf.n_points:
        conn = surf.connectivity('all')
        ids = conn.point_data['RegionId']
        keep_regions = [k for k in np.unique(ids) if (ids == k).sum() >= min_points]
        surf = conn.extract_points(np.isin(ids, keep_regions), adjacent_cells=False).extract_surface()
    val = s.interpolate(surf.points, order=1)
    scale = nci_colour_scale(*vrange)
    surf.point_data['t'] = np.clip(scale.norm(val), 0, 1)
    surf.point_data['sl2r'] = val
    sc = Scene(mol, colour=scale, translucent=opacity < 1)
    if molecule:
        # no dashed H-bond sticks here: the NCI disc itself marks the contact and a dash would pierce it
        for m, k in molecule_meshes(mol, **{'hbonds': False, **(molecule_kw or {})}):
            sc.add(m, **k)
    sc.add(surf, scalars='t', cmap=scale.cmap, clim=(0, 1), opacity=opacity, field_normals=True,
           specular=0.25, specular_power=20, diffuse=0.85, ambient=0.2, interpolate_before_map=True,
           show_scalar_bar=False, n_colors=512)
    sc.meta.update(kind='nci', iso=iso, n_vertices=int(surf.n_points), t_build=time.time() - t0,
                   surface_points=_subsample(np.asarray(surf.points)) if surf.n_points else None,
                   sl2r_min=float(np.nanmin(val)) if len(val) else None,
                   sl2r_max=float(np.nanmax(val)) if len(val) else None)
    return sc


# ----------------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------------
def px_per_angstrom(views, panel_width_mm=85.0, dpi=style.DPI, oversize=1.02):
    """Pixel density so that the WIDEST of the (framed) views fills `panel_width_mm` at `dpi`."""
    views = views if isinstance(views, (list, tuple)) else [views]
    w = max(v.width for v in views)
    return oversize * panel_width_mm / 25.4 * dpi / w


@dataclass
class Render:
    image: np.ndarray                 # (h, w, 3) uint8, cropped
    view: View
    matrix: np.ndarray                # 4x4 world -> clip coordinates (VTK composite projection)
    window: tuple                     # (W, H) final-resolution window before cropping
    offset: tuple                     # (x0, y0) of the crop in the full window (pixels, y from top)
    depth: np.ndarray                 # (H, W) float32 z-buffer in [0, 1], row 0 = TOP, nearest surface per pixel
    px_per_A: float
    colour: ColourScale | None = None
    markers: list = field(default_factory=list)
    mol: Molecule | None = None
    meta: dict = field(default_factory=dict)

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]

    def _clip(self, points):
        P = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))]
        c = P @ self.matrix.T
        return c[:, :3] / c[:, 3:4]

    def project(self, points):
        """Pixel coordinates (x right, y DOWN, in the cropped image; pixel centres at i + 0.5) and z-buffer
        depth in [0, 1] of 3D points."""
        n = self._clip(points)
        W, H = self.window
        x = (n[:, 0] + 1) / 2 * W - self.offset[0]
        y = H - (n[:, 1] + 1) / 2 * H - self.offset[1]
        z = (n[:, 2] + 1) / 2
        return np.c_[x, y], z

    def visible(self, points, towards_camera=0.0, tol=2e-3, window=1):
        """True where a 3D point (moved `towards_camera` Angstrom towards the viewer, e.g. by an atom radius)
        is not hidden behind an OPAQUE surface.  Translucent orbital lobes do not hide anything."""
        P = np.atleast_2d(points)
        P = P + np.reshape(np.broadcast_to(towards_camera, len(P)), (-1, 1)) * self.view.direction
        n = self._clip(P)
        W, H = self.window
        col = np.floor((n[:, 0] + 1) / 2 * W).astype(int)
        row = np.floor(H - (n[:, 1] + 1) / 2 * H).astype(int)
        z = (n[:, 2] + 1) / 2
        out = np.zeros(len(P), bool)
        for k, (r, c) in enumerate(zip(row, col)):
            if 0 <= r < H and 0 <= c < W:
                zb = self.depth[max(0, r - window):r + window + 1, max(0, c - window):c + window + 1].max()
                out[k] = z[k] <= zb + tol
        return out

    def px_scale(self):
        """Pixels per Angstrom in the image."""
        return self.px_per_A

    def save_png(self, path):
        from PIL import Image
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(path)
        return Path(path)


def _content_bbox(img, thresh=250):
    mask = img.min(axis=2) < thresh
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        return 0, 0, img.shape[1], img.shape[0]
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1          # x0, y0, x1, y1


def _crop(r, box, pad):
    x0, y0, x1, y1 = box
    ox, oy = r.offset
    full_h, full_w = r.image.shape[:2]
    # coordinates of the box are relative to r.image; pad with white where the pad leaves the image
    X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    out = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X1, full_w), min(Y1, full_h)
    out[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = r.image[sy0:sy1, sx0:sx1]
    return replace(r, image=out, offset=(ox + X0, oy + Y0))


def crop_together(renders, pad=14, thresh=250):
    """Crop renders of the SAME window size to the union of their content boxes (+pad px), so panels made
    with one camera stay pixel-aligned.  A single render is simply cropped tight."""
    rs = list(renders) if isinstance(renders, (list, tuple)) else [renders]
    boxes = np.array([_content_bbox(r.image, thresh) for r in rs])
    if len({r.image.shape for r in rs}) != 1 or len({r.offset for r in rs}) != 1:
        raise ValueError('crop_together needs uncropped renders of identical window size')
    box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    out = [_crop(r, box, pad) for r in rs]
    return out if isinstance(renders, (list, tuple)) else out[0]


def render(scene, view=None, px_per_A=None, panel_width_mm=85.0, supersample=2, crop=True, pad=14,
           background='white', depth_peels=6, lighting='light kit', ssao=False):
    """Render a Scene with a View.  Returns a Render (image cropped to content + pad px unless crop=False).

    px_per_A: pixel density of the final image.  If None, the framed view width fills `panel_width_mm` at
    600 dpi (use the same explicit value for panels that must share a scale).  supersample=2 renders at twice
    the size and box-filters down (anti-aliasing).  Translucent scenes use depth peeling."""
    t0 = time.time()
    if view is None:
        view = auto_view(scene.mol)
    if view.width is None:
        view = view.fit(scene)
    if px_per_A is None:
        px_per_A = px_per_angstrom(view, panel_width_mm)
    W, H = int(round(view.width * px_per_A)), int(round(view.height * px_per_A))
    ss = int(supersample)
    p = pv.Plotter(off_screen=True, window_size=(W * ss, H * ss), lighting=lighting)
    p.set_background(background)
    for mesh, kw in scene.items:
        kw = dict(kw)
        kw.setdefault('show_scalar_bar', False)
        field_normals = kw.pop('field_normals', False)
        if field_normals:
            kw['smooth_shading'] = False          # keep the gradient normals of the isosurface ...
        actor = p.add_mesh(mesh, reset_camera=False, **kw)
        if field_normals:
            actor.prop.interpolation = 'gouraud'  # ... and interpolate them (pyvista would shade flat)
    if scene.translucent:
        # pyvista's enable_depth_peeling() makes translucent actors vanish on EGL/llvmpipe; the plain VTK calls
        # work and give correct order-independent blending (checked against the analytic two-layer result)
        p.ren_win.SetAlphaBitPlanes(1)
        p.ren_win.SetMultiSamples(0)
        p.renderer.SetUseDepthPeeling(1)
        p.renderer.SetMaximumNumberOfPeels(depth_peels)
        p.renderer.SetOcclusionRatio(0.0)
    if ssao:
        p.enable_ssao(radius=0.6, bias=0.01, kernel_size=128)
    cam = p.renderer.GetActiveCamera()
    dist = view.distance()
    cam.SetFocalPoint(*view.focal)
    cam.SetPosition(*(view.focal + dist * view.direction))
    cam.SetViewUp(*view.up)
    if view.parallel:
        cam.ParallelProjectionOn()
        cam.SetParallelScale(view.height / 2)
    else:
        cam.ParallelProjectionOff()
        cam.SetViewAngle(view.view_angle)
    p.camera_set = True
    p.renderer.ResetCameraClippingRange()
    t1 = time.time()
    img = p.screenshot(return_img=True)
    t2 = time.time()
    # z-buffer (VTK rows start at the bottom) and the projection actually used
    import vtk
    arr = vtk.vtkFloatArray()
    p.ren_win.GetZbufferData(0, 0, W * ss - 1, H * ss - 1, arr)
    from vtk.util.numpy_support import vtk_to_numpy
    zb = vtk_to_numpy(arr).reshape(H * ss, W * ss)[::-1]
    aspect = p.renderer.GetTiledAspectRatio()
    M = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
    matrix = np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)])
    p.close()
    if ss > 1:
        img = img[:H * ss, :W * ss].reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
        zb = zb.reshape(H, ss, W, ss).min(axis=(1, 3))
    r = Render(image=np.ascontiguousarray(img), view=view, matrix=matrix, window=(W, H), offset=(0, 0),
               depth=zb.astype(np.float32), px_per_A=px_per_A, colour=scene.colour, markers=list(scene.markers),
               mol=scene.mol, meta=dict(scene.meta))
    r.meta.update(t_setup=t1 - t0, t_render=t2 - t1, t_total=time.time() - t0, window=(W, H), supersample=ss)
    if crop:
        r = crop_together(r, pad=pad)
    return r
