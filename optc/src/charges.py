#!/usr/bin/env python3
"""Atomic charges fitted to the periodic electrostatic potential (REPEAT: Campana, Mussard & Woo, J. Chem. Theory
Comput. 5 (2009) 2866), for the lattice field of the embedded rungs (optc/PLAN.md).

    tools/env/bin/python optc/src/charges.py fit --key KEY --point P0      # PRIMARY (REPEAT) -> charges_P0.json
    tools/env/bin/python optc/src/charges.py sc  --key KEY --point P0 [--threads 8]   # -> charges_P0_sc.json, _gas.json
    tools/env/bin/python optc/src/charges.py selftest

Three charge models (optc/PLAN.md log, 18 Oct 2026): PRIMARY = REPEAT (below); SECOND = self-consistent in-crystal
CHELPG; THIRD = gas-phase CHELPG (no polarisation by the crystal).
Self-consistent in-crystal CHELPG:  Each symmetry-unique
molecule of the state (P0: the CIF composition; P4: the composition of the collective end point) is computed with
PBE0-D3(BJ)/def2-TZVP in the Ewald-consistent field of all the others (optc/src/embed.py build_one: every molecule within
25 A + fitted boundary charges), and its own electrostatic potential (nuclei + density, the field excluded) is fitted
on a CHELPG grid (0.3 A spacing, from the Bondi surface out to 2.8 A; Breneman & Wiberg, J. Comput. Chem. 11 (1990)
361) with its formal charge fixed and a weak harmonic restraint on heavy atoms (5e-4 Eh^2/e^2, about RESP's a = 0.0005).
Start: the same fit in vacuum (the second, gas-phase charge model, kept as charges_<point>_gas.json); cycles until the
largest change is below 0.005 e (at most 6).  Symmetry copies share the charges of their CIF labels.
(An earlier version scaled the restraints by the number of points, thousands of times RESP's strength: buried atoms
went to ~0 and the REPEAT fit looked poor, RRMS 0.65. Corrected the same evening, before any production run.)

Input: the PBE0-D3(BJ)/SG15 potential of the crystal at a path point (qe.py esp: pp.x plot_num 11 = V_bare + V_H,
Ry, potential energy of an electron), so the electrostatic potential is phi = -V/2 Hartree/e.
Fit: phi(r_k) ~ sum_L q_L psi_L(r_k) + delta at the grid points r_k farther than SCALE x Bondi radius from every atom;
psi_L = tin-foil Ewald potential of unit charges on every site of CIF label L (symmetry copies share one charge), with
the neutralising background; delta = the free offset of a periodic potential.  Constraints: every molecule keeps its
formal charge (its labels' charges sum to it).  A weak restraint (weight W_RESTRAIN) towards zero keeps buried atoms
defined.  Reported: RRMS = |phi_fit - phi| / |phi - mean| over the fit points, RMS in volts.
"""
import argparse, gzip, json, math, os, sys
from pathlib import Path

import numpy as np

OPTC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = Path(os.environ.get('OPTC_RESULTS', OPTC / 'results'))
BOHR = 0.529177210903
HARTREE_V = 27.211386245988
BONDI = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85}
SCALE = 1.2
MAX_POINTS = 40000
W_RESTRAIN = 5e-4        # harmonic restraint towards 0 (Eh^2 per e^2, absolute), about RESP's a = 0.0005
ALPHA = 0.35            # Ewald splitting, 1/bohr
EPS_EWALD = 1e-9


def read_cube(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        f.readline(); f.readline()
        t = f.readline().split()
        nat, origin = int(t[0]), np.array([float(x) for x in t[1:4]])
        ax, n = [], []
        for _ in range(3):
            t = f.readline().split()
            n.append(int(t[0]))
            ax.append([float(x) for x in t[1:4]])
        atoms = []
        for _ in range(abs(nat)):
            t = f.readline().split()
            atoms.append((int(t[0]), np.array([float(x) for x in t[2:5]])))
        data = np.array(f.read().split(), dtype=float)
    data = data.reshape(n)
    return origin, np.array(ax), tuple(n), atoms, data


class Ewald:
    """Tin-foil Ewald potential (Hartree/e) of unit point charges with a neutralising background, in bohr."""

    def __init__(self, lat_bohr, alpha=ALPHA, eps=EPS_EWALD):
        self.lat = np.asarray(lat_bohr, float)
        self.V = abs(np.linalg.det(self.lat))
        self.a = alpha
        # real-space cutoff: erfc(a r)/r < eps
        r = 5.0
        while math.erfc(alpha * r) / r > eps:
            r += 0.5
        self.rc = r
        inv = np.linalg.inv(self.lat)
        self.rec = 2 * math.pi * inv.T          # rows b1, b2, b3
        gmax = 2 * alpha * math.sqrt(-math.log(eps))
        nmax = [int(math.ceil(gmax / np.linalg.norm(self.rec[i]))) for i in range(3)]
        G = []
        for i in range(-nmax[0], nmax[0] + 1):
            for j in range(-nmax[1], nmax[1] + 1):
                for k in range(-nmax[2], nmax[2] + 1):
                    if (i, j, k) == (0, 0, 0):
                        continue
                    # half space (G and -G give the same cosine)
                    if (i, j, k) < (0, 0, 0):
                        continue
                    g = i * self.rec[0] + j * self.rec[1] + k * self.rec[2]
                    g2 = g @ g
                    if g2 <= gmax * gmax:
                        G.append(g)
        self.G = np.array(G)
        g2 = np.einsum('ij,ij->i', self.G, self.G)
        self.Gw = 2 * (4 * math.pi / self.V) * np.exp(-g2 / (4 * alpha ** 2)) / g2      # x2: half space
        nimg = [int(math.ceil(self.rc / (self.V / np.linalg.norm(np.cross(self.lat[(i + 1) % 3], self.lat[(i + 2) % 3]))))) + 1
                for i in range(3)]
        self.shifts = np.array([[i, j, k] for i in range(-nimg[0], nimg[0] + 1) for j in range(-nimg[1], nimg[1] + 1)
                                for k in range(-nimg[2], nimg[2] + 1)]) @ self.lat
        self.bg = -math.pi / (alpha ** 2 * self.V)

    def basis(self, pts, sites, groups, ngroups, exclude=None, chunk=2000):
        """M[p, g] = potential at pts[p] of unit charges on all sites with groups == g.
        exclude: optional dict site -> list of image shift vectors (bohr) whose bare 1/r term is removed (QM atoms);
        for those images the real-space term becomes -erf(a d)/d (finite at d = 0)."""
        from scipy.special import erfc, erf
        pts = np.asarray(pts, float)
        M = np.zeros((len(pts), ngroups))
        for s0 in range(0, len(pts), chunk):
            P = pts[s0:s0 + chunk]
            # reciprocal part, all sites at once: sum_G w cos(G.(r - R))
            ph_r = P @ self.G.T                                  # (p, G)
            for j, R in enumerate(sites):
                ph = ph_r - (self.G @ R)[None, :]
                rec = np.cos(ph) @ self.Gw
                d = P[:, None, :] - (R[None, None, :] + self.shifts[None, :, :])
                r = np.linalg.norm(d, axis=2)
                with np.errstate(divide='ignore', invalid='ignore'):
                    real = np.where(r < self.rc, erfc(self.a * r) / r, 0.0)
                if exclude and j in exclude:
                    for t in exclude[j]:
                        k = int(np.argmin(np.linalg.norm(self.shifts - t, axis=1)))
                        rr = r[:, k]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            real[:, k] = np.where(rr > 1e-10, -erf(self.a * rr) / rr, -2 * self.a / math.sqrt(math.pi))
                M[s0:s0 + chunk, groups[j]] += rec + real.sum(axis=1) + self.bg
        return M


def fit_points(origin, ax, n, lat_bohr, sites_bohr, syms, scale=SCALE, max_points=MAX_POINTS, seed=7):
    idx = np.indices(n).reshape(3, -1).T
    pts = origin + idx @ ax
    inv = np.linalg.inv(lat_bohr)
    keep = np.ones(len(pts), bool)
    rad = np.array([BONDI[s] * scale / BOHR for s in syms])
    for j, R in enumerate(sites_bohr):
        df = (pts - R) @ inv
        df -= np.round(df)
        best = np.full(len(pts), np.inf)
        for s in np.ndindex(3, 3, 3):
            v = (df + np.array(s) - 1) @ lat_bohr
            best = np.minimum(best, np.einsum('ij,ij->i', v, v))
        keep &= best > rad[j] ** 2
    sel = np.where(keep)[0]
    rng = np.random.default_rng(seed)
    if len(sel) > max_points:
        sel = np.sort(rng.choice(sel, max_points, replace=False))
    return sel, pts[sel], int(keep.sum())


CHELPG_SPACING, CHELPG_RMAX, CHELPG_RESTRAIN = 0.3, 2.8, 5e-4
SC_TOL, SC_MAXCYC = 0.005, 6


def chelpg_points(xyz, syms, spacing=CHELPG_SPACING, rmax=CHELPG_RMAX):
    lo, hi = xyz.min(axis=0) - rmax, xyz.max(axis=0) + rmax
    g = [np.arange(lo[k], hi[k] + 1e-9, spacing) for k in range(3)]
    P = np.stack(np.meshgrid(*g, indexing='ij'), axis=-1).reshape(-1, 3)
    d = np.linalg.norm(P[:, None, :] - xyz[None, :, :], axis=2)
    rad = np.array([BONDI[s] for s in syms])
    keep = (d > rad[None, :]).all(axis=1) & (d.min(axis=1) <= rmax)
    return P[keep]


def esp_of(mol, dm, pts_A, chunk=1500):
    """potential (Hartree/e) of the molecule's nuclei and electron density at points (A)."""
    from pyscf import gto, df, lib
    pts = np.asarray(pts_A) / BOHR
    dm = np.asarray(dm)
    dmt = lib.pack_tril(dm + dm.T)
    dmt[lib.square_mat_in_trilu_indices(mol.nao).diagonal()] *= 0.5
    v = np.zeros(len(pts))
    for i0 in range(0, len(pts), chunk):
        fm = gto.fakemol_for_charges(pts[i0:i0 + chunk])
        j3c = df.incore.aux_e2(mol, fm, intor='int3c2e', aosym='s2ij')
        v[i0:i0 + chunk] = -dmt @ j3c
    R = mol.atom_coords()
    Z = mol.atom_charges()
    v += (Z[None, :] / np.linalg.norm(pts[:, None, :] - R[None, :, :], axis=2)).sum(axis=1)
    return v


def fit_chelpg(xyz, syms, pts, phi, total):
    d = np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=2) / BOHR
    A = 1.0 / d
    n = len(syms)
    H = A.T @ A
    w = CHELPG_RESTRAIN
    for i, s in enumerate(syms):
        if s != 'H':
            H[i, i] += w
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = H
    K[:n, n] = K[n, :n] = 1.0
    q = np.linalg.solve(K, np.concatenate([A.T @ phi, [total]]))[:n]
    res = A @ q - phi
    return q, float(np.sqrt((res ** 2).sum() / (phi ** 2).sum()))


def stage_sc(a):
    """self-consistent in-crystal CHELPG charges at one path point (see the module docstring)."""
    import embed as E
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mol import mk_mol, run_dft
    from pyscf import lib
    lib.num_threads(a.threads)
    state = 'S' if a.point == 'P0' else 'N'
    cr = E.Crystal(a.key, state)
    _, _, X = E.read_extxyz(RES / a.key / f'{a.point}.extxyz')
    W = cr.whole(X)
    labels = cr.rec['labels']
    # unique molecule types: the label tuple of each molecule
    types = {}
    for k, m in enumerate(cr.mols):
        types.setdefault(tuple(labels[i] for i in m), []).append(k)
    q = np.zeros(cr.nat)
    log = {'key': a.key, 'point': a.point, 'state': state, 'cycles': [], 'types': [list(t) for t in types]}
    # gas phase start (and the second charge model)
    gas = np.zeros(cr.nat)
    dms = {}
    for t, ks in types.items():
        k = ks[0]
        xyz = E.inst_xyz(cr, W, (k, (0, 0, 0)))
        sy = [cr.syms[i] for i in cr.mols[k]]
        mol = mk_mol(sy, xyz, cr.charges[k], a.mem)
        mf, e, ex = run_dft(mol, 'pbe0')
        pts = chelpg_points(xyz, sy)
        qk, rr = fit_chelpg(xyz, sy, pts, esp_of(mol, mf.make_rdm1(), pts), cr.charges[k])
        dms[t] = mf.make_rdm1()
        for kk in ks:
            for i, j in enumerate(cr.mols[kk]):
                gas[j] = qk[cr.mols[k].index(cr.mols[k][i])] if False else qk[i]
        log.setdefault('gas', {})['/'.join(t[:3]) + '...'] = {'rrms': round(rr, 4), 'n_points': int(len(pts)), 'E': e}
    # copies share label charges: assign by label (molecules of one type list atoms in the same label order)
    def by_label(qvec):
        lab = {}
        for i, L in enumerate(labels):
            lab.setdefault(L, []).append(qvec[i])
        return {L: float(np.mean(v)) for L, v in lab.items()}
    gl = by_label(gas)
    q = np.array([gl[L] for L in labels])
    out_gas = {'key': a.key, 'point': a.point, 'method': 'gas-phase CHELPG of each molecule (PBE0-D3(BJ)/def2-TZVP)',
               'charges': [round(float(x), 6) for x in q], 'charges_by_label': {L: round(v, 5) for L, v in gl.items()}}
    (RES / a.key / f'charges_{a.point}_gas.json').write_text(json.dumps(out_gas, indent=1))
    center_all = X.mean(axis=0)
    for cyc in range(1, SC_MAXCYC + 1):
        newq = q.copy()
        crow = {'cycle': cyc, 'types': {}}
        for t, ks in types.items():
            k = ks[0]
            inst = [(k, (0, 0, 0))]
            xyz = E.inst_xyz(cr, W, inst[0])
            arrays, info = E.build_one(cr, X, q, inst, W, xyz.mean(axis=0), rc=a.rc)
            sy = [cr.syms[i] for i in cr.mols[k]]
            mol = mk_mol(sy, xyz, cr.charges[k], a.mem)
            mm = (np.vstack([arrays['mm_xyz'], arrays['b_xyz']]), np.concatenate([arrays['mm_q'], arrays['b_q']]),
                  np.concatenate([arrays['mm_radii'], np.full(len(arrays['b_q']), 0.3)]))
            mf, e, ex = run_dft(mol, 'pbe0', mm=mm, dm0=dms.get(t))
            dms[t] = mf.make_rdm1()
            pts = chelpg_points(xyz, sy)
            qk, rr = fit_chelpg(xyz, sy, pts, esp_of(mol, dms[t], pts), cr.charges[k])
            for kk in ks:
                for i, j in enumerate(cr.mols[kk]):
                    newq[j] = qk[i]
            crow['types']['/'.join(t[:3]) + '...'] = {'rrms': round(rr, 4), 'E': e, 'converged': ex['converged'],
                                                      'boundary_fit_rms_V': info['fit_rms_V'], 'n_mm': info['n_mm']}
        nl = by_label(newq)
        newq = np.array([nl[L] for L in labels])
        dq = float(np.abs(newq - q).max())
        crow['max_dq'] = round(dq, 5)
        log['cycles'].append(crow)
        q = newq
        print(f'[optc.charges] {a.key} {a.point} SC cycle {cyc}: max dq {dq:.4f}', flush=True)
        if dq < SC_TOL:
            break
    lab = by_label(q)
    out = {'key': a.key, 'point': a.point, 'state': state,
           'method': 'self-consistent in-crystal CHELPG (PBE0-D3(BJ)/def2-TZVP in the Ewald-consistent field of all other '
                     'molecules), formal molecular charges fixed, weak heavy-atom restraint',
           'converged': log['cycles'][-1]['max_dq'] < SC_TOL, 'n_cycles': len(log['cycles']),
           'charges': [round(float(x), 6) for x in q], 'charges_by_label': {L: round(v, 5) for L, v in lab.items()},
           'molecule_charges': {str(k): round(float(q[m].sum()), 5) for k, m in enumerate(cr.mols)}, 'log': log}
    (RES / a.key / f'charges_{a.point}_sc.json').write_text(json.dumps(out, indent=1))


def stage_fit(a):
    import embed as E
    crs = E.Crystal(a.key, 'S' if a.point == 'P0' else 'N')
    rec = json.loads((OPTC / 'structures' / a.key / 'cell.json').read_text())
    labels = rec['labels']
    # molecule types of the state (symmetry copies share their label set; at P4 the moved H belongs to the acceptor)
    rec['molecules'] = [{'atoms': m, 'formula': '+'.join(sorted(labels[i] for i in m)), 'charge': crs.charges[k]}
                        for k, m in enumerate(crs.mols)]
    syms = rec['species']
    cube = RES / a.key / 'runs' / f'esp_{a.point}' / f'vtot_{a.point}.cube.gz'
    origin, ax, n, atoms, V = read_cube(cube)
    lat_b = np.array(ax) * np.array(n)[:, None]           # cube spans the cell
    sites = np.array([x for _, x in atoms])
    if len(sites) != len(syms):
        raise SystemExit('cube atoms != cell atoms')
    lat_cell = np.array(rec['lattice_A']) / BOHR
    if np.abs(lat_b - lat_cell).max() > 1e-3:
        raise SystemExit(f'cube cell differs from the structure cell: {np.abs(lat_b - lat_cell).max():.2e} bohr')
    phi = -V.reshape(-1) / 2.0                             # Hartree/e
    sel, P, nfree = fit_points(origin, ax, n, lat_b, sites, syms, scale=a.scale)
    if len(sel) < 2000:
        raise SystemExit(f'only {len(sel)} fit points outside {a.scale} x Bondi radii')
    b = phi[sel]
    lab_list = sorted(set(labels), key=labels.index)
    g = np.array([lab_list.index(l) for l in labels])
    ew = Ewald(lat_b)
    M = ew.basis(P, sites, g, len(lab_list))
    # constraints: per molecule species, sum of its labels' charges = formal charge
    species_of_label = {}
    for m in rec['molecules']:
        for i in m['atoms']:
            species_of_label[labels[i]] = (m['formula'], m['charge'])
    forms = sorted(set(species_of_label.values()))
    C = np.zeros((len(forms), len(lab_list)))
    d = np.zeros(len(forms))
    for k, (f, q) in enumerate(forms):
        for L, (f2, q2) in species_of_label.items():
            if f2 == f:
                C[k, lab_list.index(L)] = 1.0
        d[k] = q
    # unknowns: q (nL) and delta; minimise |M q + delta - b|^2 + w |q|^2 subject to C q = d
    nL = len(lab_list)
    A = np.hstack([M, np.ones((len(b), 1))])
    w = W_RESTRAIN
    H = A.T @ A
    H[:nL, :nL] += w * np.eye(nL)
    rhs = A.T @ b
    K = np.zeros((nL + 1 + len(forms),) * 2)
    K[:nL + 1, :nL + 1] = H
    K[:nL, nL + 1:] = C.T
    K[nL + 1:, :nL] = C
    sol = np.linalg.solve(K, np.concatenate([rhs, d]))
    q, delta = sol[:nL], sol[nL]
    res = A @ sol[:nL + 1] - b
    rrms = float(np.sqrt((res ** 2).sum() / ((b - b.mean()) ** 2).sum()))
    out = {'key': a.key, 'point': a.point, 'method': 'REPEAT-type periodic ESP fit (tin-foil Ewald), labels share charges, '
           'molecular formal charges constrained', 'source_cube': str(cube.relative_to(RES)), 'scale_bondi': a.scale,
           'n_fit_points': int(len(sel)), 'n_free_points': nfree, 'restraint_weight': W_RESTRAIN,
           'rrms': round(rrms, 5), 'rms_V': round(float(np.sqrt((res ** 2).mean())) * HARTREE_V, 5),
           'offset_V': round(float(delta) * HARTREE_V, 5),
           'charges_by_label': {L: round(float(x), 5) for L, x in zip(lab_list, q)},
           'charges': [round(float(q[g[i]]), 6) for i in range(len(labels))],
           'molecule_sums': {f'{f} ({c:+d})': round(float(sum(q[lab_list.index(L)] for L, (f2, _) in species_of_label.items()
                                                              if f2 == f)), 6) for f, c in forms}}
    (RES / a.key / f'charges_{a.point}.json').write_text(json.dumps(out, indent=1))
    print(f"[optc.charges] {a.key} {a.point}: {len(sel)} points, RRMS {rrms:.4f}, RMS {out['rms_V']:.4f} V, "
          f"offset {out['offset_V']:.3f} V")


def selftest():
    """Ewald basis against a brute-force spherical sum for a neutral pair of charges in a skewed cell, and alpha
    independence."""
    lat = np.array([[9.0, 0, 0], [2.0, 11.0, 0], [-1.5, 1.0, 13.0]])
    sites = np.array([[1.0, 2.0, 3.0], [4.5, 6.0, 7.5]])
    q = np.array([1.0, -1.0])
    pts = np.array([[3.0, 3.0, 3.0], [6.0, 1.0, 9.0], [0.5, 8.0, 2.0]])
    g = np.array([0, 1])
    v = {}
    for al in (0.25, 0.35, 0.5):
        v[al] = Ewald(lat, alpha=al).basis(pts, sites, g, 2) @ q
    # brute force: neutral cell (dipolar), spherical summation converges to the Ewald value + a shape term;
    # compare instead a quadrupolar-free check: alpha independence to 1e-7 Hartree
    spread = max(np.abs(v[0.25] - v[0.5]).max(), np.abs(v[0.35] - v[0.5]).max())
    print('alpha independence (Hartree):', spread)
    # the tin-foil potential of a cell with zero dipole equals the large-sphere direct sum
    sites2 = np.array([[1.0, 2.0, 3.0], [4.5, 6.0, 7.5], [8.0, 10.0, 12.0]])
    q2 = np.array([1.0, -2.0, 1.0]) * 0 + np.array([1.0, -2.0, 1.0])
    # make the dipole vanish: put the +1 charges symmetric about the -2 one
    sites2[2] = 2 * sites2[1] - sites2[0]
    ew = Ewald(lat).basis(pts, sites2, np.array([0, 1, 2]), 3) @ q2
    direct = np.zeros(len(pts))
    N = 14
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            for k in range(-N, N + 1):
                t = np.array([i, j, k]) @ lat
                if np.linalg.norm(t) > N * 8.5:
                    continue
                for s, qq in zip(sites2, q2):
                    direct += qq / np.linalg.norm(pts - s - t, axis=1)
    dd = (ew - direct) - (ew - direct).mean()      # the spherical sum carries a constant (quadrupole) offset
    print('tin-foil vs spherical direct sum, zero-dipole cell, up to a constant (Hartree):', np.abs(dd).max())
    ok = spread < 1e-6 and np.abs(dd).max() < 1e-4
    print('selftest', 'OK' if ok else 'FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['fit', 'sc', 'selftest'])
    ap.add_argument('--key')
    ap.add_argument('--point', default='P0')
    ap.add_argument('--scale', type=float, default=SCALE)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('DFT_THREADS', 8)))
    ap.add_argument('--mem', type=int, default=int(os.environ.get('DFT_MAX_MEMORY_MB', 16000)))
    ap.add_argument('--rc', type=float, default=25.0)
    a = ap.parse_args()
    if a.stage == 'selftest':
        sys.exit(0 if selftest() else 1)
    {'fit': stage_fit, 'sc': stage_sc}[a.stage](a)


if __name__ == '__main__':
    main()
