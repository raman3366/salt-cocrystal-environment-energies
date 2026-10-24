#!/usr/bin/env python3
"""QM regions and Ewald-consistent point-charge embedding for the molecular rungs (optc/PLAN.md).

    tools/env/bin/python optc/src/embed.py build --key KEY [--model repeat|sc|gas] [--rc 25]
        -> results/<key>/emb/<model>/{S,T}_{pair,cluster}.npz, N_pair.npz, emb.json
    tools/env/bin/python optc/src/embed.py build --key KEY --site --tag site --configs N_cluster   # one config, merged
    tools/env/bin/python optc/src/embed.py shells --key KEY        # transfer-site shell membership table (no QM)

Additions of 19 Oct 2026, 9:40 PM IST (optc/PLAN.md items B, M, O): N_cluster (pair + shell, every atom at P4, P4
charges; the environment of the reverse gap at the site level); --site-sym (the site also holds the H atoms on A, so
that the cocrystals' acceptor-side contacts count); --hb-ha (H...A cutoff, default 2.5 A); --smear-radius (radius of
the Gaussian MM charges within 4 A of the QM region, default 0.8 A).

Configurations (all from the periodic run; cell fixed):
  S  the observed state: every atom at P0, charges fitted at P0 (charges_P0.json)
  T  single-site transfer: the central pair at P4 (aligned to P0 by the heavy-atom translation that best overlays the
     two cells), everything else at P0 with the P0 charges (environment frozen in the observed state)
  N  collective transfer: every atom at P4, charges fitted at P4 (field rung only; the energy per transferring pair is
     E_emb - E_int / 2, see mol.py)
QM regions: "pair" = the donor and acceptor molecules of the first copy of the transferring bond (whole molecules);
"cluster" = the pair + every whole molecule that makes a hydrogen bond with it at P0 (H...A <= 2.5 A, angle >= 120 deg,
A in N, O), the same molecules in S and T.
MM: every whole molecule (centroid within RC of the QM centroid) not in the QM region, with its fitted charges;
atoms within 4 A of a QM atom carry a Gaussian distribution of radius 0.8 A, every other charge one of 0.3 A (PySCF's
Gaussian charge model for all; beyond ~1 A it acts as a point charge).
MM units: whole neutral formula units (each copy's donor and acceptor molecules), so the MM + QM charge is zero.
Boundary: NB charges on a sphere RC + 6 A, fitted (least squares, total charge neutral, plus a free constant that is
not applied) so that MM + boundary charges reproduce, at the QM atoms and at 14 points 1 A around each, the tin-foil
Ewald potential of every charge of the crystal minus the QM region's own (point-charge) contribution, up to that
constant.  Reported: RMS and max error (V) of the fit and the constant.
"""
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from charges import Ewald, BOHR, HARTREE_V  # noqa: E402

OPTC = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('OPTC_RESULTS', OPTC / 'results'))
RCOV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20}
RC, NB, RB_EXTRA = 25.0, 600, 6.0
SMEAR_WITHIN, SMEAR_RADIUS = 4.0, 0.8
POINT_RADIUS = 0.3          # every other MM and boundary charge: a narrow Gaussian (= a point charge beyond ~1 A)
HB_HA, HB_ANG = 2.5, 120.0


def read_extxyz(path):
    L = Path(path).read_text().splitlines()
    n = int(L[0])
    lat = np.array([float(x) for x in L[1].split('Lattice="')[1].split('"')[0].split()]).reshape(3, 3)
    syms, X = [], []
    for l in L[2:2 + n]:
        p = l.split()
        syms.append(p[0])
        X.append([float(v) for v in p[1:4]])
    return lat, syms, np.array(X)


class Crystal:
    """state 'S': molecules as in the CIF; state 'N': every copy's transferring H moved from the donor molecule to the
    acceptor molecule (the composition of the collective end point P4)."""

    def __init__(self, key, state='S'):
        self.key = key
        self.state = state
        self.rec = json.loads((OPTC / 'structures' / key / 'cell.json').read_text())
        self.lat = np.array(self.rec['lattice_A'])
        self.inv = np.linalg.inv(self.lat)
        self.syms = self.rec['species']
        self.nat = len(self.syms)
        self.mols = [list(m['atoms']) for m in self.rec['molecules']]
        self.charges = [m['charge'] for m in self.rec['molecules']]
        mol_of = {i: k for k, m in enumerate(self.mols) for i in m}
        self.moved = {}
        if state == 'N':
            for c in self.rec['copies']:
                kd, ka = mol_of[c['D']], mol_of[c['A']]
                self.mols[kd].remove(c['H'])
                self.mols[ka].append(c['H'])
                self.charges[kd] -= 1
                self.charges[ka] += 1
                self.moved[c['H']] = c['A']
        self.mols = [sorted(m) for m in self.mols]
        self.mol_of = {i: k for k, m in enumerate(self.mols) for i in m}
        self.mol_of_S = mol_of
        X0 = np.array(self.rec['frac']) @ self.lat
        self.bonds = self._bonds(X0)

    def mic(self, X, i, j):
        df = (X[j] - X[i]) @ self.inv
        df -= np.round(df)
        best = None
        for s in np.ndindex(3, 3, 3):
            v = (df + np.array(s) - 1) @ self.lat
            if best is None or v @ v < best @ best:
                best = v
        return best

    def _bonds(self, X):
        """S-topology bonds (within molecules), each H on its nearest heavy atom."""
        adj = {i: set() for i in range(self.nat)}
        for m in self.mols:
            heavy = [i for i in m if self.syms[i] != 'H']
            for a in range(len(heavy)):
                for b in range(a + 1, len(heavy)):
                    i, j = heavy[a], heavy[b]
                    if np.linalg.norm(self.mic(X, i, j)) < RCOV[self.syms[i]] + RCOV[self.syms[j]] + 0.4:
                        adj[i].add(j); adj[j].add(i)
            for h in m:
                if self.syms[h] == 'H':
                    x = self.moved.get(h) or min(heavy, key=lambda j: np.linalg.norm(self.mic(X, h, j)))
                    adj[h].add(x); adj[x].add(h)
        return adj

    def whole(self, X):
        """cartesian positions with every molecule unwrapped along its S-topology bonds (root: its first atom, wrapped
        into the cell)."""
        W = np.zeros_like(X)
        F = (X @ self.inv) % 1.0
        Xw = F @ self.lat
        for m in self.mols:
            r = m[0]
            W[r] = Xw[r]
            seen, todo = {r}, [r]
            while todo:
                i = todo.pop()
                for j in self.bonds[i]:
                    if j not in seen:
                        W[j] = W[i] + self.mic(X, i, j)
                        seen.add(j); todo.append(j)
            if len(seen) != len(m):
                raise SystemExit(f'{self.key}: molecule with root {r} not connected')
        return W


def whole_like(cr, X, Wref):
    """X unwrapped atom by atom onto the images of Wref (the same molecule instances; atoms move < half a cell)."""
    W = np.zeros_like(X)
    for i in range(cr.nat):
        df = (X[i] - Wref[i]) @ cr.inv
        df -= np.round(df)
        W[i] = Wref[i] + df @ cr.lat
    return W


def align_translation(cr, XA, XB):
    """translation t (A) minimising the heavy-atom RMSD between XB + t and XA under the minimum image."""
    heavy = [i for i, s in enumerate(cr.syms) if s != 'H']
    t = np.zeros(3)
    for _ in range(5):
        d = np.array([cr.mic(np.vstack([XA[i], XB[i] + t]), 0, 1) for i in heavy])
        t = t - d.mean(axis=0)
    rms = float(np.sqrt(np.mean([np.linalg.norm(cr.mic(np.vstack([XA[i], XB[i] + t]), 0, 1)) ** 2 for i in heavy])))
    return t, rms


def units(cr, W):
    """neutral formula units: (donor molecule, acceptor molecule, shift of the acceptor) for every copy of the
    transferring bond (each molecule belongs to exactly one copy in these 1:1 crystals); molecules of no copy stay
    single (then only neutral molecules are allowed)."""
    out, used = [], set()
    for c in cr.rec['copies']:
        kD, kA = cr.mol_of_S[c['D']], cr.mol_of_S[c['A']]
        best = min(((np.linalg.norm(W[c['A']] + (np.array(s) - 2) @ cr.lat - W[c['D']]), tuple(int(x) for x in np.array(s) - 2))
                    for s in np.ndindex(5, 5, 5)))
        out.append(((kD, (0, 0, 0)), (kA, best[1])))
        used |= {kD, kA}
    for k in range(len(cr.mols)):
        if k not in used:
            if cr.charges[k] != 0:
                raise SystemExit(f'{cr.key}: charged molecule {k} outside every formula unit')
            out.append(((k, (0, 0, 0)),))
    return out


def instances(cr, W, center, rc):
    """molecule instances (molecule, shift) of every neutral formula unit whose centroid lies within rc of center;
    shift = integer lattice vector."""
    out = []
    reach = [int(math.ceil((rc + 15) * np.linalg.norm(cr.inv[:, k]))) + 1 for k in range(3)]
    for u in units(cr, W):
        members = [(k, np.array(n)) for k, n in u]
        c0 = np.vstack([W[cr.mols[k]] + n @ cr.lat for k, n in members]).mean(axis=0)
        for s in np.ndindex(*(2 * r + 1 for r in reach)):
            n = np.array(s) - np.array(reach)
            if np.linalg.norm(c0 + n @ cr.lat - center) <= rc:
                out += [(k, tuple(int(x) for x in m + n)) for k, m in members]
    return out


def inst_xyz(cr, W, inst):
    k, n = inst
    return W[cr.mols[k]] + np.array(n) @ cr.lat


def image_nearest(cr, W, A, X):
    """lattice shift of the instance of A's molecule whose A is nearest atom X (whole positions W)."""
    return min(((np.linalg.norm(W[A] + (np.array(s) - 2) @ cr.lat - W[X]), tuple(int(x) for x in np.array(s) - 2))
                for s in np.ndindex(5, 5, 5)))[1]


def hbond_partners(cr, W, qm_insts, cands, site=None):
    """molecule instances among cands with an H-bond to any atom of the qm instances (site: only to these atoms)."""
    qm_atoms = [(i, inst_xyz(cr, W, q)[cr.mols[q[0]].index(i)]) for q in qm_insts for i in cr.mols[q[0]]
                if site is None or i in site]
    found = []
    for c in cands:
        if c in qm_insts:
            continue
        Xc = inst_xyz(cr, W, c)
        ok = False
        for ia, (i, xi) in enumerate(qm_atoms):
            for jb, j in enumerate(cr.mols[c[0]]):
                xj = Xc[jb]
                # H on the QM side, acceptor in the candidate; or the reverse
                for (h, xh, a, xa, own) in ((i, xi, j, xj, 'qm'), (j, xj, i, xi, 'mm')):
                    if cr.syms[h] != 'H' or cr.syms[a] not in ('N', 'O'):
                        continue
                    if np.linalg.norm(xh - xa) > HB_HA:
                        continue
                    d = next(iter(cr.bonds[h]))
                    if own == 'qm':
                        q = next(q for q in qm_insts if h in cr.mols[q[0]])
                        xd = inst_xyz(cr, W, q)[cr.mols[q[0]].index(d)]
                    else:
                        xd = Xc[cr.mols[c[0]].index(d)]
                    v1, v2 = xd - xh, xa - xh
                    ang = math.degrees(math.acos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
                    if ang >= HB_ANG:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if ok:
            found.append(c)
    return found


def fibonacci_sphere(n, r, c):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return c + r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def sample_points(Xqm, rad=1.0):
    d = fibonacci_sphere(14, rad, np.zeros(3))
    return np.vstack([Xqm] + [Xqm + v for v in d])


def build_one(cr, X, charges, qm_insts, W, center, rc=RC, nb=NB, tag=''):
    """MM charges + boundary charges for one QM region at one configuration; returns arrays and the fit quality."""
    q_atom = np.asarray(charges)
    qm_xyz = np.vstack([inst_xyz(cr, W, q) for q in qm_insts])
    qm_idx = [i for q in qm_insts for i in cr.mols[q[0]]]
    q_qm = q_atom[qm_idx]
    allinst = instances(cr, W, center, rc)
    mm_insts = [c for c in allinst if c not in qm_insts]
    mm_xyz = np.vstack([inst_xyz(cr, W, c) for c in mm_insts])
    mm_q = np.concatenate([q_atom[cr.mols[c[0]]] for c in mm_insts])
    dmin = np.min(np.linalg.norm(mm_xyz[:, None, :] - qm_xyz[None, :, :], axis=2), axis=1)
    radii = np.where(dmin < SMEAR_WITHIN, SMEAR_RADIUS, POINT_RADIUS)
    # target: Ewald potential of every charge of the crystal minus the QM instances' own contribution
    P = sample_points(qm_xyz)
    ew = Ewald(cr.lat / BOHR)
    # sites: atoms of the reference cell at X (wrapped); the QM instances are particular images of them
    Xw = ((X @ cr.inv) % 1.0) @ cr.lat
    exclude = {}
    for q in qm_insts:
        k, n = q
        for a, xa in zip(cr.mols[k], inst_xyz(cr, W, q)):
            exclude.setdefault(a, []).append((xa - Xw[a]) / BOHR)
    M = ew.basis(P / BOHR, Xw / BOHR, np.arange(cr.nat), cr.nat, exclude=exclude)
    target = M @ q_atom                                      # Hartree/e
    d = np.linalg.norm(P[:, None, :] - mm_xyz[None, :, :], axis=2) / BOHR
    phi_mm = (mm_q[None, :] / d).sum(axis=1)
    Bp = fibonacci_sphere(nb, rc + RB_EXTRA, center)
    B = 1.0 / (np.linalg.norm(P[:, None, :] - Bp[None, :, :], axis=2) / BOHR)
    tot = -(mm_q.sum() + q_qm.sum())
    # unknowns: the boundary charges b and a constant c0 (fitted, never applied: a uniform potential over the QM region
    # leaves every energy difference of this work unchanged, the QM region keeping its charge between configurations)
    Bc = np.hstack([B, np.ones((len(P), 1))])
    mu = 1e-6 * np.trace(B.T @ B) / nb
    K = np.zeros((nb + 2, nb + 2))
    K[:nb + 1, :nb + 1] = Bc.T @ Bc
    K[:nb, :nb] += mu * np.eye(nb)
    K[:nb, nb + 1] = K[nb + 1, :nb] = 1.0
    rhs = np.concatenate([Bc.T @ (target - phi_mm), [tot]])
    sol = np.linalg.solve(K, rhs)
    b, c0 = sol[:nb], sol[nb]
    resid = (phi_mm + B @ b + c0 - target) * HARTREE_V
    resid_raw = (phi_mm - target) * HARTREE_V
    # the potential of the narrow Gaussians is that of point charges at the sample points (d >= 1 A); the 0.8 A ones
    # differ from point charges by erfc(d / 0.8 A) / d, shown here and left in the QM run on purpose
    nq = len(qm_xyz)
    info = {'n_mm': int(len(mm_q)), 'n_mm_smeared': int((radii >= SMEAR_RADIUS).sum()), 'n_boundary': nb,
            'boundary_radius_A': rc + RB_EXTRA, 'sum_mm_charge': round(float(mm_q.sum()), 4),
            'qm_charge_from_fit': round(float(q_qm.sum()), 4),
            'fit_rms_V': round(float(np.sqrt((resid ** 2).mean())), 5), 'fit_max_V': round(float(np.abs(resid).max()), 5),
            'fit_rms_V_at_atoms': round(float(np.sqrt((resid[:nq] ** 2).mean())), 5),
            'constant_offset_V': round(float(c0) * HARTREE_V, 4),
            'without_boundary_rms_V': round(float(np.sqrt(((resid_raw - resid_raw.mean()) ** 2).mean())), 4),
            'without_boundary_mean_V': round(float(resid_raw.mean()), 4)}
    arrays = {'mm_xyz': mm_xyz, 'mm_q': mm_q, 'mm_radii': radii, 'b_xyz': Bp, 'b_q': b,
              'target_at_qm_atoms_V': target[:nq] * HARTREE_V}
    return arrays, info


SUFFIX = {'repeat': '', 'sc': '_sc', 'gas': '_gas'}


def site_atoms(cr, c, sym=False):
    """the transfer site: D, A and the H atoms on D (and with sym the H atoms on A)."""
    site = {c['D'], c['A']} | {h for h in cr.bonds[c['D']] if cr.syms[h] == 'H'}
    if sym:
        site |= {h for h in cr.bonds[c['A']] if cr.syms[h] == 'H'}
    return site


# pair configurations at P4r and at the relaxed-cell points, built only when named in --configs (results audit, 21 Oct
# 2026: the in-field pair corrections of the cocrystals at P4r and in their relaxed cells).  Each corrects the pair
# geometry of the vacuum correction at the same point in pair.json (P4r, vcS, vcP4r), so the two compare directly.
POINT_CONFIGS = {
    'R_pair': 'the P4r pair (aligned on P0) in the P0 environment with the P0 charges: single site at P4r, as T at P4',
    'NR_pair': 'every atom at P4r (aligned on P0), the N composition, the P4r charges: the P4r pair in its own field, '
               'as N at P4 (its pair geometry is pair.json P4r)',
    'VS_pair': 'every atom at vcS in the relaxed S lattice (own Ewald sum), the P0 charges fitted in the X-ray cell: '
               'the vcS pair in its own field (its pair geometry is pair.json vcS)',
    'VR_pair': 'every atom at vcP4r in the relaxed P4r lattice (own Ewald sum), the N composition, the P4r charges '
               'fitted in the X-ray cell: the vcP4r pair in its own field (its pair geometry is pair.json vcP4r)',
}


def transferred(cr, X):
    """every copy's proton nearer its acceptor than its donor (minimum image)."""
    return all(np.linalg.norm(cr.mic(X, c['A'], c['H'])) < np.linalg.norm(cr.mic(X, c['D'], c['H']))
               for c in cr.rec['copies'])


def point_sets(a, cr, crN, X0, W0, pair, center, q0, names):
    """the POINT_CONFIGS among names: (X, W env, charges, QM instances, W QM, crystal, centre) as in stage_build."""
    import copy
    out = {}
    q4r = None
    if {'R_pair', 'NR_pair', 'VR_pair'} & set(names):
        q4r = json.loads((RES / a.key / f'charges_P4r{SUFFIX[a.model]}.json').read_text())['charges']
    if {'R_pair', 'NR_pair'} & set(names):
        _, _, XR = read_extxyz(RES / a.key / 'P4r.extxyz')
        if not transferred(cr, XR):
            raise SystemExit(f'{a.key}: the proton is not transferred at P4r; R_pair/NR_pair are not defined')
        t, rms = align_translation(cr, X0, XR)
        WR = whole_like(cr, XR + t, W0)
        WRN = WR.copy()
        for h, acc in crN.moved.items():
            WRN[h] = WRN[acc] + crN.mic(XR + t, acc, h)
        out['R_pair'] = (X0, W0, q0, pair, WR, cr, center)
        out['NR_pair'] = (XR + t, WRN, q4r, pair, WRN, crN, center)
    c = cr.rec['copies'][0]
    for name, pt, q, base in (('VS_pair', 'vcS', q0, cr), ('VR_pair', 'vcP4r', q4r, crN)):
        if name not in names:
            continue
        lat, _, XV = read_extxyz(RES / a.key / f'{pt}.extxyz')
        cs, cv = copy.copy(cr), copy.copy(base)
        cs.lat = cv.lat = np.array(lat)
        cs.inv = cv.inv = np.linalg.inv(cs.lat)
        WV = cs.whole(XV)
        # the acceptor instance bonded through H, in the relaxed lattice
        pv = [(pair[0][0], (0, 0, 0)), (pair[1][0], image_nearest(cs, WV, c['A'], c['H']))]
        if base is crN:
            if not transferred(cs, XV):
                raise SystemExit(f'{a.key}: the proton is not transferred at {pt}; {name} is not defined')
            for h, acc in crN.moved.items():
                WV[h] = WV[acc] + cv.mic(XV, acc, h)
        out[name] = (XV, WV, q, pv, WV, cv, np.vstack([inst_xyz(cv, WV, x) for x in pv]).mean(axis=0))
    return {k: v for k, v in out.items() if k in names}


def stage_build(a):
    global SMEAR_RADIUS, HB_HA
    SMEAR_RADIUS, HB_HA = a.smear_radius, a.hb_ha
    cr = Crystal(a.key)
    out = RES / a.key / 'emb' / (a.model + (f'_{a.tag}' if a.tag else ''))
    out.mkdir(parents=True, exist_ok=True)
    _, _, X0 = read_extxyz(RES / a.key / 'P0.extxyz')
    _, _, X4 = read_extxyz(RES / a.key / 'P4.extxyz')
    q0 = json.loads((RES / a.key / f'charges_P0{SUFFIX[a.model]}.json').read_text())['charges']
    q4 = json.loads((RES / a.key / f'charges_P4{SUFFIX[a.model]}.json').read_text())['charges']
    W0 = cr.whole(X0)
    c = cr.rec['copies'][0]
    D, H, A = c['D'], c['H'], c['A']
    mD, mA = cr.mol_of[D], cr.mol_of[A]
    # the acceptor instance closest to the donor atom
    best = None
    for s in np.ndindex(5, 5, 5):
        n = np.array(s) - 2
        dd = np.linalg.norm(W0[A] + n @ cr.lat - W0[D])
        if best is None or dd < best[0]:
            best = (dd, tuple(int(x) for x in n))
    # shift of the acceptor instance relative to the acceptor molecule's whole positions
    pair = [(mD, (0, 0, 0)), (mA, best[1])]
    Xp0 = np.vstack([inst_xyz(cr, W0, q) for q in pair])
    center = Xp0.mean(axis=0)
    cands = instances(cr, W0, center, 14.0)
    if a.site:
        # transfer-site shell (optc/PLAN.md, 19 Oct 2026 8:26 AM IST): molecules hydrogen-bonded to the donor atom D,
        # the hydrogens on D, or the acceptor atom A of the first copy of the transferring bond
        shell = hbond_partners(cr, W0, pair, cands, site=site_atoms(cr, c, a.site_sym))
    else:
        shell = hbond_partners(cr, W0, pair, cands)
    if a.nshell:
        # basis check on a reduced cluster: the a.nshell shell molecules nearest the pair (centroid distance)
        shell = sorted(shell, key=lambda q: np.linalg.norm(inst_xyz(cr, W0, q).mean(axis=0) - center))[:a.nshell]
    cluster = pair + shell
    # T: the P4 pair aligned on P0, environment at P0
    t, rms = align_translation(cr, X0, X4)
    W4 = whole_like(cr, X4 + t, W0)
    # N: every atom at P4 in the N composition; a moved H is unwrapped next to its new partner atom
    crN = Crystal(a.key, 'N')
    W4N = W4.copy()
    for h, acc in crN.moved.items():
        W4N[h] = W4N[acc] + crN.mic(X4 + t, acc, h)
    # the pair in the N composition: the same two molecules (the acceptor instance shift is unchanged)
    # N: every atom at P4 (aligned); the instance shifts are those of P0 (same molecules, the unwrap keeps them)
    rec = {'key': a.key, 'model': a.model, 'pair': pair, 'shell': shell, 'n_shell': len(shell),
           'align_P4_on_P0_A': t.round(5).tolist(),
           'align_rms_A': round(rms, 4), 'rc_A': a.rc, 'smear_radius_A': SMEAR_RADIUS, 'hb_ha_A': HB_HA,
           'site_sym': bool(a.site_sym), 'configs': {}}
    sets = {
        'S_pair': (X0, W0, q0, pair, W0), 'S_cluster': (X0, W0, q0, cluster, W0),
        'T_pair': (X0, W0, q0, pair, W4), 'T_cluster': (X0, W0, q0, cluster, W4),
        'N_pair': (X4 + t, W4N, q4, pair, W4N), 'N_cluster': (X4 + t, W4N, q4, cluster, W4N),
    }
    asked = [n for n in a.configs.split(',') if n in POINT_CONFIGS] if a.configs else []
    if asked:
        sets.update(point_sets(a, cr, crN, X0, W0, pair, center, q0, asked))
    if a.configs:
        old = json.loads((out / 'emb.json').read_text())
        if old['pair'] != json.loads(json.dumps(pair)) or old['shell'] != json.loads(json.dumps(shell)):
            raise SystemExit(f'{a.key}: the QM region differs from the existing {out.name}/emb.json; nothing written')
        rec['configs'] = old['configs']
        sets = {k: v for k, v in sets.items() if k in a.configs.split(',')}
    elif not a.site:
        sets.pop('N_cluster')
    for name, v in sets.items():
        X, Wenv, q, qm, Wqm = v[:5]
        crs, cen = v[5:] if len(v) > 5 else (crN if name.startswith('N_') else cr, center)
        arrays, info = build_one(crs, X, q, qm, Wenv, cen, rc=a.rc)
        # the QM atoms: at T the pair takes its P4 positions inside the P0 environment; the cluster's shell stays at P0
        pair_n = sum(len(crs.mols[k]) for k, _ in pair)
        qm_syms = [crs.syms[i] for k, _ in qm for i in crs.mols[k]]
        qm_xyz = np.vstack([inst_xyz(crs, Wqm if j < len(pair) else Wenv, inst) for j, inst in enumerate(qm)])
        # distances QM (moved pair) to MM: no MM atom may sit closer than 1.2 A to a QM atom
        dmin = float(np.min(np.linalg.norm(arrays['mm_xyz'][:, None, :] - qm_xyz[None, :, :], axis=2)))
        info.update(n_qm=len(qm_syms), qm_charge=int(round(sum(crs.charges[k] for k, _ in qm))),
                    min_qm_mm_A=round(dmin, 3), pair_atoms=pair_n)
        if name in POINT_CONFIGS:
            info['note'] = POINT_CONFIGS[name]
        np.savez_compressed(out / f'{name}.npz', qm_syms=np.array(qm_syms), qm_xyz=qm_xyz, **arrays)
        rec['configs'][name] = info
        print(f"[optc.embed] {a.key} {name}: QM {len(qm_syms)} atoms (charge {info['qm_charge']:+d}), MM {info['n_mm']} "
              f"({info['n_mm_smeared']} smeared), boundary fit RMS {info['fit_rms_V']:.4f} V max {info['fit_max_V']:.4f} V "
              f"(without boundary: RMS {info['without_boundary_rms_V']:.3f} V about a mean of {info['without_boundary_mean_V']:.3f} V), "
              f"min QM-MM {dmin:.2f} A")
    (out / 'emb.json').write_text(json.dumps(rec, indent=1))


def stage_shells(a):
    """membership of the transfer-site shell (asymmetric and symmetric site; H...A cutoffs 2.4, 2.5, 2.6 A)."""
    global HB_HA
    cr = Crystal(a.key)
    _, _, X0 = read_extxyz(RES / a.key / 'P0.extxyz')
    W0 = cr.whole(X0)
    c = cr.rec['copies'][0]
    D, A = c['D'], c['A']
    mD, mA = cr.mol_of[D], cr.mol_of[A]
    best = min(((np.linalg.norm(W0[A] + (np.array(s) - 2) @ cr.lat - W0[D]), tuple(int(x) for x in np.array(s) - 2))
                for s in np.ndindex(5, 5, 5)))
    pair = [(mD, (0, 0, 0)), (mA, best[1])]
    center = np.vstack([inst_xyz(cr, W0, q) for q in pair]).mean(axis=0)
    cands = instances(cr, W0, center, 14.0)
    rows = {}
    for sym in (False, True):
        for cut in (2.4, 2.5, 2.6):
            HB_HA = cut
            sh = hbond_partners(cr, W0, pair, cands, site=site_atoms(cr, c, sym))
            rows[f"{'sym' if sym else 'asym'}_{cut}"] = {
                'n': len(sh), 'molecules': [cr.rec['molecules'][k]['formula'] for k, _ in sh],
                'atoms': int(sum(len(cr.mols[k]) for k, _ in sh) + sum(len(cr.mols[k]) for k, _ in pair))}
    out = RES / a.key / 'emb' / 'shells.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'key': a.key, 'shells': rows}, indent=1))
    print(a.key, json.dumps({k: (v['n'], v['atoms']) for k, v in rows.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'shells'])
    ap.add_argument('--key', required=True)
    ap.add_argument('--rc', type=float, default=RC)
    ap.add_argument('--model', choices=list(SUFFIX), default='repeat')
    ap.add_argument('--tag', default='', help='suffix of the output folder (e.g. rc35 for the MM-radius check)')
    ap.add_argument('--nshell', type=int, default=0, help='keep only the n nearest shell molecules (basis check)')
    ap.add_argument('--site', action='store_true', help='shell = molecules H-bonded to D, the H atoms on D, or A only')
    ap.add_argument('--site-sym', action='store_true', help='with --site: the H atoms on A belong to the site too')
    ap.add_argument('--hb-ha', type=float, default=HB_HA, help='H...A cutoff of the shell (A)')
    ap.add_argument('--smear-radius', type=float, default=SMEAR_RADIUS, help='Gaussian radius of MM charges within 4 A')
    ap.add_argument('--configs', default='', help='build only these configurations (comma list), merged into emb.json; '
                    'R_pair, NR_pair, VS_pair, VR_pair only this way (POINT_CONFIGS)')
    a = ap.parse_args()
    {'build': stage_build, 'shells': stage_shells}[a.stage](a)


if __name__ == '__main__':
    main()
