#!/usr/bin/env python3
"""Molecular stages of the option C series (optc/PLAN.md) with PySCF.

    tools/env/bin/python optc/src/mol.py req  --key KEY [--threads 8]
    tools/env/bin/python optc/src/mol.py pair --key KEY [--threads 8]      # R1f at P0-P4 + pair corrections
    tools/env/bin/python optc/src/mol.py pcm  --key KEY [--threads 8] [--radii-scale 1.2]
    tools/env/bin/python optc/src/mol.py emb  --key KEY --which pair|cluster [--model repeat|sc|gas] [--threads 24 --mem 150000]
    tools/env/bin/python optc/src/mol.py emb  --key KEY --which pair --reverse            # + Tr: P0 pair in the P4 environment
    tools/env/bin/python optc/src/mol.py emb  --key KEY --which pairout|pairshellq --model repeat_site
    tools/env/bin/python optc/src/mol.py emb  --key KEY --which cluster --model sc_site --vac-from repeat_site
    tools/env/bin/python optc/src/mol.py runner   --key KEY                               # runner-up bond, vertical
    tools/env/bin/python optc/src/mol.py monomers --key KEY                               # separated partners, def2-TZVPD

req   r_eq: the receiving partner with the transferred proton, isolated, optimised at PBE-D3(BJ)/def2-TZVP
      (density fitting, grid level 3, geomeTRIC).  r_eq = its A-H length, the target of the periodic end point P4
      (optc/src/qe.py).  Start: the partner's CIF geometry with the H placed on the A->D line (O-H 0.98 A, N-H 1.03 A).
      The result is cached per formula and charge (optc/results/_req/), so H3PO4 is computed once for the five
      phosphates.
pair  R1f: the central pair (the donor and acceptor molecules of the first copy of the transferring bond, whole) cut
      from the periodic geometry of each path point P0-P4, in vacuum: PBE0-D3(BJ)/def2-TZVP (the decomposition
      Hamiltonian), PBE-D3(BJ) (for dE_env at PBE), and for the pair correction DF-MP2 (and its SCS variant) and
      wB97X-D3(BJ), all def2-TZVP, frozen core for MP2.  -> results/<key>/mol/pair.json (energies in Eh, differences in kJ/mol)
pcm   the continuum analogue: the pair at P0 and P4 in IEF-PCM (Bondi x radii-scale cavity, Lebedev 302) at 12
      permittivities; eps_x (the sign change of E(P4) - E(P0)) and eps_eq (where it equals the periodic value)
emb   the embedded rungs at the configurations of optc/src/embed.py: S_pair, T_pair, N_pair (field only) or
      S_cluster, T_cluster (first shell), each in vacuum and in the Ewald-consistent charges.  For each embedded run
      E_int = the classical interaction of the QM nuclei and density with the charges, so that the collective energy
      per transferring pair is E_emb - E_int / 2 (every other pair transfers too).
      Additions of 19 Oct 2026, 9:40 PM IST (optc/PLAN.md, items A, B, G, H, I, S):
      --reverse (pair): Tr = the pair at P0 in the P4 environment with the P4 charges (the arrays of N_pair); with
        N_pair_emb it gives the reverse vertical gap dE(P4 env), the linear-response mean and lambda.
      pairout: the pair alone in the charges of a site cluster (everything except the transfer-site shell): R4a_outer.
      pairshellq: the pair in the shell molecules' own fitted charges only (no other charge, no boundary): R4a_shellq.
      --vac-from: the vacuum runs of a cluster are taken from another model's result (same QM geometry, checked).
runner  the runner-up bond recorded in cell.json: its pair at P0 and with its H placed at r_eq from its acceptor on the
      P0 D->A line (vertical); the chosen bond recomputed the same way; all levels of `pair`.  The acceptor is the
      instance bonded through H (fix of 21 Oct 2026; points computed on another image are recomputed).
      -> mol/pair_runner.json
monomers  the two partners of the pair separated, at the P0 and P4 geometries (each with its own hydrogens), and the
      pair itself, def2-TZVPD: the transfer energy between separated molecules and the ion-pairing part of R1f.
pair --pbe50  adds PBE50-D3(BJ) (0.5 HF + 0.5 PBE exchange, PBE correlation; the D3(BJ) term with the PBE0
      parameters, as in QE) at every point, for corrections anchored on PBE50 (optc/PLAN.md 9:40 PM item K).
cc    [CCSD(T) - MP2] with def2-SVP, frozen core, conventional integrals, on the pair at P0, P4 and P4r: the
      delta(T) correction of the MP2 pair correction (item J).
embcorr  MP2, SCS-MP2 and wB97X-D3(BJ) minus PBE0-D3(BJ) on the pair inside the charges of S_pair, T_pair, N_pair
      (--model): the pair correction computed in the field instead of in vacuum (item J).  --configs R,NR (P4r) or
      VS,VR (relaxed cells) after embed.py build --configs R_pair,NR_pair / VS_pair,VR_pair (21 Oct 2026).
Level everywhere: def2-TZVP (no diffuse functions next to charges), RI-JK (def2-universal-jkfit), grid level 3,
conv_tol 1e-9 Eh (1e-8 for clusters).
"""
import argparse, datetime, gc, json, os, sys
from pathlib import Path

import numpy as np

OPTC = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('OPTC_RESULTS', OPTC / 'results'))
HARTREE_KJ = 2625.4996394799
POINTS = ('P0', 'P1', 'P2', 'P3', 'P4')
EPS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0, 78.36)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def load(key):
    return json.loads((OPTC / 'structures' / key / 'cell.json').read_text())


def mic(rec, i, j):
    lat = np.array(rec['lattice_A'])
    F = np.array(rec['frac'])
    df = F[j] - F[i]
    df -= np.round(df)
    best = None
    for s in np.ndindex(3, 3, 3):
        v = (df + np.array(s) - 1) @ lat
        if best is None or v @ v < best @ best:
            best = v
    return best


def mol_of(rec, atom):
    for m in rec['molecules']:
        if atom in m['atoms']:
            return m
    raise KeyError(atom)


def receiving_partner(rec):
    c = rec['copies'][0]
    D, H, A = c['D'], c['H'], c['A']
    m = mol_of(rec, A)
    W = rec['whole_cart']
    syms = [rec['species'][i] for i in m['atoms']]
    xyz = [W[str(i)] for i in m['atoms']]
    XA = np.array(W[str(A)])
    u = mic(rec, A, D)
    u /= np.linalg.norm(u)
    r0 = 0.98 if rec['species'][A] == 'O' else 1.03
    syms.append('H')
    xyz.append((XA + u * r0).tolist())
    ia = m['atoms'].index(A)
    return syms, np.array(xyz), m['charge'] + 1, ia, len(syms) - 1, m['formula']


RCOV_REQ = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20}


def graph_sig(syms, xyz):
    """bond-graph signature of a molecule (Cordero radii + 0.40 A, each H on its nearest heavy atom): separates
    isomers of one formula (2,6-dimethylpyridine and 3-ethylpyridine are both C7H9N)."""
    import hashlib
    X = np.asarray(xyz, float)
    heavy = [i for i, e in enumerate(syms) if e != 'H']
    nb = {i: [] for i in range(len(syms))}
    for x, i in enumerate(heavy):
        for j in heavy[x + 1:]:
            if np.linalg.norm(X[i] - X[j]) < RCOV_REQ[syms[i]] + RCOV_REQ[syms[j]] + 0.4:
                nb[i].append(j); nb[j].append(i)
    for h in (i for i, e in enumerate(syms) if e == 'H'):
        j = min(heavy, key=lambda k: np.linalg.norm(X[k] - X[h]))
        nb[h].append(j); nb[j].append(h)
    lab = {i: syms[i] for i in nb}
    for _ in range(3):           # three rounds of neighbourhood refinement
        lab = {i: lab[i] + '(' + ','.join(sorted(lab[j] for j in nb[i])) + ')' for i in nb}
    return hashlib.md5('|'.join(sorted(lab.values())).encode()).hexdigest()[:8]


def stage_req(a):
    rec = load(a.key)
    syms, xyz, charge, ia, ih, formula = receiving_partner(rec)
    sig = graph_sig(syms, xyz)
    cache = RES / '_req' / f'{formula}+H_q{charge:+d}.json'
    if cache.exists():
        # a cache file written before 20 Oct 2026 is keyed by formula only: reused only for the same bond graph
        old = json.loads(cache.read_text())
        if graph_sig([r[0] for r in old['xyz_A']], [r[1:] for r in old['xyz_A']]) != sig:
            cache = RES / '_req' / f'{formula}+H_q{charge:+d}_{sig}.json'
    out = RES / a.key / 'req.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        r = json.loads(cache.read_text())
        r['reused_for'] = a.key
        out.write_text(json.dumps(r, indent=1))
        print(f'[optc.mol] {a.key}: r_eq {r["r_eq_A"]:.4f} A (cached {cache.name})')
        return
    from pyscf import gto, dft, lib
    from pyscf.geomopt.geometric_solver import optimize
    lib.num_threads(a.threads)
    mol = gto.M(atom=[(s, tuple(x)) for s, x in zip(syms, xyz)], basis='def2-tzvp', charge=charge, spin=0, unit='Angstrom',
                verbose=3, max_memory=a.mem)
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.grids.level = 3
    mf.disp = 'd3bj'
    mf.conv_tol = 1e-9
    t0 = datetime.datetime.now()
    mol_eq = optimize(mf, maxsteps=200)
    X = mol_eq.atom_coords(unit='Angstrom')
    r = float(np.linalg.norm(X[ih] - X[ia]))
    mf2 = dft.RKS(mol_eq, xc='pbe').density_fit()
    mf2.grids.level = 3
    mf2.disp = 'd3bj'
    e = mf2.kernel()
    res = {'r_eq_A': round(r, 5), 'molecule': f'{formula}+H', 'charge': charge, 'A_index_in_molecule': ia,
           'level': 'PBE-D3(BJ)/def2-TZVP, density fitting, grid 3, geomeTRIC', 'E_Eh': e, 'converged': bool(mf2.converged),
           'source': f'optc/src/mol.py req (first computed for {a.key})', 'utc': utc(),
           'wall_min': round((datetime.datetime.now() - t0).total_seconds() / 60, 1),
           'xyz_A': [[s] + [round(v, 6) for v in x] for s, x in zip(syms, X.tolist())]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res, indent=1))
    out.write_text(json.dumps(res, indent=1))
    print(f'[optc.mol] {a.key}: r_eq {r:.4f} A ({formula}+H, charge {charge:+d}) in {res["wall_min"]} min')


def mk_mol(syms, xyz, charge, mem, basis='def2-tzvp', n_dense=None, outer='def2-svp'):
    """n_dense: locally dense basis, the first n_dense atoms (the transferring pair) in `basis`, the rest in `outer`
    (atom labels X1 / X2 carry the two basis sets)."""
    from pyscf import gto
    if n_dense is None:
        return gto.M(atom=[(s, tuple(map(float, x))) for s, x in zip(syms, xyz)], basis=basis, charge=int(charge), spin=0,
                     unit='Angstrom', verbose=3, max_memory=mem)
    lab = [f'{s}1' if i < n_dense else f'{s}2' for i, s in enumerate(syms)]
    bas = {l: (basis if l.endswith('1') else outer) for l in set(lab)}
    return gto.M(atom=[(l, tuple(map(float, x))) for l, x in zip(lab, xyz)], basis=bas, charge=int(charge), spin=0,
                 unit='Angstrom', verbose=3, max_memory=mem)


def run_dft(mol, xc='pbe0', conv=1e-9, mm=None, dm0=None, solvent_eps=None, radii_scale=1.2, with_df=None, shift=0.0):
    """one RKS single point; mm = (coords A, charges, radii A) for the Gaussian charges; with_df = the density-fitting
    object of an earlier run on the same molecule (its three-index integrals are reused); shift = level shift (Eh) as a
    convergence aid, which does not change a converged energy (emb --level-shift; 21 Oct 2026, the brsalt whole shell,
    whose vacuum SCF did not converge in 150 cycles on 19 Oct).  Returns (mf, e, extra)."""
    from pyscf import dft, qmmm
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    if with_df is not None:
        mf.with_df = with_df
    mf.grids.level = 3
    if xc in ('pbe0', 'pbe'):
        mf.disp = 'd3bj'
    mf.conv_tol = conv
    mf.max_cycle = 150
    if shift:
        mf.level_shift, mf.diis_space = shift, 12
    if mm is not None:
        mf = qmmm.mm_charge(mf, mm[0], mm[1], radii=mm[2], unit='Angstrom')
    if solvent_eps is not None:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = solvent_eps
        mf.with_solvent.vdw_scale = radii_scale
        mf.with_solvent.lebedev_order = 29
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton() if mm is None and solvent_eps is None else mf
        e = mf.kernel(dm0=mf.make_rdm1())
    return mf, float(e), {'converged': bool(mf.converged)}


def e_int(mf, mol, mm):
    """classical interaction (Eh) of the QM nuclei and electron density with the Gaussian MM charges."""
    from pyscf import dft
    dm = mf.make_rdm1()
    h_mm = mf.get_hcore() - dft.RKS(mol).get_hcore()
    el = float(np.einsum('ij,ji->', h_mm, dm))
    nuc = float(mf.energy_nuc() - mol.energy_nuc())
    return el + nuc, el, nuc


RI_OF = {'def2-svp': 'def2-svp-ri', 'def2-tzvp': 'def2-tzvp-ri', 'def2-tzvpd': 'def2-tzvpd-ri', 'def2-qzvp': 'def2-qzvp-ri',
         'def2-qzvpp': 'def2-qzvpp-ri'}


def run_mp2(mol, basis='def2-tzvp'):
    from pyscf import scf, mp
    from pyscf.data import elements
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol = 1e-9
    ehf = mf.kernel()
    pt = mp.MP2(mf).density_fit(auxbasis=RI_OF.get(basis.lower(), 'def2-tzvp-ri'))
    pt.frozen = elements.chemcore(mol)
    ec = pt.kernel(with_t2=False)[0]     # energies only: the t2 amplitudes (40+ GB for the larger pairs) are not kept
    os_, ss = float(pt.e_corr_os), float(pt.e_corr_ss)
    return {'E_HF': float(ehf), 'E_MP2': float(ehf + ec), 'E_SCS_MP2': float(ehf + 1.2 * os_ + ss / 3.0),
            'e_corr_os': os_, 'e_corr_ss': ss, 'frozen': int(pt.frozen), 'hf_converged': bool(mf.converged)}


def pair_geometries(key):
    """the central pair at P0..P4 (whole molecules, the same instances at every point)."""
    import embed as E
    cr = E.Crystal(key)
    _, _, X0 = E.read_extxyz(RES / key / 'P0.extxyz')
    W0 = cr.whole(X0)
    c = cr.rec['copies'][0]
    mD, mA = cr.mol_of[c['D']], cr.mol_of[c['A']]
    best = min(((np.linalg.norm(W0[c['A']] + (np.array(s) - 2) @ cr.lat - W0[c['D']]), tuple(int(x) for x in np.array(s) - 2))
                for s in np.ndindex(5, 5, 5)))
    pair = [(mD, (0, 0, 0)), (mA, best[1])]
    out = {}
    import re
    extra = sorted(f.stem for f in (RES / key).glob('X*.extxyz') if re.fullmatch(r'X\d{4}[rs]', f.stem))
    for pt in POINTS + ('P4r',) + tuple(extra) + ('vcS', 'vcP4r'):
        f = RES / key / f'{pt}.extxyz'
        if not f.exists():
            continue
        lat, _, X = E.read_extxyz(f)
        syms = [cr.syms[i] for k, _ in pair for i in cr.mols[k]]
        idx = [i for k, _ in pair for i in cr.mols[k]]
        if pt.startswith('vc'):
            # relaxed cell (optc/PLAN.md 9:40 PM item E): the same two molecules, made whole in their own lattice,
            # the acceptor image nearest the donor atom chosen again
            import copy
            cv = copy.copy(cr)
            cv.lat, cv.inv = lat, np.linalg.inv(lat)
            Wv = cv.whole(X)
            bv = min(((np.linalg.norm(Wv[c['A']] + (np.array(s) - 2) @ lat - Wv[c['D']]), tuple(int(x) for x in np.array(s) - 2))
                      for s in np.ndindex(5, 5, 5)))
            xyz = np.vstack([E.inst_xyz(cv, Wv, q) for q in [(mD, (0, 0, 0)), (mA, bv[1])]])
        else:
            W = W0 if pt == 'P0' else E.whole_like(cr, X, W0)
            xyz = np.vstack([E.inst_xyz(cr, W, q) for q in pair])
        out[pt] = (syms, xyz, idx)
    charge = sum(cr.rec['molecules'][k]['charge'] for k, _ in pair)
    return out, charge, cr


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def stage_pair_basis(a):
    """basis check of R1f: PBE0-D3(BJ) and DF-MP2 at P0 and P4 (and P4r) with another basis, DF-MP2 with that basis's
    own RI basis at every point (RI_OF, recorded as aux_mp2)."""
    from pyscf import lib
    lib.num_threads(a.threads)
    geo, charge, cr = pair_geometries(a.key)
    out = RES / a.key / 'mol' / f'pair_{a.basis}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'basis': a.basis, 'points': {}}
    aux = RI_OF.get(a.basis.lower(), 'def2-tzvp-ri')
    for pt in ('P0', 'P4', 'P4r'):
        old = res['points'].get(pt, {})
        if old.get('done') and old.get('aux_mp2') != aux:
            # DF-MP2 with another RI basis (before 11f5eb6, 19 Oct 2026 10:21 PM, every basis used def2-tzvp-ri; no
            # aux_mp2 was recorded then): kept aside, recomputed, so that no difference mixes two RI bases
            res.setdefault('superseded', {})[pt] = dict(old, aux_mp2=old.get('aux_mp2', 'def2-tzvp-ri (not recorded)'))
            res['points'].pop(pt)
        if pt not in geo or res['points'].get(pt, {}).get('done'):
            continue
        syms, xyz, _ = geo[pt]
        mol = mk_mol(syms, xyz, charge, a.mem, basis=a.basis)
        mf, e, ex = run_dft(mol, 'pbe0')
        r = {'E_PBE0_D3': e, 'nao': mol.nao, 'aux_mp2': aux}
        r.update(run_mp2(mol, a.basis))
        r['done'] = True
        res['points'][pt] = r
        save_json(out, res)
    P = res['points']
    if any(P[pt].get('aux_mp2') != aux for pt in P):
        raise SystemExit(f'{a.key}: pair_{a.basis} points with different RI bases {[(pt, P[pt].get("aux_mp2")) for pt in P]}')
    res['aux_mp2'] = aux
    res['dPBE0_D3_kJ'] = round((P['P4']['E_PBE0_D3'] - P['P0']['E_PBE0_D3']) * HARTREE_KJ, 3)
    res['dMP2_kJ'] = round((P['P4']['E_MP2'] - P['P0']['E_MP2']) * HARTREE_KJ, 3)
    if 'P4r' in P:
        res['dPBE0_D3_P4r_kJ'] = round((P['P4r']['E_PBE0_D3'] - P['P0']['E_PBE0_D3']) * HARTREE_KJ, 3)
        res['dMP2_P4r_kJ'] = round((P['P4r']['E_MP2'] - P['P0']['E_MP2']) * HARTREE_KJ, 3)
    save_json(out, res)
    print(f'[optc.mol] {a.key} pair basis {a.basis}: dPBE0 {res["dPBE0_D3_kJ"]} dMP2 {res["dMP2_kJ"]} kJ/mol', flush=True)


def stage_pair(a):
    from pyscf import lib
    lib.num_threads(a.threads)
    if a.basis != 'def2-tzvp':
        return stage_pair_basis(a)
    geo, charge, cr = pair_geometries(a.key)
    out = RES / a.key / 'mol' / 'pair.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'points': {}}
    for pt, (syms, xyz, idx) in geo.items():
        r = res['points'].get(pt, {})
        if r.get('done') and a.pbe50 and 'E_PBE50_D3' not in r:
            mol = mk_mol(syms, xyz, charge, a.mem)
            r['E_PBE50_D3'], r['pbe50_converged'] = e_pbe50(mol)
            save_json(out, res)
            print(f'[optc.mol] {a.key} pair {pt}: PBE50-D3 {r["E_PBE50_D3"]:.8f}', flush=True)
        if r.get('done'):
            continue
        mol = mk_mol(syms, xyz, charge, a.mem)
        mf, e, ex = run_dft(mol, 'pbe0')
        r.update(E_PBE0_D3=e, pbe0_converged=ex['converged'], nao=mol.nao, natm=mol.natm)
        mf2, e2, ex2 = run_dft(mol, 'wb97x-d3bj', dm0=mf.make_rdm1())
        r.update(E_wB97X_D3BJ=e2, wb97x_converged=ex2['converged'])
        mf3, e3, ex3 = run_dft(mol, 'pbe', dm0=mf.make_rdm1())
        r.update(E_PBE_D3=e3, pbe_converged=ex3['converged'])
        r.update(run_mp2(mol))
        if a.pbe50:
            r['E_PBE50_D3'], r['pbe50_converged'] = e_pbe50(mol, dm0=mf.make_rdm1())
        r['xyz'] = [[s] + [round(float(v), 6) for v in x] for s, x in zip(syms, xyz)]
        r['done'] = True
        res['points'][pt] = r
        save_json(out, res)
        print(f'[optc.mol] {a.key} pair {pt}: PBE0-D3 {e:.8f}  MP2 {r["E_MP2"]:.8f}  wB97X-D3BJ {e2:.8f}', flush=True)
    P = res['points']
    if 'P0' in P:
        for pt in P:
            for lev in ('E_PBE0_D3', 'E_PBE_D3', 'E_MP2', 'E_SCS_MP2', 'E_wB97X_D3BJ', 'E_HF', 'E_PBE50_D3'):
                if lev in P[pt] and lev in P['P0']:
                    P[pt][f'd{lev[2:]}_kJ'] = round((P[pt][lev] - P['P0'][lev]) * HARTREE_KJ, 3)
            for lev in ('MP2', 'SCS_MP2', 'wB97X_D3BJ'):
                k = f'd{lev}_kJ'
                if k in P[pt] and 'dPBE0_D3_kJ' in P[pt]:
                    P[pt][f'corr_{lev}_kJ'] = round(P[pt][k] - P[pt]['dPBE0_D3_kJ'], 3)
                if k in P[pt] and 'dPBE50_D3_kJ' in P[pt]:
                    P[pt][f'corr_{lev}_pbe50_kJ'] = round(P[pt][k] - P[pt]['dPBE50_D3_kJ'], 3)
        res['charge'] = charge
        res['units'] = 'E in Eh; d* = E(Pk) - E(P0) in kJ/mol; corr_* = d(level) - d(PBE0-D3) in kJ/mol (pair correction)'
        save_json(out, res)


def e_pbe50(mol, dm0=None):
    """PBE50-D3(BJ): the SCF with 0.5 HF exchange, plus the D3(BJ) term with the PBE0 parameters."""
    from pyscf import dft
    from pyscf.dispersion import dftd3
    mf = dft.RKS(mol, xc='0.5*HF + 0.5*PBE, PBE').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 150
    e = float(mf.kernel(dm0=dm0))
    return e + float(dftd3.DFTD3Dispersion(mol, xc='pbe0', version='d3bj').get_dispersion()['energy']), bool(mf.converged)


def stage_cc(a):
    """delta(T) = [CCSD(T) - MP2] / def2-SVP, frozen core, at P0, P4, P4r (optc/PLAN.md 9:40 PM item J)."""
    from pyscf import lib, scf, mp, cc
    from pyscf.data import elements
    lib.num_threads(a.threads)
    geo, charge, cr = pair_geometries(a.key)
    out = RES / a.key / 'mol' / 'pair_cc.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'basis': 'def2-svp', 'points': {}}
    for pt in (a.points.split(',') if a.points else ('P0', 'P4', 'P4r')):
        if pt not in geo or res['points'].get(pt, {}).get('done'):
            continue
        syms, xyz, _ = geo[pt]
        mol = mk_mol(syms, xyz, charge, a.mem, basis='def2-svp')
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        ehf = float(mf.kernel())
        fz = elements.chemcore(mol)
        pt2 = mp.MP2(mf)
        pt2.frozen = fz
        emp2 = ehf + float(pt2.kernel()[0])
        mcc = cc.CCSD(mf)
        mcc.frozen = fz
        mcc.max_cycle = 100
        ecc = float(mcc.kernel()[0])
        et = float(mcc.ccsd_t())
        r = {'E_HF': ehf, 'E_MP2': emp2, 'E_CCSD': ehf + ecc, 'E_CCSD_T': ehf + ecc + et, 'ccsd_converged': bool(mcc.converged),
             'nao': mol.nao, 'frozen': int(fz), 'done': True}
        res['points'][pt] = r
        save_json(out, res)
        print(f'[optc.mol] {a.key} cc {pt}: MP2 {emp2:.8f} CCSD(T) {r["E_CCSD_T"]:.8f}', flush=True)
    save_json(out, cc_derived(res))


def cc_derived(res):
    """the P4 / P4r differences against P0 of a pair_cc.json; also run by ops/optc/merge.sh on the union of the points
    computed on several machines (22 Oct 2026: pcplut P0 on g1, P4r on g4)."""
    P = res['points']
    for pt in ('P4', 'P4r'):
        if pt in P and 'P0' in P:
            dT = ((P[pt]['E_CCSD_T'] - P[pt]['E_MP2']) - (P['P0']['E_CCSD_T'] - P['P0']['E_MP2'])) * HARTREE_KJ
            res[f'delta_T_{pt}_kJ'] = round(dT, 3)
            res[f'dMP2_{pt}_kJ'] = round((P[pt]['E_MP2'] - P['P0']['E_MP2']) * HARTREE_KJ, 3)
            res[f'dCCSD_T_{pt}_kJ'] = round((P[pt]['E_CCSD_T'] - P['P0']['E_CCSD_T']) * HARTREE_KJ, 3)
    return res


def stage_embcorr(a):
    """pair corrections inside the field (optc/PLAN.md 9:40 PM item J): PBE0, wB97X-D3(BJ), RHF + DF-MP2 with charges."""
    from pyscf import lib, scf, mp, qmmm
    from pyscf.data import elements
    lib.num_threads(a.threads)
    d = RES / a.key / 'emb' / a.model
    meta = json.loads((d / 'emb.json').read_text())
    out = RES / a.key / 'mol' / f'embcorr_pair_{a.model}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'model': a.model, 'runs': {}}
    a.which = 'pair'
    cfgs = ['S'] + [c for c in a.configs.split(',') if c and c != 'S']
    for cfg in cfgs:
        k = f'{cfg}_pair'
        if res['runs'].get(k, {}).get('done'):
            continue
        if k not in meta['configs']:
            raise SystemExit(f'{a.key}: {k} is not in emb/{a.model}/emb.json (embed.py build --configs {k})')
        syms, xyz, charge, npair, mm = emb_config(a, d, meta, cfg)
        mol = mk_mol(syms, xyz, charge, a.mem)
        mf0, e0, ex0 = run_dft(mol, 'pbe0', mm=mm)
        mf1, e1, ex1 = run_dft(mol, 'wb97x-d3bj', mm=mm, dm0=mf0.make_rdm1())
        hf = qmmm.mm_charge(scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit'), mm[0], mm[1], radii=mm[2], unit='Angstrom')
        hf.conv_tol = 1e-9
        ehf = float(hf.kernel())
        pt2 = mp.MP2(hf).density_fit(auxbasis='def2-tzvp-ri')
        pt2.frozen = elements.chemcore(mol)
        ec = float(pt2.kernel(with_t2=False)[0])
        os_, ss = float(pt2.e_corr_os), float(pt2.e_corr_ss)
        res['runs'][k] = {'E_PBE0_D3': e0, 'E_wB97X_D3BJ': e1, 'E_HF': ehf, 'E_MP2': ehf + ec,
                          'E_SCS_MP2': ehf + 1.2 * os_ + ss / 3.0, 'converged': [ex0['converged'], ex1['converged'], bool(hf.converged)],
                          'done': True}
        save_json(out, res)
        print(f'[optc.mol] {a.key} embcorr {k}: PBE0 {e0:.8f} MP2 {ehf + ec:.8f}', flush=True)
    R = res['runs']
    der = {}
    for cfg in ('T', 'N', 'R', 'NR', 'VS', 'VR'):
        if f'{cfg}_pair' in R and 'S_pair' in R:
            dd = {lev: (R[f'{cfg}_pair'][f'E_{lev}'] - R['S_pair'][f'E_{lev}']) * HARTREE_KJ
                  for lev in ('PBE0_D3', 'wB97X_D3BJ', 'MP2', 'SCS_MP2')}
            der[cfg] = {f'corr_{lev}_kJ': round(dd[lev] - dd['PBE0_D3'], 3) for lev in ('wB97X_D3BJ', 'MP2', 'SCS_MP2')}
            der[cfg]['dPBE0_kJ'] = round(dd['PBE0_D3'], 3)
    if 'VS' in der and 'VR' in der:
        # the relaxed-cell difference, as the vacuum one (analyse.py: dE_relaxed_cells_pbe0 + corr(vcP4r) - corr(vcS))
        der['VR_minus_VS'] = {k: round(der['VR'][k] - der['VS'][k], 3) for k in der['VR'] if k.startswith('corr_')}
    res['derived'] = der
    res['note'] = ('every entry: [E(level) - E(PBE0)] of the configuration minus that of S_pair (the P0 pair in the P0 '
                   'field), plain differences, no E_int/2. T: single site (P4 pair in the P0 field); N: every atom at P4 '
                   'with the P4 field. Added 21 Oct 2026 (embed.py POINT_CONFIGS): R: the P4r pair in the P0 field; NR: '
                   'every atom at P4r with the P4r field, the in-field counterpart of pair.json corr at P4r; VS / VR: the '
                   'vcS / vcP4r pairs in their relaxed cells with the X-ray-cell P0 / P4r charges, counterparts of '
                   'pair.json corr at vcS / vcP4r; VR_minus_VS: the relaxed-cell correction difference.')
    res.setdefault('configs_note', {}).update({c: meta['configs'][f'{c}_pair']['note'] for c in cfgs
                                              if meta['configs'].get(f'{c}_pair', {}).get('note')})
    save_json(out, res)
    print(f'[optc.mol] {a.key} embcorr: {der}', flush=True)


def stage_hscanpair(a):
    """pair corrections along a frozen-frame proton scan (qe.py hscan): the central pair cut from the --point geometry
    with copy 0's H moved by each offset along its D->A line, in vacuum at every level of `pair`; the corrected scan is
    the periodic PBE scan + [E(level) - E(PBE)] of the pair (optc/PLAN.md 20 Oct 2026 entry).  --wb97x adds
    wB97X-D3(BJ), also to runs already done (only that SCF), so the PBE0+wB97X-D3(BJ) frame potential exists."""
    import embed as E
    from pyscf import lib
    lib.num_threads(a.threads)
    pt = a.point
    scan = json.loads((RES / a.key / f'hscan_{pt}.json').read_text())
    geo, charge, cr = pair_geometries(a.key)
    syms, xyz0, idx = geo[pt]
    c = cr.rec['copies'][0]
    iD, iH, iA = idx.index(c['D']), idx.index(c['H']), idx.index(c['A'])
    u = (xyz0[iA] - xyz0[iD]) / np.linalg.norm(xyz0[iA] - xyz0[iD])
    out = RES / a.key / 'mol' / f'hscan_pair_{pt}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'point': pt, 'runs': {}}
    for x in scan['offsets_A']:
        k = f'{x:+.2f}'
        X = xyz0.copy()
        X[iH] = xyz0[iH] + x * u
        r = res['runs'].get(k, {})
        if r.get('done') and a.wb97x and 'E_wB97X_D3BJ' not in r:
            # --wb97x on a finished scan (pcplut, results audit 21 Oct 2026): only the wB97X-D3(BJ) SCF is added to the
            # stored run; the levels already done are kept as they are
            mf1, e1, ex1 = run_dft(mk_mol(syms, X, charge, a.mem), 'wb97x-d3bj')
            r.update(E_wB97X_D3BJ=e1, wb97x_converged=ex1['converged'])
            save_json(out, res)
            print(f'[optc.mol] {a.key} hscan pair {pt} {k}: wB97X-D3(BJ) {e1:.8f}', flush=True)
        if r.get('done'):
            continue
        r = level_energies(mk_mol(syms, X, charge, a.mem), wb97x=a.wb97x)
        r['done'] = True
        res['runs'][k] = r
        save_json(out, res)
        print(f'[optc.mol] {a.key} hscan pair {pt} {k}: PBE0-D3 {r["E_PBE0_D3"]:.8f}', flush=True)
    ks = sorted(res['runs'], key=float)
    ref = res['runs'][f'{0.0:+.2f}']
    res['offsets_A'] = [float(k) for k in ks]
    res['corr_vs_pbe_kJ'] = {lev[2:]: [round(((res['runs'][k][lev] - res['runs'][k]['E_PBE_D3']) - (ref[lev] - ref['E_PBE_D3'])) * HARTREE_KJ, 3)
                                       for k in ks] for lev in ('E_PBE0_D3', 'E_MP2', 'E_SCS_MP2', 'E_wB97X_D3BJ')
                             if all(lev in res['runs'][k] for k in ks)}
    save_json(out, res)


def stage_pcm(a):
    from pyscf import lib
    lib.num_threads(a.threads)
    geo, charge, cr = pair_geometries(a.key)
    tag = f'pcm_r{a.radii_scale:.1f}'
    out = RES / a.key / 'mol' / f'{tag}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'radii_scale': a.radii_scale, 'E': {}}
    for pt in ('P0', 'P4'):
        syms, xyz, _ = geo[pt]
        mol = mk_mol(syms, xyz, charge, a.mem)
        dm = None
        for eps in EPS:
            k = f'{pt}_{eps:g}'
            if k in res['E']:
                continue
            if eps == 1.0:
                mf, e, ex = run_dft(mol, 'pbe0', dm0=dm)
            else:
                mf, e, ex = run_dft(mol, 'pbe0', dm0=dm, solvent_eps=eps, radii_scale=a.radii_scale)
            dm = mf.make_rdm1()
            res['E'][k] = {'E': e, 'converged': ex['converged']}
            save_json(out, res)
            print(f'[optc.mol] {a.key} {tag} {k}: {e:.8f}', flush=True)
    res['dE_kJ'] = {f'{eps:g}': round((res['E'][f'P4_{eps:g}']['E'] - res['E'][f'P0_{eps:g}']['E']) * HARTREE_KJ, 3)
                    for eps in EPS if f'P4_{eps:g}' in res['E'] and f'P0_{eps:g}' in res['E']}
    save_json(out, res)


def level_energies(mol, basis='def2-tzvp', wb97x=True):
    """the levels of `pair`: PBE0-D3(BJ), wB97X-D3(BJ), PBE-D3(BJ), (SCS-)DF-MP2 with frozen core."""
    mf, e, ex = run_dft(mol, 'pbe0')
    r = {'E_PBE0_D3': e, 'pbe0_converged': ex['converged'], 'nao': mol.nao}
    if wb97x:
        mf2, e2, ex2 = run_dft(mol, 'wb97x-d3bj', dm0=mf.make_rdm1())
        r.update(E_wB97X_D3BJ=e2, wb97x_converged=ex2['converged'])
    mf3, e3, ex3 = run_dft(mol, 'pbe', dm0=mf.make_rdm1())
    r.update(E_PBE_D3=e3, pbe_converged=ex3['converged'])
    r.update(run_mp2(mol, basis))
    return r


LEVELS = ('E_PBE0_D3', 'E_PBE_D3', 'E_wB97X_D3BJ', 'E_MP2', 'E_SCS_MP2')


def diff_kj(r1, r0):
    return {lev[2:]: round((r1[lev] - r0[lev]) * HARTREE_KJ, 3) for lev in LEVELS if lev in r1 and lev in r0}


def bond_pair(cr, W0, D, H, A):
    """the donor molecule of D and the acceptor instance bonded through H (its A nearest H), whole, at the positions W0;
    also returns that instance's lattice shift.  Until 21 Oct 2026 this took the instance nearest D, the rule cell.py and
    qe.site_proton dropped on 20 Oct: in chloro and carboxy that is not the image the runner-up's H bonds to, so the
    vertical geometry put it 0.70 / 0.80 A from the chosen bond's proton (results audit, 21 Oct)."""
    import embed as E
    mD, mA = cr.mol_of[D], cr.mol_of[A]
    if mD == mA:
        raise SystemExit(f'{cr.key}: D {D} and A {A} in the same molecule')
    if cr.mol_of[H] != mD:
        raise SystemExit(f'{cr.key}: H {H} is not in the molecule of D {D}')
    img = E.image_nearest(cr, W0, A, H)
    pair = [(mD, (0, 0, 0)), (mA, img)]
    idx = [i for k, _ in pair for i in cr.mols[k]]
    xyz = np.vstack([E.inst_xyz(cr, W0, q) for q in pair])
    return [cr.syms[i] for i in idx], xyz, idx, sum(cr.rec['molecules'][k]['charge'] for k, _ in pair), img


def stage_runner(a):
    """runner-up bond (optc/PLAN.md 9:40 PM item G): vertical transfer of its H to r_eq from its acceptor on the P0
    D->A line, beside the chosen bond computed the same way."""
    import embed as E
    from pyscf import lib
    lib.num_threads(a.threads)
    cr = E.Crystal(a.key)
    ru = cr.rec.get('runner_up')
    out = RES / a.key / 'mol' / 'pair_runner.json'
    if not ru:
        save_json(out, {'key': a.key, 'skipped': 'no runner-up bond recorded in cell.json'})
        return
    _, _, X0 = E.read_extxyz(RES / a.key / 'P0.extxyz')
    W0 = cr.whole(X0)
    ch = cr.rec['copies'][0]
    req = json.loads((RES / a.key / 'req.json').read_text())['r_eq_A']
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'runner_up': ru, 'bonds': {}}
    res['runner_up'] = ru
    for tag, b in (('chosen', ch), ('runner', ru)):
        syms, xyz, idx, charge, img = bond_pair(cr, W0, b['D'], b['H'], b['A'])
        same = cr.syms[b['A']] == cr.syms[ch['A']] and \
            cr.rec['molecules'][cr.mol_of[b['A']]]['formula'] == cr.rec['molecules'][cr.mol_of[ch['A']]]['formula']
        r_eq = req if same else {'O': 0.97, 'N': 1.02}[cr.syms[b['A']]]
        iD, iH, iA = idx.index(b['D']), idx.index(b['H']), idx.index(b['A'])
        u = xyz[iD] - xyz[iA]
        xv = xyz.copy()
        xv[iH] = xyz[iA] + r_eq * u / np.linalg.norm(u)
        want = {'D': b['D'], 'H': b['H'], 'A': b['A'], 'A_image': list(img)}
        rb = res['bonds'].get(tag)
        if rb is not None and {k: rb.get(k) for k in want} != want:
            if rb.get('A_image') is None and all(rb.get(k) == want[k] for k in ('D', 'H', 'A')) \
                    and E.image_nearest(cr, W0, b['A'], b['D']) == img:
                rb['A_image'] = list(img)          # computed before the 21 Oct fix, but on the same image: kept
            else:
                # another bond or another acceptor image: kept aside for the record, recomputed
                res.setdefault('superseded', []).append(dict(rb, tag=tag, superseded='21 Oct 2026: acceptor image '
                                                             'bonded through H (bond_pair fix)'))
                res.pop('runner_minus_chosen_kJ', None)
                rb = None
        if rb is None:
            rb = res['bonds'][tag] = {'D': b['D'], 'H': b['H'], 'A': b['A'], 'A_image': list(img), 'r_eq_A': r_eq,
                                      'r_eq_from': 'req.json' if same else 'assumed (other acceptor type)',
                                      'DA_A': round(float(np.linalg.norm(u)), 4), 'points': {}}
        for pt, X in (('P0', xyz), ('vert', xv)):
            if rb['points'].get(pt, {}).get('done'):
                continue
            mol = mk_mol(syms, X, charge, a.mem)
            r = level_energies(mol)
            r['done'] = True
            rb['points'][pt] = r
            save_json(out, res)
            print(f'[optc.mol] {a.key} runner {tag} {pt}: PBE0-D3 {r["E_PBE0_D3"]:.8f}', flush=True)
        rb['vertical_kJ'] = diff_kj(rb['points']['vert'], rb['points']['P0'])
        save_json(out, res)
    B = res['bonds']
    res['runner_minus_chosen_kJ'] = {k: round(B['runner']['vertical_kJ'][k] - B['chosen']['vertical_kJ'][k], 3)
                                     for k in B['chosen']['vertical_kJ']}
    save_json(out, res)
    print(f'[optc.mol] {a.key} runner-up minus chosen (vertical): {res["runner_minus_chosen_kJ"]}', flush=True)


def stage_monomers(a):
    """the two partners separated, at each point's crystal geometry, each with the hydrogens it carries there (the
    transferring H goes with the nearer of D and A), and the pair itself, all in a.basis (default def2-TZVPD):
    dE_sep = transfer energy between separated partners; ion_pairing = R1f - dE_sep (optc/PLAN.md 9:40 PM item S)."""
    from pyscf import lib
    lib.num_threads(a.threads)
    basis = 'def2-tzvpd' if a.basis == 'def2-tzvp' else a.basis
    geo, charge, cr = pair_geometries(a.key)
    c = cr.rec['copies'][0]
    mD, mA = cr.mol_of[c['D']], cr.mol_of[c['A']]
    qD, qA = cr.rec['molecules'][mD]['charge'], cr.rec['molecules'][mA]['charge']
    out = RES / a.key / 'mol' / f'monomers_{basis}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'basis': basis, 'points': {}}
    for pt in ('P0', 'P4', 'P4r'):
        if pt not in geo:
            continue
        syms, xyz, idx = geo[pt]
        iD, iH, iA = idx.index(c['D']), idx.index(c['H']), idx.index(c['A'])
        on_acc = np.linalg.norm(xyz[iH] - xyz[iA]) < np.linalg.norm(xyz[iH] - xyz[iD])
        donor = [j for j, i in enumerate(idx) if i in cr.mols[mD] and not (on_acc and i == c['H'])]
        acc = [j for j, i in enumerate(idx) if i in cr.mols[mA]] + ([iH] if on_acc else [])
        frags = {'pair': (list(range(len(idx))), charge), 'donor': (donor, qD - int(on_acc)),
                 'acceptor': (acc, qA + int(on_acc))}
        rp = res['points'].setdefault(pt, {'H_on': 'acceptor' if on_acc else 'donor'})
        for name, (rows, q) in frags.items():
            if rp.get(name, {}).get('done'):
                continue
            mol = mk_mol([syms[j] for j in rows], xyz[rows], q, a.mem, basis=basis)
            r = level_energies(mol, basis)
            r.update(charge=q, natm=len(rows), done=True)
            rp[name] = r
            save_json(out, res)
            print(f'[optc.mol] {a.key} monomers {pt} {name} (q {q:+d}): PBE0-D3 {r["E_PBE0_D3"]:.8f}', flush=True)
    P = res['points']
    der = {}
    for pt in ('P4', 'P4r'):
        if pt not in P or 'P0' not in P:
            continue
        sep1 = {lev: P[pt]['donor'][lev] + P[pt]['acceptor'][lev] for lev in LEVELS}
        sep0 = {lev: P['P0']['donor'][lev] + P['P0']['acceptor'][lev] for lev in LEVELS}
        dpair = diff_kj(P[pt]['pair'], P['P0']['pair'])
        dsep = diff_kj(sep1, sep0)
        der[pt] = {'R1f_kJ': dpair, 'dE_sep_kJ': dsep,
                   'ion_pairing_kJ': {k: round(dpair[k] - dsep[k], 3) for k in dpair}}
    res['derived'] = der
    res['units'] = 'E in Eh; *_kJ = E(point) - E(P0) in kJ/mol; ion_pairing = R1f - dE_sep'
    save_json(out, res)
    print(f'[optc.mol] {a.key} monomers ({basis}): {json.dumps(der.get("P4", {}))}', flush=True)


def charge_set(z):
    """the MM and boundary charges of an embedding npz as (coords A, charges, Gaussian radii A)."""
    cx = np.vstack([z['mm_xyz'], z['b_xyz']])
    cq = np.concatenate([z['mm_q'], z['b_q']])
    cr_ = np.concatenate([z['mm_radii'], np.full(len(z['b_q']), 0.3)])
    return cx, cq, cr_


def emb_config(a, d, meta, cfg):
    """QM atoms, charge, number of pair atoms and charge set of one configuration of `a.which`."""
    w = a.which
    if w == 'cluster' and cfg == 'Tr':
        # the P0 pair (S composition) with the shell at P4 (N composition), in the P4 environment of N_cluster
        zS, zN = np.load(d / 'S_cluster.npz'), np.load(d / 'N_cluster.npz')
        nS, nN = meta['configs']['S_cluster']['pair_atoms'], meta['configs']['N_cluster']['pair_atoms']
        syms = list(zS['qm_syms'])[:nS] + list(zN['qm_syms'])[nN:]
        xyz = np.vstack([zS['qm_xyz'][:nS], zN['qm_xyz'][nN:]])
        return syms, xyz, meta['configs']['N_cluster']['qm_charge'], nS, charge_set(zN)
    if w in ('pair', 'cluster'):
        name = 'S_pair' if cfg == 'Tr' else f'{cfg}_{w}'
        z = np.load(d / f'{name}.npz')
        syms, xyz = list(z['qm_syms']), z['qm_xyz']
        charge, npair = meta['configs'][name]['qm_charge'], meta['configs'][name]['pair_atoms']
        # Tr: the P0 pair (S composition) inside the environment of N_pair (every other atom at P4, P4 charges)
        mm = charge_set(np.load(d / 'N_pair.npz')) if cfg == 'Tr' else charge_set(z)
        return syms, xyz, charge, npair, mm
    z = np.load(d / f'{cfg}_cluster.npz')
    npair = meta['configs'][f'{cfg}_cluster']['pair_atoms']
    syms, xyz = list(z['qm_syms'])[:npair], z['qm_xyz'][:npair]
    charge = meta['configs'][f'{cfg}_pair']['qm_charge']
    if w == 'pairout':
        # the pair in the charges of everything except the transfer-site shell (the site cluster's MM + boundary)
        return syms, xyz, charge, npair, charge_set(z)
    # pairshellq: the pair in the shell molecules' own fitted charges (P0 charges of the model), nothing else
    import embed as E
    cr = E.Crystal(a.key)
    idx = [i for k, _ in meta['shell'] for i in cr.mols[k]]
    base = a.model.split('_')[0]
    q0 = np.array(json.loads((RES / a.key / f"charges_P0{E.SUFFIX[base]}.json").read_text())['charges'])
    sx = z['qm_xyz'][npair:]
    if len(idx) != len(sx) or [cr.syms[i] for i in idx] != list(z['qm_syms'])[npair:]:
        raise SystemExit(f'{a.key}: shell atoms of {a.model} do not match the cluster file')
    # radii from the P0 positions (S) in both configurations, as embed.build_one fixes them for every other charge set;
    # until 21 Oct 2026 they followed the moved pair at T (results audit: 1-3 shell charges crossed 4 A in 7 salts)
    zS = np.load(d / 'S_cluster.npz')
    nS = meta['configs']['S_cluster']['pair_atoms']
    if list(zS['qm_syms'])[nS:] != list(z['qm_syms'])[npair:]:
        raise SystemExit(f'{a.key}: shell atoms of S_cluster and {cfg}_cluster differ')
    dmin = np.min(np.linalg.norm(zS['qm_xyz'][nS:][:, None, :] - zS['qm_xyz'][:nS][None, :, :], axis=2), axis=1)
    radii = np.where(dmin < E.SMEAR_WITHIN, E.SMEAR_RADIUS, E.POINT_RADIUS)
    return syms, xyz, charge, npair, (sx, q0[idx], radii)


def stage_emb(a):
    from pyscf import lib
    lib.num_threads(a.threads)
    d = RES / a.key / 'emb' / a.model
    ldb = a.ldb and a.which == 'cluster'
    out = RES / a.key / 'mol' / f'emb_{a.which}_{a.model}{"_ldb" if ldb else ""}.json'
    res = json.loads(out.read_text()) if out.exists() else {'key': a.key, 'which': a.which, 'model': a.model, 'runs': {},
                                                            'basis': 'def2-TZVP pair + def2-SVP shell' if ldb else 'def2-TZVP'}
    meta = json.loads((d / 'emb.json').read_text())
    if a.which in ('pairout', 'pairshellq') and not meta.get('shell'):
        res['skipped'] = 'empty transfer-site shell: R4a_outer = R4a and R4a_shellq = R1f by definition'
        save_json(out, res)
        print(f'[optc.mol] {a.key} {a.which}: {res["skipped"]}', flush=True)
        return
    rT = res['runs'].get('T_pairshellq_emb')
    if a.which == 'pairshellq' and rT and rT.get('radii_from') != 'S':
        # computed before the 21 Oct 2026 fix with the radii of the moved pair: kept where they equal the S radii
        import embed as E
        zT, nT = np.load(d / 'T_cluster.npz'), meta['configs']['T_cluster']['pair_atoms']
        dT = np.min(np.linalg.norm(zT['qm_xyz'][nT:][:, None, :] - zT['qm_xyz'][:nT][None, :, :], axis=2), axis=1)
        if np.array_equal(np.where(dT < E.SMEAR_WITHIN, E.SMEAR_RADIUS, E.POINT_RADIUS), emb_config(a, d, meta, 'T')[4][2]):
            rT['radii_from'] = 'S'
        else:
            res.setdefault('superseded', {})['T_pairshellq_emb'] = res['runs'].pop('T_pairshellq_emb')
            res.pop('derived', None)
        save_json(out, res)
    configs = {'pair': ['S', 'T', 'N'] + (['Tr'] if a.reverse else []),
               'cluster': ['S', 'T'] + (['N', 'Tr'] if a.reverse else [])}.get(a.which, ['S', 'T'])
    if a.which == 'cluster' and a.max_atoms and meta['configs']['S_cluster']['n_qm'] > a.max_atoms:
        # rule of 19 Oct 2026 (optc/PLAN.md): the whole first shell is computed only up to max_atoms QM atoms
        res['skipped'] = f"cluster of {meta['configs']['S_cluster']['n_qm']} atoms > {a.max_atoms}: not computed"
        save_json(out, res)
        print(f'[optc.mol] {a.key} {res["skipped"]}', flush=True)
        return
    if a.vac_from and a.which == 'cluster':
        # the vacuum runs depend only on the QM geometry, which is the same for every charge model: take them over
        src = json.loads((RES / a.key / 'mol' / f'emb_cluster_{a.vac_from}.json').read_text())
        for cfg in ('S', 'T'):
            z1 = np.load(d / f'{cfg}_cluster.npz')
            z2 = np.load(RES / a.key / 'emb' / a.vac_from / f'{cfg}_cluster.npz')
            if list(z1['qm_syms']) != list(z2['qm_syms']) or np.abs(z1['qm_xyz'] - z2['qm_xyz']).max() > 1e-6:
                raise SystemExit(f'{a.key}: {cfg}_cluster QM geometry differs between {a.model} and {a.vac_from}')
            k = f'{cfg}_cluster_vac'
            if k not in res['runs']:
                res['runs'][k] = dict(src['runs'][k], copied_from=f'emb_cluster_{a.vac_from}.json')
        save_json(out, res)
    if a.field_only:
        # 21 Oct 2026: the brsalt whole shell has a negative HOMO-LUMO gap in vacuum, so only its field runs are made
        res['vacuum_not_computed'] = a.field_only
        save_json(out, res)
    dms = {}
    if a.dm0_chk:
        # initial density from a PySCF chkfile of the same atoms (e.g. the S run, for T after a restart)
        from pyscf.scf import chkfile
        c = chkfile.load(a.dm0_chk, 'scf')
        dms = dict.fromkeys(('vac', 'emb'), (c['mo_coeff'] * c['mo_occ']) @ c['mo_coeff'].T)
        res['dm0_chk'] = {'file': Path(a.dm0_chk).name, 'e_tot': float(c['e_tot'])}
    for cfg in configs:
        # free the previous configuration's SCF and its three-index integrals first: held, they pushed the next
        # build out of core (brsalt whole shell, 21 Oct 2026: 161 GB in memory + 161 GB read from disk per cycle)
        mf = df_shared = None
        gc.collect()
        syms, xyz, charge, npair, mm_set = emb_config(a, d, meta, cfg)
        name = f'{cfg}_{a.which}'
        mol = mk_mol(syms, xyz, charge, a.mem, n_dense=npair if ldb else None)
        conv = 1e-8 if a.which == 'cluster' else 1e-9
        df_shared = None
        for env in ('vac', 'emb'):
            k = f'{name}_{env}'
            # vacuum: not needed for N and Tr (field rungs only) nor for pairout/pairshellq (it is R1f of emb_pair)
            if k in res['runs'] or (env == 'vac' and (a.field_only or (a.which == 'pair' and cfg in ('N', 'Tr'))
                                                     or a.which in ('pairout', 'pairshellq'))):
                continue
            mm = mm_set if env == 'emb' else None
            mf, e, ex = run_dft(mol, 'pbe0', conv=conv, mm=mm, dm0=dms.get(env), with_df=df_shared, shift=a.level_shift)
            df_shared = mf.with_df
            r = {'E': e, 'converged': ex['converged'], 'nao': mol.nao, 'natm': mol.natm, 'charge': charge}
            if a.level_shift:
                r['level_shift_Eh'] = a.level_shift
            if a.which == 'pairshellq':
                r['radii_from'] = 'S'
            if mm is not None:
                ei, el, nu = e_int(mf, mol, mm)
                r.update(E_int=ei, E_int_el=el, E_int_nuc=nu, n_charges=int(len(mm[1])), sum_charges=float(np.sum(mm[1])))
            dms[env] = mf.make_rdm1()
            res['runs'][k] = r
            save_json(out, res)
            print(f'[optc.mol] {a.key} {k}: E {e:.8f}' + (f'  E_int {r["E_int"]:.6f}' if mm is not None else ''), flush=True)
    R = res['runs']
    kj = HARTREE_KJ
    w = a.which
    der = {}
    if f'T_{w}_vac' in R and f'S_{w}_vac' in R:
        der['vacuum_kJ'] = round((R[f'T_{w}_vac']['E'] - R[f'S_{w}_vac']['E']) * kj, 3)
    if f'T_{w}_emb' in R and f'S_{w}_emb' in R:
        der['embedded_single_site_kJ'] = round((R[f'T_{w}_emb']['E'] - R[f'S_{w}_emb']['E']) * kj, 3)
    if w == 'pair' and 'N_pair_emb' in R and 'S_pair_emb' in R:
        eN = R['N_pair_emb']['E'] - R['N_pair_emb']['E_int'] / 2
        eS = R['S_pair_emb']['E'] - R['S_pair_emb']['E_int'] / 2
        der['embedded_collective_kJ'] = round((eN - eS) * kj, 3)
    # reverse vertical gaps: E(P4 pair) - E(P0 pair), both in the P4 environment (optc/PLAN.md 9:40 PM item B)
    for env, fwd, rk in (('emb', 'embedded_single_site_kJ', 'reverse_kJ'), ('vac', 'vacuum_kJ', 'reverse_vacuum_kJ')):
        if f'N_{w}_{env}' in R and f'Tr_{w}_{env}' in R:
            der[rk] = round((R[f'N_{w}_{env}']['E'] - R[f'Tr_{w}_{env}']['E']) * kj, 3)
            if fwd in der:
                sfx = '' if env == 'emb' else '_vacuum'
                der[f'lr_mean{sfx}_kJ'] = round((der[fwd] + der[rk]) / 2, 3)
                der[f'lambda{sfx}_kJ'] = round((der[fwd] - der[rk]) / 2, 3)
    res['derived'] = der
    save_json(out, res)
    print(f'[optc.mol] {a.key} emb {w}: {der}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['req', 'pair', 'pcm', 'emb', 'runner', 'monomers', 'cc', 'embcorr', 'hscanpair'])
    ap.add_argument('--point', default='P0', help='hscanpair: the frame (P0 or P4r)')
    ap.add_argument('--points', default='', help='cc: comma-separated points (default P0,P4,P4r)')
    ap.add_argument('--wb97x', action='store_true', help='hscanpair: add wB97X-D3(BJ) (to finished runs too)')
    ap.add_argument('--configs', default='S,T,N', help='embcorr: configurations of emb.json, S always included '
                    '(S,T,N; R,NR,VS,VR after embed.py build --configs R_pair,...)')
    ap.add_argument('--pbe50', action='store_true', help='pair: add PBE50-D3(BJ) at every point')
    ap.add_argument('--key', required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('DFT_THREADS', 8)))
    ap.add_argument('--mem', type=int, default=int(os.environ.get('DFT_MAX_MEMORY_MB', 16000)))
    ap.add_argument('--radii-scale', type=float, default=1.2)
    ap.add_argument('--which', choices=['pair', 'cluster', 'pairout', 'pairshellq'], default='pair')
    ap.add_argument('--reverse', action='store_true', help='emb pair: add Tr, the P0 pair in the P4 environment')
    ap.add_argument('--vac-from', default='', help='emb cluster: take the vacuum runs from this model (same geometry)')
    ap.add_argument('--model', default='repeat', help='folder under emb/: repeat, sc, gas, or e.g. repeat_rc35')
    ap.add_argument('--ldb', action='store_true', help='emb cluster: def2-TZVP on the pair, def2-SVP on the shell')
    ap.add_argument('--max-atoms', type=int, default=0, help='emb cluster: skip (with a record) above this many QM atoms')
    ap.add_argument('--level-shift', type=float, default=0.0, help='emb: level shift (Eh) as an SCF convergence aid')
    ap.add_argument('--field-only', default='', help='emb: skip the vacuum runs; the text (the reason) is recorded in the result')
    ap.add_argument('--dm0-chk', default='', help='emb: initial density of the first SCF from this PySCF chkfile (same atoms)')
    ap.add_argument('--basis', default='def2-tzvp', help='pair stage: basis of a check run (written to pair_<basis>.json)')
    a = ap.parse_args()
    {'req': stage_req, 'pair': stage_pair, 'pcm': stage_pcm, 'emb': stage_emb, 'runner': stage_runner,
     'monomers': stage_monomers, 'cc': stage_cc, 'embcorr': stage_embcorr, 'hscanpair': stage_hscanpair}[a.stage](a)


if __name__ == '__main__':
    main()
