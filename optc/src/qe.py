#!/usr/bin/env python3
"""Periodic rung R5 of the option C series (optc/PLAN.md) with Quantum ESPRESSO 7.5, one crystal at a time.

    tools/qe/bin/python optc/src/qe.py <stage> --key KEY [--np 12] [--point P2]
stages
    relax     S = the observed state: fixed X-ray cell, BFGS, from the CIF with X-H set to neutron lengths
    point     one constrained point P1..P4 (--point): every copy of the transferring H held by a pw.x 'distance'
              constraint, everything else relaxed (damped dynamics; QE refuses constraints with BFGS)
                P4 = the other end: H at r_eq from the acceptor A (r_eq = the receiving partner's isolated D-H
                     length, optc/results/<key>/req.json, written by optc/src/mol.py req)
                Pk (k = 1, 2, 3): d(D-H) = d_S + k/4 (L_S - r_eq - d_S), d_S and L_S = D-H and D...A at S
              seeds: S with each copy's H moved along its D->A line to the target (all points independent)
    hybrid    PBE-D3(BJ) and PBE0-D3(BJ) SCFs with SG15 at 100/400 Ry at one path point (--point P0..P4, P0 = S);
              the PBE0 path = PBE-D3/SSSP path + [E(PBE0) - E(PBE)]_SG15 (same rule as the SAPH paper)
    esp       the PBE0-D3/SG15 density at P0 or P4 kept, pp.x total electrostatic potential (V_bare + V_H) as a
              cube for the periodic charge fit (optc/src/charges.py)
    kcheck    SSSP SCF at S and P4 on the next denser k mesh (n_i + 1 for every axis)
    p4r       an unconstrained BFGS relax started from the converged P4: is the other state a minimum at PBE-D3?
    vc        cell check (saph, dapsone, mppcp): at 80/640 Ry, (a) S and P4 re-relaxed in the X-ray cell, (b) a
              variable-cell BFGS relax of S (zero pressure), (c) P4 constrained in that cell; the shift of E(P4) - E(P0)
              between the two cells is the cell uncertainty component (QE allows constraints only at fixed cell)
    box       bridge gate G2b: the isolated pair of mol.py pair (P0 and P4 geometries) in a cubic box with 12 A of vacuum
              (Martyna-Tuckerman, Gamma point), PBE-D3(BJ) and PBE0-D3(BJ) with SG15 at 100/400 Ry; its E(P4) - E(P0)
              must equal PySCF's PBE0-D3(BJ)/def2-TZVP value within 4 kJ/mol
    summary   optc/results/<key>/periodic.json: the path, per formula unit, PBE and PBE0, with the checks
  additions of 19 Oct 2026, 9:40 PM IST (optc/PLAN.md items C, D, E, K, P):
    xpoint    an extra constrained point: every copy's d(D-H) held at --dh, seeded from --seed (P0 or P4r) with each H
              moved along its D->A line; tag path_<name>, <name>.extxyz (--name X<dh*1000><r|s>, e.g. X1300r)
    hess      partial Hessian of copy 0's H, D and A at --point (P0, P4r, or an extra point): central differences of
              the PBE-D3(BJ)/SSSP forces, step 0.01 A, 18 SCFs; harmonic frequencies and zero-point energies (H and D)
    vcfree    variable-cell BFGS (80/640 Ry, zero pressure) from --point (P4r, or P0 = vc_S of stage vc), its 80 Ry
              fixed-cell reference, and SG15 PBE/PBE0 single points in the relaxed cell; vc<S|P4r>.extxyz
    hscan     the proton alone moved along each copy's D->A line (all copies in phase) in the frozen heavy-atom frame
              of --point (P0 or P4r), 11 PBE-D3/SSSP single points from -0.20 to +0.30 A: the potential of the proton
              stretch for the 1D DVR (heavy atoms fixed; the relaxed path is the adiabatic limit)
    vert      periodic single points without relaxation at the cluster configurations: T1 (copy 0's pair at its P4
              geometry, aligned as in embed.py, all else at P0), T1v (copy 0's H only at r_eq from A), Nv (every
              copy's H at r_eq from A), T2 (T1 in the 2x supercell along the axis with k = 2, PBE only)
    hybrid    --exx 0.5: PBE50 (exx_fraction 0.5, tags pbe50_sg15_*); --nq 1,1,2: exact-exchange q mesh check
    kcheck    --point P4r: the denser mesh at P4r too
Level: PBE-D3(BJ) two-body (dftd3_version 4, three-body off), SSSP PBE efficiency 1.3.0 at 70/560 Ry; fixed X-ray
cell; Monkhorst-Pack mesh n_i = ceil(9.0 A / |a_i|), unshifted.  Parsers are those of the SAPH paper
(dft/periodic_qe.py parse_pw / parse_hybrid), imported unchanged.
"""
import argparse, datetime, hashlib, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

import numpy as np

OPTC = Path(__file__).resolve().parent.parent
ROOT = OPTC.parent
sys.path.insert(0, str(ROOT / 'dft'))
from periodic_qe import parse_pw, parse_hybrid, fortran  # noqa: E402  (the SAPH paper's tested parsers)

QE = Path(os.environ.get('OPTC_QE', ROOT / 'tools/qe'))
RES = Path(os.environ.get('OPTC_RESULTS', OPTC / 'results'))
SCRATCH = Path(os.environ.get('OPTC_SCRATCH', '/mnt/scratch/optc' if os.path.isdir('/mnt/scratch') else ROOT / 'scratch/optc'))
RY_EV, KJMOL_PER_EV, BOHR_A = 13.605693122994, 96.485332, 0.529177210903

# SSSP PBE efficiency 1.3.0 (SSSP_1.3.0_PBE_efficiency.json, Materials Cloud record rcyfm-68h65)
SSSP = {'H': ('H.pbe-rrkjus_psl.1.0.0.UPF', 'f52b6d4d1c606e5624b1dc7b2218f220'),
        'C': ('C.pbe-n-kjpaw_psl.1.0.0.UPF', '5d2aebdfa2cae82b50a7e79e9516da0f'),
        'N': ('N.pbe-n-radius_5.UPF', '16739722b17309cd8fe442a2ace49922'),
        'O': ('O.pbe-n-kjpaw_psl.0.1.UPF', '0234752ac141de4415c5fc33072bef88'),
        'P': ('P.pbe-n-rrkjus_psl.1.0.0.UPF', '8b930f418f0a4573dca56cd030ffe088'),
        'S': ('s_pbe_v1.4.uspp.F.UPF', '88d86576ff6df21479756cfb9bdac1df'),
        'Cl': ('cl_pbe_v1.4.uspp.F.UPF', 'fc6f6913ecf08c9257cb748ef0700058'),
        'Br': ('br_pbe_v1.4.uspp.F.UPF', 'd3ffb7b29f6225aa16fe06858fb2a80b')}
# SG15 ONCV v1.2 (sg15_oncv_upf_2020-02-06.tar.gz, md5 0d72299c3962fb9de17a22d81cb79734; Cl, Br md5 on the Pi 18 Oct 2026)
SG15 = {'H': ('H_ONCV_PBE-1.2.upf', 'bf21e08b90ac4ce7d0149ac4e8138190'),
        'C': ('C_ONCV_PBE-1.2.upf', '1a5f83a7b1f58d24996abe00ed223ac6'),
        'N': ('N_ONCV_PBE-1.2.upf', 'c233fa8df51455e992258b7c3dee8a9f'),
        'O': ('O_ONCV_PBE-1.2.upf', 'e36c48deb68f4e355fffe4602be2c42c'),
        'P': ('P_ONCV_PBE-1.2.upf', '0688a67b2b263c6b42d3d97c42745426'),
        'S': ('S_ONCV_PBE-1.2.upf', 'f9d020296ee6110346977bfd66d424f6'),
        'Cl': ('Cl_ONCV_PBE-1.2.upf', '0f993535b366f580ecbcdacce103c843'),
        'Br': ('Br_ONCV_PBE-1.2.upf', '5b50323dbf1c0ae0ed9e7f1e72c3f86c')}
MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904}
ECUT = (70, 560)
ECUT_SG15 = (100, 400)
K_LEN = 9.0
RELAX_FORC, RELAX_ETOT = 5e-4, 1e-5          # as relax_i of the SAPH paper
PATH_FORC, PATH_ETOT = 1e-3, 1e-4            # as its constrained profile
CONV_THR, HYB_CONV_THR = 1e-9, 1e-6
# starting X-H lengths for the X-ray structures (Allen & Bruno, Acta B 66 (2010) 380, neutron means, rounded)
XH_START = {'C': 1.083, 'N': 1.030, 'O': 0.983}
POINTS = ('P0', 'P1', 'P2', 'P3', 'P4')
# smoke tests only (the Pi): OPTC_SMOKE="ecut,ecutrho,ecut_sg15,ecutrho_sg15,nstep" with a Gamma-point mesh; every
# run records its setting in runs.jsonl, and summary refuses to mix settings
SMOKE = os.environ.get('OPTC_SMOKE')
if SMOKE:
    _s = [int(x) for x in SMOKE.split(',')]
    ECUT, ECUT_SG15, SMOKE_NSTEP = (_s[0], _s[1]), (_s[2], _s[3]), _s[4]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def note(out, text):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'STAGES.md', 'a') as f:
        f.write(f'- {utc()}  {text}\n')
    print(f'[optc.qe] {text}', flush=True)


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class Crystal:
    def __init__(self, key):
        self.key = key
        self.rec = json.loads((OPTC / 'structures' / key / 'cell.json').read_text())
        self.lat = np.array(self.rec['lattice_A'])
        self.syms = self.rec['species']
        self.frac0 = np.array(self.rec['frac'])
        self.nat = len(self.syms)
        self.copies = self.rec['copies']
        self.nfu = len(self.copies) if self.copies else int(self.rec['meta']['Z'])
        self.out = RES / key
        L = np.linalg.norm(self.lat, axis=1)
        self.kgrid = (1, 1, 1) if SMOKE else tuple(int(math.ceil(K_LEN / x)) for x in L)
        self.ops = self.rec['meta']['symops']

    def cart(self, F):
        return np.asarray(F) @ self.lat

    def frac(self, X):
        return np.asarray(X) @ np.linalg.inv(self.lat)

    def mic(self, X, i, j):
        """minimum-image vector X[j] - X[i] (cartesian)."""
        F = self.frac(X)
        best = None
        df = F[j] - F[i]
        df -= np.round(df)
        for s in np.ndindex(3, 3, 3):
            v = (df + np.array(s) - 1) @ self.lat
            if best is None or v @ v < best @ best:
                best = v
        return best

    def d(self, X, i, j):
        return float(np.linalg.norm(self.mic(X, i, j)))


def irreducible_k(cr, grid):
    """Irreducible points of an unshifted MP grid under the CIF point group plus time reversal (pool sizing)."""
    import gemmi
    n = np.array(grid)
    rots = []
    for t in cr.ops:
        op = gemmi.Op(t)
        W = np.array([[op.rot[i][j] / op.DEN for j in range(3)] for i in range(3)])
        R = np.linalg.inv(W).T
        rots += [R, -R]
    seen, count = set(), 0
    for idx in np.ndindex(*grid):
        k = np.array(idx) / n
        key = tuple(np.round((k % 1) * n).astype(int) % n)
        if key in seen:
            continue
        count += 1
        for R in rots:
            seen.add(tuple(np.round(((R @ k) % 1) * n).astype(int) % n))
    return count


def pools(np_ranks, nirr):
    divs = [d for d in range(1, np_ranks + 1) if np_ranks % d == 0 and d <= nirr]
    even = [d for d in divs if nirr % d == 0]
    return max(even) if even and max(even) * 2 > max(divs) else max(divs)


def pw_input(cr, X, *, calc, prefix, outdir, pseudo_dir, pseudos, ecut, kgrid, conv_thr=CONV_THR, forc=None, etot=None,
             dyn='bfgs', constraints=(), hybrid=False, nq=(1, 1, 1), disk_io='low', nstep=300, max_seconds=None, box=False,
             exx_fraction=None, mixing=None):
    species = [s for s in pseudos if s in cr.syms]
    ctrl = [f"calculation = '{calc}'", f"prefix = '{prefix}'", f"outdir = '{outdir}'", f"pseudo_dir = '{pseudo_dir}'",
            "tprnfor = .true.", "tstress = .false.", f"disk_io = '{disk_io}'", "verbosity = 'low'"]
    if max_seconds:
        ctrl.append(f'max_seconds = {int(max_seconds)}')
    if calc in ('relax', 'vc-relax'):
        ctrl += [f'forc_conv_thr = {fortran(forc)}', f'etot_conv_thr = {fortran(etot)}',
                 f'nstep = {SMOKE_NSTEP if SMOKE else nstep}']
    sysn = ['ibrav = 0', f'nat = {cr.nat}', f'ntyp = {len(species)}', f'ecutwfc = {ecut[0]}', f'ecutrho = {ecut[1]}',
            "occupations = 'fixed'", "vdw_corr = 'dft-d3'", 'dftd3_version = 4', 'dftd3_threebody = .false.']
    if hybrid:
        sysn += ["input_dft = 'pbe0'", f'nqx1 = {nq[0]}', f'nqx2 = {nq[1]}', f'nqx3 = {nq[2]}']
        if exx_fraction is not None:
            sysn += [f'exx_fraction = {exx_fraction}']
    if box:
        sysn += ["assume_isolated = 'mt'"]
    elec = [f'conv_thr = {fortran(conv_thr)}', 'mixing_beta = 0.4', 'electron_maxstep = 200']
    if mixing:
        # (mixing_mode, mixing_beta, electron_maxstep) for a run that sloshes with the default plain mixing
        elec = [f'conv_thr = {fortran(conv_thr)}', f"mixing_mode = '{mixing[0]}'", f'mixing_beta = {mixing[1]}',
                f'electron_maxstep = {mixing[2]}']
    L = ['&CONTROL'] + [f'  {x}' for x in ctrl] + ['/', '&SYSTEM'] + [f'  {x}' for x in sysn] + ['/', '&ELECTRONS'] + \
        [f'  {x}' for x in elec] + ['/']
    if calc == 'relax':
        L += ['&IONS', f"  ion_dynamics = '{dyn}'", '/']
    if calc == 'vc-relax':
        L += ['&IONS', "  ion_dynamics = 'bfgs'", '/', '&CELL', "  cell_dynamics = 'bfgs'", "  cell_dofree = 'all'",
              '  press = 0.0', '  press_conv_thr = 0.5', '/']
    L += ['ATOMIC_SPECIES'] + [f'  {s} {MASS[s]:.4f} {pseudos[s][0]}' for s in species]
    L += ['CELL_PARAMETERS angstrom'] + ['  ' + ' '.join(f'{v:16.10f}' for v in row) for row in cr.lat]
    F = cr.frac(X)
    L += ['ATOMIC_POSITIONS crystal'] + [f'  {s:2s} ' + ' '.join(f'{v:15.10f}' for v in f) for s, f in zip(cr.syms, F)]
    L += ['K_POINTS gamma'] if box else ['K_POINTS automatic', '  {} {} {} 0 0 0'.format(*kgrid)]
    if constraints:
        if calc != 'relax' or dyn == 'bfgs':
            raise ValueError('pw.x 7.5 applies CONSTRAINTS only in damped dynamics / FIRE')
        L += ['CONSTRAINTS', f'  {len(constraints)}'] + [f"  'distance' {i + 1} {j + 1} {t / BOHR_A:.10f}"
                                                         for i, j, t in constraints]
    return '\n'.join(L) + '\n'


def read_input_back(txt, nat):
    """G1 round trip: the lattice (A) and fractional positions as pw.x will read them."""
    lines = txt.splitlines()
    i = lines.index('CELL_PARAMETERS angstrom')
    lat = np.array([[float(v) for v in lines[i + k].split()] for k in (1, 2, 3)])
    j = lines.index('ATOMIC_POSITIONS crystal')
    F = np.array([[float(v) for v in lines[j + 1 + k].split()[1:4]] for k in range(nat)])
    return lat, F


def check_pseudos(pdir, table, syms):
    for s in sorted(set(syms)):
        f, h = table[s]
        p = Path(pdir) / f
        if not p.exists() or md5(p) != h:
            raise SystemExit(f'pseudopotential {p} missing or md5 mismatch')


def run_pw(a, cr, tag, X, *, calc, ecut=ECUT, kgrid=None, pseudos=SSSP, pdir=None, hybrid=False, keep_outdir=False,
           **kw):
    """Run (or reuse) one pw.x calculation in <results>/<key>/runs/<tag>/; returns the parsed output."""
    kgrid = kgrid or cr.kgrid
    pdir = pdir or (QE / ('pseudo_sg15' if pseudos is SG15 else 'pseudo'))
    check_pseudos(pdir, pseudos, cr.syms)
    rd = cr.out / 'runs' / tag
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / f'{tag}.out'

    def finished(p):
        return (p['job_done'] and not p['scf_failed'] and (calc not in ('relax', 'vc-relax') or p['relax_converged'] or bool(SMOKE))
                and (not hybrid or p['hybrid']['exx_converged']))
    if out.exists():
        p = parse_pw(out, cr.nat)
        if hybrid:
            p['hybrid'] = parse_hybrid(out)
        if finished(p) and (not keep_outdir or (SCRATCH / cr.key / tag).exists()):
            note(cr.out, f'{tag}: finished earlier, reused')
            return p
        n = len(list(rd.glob(f'{tag}.out.part*')))
        out.rename(rd / f'{tag}.out.part{n + 1}')
        if calc == 'relax' and p['positions_crystal']:
            X = np.array(p['positions_crystal']) @ cr.lat
            note(cr.out, f'{tag}: restarting from the last geometry of part {n + 1}')
    scr = SCRATCH / cr.key / tag
    shutil.rmtree(scr, ignore_errors=True)
    scr.mkdir(parents=True, exist_ok=True)
    inp = rd / f'{tag}.in'
    txt = pw_input(cr, X, calc=calc, prefix=tag, outdir=scr, pseudo_dir=pdir, pseudos=pseudos, ecut=ecut, kgrid=kgrid,
                   hybrid=hybrid, **kw)
    lat2, F2 = read_input_back(txt, cr.nat)
    if np.abs(lat2 - cr.lat).max() > 1e-8 or np.abs(F2 @ lat2 - X).max() > 1e-6:
        raise SystemExit(f'{tag}: pw.x input does not reproduce the structure')
    inp.write_text(txt)
    nirr = irreducible_k(cr, kgrid)
    nk = pools(a.np, nirr)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    cmd = [str(QE / 'bin/mpirun'), '-np', str(a.np), '--bind-to', 'none', '--oversubscribe',
           str(QE / 'bin/pw.x'), '-nk', str(nk), '-nd', '1', '-in', str(inp)]
    note(cr.out, f'{tag}: start {calc} {"PBE0 " if hybrid else ""}ecut {ecut[0]}/{ecut[1]} k {kgrid} ({nirr} irr) '
         f'np {a.np} nk {nk} host {os.uname().nodename}')
    t0 = time.time()
    with open(out, 'w') as fo:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=rd, env=env, start_new_session=True)
        try:
            rc = proc.wait()
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                os.killpg(proc.pid, signal.SIGKILL)
            raise
    wall = time.time() - t0
    p = parse_pw(out, cr.nat)
    if hybrid:
        p['hybrid'] = parse_hybrid(out)
    rec = {'tag': tag, 'calc': calc, 'hybrid': hybrid, 'ecut': ecut, 'kgrid': kgrid, 'nk_irr': p['nk_irr'], 'np': a.np,
           'nk': nk, 'rc': rc, 'wall_s': round(wall, 1), 'n_scf': len(p['scf_iterations']),
           'energy_Ry': p['hybrid']['energy_Ry'] if hybrid else p['energy_Ry'], 'utc': utc(), 'host': os.uname().nodename}
    with open(cr.out / 'runs.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    if not keep_outdir:
        shutil.rmtree(scr, ignore_errors=True)
    ok = rc == 0 and finished(p)
    note(cr.out, f'{tag}: {"done" if ok else "FAILED"} rc={rc} wall {wall / 60:.1f} min E = {rec["energy_Ry"]} Ry')
    if not ok:
        raise RuntimeError(f'pw.x {tag} failed (rc={rc}); see {out}')
    return p


# ----------------------------------------------------------------------------------------------- geometry
def start_geometry(cr):
    """CIF positions with X-H set to the neutron means along the X->H direction (X-ray structures only)."""
    X = cr.cart(cr.frac0)
    if 'neutron' in (cr.rec['meta'].get('radiation', '') + cr.rec['meta'].get('probe', '')).lower():
        return X, 'neutron positions as deposited'
    hpar = {}
    for i, s in enumerate(cr.syms):
        if s != 'H':
            continue
        best = min(((cr.d(X, i, j), j) for j in range(cr.nat) if cr.syms[j] != 'H'))
        hpar[i] = best[1]
    for h, x in hpar.items():
        v = cr.mic(X, x, h)
        r = XH_START.get(cr.syms[x])
        if r:
            X[h] = X[x] + v / np.linalg.norm(v) * r
    return X, 'X-H set to neutron means (C 1.083, N 1.030, O 0.983 A) along X->H'


def final_geometry(cr, p):
    return np.array(p['positions_crystal']) @ cr.lat


def geometry(cr, tag):
    """converged geometry of an earlier run (cartesian, A)."""
    out = cr.out / 'runs' / tag / f'{tag}.out'
    p = parse_pw(out, cr.nat)
    if not p['relax_converged'] and not SMOKE:
        raise SystemExit(f'{tag} is not converged')
    return final_geometry(cr, p)


def path_targets(cr, XS):
    """Target distances (A) of the constrained points from the relaxed S and r_eq (req.json)."""
    req = json.loads((cr.out / 'req.json').read_text())
    r_eq = req['r_eq_A']
    dS = [cr.d(XS, c['D'], site_proton(cr, XS, c)) for c in cr.copies]
    LS = [cr.d(XS, c['D'], c['A']) for c in cr.copies]
    d_s, L_s = float(np.mean(dS)), float(np.mean(LS))
    t = {'P4': ('A-H', r_eq)}
    for k in (1, 2, 3):
        t[f'P{k}'] = ('D-H', d_s + k / 4 * (L_s - r_eq - d_s))
    return t, {'d_S': d_s, 'L_S': L_s, 'r_eq': r_eq, 'r_eq_source': req.get('source'), 'dS_copies': dS, 'LS_copies': LS}


def held_targets(cr, f, kind):
    """the distances a constrained run held, read from the CONSTRAINTS block of its pw.x input f (bohr -> A): copy 0's
    constraint on its D (kind 'D-H') or A ('A-H') as (kind, d), and any other constraint on that H (the off-path rule's
    H...A) as extra [(label, d)].  None if f holds no constraint.  path_targets() recomputes the targets from today's
    site_proton, which can differ from what ran (sulfamate P1-P3 held d(N4-H41); gap map B7, 22 Oct 2026)."""
    if not f.exists():
        return None
    L = [l.strip() for l in f.read_text().splitlines()]
    if 'CONSTRAINTS' not in L:
        return None
    i = L.index('CONSTRAINTS')
    cons = [(int(w[1]) - 1, int(w[2]) - 1, round(float(w[3]) * BOHR_A, 8))
            for w in (l.split() for l in L[i + 2:i + 2 + int(L[i + 1].split()[0])])]
    c = cr.copies[0]
    end = c['D'] if kind == 'D-H' else c['A']
    first = next(((j, t) for a, j, t in cons if a == end), None)
    if first is None:
        return None
    H = first[0]
    name = {c['D']: 'D-H', c['A']: 'H...A'}
    extra = [(name.get(b if a == H else a, f'{a}-{b}'), t) for a, b, t in cons if H in (a, b) and (a, b) != (end, H)]
    return {'target': (kind, first[1]), 'extra': extra}


def site_proton(cr, X, c):
    """the transferring proton of copy c in geometry X: of D's own protons (those bonded to D in the CIF, and c['H']),
    the one nearest the acceptor instance closest to D.  The protons of an NH3+ donor are equivalent and can exchange
    roles: in sulfamate (20 Oct 2026) the transferring bond is N4-H43...O21 (2.83 A) and along the computed path the
    proton labelled H41 carries it (P4r is P0 with H41 and H43 exchanged, dE -0.03 kJ/mol); for every other crystal
    this is c['H'] at every stored point (checked 20 Oct 2026)."""
    D, A = c['D'], c['A']
    X0 = cr.cart(cr.frac0)
    cand = {c['H']} | {h for h in range(cr.nat) if cr.syms[h] == 'H' and np.linalg.norm(cr.mic(X0, D, h)) < 1.3}
    tA = cr.mic(X, D, A)
    return min(sorted(cand), key=lambda h: np.linalg.norm(cr.mic(X, D, h) - tA))


def seed_point(cr, XS, kind, target):
    X = XS.copy()
    cons = []
    for c in cr.copies:
        D, H, A = c['D'], site_proton(cr, XS, c), c['A']
        u = cr.mic(XS, D, A)
        L = np.linalg.norm(u)
        u /= L
        XD = XS[D]
        XA = XD + u * L
        X[H] = XD + u * target if kind == 'D-H' else XA - u * target
        cons.append((D, H, target) if kind == 'D-H' else (A, H, target))
    # wrap the moved H back into the cell
    F = cr.frac(X)
    F = F % 1.0
    return F @ cr.lat, cons


def proton_report(cr, X):
    rows = []
    for c in cr.copies:
        D, H, A = c['D'], site_proton(cr, X, c), c['A']
        dh, ha, da = cr.d(X, D, H), cr.d(X, H, A), cr.d(X, D, A)
        v1, v2 = cr.mic(X, H, D), cr.mic(X, H, A)
        ang = math.degrees(math.acos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
        rows.append({'D-H': round(dh, 4), 'H...A': round(ha, 4), 'D...A': round(da, 4), 'angle': round(ang, 2),
                     'delta': round(dh - ha, 4)})
    return rows


def other_protons(cr, X):
    """Every H whose nearest heavy atom differs from the CIF's: a proton that moved by itself."""
    X0 = cr.cart(cr.frac0)
    moved = []
    transferring = {c['H'] for c in (cr.copies or [])}
    for i, s in enumerate(cr.syms):
        if s != 'H' or i in transferring:
            continue
        n0 = min((cr.d(X0, i, j), j) for j in range(cr.nat) if cr.syms[j] != 'H')[1]
        n1 = min((cr.d(X, i, j), j) for j in range(cr.nat) if cr.syms[j] != 'H')[1]
        if n0 != n1:
            moved.append({'H': cr.rec['labels'][i], 'from': cr.rec['labels'][n0], 'to': cr.rec['labels'][n1]})
    return moved


def write_extxyz(cr, X, path, comment=''):
    lines = [str(cr.nat), 'Lattice="' + ' '.join(f'{x:.8f}' for x in cr.lat.ravel()) + '" Properties=species:S:1:pos:R:3 '
             + f'pbc="T T T" {comment}']
    lines += [f'{s:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}' for s, x in zip(cr.syms, X)]
    Path(path).write_text('\n'.join(lines) + '\n')


def point_tag(pt):
    return {'P0': 'relax_S', 'P4r': 'relax_P4r'}.get(pt, f'path_{pt}')


# ----------------------------------------------------------------------------------------------- stages
def stage_relax(a, cr):
    X, how = start_geometry(cr)
    write_extxyz(cr, X, cr.out / 'start.extxyz', f'"{how}"')
    p = run_pw(a, cr, 'relax_S', X, calc='relax', forc=RELAX_FORC, etot=RELAX_ETOT)
    XS = final_geometry(cr, p)
    write_extxyz(cr, XS, cr.out / 'P0.extxyz', f'E_Ry={p["energy_Ry"]}')


OFFPATH_ANGLE, OFFPATH_DA = 140.0, 0.25


def stage_point(a, cr):
    XS = geometry(cr, 'relax_S')
    tg, info = path_targets(cr, XS)
    kind, target = tg[a.point]
    X, cons = seed_point(cr, XS, kind, target)
    if a.pin_ha:
        # rule of optc/PLAN.md (19 Oct 2026): a point that left the path (D-H...A < 140 deg or D...A > L_S + 0.25 A) is
        # rerun with d(H...A) held as well, at L_S - target (the proton on the D->A line); the off-path runs are kept
        if kind != 'D-H':
            raise SystemExit('--pin-ha applies to the interior points P1-P3')
        for t in (point_tag(a.point), f'pbe_sg15_{a.point}', f'pbe0_sg15_{a.point}'):
            d = cr.out / 'runs' / t
            if d.exists() and not (cr.out / 'runs' / f'{t}_offpath').exists():
                d.rename(cr.out / 'runs' / f'{t}_offpath')
        cons = cons + [(site_proton(cr, XS, c), c['A'], info['L_S'] - target) for c in cr.copies]
        note(cr.out, f'{point_tag(a.point)}: rerun with d(H...A) = {info["L_S"] - target:.4f} A held too (off-path rule)')
    p = run_pw(a, cr, point_tag(a.point), X, calc='relax', dyn='damp', constraints=cons, forc=PATH_FORC, etot=PATH_ETOT)
    write_extxyz(cr, final_geometry(cr, p), cr.out / f'{a.point}.extxyz', f'E_Ry={p["energy_Ry"]} {kind}={target:.4f}')


def stage_hybrid(a, cr):
    X = geometry(cr, point_tag(a.point))
    pb = run_pw(a, cr, f'pbe_sg15_{a.point}', X, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR)
    # P4r too since 21 Oct 2026: its charges give the field of the P4r in-field pair corrections (embed.py R/NR)
    keep = a.point in ('P0', 'P4', 'P4r') and a.keep
    nq = tuple(int(x) for x in a.nq.split(',')) if a.nq else (1, 1, 1)
    if a.exx is not None:
        # third anchor (optc/PLAN.md 9:40 PM item K): PBE50; the D3(BJ) parameters stay those of PBE0
        h = run_pw(a, cr, f'pbe{int(round(a.exx * 100))}_sg15_{a.point}', X, calc='scf', ecut=ECUT_SG15, pseudos=SG15,
                   conv_thr=HYB_CONV_THR, hybrid=True, exx_fraction=a.exx)
        return pb, h
    tag = f'pbe0_sg15_{a.point}' + (f'_nq{"".join(map(str, nq))}' if a.nq else '')
    h = run_pw(a, cr, tag, X, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR,
               hybrid=True, keep_outdir=keep, nq=nq)
    return pb, h


def stage_xpoint(a, cr):
    """extra constrained point (optc/PLAN.md 9:40 PM item D)."""
    if not re.fullmatch(r'X\d{4}[rs]', a.name or '') or a.dh is None:
        raise SystemExit('xpoint needs --name X<dh in mA><r|s> and --dh')
    XS = geometry(cr, point_tag(a.seed))
    X, cons = seed_point(cr, XS, 'D-H', a.dh)
    if a.pin_ha:
        _, info = path_targets(cr, geometry(cr, 'relax_S'))
        cons = cons + [(site_proton(cr, XS, c), c['A'], info['L_S'] - a.dh) for c in cr.copies]
    p = run_pw(a, cr, f'path_{a.name}', X, calc='relax', dyn='damp', constraints=cons, forc=PATH_FORC, etot=PATH_ETOT)
    write_extxyz(cr, final_geometry(cr, p), cr.out / f'{a.name}.extxyz', f'E_Ry={p["energy_Ry"]} D-H={a.dh:.4f} seed={a.seed}')
    note(cr.out, f'path_{a.name}: proton {proton_report(cr, final_geometry(cr, p))[0]}')


HESS_STEP = 0.01           # A
CM1_TO_KJMOL = 0.01196266  # h c N_A per cm-1, kJ/mol


def frequencies(Hm, masses):
    """harmonic wavenumbers (cm-1) of a Hessian in Ry/bohr^2 with atomic masses in amu (3 coordinates per atom)."""
    ry_j, bohr_m, amu_kg, c_cm = 2.1798723611e-18, 0.529177210903e-10, 1.66053906660e-27, 2.99792458e10
    m = np.repeat(masses, 3)
    Mw = Hm / np.sqrt(np.outer(m, m)) * ry_j / bohr_m ** 2 / amu_kg
    w2 = np.linalg.eigvalsh(Mw)
    return [float(np.sign(x) * math.sqrt(abs(x)) / (2 * math.pi * c_cm)) for x in w2]


def stage_hess(a, cr):
    """partial Hessian of copy 0's H, D, A (optc/PLAN.md 9:40 PM item D)."""
    X0 = geometry(cr, point_tag(a.point))
    c = cr.copies[0]
    atoms = [site_proton(cr, X0, c), c['D'], c['A']]
    F = {}
    for ia, i in enumerate(atoms):
        for k in range(3):
            for sgn in (1, -1):
                X = X0.copy()
                X[i, k] += sgn * HESS_STEP
                p = run_pw(a, cr, f'hess_{a.point}_{ia}{"xyz"[k]}{"p" if sgn > 0 else "m"}', X, calc='scf', conv_thr=1e-10)
                F[(ia, k, sgn)] = np.array(p['forces_Ry_bohr'])[atoms]
    Hm = np.zeros((9, 9))
    for ia in range(3):
        for k in range(3):
            Hm[3 * ia + k] = -((F[(ia, k, 1)] - F[(ia, k, -1)]) / (2 * HESS_STEP / BOHR_A)).reshape(-1)
    asym = float(np.abs(Hm - Hm.T).max())
    Hm = 0.5 * (Hm + Hm.T)
    out = {'point': a.point, 'atoms': atoms, 'labels': [cr.rec['labels'][i] for i in atoms], 'step_A': HESS_STEP,
           'hessian_Ry_bohr2': Hm.round(8).tolist(), 'asymmetry_Ry_bohr2': round(asym, 8)}
    for iso, mH in (('H', MASS['H']), ('D', 2.01410)):
        mass = [mH, MASS[cr.syms[c['D']]], MASS[cr.syms[c['A']]]]
        nu9 = frequencies(Hm, mass)
        nu3 = frequencies(Hm[:3, :3], mass[:1])
        out[iso] = {'wavenumbers_HDA_cm1': [round(x, 1) for x in nu9], 'wavenumbers_H_only_cm1': [round(x, 1) for x in nu3],
                    'zpe_HDA_kJmol': round(0.5 * CM1_TO_KJMOL * sum(x for x in nu9 if x > 0), 3),
                    'zpe_H_only_kJmol': round(0.5 * CM1_TO_KJMOL * sum(x for x in nu3 if x > 0), 3)}
    atomic_json(cr.out / f'hess_{a.point}.json', out)
    note(cr.out, f"hess {a.point}: H modes {out['H']['wavenumbers_H_only_cm1']} cm-1, ZPE(H,D,A) {out['H']['zpe_HDA_kJmol']} kJ/mol")


HSCAN = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
# both wells inside each frozen frame (optc/PLAN.md 20 Oct 2026 entry): the P0 frame reaches toward A, the P4r frame
# back toward D
HSCAN_OF = {'P0': HSCAN + (0.35, 0.40, 0.45, 0.50, 0.60),
            'P4r': (-0.60, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25) + HSCAN[:9]}


def stage_hscan(a, cr):
    """proton stretch in the frozen frame of a stationary point (optc/PLAN.md 9:40 PM item D)."""
    X0 = geometry(cr, point_tag(a.point))
    res = {'point': a.point, 'offsets_A': [], 'E_Ry': [], 'DH_A': [], 'n_fu': cr.nfu,
           'note': 'every copy\'s H moved by the offset along its own D->A unit vector, all other atoms fixed'}
    for x in HSCAN_OF.get(a.point, HSCAN):
        X = X0.copy()
        for c in cr.copies:
            u = cr.mic(X0, c['D'], c['A'])
            h = site_proton(cr, X0, c)
            X[h] = X0[h] + x * u / np.linalg.norm(u)
        tag = f"hscan_{a.point}_{'m' if x < 0 else 'p'}{abs(int(round(x * 100))):02d}"
        p = run_pw(a, cr, tag, wrap(cr, X), calc='scf')
        res['offsets_A'].append(x)
        res['E_Ry'].append(p['energy_Ry'])
        res['DH_A'].append(round(float(np.mean([cr.d(X, c['D'], site_proton(cr, X0, c)) for c in cr.copies])), 4))
        e0 = res['E_Ry'][res['offsets_A'].index(0.0)] if 0.0 in res['offsets_A'] else res['E_Ry'][0]
        res['dE_kJ_per_pair'] = [round((e - e0) * RY_EV * KJMOL_PER_EV / cr.nfu, 3) for e in res['E_Ry']]
        atomic_json(cr.out / f'hscan_{a.point}.json', res)
    note(cr.out, f"hscan {a.point}: {res['dE_kJ_per_pair']}")


def cell_copy(cr, lat):
    import copy
    cv = copy.copy(cr)
    cv.lat = np.array(lat)
    return cv


def stage_vcfree(a, cr):
    """the other state's own lattice (optc/PLAN.md 9:40 PM item E)."""
    nm = 'P4r' if a.point == 'P4r' else 'S'
    X = geometry(cr, point_tag(a.point))
    run_pw(a, cr, f'x80_{a.point}', X, calc='relax', ecut=ECUT_VC, forc=RELAX_FORC, etot=RELAX_ETOT)
    pv = run_pw(a, cr, f'vc_{nm}', X, calc='vc-relax', ecut=ECUT_VC, forc=RELAX_FORC, etot=RELAX_ETOT)
    lat = final_cell(cr.out / 'runs' / f'vc_{nm}' / f'vc_{nm}.out')
    cv = cell_copy(cr, lat)
    XV = np.array(pv['positions_crystal']) @ lat
    write_extxyz(cv, XV, cr.out / f'vc{nm}.extxyz', f'E_Ry={pv["energy_Ry"]} vc-relax from {a.point}')
    note(cr.out, f'vc_{nm}: proton {proton_report(cv, XV)[0]}, volume {100 * (abs(np.linalg.det(lat)) / abs(np.linalg.det(cr.lat)) - 1):+.2f} %')
    run_pw(a, cv, f'pbe_sg15_vc{nm}', XV, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR)
    run_pw(a, cv, f'pbe0_sg15_vc{nm}', XV, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR, hybrid=True)
    vcfree_summary(cr)


def vcfree_summary(cr):
    """vcfree.json once both states have their own lattice: E(P4r) - E(S) per formula unit, fixed X-ray cell at
    80 Ry against both relaxed cells, PBE (80/640 SSSP) and PBE0 (+ the SG15 hybrid correction in each relaxed cell)."""
    R = cr.out / 'runs'
    def e(tag, hyb=False):
        f = R / tag / f'{tag}.out'
        if not f.exists():
            return None
        return parse_hybrid(f)['energy_Ry'] if hyb else parse_pw(f, cr.nat)['energy_Ry']
    E = {t: e(t) for t in ('x80_P0', 'x80_P4r', 'vc_S', 'vc_P4r', 'pbe_sg15_vcS', 'pbe_sg15_vcP4r')}
    E.update({t: e(t, True) for t in ('pbe0_sg15_vcS', 'pbe0_sg15_vcP4r')})
    kj = RY_EV * KJMOL_PER_EV / cr.nfu
    out = {'key': cr.key, 'E_Ry': E, 'units': 'kJ/mol per formula unit, E(P4r state) - E(S state)'}
    if E['x80_P0'] is not None and E['x80_P4r'] is not None:
        out['dE_xray_cell_pbe80'] = round((E['x80_P4r'] - E['x80_P0']) * kj, 3)
    if E['vc_S'] is not None and E['vc_P4r'] is not None:
        out['dE_relaxed_cells_pbe80'] = round((E['vc_P4r'] - E['vc_S']) * kj, 3)
        if None not in (E['pbe_sg15_vcS'], E['pbe_sg15_vcP4r'], E['pbe0_sg15_vcS'], E['pbe0_sg15_vcP4r']):
            corr = (E['pbe0_sg15_vcP4r'] - E['pbe_sg15_vcP4r']) - (E['pbe0_sg15_vcS'] - E['pbe_sg15_vcS'])
            out['dE_relaxed_cells_pbe0'] = round(out['dE_relaxed_cells_pbe80'] + corr * kj, 3)
        vols = {}
        for t in ('vc_S', 'vc_P4r'):
            lat = final_cell(R / t / f'{t}.out')
            if lat is not None:
                vols[t] = round(100 * (abs(np.linalg.det(lat)) / abs(np.linalg.det(cr.lat)) - 1), 2)
        out['volume_change_pct'] = vols
    atomic_json(cr.out / 'vcfree.json', out)
    shown = {k: v for k, v in out.items() if k.startswith('dE') or k.startswith('vol')}
    note(cr.out, f'vcfree: {shown}')


class Super:
    """the cell doubled along one axis, without symmetry (for run_pw/pw_input)."""

    def __init__(self, cr, axis):
        self.key, self.out, self.ops = cr.key, cr.out, ['x,y,z']
        self.lat = cr.lat.copy()
        self.lat[axis] *= 2
        self.syms = list(cr.syms) * 2
        self.nat = 2 * cr.nat
        k = list(cr.kgrid)
        k[axis] //= 2
        self.kgrid = tuple(k)
        self.shift = cr.lat[axis]

    def frac(self, X):
        return np.asarray(X) @ np.linalg.inv(self.lat)


def wrap(cr, X):
    return (cr.frac(X) % 1.0) @ cr.lat


def doubled_T2(cr, ec, X0, W0, W4, c, ax):
    """the cell doubled along axis ax: both images of the observed structure, then ONE instance of the transferring pair
    (the donor's molecule and the acceptor's molecule that is its hydrogen-bond partner) replaced by its P4 geometry.

    The acceptor's molecule is first translated by the lattice vector that puts its A next to the donor's D (whole
    molecules are made whole one by one, so the two can sit a lattice vector apart); the image of each pair atom is
    then the one its P0 position lies in.  (The first construction, vert_T2, wrapped wrap(T1) atom by atom into image
    1: a pair that straddles the cell edge along the doubled axis was split between the images, leaving one hydrogen
    bond with two protons and its neighbour with none; found on carboxy, 20 Oct 2026, H...H 0.17 A, T2 +5706 kJ/mol.)
    Returns the positions and {"D", "H", "A"}: the image of each."""
    sc = Super(cr, ax)
    D, H, A = c['D'], site_proton(cr, W4, c), c['A']
    s_a = W0[D] + cr.mic(X0, D, A) - W0[A]
    n = cr.frac(s_a[None])[0]
    if np.abs(n - np.round(n)).max() > 1e-6:
        raise RuntimeError(f'doubled_T2: the shift of the acceptor molecule is not a lattice vector ({n})')
    dmol, amol = ec.mols[ec.mol_of[D]], ec.mols[ec.mol_of[A]]
    P0, P4 = W0.copy(), W4.copy()
    P0[amol] += s_a
    P4[amol] += s_a
    X2 = np.vstack([wrap(cr, X0), wrap(cr, X0) + sc.shift])
    im = {}
    for i in list(dmol) + list(amol):
        im[i] = int(np.floor(cr.frac(P0[[i]])[0, ax])) % 2
        X2[i + im[i] * cr.nat] = P4[i]
    # the moved proton must sit on its own acceptor (same instance) and nowhere else
    F = sc.frac(X2)
    h, a = H + im[H] * cr.nat, A + im[A] * cr.nat
    heavy = [k for k, s in enumerate(sc.syms) if s != 'H']
    d = F[heavy] - F[h]
    d -= np.round(d)
    dist = np.linalg.norm(d @ sc.lat, axis=1)
    if heavy[int(np.argmin(dist))] != a:
        raise RuntimeError('doubled_T2: the moved proton is not next to its own acceptor')
    return X2, {'D': im[D], 'H': im[H], 'A': im[A]}


def stage_vert(a, cr):
    """periodic closure of the cluster ladder (optc/PLAN.md 9:40 PM item C)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import embed as E
    ec = E.Crystal(cr.key)
    X0, X4 = geometry(cr, 'relax_S'), geometry(cr, 'path_P4')
    t, rms = E.align_translation(ec, X0, X4)
    W0 = ec.whole(X0)
    W4 = E.whole_like(ec, X4 + t, W0)
    c = cr.copies[0]
    pair_atoms = ec.mols[ec.mol_of[c['D']]] + ec.mols[ec.mol_of[c['A']]]
    r_eq = json.loads((cr.out / 'req.json').read_text())['r_eq_A']
    T1 = W0.copy()
    T1[pair_atoms] = W4[pair_atoms]
    T1v = X0.copy()
    u = cr.mic(X0, c['D'], c['A'])
    T1v[site_proton(cr, X0, c)] = X0[c['D']] + u * (1 - r_eq / np.linalg.norm(u))
    Nv, _ = seed_point(cr, X0, 'A-H', r_eq)
    cfgs = {'P0': X0, 'T1': wrap(cr, T1), 'T1v': wrap(cr, T1v), 'Nv': Nv}
    res = {'key': cr.key, 'align_rms_A': round(rms, 4), 'n_fu': cr.nfu, 'runs': {}}
    for name, X in cfgs.items():
        res['runs'][name] = {'E_pbe_Ry': run_pw(a, cr, f'vert_{name}', X, calc='scf')['energy_Ry']}
        if name in ('P0', 'T1', 'Nv'):
            # P0: the path's own SG15 runs at relax_S (same geometry) are reused
            sfx = 'P0' if name == 'P0' else f'v{name}'
            pb = run_pw(a, cr, f'pbe_sg15_{sfx}', X, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR)
            h = run_pw(a, cr, f'pbe0_sg15_{sfx}', X, calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR,
                       hybrid=True)
            res['runs'][name].update(E_pbe_sg15_Ry=pb['energy_Ry'], E_pbe0_sg15_Ry=h['hybrid']['energy_Ry'])
        atomic_json(cr.out / 'vert.json', res)
    axes = [i for i, k in enumerate(cr.kgrid) if k == 2]
    if axes:
        ax = axes[0]
        sc = Super(cr, ax)
        X2, img = doubled_T2(cr, ec, X0, W0, W4, c, ax)
        # carboxy (20 Oct 2026) sloshed with plain mixing (scf accuracy back to ~1 Ry again and again over 100
        # iterations): local-TF mixing, beta 0.2, 300 steps, conv_thr 1e-8 Ry (1e-5 kJ/mol) in this cell of twice the atoms
        res['runs']['T2'] = {'E_pbe_Ry': run_pw(a, sc, 'vert_T2s', X2, calc='scf', kgrid=sc.kgrid, conv_thr=1e-8,
                                                mixing=('local-TF', 0.2, 300))['energy_Ry'],
                             'axis': ax, 'construction': 'vert_T2s: one whole pair instance moved (20 Oct 2026)',
                             'pair_images': img}
    kj = RY_EV * KJMOL_PER_EV
    R = res['runs']
    d = {'T1_pbe_kJ': (R['T1']['E_pbe_Ry'] - R['P0']['E_pbe_Ry']) * kj,
         'T1v_pbe_kJ': (R['T1v']['E_pbe_Ry'] - R['P0']['E_pbe_Ry']) * kj,
         'Nv_pbe_kJ_per_pair': (R['Nv']['E_pbe_Ry'] - R['P0']['E_pbe_Ry']) * kj / cr.nfu}
    for name in ('T1', 'Nv'):
        corr = (R[name]['E_pbe0_sg15_Ry'] - R[name]['E_pbe_sg15_Ry']) - (R['P0']['E_pbe0_sg15_Ry'] - R['P0']['E_pbe_sg15_Ry'])
        div = cr.nfu if name == 'Nv' else 1
        d[f'{name}_pbe0_kJ' + ('_per_pair' if name == 'Nv' else '')] = d[f'{name}_pbe_kJ' + ('_per_pair' if name == 'Nv' else '')] + corr * kj / div
    if 'T2' in R:
        d['T2_pbe_kJ'] = (R['T2']['E_pbe_Ry'] - 2 * R['P0']['E_pbe_Ry']) * kj
    d['cooperativity_vertical_pbe_kJ'] = d['Nv_pbe_kJ_per_pair'] - d['T1v_pbe_kJ']
    res['derived'] = {k: round(v, 3) for k, v in d.items()}
    res['units'] = 'kJ/mol: T1, T1v, T2 per moved pair (one pair of the cell, or of the supercell); Nv per pair'
    atomic_json(cr.out / 'vert.json', res)
    note(cr.out, f"vert: {res['derived']}")


def stage_runner(a, cr):
    """periodic analogue of the pair-level runner-up check (optc/PLAN.md item G; added 21 Oct 2026 after the pair-level
    chloro value, -63 kJ/mol, came out larger than chloro's periodic margin): in relax_S, the runner-up proton of
    cell.json placed at r_eq from its acceptor on the D->A line through H (the acceptor instance cell.py chose), against
    T1v of stage_vert (copy 0's transferring proton moved the same way on its own bond).  PBE (SSSP) always; with
    --pbe0 also [PBE0 - PBE] from SG15 runs of both vertical configurations."""
    ru = cr.rec.get('runner_up')
    if not ru:
        raise SystemExit(f'{cr.key}: no runner-up bond in cell.json')
    vert = json.loads((cr.out / 'vert.json').read_text())['runs']
    X0 = geometry(cr, 'relax_S')
    r_eq = json.loads((cr.out / 'req.json').read_text())['r_eq_A']
    c = cr.copies[0]
    T1v = X0.copy()
    u = cr.mic(X0, c['D'], c['A'])
    T1v[site_proton(cr, X0, c)] = X0[c['D']] + u * (1 - r_eq / np.linalg.norm(u))
    Rv = X0.copy()
    ur = cr.mic(X0, ru['D'], ru['H']) + cr.mic(X0, ru['H'], ru['A'])
    Rv[ru['H']] = X0[ru['D']] + ur * (1 - r_eq / np.linalg.norm(ur))
    near = sorted((float(np.linalg.norm(cr.mic(Rv, ru['H'], j))), j) for j in range(len(Rv)) if j != ru['H'])
    if near[0][1] != ru['A'] or near[0][0] > r_eq + 1e-6 or near[1][0] < 0.9:
        raise SystemExit(f'{cr.key}: runner-up proton not on its own acceptor after the move: nearest {near[:3]}')
    res = {'key': cr.key, 'runner_up': ru, 'r_eq_A': r_eq, 'n_fu': cr.nfu,
           'moved_proton_nearest_A': [[round(dd, 4), j] for dd, j in near[:3]], 'runs': {}}
    res['runs']['Rv'] = {'E_pbe_Ry': run_pw(a, cr, 'vert_Rv', wrap(cr, Rv), calc='scf')['energy_Ry']}
    kj = RY_EV * KJMOL_PER_EV
    E0, ET = vert['P0']['E_pbe_Ry'], vert['T1v']['E_pbe_Ry']
    d = {'Rv_pbe_kJ': (res['runs']['Rv']['E_pbe_Ry'] - E0) * kj, 'T1v_pbe_kJ': (ET - E0) * kj}
    d['runner_minus_chosen_pbe_kJ'] = d['Rv_pbe_kJ'] - d['T1v_pbe_kJ']
    if a.pbe0:
        for name, X in (('Rv', Rv), ('T1v', T1v)):
            pb = run_pw(a, cr, f'pbe_sg15_v{name}', wrap(cr, X), calc='scf', ecut=ECUT_SG15, pseudos=SG15,
                        conv_thr=HYB_CONV_THR)
            h = run_pw(a, cr, f'pbe0_sg15_v{name}', wrap(cr, X), calc='scf', ecut=ECUT_SG15, pseudos=SG15,
                       conv_thr=HYB_CONV_THR, hybrid=True)
            res['runs'].setdefault(name, {}).update(E_pbe_sg15_Ry=pb['energy_Ry'], E_pbe0_sg15_Ry=h['hybrid']['energy_Ry'])
            atomic_json(cr.out / 'runner_periodic.json', res)
        R = res['runs']
        corr = (R['Rv']['E_pbe0_sg15_Ry'] - R['Rv']['E_pbe_sg15_Ry']) - (R['T1v']['E_pbe0_sg15_Ry'] - R['T1v']['E_pbe_sg15_Ry'])
        d['runner_minus_chosen_pbe0_kJ'] = d['runner_minus_chosen_pbe_kJ'] + corr * kj
    res['derived'] = {k: round(v, 3) for k, v in d.items()}
    res['units'] = 'kJ/mol per moved proton (one proton of the cell), relative to P0; T1v from vert.json'
    atomic_json(cr.out / 'runner_periodic.json', res)
    note(cr.out, f"runner (periodic, vertical): {res['derived']}")


def stage_esp(a, cr):
    """pp.x on the kept PBE0 density: total electrostatic potential (plot_num 11) as a Gaussian cube."""
    tag = f'pbe0_sg15_{a.point}'
    scr = SCRATCH / cr.key / tag
    if not scr.exists():
        a.keep = True
        stage_hybrid(a, cr)
    rd = cr.out / 'runs' / f'esp_{a.point}'
    rd.mkdir(parents=True, exist_ok=True)
    cube = rd / f'vtot_{a.point}.cube'
    inp = rd / 'pp.in'
    inp.write_text(f"&INPUTPP\n  prefix = '{tag}'\n  outdir = '{scr}'\n  filplot = '{rd / 'vtot.dat'}'\n  plot_num = 11\n/\n"
                   f"&PLOT\n  iflag = 3\n  output_format = 6\n  fileout = '{cube}'\n/\n")
    cmd = [str(QE / 'bin/mpirun'), '-np', str(min(a.np, 8)), '--bind-to', 'none', '--oversubscribe', str(QE / 'bin/pp.x'),
           '-in', str(inp)]
    with open(rd / 'pp.out', 'w') as fo:
        rc = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=rd).returncode
    (rd / 'vtot.dat').unlink(missing_ok=True)
    if rc != 0 or not cube.exists():
        raise RuntimeError(f'pp.x failed for {tag} (rc {rc})')
    subprocess.run(['gzip', '-f', str(cube)], check=True)
    shutil.rmtree(scr, ignore_errors=True)
    note(cr.out, f'esp_{a.point}: cube written ({cube.name}.gz)')


def stage_p4r(a, cr):
    X = geometry(cr, 'path_P4')
    p = run_pw(a, cr, 'relax_P4r', X, calc='relax', forc=RELAX_FORC, etot=RELAX_ETOT)
    XR = final_geometry(cr, p)
    write_extxyz(cr, XR, cr.out / 'P4r.extxyz', f'E_Ry={p["energy_Ry"]}')
    rows = proton_report(cr, XR)
    note(cr.out, f'relax_P4r: proton {"stays on the acceptor" if all(r["delta"] > 0 for r in rows) else "returns to the donor"} '
         f'(delta {[r["delta"] for r in rows]})')


ECUT_VC = (80, 640)


def final_cell(path):
    """last CELL_PARAMETERS block of a vc-relax output, in A."""
    txt = Path(path).read_text(errors='replace')
    blocks = re.findall(r'CELL_PARAMETERS \((alat=\s*[\d.]+|angstrom|bohr)\)\n((?:\s*-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s*\n){3})', txt)
    if not blocks:
        return None
    unit, body = blocks[-1]
    M = np.array([[float(v) for v in l.split()] for l in body.strip().splitlines()])
    if unit.startswith('alat'):
        M *= float(unit.split('=')[1]) * BOHR_A
    elif unit == 'bohr':
        M *= BOHR_A
    return M


def stage_vc(a, cr):
    import copy
    XS = geometry(cr, 'relax_S')
    tg, info = path_targets(cr, XS)
    kind, target = tg['P4']
    # (a) X-ray cell at 80/640
    p0 = run_pw(a, cr, 'x80_P0', XS, calc='relax', ecut=ECUT_VC, forc=RELAX_FORC, etot=RELAX_ETOT)
    X4 = geometry(cr, 'path_P4')
    cons = [(c['A'], site_proton(cr, X4, c), target) for c in cr.copies]
    p4 = run_pw(a, cr, 'x80_P4', X4, calc='relax', ecut=ECUT_VC, dyn='damp', constraints=cons, forc=PATH_FORC,
                etot=PATH_ETOT)
    # (b) variable cell
    pv = run_pw(a, cr, 'vc_S', XS, calc='vc-relax', ecut=ECUT_VC, forc=RELAX_FORC, etot=RELAX_ETOT)
    lat = final_cell(cr.out / 'runs' / 'vc_S' / 'vc_S.out')
    cv = copy.copy(cr)
    cv.lat = lat
    XV = np.array(pv['positions_crystal']) @ lat
    # (c) P4 in the relaxed cell
    Xs, cons2 = seed_point(cv, XV, kind, target)
    pv4 = run_pw(a, cv, 'vc_P4', Xs, calc='relax', ecut=ECUT_VC, dyn='damp', constraints=cons2, forc=PATH_FORC,
                 etot=PATH_ETOT)
    kj = RY_EV * KJMOL_PER_EV / cr.nfu
    out = {'ecut': ECUT_VC, 'xray_cell_A': cr.lat.tolist(), 'vc_cell_A': lat.tolist(),
           'volume_change_pct': round(100 * (abs(np.linalg.det(lat)) / abs(np.linalg.det(cr.lat)) - 1), 2),
           'dE_xray_kJmol_fu': round((p4['energy_Ry'] - p0['energy_Ry']) * kj, 3),
           'dE_vc_kJmol_fu': round((pv4['energy_Ry'] - pv['energy_Ry']) * kj, 3)}
    out['shift_kJmol_fu'] = round(out['dE_vc_kJmol_fu'] - out['dE_xray_kJmol_fu'], 3)
    atomic_json(cr.out / 'vc.json', out)
    note(cr.out, f"vc: volume {out['volume_change_pct']:+.1f} %, dE(P4-P0) X-ray cell {out['dE_xray_kJmol_fu']} vs "
         f"relaxed cell {out['dE_vc_kJmol_fu']} kJ/mol per f.u. (shift {out['shift_kJmol_fu']})")


class Box:
    """a pseudo-crystal holding one isolated pair (for run_pw/pw_input): cubic cell, no symmetry, Gamma point."""

    def __init__(self, key, syms, xyz, edge):
        self.key = key
        self.syms = list(syms)
        self.nat = len(syms)
        self.lat = np.eye(3) * edge
        self.ops = ['x,y,z']
        self.kgrid = (1, 1, 1)
        self.out = RES / key
        c = xyz.mean(axis=0)
        self.X = xyz - c + edge / 2

    def frac(self, X):
        return np.asarray(X) @ np.linalg.inv(self.lat)


def stage_box(a, cr):
    pj = json.loads((cr.out / 'mol' / 'pair.json').read_text())
    xyz0 = np.array([r[1:] for r in pj['points']['P0']['xyz']])
    ext = max(np.ptp(np.array([r[1:] for r in pj['points'][pt]['xyz']]), axis=0).max() for pt in ('P0', 'P4'))
    edge = float(math.ceil(ext + 12.0))
    res = {}
    for pt in ('P0', 'P4'):
        rows = pj['points'][pt]['xyz']
        bx = Box(cr.key, [r[0] for r in rows], np.array([r[1:] for r in rows], float), edge)
        extra = dict(calc='scf', ecut=ECUT_SG15, pseudos=SG15, conv_thr=HYB_CONV_THR, kgrid=(1, 1, 1), box=True)
        pb = run_pw(a, bx, f'box_pbe_{pt}', bx.X, **extra)
        h = run_pw(a, bx, f'box_pbe0_{pt}', bx.X, hybrid=True, **extra)
        res[pt] = {'E_pbe_Ry': pb['energy_Ry'], 'E_pbe0_Ry': h['hybrid']['energy_Ry']}
    kj = RY_EV * KJMOL_PER_EV
    out = {'edge_A': edge, 'points': res,
           'dE_pbe_kJ': round((res['P4']['E_pbe_Ry'] - res['P0']['E_pbe_Ry']) * kj, 3),
           'dE_pbe0_kJ': round((res['P4']['E_pbe0_Ry'] - res['P0']['E_pbe0_Ry']) * kj, 3),
           'pyscf_dE_pbe0_kJ': pj['points']['P4'].get('dPBE0_D3_kJ')}
    out['difference_kJ'] = round(out['dE_pbe0_kJ'] - out['pyscf_dE_pbe0_kJ'], 3) if out['pyscf_dE_pbe0_kJ'] is not None else None
    out['gate_G2b_pass'] = out['difference_kJ'] is not None and abs(out['difference_kJ']) <= 4.0
    atomic_json(cr.out / 'bridge.json', out)
    note(cr.out, f"bridge G2b: QE {out['dE_pbe0_kJ']} vs PySCF {out['pyscf_dE_pbe0_kJ']} kJ/mol -> "
         f"{'PASS' if out['gate_G2b_pass'] else 'FAIL'}")


def stage_kcheck(a, cr):
    k2 = tuple(n + 1 for n in cr.kgrid)
    pts = ('P4r',) if a.point == 'P4r' else (('P0', 'P4') if cr.copies else ('P0',))
    for pt in pts:
        X = geometry(cr, point_tag(pt))
        run_pw(a, cr, f'kcheck_{pt}', X, calc='scf', kgrid=k2)


def stage_summary(a, cr):
    rows = {}
    XS = geometry(cr, 'relax_S')
    tg, info = path_targets(cr, XS) if cr.copies else ({}, None)
    kj = RY_EV * KJMOL_PER_EV / cr.nfu
    extra = sorted(d.name[5:] for d in (cr.out / 'runs').glob('path_X*') if not d.name.endswith('_offpath'))
    for pt in POINTS + tuple(extra):
        tag = point_tag(pt)
        f = cr.out / 'runs' / tag / f'{tag}.out'
        if not f.exists():
            continue
        p = parse_pw(f, cr.nat)
        if not p['relax_converged'] and not SMOKE:
            rows[pt] = {'status': 'not converged'}
            continue
        X = final_geometry(cr, p)
        rec = tg.get(pt) if pt in POINTS else ('D-H', int(pt[1:5]) / 1000.0)
        # 'target': what the run held (its input); 'target_recomputed': path_targets() today (gap map B7, 22 Oct 2026)
        held = held_targets(cr, f.with_suffix('.in'), rec[0]) if rec and cr.copies else None
        r = {'E_pbe_Ry': p['energy_Ry'], 'proton': proton_report(cr, X), 'other_protons_moved': other_protons(cr, X),
             'target': held['target'] if held else rec, 'extra': pt not in POINTS,
             'target_source': 'input' if held else 'recomputed'}
        if rec:
            r['target_recomputed' if pt in POINTS else 'target_from_name'] = rec
        if held and held['extra']:
            r['target_extra'] = held['extra']
        for lev, t in (('pbe_sg15', f'pbe_sg15_{pt}'), ('pbe0_sg15', f'pbe0_sg15_{pt}'), ('pbe50_sg15', f'pbe50_sg15_{pt}')):
            fo = cr.out / 'runs' / t / f'{t}.out'
            if fo.exists():
                q = parse_pw(fo, cr.nat)
                r[f'E_{lev}_Ry'] = parse_hybrid(fo)['energy_Ry'] if lev in ('pbe0_sg15', 'pbe50_sg15') else q['energy_Ry']
        rows[pt] = r
    if 'P0' in rows and 'E_pbe_Ry' in rows['P0']:
        e0 = rows['P0']
        for pt, r in rows.items():
            if 'E_pbe_Ry' not in r:
                continue
            r['dE_pbe_kJmol_fu'] = round((r['E_pbe_Ry'] - e0['E_pbe_Ry']) * kj, 3)
            if all(r.get(k) is not None and e0.get(k) is not None for k in ('E_pbe_sg15_Ry', 'E_pbe0_sg15_Ry')):
                corr = (r['E_pbe0_sg15_Ry'] - r['E_pbe_sg15_Ry']) - (e0['E_pbe0_sg15_Ry'] - e0['E_pbe_sg15_Ry'])
                r['hybrid_corr_kJmol_fu'] = round(corr * kj, 3)
                r['dE_pbe0_kJmol_fu'] = round(r['dE_pbe_kJmol_fu'] + corr * kj, 3)
            if all(r.get(k) is not None and e0.get(k) is not None for k in ('E_pbe_sg15_Ry', 'E_pbe50_sg15_Ry')):
                corr = (r['E_pbe50_sg15_Ry'] - r['E_pbe_sg15_Ry']) - (e0['E_pbe50_sg15_Ry'] - e0['E_pbe_sg15_Ry'])
                r['dE_pbe50_kJmol_fu'] = round(r['dE_pbe_kJmol_fu'] + corr * kj, 3)
    f = cr.out / 'runs' / 'relax_P4r' / 'relax_P4r.out'
    p4r = None
    if f.exists():
        p = parse_pw(f, cr.nat)
        if p['relax_converged']:
            XR = final_geometry(cr, p)
            p4r = {'E_pbe_Ry': p['energy_Ry'], 'dE_pbe_kJmol_fu': round((p['energy_Ry'] - rows['P0']['E_pbe_Ry']) * kj, 3)
                   if 'P0' in rows and 'E_pbe_Ry' in rows['P0'] else None, 'proton': proton_report(cr, XR),
                   'other_protons_moved': other_protons(cr, XR)}
            e = {}
            for lev, t in (('pbe_sg15', 'pbe_sg15_P4r'), ('pbe0_sg15', 'pbe0_sg15_P4r'), ('pbe50_sg15', 'pbe50_sg15_P4r')):
                fo = cr.out / 'runs' / t / f'{t}.out'
                if fo.exists():
                    e[lev] = parse_hybrid(fo)['energy_Ry'] if lev != 'pbe_sg15' else parse_pw(fo, cr.nat)['energy_Ry']
            e0 = rows.get('P0', {})
            if None not in (e.get('pbe_sg15'), e.get('pbe0_sg15'), e0.get('E_pbe_sg15_Ry'), e0.get('E_pbe0_sg15_Ry'),
                            p4r['dE_pbe_kJmol_fu']):
                corr = (e['pbe0_sg15'] - e['pbe_sg15']) - (e0['E_pbe0_sg15_Ry'] - e0['E_pbe_sg15_Ry'])
                p4r['dE_pbe0_kJmol_fu'] = round(p4r['dE_pbe_kJmol_fu'] + corr * kj, 3)
                if e.get('pbe50_sg15') is not None and e0.get('E_pbe50_sg15_Ry') is not None:
                    corr = (e['pbe50_sg15'] - e['pbe_sg15']) - (e0['E_pbe50_sg15_Ry'] - e0['E_pbe_sg15_Ry'])
                    p4r['dE_pbe50_kJmol_fu'] = round(p4r['dE_pbe_kJmol_fu'] + corr * kj, 3)
    kc = {}
    for pt in ('P0', 'P4', 'P4r'):
        f = cr.out / 'runs' / f'kcheck_{pt}' / f'kcheck_{pt}.out'
        if f.exists():
            kc[pt] = parse_pw(f, cr.nat)['energy_Ry']
    kcheck = {'k_dense': [n + 1 for n in cr.kgrid], 'E_dense_Ry': kc} if kc else None
    if 'P0' in kc and 'P4' in kc and 'P4' in rows and 'E_pbe_Ry' in rows['P4']:
        # added to the dict, E_dense_Ry kept (analyse.py polymorph_ranking reads its P0; 22 Oct 2026)
        kcheck.update({'dE_P4_dense_kJmol_fu': round((kc['P4'] - kc['P0']) * kj, 3),
                       'dE_P4_kJmol_fu': rows['P4'].get('dE_pbe_kJmol_fu')})
        if 'P4r' in kc and p4r:
            kcheck['dE_P4r_dense_kJmol_fu'] = round((kc['P4r'] - kc['P0']) * kj, 3)
            kcheck['dE_P4r_kJmol_fu'] = p4r.get('dE_pbe_kJmol_fu')
    atomic_json(cr.out / 'periodic.json', {'key': cr.key, 'generated_utc': utc(), 'nat': cr.nat, 'n_fu': cr.nfu,
                                           'kgrid': cr.kgrid, 'ecut_sssp': ECUT, 'ecut_sg15': ECUT_SG15, 'path': info,
                                           'points': rows, 'P4r': p4r, 'kcheck': kcheck,
                                           'units': 'kJ/mol per formula unit (all copies move: per cell / n_fu)'})
    for pt, r in rows.items():
        print(pt, r.get('dE_pbe_kJmol_fu'), r.get('dE_pbe0_kJmol_fu'), r.get('proton', [{}])[0] if r.get('proton') else '')


def stage_inputs(a, cr):
    """Pi-side G1 helper: the relax input written and read back, distances compared to 1e-4 A."""
    X, how = start_geometry(cr)
    txt = pw_input(cr, X, calc='relax', prefix='g1', outdir='/tmp', pseudo_dir='/tmp', pseudos=SSSP, ecut=ECUT,
                   kgrid=cr.kgrid, forc=RELAX_FORC, etot=RELAX_ETOT)
    lat2, F2 = read_input_back(txt, cr.nat)
    X2 = F2 @ lat2
    X0 = cr.cart(cr.frac0)
    worst = 0.0
    for i in range(cr.nat):
        for j in range(i + 1, cr.nat):
            if cr.syms[i] == 'H' or cr.syms[j] == 'H':
                continue
            worst = max(worst, abs(cr.d(X0, i, j) - cr.d(X2, i, j)))
    nirr = irreducible_k(cr, cr.kgrid)
    print(f'{cr.key:11s} nat {cr.nat:3d} k {cr.kgrid} ({nirr} irr) heavy-atom distance round trip max |dd| = {worst:.2e} A '
          f'({"ok" if worst < 1e-4 else "FAIL"})')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['relax', 'point', 'hybrid', 'esp', 'kcheck', 'p4r', 'box', 'vc', 'summary', 'inputs',
                                      'xpoint', 'hess', 'vcfree', 'vert', 'hscan', 'runner'])
    ap.add_argument('--key', required=True)
    ap.add_argument('--point', default='P0', help='P0..P4, P4r, or an extra point X<dh><r|s>')
    ap.add_argument('--name', default='', help='xpoint: name of the extra point (X<dh in mA><r|s>)')
    ap.add_argument('--dh', type=float, default=None, help='xpoint: d(D-H) held (A)')
    ap.add_argument('--seed', default='P0', choices=['P0', 'P4r'], help='xpoint: geometry the H is moved in')
    ap.add_argument('--exx', type=float, default=None, help='hybrid: exact-exchange fraction other than PBE0 (e.g. 0.5)')
    ap.add_argument('--nq', default='', help='hybrid: exact-exchange q mesh, e.g. 1,1,2 (default 1,1,1)')
    ap.add_argument('--np', type=int, default=int(os.environ.get('QE_NP', 12)))
    ap.add_argument('--keep', action='store_true', help='hybrid: keep the PBE0 density of P0/P4/P4r for the esp stage')
    ap.add_argument('--pin-ha', action='store_true', help='point: hold d(H...A) too (the off-path rule)')
    ap.add_argument('--pbe0', action='store_true', help='runner: add the PBE0 correction (SG15 runs of both configurations)')
    a = ap.parse_args()
    cr = Crystal(a.key)
    cr.out.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r'P[0-4]|P4r|X\d{4}[rs]', a.point):
        raise SystemExit(f'--point {a.point}: not a path point')
    {'relax': stage_relax, 'point': stage_point, 'hybrid': stage_hybrid, 'esp': stage_esp, 'kcheck': stage_kcheck,
     'p4r': stage_p4r, 'box': stage_box, 'vc': stage_vc, 'summary': stage_summary, 'inputs': stage_inputs,
     'xpoint': stage_xpoint, 'hess': stage_hess, 'vcfree': stage_vcfree, 'vert': stage_vert,
     'hscan': stage_hscan, 'runner': stage_runner}[a.stage](a, cr)


if __name__ == '__main__':
    main()
