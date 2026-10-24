#!/usr/bin/env python3
"""Numbers module of the CTC series paper: every number its text, tables and SI may print -> data/facts.json.

    tools/paper-env/bin/python paper/ctc/numbers/facts.py            (from the repository root)

Reads only: optc/results/summary.json (md5 pinned below; refuses any other file), optc/crystals.json,
optc/structures/<key>/cell.json, optc/PLAN.md (the G2b threshold) and optc/CTC_REWRITE_PLAN.md (the plan commit),
plus, for D...A along the periodic profiles only, the merged raw results ($OPTC_MERGED, default
/root/cqc-sfa-archive/optc-merged/results/<key>/periodic.json).  Writes only data/facts.json.

Fact record: key -> {value, unit, precision, sign, text, source, meaning[, note, of, case]}; text = value at its
precision, rounded half to even, true minus.  Sign conventions (_meta.signs):
  dE_PT            E(ion pair) - E(neutral pair), kJ/mol per transferring pair.  summary.json stores most energies as
                   other state minus observed state (om); dE_PT = s * om, s = +1 cocrystal, -1 salt, asserted against
                   crystals[].fixed_ion_minus_neutral.  Same for shifts of a dE_PT quantity.
  margin           other state minus observed (> 0: observed lower), only for a classification margin and its checks.
  toward_observed  other minus observed for the two closure-cycle ranges the plan quotes in that sense.
  profile          E(point) - E(P0), P0 = the observed-state end of the periodic path (SI profile tables).
"""
import hashlib
import json
import os
import re
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OPTC = REPO / 'optc'
SUMMARY, MD5 = OPTC / 'results/summary.json', '411d4c2ad29499c8961de1bae2fc5b02'
RAW = Path(os.environ.get('OPTC_MERGED', '/root/cqc-sfa-archive/optc-merged/results'))
KJ, A = 'kJ mol-1', 'A'
LEV = ['PBE', 'PBE0', 'PBE50', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ']
CORR = LEV[3:]
MOL = {'PBE0+MP2': 'MP2', 'PBE0+SCS_MP2': 'SCS_MP2', 'PBE0+wB97X_D3BJ': 'wB97X_D3BJ'}
CM = ['repeat', 'sc', 'gas']
SIGNS = {
    'dE_PT': 'E(ion pair) - E(neutral pair), kJ/mol per transferring pair (summary.json crystals[].fixed_ion_minus_neutral)',
    'margin': 'other state minus observed state (> 0: observed state lower); classification margins and their checks only',
    'toward_observed': 'other state minus observed state for a non-margin quantity (> 0: favours the observed state)',
    'profile': 'E(point) - E(P0) along the periodic path, P0 = observed-state end (not a dE_PT)',
    'lowering': '-dE_PT term: by how much the environment lowers the ion pair relative to the neutral pair (> 0)',
    'unsigned': 'magnitude or a quantity without orientation',
    'as_stored': 'as stored in summary.json, orientation given in meaning'}
SIGNED = {'dE_PT', 'margin', 'toward_observed', 'profile', 'as_stored'}

raw = SUMMARY.read_bytes()
assert hashlib.md5(raw).hexdigest() == MD5, 'summary.json is not the pinned file'
SJ = json.loads(raw)
CJ = json.loads((OPTC / 'crystals.json').read_text())['crystals']
C = {c['key']: c for c in SJ['crystals']}
KEYS = [k for k in C if k != 'brcoc']                         # the 13 with a transferring bond, summary order (dpKa)
CELL = {k: json.loads((OPTC / f'structures/{k}/cell.json').read_text()) for k in C}
S = {k: (1 if C[k]['observed'] == 'cocrystal' else -1) for k in KEYS}
SALT = [k for k in KEYS if C[k]['observed'] == 'salt']
COC = [k for k in KEYS if C[k]['observed'] == 'cocrystal']
CORE = [k for k in KEYS if CJ[k]['set'] == 'core']
F, MISSING, CHECKS = {}, [], []


def fmt(x, d, signed=False):
    """x at d decimals, half to even on the shortest repr, true minus; '+' when signed."""
    if x is None:
        return '–'
    if isinstance(x, (bool, str)) or d is None:
        return str(x)
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN)
    s = f'{q:+.{d}f}' if signed else f'{q:.{d}f}'
    if q == 0:
        s = s.lstrip('+-')
    return s.replace('-', '−')


def add(key, value, unit=None, prec=None, sign=None, src='', meaning='', **kw):
    assert key not in F, key
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    rec = dict(value=value, unit=unit, precision=prec, sign=sign,
               text=fmt(value, prec, sign in SIGNED) if isinstance(value, (int, float)) and not isinstance(value, bool)
               else value, source=src, meaning=meaning)
    rec.update({k: v for k, v in kw.items() if v is not None})
    F[key] = rec
    return value


def path_str(path):
    return ''.join(f'[{p}]' if isinstance(p, int) else f'.{p}' if re.fullmatch(r'[A-Za-z0-9_+\-]+', p) else f'["{p}"]' for p in path)


def src(k, *path, flip=False):
    return f'summary.json crystals[{k}]' + path_str(path) + (' × (−1) (salt: om → dE_PT)' if flip else '')


def get(x, *path):
    for p in path:
        if x is None:
            return None
        x = x.get(p) if isinstance(x, dict) else x[p]
    return x


def E(key, k, path, meaning, prec=1, conv='dE_PT', unit=KJ):
    """An energy of crystal k from summary.json; om -> dE_PT with s[k] unless conv says otherwise."""
    v = get(C[k], *path)
    if v is None:
        return None
    flip = conv == 'dE_PT' and S[k] < 0
    return add(key, S[k] * v if conv == 'dE_PT' else v, unit, prec, conv, src(k, *path, flip=flip), meaning, case=k)


def slug(s):
    return re.sub(r'_+', '_', re.sub(r'[^A-Za-z0-9+]+', '_', s)).strip('_').lower()


def rng(prefix, vals, unit, prec, sign, meaning, srcs):
    """min / max (and where) of {crystal: value}."""
    kmin, kmax = min(vals, key=vals.get), max(vals, key=vals.get)
    add(f'{prefix}.min', vals[kmin], unit, prec, sign, srcs, meaning + ' (minimum)', at=kmin)
    add(f'{prefix}.max', vals[kmax], unit, prec, sign, srcs, meaning + ' (maximum)', at=kmax)
    add(f'{prefix}.n', len(vals), None, 0, None, srcs, meaning + ' (number of crystals)', members=list(vals))


def count(prefix, members, ok, meaning, srcs):
    n = sum(bool(ok[m]) for m in members)
    add(prefix, n, None, 0, None, srcs, meaning, of=len(members), text_of=f'{n}/{len(members)}',
        yes=[m for m in members if ok[m]], no=[m for m in members if not ok[m]])


def cp95(x, n):
    lo = 0.0 if x == 0 else stats.beta.ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(0.975, x + 1, n - x)
    return [round(float(lo), 3), round(float(hi), 3)]


def check(name, ok, detail=''):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    assert ok, (name, detail)


# ===================================================================================================== sign factor
for k in KEYS:
    fx = C[k]['fixed_ion_minus_neutral']
    pairs = {'R1f': ('R1f', 'PBE0'), 'R4a_single': ('ladder', 'repeat', 'R4a_single'),
             'R4a_collective': ('ladder', 'repeat', 'R4a_collective'), 'R3': ('ladder', 'repeat', 'R3'),
             'R4_single': ('ladder', 'repeat', 'R4_single'), 'R5_PBE0': ('R5', 'PBE0'),
             'R5_PBE0+MP2': ('R5', 'PBE0+MP2'), 'dE_env_PBE0': ('dE_env', 'PBE0')}
    for fk, p in pairs.items():
        check(f'sign {k} {fk}', abs(fx[fk] - S[k] * get(C[k], *p)) < 1e-9, f'{fx[fk]} vs s*{get(C[k], *p)}')

# ===================================================================================================== per crystal
for k in list(C):
    c, j, cell, p = C[k], CJ[k], CELL[k], f'x.{k}'
    cj = f'crystals.json crystals.{k}'
    for f_ in ['name', 'set', 'family', 'role', 'observed', 'proton_evidence', 'ccdc', 'cif']:
        add(f'{p}.{f_}', j.get(f_), None, None, None, f'{cj}.{f_}', f_.replace('_', ' '), case=k)
    add(f'{p}.cod', j.get('cod') or j['cif'].removesuffix('.cif'), None, None, None,
        f'{cj}.cod' if j.get('cod') else f'{cj}.cif (COD number = CIF file name)', 'COD number', case=k)
    add(f'{p}.dpka', j['dpka'], None, 2, None, f'{cj}.dpka', 'dpKa = pKa(BH+) - pKa(HA), aqueous, 25 C (plan value)', case=k)
    add(f'{p}.dpka_lo', j['dpka_range'][0], None, 2, None, f'{cj}.dpka_range[0]', 'dpKa range, low', case=k)
    add(f'{p}.dpka_hi', j['dpka_range'][1], None, 2, None, f'{cj}.dpka_range[1]', 'dpKa range, high', case=k)
    flags = [f_ for f_, on in [('compilation', j.get('dpka_quality')), ('off-axis', j.get('dpka_off_axis')),
                               ('site value', j.get('dpka_site') is not None), ('alternative', j.get('dpka_alt') is not None)] if on]
    add(f'{p}.dpka_flags', flags, None, None, None, f'{cj}.dpka_quality / dpka_off_axis / dpka_site / dpka_alt',
        'quality flags of dpKa', case=k)
    for f_, prec in [('dpka_quality', None), ('dpka_off_axis_reason', None), ('dpka_site', 2), ('dpka_alt', 2)]:
        if j.get(f_) is not None:
            add(f'{p}.{f_}', j[f_], None, prec, None, f'{cj}.{f_}', f_.replace('_', ' '), case=k)
    if j.get('dpka_alt_range'):
        add(f'{p}.dpka_alt_lo', j['dpka_alt_range'][0], None, 2, None, f'{cj}.dpka_alt_range[0]', 'alternative dpKa, low', case=k)
        add(f'{p}.dpka_alt_hi', j['dpka_alt_range'][1], None, 2, None, f'{cj}.dpka_alt_range[1]', 'alternative dpKa, high', case=k)
    cs = f'optc/structures/{k}/cell.json'
    add(f'{p}.nat', cell['nat'], None, 0, None, f'{cs}:nat', 'atoms in the periodic cell', case=k)
    add(f'{p}.Z', int(cell['Z_cif']), None, 0, None, f'{cs}:Z_cif', 'Z of the CIF', case=k)
    add(f'{p}.n_fu', cell['n_formula_units'], None, 0, None, f'{cs}:n_formula_units', 'formula units (transferring pairs) in the cell', case=k)
    add(f'{p}.space_group', cell['meta']['space_group'], None, None, None, f'{cs}:meta.space_group', 'space group', case=k)
    add(f'{p}.T_K', cell['meta']['T'], None, None, None, f'{cs}:meta.T', 'data-collection temperature (CIF string)', case=k)
    add(f'{p}.bond', cell['transfer']['labels'] if cell.get('transfer') else None, None, None, None,
        f'{cs}:transfer.labels', 'transferring hydrogen bond D-H...A (CIF labels)', case=k)
    if k == 'brcoc':
        continue
    check(f'DA {k}', abs(cell['transfer']['DA'] - c['DA_xray_A']) < 1e-4)
    E(f'{p}.DA.xray', k, ['DA_xray_A'], 'D...A of the transferring bond, X-ray', 3, None, A)
    E(f'{p}.DA.P0', k, ['DA_P0_A'], 'D...A at P0 (periodic, observed state)', 3, None, A)
    # ---- isolated pair, field and cluster rungs (dE_PT)
    for lev in ['PBE0', 'PBE', 'MP2', 'SCS_MP2', 'wB97X_D3BJ']:
        E(f'{p}.R1f.{lev}', k, ['R1f', lev], f'R1f: isolated pair at the crystal geometry, {lev}')
    for cm in CM:
        L = c['ladder'][cm]
        E(f'{p}.R4a_single.{cm}', k, ['ladder', cm, 'R4a_single'], f'R4a single site: pair in the lattice field ({cm} charges), observed environment')
        E(f'{p}.R4a_coll_Eint2.{cm}', k, ['ladder', cm, 'R4a_collective'], f'R4a collective, estimator E_emb - E_int/2 (plan), {cm} charges')
        E(f'{p}.R4a_coll_LR.{cm}', k, ['ladder', cm, 'R4a_collective_lr'], f'R4a collective, LR mean of forward and reverse gaps (added), {cm} charges')
    E(f'{p}.R3', k, ['ladder', 'repeat', 'R3'], 'R3: pair + first H-bonded shell (transfer-site rule), vacuum')
    E(f'{p}.R4', k, ['ladder', 'repeat', 'R4_single'], 'R4: R3 in the lattice field (REPEAT), single site')
    add(f'{p}.site_shell_empty', bool(c['ladder']['repeat'].get('site_shell_empty')), None, None, None,
        src(k, 'ladder', 'repeat', 'site_shell_empty'), 'transfer-site shell empty: R3 = R1f and R4 = R4a by definition', case=k)
    E(f'{p}.n_site_shell', k, ['ladder', 'repeat', 'n_site_shell'], 'molecules in the transfer-site shell', 0, None, None)
    # ---- periodic rung
    for lev in LEV:
        E(f'{p}.R5.P4.{lev}', k, ['R5', lev], f'R5 at P4: periodic crystal, fixed X-ray cell, {lev}')
        for cl, tag in [('class', 'full'), ('class_preregistered_profile', 'plan')]:
            v = get(c, cl, lev)
            if v:
                add(f'{p}.class.{tag}.{lev}', v[0], None, None, None, src(k, cl, lev, 0),
                    f'class (minimum of the PCHIP profile), {"full profile incl. P4r and extra points" if tag == "full" else "plan profile P0-P4"}, {lev}', case=k)
                add(f'{p}.class.{tag}.{lev}.delta_min', v[1], A, 2, 'as_stored', src(k, cl, lev, 1),
                    'delta of the profile minimum (< 0: observed side)', case=k)
        m = get(c, 'margin_other_side', lev)
        if m and m.get('margin') is not None:
            E(f'{p}.margin.{lev}', k, ['margin_other_side', lev, 'margin'],
              f'classification margin: lowest sampled other-side point minus P0 ({m["point"]}), {lev}', conv='margin')
            add(f'{p}.margin.{lev}.point', m['point'], None, None, None, src(k, 'margin_other_side', lev, 'point'), 'point of the margin', case=k)
        if m and m.get('crossing_energy_delta0') is not None:
            E(f'{p}.crossing.{lev}', k, ['margin_other_side', lev, 'crossing_energy_delta0'],
              f'crossing energy E(delta = 0) - E(P0), not a state, {lev}', conv='margin')
        if m and 'no_other_side_minimum' in m:
            add(f'{p}.no_other_side_minimum.{lev}', bool(m['no_other_side_minimum']), None, None, None,
                src(k, 'margin_other_side', lev, 'no_other_side_minimum'), 'no other-side minimum on the profile', case=k)
    # ---- environment term and compensation
    for lev in ['PBE0', 'PBE']:
        env, r1 = E(f'{p}.dE_env.{lev}', k, ['dE_env', lev], f'dE_env = R5 - R1f at P4, {lev} (environment term)'), S[k] * c['R1f'][lev]
        add(f'{p}.dE_env_lowering.{lev}', -env, KJ, 1, 'lowering', src(k, 'dE_env', lev) + ' (toward the ion pair)',
            f'lowering of the ion pair relative to the neutral pair by the observed environment, {lev}', case=k)
        add(f'{p}.compensation.{lev}', -env / r1, None, 2, 'unsigned', f'-[{src(k, "dE_env", lev)}] / [{src(k, "R1f", lev)}], both dE_PT',
            f'fraction of the isolated pair cost R1f cancelled by the environment, -dE_env / R1f, {lev} (> 1: over-compensated)', case=k)
        check(f'R5 = R1f + dE_env {k} {lev}', abs(r1 + env - S[k] * c['R5'][lev]) < 2e-3)
    # ---- budgets (estimate: shifts at different settings combined additively)
    for q, b in c['budget'].items():
        if q == 'label':
            continue
        conv = 'margin' if q in ('margin', 'P4r') else 'dE_PT'
        sg = S[k] if conv == 'dE_PT' else 1
        bp, fl = f'{p}.budget.{q}', conv == 'dE_PT' and S[k] < 0
        what = {'margin': 'margin at P4r', 'P4r': 'P4r - P0 (no other-side minimum: not a margin)', 'R5_P4': 'R5 PBE0+MP2 at P4',
                'dE_env': 'dE_env PBE0 at P4', 'R4a_single': 'R4a single site (REPEAT)', 'R4a_collective': 'R4a collective E_int/2 (REPEAT)',
                'R4_single': 'R4 single site (REPEAT)'}[q]
        add(f'{bp}.central', sg * b['central'], KJ, 1, conv, src(k, 'budget', q, 'central', flip=fl), f'budget central value, {what}', case=k)
        lo, hi = sorted(sg * x for x in b['envelope'])
        add(f'{bp}.lo', lo, KJ, 1, conv, src(k, 'budget', q, 'envelope', flip=fl), f'budget envelope low, {what}', case=k)
        add(f'{bp}.hi', hi, KJ, 1, conv, src(k, 'budget', q, 'envelope', flip=fl), f'budget envelope high, {what}', case=k)
        add(f'{bp}.rss', b['rss'], KJ, 1, 'unsigned', src(k, 'budget', q, 'rss'), f'budget root sum of squares of the shifts, {what}', case=k)
        for g, comp in (b.get('components') or {}).items():
            for lab, v in (comp or {}).items():
                key = f'{bp}.{g}' if len(comp) == 1 else f'{bp}.{g}.{slug(lab)}'
                add(key, sg * v, KJ, 1, conv, src(k, 'budget', q, 'components', g, lab, flip=fl), f'shift of the {what}: {g}, {lab}', case=k)
        for lab, v in (b.get('combined') or {}).items():
            add(f'{bp}.combined.{slug(lab)}', sg * v, KJ, 1, conv, src(k, 'budget', q, 'combined', lab, flip=fl),
                f'{what} with the combined checks: {lab}', case=k)
        for g, why in (b.get('none_reason') or {}).items():
            add(f'{bp}.{g}.none_reason', why, None, None, None, src(k, 'budget', q, 'none_reason', g), f'why the {g} check is absent', case=k)
    r = c['resolution']
    add(f'{p}.resolution.verdict', r['verdict'], None, None, None, src(k, 'resolution', 'verdict'), 'resolution of the corrected class by the budget', case=k)
    add(f'{p}.resolution.quantity', r['quantity'], None, None, None, src(k, 'resolution', 'quantity'), 'quantity the resolution rule uses', case=k)
    add(f'{p}.resolved', r['verdict'].startswith('resolved'), None, None, None, src(k, 'resolution', 'verdict'), 'resolved by the budget', case=k)
    # ---- reverse gaps (whose environment): fwd = observed environment, rev = the other state's own environment
    for g, rv in c['reverse'].items():
        rp = f'{p}.reverse.{g}'
        for f_, lab in [('embedded_single_site_kJ', 'forward gap, observed-state environment'),
                        ('reverse_kJ', "reverse gap, the other state's environment"), ('lr_mean_kJ', 'LR mean of the two gaps'),
                        ('vacuum_kJ', 'cluster in vacuum, forward'), ('reverse_vacuum_kJ', 'cluster in vacuum, reverse'),
                        ('lr_mean_vacuum_kJ', 'cluster in vacuum, LR mean')]:
            if f_ in rv:
                E(f'{rp}.{f_.removesuffix("_kJ")}', k, ['reverse', g, f_], f'{g}: {lab}')
        for f_ in ['lambda_kJ', 'lambda_vacuum_kJ']:
            if f_ in rv:
                E(f'{rp}.{f_.removesuffix("_kJ")}', k, ['reverse', g, f_], f'{g}: lambda = (forward - reverse)/2 in the observed-first orientation (> 0)', conv='unsigned')
        add(f'{rp}.other_wins', rv['reverse_kJ'] < 0, None, None, None, src(k, 'reverse', g, 'reverse_kJ') + ' < 0',
            "the other state wins in its own environment", case=k)
        add(f'{rp}.lr_observed', rv['lr_mean_kJ'] > 0, None, None, None, src(k, 'reverse', g, 'lr_mean_kJ') + ' > 0',
            'the LR mean puts the proton on the observed side', case=k)
    # ---- periodic closure  R1f -> R4 -> T1 -> N_vert -> R5
    cl = c['closure']
    for lev in ['pbe0', 'pbe']:
        for t, v in c['closure'][f'cycle_{lev}'].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                E(f'{p}.cycle.{lev}.{t}', k, ['closure', f'cycle_{lev}', t], f'closure cycle term {t} ({lev.upper()})')
        check(f'cycle closes {k} {lev}', cl[f'cycle_{lev}']['sum_equals_R5'])
    for t, lab in [('T1_pbe0_kJ', 'T1 periodic single site (one pair of Z moved), PBE0'), ('T1_pbe_kJ', 'T1, PBE'),
                   ('T1v_pbe_kJ', 'T1 vertical (H only), PBE'), ('T2_pbe_kJ', 'T2 doubled cell single site, PBE'),
                   ('T2_pbe0_est_kJ', 'T2 PBE0 estimate = T2(PBE) + [T1(PBE0) - T1(PBE)]'),
                   ('Nv_pbe0_kJ_per_pair', 'N_vert every H moved, per pair, PBE0'), ('Nv_pbe_kJ_per_pair', 'N_vert, PBE'),
                   ('cooperativity_preregistered_pbe0', 'cooperativity N_vert - T1 (item C), PBE0'),
                   ('cooperativity_preregistered_pbe', 'cooperativity N_vert - T1 (item C), PBE'),
                   ('cooperativity_vertical_f1overZ_to_1_pbe', 'cooperativity N_vert - T1v (both vertical), PBE'),
                   ('pair_relaxation_single_site_pbe', 'T1 - T1v, relaxation of the one pair, PBE'),
                   ('reorganization_pbe0', 'lattice reorganisation R5 - N_vert, PBE0'), ('reorganization_pbe', 'R5 - N_vert, PBE'),
                   ('cluster_minus_periodic_single_site', 'R4 (cluster) - T1'), ('cluster_minus_periodic_single_site_T2', 'R4 - T2 PBE0 est.'),
                   ('field_minus_periodic_single_site', 'R4a (field) - T1'), ('field_minus_periodic_single_site_T2', 'R4a - T2 PBE0 est.')]:
        E(f'{p}.closure.{t.removesuffix("_kJ")}', k, ['closure', t], lab)
    E(f'{p}.closure.cluster_overshoot', k, ['closure', 'cluster_minus_periodic_single_site'],
      'R4 - T1 toward the observed state: how much the embedded cluster overstates the periodic single site', conv='toward_observed')
    add(f'{p}.closure.relaxation_giveback', -cl['reorganization_pbe0'], KJ, 1, 'toward_observed',
        src(k, 'closure', 'reorganization_pbe0') + ' × (−1)', 'lattice relaxation R5 - N_vert, given back from the observed state (> 0), PBE0', case=k)
    add(f'{p}.closure.image_T2_minus_T1_pbe', S[k] * (cl['T2_pbe_kJ'] - cl['T1_pbe_kJ']), KJ, 1, 'dE_PT',
        f'{src(k, "closure", "T2_pbe_kJ")} - {src(k, "closure", "T1_pbe_kJ")}' + (' × (−1)' if S[k] < 0 else ''),
        'image effect of the single site, T2 - T1 (PBE)', case=k)
    add(f'{p}.closure.sign_undetermined', cl['cluster_minus_periodic_single_site_sign'].startswith('sign undetermined'), None, None, None,
        src(k, 'closure', 'cluster_minus_periodic_single_site_sign'), 'sign of R4 - T1 differs with T1 and the T2 estimate', case=k)
    # ---- checks: k, q, basis, delta(T), relaxed cells, bridge
    E(f'{p}.shift.k.P4', k, ['kcheck_shift'], 'k-mesh shift at P4 (PBE, dense - production k)')
    if c.get('kcheck_shift_P4r') is not None:
        E(f'{p}.shift.k.P4r', k, ['kcheck_shift_P4r'], 'k-mesh shift at P4r (PBE)', conv='margin')
    ck = c['checks']
    if ck.get('qmesh'):
        E(f'{p}.shift.q.P4r', k, ['checks', 'qmesh', 'shift_kJ_per_fu'], 'EXX q-mesh shift at P4r (q = k mesh - q = 1x1x1), PBE0', conv='margin')
    if ck.get('pair_cc'):
        for t in ['P4', 'P4r']:
            if f'delta_T_{t}_kJ' in ck['pair_cc']:
                E(f'{p}.shift.dT.{t}', k, ['checks', 'pair_cc', f'delta_T_{t}_kJ'], f'delta(T) = CCSD(T) - MP2 of the pair (def2-SVP) at {t}')
    if ck.get('pair_def2-qzvp'):
        for t, f_ in [('P4', 'dPBE0_D3_kJ'), ('P4', 'dMP2_kJ'), ('P4r', 'dPBE0_D3_P4r_kJ'), ('P4r', 'dMP2_P4r_kJ')]:
            E(f'{p}.qzvp.{t}.{f_.split("_")[0][1:]}', k, ['checks', 'pair_def2-qzvp', f_], f'pair transfer energy in def2-QZVP at {t}, {f_}')
    if c.get('vc'):
        E(f'{p}.shift.cell_vc.P4', k, ['vc', 'shift_kJmol_fu'], 'variable-cell relax at P4 (PBE, 80 Ry): shift of R5')
        E(f'{p}.vc.volume_change_pct', k, ['vc', 'volume_change_pct'], 'cell volume change of the vc relax', 2, None, '%')
    vf = ck.get('vcfree')
    if vf:
        for f_, v in vf.items():
            if f_.startswith('dE_relaxed_cells') or f_ == 'dE_xray_cell_pbe80':
                E(f'{p}.vcfree.{f_.removeprefix("dE_")}', k, ['checks', 'vcfree', f_], f'P4r - P0 with both cells relaxed: {f_}', conv='margin')
        for t, v in vf['volume_change_pct'].items():
            E(f'{p}.vcfree.volume_change_pct.{t}', k, ['checks', 'vcfree', 'volume_change_pct', t], f'volume change of the relaxed cell {t}', 2, None, '%')
    for f_ in ['runner_up_minus_chosen_periodic']:
        if ck.get(f_):
            E(f'{p}.runner_up.periodic', k, ['checks', f_, 'runner_minus_chosen_pbe_kJ'],
              'runner-up minus chosen bond, periodic vertical (PBE): E(other state via runner-up) - E(via chosen)', conv='as_stored')
    if ck.get('runner_up_minus_chosen_vertical'):
        for lev, v in ck['runner_up_minus_chosen_vertical'].items():
            E(f'{p}.runner_up.pair.{lev}', k, ['checks', 'runner_up_minus_chosen_vertical', lev], f'runner-up minus chosen, pair level {lev}', conv='as_stored')
    if c.get('bridge'):
        b = c['bridge']
        E(f'{p}.bridge.QE_PBE0', k, ['bridge', 'dE_pbe0_kJ'], 'gate G2b: pair transfer energy, plane-wave PBE0 (QE)')
        E(f'{p}.bridge.PySCF_PBE0', k, ['bridge', 'pyscf_dE_pbe0_kJ'], 'gate G2b: pair transfer energy, PySCF PBE0-D3(BJ)/def2-TZVP')
        add(f'{p}.bridge.abs_difference', abs(b['difference_kJ']), KJ, 2, 'unsigned', src(k, 'bridge', 'difference_kJ') + ' (absolute)',
            'gate G2b: |QE - PySCF| of the pair transfer energy', case=k)
        add(f'{p}.bridge.pass', b['gate_G2b_pass'], None, None, None, src(k, 'bridge', 'gate_G2b_pass'), 'gate G2b passed', case=k)
        if ck.get('pair_def2-qzvp'):
            add(f'{p}.bridge.abs_difference_qzvp', abs(b['dE_pbe0_kJ'] - ck['pair_def2-qzvp']['dPBE0_D3_kJ']), KJ, 2, 'unsigned',
                f'|{src(k, "bridge", "dE_pbe0_kJ")} - {src(k, "checks", "pair_def2-qzvp", "dPBE0_D3_kJ")}|', 'gate G2b difference with the pair in def2-QZVP', case=k)
    # ---- nuclear quantum checks
    nu = c.get('nuclear') or {}
    for lev, z in (get(nu, 'zpe_frozen_frame', 'H') or {}).items():
        zp = ['nuclear', 'zpe_frozen_frame', 'H', lev]
        E(f'{p}.zpe.H.{lev}.dZPE', k, zp + ['dZPE_kJ'], f'frozen-frame proton dZPE (P4r - P0), H, {lev}', conv='margin')
        E(f'{p}.zpe.H.{lev}.E0_difference', k, zp + ['E0_difference_kJ'], f'frozen-frame E0 difference ({z["kind"]}), {lev}', conv='margin')
        add(f'{p}.zpe.H.{lev}.kind', z['kind'], None, None, None, src(k, *zp, 'kind'), 'kind of the frozen-frame result', case=k)
        if z.get('margin_P4r_zpe_corrected') is not None:
            E(f'{p}.zpe.H.{lev}.margin', k, zp + ['margin_P4r_zpe_corrected'], f'ZPE-corrected margin at P4r, H, {lev}', conv='margin')
    for iso in ['H']:
        for f_ in ['zpe_HDA_kJmol', 'zpe_H_only_kJmol']:
            if get(nu, 'dZPE_P4r_minus_P0', iso, f_) is not None:
                E(f'{p}.zpe.harmonic.{f_.split("_kJ")[0].removeprefix("zpe_")}', k, ['nuclear', 'dZPE_P4r_minus_P0', iso, f_],
                  f'harmonic partial-Hessian dZPE P4r - P0 (PBE), {f_}', conv='margin')
    if nu.get('p_other_H_printed'):
        add(f'{p}.p_other_H', nu['p_other_H_printed']['value'], None, 4, None, src(k, 'nuclear', 'p_other_H_printed', 'value'),
            f'probability of the proton beyond the dividing point ({nu["p_other_H_printed"]["source"]})', case=k)
    # ---- continuum (secondary)
    add(f'{p}.pcm.eps_x', c['eps_x'], None, None, None, src(k, 'eps_x'), 'epsilon at which the continuum pair changes sign (string)', case=k)
    add(f'{p}.pcm.eps_eq', c['eps_eq'], None, None, None, src(k, 'eps_eq'), 'epsilon at which the continuum reproduces R5 PBE0+MP2 (string)', case=k)
    E(f'{p}.pcm.eps78', k, ['pcm_dE', -1, 1], 'pair in the continuum at epsilon = 78.36')
    for sc, v in (ck.get('pcm_radii') or {}).items():
        E(f'{p}.pcm.radii{sc}.eps78', k, ['checks', 'pcm_radii', sc, 'eps78'], f'pair in the continuum at epsilon 78, radius scale {sc}')
    # ---- field/shell split (plan, verdict) and exploratory decomposition
    lr = c['ladder']['repeat']
    for t in ['field', 'shell', 'nonadditivity', 'field_only', 'cooperativity']:
        E(f'{p}.split.{t}', k, ['ladder', 'repeat', t], f'field/shell split (plan, REPEAT): {t}')
    E(f'{p}.split.field_share', k, ['ladder', 'repeat', 'field_share'], 'field share of the split', 2, None, None)
    add(f'{p}.split.presentable', lr['split_presentable'], None, None, None, src(k, 'ladder', 'repeat', 'split_presentable'),
        'split presentable (|nonadditivity| < min(|field|, |shell|))', case=k)
    for t, v in c['decomposition']['repeat'].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not t.startswith('R4a'):
            E(f'{p}.decomp.{t}', k, ['decomposition', 'repeat', t], f'exploratory decomposition (added 19 Oct), REPEAT: {t}')

# ===================================================================================================== profiles (SI)
PTS = {}
for k in KEYS:
    pj = RAW / k / 'periodic.json'
    rp = json.loads(pj.read_text()) if pj.exists() else None
    if rp is None:
        MISSING.append(f'{k}: D...A along the profile ({pj} not found)')
    rsrc = f'optc-merged/results/{k}/periodic.json'
    names = list(C[k]['profile'])
    if 'P4r' not in names and C[k].get('P4r'):
        names.append('P4r')                                       # salts: P4r relaxed back to the observed side
    PTS[k] = names
    for n in names:
        pp, pt = f'p.{k}.{n}', C[k]['profile'].get(n)
        if pt is None:                                            # salt P4r: summary.json crystals[].P4r (PBE only)
            q = C[k]['P4r']['proton']
            add(f'{pp}.delta', float(np.mean([x['delta'] for x in q])), A, 3, 'as_stored', src(k, 'P4r', 'proton', '*', 'delta'),
                'proton metric delta = d(D-H) - d(H...A) (< 0: observed side)', case=k)
            E(f'{pp}.PBE', k, ['P4r', 'dE_pbe_kJmol_fu'], 'E(P4r) - E(P0), PBE (P4r relaxed from P4)', conv='profile')
            add(f'{pp}.target', None, A, 3, None, '', 'held distance (none: unconstrained relax)', case=k, kind='relaxed')
        else:
            E(f'{pp}.delta', k, ['profile', n, 'delta'], 'proton metric delta = d(D-H) - d(H...A) (< 0: observed side)', 3, 'as_stored', A)
            for lev in LEV:
                E(f'{pp}.{lev}', k, ['profile', n, lev], f'E({n}) - E(P0), {lev}', conv='profile')
            tt = rp['points'][n].get('target') if rp and n in rp['points'] else None
            add(f'{pp}.target', tt[1] if tt else None, A, 3, None, (f'{rsrc} points.{n}.target' if tt else ''),
                f'held distance {tt[0] if tt else "(none)"}', case=k, target_type=tt[0] if tt else None,
                kind='relaxed' if n == 'P4r' else ('extra' if pt.get('extra') else 'path'))
            if pt.get('other_protons_moved'):
                MISSING.append(f'{k} {n}: other protons moved {pt["other_protons_moved"]}')
        if rp:
            q = (rp['P4r'] if n == 'P4r' else rp['points'][n])['proton']
            da = [x['D...A'] for x in q]
            check(f'copies {k} {n}', max(da) - min(da) < 2e-3, str(da))
            check(f'delta raw {k} {n}', abs(np.mean([x['delta'] for x in q]) - F[f'{pp}.delta']['value']) < 2e-4)
            add(f'{pp}.DA', float(np.mean(da)), A, 3, None, f'{rsrc} {"P4r" if n == "P4r" else "points." + n}.proton[*].D...A (mean of copies)',
                'D...A of the transferring bond at this point', case=k)
        else:
            add(f'{pp}.DA', None, A, 3, None, '', 'D...A (raw results not available)', case=k)
    add(f'p.{k}.points', names, None, None, None, src(k, 'profile'), 'points of the periodic profile (salts: + P4r, relaxed back)', case=k)

# ===================================================================================================== table 4 (cocrystal checks at P4r, per level)
for k in COC:
    c, ck, b = C[k], C[k]['checks'], C[k]['budget'].get('margin') or C[k]['budget']['P4r']
    bq = 'margin' if 'margin' in C[k]['budget'] else 'P4r'
    P4r = c['profile']['P4r']
    for lev in CORR:
        t, ml = f't4.{k}.{lev}', MOL[lev]
        cen = P4r[lev]
        vac = P4r[lev] - P4r['PBE0']                                  # vacuum pair correction at P4r (default basis)
        sh = {}
        sh['in_field'] = ck['embcorr_pair_repeat']['NR'][f'corr_{ml}_kJ'] - vac
        vf = ck['vcfree']
        rel_vac, rel_inf = vf[f'dE_relaxed_cells_pbe0_{ml}'], vf[f'dE_relaxed_cells_pbe0_{ml}_infield']
        sh['relaxed_vacuum'], sh['relaxed_in_field'] = rel_vac - cen, rel_inf - cen
        sh['in_field_relaxed'] = rel_inf - rel_vac
        sh['k'] = c['kcheck_shift_P4r']
        sh['q'] = ck['qmesh']['P4r_shifted'][lev]['q'] - ck['qmesh']['P4r_shifted'][lev]['q111']
        tz = ck['monomers_tzvpd']['P4r']['R1f_kJ']
        sh['TZVPD'] = (tz[ml] - tz['PBE0_D3']) - vac
        if lev == 'PBE0+MP2':
            if ck.get('pair_cc'):
                sh['delta_T'] = ck['pair_cc']['delta_T_P4r_kJ']
            if ck.get('pair_def2-qzvp'):
                qz = ck['pair_def2-qzvp']
                sh['QZVP'] = (qz['dMP2_P4r_kJ'] - qz['dPBE0_D3_P4r_kJ']) - vac
        z = get(c, 'nuclear', 'zpe_frozen_frame', 'H', lev)
        if z:
            sh['ZPE_frozen'] = z['dZPE_kJ']
            add(f'{t}.ZPE_frozen.kind', z['kind'], None, None, None, src(k, 'nuclear', 'zpe_frozen_frame', 'H', lev, 'kind'), 'frozen-frame ZPE kind', case=k)
        for f_ in ['zpe_HDA_kJmol', 'zpe_H_only_kJmol']:
            sh[f'ZPE_harmonic_{f_.split("_kJ")[0].removeprefix("zpe_")}'] = c['nuclear']['dZPE_P4r_minus_P0']['H'][f_]
        dT = sh.get('delta_T', 0.0)
        comb = {'relaxed cells, in-field': rel_inf,
                'relaxed cells, vacuum + delta(T) + k + q' if 'delta_T' in sh else 'relaxed cells, vacuum + k + q': rel_vac + dT + sh['k'] + sh['q'],
                'relaxed cells, in-field + delta(T) + k + q' if 'delta_T' in sh else 'relaxed cells, in-field + k + q': rel_inf + dT + sh['k'] + sh['q']}
        zfr = 'ZPE_frozen' if z and z.get('margin_P4r_zpe_corrected') is not None else None    # a bound enters no envelope
        use = {g: v for g, v in sh.items() if g != 'ZPE_frozen' or zfr}
        vals = [cen] + [cen + v for v in use.values()] + list(comb.values())
        if lev == 'PBE0+MP2':                                         # the rule must reproduce summary.json's budget
            bc = b['components']
            ref = {'in_field': bc['in_field_vs_vacuum']['X-ray cell, NR (REPEAT field)'],
                   'relaxed_vacuum': bc['cell']['relaxed cells (vcfree), vacuum pair correction'],
                   'in_field_relaxed': bc['in_field_vs_vacuum']['relaxed cells, (VR - VS) - (vcP4r - vcS)'],
                   'k': bc['k_mesh']['PBE, dense - production k at P4r'], 'q': bc['q_mesh']['PBE0, EXX q = k mesh - q = 1x1x1'],
                   'TZVPD': bc['basis_TZVPD']['[MP2 - PBE0] of the pair, def2-TZVPD']}
            if bc.get('delta_T'):
                ref['delta_T'] = bc['delta_T']['[CCSD(T) - MP2] of the pair, def2-SVP']
            if bc.get('basis_QZVP'):
                ref['QZVP'] = bc['basis_QZVP']['[MP2 - PBE0] of the pair, def2-QZVP']
            if zfr:
                ref['ZPE_frozen'] = bc['ZPE']['frozen frame, H']
            ref['ZPE_harmonic_HDA'] = bc['ZPE']['harmonic partial Hessian, H D A (PBE)']
            ref['ZPE_harmonic_H_only'] = bc['ZPE']['harmonic partial Hessian, H only (PBE)']
            for g, v in ref.items():
                check(f't4 rule {k} {g}', abs(sh[g] - v) < 2e-3, f'{sh[g]} vs {v}')
            for lab, v in b['combined'].items():
                check(f't4 combined {k} {lab}', abs(comb[lab] - v) < 2e-3, f'{comb[lab]} vs {v}')
            check(f't4 central {k}', abs(cen - b['central']) < 1e-9)
            extra = [cen + v for g in ('level_spread', 'anchor_PBE', 'anchor_PBE50') for v in (bc.get(g) or {}).values()]
            env = [min(vals + extra), max(vals + extra)]
            check(f't4 envelope {k}', abs(env[0] - b['envelope'][0]) < 2e-3 and abs(env[1] - b['envelope'][1]) < 2e-3, f'{env} vs {b["envelope"]}')
        conv = 'margin'
        mo = c['margin_other_side'][lev]
        add(f'{t}.xray', cen, KJ, 1, conv, src(k, 'profile', 'P4r', lev),
            f'P4r - P0 at the X-ray cell, {lev} ({"a margin" if mo.get("margin") is not None and mo.get("point") == "P4r" else "not the margin of this level: see x." + k + ".margin"})', case=k)
        rule = {'in_field': f'checks.embcorr_pair_repeat.NR.corr_{ml}_kJ - (profile.P4r.{lev} - profile.P4r.PBE0)',
                'relaxed_vacuum': f'checks.vcfree.dE_relaxed_cells_pbe0_{ml} - profile.P4r.{lev}',
                'relaxed_in_field': f'checks.vcfree.dE_relaxed_cells_pbe0_{ml}_infield - profile.P4r.{lev}',
                'in_field_relaxed': f'checks.vcfree.dE_relaxed_cells_pbe0_{ml}_infield - dE_relaxed_cells_pbe0_{ml}',
                'k': 'kcheck_shift_P4r (PBE)', 'q': f'checks.qmesh.P4r_shifted.{lev}: q - q111',
                'TZVPD': f'checks.monomers_tzvpd.P4r.R1f_kJ: ({ml} - PBE0_D3) - (profile.P4r.{lev} - profile.P4r.PBE0)',
                'delta_T': 'checks.pair_cc.delta_T_P4r_kJ', 'QZVP': 'checks.pair_def2-qzvp: (dMP2_P4r - dPBE0_D3_P4r) - (profile.P4r.PBE0+MP2 - profile.P4r.PBE0)',
                'ZPE_frozen': f'nuclear.zpe_frozen_frame.H.{lev}.dZPE_kJ', 'ZPE_harmonic_HDA': 'nuclear.dZPE_P4r_minus_P0.H.zpe_HDA_kJmol (PBE)',
                'ZPE_harmonic_H_only': 'nuclear.dZPE_P4r_minus_P0.H.zpe_H_only_kJmol (PBE)'}
        for g, v in sh.items():
            add(f'{t}.shift.{g}', v, KJ, 1, conv, f'summary.json crystals[{k}].' + rule[g], f'shift of P4r - P0 at {lev}: {g}', case=k,
                note=None if lev == 'PBE0+MP2' else 'same rule as the PBE0+MP2 budget component (asserted), applied to this level')
            add(f'{t}.value.{g}', cen + v, KJ, 1, conv, f'{t}.xray + {t}.shift.{g}', f'P4r - P0 at {lev} with the {g} check', case=k)
        for lab, v in comb.items():
            add(f'{t}.combined.{slug(lab)}', v, KJ, 1, conv, 'computed: ' + lab, f'P4r - P0 at {lev}, {lab}', case=k)
        if z and z.get('margin_P4r_zpe_corrected') is None:
            add(f'{t}.ZPE_frozen.bound', z['E0_difference_kJ'], KJ, 1, conv, src(k, 'nuclear', 'zpe_frozen_frame', 'H', lev, 'E0_difference_kJ'),
                f'frozen-frame E0 difference, {z["kind"]}: not in the envelope', case=k)
        lo, hi = min(vals), max(vals)
        add(f'{t}.env_lo', lo, KJ, 1, conv, 'min over xray, xray + each check, combined', f'envelope over the checks at this level, low ({lev})', case=k)
        add(f'{t}.env_hi', hi, KJ, 1, conv, 'max over xray, xray + each check, combined', f'envelope over the checks at this level, high ({lev})', case=k)
        add(f'{t}.verdict', 'observed state lower' if lo > 0 else ('other state lower' if hi < 0 else 'spans 0'), None, None, None,
            f'{t}.env_lo / env_hi', f'sign of P4r - P0 under every check at this level ({lev})', case=k)

# ===================================================================================================== series level
SS = 'summary.json crystals[*]'
add('s.order', list(C), None, None, None, 'summary.json crystals[*].key', 'crystal order of summary.json (sorted by dpKa)')
for grp, mem in [('all', KEYS), ('salt', SALT), ('cocrystal', COC), ('core', CORE)]:
    add(f's.n.{grp}', len(mem), None, 0, None, 'crystals.json set / observed, summary.json crystals', f'crystals counted: {grp}', members=mem)
for st in ['core', 'optional', 'extension']:
    m = [k for k in KEYS if CJ[k]['set'] == st]
    add(f's.n.set.{st}', len(m), None, 0, None, 'crystals.json crystals[*].set', f'crystals of the {st} set with results', members=m)
add('s.not_counted', [f'{a}: {b}' for a, b in SJ['counts_table'][0]['excluded']], None, None, None, 'summary.json counts_table[0].excluded', 'not counted in any row')
dp = {k: CJ[k]['dpka'] for k in KEYS}
rng('s.dpka.all', dp, None, 2, None, 'dpKa of the 13', 'crystals.json crystals[*].dpka')
rng('s.dpka.without_perchlorate', {k: v for k, v in dp.items() if k != 'perchlorate'}, None, 2, None, 'dpKa without the positive control', 'crystals.json')
V = lambda key: {k: F[f'x.{k}.{key}']['value'] for k in KEYS if F.get(f'x.{k}.{key}') and F[f'x.{k}.{key}']['value'] is not None}
for lev in ['PBE0', 'PBE']:
    r1 = V(f'R1f.{lev}')
    for grp, mem in [('all', KEYS), ('salt', SALT), ('cocrystal', COC)]:
        rng(f's.R1f.{lev}.{grp}', {k: r1[k] for k in mem}, KJ, 1, 'dE_PT', f'R1f {lev} (isolated pair), {grp}', f'{SS}.R1f.{lev} (dE_PT)')
    count(f's.R1f.{lev}.neutral_lower', KEYS, {k: r1[k] > 0 for k in KEYS}, f'isolated pair neutral (R1f {lev} > 0)', f'{SS}.R1f.{lev}')
    env, low, comp = V(f'dE_env.{lev}'), V(f'dE_env_lowering.{lev}'), V(f'compensation.{lev}')
    for grp, mem in [('all', KEYS), ('salt', SALT), ('cocrystal', COC)]:
        rng(f's.dE_env.{lev}.{grp}', {k: env[k] for k in mem}, KJ, 1, 'dE_PT', f'dE_env {lev}, {grp}', f'{SS}.dE_env.{lev} (dE_PT)')
        rng(f's.dE_env_lowering.{lev}.{grp}', {k: low[k] for k in mem}, KJ, 1, 'lowering', f'lowering of the ion pair by the environment, {lev}, {grp}', f'{SS}.dE_env.{lev}')
        rng(f's.compensation.{lev}.{grp}', {k: comp[k] for k in mem}, None, 2, 'unsigned', f'-dE_env / R1f, {lev}, {grp}', f'{SS}.dE_env / R1f')
    count(f's.dE_env.{lev}.toward_ion_pair', KEYS, {k: env[k] < 0 for k in KEYS}, f'environment lowers the ion pair (dE_env {lev} < 0)', f'{SS}.dE_env.{lev}')
for lev in LEV:
    r5 = V(f'R5.P4.{lev}')
    for grp, mem in [('all', KEYS), ('salt', SALT), ('cocrystal', COC)]:
        if all(k in r5 for k in mem):
            rng(f's.R5.P4.{lev}.{grp}', {k: r5[k] for k in mem}, KJ, 1, 'dE_PT', f'R5 at P4, {lev}, {grp}', f'{SS}.R5.{lev} (dE_PT)')
# cocrystal "margins of 5-18": R5 PBE0+MP2 at P4 (= dE_PT); spread between pair corrections at P4; budget sizes
sp = {k: max(F[f'x.{k}.R5.P4.{l}']['value'] for l in CORR) - min(F[f'x.{k}.R5.P4.{l}']['value'] for l in CORR) for k in KEYS}
for grp, mem in [('all', KEYS), ('salt', SALT), ('cocrystal', COC)]:
    rng(f's.pair_correction_spread.P4.{grp}', {k: sp[k] for k in mem}, KJ, 1, 'unsigned',
        f'spread of R5 at P4 over MP2 / SCS-MP2 / wB97X-D3(BJ), {grp}', f'{SS}.R5.PBE0+(MP2|SCS_MP2|wB97X_D3BJ)')
    rng(f's.budget.R5_P4.rss.{grp}', {k: C[k]['budget']['R5_P4']['rss'] for k in mem}, KJ, 1, 'unsigned', f'budget rss of R5 at P4, {grp}', f'{SS}.budget.R5_P4.rss')
    rng(f's.budget.R5_P4.width.{grp}', {k: np.ptp(C[k]['budget']['R5_P4']['envelope']) for k in mem}, KJ, 1, 'unsigned',
        f'budget envelope width of R5 at P4, {grp}', f'{SS}.budget.R5_P4.envelope')
mg = {k: F[f'x.{k}.margin.PBE0+MP2']['value'] for k in COC if f'x.{k}.margin.PBE0+MP2' in F}
rng('s.margin.PBE0+MP2.cocrystal', mg, KJ, 1, 'margin', 'classification margin at PBE0+MP2 (cocrystals with an other-side minimum)', f'{SS}.margin_other_side.PBE0+MP2.margin')
nm = {k: F[f'x.{k}.no_other_side_minimum.PBE0+MP2']['value'] for k in COC}
count('s.no_other_side_minimum.PBE0+MP2.cocrystal', COC, nm, 'cocrystals without an other-side minimum at PBE0+MP2', f'{SS}.margin_other_side.PBE0+MP2')
cr = {k: F[f'x.{k}.crossing.PBE0+MP2']['value'] for k in SALT}
rng('s.crossing.PBE0+MP2.salt', cr, KJ, 1, 'margin', 'crossing energy at delta = 0, PBE0+MP2, salts', f'{SS}.margin_other_side.PBE0+MP2.crossing_energy_delta0')
# reverse gaps and LR means
for g in ['pair_repeat', 'pair_sc', 'pair_gas', 'site_repeat']:
    mem = [k for k in KEYS if g in C[k]['reverse']]
    count(f's.reverse.{g}.other_wins', mem, {k: F[f'x.{k}.reverse.{g}.other_wins']['value'] for k in mem},
          f"{g}: the other state wins in its own environment (reverse gap)", f'{SS}.reverse.{g}.reverse_kJ < 0')
    count(f's.reverse.{g}.lr_observed', mem, {k: F[f'x.{k}.reverse.{g}.lr_observed']['value'] for k in mem},
          f'{g}: LR mean of the two gaps on the observed side', f'{SS}.reverse.{g}.lr_mean_kJ > 0')
    count(f's.reverse.{g}.fwd_observed', mem, {k: C[k]['reverse'][g]['embedded_single_site_kJ'] > 0 for k in mem},
          f'{g}: forward gap (observed environment) on the observed side', f'{SS}.reverse.{g}.embedded_single_site_kJ > 0')
    rng(f's.reverse.{g}.lambda', {k: C[k]['reverse'][g]['lambda_kJ'] for k in mem}, KJ, 1, 'unsigned', f'{g}: lambda', f'{SS}.reverse.{g}.lambda_kJ')
# closure
for t, lab, sg in [('cluster_overshoot', 'R4 - T1 toward the observed state (cluster overshoots the periodic single site)', 'toward_observed'),
                   ('relaxation_giveback', 'lattice relaxation R5 - N_vert given back from the observed state, PBE0', 'toward_observed')]:
    rng(f's.closure.{t}', V(f'closure.{t}'), KJ, 1, sg, lab, f'{SS}.closure')
for t in ['cluster_minus_periodic_single_site', 'reorganization_pbe0', 'cooperativity_preregistered_pbe0', 'image_T2_minus_T1_pbe']:
    for grp, mem in [('salt', SALT), ('cocrystal', COC)]:
        rng(f's.closure.{t}.{grp}', {k: V(f'closure.{t}')[k] for k in mem}, KJ, 1, 'dE_PT', f'{t} (dE_PT), {grp}', f'{SS}.closure.{t}')
rng('s.closure.cooperativity_toward_observed', {k: C[k]['closure']['cooperativity_preregistered_pbe0'] for k in KEYS}, KJ, 1, 'toward_observed',
    'cooperativity N_vert - T1 (item C, PBE0): extra cost per pair of moving every copy', f'{SS}.closure.cooperativity_preregistered_pbe0')
rng('s.closure.image_abs', {k: abs(V('closure.image_T2_minus_T1_pbe')[k]) for k in KEYS}, KJ, 1, 'unsigned', '|T2 - T1| (PBE), image effect', f'{SS}.closure.T2_pbe_kJ - T1_pbe_kJ')
count('s.closure.sign_undetermined', KEYS, V('closure.sign_undetermined'), 'sign of R4 - T1 undetermined with the T2 estimate', f'{SS}.closure.cluster_minus_periodic_single_site_sign')
# counts by rung (summary.json counts_table) with the core column computed from its missed / excluded lists
STATUS = lambda s: 'plan' if s == 'pre-registered' else ('extension' if s.startswith('extension') else 'added')
for i, row in enumerate(SJ['counts_table']):
    excl = {e[0] for e in row['excluded']}
    counted = [k for k in KEYS if k not in excl]
    missed = {m.split(' ')[0] for m in row['all']['missed']}
    rk = f's.count.{i:02d}'
    add(f'{rk}.rung', row['rung'], None, None, None, f'summary.json counts_table[{i}].rung', 'rung')
    add(f'{rk}.criterion', row['criterion'], None, None, None, f'summary.json counts_table[{i}].criterion', 'criterion')
    add(f'{rk}.level', row['level'], None, None, None, f'summary.json counts_table[{i}].level', 'level')
    add(f'{rk}.status', STATUS(row['status']), None, None, None, f'summary.json counts_table[{i}].status ("{row["status"]}")',
        'plan = fixed in the dated plan before the production calculations; extension = P4r and extra points (19-20 Oct); added = later')
    for grp in ['all', 'salt', 'cocrystal']:
        g = row[grp]
        add(f'{rk}.{grp}', g['reproduced'], None, 0, None, f'summary.json counts_table[{i}].{grp}.reproduced', f'reproduced, {grp}',
            of=g['of'], text_of=f'{g["reproduced"]}/{g["of"]}', ci95=g['ci95'], missed=g['missed'])
        mm = [k for k in counted if (grp == 'all' or C[k]['observed'] == grp)]
        check(f'count {i} {grp}', len(mm) == g['of'] and len([k for k in mm if k not in missed]) == g['reproduced'])
        if grp == 'all':
            check(f'cp95 {i}', cp95(g['reproduced'], g['of']) == g['ci95'], f'{cp95(g["reproduced"], g["of"])} vs {g["ci95"]}')
    core = [k for k in counted if k in CORE]
    n = len([k for k in core if k not in missed])
    add(f'{rk}.core', n, None, 0, None, f'summary.json counts_table[{i}].all.missed / excluded ∩ crystals.json set == core',
        'reproduced, core set', of=len(core), text_of=f'{n}/{len(core)}', ci95=cp95(n, len(core)), missed=[k for k in core if k in missed])
ROW = {(r['rung'], r['criterion'], r['level']): i for i, r in enumerate(SJ['counts_table'])}
FULL = 'class, full profile (P4r + extra points)'
for name, key in [('R1f', ('R1f', 'sign at P4 (P4 - P0 > 0)', 'PBE0')),
                  ('R5_full_MP2', ('R5', FULL, 'PBE0+MP2')), ('R5_full_SCS', ('R5', FULL, 'PBE0+SCS_MP2')),
                  ('R5_full_wB97X', ('R5', FULL, 'PBE0+wB97X_D3BJ')), ('R5_full_PBE0', ('R5', FULL, 'PBE0')), ('R5_full_PBE', ('R5', FULL, 'PBE')),
                  ('R5_plan_wB97X', ('R5', 'class, pre-registered profile (P0-P4)', 'PBE0+wB97X_D3BJ')),
                  ('R5_resolved', ('R5 corrected, resolved', 'class observed and budget envelope excludes 0 (margin at P4r, else R5 at P4)', 'PBE0+MP2')),
                  ('R4a_single', ('R4a single site (field, REPEAT)', 'sign at P4 (P4 - P0 > 0)', 'PBE0')),
                  ('R4a_coll_Eint2', ('R4a collective (field, REPEAT, E_int/2)', 'sign at P4 (P4 - P0 > 0)', 'PBE0')),
                  ('R4a_coll_LR', ('R4a collective (field, REPEAT, LR mean)', 'sign at P4 (P4 - P0 > 0)', 'PBE0'))]:
    i = ROW[key]
    for grp in ['all', 'salt', 'cocrystal', 'core']:
        r0 = F[f's.count.{i:02d}.{grp}']
        add(f's.count.{name}.{grp}', r0['value'], None, 0, None, f's.count.{i:02d}.{grp} ({r0["source"]})', f'{key[0]} | {key[1]} | {key[2]}: {grp}',
            of=r0['of'], text_of=r0['text_of'], missed=r0['missed'])
# independent re-scoring of three rows from the crystal records
for name, ok in [('R1f', {k: C[k]['R1f']['PBE0'] > 0 for k in KEYS}),
                 ('R5_full_MP2', {k: C[k]['class']['PBE0+MP2'][0] == 'observed' for k in KEYS}),
                 ('R5_resolved', {k: C[k]['class']['PBE0+MP2'][0] == 'observed' and C[k]['resolution']['verdict'].startswith('resolved') for k in KEYS})]:
    for grp, mem in [('all', KEYS), ('core', CORE)]:
        check(f'rescore {name} {grp}', sum(ok[k] for k in mem) == F[f's.count.{name}.{grp}']['value'])
add('s.unresolved', [k for k in KEYS if not C[k]['resolution']['verdict'].startswith('resolved')], None, None, None,
    f'{SS}.resolution.verdict', 'crystals whose corrected class the budget does not resolve')
for rk_, g in SJ['counts_by_class'].items():
    add(f's.count_by_class.{slug(rk_)}', g['reproduced'], None, 0, None, f'summary.json counts_by_class["{rk_}"]', rk_,
        of=g['of'], text_of=f'{g["reproduced"]}/{g["of"]}', ci95=g['ci95'])
# continuum, field/shell, decomposition, robustness ranges
eq = {k: C[k]['eps_eq'] for k in KEYS}
num = lambda s: float(s) if isinstance(s, (int, float)) or re.fullmatch(r'[0-9.]+', s) else None
count('s.pcm.reproduces_R5', KEYS, {k: num(eq[k]) is not None for k in KEYS}, 'a continuum epsilon in 1-78 reproduces R5 PBE0+MP2', f'{SS}.eps_eq')
rng('s.pcm.eps_eq', {k: num(eq[k]) for k in KEYS if num(eq[k]) is not None}, None, 1, None, 'epsilon that reproduces R5 PBE0+MP2', f'{SS}.eps_eq')
count('s.split.presentable.salt', SALT, {k: C[k]['ladder']['repeat']['split_presentable'] is True for k in SALT}, 'field/shell split presentable (salts)', f'{SS}.ladder.repeat.split_presentable')
for t in ['field', 'shell', 'nonadditivity']:
    rng(f's.split.{t}.salt', {k: V(f'split.{t}')[k] for k in SALT}, KJ, 1, 'dE_PT', f'field/shell split {t} (plan), salts', f'{SS}.ladder.repeat.{t}')
for t in ['shell', 'shell_charges', 'electrostatic', 'outer', 'beyond_electrostatic', 'cross']:
    rng(f's.decomp.{t}.salt', {k: V(f'decomp.{t}')[k] for k in SALT}, KJ, 1, 'dE_PT', f'exploratory decomposition {t}, salts', f'{SS}.decomposition.repeat.{t}')
rng('s.shift.k.P4', V('shift.k.P4'), KJ, 1, 'dE_PT', 'k-mesh shift of R5 at P4 (PBE)', f'{SS}.kcheck_shift')
rng('s.shift.k.P4r', V('shift.k.P4r'), KJ, 1, 'margin', 'k-mesh shift at P4r (PBE), cocrystals', f'{SS}.kcheck_shift_P4r')
rng('s.shift.q.P4r', V('shift.q.P4r'), KJ, 1, 'margin', 'q-mesh shift at P4r (PBE0), cocrystals', f'{SS}.checks.qmesh.shift_kJ_per_fu')
for grp, mem in [('salt', SALT), ('cocrystal', COC)]:
    rng(f's.shift.TZVPD.R5_P4.{grp}', {k: V('budget.R5_P4.basis_TZVPD')[k] for k in mem}, KJ, 1, 'dE_PT', f'def2-TZVPD shift of the MP2 pair correction at P4, {grp}', f'{SS}.budget.R5_P4.components.basis_TZVPD')
    rng(f's.shift.TZVPD.dE_env.{grp}', {k: V('budget.dE_env.basis_TZVPD')[k] for k in mem}, KJ, 1, 'dE_PT', f'def2-TZVPD shift of dE_env (R1f in TZVPD), {grp}', f'{SS}.budget.dE_env.components.basis_TZVPD')
# dpKa (secondary): within-family slopes are recomputed from the dE_PT values to pin their sign
sec = SJ['secondary']
add('s.dpka.RT2303', sec['RT2303_kJ_per_pKa'], KJ, 1, 'unsigned', 'summary.json secondary.RT2303_kJ_per_pKa', '2.303 RT per pKa unit (298 K)')
QK = {'R1f PBE0': 'R1f.PBE0', 'R5 PBE0+MP2': 'R5.P4.PBE0+MP2', 'dE_env PBE0': 'dE_env.PBE0'}
for fam, w in sec['within_family'].items():
    fk = slug(fam)
    add(f's.dpka.family.{fk}.n', w['n'], None, 0, None, f'summary.json secondary.within_family["{fam}"].n', f'{fam} family size', members=w['members'])
    for q, qk in QK.items():
        if q in w:
            x = [CJ[m]['dpka'] for m in w['members']]
            y = [F[f'x.{m}.{qk}']['value'] for m in w['members']]
            sl = np.polyfit(x, y, 1)[0]
            check(f'slope sign {fam} {q}', abs(sl - w[q]['slope_kJ_per_pKa']) < 0.01, f'{sl} vs {w[q]["slope_kJ_per_pKa"]}')
            check(f'spearman {fam} {q}', abs(stats.spearmanr(x, y)[0] - w[q]['spearman']) < 0.01)
            add(f's.dpka.family.{fk}.{slug(q)}.slope', w[q]['slope_kJ_per_pKa'], KJ, 1, 'dE_PT', f'summary.json secondary.within_family["{fam}"]["{q}"].slope_kJ_per_pKa',
                f'least-squares slope of {q} (dE_PT) against dpKa, per pKa unit, {fam} family')
            add(f's.dpka.family.{fk}.{slug(q)}.spearman', w[q]['spearman'], None, 1, None, f'summary.json secondary.within_family["{fam}"]["{q}"].spearman', f'Spearman rho, {q}, {fam}')
for q, w in sec['pooled_added_22oct'].items():
    x = [CJ[k]['dpka'] for k in KEYS]
    y = [F[f'x.{k}.{QK[q]}']['value'] for k in KEYS]
    check(f'pooled {q}', abs(stats.spearmanr(x, y)[0] - w['spearman']) < 1e-3)
    qq = f's.dpka.pooled.{slug(q)}'
    pq = f'summary.json secondary.pooled_added_22oct["{q}"]'
    add(f'{qq}.spearman', w['spearman'], None, 2, None, f'{pq}.spearman', f'Spearman rho against dpKa, all 13 (added), {q}', n=w['n'])
    add(f'{qq}.p', w['p'], None, 3, None, f'{pq}.p', f'p value, {q}')
    add(f'{qq}.ci_lo', w['ci95_fisher_z'][0], None, 2, None, f'{pq}.ci95_fisher_z[0]', f'95 % interval (Fisher z), low, {q}')
    add(f'{qq}.ci_hi', w['ci95_fisher_z'][1], None, 2, None, f'{pq}.ci95_fisher_z[1]', f'95 % interval (Fisher z), high, {q}')
    add(f'{qq}.spearman_without_off_axis', w['spearman_without_off_axis'], None, 2, None, f'{pq}.spearman_without_off_axis', f'rho without sulfamate, {q}')
for q, w in sec['DA_vs_dpka'].items():
    add(f's.dpka.DA.{q}.spearman', w['spearman'], None, 2, None, f'summary.json secondary.DA_vs_dpka.{q}.spearman', f'Spearman rho, {q} against dpKa', n=w['n'])
    add(f's.dpka.DA.{q}.p', w['p'], None, 3, None, f'summary.json secondary.DA_vs_dpka.{q}.p', f'p value, {q} against dpKa')
rng('s.DA.xray', V('DA.xray'), A, 3, None, 'D...A X-ray of the transferring bond', f'{SS}.DA_xray_A')
# gate G2b and the polymorph pair
g2 = re.search(r'Bridge G2b:.*?within\s+([0-9.]+)\s*kJ/mol', (OPTC / 'PLAN.md').read_text(), re.S)
add('s.G2b.threshold', float(g2.group(1)), KJ, 1, 'unsigned', 'optc/PLAN.md: "Bridge G2b: PySCF and QE pair ΔE within N kJ/mol"', 'gate G2b threshold')
add('s.G2b.saph.miss', F['x.saph.bridge.abs_difference']['value'] - F['s.G2b.threshold']['value'], KJ, 1, 'unsigned',
    'x.saph.bridge.abs_difference - s.G2b.threshold', 'by how much saph misses gate G2b')
for f_, v in SJ['polymorph_brsalt_brcoc'].items():
    if isinstance(v, (int, float)):
        add(f's.polymorph.{f_.removesuffix("_kJ_per_fu")}', v, KJ, 1, 'as_stored', f'summary.json polymorph_brsalt_brcoc.{f_}',
            'E(brcoc) - E(brsalt) per formula unit, static lattice energy (q_shift: change from the q-mesh step)')
m7 = re.search(r'\(commit ([0-9a-f]{7})\)', (OPTC / 'CTC_REWRITE_PLAN.md').read_text())
add('s.plan_commit', m7.group(1), None, None, None, 'optc/CTC_REWRITE_PLAN.md section 2 wording rule', 'commit of the dated plan')

# ===================================================================================================== plan section 1 against the facts
PLAN = [  # (what the plan says, fact keys, decimals, how)
    ('dE_PT +35 to +122', ['s.R1f.PBE0.all.min', 's.R1f.PBE0.all.max'], 0, None),
    ('lowers the ion pair by 116-163 in the salts', ['s.dE_env_lowering.PBE0.salt.min', 's.dE_env_lowering.PBE0.salt.max'], 0, None),
    ('by 33-86 in the cocrystals', ['s.dE_env_lowering.PBE0.cocrystal.min', 's.dE_env_lowering.PBE0.cocrystal.max'], 0, None),
    ('cancels 75-95 %', ['s.compensation.PBE0.cocrystal.min', 's.compensation.PBE0.cocrystal.max'], 2, None),
    ('dapsone 0.92, mppcp 0.82, pcplut 0.95, nbaetpy 0.75', [f'x.{k}.compensation.PBE0' for k in ['dapsone', 'mppcp', 'pcplut', 'nbaetpy']], 2, None),
    ('margins of only 5-18', ['s.R5.P4.PBE0+MP2.cocrystal.min', 's.R5.P4.PBE0+MP2.cocrystal.max'], 0, None),
    ('12/13, 8/13, 8/13 (reverse, three charge models)', ['s.reverse.pair_repeat.other_wins', 's.reverse.pair_sc.other_wins', 's.reverse.pair_gas.other_wins'], 0, 'of'),
    ('2/5 for the embedded cluster', ['s.reverse.site_repeat.other_wins'], 0, 'of'),
    ('cluster exceeds the periodic single site by 7-102', ['s.closure.cluster_overshoot.min', 's.closure.cluster_overshoot.max'], 0, None),
    ('lattice relaxation gives back 14-78', ['s.closure.relaxation_giveback.min', 's.closure.relaxation_giveback.max'], 0, None),
    ('core 9/9', ['s.count.R5_full_MP2.core'], 0, 'of'), ('resolved 8/9', ['s.count.R5_resolved.core'], 0, 'of'),
    ('all 13/13', ['s.count.R5_full_MP2.all'], 0, 'of'), ('12/13 with wB97X on the extended profile', ['s.count.R5_full_wB97X.all'], 0, 'of'),
    ('11/13 resolved', ['s.count.R5_resolved.all'], 0, 'of'),
    ('phosphate family -9.7 per pKa unit', ['s.dpka.family.h3po4.r1f_pbe0.slope'], 1, None)]
PLAN_CHECK = []
for said, keys, d, how in PLAN:
    got = ', '.join(F[k]['text_of'] if how == 'of' else fmt(F[k]['value'] * (100 if '%' in said else 1), 0 if '%' in said else d, False) for k in keys)
    PLAN_CHECK.append(dict(plan=said, facts=keys, computed=got))

# ===================================================================================================== write
def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


meta = dict(generated_by='paper/ctc/numbers/facts.py', n_facts=len(F), signs=SIGNS,
            sign_factor={k: S[k] for k in KEYS},
            rounding='text = value rounded half to even at precision (shortest repr), true minus; + on signed energies',
            inputs={'optc/results/summary.json': MD5, 'optc/crystals.json': md5(OPTC / 'crystals.json'),
                    **{f'optc/structures/{k}/cell.json': md5(OPTC / f'structures/{k}/cell.json') for k in C}},
            raw_profiles='optc-merged/results/<key>/periodic.json (D...A along the profile only)',
            budget_label=SJ['budget_label'], counts_note=SJ['counts_table_note'],
            not_in_results={'nitrate': 'optional; no results directory', 'brcoc': 'no transferring bond (polymorph ranking only)'},
            checks_passed=len(CHECKS), missing=MISSING, plan_check=PLAN_CHECK)
(HERE / 'data').mkdir(exist_ok=True)
(HERE / 'data/facts.json').write_text(json.dumps({'_meta': meta, **F}, indent=1, ensure_ascii=False) + '\n')
print(f'{len(F)} facts, {len(CHECKS)} checks passed, {len(MISSING)} missing -> {HERE / "data/facts.json"}')
for r in PLAN_CHECK:
    print(f'  plan: {r["plan"]:55s} computed: {r["computed"]}')
