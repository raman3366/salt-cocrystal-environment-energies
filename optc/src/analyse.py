#!/usr/bin/env python3
"""Collect the option C results into one summary (optc/PLAN.md: endpoints, classification rule, decomposition).

    python3 optc/src/analyse.py [--results DIR] [--out optc/results/summary]   -> summary.json + summary.md

Per crystal (all energies kJ/mol per transferring pair; "other - observed", so > 0 = the observed state is lower):
  periodic   R5 at PBE-D3 (SSSP), PBE0-D3 (SG15 correction) and PBE0 + pair corrections (DF-MP2, SCS-MP2,
             wB97X-D3(BJ)), the profile against delta = d(D-H) - d(H...A): P0-P4 (pre-registered), P4r where the relax
             from P4 stays on the acceptor side and the extra points X (19 Oct 2026); class = position of the minimum of a
             monotone-preserving interpolation (PCHIP) of the corrected profile: 'observed' if delta_min < -0.1 A,
             'other' if > +0.1 A, 'shared' in between (the plan's 0.1 A window); a level without a class records why
             (class_missing, class_skipped_reason) and the counts list every crystal they leave out
  ladder     R1f (pair in vacuum), R4a single site / collective (pair in the field), R3 (pair + shell in vacuum),
             R4 single site (shell in the field); field and shell shares averaged over both orders; non-additivity;
             cooperativity R4a(collective) - R4a(single site); R4 + cooperativity against R5
  dE_env     R5 - R1f at PBE0 (and at PBE)
  continuum  eps_x (sign change of the PCM pair difference) and eps_eq (the permittivity at which it equals R5)
Counts per rung: crystals whose observed state the rung reproduces, with exact (Clopper-Pearson) 95 % intervals.

Additions of 19 Oct 2026 (optc/PLAN.md, 9:40 PM IST), each read only where its files exist:
  fixed      every main term also as E(ion pair) - E(neutral pair) (the same physical direction for salts and cocrystals)
  decomp     near-to-far, disjoint: shell = R3 - R1f, outer = R4 - R3; reverse order with R4a_outer (pair in the
             charges of everything but the transfer-site shell): outer' = R4a_outer - R1f, shell' = R4 - R4a_outer,
             cross = R4 - R4a_outer - R3 + R1f; electrostatic R4a - R1f vs the rest R4 - R4a; shell charges alone
             R4a_shellq - R1f; the same with the sc charges and the robustness rule (sign kept, change < 1/2) per term
  reverse    vertical gaps in the P4 environment (pair: three charge models; site cluster: REPEAT), linear-response
             mean and lambda
  closure    periodic single points (vert.json): T1, T1v, T2 single site, Nv collective vertical; cooperativity
             Nv - T1 (pre-registered) and Nv - T1v (vert.json); reorganization R5 - Nv; cluster minus periodic single site
             R4 - T1; the cycle R1f -> R4 -> T1 -> Nv -> R5
  checks     whole shell raw and corrected, PCM radii, QZVP and TZVPD pairs, rc35, smearing radius, runner-up bond,
             separated partners (proton-transfer and ion-pairing split of R1f), k mesh at P4/P4r, bridge, cells (vc,
             vcfree), the brsalt/brcoc ranking
  nuclear    partial-Hessian zero-point energies (hess_*.json) and a 1D sinc-DVR of the proton (H and D) on the
             corrected profile (all points, hysteresis pairs at one d(D-H) reduced to the lower energy)
  class      with and without P4r and the extra points (class_preregistered_profile: P0-P4 only); counts by class
             (salts / cocrystals) with one denominator; per rung the crystals not counted and why (counts[].excluded)

Additions of 22 Oct 2026 (gap map set A; where a printed column changed, the old value is kept under *_legacy):
  margin     margin_other_side: defined only by an other-side minimum (P4r, interior PCHIP minimum at delta > +0.1 A),
             else the crossing energy at delta = 0 (not a state); hysteresis pairs reduced; PBE50 like-for-like only
  counts     counts_table: rung x criterion (full-profile class / pre-registered P0-P4 class / sign at P4) x level,
             by class, with the crystals missed and not counted; R3 / R4 also without the empty site shells
  closure    T2_pbe0_est (T2 at PBE + the T1 hybrid shift) and R4 / R4a minus it (T1 moves 1/Z of the pairs)
  checks     kcheck_shift_P4r; EXX q mesh (checks.qmesh, polymorph pbe0_qmesh); symmetric-site terms; relaxed cells
             with the in-field correction, + delta(T), + q
  nuclear    bends perpendicular to D->A (bends_perp); frame localization (E0 against the dividing point) and the
             bound it implies; relaxed-profile DVR measured from P0; the printed p(other) from the P0 frozen frame

Additions of 22 Oct 2026 (gap map set B; pre-registered keys unchanged, new ones beside them):
  closure    pair_relaxation_single_site_pbe (T1 - T1v), cooperativity_preregistered_pbe0 / _pbe (Nv - T1),
             cooperativity_vertical_f1overZ_to_1_pbe (= Nv - T1v), cycle_pbe0 / cycle_pbe with the sum == R5 check
  ladder     both collective estimators: R4a_collective (E_int/2) and R4a_collective_lr (LR mean), their gap, and the
             cooperativity, R4 estimate and R5 residual from the LR mean; split_presentable 'n/a' for an empty site shell
  decomp     status 'exploratory' (item A, after the pre-registered split failed)
  budget     per crystal and quantity (margin at P4r, R5 at P4, dE_env, the REPEAT field and cluster rungs): every check as
             a signed shift, envelope and rss, an estimate (shifts at different settings, combined additively)
  resolution 'unresolved (|margin| < uncertainty)' where the budget envelope spans 0; count 'R5 corrected, resolved'
  dpka       dpka_alt, dpka_site, dpka_off_axis, dpka_quality from crystals.json, printed beside the pre-registered dpka
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np

OPTC = Path(__file__).resolve().parent.parent
HARTREE_KJ = 2625.4996394799


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return round(float(lo), 3), round(float(hi), 3)


def delta_of(row):
    p = row.get('proton') or []
    return float(np.mean([r['delta'] for r in p])) if p else None


def classify(deltas, energies):
    """position of the minimum of a PCHIP interpolation of E(delta) (both lists sorted by delta)."""
    from scipy.interpolate import PchipInterpolator
    d, e = np.array(deltas), np.array(energies)
    o = np.argsort(d)
    d, e = d[o], e[o]
    f = PchipInterpolator(d, e)
    x = np.linspace(d[0], d[-1], 2001)
    xm = float(x[int(np.argmin(f(x)))])
    cls = 'observed' if xm < -0.1 else ('other' if xm > 0.1 else 'shared')
    return cls, round(xm, 3)


def crystal_summary(key, reg, R):
    c = reg['crystals'][key]
    d = R / key
    out = {'key': key, 'name': c['name'], 'role': c['role'], 'observed': c['observed'], 'dpka': c['dpka'],
           'dpka_range': c.get('dpka_range')}
    # beside the pre-registered dpka (gap map B6, 22 Oct 2026), where crystals.json has them
    out.update({k: c[k] for k in ('dpka_alt', 'dpka_alt_range', 'dpka_alt_source', 'dpka_site', 'dpka_site_note',
                                  'dpka_off_axis', 'dpka_off_axis_reason', 'dpka_quality') if k in c})
    per = load(d / 'periodic.json')
    pair = load(d / 'mol' / 'pair.json')
    # secondary endpoint D...A against dpKa (PLAN_FINAL): the chosen bond in the X-ray structure (cell.py) and at P0
    tr = (load(OPTC / 'structures' / key / 'cell.json') or {}).get('transfer') or {}
    out.update({'family': c.get('family'), 'set': c.get('set'), 'DA_xray_A': tr.get('DA'),
                'DA_P0_A': round(per['path']['L_S'], 4) if per and (per.get('path') or {}).get('L_S') else None})
    if per:
        pts = per['points']
        out['kgrid'] = per.get('kgrid')
        prof = {}
        for pt, r in pts.items():
            if 'dE_pbe_kJmol_fu' not in r:
                continue
            tg = r.get('target')
            prof[pt] = {'delta': delta_of(r), 'PBE': r.get('dE_pbe_kJmol_fu'), 'PBE0': r.get('dE_pbe0_kJmol_fu'),
                        'PBE50': r.get('dE_pbe50_kJmol_fu'), 'other_protons_moved': r.get('other_protons_moved'),
                        'target_DH': tg[1] if tg and tg[0] == 'D-H' else None, 'extra': bool(r.get('extra'))}
            if pair and pt in pair['points']:
                for lev in ('MP2', 'SCS_MP2', 'wB97X_D3BJ'):
                    corr = pair['points'][pt].get(f'corr_{lev}_kJ')
                    if corr is not None and prof[pt]['PBE0'] is not None:
                        prof[pt][f'PBE0+{lev}'] = round(prof[pt]['PBE0'] + corr, 3)
        out['profile'] = prof
        if 'P4' in prof:
            out['R5'] = {k: v for k, v in prof['P4'].items() if k not in ('delta', 'other_protons_moved')}
        if per.get('kcheck') and per['kcheck'].get('dE_P4_dense_kJmol_fu') is not None:
            out['kcheck_shift'] = round(per['kcheck']['dE_P4_dense_kJmol_fu'] - per['kcheck']['dE_P4_kJmol_fu'], 3)
        kc = per.get('kcheck') or {}
        if kc.get('dE_P4r_dense_kJmol_fu') is not None and kc.get('dE_P4r_kJmol_fu') is not None:
            out['kcheck_shift_P4r'] = round(kc['dE_P4r_dense_kJmol_fu'] - kc['dE_P4r_kJmol_fu'], 3)   # PBE, dense minus production
        out['P4r'] = per.get('P4r')
        # a second minimum found by the unconstrained relax from P4 (proton stays on the acceptor side) joins the profile
        p4r = per.get('P4r') or {}
        if p4r.get('proton') and p4r['proton'][0]['delta'] > 0 and p4r.get('dE_pbe_kJmol_fu') is not None:
            prof['P4r'] = {'delta': delta_of(p4r), 'PBE': p4r.get('dE_pbe_kJmol_fu'), 'PBE0': p4r.get('dE_pbe0_kJmol_fu'),
                           'PBE50': p4r.get('dE_pbe50_kJmol_fu')}
            if pair and 'P4r' in pair['points']:
                for lev in ('MP2', 'SCS_MP2', 'wB97X_D3BJ'):
                    corr = pair['points']['P4r'].get(f'corr_{lev}_kJ')
                    if corr is not None and prof['P4r']['PBE0'] is not None:
                        prof['P4r'][f'PBE0+{lev}'] = round(prof['P4r']['PBE0'] + corr, 3)
        cls, miss = {}, {}
        if len(prof) >= 5:
            for lev in ('PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ'):
                if all(lev in prof[p] and prof[p][lev] is not None for p in prof):
                    cls[lev] = classify([prof[p]['delta'] for p in prof], [prof[p][lev] for p in prof])
                else:
                    miss[lev] = [p for p in prof if prof[p].get(lev) is None]
        else:
            out['class_skipped_reason'] = f"{len(prof)} profile points (< 5): {', '.join(prof) or 'none'}"
        out['class'] = cls
        out['class_missing'] = miss       # gap map B8: the levels left without a class and the points that lack them
    else:
        out['class_skipped_reason'] = 'no periodic.json'
    if pair:
        P = pair['points']
        if 'P4' in P:
            out['R1f'] = {'PBE0': P['P4'].get('dPBE0_D3_kJ'), 'PBE': P['P4'].get('dPBE_D3_kJ'),
                          'MP2': P['P4'].get('dMP2_kJ'), 'SCS_MP2': P['P4'].get('dSCS_MP2_kJ'),
                          'wB97X_D3BJ': P['P4'].get('dwB97X_D3BJ_kJ')}
    if out.get('R5') and out.get('R1f'):
        out['dE_env'] = {'PBE0': round(out['R5']['PBE0'] - out['R1f']['PBE0'], 3) if out['R5'].get('PBE0') is not None else None,
                         'PBE': round(out['R5']['PBE'] - out['R1f']['PBE'], 3) if out['R1f'].get('PBE') is not None else None}
    lad = {}
    for model in ('repeat', 'sc', 'gas'):
        ep = load(d / 'mol' / f'emb_pair_{model}.json')
        # shell rung (optc/PLAN.md, 19 Oct 2026 8:26 AM IST): PRIMARY = the transfer-site shell (molecules H-bonded to D,
        # the H atoms on D or A), all def2-TZVP; a crystal with no such molecule has R3 = R1f and R4 = R4a by definition.
        # CHECK = the whole first shell (<= 150 QM atoms), locally dense basis + [TZVP - locally dense] of its site cluster.
        ec = load(d / 'mol' / f'emb_cluster_{model}_site.json')
        site_meta = load(d / 'emb' / f'{model}_site' / 'emb.json')
        full_ldb = load(d / 'mol' / f'emb_cluster_{model}_ldb.json')
        site_ldb = load(d / 'mol' / f'emb_cluster_{model}_site_ldb.json')
        if full_ldb and ec and site_ldb and not full_ldb.get('skipped'):
            corr = {k: ec['derived'][k] - site_ldb['derived'][k] for k in ('vacuum_kJ', 'embedded_single_site_kJ')
                    if k in ec.get('derived', {}) and k in site_ldb.get('derived', {})}
            full = {k: round(full_ldb['derived'][k] + corr[k], 3) for k in corr if k in full_ldb.get('derived', {})}
        else:
            full = None
        if not ep:
            continue
        m = {}
        dp, dc = ep.get('derived', {}), (ec or {}).get('derived', {})
        m['R1f'] = dp.get('vacuum_kJ')
        m['R4a_single'] = dp.get('embedded_single_site_kJ')
        m['R4a_collective'] = dp.get('embedded_collective_kJ')
        m['R3'] = dc.get('vacuum_kJ')
        m['R4_single'] = dc.get('embedded_single_site_kJ')
        if site_meta and site_meta.get('n_shell') == 0 and m.get('R1f') is not None:
            m['R3'], m['R4_single'] = m['R1f'], m.get('R4a_single')
            m['site_shell_empty'] = True
        if full:
            m['R3_full_shell'] = full.get('vacuum_kJ')
            m['R4_full_shell'] = full.get('embedded_single_site_kJ')
        m['n_site_shell'] = site_meta.get('n_shell') if site_meta else None
        if None not in (m['R1f'], m['R4a_single']):
            m['field_only'] = round(m['R4a_single'] - m['R1f'], 3)
        if None not in (m['R1f'], m['R4a_single'], m['R3'], m['R4_single']):
            f1, f2 = m['R4a_single'] - m['R1f'], m['R4_single'] - m['R3']
            s1, s2 = m['R3'] - m['R1f'], m['R4_single'] - m['R4a_single']
            m['field'] = round((f1 + f2) / 2, 3)
            m['shell'] = round((s1 + s2) / 2, 3)
            m['field_order_spread'] = round(abs(f1 - f2), 3)
            m['nonadditivity'] = round(m['R4_single'] - m['R4a_single'] - m['R3'] + m['R1f'], 3)
            tot = m['R4_single'] - m['R1f']
            m['field_share'] = round(m['field'] / tot, 3) if abs(tot) > 1e-6 else None
            if m.get('site_shell_empty'):     # shell = 0 by definition: no split to judge (gap map B3; was False)
                m['split_presentable'], m['split_presentable_legacy'] = 'n/a (empty site shell)', False
            else:
                m['split_presentable'] = abs(m['nonadditivity']) < min(abs(m['field']), abs(m['shell'])) if m['field'] and m['shell'] else False
        if None not in (m.get('R4a_collective'), m.get('R4a_single')):
            m['cooperativity'] = round(m['R4a_collective'] - m['R4a_single'], 3)
            if m.get('R4_single') is not None:
                m['R4_collective_est'] = round(m['R4_single'] + m['cooperativity'], 3)
                if out.get('R5', {}).get('PBE0') is not None:
                    m['residual_R5_minus_R4'] = round(out['R5']['PBE0'] - m['R4_collective_est'], 3)
        # gap map B2: the second collective estimator, the linear-response mean of the forward (P0 environment) and
        # reverse (P4 environment) vertical gaps; the pre-registered one above is E_emb - E_int/2
        m['R4a_collective_lr'] = dp.get('lr_mean_kJ')
        if None not in (m.get('R4a_collective'), m['R4a_collective_lr']):
            m['collective_estimator_gap'] = round(m['R4a_collective'] - m['R4a_collective_lr'], 3)
        if None not in (m['R4a_collective_lr'], m.get('R4a_single')):
            m['cooperativity_lr'] = round(m['R4a_collective_lr'] - m['R4a_single'], 3)
            if m.get('R4_single') is not None:
                m['R4_collective_est_lr'] = round(m['R4_single'] + m['cooperativity_lr'], 3)
                if out.get('R5', {}).get('PBE0') is not None:
                    m['residual_R5_minus_R4_lr'] = round(out['R5']['PBE0'] - m['R4_collective_est_lr'], 3)
        lad[model] = m
    out['ladder'] = lad
    if 'repeat' in lad and 'sc' in lad and lad['repeat'].get('field_share') is not None and lad['sc'].get('field_share') is not None:
        a, b = lad['repeat']['field_share'], lad['sc']['field_share']
        out['field_split_robust'] = (a * b > 0) and abs(a - b) <= 0.5 * abs(a)
    pcm = load(d / 'mol' / 'pcm_r1.2.json')
    if pcm and pcm.get('dE_kJ'):
        e = sorted((float(k), v) for k, v in pcm['dE_kJ'].items())
        out['pcm_dE'] = e
        epsx = None
        for (e1, v1), (e2, v2) in zip(e, e[1:]):
            if v1 * v2 <= 0 and v1 != v2:
                epsx = round(math.exp(math.log(e1) + (math.log(e2) - math.log(e1)) * v1 / (v1 - v2)), 2)
                break
        out['eps_x'] = epsx if epsx else f'not reached in 1-78 (dE at 78: {e[-1][1]})'
        if out.get('R5', {}).get('PBE0') is not None:
            t = out['R5']['PBE0']
            epseq = None
            for (e1, v1), (e2, v2) in zip(e, e[1:]):
                if (v1 - t) * (v2 - t) <= 0 and v1 != v2:
                    epseq = round(math.exp(math.log(e1) + (math.log(e2) - math.log(e1)) * (v1 - t) / (v1 - v2)), 2)
                    break
            out['eps_eq'] = epseq if epseq else 'outside 1-78'
    out['bridge'] = load(d / 'bridge.json')
    out['vc'] = load(d / 'vc.json')
    extend(out, d)
    return out


KIN_KJ = 0.2016609     # hbar^2 / (2 amu A^2) in kJ/mol
MASS_H, MASS_D = 1.007825, 2.014102


def dvr_1d(delta, energy, mass, temp=None, wall=150.0, n=601, ref=0.0):
    """sinc-DVR (Colbert-Miller) of the transferring proton on E(delta) (kJ/mol per pair): x = delta / 2 is its
    displacement along D...A (A); the PCHIP profile is continued beyond the sampled ends by a quadratic wall that
    keeps the end slope and rises by `wall` kJ/mol over 0.3 A of delta.  Returns the ground state and the
    probabilities on each side of the barrier top between the two lowest minima (or of delta = 0).
    E0_above_ref = E0_above_min - V_ref_minus_min (zero point above the lowest V, less the potential relaxation from
    `ref` to that minimum); E0_minus_divide < 0: the ground state lies below the dividing point (localized on one
    side); p_ref_side: the population on the side of the divide that holds `ref` (a frozen frame's own side)."""
    from scipy.interpolate import PchipInterpolator
    o = np.argsort(delta)
    d, e = np.asarray(delta, float)[o], np.asarray(energy, float)[o]
    f = PchipInterpolator(d, e)
    lo, hi = d[0] - 0.3, d[-1] + 0.3
    g = np.linspace(lo, hi, n)
    k = wall / 0.3 ** 2
    s0, s1 = f.derivative()(d[0]), f.derivative()(d[-1])
    V = np.where(g < d[0], e[0] + s0 * (g - d[0]) + k * (g - d[0]) ** 2,
                 np.where(g > d[-1], e[-1] + s1 * (g - d[-1]) + k * (g - d[-1]) ** 2, f(np.clip(g, d[0], d[-1]))))
    x = g / 2
    dx = x[1] - x[0]
    i = np.arange(n)
    D = i[:, None] - i[None, :]
    with np.errstate(divide='ignore'):
        T = np.where(D == 0, np.pi ** 2 / 3, 2.0 * (-1.0) ** D / np.where(D == 0, 1, D) ** 2)
    T = T * KIN_KJ / (mass * dx ** 2)
    w, v = np.linalg.eigh(T + np.diag(V))
    psi2 = v[:, 0] ** 2
    # dividing point: the highest V between the global minimum and the lowest minimum on the other side of delta 0
    inner = (g >= d[0]) & (g <= d[-1])
    gi, Vi = g[inner], V[inner]
    left, right = Vi[gi < 0], Vi[gi >= 0]
    cut = 0.0
    if len(left) and len(right):
        il, ir = int(np.argmin(left)), int(np.argmin(right)) + len(left)
        a, b = sorted((il, ir))
        cut = float(gi[a + int(np.argmax(Vi[a:b + 1]))])
    vref, vcut = float(f(min(max(ref, d[0]), d[-1]))), float(f(cut))
    out = {'E0_above_min_kJ': round(float(w[0] - V.min()), 3), 'E1_minus_E0_kJ': round(float(w[1] - w[0]), 3),
           'E0_above_ref_kJ': round(float(w[0] - vref), 3),
           'E0_above_delta0_kJ': round(float(w[0] - f(min(max(0.0, d[0]), d[-1]))), 3),     # E0_above_ref with ref = 0 (legacy)
           'mean_delta_A': round(float((psi2 * g).sum()), 4), 'divide_delta_A': round(cut, 4),
           'p_delta_above_divide': round(float(psi2[g > cut].sum()), 4),
           'V_ref_minus_min_kJ': round(vref - float(V.min()), 3), 'V_min_delta_A': round(float(g[int(np.argmin(V))]), 4),
           'V_divide_kJ': round(vcut - vref, 3), 'E0_minus_divide_kJ': round(float(w[0]) - vcut, 3),
           'p_ref_side': round(float(psi2[g > cut].sum() if ref > cut else psi2[g <= cut].sum()), 4)}
    if temp:
        beta = 1.0 / (0.0083144626 * temp)
        pw = np.exp(-beta * (w[:20] - w[0]))
        side = np.array([(v[:, j] ** 2)[g > cut].sum() for j in range(20)])
        out[f'p_above_divide_at_{temp:g}K'] = round(float((pw * side).sum() / pw.sum()), 4)
    return out


def robust_text(rb):
    """'all', 'not: <terms>' or '' (not evaluable) for the robustness flags of one crystal."""
    known = {k: v for k, v in (rb or {}).items() if v is not None}
    if not known:
        return ''
    bad = [k for k, v in known.items() if not v]
    return ('not: ' + ', '.join(bad)) if bad else ('all' if len(known) == len(rb) else 'evaluated: ' + ', '.join(known))


def term(x, y):
    return None if x is None or y is None else round(x - y, 3)


def robust(a, b):
    """the pre-registered rule applied to one term: same sign and |b - a| < |a| / 2 (a = REPEAT, b = sc)."""
    if a is None or b is None:
        return None
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return True                      # zero by definition in both models (empty transfer-site shell)
    return a * b > 0 and abs(b - a) < 0.5 * abs(a)


EXP_T = {'mppcp': 30.0, 'dapsone': 200.0, 'pcplut': 100.0, 'nbaetpy': 110.0}   # diffraction temperatures (crystals.json names)


def basis_check(bj):
    """P4 - P0 and P4r - P0 of a pair_<basis>.json, recomputed from its points.  A DF-MP2 difference is kept only when
    both points record the same RI basis (mppcp's def2-QZVP P0/P4 ran with def2-tzvp-ri, unrecorded, and P4r with
    def2-qzvp-ri: results audit, 21 Oct 2026); PBE0 does not use it."""
    P = {pt: r for pt, r in (bj.get('points') or {}).items() if r.get('done')}
    out, bad = {}, []
    for pt, sfx in (('P4', ''), ('P4r', '_P4r')):
        if pt not in P or 'P0' not in P:
            continue
        out[f'dPBE0_D3{sfx}_kJ'] = round((P[pt]['E_PBE0_D3'] - P['P0']['E_PBE0_D3']) * HARTREE_KJ, 3)
        ax = (P['P0'].get('aux_mp2'), P[pt].get('aux_mp2'))
        if None in ax or ax[0] != ax[1]:
            out[f'dMP2{sfx}_kJ'] = None
            bad.append(f'{pt} - P0 mixes RI bases {ax[1] or "unrecorded"} / {ax[0] or "unrecorded"}')
        else:
            out[f'dMP2{sfx}_kJ'] = round((P[pt]['E_MP2'] - P['P0']['E_MP2']) * HARTREE_KJ, 3)
    if bad:
        out['mp2_invalid'] = '; '.join(bad) + ' (recompute the unrecorded points: mol.py pair --basis)'
    return out


def runner_invalid(key, d, rj):
    """why pair_runner.json cannot be used, or None.  Both bonds must be the ones cell.json records now, each on the
    acceptor instance bonded through its H (mol.bond_pair, fixed 21 Oct 2026; files written before carry no A_image
    and are checked against the rule they were computed with, the instance nearest D)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import embed as E
    cr = E.Crystal(key)
    ru = cr.rec.get('runner_up')
    if not ru:
        return 'no runner-up bond in cell.json'
    _, _, X0 = E.read_extxyz(d / 'P0.extxyz')
    W0 = cr.whole(X0)
    for tag, b in (('chosen', cr.rec['copies'][0]), ('runner', ru)):
        rb = (rj.get('bonds') or {}).get(tag) or {}
        if any(rb.get(k) != b[k] for k in ('D', 'H', 'A')):
            return f"{tag}: computed for {rb.get('D')}/{rb.get('H')}/{rb.get('A')}, cell.json has {b['D']}/{b['H']}/{b['A']}"
        used = tuple(rb['A_image']) if rb.get('A_image') is not None else E.image_nearest(cr, W0, b['A'], b['D'])
        if used != E.image_nearest(cr, W0, b['A'], b['H']):
            return f'{tag}: acceptor image {list(used)} is not the one bonded through H (before the 21 Oct 2026 fix)'
    return None


RY_KJ = 13.605693122994 * 96.485332    # kJ/mol per Ry (qe.py RY_EV * KJMOL_PER_EV)
CM1_TO_KJMOL = 0.01196266              # h c N_A per cm-1 (qe.py)
LEVELS = ('PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ')


def lowest_per_target(prof, lev):
    """(delta, E, point) of the profile points carrying `lev`, sorted by delta; hysteresis pairs (two relaxed
    geometries at one constrained d(D-H)) reduced to the lower energy, points without a target keyed by delta."""
    best = {}
    for pt, p in prof.items():
        e, dl, t = p.get(lev), p.get('delta'), p.get('target_DH')
        if e is None or dl is None:
            continue
        k = ('t', round(t, 3)) if t else ('d', round(dl, 3))
        if k not in best or e < best[k][1]:
            best[k] = (dl, e, pt)
    return sorted(best.values())


def margin(prof, lev, window=0.1):
    """margin of one level (gap map A1, 22 Oct 2026): the lowest sampled energy at delta > +window minus E(P0), only
    where the profile holds P4r and the PCHIP of the level has an interior local minimum at delta > +window (a state
    on the other side).  Otherwise margin None, with the PCHIP value at delta = 0 minus E(P0) (a crossing energy, not
    a state) and the lowest sampled other-side point.  Hysteresis pairs reduced to the lower energy; the points with
    |delta| <= window are listed apart; an observed-side point below P0 is noted (the reference stays P0)."""
    from scipy.interpolate import PchipInterpolator
    pts = lowest_per_target(prof, lev)
    e0 = (prof.get('P0') or {}).get(lev)
    if len(pts) < 3 or e0 is None:
        return None
    d, e = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    f = PchipInterpolator(d, e)
    x = np.linspace(d[0], d[-1], 2001)
    y = f(x)
    imin = [round(float(x[i]), 3) for i in range(1, len(x) - 1) if x[i] > window and y[i] < y[i - 1] and y[i] <= y[i + 1]]
    pt3 = lambda q: {'E': round(q[1] - e0, 3), 'point': q[2], 'delta': q[0]}
    oth = [pt3(q) for q in pts if q[0] > window]
    low = min(oth, key=lambda q: q['E']) if oth else None
    below = [pt3(q) for q in pts if q[0] < -window and q[1] < e0]
    has = (prof.get('P4r') or {}).get(lev) is not None and bool(imin) and low is not None
    pm = min(((round(float(f(xm)) - e0, 3), xm) for xm in imin), default=None)
    return {'margin': low['E'] if has else None, 'point': low['point'] if has else None, 'delta': low['delta'] if has else None,
            'no_other_side_minimum': not has, 'interior_minima_delta_A': imin,
            # the lowest sampled point can be a constrained path end (P4) away from the minimum that makes the state
            'pchip_interior_min': {'E': pm[0], 'delta': pm[1]} if pm else None,
            'point_not_at_minimum': bool(has and low['point'] != 'P4r' and all(abs(low['delta'] - xm) > window for xm in imin)),
            'crossing_energy_delta0': round(float(f(0.0)) - e0, 3), 'lowest_sampled_other_side': low,
            'window_points': [pt3(q) for q in pts if abs(q[0]) <= window],
            'obs_side_min_below_P0': min(below, key=lambda q: q['E']) if below else None,
            'reduced_away': sorted(set(p for p in prof if prof[p].get(lev) is not None) - {q[2] for q in pts})}


def bends_perp(d, pt, hz, mass):
    """bend zero point of the transferring H (gap map A4): the H block of the partial Hessian projected on the plane
    perpendicular to u = mic(D -> A) of <pt>.extxyz (the direction the frozen-frame scan moves H along, qe.py
    stage_hscan), wavenumbers as qe.frequencies; also the overlaps |<mode|u>| of the three H-only modes."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import embed as E
    lat, _, X = E.read_extxyz(d / f'{pt}.extxyz')
    _, iD, iA = hz['atoms']
    df = (X[iA] - X[iD]) @ np.linalg.inv(lat)
    df -= np.round(df)
    v = min(((df + np.array(s) - 1) @ lat for s in np.ndindex(3, 3, 3)), key=lambda v: v @ v)
    u = v / np.linalg.norm(v)
    B = np.linalg.qr(np.column_stack([u, np.eye(3)]))[0][:, 1:3]      # orthonormal, perpendicular to u
    K = np.array(hz['hessian_Ry_bohr2'])[:3, :3]
    K = 0.5 * (K + K.T)
    si = 2.1798723611e-18 / 0.529177210903e-10 ** 2 / 1.66053906660e-27 / mass   # Ry/bohr^2/amu -> s^-2
    cm = lambda w2: [float(np.sign(x) * math.sqrt(abs(x) * si) / (2 * math.pi * 2.99792458e10)) for x in w2]
    wb = cm(np.linalg.eigvalsh(B.T @ K @ B))
    w3, v3 = np.linalg.eigh(K)
    return {'zpe_bend_kJ': round(0.5 * CM1_TO_KJMOL * sum(x for x in wb if x > 0), 3), 'bend_cm1': [round(x, 1) for x in wb],
            'H_only_cm1': [round(x, 1) for x in cm(w3)], 'H_only_overlap_u': [round(float(abs(v3[:, j] @ u)), 3) for j in range(3)]}


def hybrid_energy(d, tag):
    """(total energy in Ry, q mesh) of runs/<tag>/<tag>.out: the last '!!' line of a finished hybrid run (JOB DONE,
    EXX self-consistency; parse_hybrid as qe.py summary), else (None, None)."""
    import sys
    sys.path.insert(0, str(OPTC.parent / 'dft'))
    from periodic_qe import parse_hybrid
    f = d / 'runs' / tag / f'{tag}.out'
    if not f.exists() or 'JOB DONE' not in f.read_text(errors='replace'):
        return None, None
    h = parse_hybrid(f)
    return h.get('energy_Ry'), h.get('nq')


def qmesh_runs(d):
    """{point: {nq, E_nq111_Ry, E_nq_Ry}} for every runs/pbe0_sg15_<pt>_nq<abc> (qe.py hybrid --nq) with its q = 1x1x1
    partner pbe0_sg15_<pt>, both finished and the q mesh printed in each output equal to its tag."""
    out = {}
    for run in sorted((d / 'runs').glob('pbe0_sg15_*_nq*')) if (d / 'runs').is_dir() else ():
        pt, tag = run.name[len('pbe0_sg15_'):].rsplit('_nq', 1)
        e, nq = hybrid_energy(d, run.name)
        e1, nq1 = hybrid_energy(d, f'pbe0_sg15_{pt}')
        if None in (e, e1) or ''.join(map(str, nq or [])) != tag or nq1 != [1, 1, 1]:
            continue
        out[pt] = {'nq': nq, 'E_nq111_Ry': e1, 'E_nq_Ry': e}
    return out


def qmesh(d, per, prof):
    """EXX q-mesh check (gap map A8): PBE0 (SG15) E(P4r) - E(P0) per formula unit at q = 1x1x1 and at the q mesh of
    the runs (= the k mesh), and the P4r profile values shifted by the difference."""
    q = qmesh_runs(d)
    if not q or not per:
        return None
    kj = RY_KJ / per['n_fu']
    out = {'runs': q, 'shift_E_kJ_per_fu': {pt: round((r['E_nq_Ry'] - r['E_nq111_Ry']) * kj, 3) for pt, r in q.items()}}
    if 'P0' in q and 'P4r' in q and q['P0']['nq'] == q['P4r']['nq']:
        a = (q['P4r']['E_nq111_Ry'] - q['P0']['E_nq111_Ry']) * kj
        b = (q['P4r']['E_nq_Ry'] - q['P0']['E_nq_Ry']) * kj
        out.update({'nq': q['P0']['nq'], 'dE_P4r_minus_P0_pbe0_sg15_nq111': round(a, 3),
                    'dE_P4r_minus_P0_pbe0_sg15_nq': round(b, 3), 'shift_kJ_per_fu': round(b - a, 3)})
        if 'P4r' in prof:
            out['P4r_shifted'] = {lev: {'q111': prof['P4r'][lev], 'q': round(prof['P4r'][lev] + b - a, 3)}
                                  for lev in LEVELS[1:] if prof['P4r'].get(lev) is not None}
    out['note'] = ('PBE0 (SG15) total energies, q = 1x1x1 vs q = nq; the shift applies to the composite PBE0 profile and '
                   'to every PBE0 + pair-correction level (PBE does not depend on q); one q mesh only (= the k mesh)')
    return out


BUDGET_LABEL = 'estimate: shifts computed at different settings, combined additively'
RSS_GROUPS = {'anchor': ('anchor_PBE', 'anchor_PBE50'), 'basis': ('basis_QZVP', 'basis_TZVPD')}


def budget_entry(central, point, comps, notes=None, combined=None):
    """one quantity of the budget: comps {component: {alternative: signed shift from central} or None (not computed)};
    envelope = min / max of central, of central + each shift (one at a time) and of the combinations computed directly
    or summed; rss over the components of the largest |shift| of each (the PBE / PBE50 anchors and the QZVP / TZVPD
    bases each counted once, by the larger)."""
    if central is None:
        return None
    cp = {k: ({a: round(x, 3) for a, x in v.items() if x is not None} or None) if v else None for k, v in comps.items()}
    mag = {k: max(abs(x) for x in v.values()) for k, v in cp.items() if v}
    for g, ks in RSS_GROUPS.items():
        if any(k in mag for k in ks):
            mag[g] = max(mag.pop(k, 0.0) for k in ks)
    cb = {a: round(x, 3) for a, x in (combined or {}).items() if x is not None}
    vals = [central] + [central + x for v in cp.values() if v for x in v.values()] + list(cb.values())
    return {'central': round(central, 3), 'point': point, 'components': cp, 'none_reason': notes or None, 'combined': cb or None,
            'envelope': [round(min(vals), 3), round(max(vals), 3)], 'rss': round(math.sqrt(sum(x * x for x in mag.values())), 3)}


def budget(out, d):
    """uncertainty budget per crystal and quantity (gap map B4, 22 Oct 2026; PLAN_FINAL endpoint 2, optc/PLAN.md 19 Oct
    9:40 PM 'an uncertainty budget per term'): every check that exists, as a signed shift from the central value, None
    where not computed.  margin: PBE0+MP2 at P4r in the X-ray cell where margin() finds an other-side minimum at
    PBE0+MP2 (else the same budget under 'P4r', not a margin); R5_P4: PBE0+MP2 at P4; dE_env: PBE0; R4a_single / R4a_collective / R4_single: the REPEAT field and
    cluster rungs (the charge and shell components).  The shifts were computed at different settings (k at PBE in the
    X-ray cell, q at PBE0, cells relaxed at PBE, delta(T) in def2-SVP) and are combined additively: an estimate."""
    prof, ck, nu = out.get('profile') or {}, out.get('checks') or {}, out.get('nuclear') or {}
    pair = (load(d / 'mol' / 'pair.json') or {}).get('points') or {}
    qz, tz = ck.get('pair_def2-qzvp') or {}, ck.get('monomers_tzvpd') or {}
    pc, ec = ck.get('pair_cc') or {}, ck.get('embcorr_pair_repeat') or {}
    dt_ok = 'P0' in (pc.get('points_done') or ())
    vc, qm = out.get('vc') or {}, ck.get('qmesh') or {}
    res = {}
    none_pt = {'charges': 'the periodic profile and its pair correction hold no point charges (the in-field check is '
                          'REPEAT only)',
               'shell_site_vs_whole': 'not computed: the pair correction was not repeated on a larger QM region'}

    def at(pt, c, field):
        """the shifts of a PBE0+MP2 profile value c at pt common to margin and R5_P4"""
        p, q = prof.get(pt) or {}, pair.get(pt) or {}
        sfx, f = ('' if pt == 'P4' else f'_{pt}'), ec.get(field) or {}
        t = ((tz.get(pt) or {}).get('R1f_kJ')) or {}
        return {
            'level_spread': {lev: p[lev] - c for lev in ('PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ') if p.get(lev) is not None},
            'delta_T': {'[CCSD(T) - MP2] of the pair, def2-SVP': pc.get(f'delta_T_{pt}_kJ')} if dt_ok else None,
            'anchor_PBE': {'PBE + [MP2 - PBE] of the pair': None if None in (p.get('PBE'), q.get('dMP2_kJ'), q.get('dPBE_D3_kJ'))
                           else p['PBE'] + q['dMP2_kJ'] - q['dPBE_D3_kJ'] - c},
            'anchor_PBE50': {'PBE50 + [MP2 - PBE50] of the pair': None if None in (p.get('PBE50'), q.get('corr_MP2_pbe50_kJ'))
                             else p['PBE50'] + q['corr_MP2_pbe50_kJ'] - c},
            'in_field_vs_vacuum': {f'X-ray cell, {field} (REPEAT field)': None if None in (f.get('corr_MP2_kJ'), q.get('corr_MP2_kJ'))
                                   else f['corr_MP2_kJ'] - q['corr_MP2_kJ']},
            'basis_QZVP': {'[MP2 - PBE0] of the pair, def2-QZVP': None if None in (qz.get(f'dMP2{sfx}_kJ'), qz.get(f'dPBE0_D3{sfx}_kJ'), q.get('corr_MP2_kJ'))
                           else qz[f'dMP2{sfx}_kJ'] - qz[f'dPBE0_D3{sfx}_kJ'] - q['corr_MP2_kJ']},
            'basis_TZVPD': {'[MP2 - PBE0] of the pair, def2-TZVPD': None if None in (t.get('MP2'), t.get('PBE0_D3'), q.get('corr_MP2_kJ'))
                            else t['MP2'] - t['PBE0_D3'] - q['corr_MP2_kJ']},
            'charges': None, 'shell_site_vs_whole': None}
    # -- margin: the other-side state P4r (where every check of item D/E/J and the q mesh was run)
    c = (prof.get('P4r') or {}).get('PBE0+MP2')
    if c is not None:
        b, vf = 'dE_relaxed_cells_pbe0', ck.get('vcfree') or {}
        cp = at('P4r', c, 'NR')
        cp['in_field_vs_vacuum']['relaxed cells, (VR - VS) - (vcP4r - vcS)'] = term(vf.get(f'{b}_MP2_infield'), vf.get(f'{b}_MP2'))
        cp['cell'] = {'relaxed cells (vcfree), vacuum pair correction': term(vf.get(f'{b}_MP2'), c)}
        cp['k_mesh'] = {'PBE, dense - production k at P4r': out.get('kcheck_shift_P4r')}
        cp['q_mesh'] = {'PBE0, EXX q = k mesh - q = 1x1x1': qm.get('shift_kJ_per_fu')}
        zf = ((nu.get('zpe_frozen_frame') or {}).get('H') or {}).get('PBE0+MP2') or {}
        hz = (nu.get('dZPE_P4r_minus_P0') or {}).get('H') or {}
        # a frozen frame with a delocalized ground state gives a bound, not a shift: kept apart from rss and envelope
        zloc = (zf.get('kind') or '').startswith('ZPE')
        cp['ZPE'] = {'frozen frame, H': zf.get('dZPE_kJ') if zloc else None, 'harmonic partial Hessian, H D A (PBE)': hz.get('zpe_HDA_kJmol'),
                     'harmonic partial Hessian, H only (PBE)': hz.get('zpe_H_only_kJmol')}
        mg = (out.get('margin_other_side') or {}).get('PBE0+MP2') or {}
        lo, ob = mg.get('lowest_sampled_other_side') or {}, mg.get('obs_side_min_below_P0') or {}
        # other choices of the reference points: a definition, not an uncertainty (gap map A1 keeps P0); outside rss
        alt = {f"lowest sampled other-side point ({lo.get('point')})": None if lo.get('point') in (None, 'P4r') else round(lo['E'] - c, 3),
               f"observed-side point below P0 ({ob.get('point')})": round(-ob['E'], 3) if ob else None}
        notes = dict(none_pt, ZPE=f"frozen frame: {zf['kind']}" if zf.get('kind') else 'not computed')
        dt = pc.get('delta_T_P4r_kJ') if dt_ok else None
        add = [(n, x) for n, x in (('delta(T)', dt), ('k', out.get('kcheck_shift_P4r')), ('q', qm.get('shift_kJ_per_fu'))) if x is not None]
        cb = {'relaxed cells, in-field': vf.get(f'{b}_MP2_infield')}
        for w, kk in (('vacuum', f'{b}_MP2'), ('in-field', f'{b}_MP2_infield')):
            if vf.get(kk) is not None and add:
                cb[f"relaxed cells, {w} + {' + '.join(n for n, _ in add)}"] = vf[kk] + sum(x for _, x in add)
        # 'margin' only where margin() finds an other-side minimum at PBE0+MP2; else the same budget of P4r - P0 under 'P4r'
        mk = 'margin' if mg.get('margin') is not None else 'P4r'
        res[mk] = q = budget_entry(c, 'P4r', cp, notes, cb)
        q['quantity'] = ('margin (P4r, other-side minimum at PBE0+MP2)' if mk == 'margin' else
                         'P4r - P0 (no other-side minimum at PBE0+MP2: not a margin; margin(), gap map A1)')
        q['margin_other_side'] = {k: mg.get(k) for k in ('margin', 'point', 'no_other_side_minimum', 'crossing_energy_delta0')}
        q['definition_alternatives'] = {k: x for k, x in alt.items() if x is not None} or None
        q['bounds'] = None if zloc or zf.get('dZPE_kJ') is None else {'frozen frame, H (dZPE)': zf['dZPE_kJ'], 'kind': zf['kind']}
        q['relaxed_cells'] = {lev: vf.get(f'{b}{s}') for lev, s in (('PBE0', ''), ('PBE0+MP2', '_MP2'),
                              ('PBE0+SCS_MP2', '_SCS_MP2'), ('PBE0+wB97X_D3BJ', '_wB97X_D3BJ'))} if vf.get(b) is not None else None
    # -- R5 at P4 (the pre-registered end point; the only one for the salts, which have no other-side minimum)
    p4 = prof.get('P4') or {}
    c = p4.get('PBE0+MP2')
    if c is not None:
        cp = at('P4', c, 'N')
        cp.update({'cell': {'vc relax at P4 (PBE, 80 Ry)': vc.get('shift_kJmol_fu')}, 'k_mesh': {'PBE, dense - production k at P4': out.get('kcheck_shift')},
                   'q_mesh': None, 'ZPE': None})
        res['R5_P4'] = budget_entry(c, 'P4', cp, dict(none_pt, q_mesh='not computed at P4', ZPE='not computed at P4 (no Hessian)'))
    # -- dE_env = R5 - R1f at PBE0
    de, r1 = out.get('dE_env') or {}, (out.get('R1f') or {}).get('PBE0')
    c = de.get('PBE0')
    if c is not None:
        q4, t4 = pair.get('P4') or {}, ((tz.get('P4') or {}).get('R1f_kJ')) or {}
        cp = {'level_spread': None, 'delta_T': None, 'in_field_vs_vacuum': None, 'q_mesh': None, 'ZPE': None, 'charges': None,
              'shell_site_vs_whole': None,
              'anchor_PBE': {'R5 - R1f at PBE': term(de.get('PBE'), c)},
              'anchor_PBE50': {'R5 - R1f at PBE50': None if None in (p4.get('PBE50'), q4.get('dPBE50_D3_kJ')) else p4['PBE50'] - q4['dPBE50_D3_kJ'] - c},
              'cell': {'vc relax at P4 (PBE, R5 only)': vc.get('shift_kJmol_fu')},
              'k_mesh': {'PBE, dense - production k at P4 (R5 only)': out.get('kcheck_shift')},
              'basis_QZVP': {'R1f in def2-QZVP (PBE0)': None if None in (qz.get('dPBE0_D3_kJ'), r1) else r1 - qz['dPBE0_D3_kJ']},
              'basis_TZVPD': {'R1f in def2-TZVPD (PBE0)': None if None in (t4.get('PBE0_D3'), r1) else r1 - t4['PBE0_D3']}}
        same = 'the same pair correction enters R5 and R1f and cancels'
        res['dE_env'] = budget_entry(c, 'P4', cp, {'level_spread': same, 'delta_T': same, 'in_field_vs_vacuum': 'R1f is the pair in vacuum by definition',
                                                   'q_mesh': 'not computed at P4', 'ZPE': 'not computed at P4', 'charges': 'no point charges in R5 or R1f',
                                                   'shell_site_vs_whole': 'no QM cluster in R5 or R1f'})
    # -- the field and cluster rungs (REPEAT): charge models, smearing and cutoff, collective estimator, shell size
    L = out.get('ladder') or {}
    lr, sc, gs = (L.get(m) or {} for m in ('repeat', 'sc', 'gas'))
    st = ck.get('sitesym_repeat_terms') or {}
    for qn in ('R4a_single', 'R4a_collective', 'R4_single'):
        c = lr.get(qn)
        if c is None:
            continue
        alt = 'R4a_single' if qn == 'R4_single' and lr.get('site_shell_empty') else qn     # R4 = R4a by definition
        cp = {'charges': {'sc': term(sc.get(alt), c), 'gas': term(gs.get(alt), c)}}
        if qn == 'R4a_single':
            cp['smearing'] = {'0.5 A': term(ck.get('R4a_repeat_smear05'), c), '1.2 A': term(ck.get('R4a_repeat_smear12'), c)}
            cp['cutoff'] = {'35 A': term(ck.get('R4a_repeat_rc35'), c)}
        if qn == 'R4a_collective':
            cp['collective_estimator'] = {'LR mean': term(lr.get('R4a_collective_lr'), c)}
        if qn == 'R4_single':
            cp['shell_site_vs_whole'] = {'whole first shell (corrected)': term(lr.get('R4_full_shell'), c),
                                         'symmetric site (item M)': term(st.get('R4_single'), c) if st and not st.get('empty_site_shell') else None}
        res[qn] = budget_entry(c, 'P4 (REPEAT field)', cp)
    if res:
        res['label'] = BUDGET_LABEL
    return res


def resolution(out):
    """gap map B5: is the sign of the corrected margin outside its budget?  The margin (P4r) where margin() finds an
    other-side state at PBE0+MP2, else R5 at P4; 'unresolved' when the envelope spans 0 (the rss test is recorded
    beside it).  The pre-registered class is not changed."""
    bd = out.get('budget') or {}
    q, name = (bd.get('margin'), 'margin') if bd.get('margin') else (bd.get('R5_P4'), 'R5_P4')
    if not q:
        return None
    lo, hi = q['envelope']
    sp = lo <= 0 <= hi
    return {'quantity': name, 'point': q['point'], 'central': q['central'], 'envelope': q['envelope'], 'rss': q['rss'],
            'envelope_spans_0': sp, 'rss_ge_central': q['rss'] >= abs(q['central']),
            'central_relaxed_cells': (q.get('relaxed_cells') or {}).get('PBE0+MP2'),
            'verdict': f'unresolved (budget envelope of {name} spans 0)' if sp else
            ('resolved: observed state lower' if q['central'] > 0 else 'resolved: other state lower'), 'label': BUDGET_LABEL}


def extend(out, d):
    key = out['key']
    sgn = 1.0 if out['observed'] == 'cocrystal' else -1.0
    # -- fixed direction E(ion pair) - E(neutral)
    lr = out.get('ladder', {}).get('repeat', {})
    fx = {}
    for name, v in (('R1f', (out.get('R1f') or {}).get('PBE0')), ('R4a_single', lr.get('R4a_single')),
                    ('R4a_collective', lr.get('R4a_collective')), ('R3', lr.get('R3')), ('R4_single', lr.get('R4_single')),
                    ('R5_PBE0', (out.get('R5') or {}).get('PBE0')), ('R5_PBE0+MP2', (out.get('R5') or {}).get('PBE0+MP2')),
                    ('dE_env_PBE0', (out.get('dE_env') or {}).get('PBE0'))):
        fx[name] = None if v is None else round(sgn * v, 3)
    out['fixed_ion_minus_neutral'] = fx
    # -- disjoint decomposition per charge model
    dec = {}
    for model in ('repeat', 'sc'):
        m = out.get('ladder', {}).get(model, {})
        R1f, R4a, R3, R4 = m.get('R1f'), m.get('R4a_single'), m.get('R3'), m.get('R4_single')
        if model == 'sc':
            R3 = R3 if R3 is not None else lr.get('R3')
        po = load(d / 'mol' / f'emb_pairout_{model}_site.json')
        pq = load(d / 'mol' / f'emb_pairshellq_{model}_site.json')
        empty = m.get('site_shell_empty') or (po or {}).get('skipped')
        R4ao = R4a if empty else ((po or {}).get('derived') or {}).get('embedded_single_site_kJ')
        R4aq = R1f if empty else ((pq or {}).get('derived') or {}).get('embedded_single_site_kJ')
        dec[model] = {'shell': term(R3, R1f), 'outer': term(R4, R3), 'electrostatic': term(R4a, R1f),
                      'beyond_electrostatic': term(R4, R4a), 'R4a_outer': R4ao, 'outer_rev': term(R4ao, R1f),
                      'shell_rev': term(R4, R4ao), 'cross': None if None in (R4, R4ao, R3, R1f) else round(R4 - R4ao - R3 + R1f, 3),
                      'R4a_shellq': R4aq, 'shell_charges': term(R4aq, R1f), 'empty_site_shell': bool(empty)}
    if dec.get('repeat') and dec.get('sc'):
        dec['robust'] = {t: robust(dec['repeat'][t], dec['sc'][t]) for t in ('shell', 'outer', 'outer_rev', 'shell_rev',
                                                                             'electrostatic', 'beyond_electrostatic')}
    if dec:
        dec['status'] = 'exploratory'
        dec['status_note'] = ('added 19 Oct 2026, 9:40 PM IST (optc/PLAN.md item A), after the pre-registered field/shell '
                              'split (ladder field, shell, nonadditivity, split_presentable) was found not presentable')
    out['decomposition'] = dec
    # -- reverse gaps
    rev = {}
    for model in ('repeat', 'sc', 'gas'):
        dd = (load(d / 'mol' / f'emb_pair_{model}.json') or {}).get('derived') or {}
        if 'reverse_kJ' in dd:
            rev[f'pair_{model}'] = {k: dd.get(k) for k in ('embedded_single_site_kJ', 'reverse_kJ', 'lr_mean_kJ', 'lambda_kJ')}
    dd = (load(d / 'mol' / 'emb_cluster_repeat_site.json') or {}).get('derived') or {}
    if 'reverse_kJ' in dd or 'reverse_vacuum_kJ' in dd:
        rev['site_repeat'] = {k: dd.get(k) for k in ('vacuum_kJ', 'reverse_vacuum_kJ', 'lr_mean_vacuum_kJ', 'lambda_vacuum_kJ',
                                                     'embedded_single_site_kJ', 'reverse_kJ', 'lr_mean_kJ', 'lambda_kJ')}
    out['reverse'] = rev or None
    # -- periodic closure
    vj = load(d / 'vert.json')
    if vj and vj.get('derived'):
        v = vj['derived']
        cl = dict(v)
        R5 = out.get('R5') or {}
        cl['reorganization_pbe0'] = term(R5.get('PBE0'), v.get('Nv_pbe0_kJ_per_pair'))
        cl['reorganization_pbe'] = term(R5.get('PBE'), v.get('Nv_pbe_kJ_per_pair'))
        cl['cluster_minus_periodic_single_site'] = term(lr.get('R4_single'), v.get('T1_pbe0_kJ'))
        cl['field_minus_periodic_single_site'] = term(lr.get('R4a_single'), v.get('T1_pbe0_kJ'))
        # gap map A7: T1 transfers one pair of the cell's Z (f = 1/Z); T2 (vert_T2s, doubled cell) was run at PBE/SSSP
        # only, so its PBE0 value is estimated with T1's hybrid shift (a true PBE0 T2 was not computed)
        cl['cluster_minus_periodic_single_site_f1overZ'] = cl['cluster_minus_periodic_single_site']
        cl['field_minus_periodic_single_site_f1overZ'] = cl['field_minus_periodic_single_site']
        t2 = None if None in (v.get('T2_pbe_kJ'), v.get('T1_pbe0_kJ'), v.get('T1_pbe_kJ')) else \
            round(v['T2_pbe_kJ'] + v['T1_pbe0_kJ'] - v['T1_pbe_kJ'], 3)
        cl['T2_pbe0_est_kJ'] = t2
        cl['cluster_minus_periodic_single_site_T2'] = term(lr.get('R4_single'), t2)
        cl['field_minus_periodic_single_site_T2'] = term(lr.get('R4a_single'), t2)
        a, b = cl['cluster_minus_periodic_single_site'], cl['cluster_minus_periodic_single_site_T2']
        cl['cluster_minus_periodic_single_site_sign'] = None if None in (a, b) else (
            'sign undetermined (R4 - T1 and R4 - T2 est. differ in sign)' if a * b < 0 else 'same sign with T1 and T2 est.')
        cl['labels'] = {'T1': 'periodic single site, one pair of Z moved (f = 1/Z); *_f1overZ = the unsuffixed keys',
                        'T2_pbe0_est_kJ': 'T2(PBE) + [T1(PBE0) - T1(PBE)]: an estimate, a true PBE0 T2 was not computed',
                        '*_T2': 'R4 (cluster) or R4a (field) minus T2_pbe0_est_kJ'}
        # gap map B1: the cooperativity as pre-registered (PLAN 9:40 PM item C: N_vert - single site, the single site
        # being T1, whose pair is the relaxed P4 pair) and as vert.json derives it (N_vert - T1v, both vertical: H moved
        # only); they differ by the relaxation of the one pair, T1 - T1v (T1v at PBE only)
        t1_0, t1_p, t1v_p, nv_0, nv_p = (v.get(k) for k in ('T1_pbe0_kJ', 'T1_pbe_kJ', 'T1v_pbe_kJ', 'Nv_pbe0_kJ_per_pair',
                                                            'Nv_pbe_kJ_per_pair'))
        cl['pair_relaxation_single_site_pbe'] = term(t1_p, t1v_p)
        cl['cooperativity_preregistered_pbe0'] = term(nv_0, t1_0)
        cl['cooperativity_preregistered_pbe'] = term(nv_p, t1_p)
        cl['cooperativity_vertical_f1overZ_to_1_pbe'] = v.get('cooperativity_vertical_pbe_kJ')
        R1f = out.get('R1f') or {}

        def cycle(r1f, r4, t1, nv, r5):
            st = {'R1f': r1f, 'R4_minus_R1f': term(r4, r1f), 'T1_minus_R4': term(t1, r4), 'Nv_minus_T1': term(nv, t1),
                  'R5_minus_Nv': term(r5, nv)}
            if r4 is None:
                st.pop('R4_minus_R1f'), st.pop('T1_minus_R4')
                st['T1_minus_R1f'] = term(t1, r1f)
            sm = None if None in st.values() else round(sum(st.values()), 3)
            return dict(st, sum=sm, R5=r5, sum_equals_R5=None if None in (sm, r5) else abs(sm - r5) < 0.01)
        cl['cycle_pbe0'] = cycle(R1f.get('PBE0'), lr.get('R4_single'), t1_0, nv_0, R5.get('PBE0'))
        cl['cycle_pbe'] = dict(cycle(R1f.get('PBE'), None, t1_p, nv_p, R5.get('PBE')), Nv_minus_T1v=v.get('cooperativity_vertical_pbe_kJ'),
                               T1_minus_T1v=cl['pair_relaxation_single_site_pbe'],
                               note='no cluster rung at PBE (R4 is PBE0); Nv - T1 = (Nv - T1v) - (T1 - T1v)')
        cl['labels'].update({
            'cooperativity_preregistered_*': 'as defined before its results (optc/PLAN.md 19 Oct 2026 9:40 PM item C, an addition to the 18 Oct pre-registration): N_vert - T1, T1 moves 1/Z of the pairs (its pair at the relaxed P4 '
                                             'geometry, N_vert moves only the H of every pair)',
            'cooperativity_vertical_f1overZ_to_1_pbe': 'N_vert - T1v (= cooperativity_vertical_pbe_kJ of vert.json): both '
                                                       'vertical (H moved only), f = 1/Z to 1',
            'pair_relaxation_single_site_pbe': 'T1 - T1v: relaxation of the one transferred pair (PBE only)',
            'cycle_*': 'R1f -> R4 -> T1 -> N_vert -> R5 (PLAN item C), each step a difference; sum == R5 by construction',
            'note': 'neither cooperativity variant is a like-for-like test of the E_int/2 term (ladder cooperativity '
                    'R4a_collective - R4a_single), which is computed in point charges at the P4 geometry'})
        out['closure'] = cl
    # -- checks
    ck = {}
    fl = load(d / 'mol' / 'emb_cluster_repeat_ldb.json')
    if fl and not fl.get('skipped') and fl.get('derived'):
        ck['whole_shell_raw'] = {'R3': fl['derived'].get('vacuum_kJ'), 'R4': fl['derived'].get('embedded_single_site_kJ')}
        ck['whole_shell_corrected'] = {'R3': lr.get('R3_full_shell'), 'R4': lr.get('R4_full_shell')}
    elif fl and fl.get('skipped'):
        ck['whole_shell_raw'] = fl['skipped']
    pc = {}
    for rs in ('1.1', '1.2', '1.3'):
        pj = load(d / 'mol' / f'pcm_r{rs}.json')
        if pj and pj.get('dE_kJ'):
            pc[rs] = {'eps1': pj['dE_kJ'].get('1'), 'eps78': pj['dE_kJ'].get('78.36')}
    ck['pcm_radii'] = pc or None
    for b in ('def2-qzvp',):
        bj = load(d / 'mol' / f'pair_{b}.json')
        if bj:
            ck[f'pair_{b}'] = basis_check(bj)
    mo = load(d / 'mol' / 'monomers_def2-tzvpd.json')
    if mo and mo.get('derived'):
        ck['monomers_tzvpd'] = mo['derived']
    for tag in ('repeat_rc35', 'repeat_smear05', 'repeat_smear12'):
        ej = load(d / 'mol' / f'emb_pair_{tag}.json')
        if ej and ej.get('derived'):
            ck[f'R4a_{tag}'] = ej['derived'].get('embedded_single_site_kJ')
    rj = load(d / 'mol' / 'pair_runner.json')
    if rj and rj.get('runner_minus_chosen_kJ'):
        bad = runner_invalid(key, d, rj)
        ck['runner_up_minus_chosen_vertical'] = {'invalid': bad} if bad else rj['runner_minus_chosen_kJ']
    # the same bond choice in the crystal (qe.py runner, 21 Oct 2026): vertical, runner minus chosen per pair
    rp = load(d / 'runner_periodic.json')
    if rp and rp.get('derived'):
        ck['runner_up_minus_chosen_periodic'] = {k: v for k, v in rp['derived'].items() if k.startswith('runner_minus_chosen')}
    sb = load(d / 'mol' / 'emb_cluster_repeat_sub4.json'), load(d / 'mol' / 'emb_cluster_repeat_sub4_ldb.json')
    if all(x and x.get('derived') for x in sb):
        ck['basis_reduced_cluster'] = {'tzvp': sb[0]['derived'], 'ldb': sb[1]['derived']}
    # delta(T) of the pair (mol.py cc): [CCSD(T) - MP2] at a point minus at P0, small basis, frozen core
    cc = load(d / 'mol' / 'pair_cc.json')
    if cc:
        ck['pair_cc'] = dict({k: v for k, v in cc.items() if k.startswith(('delta_T', 'dMP2', 'dCCSD'))}, basis=cc.get('basis'),
                             points_done=sorted(p for p, r in (cc.get('points') or {}).items() if r.get('done')))
    # pair corrections in the field (mol.py embcorr): [level - PBE0] per configuration, and the symmetric site (item M)
    ecj = load(d / 'mol' / 'embcorr_pair_repeat.json')
    if ecj and ecj.get('derived'):
        ck['embcorr_pair_repeat'] = ecj['derived']
    ss = {w: (load(d / 'mol' / f'emb_{w}_repeat_sitesym.json') or {}) for w in ('cluster', 'pairout', 'pairshellq')}
    ss = {w: x.get('derived') or ({'skipped': x['skipped']} if x.get('skipped') else None) for w, x in ss.items()}
    if any(ss.values()):
        ck['sitesym_repeat'] = ss
    # the symmetric-site terms, as the decomposition (an empty symmetric shell: R4a_outer = R4a, R4a_shellq = R1f)
    sc = ss.get('cluster') or {}
    if sc.get('vacuum_kJ') is not None:
        sm = load(d / 'emb' / 'repeat_sitesym' / 'emb.json') or {}
        R1f, R4a, R3s, R4s = lr.get('R1f'), lr.get('R4a_single'), sc.get('vacuum_kJ'), sc.get('embedded_single_site_kJ')
        po, pq = ss.get('pairout') or {}, ss.get('pairshellq') or {}
        empty = sm.get('n_shell') == 0 or 'skipped' in po
        R4ao = R4a if empty else po.get('embedded_single_site_kJ')
        R4aq = R1f if empty else pq.get('embedded_single_site_kJ')
        ck['sitesym_repeat_terms'] = {
            'n_shell': sm.get('n_shell'), 'empty_site_shell': bool(empty), 'R3': R3s, 'R4_single': R4s, 'R4a_outer': R4ao,
            'R4a_shellq': R4aq, 'shell': term(R3s, R1f), 'outer': term(R4s, R3s), 'outer_rev': term(R4ao, R1f),
            'shell_rev': term(R4s, R4ao), 'cross': None if None in (R4s, R4ao, R3s, R1f) else round(R4s - R4ao - R3s + R1f, 3),
            'shell_charges': term(R4aq, R1f), 'electrostatic': term(R4a, R1f), 'beyond_electrostatic': term(R4s, R4a)}
    pj = load(d / 'periodic.json') or {}
    ck['kcheck'] = pj.get('kcheck')
    ck['qmesh'] = qmesh(d, pj, out.get('profile') or {})
    ck['vcfree'] = load(d / 'vcfree.json')
    # relaxed cells + the pair correction at the two relaxed-cell geometries (as on the profile: per-pair correction
    # added to the per-formula-unit PBE0 difference)
    vf = ck['vcfree'] or {}
    pair = load(d / 'mol' / 'pair.json')
    if vf.get('dE_relaxed_cells_pbe0') is not None and pair and all(p in pair['points'] for p in ('vcS', 'vcP4r')):
        for lev in ('MP2', 'SCS_MP2', 'wB97X_D3BJ'):
            cs, cp = (pair['points'][p].get(f'corr_{lev}_kJ') for p in ('vcS', 'vcP4r'))
            if cs is not None and cp is not None:
                vf[f'dE_relaxed_cells_pbe0_{lev}'] = round(vf['dE_relaxed_cells_pbe0'] + cp - cs, 3)
    # gap map A9: the same with the in-field correction difference (embcorr VR - VS), + delta(T) and + the q shift,
    # both taken from the X-ray-cell P0/P4r pair (assumed transferable to the relaxed cells)
    if vf.get('dE_relaxed_cells_pbe0') is not None:
        vrs = (ck.get('embcorr_pair_repeat') or {}).get('VR_minus_VS') or {}
        for lev in ('MP2', 'SCS_MP2', 'wB97X_D3BJ'):
            if vrs.get(f'corr_{lev}_kJ') is not None:
                vf[f'dE_relaxed_cells_pbe0_{lev}_infield'] = round(vf['dE_relaxed_cells_pbe0'] + vrs[f'corr_{lev}_kJ'], 3)
        pc = ck.get('pair_cc') or {}
        if {'P0', 'P4r'} <= set(pc.get('points_done') or ()) and pc.get('delta_T_P4r_kJ') is not None:
            for k in ('MP2', 'MP2_infield'):
                if vf.get(f'dE_relaxed_cells_pbe0_{k}') is not None:
                    vf[f'dE_relaxed_cells_pbe0_{k.replace("MP2", "MP2_dT")}'] = round(vf[f'dE_relaxed_cells_pbe0_{k}'] + pc['delta_T_P4r_kJ'], 3)
        qs = (ck.get('qmesh') or {}).get('shift_kJ_per_fu')
        if qs is not None:
            for k in [k for k in vf if k.startswith('dE_relaxed_cells_pbe0')]:
                vf[f'{k}_q'] = round(vf[k] + qs, 3)
        vf['labels_added'] = {'_<lev>': 'vacuum pair correction at vcP4r minus at vcS (mol/pair.json)',
                              '_infield': 'in-field pair correction difference (embcorr_pair_repeat VR_minus_VS)',
                              '_MP2_dT': '+ delta(T) of the X-ray-cell P0/P4r pair (mol/pair_cc.json)',
                              '_q': '+ the q-mesh shift of the X-ray-cell P0/P4r pair (checks.qmesh)'}
    out['checks'] = ck
    # -- nuclear: Hessian ZPE and 1D DVR on the corrected profile
    nu = {}
    hz = {pt: load(d / f'hess_{pt}.json') for pt in ('P0', 'P4r')}
    if all(hz.values()):
        nu['dZPE_P4r_minus_P0'] = {iso: {k: round(hz['P4r'][iso][k] - hz['P0'][iso][k], 3)
                                         for k in ('zpe_HDA_kJmol', 'zpe_H_only_kJmol')} for iso in ('H', 'D')}
        nu['wavenumbers_H_only_cm1'] = {pt: hz[pt]['H']['wavenumbers_H_only_cm1'] for pt in hz}
    # frozen heavy-atom frames (optc/PLAN.md, 20 Oct 2026 entry): the periodic PBE proton scan + the pair correction
    # [E(level) - E(PBE)] at each offset; DVR per level; E0 measured from the frame's own geometry (offset 0), which is
    # the scan's minimum at PBE only: at the corrected levels E0_above_ref = E0_above_min - (V(0) - Vmin), the zero
    # point less the potential relaxation along the scan.  Bends: the H-only Hessian block (PBE) projected
    # perpendicular to D->A (bends_perp, gap map A4; *_legacy: the two lowest H-only modes, which at nbaetpy P4r
    # include the stretch).  A frame whose ground state lies above its dividing point (E0_minus_divide >= 0) has no
    # localized level on its own side (gap map A5): the E0 difference is then a bound, not a margin.
    prof = out.get('profile') or {}
    LEVS = {'PBE': None, 'PBE0': 'PBE0_D3', 'PBE0+MP2': 'MP2', 'PBE0+SCS_MP2': 'SCS_MP2', 'PBE0+wB97X_D3BJ': 'wB97X_D3BJ'}
    ff = {}
    for pt in ('P0', 'P4r'):
        h, hp = load(d / f'hscan_{pt}.json'), load(d / 'mol' / f'hscan_pair_{pt}.json')
        if not h or len(h.get('dE_kJ_per_pair', [])) < 7:
            continue
        xs, v0 = h['offsets_A'], dict(zip(h['offsets_A'], h['dE_kJ_per_pair']))
        # true delta of each scan point: the frame's own delta + 2 x offset (the H moves along D...A, D and A fixed)
        dref = (prof.get(pt) or {}).get('delta')
        if dref is None:
            continue
        ff[pt] = {}
        for lev, key in LEVS.items():
            if key is None:
                V = [v0[x] for x in xs]
            elif hp and hp.get('corr_vs_pbe_kJ', {}).get(key) and len(hp['offsets_A']) == len(xs):
                cmap = dict(zip(hp['offsets_A'], hp['corr_vs_pbe_kJ'][key]))
                if not all(round(x, 2) in {round(y, 2) for y in cmap} for x in xs):
                    continue
                cm = {round(y, 2): c for y, c in cmap.items()}
                V = [v0[x] + cm[round(x, 2)] for x in xs]
            else:
                continue
            ff[pt][lev] = {iso: dvr_1d([dref + 2 * x for x in xs], V, m, EXP_T.get(out['key']), ref=dref)
                           for iso, m in (('H', MASS_H), ('D', MASS_D))}
    if ff:
        nu['frozen_frame_stretch'] = ff
    if 'P0' in ff and 'P4r' in ff and all(hz.values()) and 'P4r' in prof:
        zc = {}
        # no <pt>.extxyz (no lattice for u): the legacy bends, labelled
        bp = {pt: {iso: bends_perp(d, pt, hz[pt], m) for iso, m in (('H', MASS_H), ('D', MASS_D))}
              if (d / f'{pt}.extxyz').exists() else None for pt in hz}
        nu['bends_perp'] = {pt: v or f'missing {pt}.extxyz: legacy bends (two lowest H-only modes)' for pt, v in bp.items()}
        KIND = {(True, True): 'ZPE-corrected margin',
                (False, True): 'P4r-frame E0 difference, lower bound (no localized other-state level)',
                (True, False): 'P4r-frame E0 difference, upper bound (no localized observed-state level in the P0 frame)',
                (False, False): 'P4r-frame E0 difference, not bounded (no localized level in either frame)'}
        for iso in ('H', 'D'):
            def bends(pt):
                w = sorted(x for x in hz[pt][iso]['wavenumbers_H_only_cm1'] if x > 0)[:2]
                return 0.5 * 0.01196266 * sum(w)
            zc[iso] = {}
            for lev in LEVS:
                if lev in ff['P0'] and lev in ff['P4r'] and prof['P4r'].get(lev) is not None:
                    f0, f4 = ff['P0'][lev][iso], ff['P4r'][lev][iso]
                    dzl = (f4['E0_above_ref_kJ'] + bends('P4r')) - (f0['E0_above_ref_kJ'] + bends('P0'))
                    zb = lambda pt: bp[pt][iso]['zpe_bend_kJ'] if bp[pt] else bends(pt)
                    dz = (f4['E0_above_ref_kJ'] + zb('P4r')) - (f0['E0_above_ref_kJ'] + zb('P0'))
                    loc = (f4['E0_minus_divide_kJ'] < 0, f0['E0_minus_divide_kJ'] < 0)
                    val = round(prof['P4r'][lev] + dz, 3)
                    zc[iso][lev] = {'dZPE_kJ': round(dz, 3), 'margin_P4r_zpe_corrected': val if all(loc) else None,
                                    'E0_difference_kJ': val, 'kind': KIND[loc],
                                    'P4r_frame_localized': loc[0], 'P0_frame_localized': loc[1],
                                    'p_far_side_P0_frame': f0['p_delta_above_divide'],
                                    'dZPE_kJ_legacy': round(dzl, 3), 'margin_P4r_zpe_corrected_legacy': round(prof['P4r'][lev] + dzl, 3)}
        nu['zpe_frozen_frame'] = zc
    # adiabatic DVR on the relaxed profile, E0 measured from P0 (gap map A6)
    for lev in ('PBE0+MP2', 'PBE0'):
        pts = lowest_per_target(prof, lev)
        if len(pts) >= 5 and 'P0' in prof:
            dl, e, _ = zip(*pts)
            nu[f'dvr_{lev}'] = {iso: dvr_1d(dl, e, m, EXP_T.get(key), ref=prof['P0']['delta']) for iso, m in (('H', MASS_H), ('D', MASS_D))}
    # the p(other) H column of summary.md: the P0 frozen frame where there is one; the legacy column was the adiabatic DVR
    pf = (((nu.get('zpe_frozen_frame') or {}).get('H') or {}).get('PBE0+MP2') or {}).get('p_far_side_P0_frame')
    pa = ((nu.get('dvr_PBE0+MP2') or {}).get('H') or {}).get('p_delta_above_divide')
    if pf is not None or pa is not None:
        src = 'dvr_PBE0+MP2.H.p_delta_above_divide (relaxed profile, adiabatic)'
        nu['p_other_H_printed'] = {'value': pf, 'source': 'zpe_frozen_frame.H.PBE0+MP2.p_far_side_P0_frame'} \
            if pf is not None else {'value': pa, 'source': src}
        nu['p_other_H_printed_legacy'] = {'value': pa, 'source': src}
    out['nuclear'] = nu or None
    # -- class without P4r and the extra points (the pre-registered profile)
    if prof:
        base = {k: v for k, v in prof.items() if k in ('P0', 'P1', 'P2', 'P3', 'P4')}
        cl = {}
        for lev in ('PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ'):
            if len(base) == 5 and all(base[p].get(lev) is not None for p in base):
                cl[lev] = classify([base[p]['delta'] for p in base], [base[p][lev] for p in base])
        out['class_preregistered_profile'] = cl
        # legacy margin (to 22 Oct 2026): the lowest energy at delta > 0, window points included, PBE50 on its few points
        mg = {}
        for lev in ('PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ', 'PBE50'):
            vals = [(p['delta'], p.get(lev)) for p in prof.values() if p.get(lev) is not None and p.get('delta') is not None]
            other = [e for dl, e in vals if dl > 0]       # delta > 0 is the other state for salts and cocrystals alike
            if other:
                mg[lev] = round(min(other), 3)
        out['margin_other_side_legacy'] = mg
        # margin (gap map A1): defined only by an other-side minimum; PBE50 by the same rule only on a profile as complete
        # as PBE0's, otherwise point by point against PBE0 at the other-side points it has
        mg = {lev: margin(prof, lev) for lev in LEVELS}
        full = [p for p in prof if prof[p].get('PBE0') is not None]
        have = [p for p in full if prof[p].get('PBE50') is not None]
        if have and len(have) == len(full):
            mg['PBE50'] = margin(prof, 'PBE50')
        elif have:
            mg['PBE50'] = {'margin': None, 'not_like_for_like': f"PBE50 at {', '.join(have)} only (PBE0 at {len(full)} points)",
                           'pointwise_other_side': {p: {'PBE50': prof[p]['PBE50'], 'PBE0': prof[p]['PBE0'],
                                                        'PBE50_minus_PBE0': round(prof[p]['PBE50'] - prof[p]['PBE0'], 3)}
                                                    for p in have if prof[p]['delta'] > 0.1}}
        out['margin_other_side'] = {k: v for k, v in mg.items() if v is not None}
        out['margin_other_side_note'] = ('margin: lowest sampled point at delta > +0.1 A minus P0, only with an other-side '
                                         'minimum (P4r present, interior PCHIP minimum at delta > +0.1 A); else None and '
                                         'crossing_energy_delta0 = crossing energy at delta = 0 (not a state)')
    # -- uncertainty budget and the resolution of the corrected sign (gap map B4, B5)
    out['budget'] = budget(out, d) or None
    out['resolution'] = resolution(out)


def polymorph_ranking(R):
    """brsalt (salt) vs brcoc (cocrystal), same composition: E(brcoc) - E(brsalt) per formula unit at P0."""
    a, b = load(R / 'brsalt' / 'periodic.json'), load(R / 'brcoc' / 'periodic.json')
    if not a or not b:
        return None
    kj = RY_KJ
    pa, pb = a['points']['P0'], b['points']['P0']
    out = {'pbe_kJ_per_fu': round((pb['E_pbe_Ry'] / b['n_fu'] - pa['E_pbe_Ry'] / a['n_fu']) * kj, 3)}
    hyb = qs = dk = None
    if all(x.get(k) is not None for x in (pa, pb) for k in ('E_pbe_sg15_Ry', 'E_pbe0_sg15_Ry')):
        ca = (pa['E_pbe0_sg15_Ry'] - pa['E_pbe_sg15_Ry']) / a['n_fu']
        cb = (pb['E_pbe0_sg15_Ry'] - pb['E_pbe_sg15_Ry']) / b['n_fu']
        out['pbe0_kJ_per_fu'] = round(out['pbe_kJ_per_fu'] + (cb - ca) * kj, 3)
        hyb = (cb - ca) * kj
    ka, kb = (a.get('kcheck') or {}).get('E_dense_Ry') or {}, (b.get('kcheck') or {}).get('E_dense_Ry') or {}
    if 'P0' in ka and 'P0' in kb:
        dk = (kb['P0'] / b['n_fu'] - ka['P0'] / a['n_fu']) * kj
        out['pbe_dense_k_kJ_per_fu'] = round(dk, 3)
    else:
        out['pbe_dense_k_kJ_per_fu'] = None
        out['dense_k_missing'] = [k for k, x in (('brsalt', ka), ('brcoc', kb)) if 'P0' not in x]
    # EXX q mesh (gap map A8): PBE0 correction of each cell at its own q mesh (= its k mesh) instead of q = 1x1x1
    qa, qb = qmesh_runs(R / 'brsalt').get('P0'), qmesh_runs(R / 'brcoc').get('P0')
    if hyb is not None and qa and qb:
        qs = ((qb['E_nq_Ry'] - qb['E_nq111_Ry']) / b['n_fu'] - (qa['E_nq_Ry'] - qa['E_nq111_Ry']) / a['n_fu']) * kj
        out.update({'nq': {'brsalt': qa['nq'], 'brcoc': qb['nq']}, 'q_shift_kJ_per_fu': round(qs, 3),
                    'pbe0_qmesh_kJ_per_fu': round(out['pbe_kJ_per_fu'] + hyb + qs, 3)})
        if dk is not None:
            out['pbe0_dense_k_qmesh_kJ_per_fu'] = round(dk + hyb + qs, 3)
    out['note'] = ('fixed X-ray cells; pbe0: PBE0 correction with q = 1x1x1 in two different cells; pbe0_qmesh: at q = '
                   'each cell\'s k mesh; dense k at PBE only (pbe0_dense_k_qmesh adds the PBE0 and q terms to it); no '
                   'vibrational free energy; not done: the monomer (item S) correction' if qs is not None else
                   'fixed X-ray cells; PBE0 correction with q = 1x1x1 in two different cells; no vibrational free energy; '
                   'not done: the monomer (item S) correction')
    return out


def agrees(observed, value):
    """a two-configuration rung reproduces the observed state if the other state is higher (value > 0)."""
    return None if value is None else value > 0


def rung(r, k, model='repeat'):
    return r.get('ladder', {}).get(model, {}).get(k)


def class_ok(field, lev):
    """profile criterion: the class at `lev` ('class' with P4r and the extra points, or 'class_preregistered_profile')
    is 'observed'; a reason string where there is none."""
    def get(r):
        c = (r.get(field) or {}).get(lev)
        return c[0] == 'observed' if c else f'no {field} at {lev}'
    return get


def nontrivial(k, sym=False):
    """sign criterion of R3 / R4_single without the empty transfer-site shells (R3 = R1f, R4 = R4a by definition);
    sym: such a crystal counted with its symmetric site (item M) where that shell is not empty."""
    def get(r):
        if not rung(r, 'site_shell_empty'):
            return agrees(r['observed'], rung(r, k))
        st = (r.get('checks') or {}).get('sitesym_repeat_terms')
        if sym and st and not st['empty_site_shell']:
            return agrees(r['observed'], st.get(k))
        return 'site shell empty' + ('' if not sym else ', symmetric site empty too' if st else ', no symmetric-site run')
    return get


def tally(rows, get, absent=()):
    """k of n, overall and by class, with the crystals missed and those not counted (and why); get(r): True / False,
    or None / a reason string."""
    t = {'excluded': [[k, 'no results directory'] for k in absent]}
    v, why = {}, getattr(get, 'why', None)          # why(r): the label of a miss that is not a wrong class
    for r in rows:
        x = 'ranking only, no transferring bond' if r['key'] == 'brcoc' else get(r)
        if x is None or isinstance(x, str):
            t['excluded'].append([r['key'], x or 'not computed'])
        else:
            v[r['key']] = (bool(x), r['observed'], why(r) if why and not x else None)
    for cls in ('all', 'salt', 'cocrystal'):
        s = {key: (x, w) for key, (x, c, w) in v.items() if cls in ('all', c)}
        k, n = sum(x for x, _ in s.values()), len(s)
        t[cls] = {'reproduced': k, 'of': n, 'ci95': clopper_pearson(k, n) if n else None,
                  'missed': [key + (f' ({w})' if w else '') for key, (x, w) in s.items() if not x]}
    return t


SIGN = 'sign at P4 (P4 - P0 > 0)'


def count_rows():
    """rung x criterion x level (gap map A2, 22 Oct 2026): the molecular rungs by their two-configuration sign, R5 by
    the class of the full profile, the class of the pre-registered P0-P4 profile and the same sign at P4; each row
    with its status (pre-registered / extension of 19-20 Oct / added 22 Oct 2026)."""
    P, X, A = 'pre-registered', 'extension (P4r and the extra points, 19-20 Oct 2026)', 'added 22 Oct 2026, not pre-registered'
    rows = [('R1f', SIGN, 'PBE0', lambda r: agrees(r['observed'], (r.get('R1f') or {}).get('PBE0')), P),
            ('R4a single site (field, REPEAT)', SIGN, 'PBE0', lambda r: agrees(r['observed'], rung(r, 'R4a_single')), P),
            ('R4a collective (field, REPEAT, E_int/2)', SIGN, 'PBE0', lambda r: agrees(r['observed'], rung(r, 'R4a_collective')), P),
            ('R4a collective (field, REPEAT, LR mean)', SIGN, 'PBE0', lambda r: agrees(r['observed'], rung(r, 'R4a_collective_lr')), A)]
    # gap map B2: both collective estimators with the second and third charge models
    rows += [(f'R4a collective (field, {m}, {e})', SIGN, 'PBE0', lambda r, m=m, k=k: agrees(r['observed'], rung(r, k, m)), A)
             for m in ('sc', 'gas') for e, k in (('E_int/2', 'R4a_collective'), ('LR mean', 'R4a_collective_lr'))]
    rows += [('R3 (shell)', SIGN, 'PBE0', lambda r: agrees(r['observed'], rung(r, 'R3')), P),
             ('R3 (shell), non-trivial', SIGN, 'PBE0', nontrivial('R3'), A),
             ('R3 (shell), non-trivial + symmetric site', SIGN, 'PBE0', nontrivial('R3', True), A),
             ('R4 single site (shell + field)', SIGN, 'PBE0', lambda r: agrees(r['observed'], rung(r, 'R4_single')), P),
             ('R4 single site, non-trivial', SIGN, 'PBE0', nontrivial('R4_single'), A),
             ('R4 single site, non-trivial + symmetric site', SIGN, 'PBE0', nontrivial('R4_single', True), A)]
    for crit, field, st in (('class, full profile (P4r + extra points)', 'class', X),
                            ('class, pre-registered profile (P0-P4)', 'class_preregistered_profile', P)):
        rows += [('R5', crit, lev, class_ok(field, lev), st) for lev in LEVELS]
    rows += [('R5', SIGN, lev, lambda r, lev=lev: agrees(r['observed'], (r.get('R5') or {}).get(lev)), A) for lev in LEVELS]
    rows += [('R5 corrected, resolved', RESOLVED, 'PBE0+MP2', resolved_ok, A)]
    return rows


ADDED_COUNTS = ('R4a collective (field, LR mean)', 'R5 corrected, resolved')
RESOLVED = 'class observed and budget envelope excludes 0 (margin at P4r, else R5 at P4)'


def resolved_ok(r):
    """gap map B5: the full-profile class at PBE0+MP2 is 'observed' and the budget envelope of the corrected sign
    excludes 0; None where either is missing."""
    c, rs = (r.get('class') or {}).get('PBE0+MP2'), r.get('resolution')
    if not c or not rs:
        return None
    return c[0] == 'observed' and rs['verdict'].startswith('resolved: observed')


resolved_ok.why = lambda r: ('unresolved, class observed' if (r.get('resolution') or {}).get('envelope_spans_0')
                             and ((r.get('class') or {}).get('PBE0+MP2') or [''])[0] == 'observed' else 'wrong class or sign')


def why_missing(r, name, lev=None):
    """the reason a crystal is not counted in a rung (gap map B8)"""
    if r['key'] == 'brcoc':
        return 'ranking only, no transferring bond'
    if lev:
        if r.get('class_skipped_reason'):
            return r['class_skipped_reason']
        m = (r.get('class_missing') or {}).get(lev)
        return f"{lev} missing at {', '.join(m)}" if m else f'no class at {lev}'
    site = 'no transfer-site cluster (mol/emb_cluster_repeat_site.json)'
    return next((w for k, w in (('R1f', 'no R1f (mol/pair.json at P4)'), ('R4a', 'no field rung (mol/emb_pair_repeat.json)'),
                                ('R3', site), ('R4 ', site)) if name.startswith(k)), 'not computed')


def secondary(rows):
    """the secondary endpoints of PLAN_FINAL: within-family trends against 2.303RT dpKa (descriptive; families of at
    least 3 crystals) and D...A against dpKa; beside them the rank correlation pooled over all crystals (added 22 Oct
    2026, not the pre-registered form).  Energies as E(ion pair) - E(neutral) (fixed_ion_minus_neutral)."""
    from scipy.stats import spearmanr
    rs = [r for r in rows if r['key'] != 'brcoc' and r.get('fixed_ion_minus_neutral')]
    qs = {'R1f PBE0': 'R1f', 'R5 PBE0+MP2': 'R5_PBE0+MP2', 'dE_env PBE0': 'dE_env_PBE0'}

    def rho(x, y):
        s, p = spearmanr(x, y)
        n, z = len(x), math.atanh(max(min(s, 0.999999), -0.999999))
        se = 1 / math.sqrt(n - 3)
        return {'n': n, 'spearman': round(float(s), 3), 'p': round(float(p), 4),
                'ci95_fisher_z': [round(math.tanh(z - 1.96 * se), 2), round(math.tanh(z + 1.96 * se), 2)]}
    fam = {}
    for r in rs:
        fam.setdefault(r.get('family') or '?', []).append(r)
    within = {}
    for f, g in sorted(fam.items()):
        e = within[f] = {'n': len(g), 'members': [r['key'] for r in g], 'observed': sorted({r['observed'] for r in g})}
        if len(g) < 3:
            continue
        for name, k in qs.items():
            y = [r['fixed_ion_minus_neutral'].get(k) for r in g]
            if None not in y:
                x = [r['dpka'] for r in g]
                e[name] = {'slope_kJ_per_pKa': round(float(np.polyfit(x, y, 1)[0]), 2), 'spearman': round(float(spearmanr(x, y)[0]), 3)}
    pooled = {}
    for name, k in qs.items():
        g = [r for r in rs if r['fixed_ion_minus_neutral'].get(k) is not None]
        pooled[name] = rho([r['dpka'] for r in g], [r['fixed_ion_minus_neutral'][k] for r in g])
        h = [r for r in g if not r.get('dpka_off_axis')]
        pooled[name]['spearman_without_off_axis'] = round(float(spearmanr([r['dpka'] for r in h], [r['fixed_ion_minus_neutral'][k] for r in h])[0]), 3)
    da = {k: rho([r['dpka'] for r in rs if r.get(k)], [r[k] for r in rs if r.get(k)]) for k in ('DA_xray_A', 'DA_P0_A')
          if sum(1 for r in rs if r.get(k)) > 3}
    return {'RT2303_kJ_per_pKa': round(2.303 * 0.0083144626 * 298.15, 3), 'within_family': within,
            'pooled_added_22oct': pooled, 'DA_vs_dpka': da,
            'note': 'descriptive; the crystals were chosen where the dpKa rule fails, so a missing trend is partly built in'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(OPTC / 'results'))
    ap.add_argument('--out', default=str(OPTC / 'results' / 'summary'))
    a = ap.parse_args()
    R = Path(a.results)
    reg = json.loads((OPTC / 'crystals.json').read_text())
    rows = [crystal_summary(k, reg, R) for k in reg['crystals'] if (R / k).exists()]
    absent = [k for k in reg['crystals'] if not (R / k).exists()]
    counts = {}
    R5L = {'R5 PBE': 'PBE', 'R5 PBE0': 'PBE0', 'R5 PBE0 + MP2 pair correction': 'PBE0+MP2',
           'R5 PBE0 + SCS-MP2 pair correction': 'PBE0+SCS_MP2', 'R5 PBE0 + wB97X-D3(BJ) pair correction': 'PBE0+wB97X_D3BJ',
           'R5 corrected, resolved': 'PBE0+MP2'}
    for name, get in (
            ('R1f PBE0', lambda r: agrees(r['observed'], (r.get('R1f') or {}).get('PBE0'))),
            ('R4a single site (field)', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R4a_single'))),
            ('R4a collective (field)', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R4a_collective'))),
            ('R4a collective (field, LR mean)', lambda r: agrees(r['observed'], rung(r, 'R4a_collective_lr'))),
            ('R3 (shell)', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R3'))),
            ('R4 single site (shell + field)', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R4_single'))),
            ('R5 PBE', lambda r: (r.get('class', {}).get('PBE') or [None])[0] == 'observed' if r.get('class', {}).get('PBE') else None),
            ('R5 PBE0', lambda r: (r.get('class', {}).get('PBE0') or [None])[0] == 'observed' if r.get('class', {}).get('PBE0') else None),
            ('R5 PBE0 + MP2 pair correction', lambda r: (r.get('class', {}).get('PBE0+MP2') or [None])[0] == 'observed' if r.get('class', {}).get('PBE0+MP2') else None),
            ('R5 PBE0 + SCS-MP2 pair correction', lambda r: (r.get('class', {}).get('PBE0+SCS_MP2') or [None])[0] == 'observed' if r.get('class', {}).get('PBE0+SCS_MP2') else None),
            ('R5 PBE0 + wB97X-D3(BJ) pair correction', lambda r: (r.get('class', {}).get('PBE0+wB97X_D3BJ') or [None])[0] == 'observed' if r.get('class', {}).get('PBE0+wB97X_D3BJ') else None),
            ('R5 corrected, resolved', resolved_ok)):
        v = [get(r) for r in rows if r['key'] != 'brcoc']
        v = [x for x in v if x is not None]
        k, n = sum(v), len(v)
        # gap map B8: the crystals not counted and why (the numbers above are as before)
        ex = [[x, 'no results directory'] for x in absent] + [[r['key'], why_missing(r, name, R5L.get(name))] for r in rows
                                                               if r['key'] == 'brcoc' or get(r) is None]
        counts[name] = {'reproduced': k, 'of': n, 'ci95': clopper_pearson(k, n) if n else None, 'excluded': ex}
        if name in ADDED_COUNTS:
            counts[name]['status'] = 'added 22 Oct 2026, not pre-registered'
    nr5 = {n: counts[n]['of'] for n in R5L if n in counts and n != 'R5 corrected, resolved'}
    warn = [f"R5 counted over different n by level: {', '.join(f'{n} {v}' for n, v in nr5.items())}"] if len(set(nr5.values())) > 1 else []
    # counts by class (one denominator per rung: the crystals with a transferring bond and that rung)
    by_class = {}
    for name, get in (('R1f PBE0', lambda r: agrees(r['observed'], (r.get('R1f') or {}).get('PBE0'))),
                      ('R4a single site', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R4a_single'))),
                      ('R4 single site', lambda r: agrees(r['observed'], r.get('ladder', {}).get('repeat', {}).get('R4_single'))),
                      ('R5 PBE0 + MP2', lambda r: (r.get('class', {}).get('PBE0+MP2') or [None])[0] == 'observed' if r.get('class', {}).get('PBE0+MP2') else None)):
        for cls in ('salt', 'cocrystal'):
            v = [get(r) for r in rows if r['key'] != 'brcoc' and r['observed'] == cls]
            v = [x for x in v if x is not None]
            by_class[f'{name} | {cls}'] = {'reproduced': sum(v), 'of': len(v), 'ci95': clopper_pearson(sum(v), len(v)) if v else None}
    # every other rung by class (gap map A2, 22 Oct 2026), the same format
    for name, get in (('R4a collective', lambda r: agrees(r['observed'], rung(r, 'R4a_collective'))),
                      ('R3', lambda r: agrees(r['observed'], rung(r, 'R3'))), ('R5 PBE', class_ok('class', 'PBE')),
                      ('R5 PBE0', class_ok('class', 'PBE0')), ('R5 PBE0 + SCS-MP2', class_ok('class', 'PBE0+SCS_MP2')),
                      ('R5 PBE0 + wB97X-D3(BJ)', class_ok('class', 'PBE0+wB97X_D3BJ'))):
        t = tally(rows, get)
        for cls in ('salt', 'cocrystal'):
            by_class[f'{name} | {cls}'] = {k: t[cls][k] for k in ('reproduced', 'of', 'ci95')}
    table = [dict(rung=n, criterion=c, level=lev, status=st, **tally(rows, get, absent)) for n, c, lev, get, st in count_rows()]
    summ = {'crystals': rows, 'counts': counts, 'counts_by_class': by_class, 'counts_table': table,
            'counts_table_note': ('R5 in counts is the class of the full profile (P4r and the extra points included); the '
                                  'molecular rungs are scored by their two-configuration sign by design; the 95 % '
                                  'intervals are descriptive only (the crystals are not a random sample)'),
            'polymorph_brsalt_brcoc': polymorph_ranking(R), 'counts_warnings': warn, 'secondary': secondary(rows),
            'budget_label': BUDGET_LABEL + ' (crystals[].budget, crystals[].resolution; gap map B4, B5)'}
    Path(a.out + '.json').write_text(json.dumps(summ, indent=1))
    for w in warn:
        print('WARNING:', w, file=sys.stderr)
    L = ['# Option C results (generated by optc/src/analyse.py)', '',
         'kJ/mol per transferring pair, other state minus observed state (> 0: the observed state is lower).', '',
         '| crystal | dpKa | observed | R1f | R4a ss | R4a coll E_int/2 / LR | R3 | R4 ss | R5 PBE | R5 PBE0 | R5 PBE0+MP2 | dE_env | class (PBE0+MP2) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def pk(r):
        """the pre-registered dpka, marked off-axis and with the alternative / site value where crystals.json has one"""
        return f"{r['dpka']:+.2f}" + (' (off-axis)' if r.get('dpka_off_axis') else '') + \
            (f" (alt {r['dpka_alt']:+.1f})" if r.get('dpka_alt') is not None else '') + \
            (f" (site {r['dpka_site']:+.2f})" if r.get('dpka_site') is not None else '')
    for r in sorted(rows, key=lambda r: r['dpka']):
        lr = r.get('ladder', {}).get('repeat', {})
        g = lambda x: '' if x is None else f'{x:.1f}' if isinstance(x, (int, float)) else str(x)
        e = (' (= R1f)', ' (= R4a)') if lr.get('site_shell_empty') else ('', '')
        L.append(f"| {r['key']} | {pk(r)} | {r['observed']} | {g((r.get('R1f') or {}).get('PBE0'))} | "
                 f"{g(lr.get('R4a_single'))} | {g(lr.get('R4a_collective'))} / {g(lr.get('R4a_collective_lr'))} | "
                 f"{g(lr.get('R3'))}{e[0]} | {g(lr.get('R4_single'))}{e[1]} | "
                 f"{g((r.get('R5') or {}).get('PBE'))} | {g((r.get('R5') or {}).get('PBE0'))} | "
                 f"{g((r.get('R5') or {}).get('PBE0+MP2'))} | {g((r.get('dE_env') or {}).get('PBE0'))} | "
                 f"{(r.get('class', {}).get('PBE0+MP2') or ['', ''])[0]} |")
    L += ['', '(= R1f) / (= R4a): empty transfer-site shell, R3 = R1f and R4 = R4a by definition (optc/PLAN.md, 19 Oct 2026 '
          '8:26 AM IST); counted in R3 / R4 below, left out of the non-trivial rows of the criterion table.']
    # gap map B2: the two collective estimators side by side
    gaps = [(abs(rung(r, 'collective_estimator_gap', m)), r['key'], m) for r in rows for m in ('repeat', 'sc', 'gas')
            if rung(r, 'collective_estimator_gap', m) is not None]
    gr = max((x for x in gaps if x[2] == 'repeat'), default=None)
    miss = [r for r in rows if agrees(r['observed'], rung(r, 'R4a_collective')) is False]
    if gr:
        ga = max(gaps)
        L += ['', f"R4a coll: the pre-registered collective estimator E_emb - E_int/2 / the linear-response mean of the forward "
              f"(P0 environment) and reverse (P4 environment) vertical gaps (item B, added 19 Oct 2026 9:40 PM IST); the "
              f"pre-registered estimator is E_emb - E_int/2 (optc/PLAN.md, R4a), the LR mean is a second estimator.  "
              f"Their difference, up to {gr[0]:.1f} kJ/mol with REPEAT ({gr[1]}; {ga[0]:.1f} over all three charge models, "
              f"{ga[1]} {ga[2]}), is the uncertainty of the collective field rung"
              + ('; the E_int/2 misses with the LR mean: ' + ', '.join(f"{r['key']} {rung(r, 'R4a_collective_lr'):+.1f}" for r in miss
                                                                     if rung(r, 'R4a_collective_lr') is not None) if miss else '')
              + '.  Counts of both estimators with all three charge models: criterion table.']
    # gap map B6: dpKa metadata beside the pre-registered value
    md = [r for r in sorted(rows, key=lambda r: r['dpka']) if any(r.get(k) is not None for k in ('dpka_off_axis', 'dpka_alt', 'dpka_site', 'dpka_quality'))]
    if md:
        L += ['', 'dpKa: the pre-registered value (crystals.json dpka, unchanged; the rows stay sorted by it).  ' + '; '.join(
            f"{r['key']}: " + ', '.join(x for x in (
                f"off-axis ({r.get('dpka_off_axis_reason')})" if r.get('dpka_off_axis') else '',
                f"alternative {r['dpka_alt']:+.1f} {r.get('dpka_alt_range')}" if r.get('dpka_alt') is not None else '',
                f"site value {r['dpka_site']:+.2f}" if r.get('dpka_site') is not None else '',
                (r.get('dpka_quality') or '').split(':')[0]) if x) for r in md) + ' (sources: crystals.json).']
    L += ['', '## Class per level (minimum of the PCHIP profile; full profile with P4r and the extra points / '
          'pre-registered P0-P4 profile; delta_min in A)', '',
          '| crystal | observed | ' + ' | '.join(LEVELS) + ' |', '|---|---|' + '---|' * len(LEVELS)]
    ab = {'observed': 'obs', 'other': 'OTHER', 'shared': 'shared'}
    cc = lambda c: f'{ab[c[0]]} {c[1]:+.2f}' if c else '-'
    for r in sorted(rows, key=lambda r: r['dpka']):
        if r.get('class'):
            L.append(f"| {r['key']} | {r['observed']} | " + ' | '.join(
                f"{cc(r['class'].get(lev))} / {cc((r.get('class_preregistered_profile') or {}).get(lev))}" for lev in LEVELS) + ' |')
    L += ['', '| rung | reproduced | of | 95 % interval |', '|---|---|---|---|']
    for k, v in counts.items():
        L.append(f"| {k}{' (added 22 Oct 2026, not pre-registered)' if v.get('status') else ''} | {v['reproduced']} | {v['of']} | {v['ci95']} |")
    L += ['', 'R5 rows: the class of the full profile (P4r and the extra points included); the pre-registered P0-P4 class '
          'and the sign at P4 are in the criterion table below.  R5 corrected, resolved (gap map B5): the class at PBE0 + MP2 '
          'is observed and the envelope of the uncertainty budget of the corrected sign excludes 0 (the margin at P4r where '
          'there is an other-side state, else R5 at P4); the pre-registered classes and counts above are unchanged.']
    L += ['', 'Not counted, per rung (gap map B8):'] + [
        f"- {k}: " + ('; '.join(f'{x} ({why})' for x, why in v['excluded']) or 'none') for k, v in counts.items()]
    L += [f'- WARNING: {w}' for w in warn]
    L += ['', '| rung, class | reproduced | of | 95 % interval |', '|---|---|---|---|']
    for k, v in by_class.items():
        L.append(f"| {k} | {v['reproduced']} | {v['of']} | {v['ci95']} |")
    common = [x for x in table[0]['excluded'] if all(x in t['excluded'] for t in table)]
    L += ['', '## Counts by rung, criterion and level (gap map A2)', '',
          '| rung | criterion | level | status | all | 95 % interval | salts | cocrystals | missed | not counted (this row) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for t in table:
        kn = lambda c: f"{t[c]['reproduced']}/{t[c]['of']}"
        L.append(f"| {t['rung']} | {t['criterion']} | {t['level']} | {t['status']} | {kn('all')} | {t['all']['ci95']} | {kn('salt')} | "
                 f"{kn('cocrystal')} | {', '.join(t['all']['missed'])} | "
                 f"{'; '.join(f'{k} ({why})' for k, why in t['excluded'] if [k, why] not in common)} |")
    L += ['', 'Not counted in any row: ' + ('; '.join(f'{k} ({why})' for k, why in common) or 'none') + '.  The molecular '
          'rungs are two-configuration quantities and are scored by their sign at P4 by design; R5 is scored by its '
          'profile.  Symmetric site: an empty-shell crystal counted with its symmetric-site cluster (item M) where that '
          'shell is not empty.  95 % intervals (Clopper-Pearson) are descriptive only: the crystals are not a random sample.']
    g = lambda x: '' if x is None else (f'{x:.1f}' if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
    # gap map B3: the pre-registered split and its verdict first; the disjoint decomposition is exploratory
    sp = [r for r in rows if r['observed'] == 'salt' and rung(r, 'split_presentable') is not None]
    ok = [r['key'] for r in sp if rung(r, 'split_presentable') is True]
    L += ['', f"## Pre-registered field/shell split (PLAN_FINAL endpoint 3): verdict "
          f"{'presentable in ' + ', '.join(ok) if ok else 'not presentable in any salt'} (REPEAT; kJ/mol)", '',
          '| crystal | field | shell | nonadditivity | field order spread | field share | split presentable | field split robust (sc) |',
          '|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: r['dpka']):
        m = r.get('ladder', {}).get('repeat', {})
        if m.get('field') is not None:
            fs = '' if m.get('field_share') is None else f"{m['field_share']:.2f}"
            L.append(f"| {r['key']} | {g(m.get('field'))} | {g(m.get('shell'))} | {g(m.get('nonadditivity'))} | "
                     f"{g(m.get('field_order_spread'))} | {fs} | "
                     f"{g(m.get('split_presentable'))} | {g(r.get('field_split_robust'))} |")
    L += ['', 'Field and shell: averages over both orders; split presentable (pre-registered): |nonadditivity| < min(|field|, '
          '|shell|); n/a: empty transfer-site shell (shell = 0 by definition; split_presentable_legacy False in summary.json); '
          'field split robust: the field share keeps its sign and changes by less than half with the sc charges.']
    L += ['', '## EXPLORATORY disjoint decomposition (added 19 Oct 2026 9:40 PM IST, after the pre-registered split failed; '
          'PLAN item A; REPEAT; kJ/mol, other - observed)', '',
          '| crystal | shell R3-R1f | outer R4-R3 | outer rev R4a_out-R1f | shell rev R4-R4a_out | cross | shell charges | electrostatic R4a-R1f | beyond R4-R4a | robust (sc) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: r['dpka']):
        dd = (r.get('decomposition') or {}).get('repeat') or {}
        rb = (r.get('decomposition') or {}).get('robust') or {}
        L.append(f"| {r['key']} | {g(dd.get('shell'))} | {g(dd.get('outer'))} | {g(dd.get('outer_rev'))} | {g(dd.get('shell_rev'))} | "
                 f"{g(dd.get('cross'))} | {g(dd.get('shell_charges'))} | {g(dd.get('electrostatic'))} | {g(dd.get('beyond_electrostatic'))} | "
                 f"{robust_text(rb)} |")
        st = (r.get('checks') or {}).get('sitesym_repeat_terms') or {}
        if st and not st.get('empty_site_shell'):
            L.append(f"| {r['key']}, symmetric site (item M) | {g(st.get('shell'))} | {g(st.get('outer'))} | {g(st.get('outer_rev'))} | "
                     f"{g(st.get('shell_rev'))} | {g(st.get('cross'))} | {g(st.get('shell_charges'))} | {g(st.get('electrostatic'))} | "
                     f"{g(st.get('beyond_electrostatic'))} |  |")
    L += ['', 'Symmetric site (item M): the shell H-bonded to D, A and the H atoms on either, the same terms (REPEAT only); '
          'the asymmetric transfer-site shell of the cocrystals is empty, so their shell = 0 holds by that definition only.']
    L += ['', '## Reverse gaps, periodic closure, margins and nuclear quantum effects', '',
          '| crystal | R4a fwd / rev / lambda (REPEAT) | site R4 fwd / rev | T1 PBE0 (f = 1/Z) | T2 PBE0 (est.) | '
          'R4 - T1 / R4 - T2 est. | Nv PBE0 | coop. Nv-T1 (item C, PBE0) | Nv-T1v (vert, PBE) | T1-T1v (PBE) | reorg. | '
          'margin PBE0+MP2 | ZPE-corr. margin (H, PBE0+MP2) | p(other) H, P0 frame |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def mtext(m):
        if not m:
            return ''
        return f"{m['margin']:.1f} ({m['point']})" if m.get('margin') is not None else f"none; crossing {m['crossing_energy_delta0']:.1f}"

    def ztext(z):
        if not z:
            return ''
        v, k = z['E0_difference_kJ'], z['kind']
        return f'{v:.1f}' if k.startswith('ZPE') else f'>={v:.1f} (deloc.)' if 'lower' in k else \
            f'<={v:.1f} (P0 deloc.)' if 'upper' in k else f'~{v:.1f} (both deloc.)'
    for r in sorted(rows, key=lambda r: r['dpka']):
        rv = (r.get('reverse') or {})
        pr, sr = rv.get('pair_repeat') or {}, rv.get('site_repeat') or {}
        cl = r.get('closure') or {}
        nu = r.get('nuclear') or {}
        zc = ((nu.get('zpe_frozen_frame') or {}).get('H') or {}).get('PBE0+MP2')
        po = nu.get('p_other_H_printed') or {}
        dv = '' if po.get('value') is None else f"{po['value']:.4f}" + ('' if 'P0_frame' in po['source'] else ' (adiab.)')
        sg = ' (sign undetermined)' if 'undetermined' in (cl.get('cluster_minus_periodic_single_site_sign') or '') else ''
        L.append(f"| {r['key']} | {g(pr.get('embedded_single_site_kJ'))} / {g(pr.get('reverse_kJ'))} / {g(pr.get('lambda_kJ'))} | "
                 f"{g(sr.get('embedded_single_site_kJ'))} / {g(sr.get('reverse_kJ'))} | {g(cl.get('T1_pbe0_kJ'))} | "
                 f"{g(cl.get('T2_pbe0_est_kJ'))} | {g(cl.get('cluster_minus_periodic_single_site'))} / "
                 f"{g(cl.get('cluster_minus_periodic_single_site_T2'))}{sg} | "
                 f"{g(cl.get('Nv_pbe0_kJ_per_pair'))} | {g(cl.get('cooperativity_preregistered_pbe0'))} | "
                 f"{g(cl.get('cooperativity_vertical_pbe_kJ'))} | {g(cl.get('pair_relaxation_single_site_pbe'))} | "
                 f"{g(cl.get('reorganization_pbe0'))} | "
                 f"{mtext((r.get('margin_other_side') or {}).get('PBE0+MP2'))} | {ztext(zc)} | {dv} |")
    L += ['', 'T1: periodic single site, one pair of the Z in the cell moved (f = 1/Z).  T2 PBE0 (est.) = T2(PBE) + '
          '[T1(PBE0) - T1(PBE)]: T2 (doubled cell) was run at PBE only, a true PBE0 T2 was not computed.  Sign '
          'undetermined: R4 - T1 and R4 - T2 est. differ in sign.',
          'Cooperativity (gap map B1): N_vert - T1 as defined before its results (PLAN 19 Oct 9:40 PM item C, an addition to the pre-registration; T1 holds its one pair at the relaxed '
          'P4 geometry, N_vert moves only the H of every pair); vert.json\'s N_vert - T1v (both vertical); they differ by '
          'T1 - T1v, the relaxation of the one pair (PBE only).  Neither is a like-for-like test of the E_int/2 term (ladder '
          'cooperativity R4a_collective - R4a_single).  The cycle R1f -> R4 -> T1 -> N_vert -> R5 is closure.cycle_pbe0 / '
          'cycle_pbe in summary.json.',
          'Margin: the lowest sampled point at delta > +0.1 A minus P0, where the profile has an other-side minimum (P4r '
          'present, interior PCHIP minimum at delta > +0.1 A); otherwise none, with the crossing energy at delta = 0 (not a '
          'state).  Hysteresis pairs reduced to the lower energy; the legacy value (lowest point at delta > 0) is '
          'margin_other_side_legacy in summary.json.',
          'ZPE-corr. margin: bends perpendicular to D->A; >= x (deloc.): the P4r-frame ground state lies above its dividing '
          'point, so x is a lower bound (no localized other-state level).  p(other): the far-side population of the P0 '
          'frozen frame; (adiab.): no frozen frame (salts), the 1D DVR on the relaxed profile (adiabatic limit).']
    # -- uncertainty budget and the resolution of the corrected sign (gap map B4, B5)
    f2 = lambda x: '' if x is None else f'{x:+.2f}'

    def pm(q):
        return '' if not q else f"{q['central']:.1f} +/- {q['rss']:.1f} [{q['envelope'][0]:.1f}, {q['envelope'][1]:.1f}]"

    def cv(q, k):
        v = ((q or {}).get('components') or {}).get(k)
        return ' / '.join(f'{x:+.2f}' for x in v.values()) if v else ''
    bud = [r for r in sorted(rows, key=lambda r: r['dpka']) if r.get('budget')]
    if bud:
        L += ['', f'## Uncertainty budget ({BUDGET_LABEL}; kJ/mol)', '',
              '| crystal | margin at P4r (PBE0+MP2) | resolution | R5 at P4 (PBE0+MP2) | dE_env (PBE0) | R4a ss | R4a coll (E_int/2) | R4 ss |',
              '|---|---|---|---|---|---|---|---|']
        for r in bud:
            bd, rs = r['budget'], r.get('resolution') or {}
            mq = pm(bd.get('margin')) or (pm(bd.get('P4r')) + ' (P4r - P0, no other-side minimum: not a margin)' if bd.get('P4r') else '')
            L.append(f"| {r['key']} | {mq} | {rs.get('verdict', '')} | "
                     f"{pm(bd.get('R5_P4'))} | {pm(bd.get('dE_env'))} | {pm(bd.get('R4a_single'))} | {pm(bd.get('R4a_collective'))} | "
                     f"{pm(bd.get('R4_single'))} |")
        L += ['', 'value +/- rss [min, max]: the central value; the root sum of squares over the components of the largest '
              '|shift| of each (the PBE / PBE50 anchors and the def2-QZVP / def2-TZVPD bases each counted once); the envelope, '
              'central + each shift one at a time and the relaxed-cell combinations (computed directly, or summed).  Margin '
              '(where there is an other-side P4r; the X-ray cell, the setting of the pre-registered profile): level spread '
              '(SCS-MP2, wB97X-D3(BJ)), delta(T), PBE / PBE50 anchors, in-field pair correction, relaxed cells, k and q mesh, '
              'def2-QZVP / def2-TZVPD pair correction, ZPE (frozen frame where both frames are localized; harmonic H D A and H '
              'only); a delocalized frame\'s ZPE (a bound) and other reference points (a definition) are listed apart, outside '
              'rss and the envelope.  Where margin() finds no other-side minimum at PBE0+MP2 (mppcp, nbaetpy) the same budget is '
              'that of P4r - P0, not a margin.  R5 at '
              'P4: the same at P4 with the vc cell shift.  dE_env: the PBE / PBE50 anchors, the cell and k mesh of R5, the basis '
              'of R1f (the pair corrections cancel).  Field and cluster rungs (REPEAT): the sc / gas charges, smearing radius and '
              'cutoff, the LR-mean collective estimator, the whole first shell / symmetric site.  The charge models and the shell '
              'size do not enter the periodic margin; every component not computed is None in summary.json (budget.*.none_reason).',
              'Resolution: the margin where margin() finds an other-side state at PBE0+MP2, else R5 at P4; unresolved when the '
              'budget envelope of that quantity spans 0.  The pre-registered classes and counts are unchanged.']
        mrg = [r for r in bud if r['budget'].get('margin') or r['budget'].get('P4r')]
        if mrg:
            L += ['', '### Margin components (shift from the central value; kJ/mol)', '',
                  '| crystal | central | SCS / wB97X | delta(T) | anchor PBE / PBE50 | in-field X-ray / relaxed | relaxed cells | k | q | '
                  'QZVP / TZVPD | ZPE (frozen) / HDA / H only | outside rss: bounds, other reference points | relaxed-cell combinations |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for r in mrg:
                q = r['budget'].get('margin') or r['budget']['P4r']
                rp = dict(q.get('definition_alternatives') or {})
                if q.get('bounds'):
                    rp[f"ZPE bound, {q['bounds']['kind'].split(', ')[-1].split(' (')[0]}"] = q['bounds']['frozen frame, H (dZPE)']
                L.append(f"| {r['key']}{'' if r['budget'].get('margin') else ' (P4r - P0)'} | {q['central']:.2f} | {cv(q, 'level_spread')} | {cv(q, 'delta_T')} | "
                         f"{cv(q, 'anchor_PBE') or '-'} / {cv(q, 'anchor_PBE50') or '-'} | {cv(q, 'in_field_vs_vacuum')} | {cv(q, 'cell')} | "
                         f"{cv(q, 'k_mesh')} | {cv(q, 'q_mesh')} | {cv(q, 'basis_QZVP') or '-'} / {cv(q, 'basis_TZVPD') or '-'} | "
                         f"{cv(q, 'ZPE')} | {'; '.join(f'{k} {x:+.2f}' for k, x in rp.items())} | "
                         f"{'; '.join(f'{k}: {x:.2f}' for k, x in (q.get('combined') or {}).items())} |")
        un = [r for r in bud if (r.get('resolution') or {}).get('envelope_spans_0')]
        L += ['', 'Unresolved: ' + (', '.join(r['key'] for r in un) or 'none') + '.']
        for r in [x for x in bud if x['key'] == 'dapsone' or x in un]:
            q, rs = (r['budget'].get('margin') or r['budget'].get('R5_P4')), r['resolution']
            p4, vf = (r.get('profile') or {}).get(q['point']) or {}, (r.get('checks') or {}).get('vcfree') or {}
            b, cp, c = 'dE_relaxed_cells_pbe0', q['components'], q['central']
            one = lambda k, a=None: None if not cp.get(k) else c + (cp[k][a] if a else next(iter(cp[k].values())))
            L.append(f"- {r['key']} ({q['point']} - P0, kJ/mol per pair): X-ray cell PBE0+MP2 {f2(c)} (SCS-MP2 {f2(p4.get('PBE0+SCS_MP2'))}, "
                     f"wB97X-D3(BJ) {f2(p4.get('PBE0+wB97X_D3BJ'))}); relaxed cells {f2(vf.get(b + '_MP2'))} / {f2(vf.get(b + '_SCS_MP2'))} / "
                     f"{f2(vf.get(b + '_wB97X_D3BJ'))}, in-field {f2(vf.get(b + '_MP2_infield'))} / {f2(vf.get(b + '_SCS_MP2_infield'))} / "
                     f"{f2(vf.get(b + '_wB97X_D3BJ_infield'))} (MP2 / SCS-MP2 / wB97X-D3(BJ)); "
                     + ''.join(f'{k} {x:+.2f}; ' for k, x in (q.get('combined') or {}).items() if '+' in k)
                     + f"PBE-anchored {f2(one('anchor_PBE')) or 'n/a'}, PBE50-anchored {f2(one('anchor_PBE50')) or 'not run'}; in-field "
                     f"pair correction (X-ray cell) {f2(one('in_field_vs_vacuum', next(iter(cp.get('in_field_vs_vacuum') or {'': 0})))) or 'n/a'}; "
                     f"def2-QZVP {f2(one('basis_QZVP')) or 'not run'}; + delta(T) {f2(one('delta_T')) or 'not run'}; rss {q['rss']:.2f}, envelope "
                     f"[{q['envelope'][0]:+.2f}, {q['envelope'][1]:+.2f}]: {rs['verdict']}.")
    # -- cocrystal checks (gap map A3, A8, A9)
    coc = [r for r in sorted(rows, key=lambda r: r['dpka']) if r['observed'] == 'cocrystal' and 'P4r' in (r.get('profile') or {})]
    g = lambda x: '' if x is None else f'{x:.2f}'
    if coc:
        L += ['', '## Cocrystal checks at P4r (kJ/mol, P4r - P0; X-ray cells)', '',
              '| crystal | PBE0 | PBE0+MP2 | delta(T) (def2-SVP) | MP2 corr. vacuum / in-field (NR) | k shift (PBE, dense - prod.) | '
              'q mesh | q shift (PBE0) | PBE0 / PBE0+MP2 at q |', '|---|---|---|---|---|---|---|---|---|']
        for r in coc:
            p4, ck = r['profile']['P4r'], r.get('checks') or {}
            pc, qm = ck.get('pair_cc') or {}, ck.get('qmesh') or {}
            nr = ((ck.get('embcorr_pair_repeat') or {}).get('NR') or {}).get('corr_MP2_kJ')
            vac = None if None in (p4.get('PBE0+MP2'), p4.get('PBE0')) else round(p4['PBE0+MP2'] - p4['PBE0'], 3)
            qs = qm.get('P4r_shifted') or {}
            L.append(f"| {r['key']} | {g(p4.get('PBE0'))} | {g(p4.get('PBE0+MP2'))} | {g(pc.get('delta_T_P4r_kJ'))} | "
                     f"{g(vac)} / {g(nr)} | {g(r.get('kcheck_shift_P4r'))} | {''.join(map(str, qm.get('nq') or [])) or ''} | "
                     f"{g(qm.get('shift_kJ_per_fu'))} | {g((qs.get('PBE0') or {}).get('q'))} / {g((qs.get('PBE0+MP2') or {}).get('q'))} |")
        L += ['', 'delta(T): [CCSD(T) - MP2] of the pair at P4r minus at P0 (mol/pair_cc.json; not run where blank).  MP2 '
              'corr.: [MP2 - PBE0] P4r - P0 of the pair in vacuum (pair.json) and in the field (embcorr NR: every atom at P4r '
              'in the P4r field; plain differences, no E_int/2).  k shift: PBE dE(P4r) at the denser mesh minus production.  '
              'q: PBE0 (SG15) at the EXX q mesh = the k mesh against q = 1x1x1, applied to the P4r profile values.']
        L += ['', '### Relaxed cells (vcfree; vacuum / in-field pair corrections; delta(T) and q shift of the X-ray-cell pair)', '',
              '| crystal | PBE0 | +MP2 | +SCS-MP2 | +wB97X-D3(BJ) | +MP2 +delta(T) | +MP2 +q | +MP2 +delta(T) +q |',
              '|---|---|---|---|---|---|---|---|']
        for r in coc:
            vf = (r.get('checks') or {}).get('vcfree') or {}
            if vf.get('dE_relaxed_cells_pbe0') is None:
                continue
            b = 'dE_relaxed_cells_pbe0'
            two = lambda s, q='': f"{g(vf.get(f'{b}_{s}{q}'))} / {g(vf.get(f'{b}_{s}_infield{q}'))}"
            dt = lambda q='': f"{g(vf.get(f'{b}_MP2_dT{q}'))} / {g(vf.get(f'{b}_MP2_dT_infield{q}'))}" if f'{b}_MP2_dT' in vf else ''
            L.append(f"| {r['key']} | {g(vf.get(b))} | {two('MP2')} | {two('SCS_MP2')} | {two('wB97X_D3BJ')} | {dt()} | "
                     f"{two('MP2', '_q')} | {dt('_q')} |")
        sym = [r for r in coc if (r.get('checks') or {}).get('sitesym_repeat_terms')]
        if sym:
            L += ['', '### Symmetric site (item M, REPEAT; kJ/mol per pair)', '',
                  '| crystal | n shell | R3 | R4 ss | shell R3-R1f | outer R4-R3 | outer rev | shell rev | cross | shell charges |',
                  '|---|---|---|---|---|---|---|---|---|---|']
            for r in sym:
                st = r['checks']['sitesym_repeat_terms']
                L.append(f"| {r['key']} | {st.get('n_shell')} | {g(st.get('R3'))} | {g(st.get('R4_single'))} | {g(st.get('shell'))} | "
                         f"{g(st.get('outer'))} | {g(st.get('outer_rev'))} | {g(st.get('shell_rev'))} | {g(st.get('cross'))} | "
                         f"{g(st.get('shell_charges'))} |")
        # gap map B11: every check at P4r, per level
        L += ['', '### Cocrystal checks per level at P4r (kJ/mol, P4r - P0; X-ray cells unless relaxed)', '',
              '| crystal | level | X-ray cell | + delta(T) | in-field (NR) | relaxed cells | relaxed, in-field | k shift | q shift | '
              'def2-QZVP | ZPE-corr. (H) |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in coc:
            p4, ck = r['profile']['P4r'], r.get('checks') or {}
            vf, nr = ck.get('vcfree') or {}, (ck.get('embcorr_pair_repeat') or {}).get('NR') or {}
            pc, qz = ck.get('pair_cc') or {}, ck.get('pair_def2-qzvp') or {}
            zf = ((r.get('nuclear') or {}).get('zpe_frozen_frame') or {}).get('H') or {}
            dt = pc.get('delta_T_P4r_kJ') if 'P0' in (pc.get('points_done') or ()) else None
            for lev, k in (('PBE0', None), ('PBE0+MP2', 'MP2'), ('PBE0+SCS_MP2', 'SCS_MP2'), ('PBE0+wB97X_D3BJ', 'wB97X_D3BJ')):
                x, sfx = p4.get(lev), f'_{k}' if k else ''
                L.append(f"| {r['key']} | {lev} | {g(x)} | {g(x + dt) if k == 'MP2' and None not in (x, dt) else ''} | "
                         f"{g(p4['PBE0'] + nr[f'corr_{k}_kJ']) if k and None not in (p4.get('PBE0'), nr.get(f'corr_{k}_kJ')) else ''} | "
                         f"{g(vf.get(f'dE_relaxed_cells_pbe0{sfx}'))} | {g(vf.get(f'dE_relaxed_cells_pbe0{sfx}_infield')) if k else ''} | "
                         f"{g(r.get('kcheck_shift_P4r'))} | {g((ck.get('qmesh') or {}).get('shift_kJ_per_fu'))} | "
                         f"{g(p4['PBE0'] + qz['dMP2_P4r_kJ'] - qz['dPBE0_D3_P4r_kJ']) if k == 'MP2' and None not in (p4.get('PBE0'), qz.get('dMP2_P4r_kJ'), qz.get('dPBE0_D3_P4r_kJ')) else ''} | "
                         f"{ztext(zf.get(lev))} |")
        L += ['', 'X-ray cell: the profile value; + delta(T): [CCSD(T) - MP2]/def2-SVP of the pair; in-field: PBE0 + the pair '
              'correction in the P4r field (NR); relaxed cells: vcfree with the vacuum / in-field correction difference (VR - VS); '
              'k and q shifts as above (they apply to every level); def2-QZVP: the MP2 pair correction in def2-QZVP; ZPE-corr.: '
              'the frozen-frame E0 difference (bound where a frame is delocalized).']
    # -- runner-up bond (gap map B11): pair and periodic, against each crystal's margin
    ru = [r for r in sorted(rows, key=lambda r: r['dpka']) if any((r.get('checks') or {}).get(k) for k in
                                                                ('runner_up_minus_chosen_vertical', 'runner_up_minus_chosen_periodic'))]
    if ru:
        L += ['', '## Runner-up bond (runner-up minus chosen, vertical; kJ/mol per pair)', '',
              '| crystal | pair: PBE0 / MP2 / SCS-MP2 / wB97X-D3(BJ) | periodic: PBE / PBE0 | margin PBE0 | margin PBE0+MP2 | '
              'periodic + margin (or crossing) |', '|---|---|---|---|---|---|']
        for r in ru:
            ck, mo = r['checks'], r.get('margin_other_side') or {}
            pv, pp = ck.get('runner_up_minus_chosen_vertical') or {}, ck.get('runner_up_minus_chosen_periodic') or {}
            lev = 'PBE0' if pp.get('runner_minus_chosen_pbe0_kJ') is not None else 'PBE'
            m = mo.get(lev) or {}
            ref = m.get('margin') if m.get('margin') is not None else m.get('crossing_energy_delta0')
            dv = pp.get(f'runner_minus_chosen_{lev.lower()}_kJ')
            L.append(f"| {r['key']} | {('invalid: ' + pv['invalid']) if 'invalid' in pv else ' / '.join(g(pv.get(k)) for k in ('PBE0_D3', 'MP2', 'SCS_MP2', 'wB97X_D3BJ'))} | "
                     f"{g(pp.get('runner_minus_chosen_pbe_kJ'))} / {g(pp.get('runner_minus_chosen_pbe0_kJ'))} | {mtext(mo.get('PBE0'))} | "
                     f"{mtext(mo.get('PBE0+MP2'))} | {'' if None in (ref, dv) else f'{ref + dv:.1f} ({lev})'} |")
        L += ['', 'Periodic + margin: the runner-up minus chosen difference added to the margin of that level, or to the crossing '
              'energy at delta = 0 where there is no other-side minimum (all salts); > 0: the runner-up bond does not give a '
              'lower other state.  Pair: mol/pair_runner.json; periodic: runner_periodic.json (qe.py runner).']
    # -- secondary endpoints (PLAN_FINAL): dpKa trends and D...A
    se = summ.get('secondary')
    if se:
        L += ['', f"## Secondary endpoints: dpKa trends and D...A (descriptive; E(ion pair) - E(neutral), kJ/mol; "
              f"2.303RT = {se['RT2303_kJ_per_pKa']} kJ/mol per pKa unit)", '',
              '| family | n | observed | R1f PBE0: slope / rho | R5 PBE0+MP2: slope / rho | dE_env PBE0: slope / rho |', '|---|---|---|---|---|---|']
        for f, e in se['within_family'].items():
            sr = lambda q: f"{e[q]['slope_kJ_per_pKa']:+.1f} / {e[q]['spearman']:+.2f}" if q in e else '-'
            L.append(f"| {f} | {e['n']} | {', '.join(e['observed'])} | {sr('R1f PBE0')} | {sr('R5 PBE0+MP2')} | {sr('dE_env PBE0')} |")
        L += ['', 'Pooled over all crystals with a transferring bond (added 22 Oct 2026, not the pre-registered within-family '
              'form): ' + '; '.join(f"{n} rho {v['spearman']:+.2f} (p {v['p']}, 95 % Fisher z {v['ci95_fisher_z']}; "
                                    f"{v['spearman_without_off_axis']:+.2f} without the off-axis dpKa)" for n, v in se['pooled_added_22oct'].items()) + '.',
              '', '| crystal | dpKa | observed | D...A X-ray (A) | D...A P0, PBE (A) |', '|---|---|---|---|---|']
        g4 = lambda x: '' if x is None else f'{x:.3f}'
        for r in sorted((r for r in rows if r['key'] != 'brcoc'), key=lambda r: r['dpka']):
            L.append(f"| {r['key']} | {r['dpka']:+.2f} | {r['observed']} | {g4(r.get('DA_xray_A'))} | {g4(r.get('DA_P0_A'))} |")
        L += ['', 'D...A against dpKa: ' + '; '.join(f"{k.replace('_A', '')} rho {v['spearman']:+.2f} (p {v['p']}, n {v['n']})"
                                                    for k, v in se['DA_vs_dpka'].items()) + '.  ' + se['note'] + '.']
    pr = summ.get('polymorph_brsalt_brcoc')
    if pr:
        h = lambda k: '-' if pr.get(k) is None else f'{pr[k]:+.2f}'
        nq = pr.get('nq') or {}
        L += ['', '## brsalt vs brcoc (E(brcoc) - E(brsalt), kJ/mol per formula unit; > 0: the salt is lower)', '',
              '| PBE | PBE dense k | PBE0 (q = 1x1x1) | PBE0 (q = k mesh) | PBE0, dense k, q = k mesh |', '|---|---|---|---|---|',
              f"| {h('pbe_kJ_per_fu')} | {h('pbe_dense_k_kJ_per_fu')}{' (missing: ' + ', '.join(pr['dense_k_missing']) + ')' if pr.get('dense_k_missing') else ''} | "
              f"{h('pbe0_kJ_per_fu')} | {h('pbe0_qmesh_kJ_per_fu')} | {h('pbe0_dense_k_qmesh_kJ_per_fu')} |",
              '', pr['note'] + ('; q meshes ' + ', '.join(f"{k} {''.join(map(str, v))}" for k, v in nq.items()) if nq else '') + '.']
    Path(a.out + '.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
