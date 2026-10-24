"""Tables S1-S15 of the Supplementary material, from paper/ctc/numbers/data/facts.json (signed, rounded facts) and, for the
few quantities it does not carry, the pinned optc/results/summary.json and optc/crystals.json.

    python3 paper/ctc/si/make_si_tables.py          # -> paper/ctc/si/tables/tableS*.md
"""
import csv, hashlib, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NUM = HERE.parent / 'numbers'
F = json.loads((NUM / 'data' / 'facts.json').read_text())
SUMMARY = REPO / 'optc' / 'results' / 'summary.json'
assert hashlib.md5(SUMMARY.read_bytes()).hexdigest() == '411d4c2ad29499c8961de1bae2fc5b02'
S = {r['key']: r for r in json.loads(SUMMARY.read_text())['crystals']}
SUM = json.loads(SUMMARY.read_text())
C = json.loads((REPO / 'optc' / 'crystals.json').read_text())['crystals']
NAME = {'cyano': 'CNAPH', 'saph': 'SAPH', 'carboxy': 'CXAPH', 'chloro': 'CLAPH', 'anis': 'MOAPH', 'perchlorate': 'SAPC',
        'sulfamate': 'SASF', 'dapsone': 'DDS·DNBA', 'mppcp': 'MP·PCP', 'pcplut': 'LUT·PCP', 'nbaetpy': 'EPY·NBA',
        'brsalt': 'BMA–DNB', 'clsalt': 'CMA–DNB', 'brcoc': 'BMA·DNBA'}
ORDER = [k for k in F['s.order']['value'] if k != 'brcoc']
SALTS = [k for k in ORDER if F[f'x.{k}.observed']['value'] == 'salt']
COC = [k for k in ORDER if F[f'x.{k}.observed']['value'] == 'cocrystal']
OUT = HERE / 'tables'
OUT.mkdir(exist_ok=True)
DPT = 'ΔE~PT~ = E(ion pair) − E(neutral pair), kJ mol⁻¹ per transferring pair (> 0: the neutral pair is lower)'
MRG = 'other state − observed state, kJ mol⁻¹ per transferring pair (> 0: the observed state is lower)'


def t(key, dash='–'):
    v = F.get(key)
    if v is None or v.get('value') is None:
        return dash
    return v.get('text_of') or v['text']


def table(name, label, title, head, rows, note):
    num = lambda c: bool(re.fullmatch(r'[−+–\-]?[\d.]+( to [−+]?[\d.]+)?%?|\d+/\d+|–', str(c).strip()))
    al = [':---' if i == 0 or not all(num(r[i]) for r in rows if r[i] not in ('', '–')) else '---:' for i in range(len(head))]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(al) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    out += ['', f'Table: {title} {{#tbl:{label}}}', '', note]
    (OUT / f'{name}.md').write_text('\n'.join(out) + '\n')


def fmt(x, nd=1, sign=True):
    if x is None:
        return '–'
    s = f'{abs(x):.{nd}f}'
    return ('−' + s) if x < 0 and float(s) != 0 else (('+' + s) if sign else s)


def s1():
    rows = []
    for k in F['s.order']['value']:
        c = C[k]
        note = (c.get('pka_note') or '').split(';')[0][:150]
        rng = c.get('dpka_range') or ['', '']
        q = c.get('dpka_quality', 'primary measurements').split(':')[0]
        rows.append([NAME[k], fmt(c['dpka'], 2), f'{fmt(rng[0], 2)} to {fmt(rng[1], 2)}', q, note])
    table('tableS1', 'pka', 'ΔpK~a~ values, their ranges and sources.', ['Crystal', 'ΔpK~a~', 'Range', 'Quality', 'Sources (abridged)'],
          rows, 'ΔpK~a~ = pK~a~(BH^+^) − pK~a~(HA), aqueous, 25 °C, fixed in the plan before the production calculations. The '
          'full source notes (constants, temperatures, ionic strengths, primary references) are in the data deposit '
          '(optc/research/pka_sources.md).')


def s2():
    rows = []
    for r in SUM['counts_table']:
        a = r['all']
        ci = a.get('ci95') or [None, None]
        miss = ', '.join(NAME.get(m.split(' ')[0], m) + (m[len(m.split(' ')[0]):] if ' ' in m else '') for m in a['missed']) or '–'
        rows.append([r['rung'], r['criterion'], r['level'], r['status'], f"{a['reproduced']}/{a['of']}",
                     f'{ci[0]:.2f}–{ci[1]:.2f}' if ci[0] is not None else '–',
                     f"{r['salt']['reproduced']}/{r['salt']['of']}", f"{r['cocrystal']['reproduced']}/{r['cocrystal']['of']}", miss])
    table('tableS2', 'countsall', 'Observed states reproduced: every rung, criterion and level.',
          ['Rung', 'Criterion', 'Level', 'Status', 'All', '95% interval', 'Salts', 'Cocrystals', 'Missed'], rows,
          '95% intervals: Clopper–Pearson, descriptive (the crystals are not a random sample). Status as in Table 2 of the '
          'main text. Not counted in any row: the optional nitrate salt (no results) and BMA·DNBA (no acid–base hydrogen bond).')


def s3():
    src = (NUM / 'si' / 'profiles_all.md').read_text()
    for k, v in NAME.items():
        src = re.sub(rf'(?<![\w-]){k}(?![\w-])', v, src)
    (OUT / 'tableS3.md').write_text(src if '{#tbl:' in src else src + '\n\nTable: Periodic profile points of every crystal. {#tbl:profiles}\n')


def s4():
    src = (NUM / 'tables' / 'table4_cocrystal_margins.md').read_text()
    for k, v in NAME.items():
        src = re.sub(rf'(?<![\w-]){k}(?![\w-])', v, src)
    src = re.sub(r'\{#tbl:[^}]*\}', '{#tbl:cocmargins}', src)
    (OUT / 'tableS4.md').write_text(src)


def s5():
    rows = []
    for k in ORDER:
        r = [NAME[k]]
        for m in ('repeat', 'sc', 'gas'):
            r += [t(f'x.{k}.R4a_single.{m}'), t(f'x.{k}.R4a_coll_Eint2.{m}'), t(f'x.{k}.R4a_coll_LR.{m}')]
        r += [t(f'x.{k}.R3') if not F[f'x.{k}.site_shell_empty']['value'] else '= R1f',
              t(f'x.{k}.R4') if not F[f'x.{k}.site_shell_empty']['value'] else '= R4a', t(f'x.{k}.n_site_shell')]
        rows.append(r)
    table('tableS5', 'rungs', 'Molecular rungs with the three charge models.',
          ['Crystal', 'R4a ss REPEAT', 'coll. E~int~/2', 'coll. mean', 'R4a ss sc', 'coll. E~int~/2', 'coll. mean',
           'R4a ss gas', 'coll. E~int~/2', 'coll. mean', 'R3', 'R4', 'Shell molecules'], rows,
          f'{DPT}. R4a ss: the pair in the field, single site; coll.: every pair transferred, estimator E~emb~ − E~int~/2 (the '
          'plan) and the mean of the forward and reverse vertical gaps (added). REPEAT: charges fitted to the periodic PBE0 '
          'potential; sc: self-consistent in-crystal CHELPG; gas: gas-phase CHELPG. R3, R4: transfer-site shell, REPEAT field.')


def s6():
    rows = []
    for k in ORDER:
        r = [NAME[k]]
        for m in ('pair_repeat', 'pair_sc', 'pair_gas', 'site_repeat'):
            rv = (S[k].get('reverse') or {}).get(m) or {}
            r += [fmt(rv.get('embedded_single_site_kJ')), fmt(rv.get('reverse_kJ')), fmt(rv.get('lambda_kJ'), sign=False)]
        rows.append(r)
    table('tableS6', 'reverse', 'Forward and reverse vertical gaps.',
          ['Crystal', 'REPEAT fwd', 'rev', 'λ', 'sc fwd', 'rev', 'λ', 'gas fwd', 'rev', 'λ', 'cluster fwd', 'rev', 'λ'], rows,
          f'{MRG}. fwd: the pair transferred in the P0 (observed) environment; rev: the P0 pair in the P4 environment with the P4 '
          'charges; λ = (fwd − rev)/2. Cluster: the transfer-site cluster in the REPEAT field, five salts.')


def s7():
    rows = []
    for k in SALTS:
        ws = (S[k].get('checks') or {}).get('whole_shell_corrected') or {}
        sg = -1   # salts: summary stores other - observed; dE_PT = -(that)
        r3w, r4w = ws.get('R3'), ws.get('R4')
        rows.append([NAME[k], t(f'x.{k}.R3'), fmt(sg * r3w) if r3w is not None else 'not obtained' if k == 'brsalt' else '–',
                     t(f'x.{k}.R4'), fmt(sg * r4w) if r4w is not None else '–'])
    table('tableS7', 'wholeshell', 'Transfer-site shell against the whole first shell (salts).',
          ['Crystal', 'R3 site', 'R3 whole', 'R4 site', 'R4 whole'], rows,
          f'{DPT}. Whole first shell: every molecule hydrogen-bonded to any atom of the pair, locally dense basis plus a '
          '[def2-TZVP − locally dense] correction from the site cluster, where the cluster held at most 150 atoms. BMA–DNB: '
          'the ionic whole-shell cluster has a negative HOMO–LUMO gap in vacuum, so R3 was not obtained; R4 was computed in the '
          'field only.')


def s8():
    rows = []
    for k in SALTS:
        rows.append([NAME[k], t(f'x.{k}.split.field'), t(f'x.{k}.split.shell'), t(f'x.{k}.split.nonadditivity'),
                     t(f'x.{k}.split.field_share'), t(f'x.{k}.decomp.shell'), t(f'x.{k}.decomp.outer'),
                     t(f'x.{k}.decomp.electrostatic'), t(f'x.{k}.decomp.beyond_electrostatic'), t(f'x.{k}.decomp.cross')])
    table('tableS8', 'split', 'The planned field/shell split and the disjoint decomposition (salts).',
          ['Crystal', 'Field', 'Shell', 'Order dependence', 'Field share', 'Shell R3 − R1f', 'Outer R4 − R3',
           'Field R4a − R1f', 'Beyond R4 − R4a', 'Cross'], rows,
          f'{DPT}. Field, shell, order dependence, field share: the split fixed in the plan (averages over both orders); it is '
          'reported only where the order dependence is smaller than both terms, which holds for none. Shell R3 − R1f to cross: '
          'the disjoint decomposition added afterwards (single site, vertical, REPEAT).')


def s9():
    rows = []
    for k in ORDER:
        rows.append([NAME[k], t(f'x.{k}.cycle.pbe0.R1f'), t(f'x.{k}.cycle.pbe0.R4_minus_R1f'), t(f'x.{k}.cycle.pbe0.T1_minus_R4'),
                     t(f'x.{k}.cycle.pbe0.Nv_minus_T1'), t(f'x.{k}.cycle.pbe0.R5_minus_Nv'), t(f'x.{k}.cycle.pbe0.R5'),
                     t(f'x.{k}.closure.T1v_pbe'), t(f'x.{k}.closure.T2_pbe'), t(f'x.{k}.closure.image_T2_minus_T1_pbe')])
    table('tableS9', 'closure', 'The periodic closure cycle.',
          ['Crystal', 'R1f', 'R4 − R1f', 'T1 − R4', 'N~vert~ − T1', 'R5 − N~vert~', 'R5', 'T1v (PBE)', 'T2 (PBE)', 'T2 − T1 (PBE)'],
          rows, f'{DPT}; PBE0 unless stated. T1: one pair of the cell at its P4 geometry, all else at P0; T1v: the same with only '
          'the proton moved; T2: T1 in the doubled cell; N~vert~: every proton moved vertically. The steps sum to R5.')


def factor(k):
    """stored 'other - observed' -> dE_PT: +1 for cocrystals, -1 for salts (checked against summary.json itself)"""
    f = S[k]['fixed_ion_minus_neutral']['R1f'] / S[k]['R1f']['PBE0']
    assert abs(abs(f) - 1) < 1e-9 and (f > 0) == (S[k]['observed'] == 'cocrystal'), k
    return round(f)


def s10():
    rows = []
    for k in ORDER:
        bd = S[k]['budget']
        for q in ('margin', 'P4r', 'R5_P4', 'dE_env'):
            b = bd.get(q)
            if not b:
                continue
            f = 1 if q in ('margin', 'P4r') else factor(k)      # margins stay in the margin sign; R5 and dE_env as dE_PT
            lo, hi = sorted(f * x for x in b['envelope'])
            comps = '; '.join(f"{c.replace('_', ' ')}: " + ', '.join(fmt(f * x) for x in v.values())
                              for c, v in b['components'].items() if v)
            rows.append([NAME[k], {'margin': 'margin at P4r', 'P4r': 'P4r − P0', 'R5_P4': 'R5 at P4', 'dE_env': 'ΔE~env~'}[q],
                         fmt(f * b['central']), f"{fmt(lo)} to {fmt(hi)}", fmt(b['rss'], sign=False), comps])
    table('tableS10', 'budget', 'Error budgets.', ['Crystal', 'Quantity', 'Central', 'Envelope', 'rss', 'Components (shift from central)'],
          rows, f'Margin at P4r and P4r − P0: {MRG}. R5 at P4 and ΔE~env~: {DPT}. Central: PBE0 + MP2 in the X-ray cell '
          '(ΔE~env~: PBE0). Envelope: the '
          'central value plus each shift alone and the combinations computed directly (an estimate). rss: root sum of squares '
          'over the largest shift of each component. Components not computed are omitted.')


def s11():
    rows = []
    for k in COC:
        for lev in ('PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ'):
            z = (((S[k].get('nuclear') or {}).get('zpe_frozen_frame') or {}).get('H') or {}).get(lev)
            if not z:
                continue
            rows.append([NAME[k], lev, fmt(z['dZPE_kJ']), fmt(z['E0_difference_kJ']), z['kind'],
                         f"{100 * z['p_far_side_P0_frame']:.1f}%"])
    table('tableS11', 'nqe', 'Zero-point and proton-delocalisation checks (cocrystals, H).',
          ['Crystal', 'Level', 'ΔZPE', 'E~0~ difference', 'Kind', 'P0 frame beyond barrier'], rows,
          f'{MRG}. Frozen-frame proton problem: the proton moved along D···A in the fixed P0 and P4r frames, a 1D sinc DVR on the '
          'corrected potential plus the two bends from the partial Hessian projected perpendicular to D···A. Kind: a '
          'ZPE-corrected margin when both frames hold a localised ground state, otherwise a bound.')


def s12():
    rows = []
    for k in ORDER:
        ck = S[k].get('checks') or {}
        pv, pp = ck.get('runner_up_minus_chosen_vertical') or {}, ck.get('runner_up_minus_chosen_periodic') or {}
        if not pv and not pp:
            continue
        rows.append([NAME[k], ' / '.join(fmt(pv.get(x)) for x in ('PBE0_D3', 'MP2', 'SCS_MP2', 'wB97X_D3BJ')),
                     fmt(pp.get('runner_minus_chosen_pbe_kJ')), fmt(pp.get('runner_minus_chosen_pbe0_kJ'))])
    table('tableS12', 'runner', 'Runner-up hydrogen bonds.', ['Crystal', 'Pair: PBE0 / MP2 / SCS-MP2 / ωB97X', 'Periodic PBE', 'Periodic PBE0'],
          rows, 'Runner-up minus chosen bond, vertical, kJ mol⁻¹ per pair (< 0: transfer along the runner-up is easier). Periodic: '
          'the runner-up proton placed at r~eq~ from its acceptor in the P0 cell.')


def s13():
    rows = []
    for k in ORDER:
        rows.append([NAME[k], t(f'x.{k}.pcm.eps_x'), t(f'x.{k}.pcm.eps_eq'), t(f'x.{k}.pcm.eps78'),
                     t(f'x.{k}.pcm.radii1.1.eps78'), t(f'x.{k}.pcm.radii1.3.eps78')])
    table('tableS13', 'pcm', 'The pair in a dielectric continuum.', ['Crystal', 'ε at sign change', 'ε reproducing R5', 'ε = 78',
          'ε = 78, radii × 1.1', 'ε = 78, radii × 1.3'], rows,
          f'{DPT}. IEF-PCM around the R1f pair, PBE0-D3(BJ)/def2-TZVP, cavity radii × 1.2 unless stated. – : no ε between 1 and 78.')


def s14():
    rows = []
    for k in ORDER:
        ck = S[k].get('checks') or {}
        pc = ck.get('pair_cc') or {}
        rows.append([NAME[k], t(f'x.{k}.shift.k.P4'), t(f'x.{k}.shift.k.P4r'), t(f'x.{k}.shift.q.P4r'),
                     fmt(pc.get('delta_T_P4_kJ')), fmt(pc.get('delta_T_P4r_kJ')), t(f'x.{k}.shift.cell_vc.P4')])
    table('tableS14', 'checks', 'Mesh, cell and δ(T) checks.', ['Crystal', 'k, P4', 'k, P4r', 'q, P4r', 'δ(T), P4', 'δ(T), P4r', 'cell, P4'],
          rows, 'Shifts in kJ mol⁻¹ per pair. k: PBE, one step denser mesh minus production; q: PBE0, exchange q mesh = k mesh minus '
          '1 × 1 × 1; δ(T): [CCSD(T) − MP2]/def2-SVP of the pair, P4 or P4r minus P0; cell: variable-cell relaxation. Signs as '
          'stored (other − observed for P4r, ΔE~PT~ for P4).')


def s15():
    pr = SUM['polymorph_brsalt_brcoc']
    rows = [[lab, fmt(pr.get(key))] for lab, key in (('PBE', 'pbe_kJ_per_fu'), ('PBE, dense k', 'pbe_dense_k_kJ_per_fu'),
                                                    ('PBE0, q = 1 × 1 × 1', 'pbe0_kJ_per_fu'), ('PBE0, q = k mesh', 'pbe0_qmesh_kJ_per_fu'),
                                                    ('PBE0, dense k, q = k mesh', 'pbe0_dense_k_qmesh_kJ_per_fu'))]
    table('tableS15', 'polymorph', 'The polymorph pair BMA–DNB (salt) and BMA·DNBA (cocrystal).', ['Level', 'E(BMA·DNBA) − E(BMA–DNB)'],
          rows, 'kJ mol⁻¹ per formula unit, static lattice energies in the X-ray cells; > 0: the salt is lower. No pair, monomer, '
          'zero-point or thermal correction.')


if __name__ == '__main__':
    for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15):
        f()
    print('wrote', ', '.join(sorted(p.name for p in OUT.glob('tableS*.md'))))
