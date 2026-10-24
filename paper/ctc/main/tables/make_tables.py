"""Main-text tables of the CTC paper from the numbers module (paper/ctc/numbers): display names, the observed-state
evidence in short form, and the count rows that carry the argument (the full tables are in the Supplementary material).

    python3 paper/ctc/main/tables/make_tables.py        # -> paper/ctc/main/tables/table{1,2,3}.md
"""
import csv, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = HERE.parents[1] / 'numbers'
F = json.loads((NUM / 'data' / 'facts.json').read_text())
NAME = {'cyano': 'CNAPH', 'saph': 'SAPH', 'carboxy': 'CXAPH', 'chloro': 'CLAPH', 'anis': 'MOAPH', 'perchlorate': 'SAPC',
        'sulfamate': 'SASF', 'dapsone': 'DDS·DNBA', 'mppcp': 'MP·PCP', 'pcplut': 'LUT·PCP', 'nbaetpy': 'EPY·NBA',
        'brsalt': 'BMA–DNB', 'clsalt': 'CMA–DNB', 'brcoc': 'BMA·DNBA'}
CHEM = {'cyano': '2-cyanoanilinium dihydrogen phosphate', 'saph': '4-sulfamoylanilinium dihydrogen phosphate',
        'carboxy': '2-carboxyanilinium dihydrogen phosphate', 'chloro': '2-chloroanilinium dihydrogen phosphate',
        'anis': '2-methoxyanilinium dihydrogen phosphate', 'perchlorate': '4-sulfamoylanilinium perchlorate',
        'sulfamate': '4-sulfamoylanilinium sulfamate', 'dapsone': 'dapsone · 3,5-dinitrobenzoic acid',
        'mppcp': '2-methylpyridine · pentachlorophenol', 'pcplut': '2,6-dimethylpyridine · pentachlorophenol',
        'nbaetpy': '3-ethylpyridine · 4-nitrobenzoic acid', 'brsalt': '4-bromo-2-methylanilinium 3,5-dinitrobenzoate',
        'clsalt': '4-chloro-2-methylanilinium 3,5-dinitrobenzoate',
        'brcoc': '4-bromo-2-methylaniline · 3,5-dinitrobenzoic acid'}
REF = {'cyano': 'zhang20092', 'saph': 'muthuselvi2014crystal', 'carboxy': 'benalicherif2007hydrogen',
       'chloro': 'balamurugan2010dihydrogen', 'anis': 'ouldabdellahi1995o', 'perchlorate': 'anitha20134',
       'sulfamate': 'wojnarska2018crystal', 'dapsone': 'smith20124', 'mppcp': 'steiner2000neutron',
       'pcplut': 'schmidtmann2008hydrogen', 'nbaetpy': 'rajput2017exploring', 'brsalt': 'jones2019living',
       'clsalt': 'jones2019living', 'brcoc': 'jones2019living'}
ORDER = [k for k in F['s.order']['value']]


def t(key):
    return F[key]['text']


def evidence(k):
    e = F.get(f'x.{k}.proton_evidence', {}).get('value') or ''
    temp = re.search(r'(\d+(?:\.\d+)?)\(\d+\) K|(\d+) K', e)
    T = (temp.group(1) or temp.group(2)) if temp else ''
    if 'ssNMR' in e:
        return f'¹⁵N–¹H ssNMR [@{REF[k]}]'
    if 'NPD' in e or 'neutron powder' in e:
        return 'neutron powder + X-ray, 100 K [@schmidtmann2014determining]'
    if e.startswith('neutron'):
        return f'neutron, {T} K'
    if 'constr' in e or 'riding' in e:
        return f'X-ray, {T} K, H riding'
    return f'X-ray, {T} K, H located'


def table1():
    rows = ['| Name | Crystal | CCDC | Set | Role | ΔpK~a~ | Observed | Evidence | Transferring bond | D···A (Å) |',
            '|:---|:---|:---|:---|:---|---:|:---|:---|:---|---:|']
    for k in ORDER:
        cc = F[f'x.{k}.ccdc']['value'] if f'x.{k}.ccdc' in F and F[f'x.{k}.ccdc']['value'] else '–'
        v = F[f'x.{k}.dpka']['value']
        dp = ('+' if v > 0 else '') + t(f'x.{k}.dpka')
        fl = F.get(f'x.{k}.dpka_flags', {}).get('value') or []
        mark = ''.join({'compilation': '^c^', 'off-axis': '^o^', 'site value': '^s^', 'alternative': '^a^'}[f] for f in fl)
        role = F.get(f'x.{k}.role', {}).get('value') or ''
        role = {'test': 'test', 'test (the paper\'s crystal)': 'test'}.get(role, role)
        if k == 'brcoc':
            rows.append(f'| {NAME[k]} | {CHEM[k]} [@{REF[k]}] | {cc} | core | polymorph of BMA–DNB | {dp}{mark} | cocrystal | '
                        'X-ray, 150 K | – (ranking only) | – |')
            continue
        bond = F[f'x.{k}.bond']['value'].replace("'", '′').replace('...', '···').replace('-', '–')
        rows.append(f"| {NAME[k]} | {CHEM[k]} [@{REF[k]}] | {cc} | {F[f'x.{k}.set']['value']} | {role} | {dp}{mark} | "
                    f"{F[f'x.{k}.observed']['value']} | {evidence(k)} | {bond} | {t(f'x.{k}.DA.xray')} |")
    rows += ['', 'Table: The crystal set, ordered by family and ΔpK~a~. {#tbl:set}', '',
             'ΔpK~a~ = pK~a~(BH^+^) − pK~a~(HA), aqueous, 25 °C, fixed in the dated plan before the production '
             'calculations. ^c^ Compilation value, no primary measurement located. ^o^ Off-axis: the sulfamic acid pK~a~ '
             'refers to N–H deprotonation, while the computed transfer protonates O. ^s^ Macroscopic value (site constant '
             '+1.03). ^a^ An alternative estimate is +17. Set: core (fixed in the plan), optional, extension (two negative '
             'controls added by a rule fixed before their calculations). D···A: X-ray structure, transferring bond. '
             'BMA·DNBA has no acid–base hydrogen bond and enters the polymorph ranking only.']
    (HERE / 'table1.md').write_text('\n'.join(rows) + '\n')


def table2():
    rows = list(csv.DictReader(open(NUM / 'tables' / 'table2_counts.csv')))
    want = [('R1f', 'sign at P4', 'PBE0'), ('R4a single site (field, REPEAT)', 'sign at P4', 'PBE0'),
            ('R4a collective (field, REPEAT, E_int/2)', 'sign at P4', 'PBE0'),
            ('R4a collective (field, REPEAT, LR mean)', 'sign at P4', 'PBE0'),
            ('R3 (shell), non-trivial', 'sign at P4', 'PBE0'), ('R4 single site, non-trivial', 'sign at P4', 'PBE0')] + \
        [('R5', 'class, plan profile (P0-P4)', lev) for lev in ('PBE', 'PBE0', 'PBE0 + MP2', 'PBE0 + SCS-MP2', 'PBE0 + ωB97X-D3(BJ)')] + \
        [('R5', 'class, full profile (P4r + extra points)', lev) for lev in ('PBE0 + MP2', 'PBE0 + ωB97X-D3(BJ)')] + \
        [('R5 corrected, resolved', None, 'PBE0 + MP2')]
    lab = {'R1f': 'R1f, isolated pair', 'R4a single site (field, REPEAT)': 'R4a, pair in the field, single site',
           'R4a collective (field, REPEAT, E_int/2)': 'R4a, collective, E~emb~ − E~int~/2',
           'R4a collective (field, REPEAT, LR mean)': 'R4a, collective, mean of both gaps',
           'R3 (shell), non-trivial': 'R3, pair + shell (non-empty shells)',
           'R4 single site, non-trivial': 'R4, R3 in the field (non-empty shells)'}
    head = list(rows[0].keys())
    out = ['| Rung | Criterion | Level | All | Salts | Cocrystals | Core 9 | Status | Missed |',
           '|:---|:---|:---|---:|---:|---:|---:|:---|:---|']
    for r_, c_, l_ in want:
        m = [r for r in rows if r[head[0]] == r_ and (c_ is None or r[head[1]] == c_) and r[head[2]] == l_]
        assert len(m) == 1, (r_, c_, l_, len(m))
        r = m[0]
        missed = r[head[-1]]
        for k, v in NAME.items():
            missed = re.sub(rf'\b{k}\b', v, missed)
        if r_ == 'R1f':
            missed = 'all 9 salts'
        missed = '–' if missed.strip() in ('-', '') else missed
        crit = {'sign at P4': 'sign at P4', 'class, plan profile (P0-P4)': 'class, P0–P4 profile',
                'class, full profile (P4r + extra points)': 'class, full profile'}.get(r[head[1]], 'class observed, envelope excludes 0')
        out.append(f"| {lab.get(r_, r_)} | {crit} | {r[head[2]]} | {r['All']} | {r['Salts']} | {r['Cocrystals']} | {r['Core']} | "
                   f"{r['Status']} | {missed} |")
    out += ['', 'Table: Observed states reproduced, by rung and criterion. {#tbl:counts}', '',
            'Status: plan, fixed before the production calculations; extension, profiles with the relaxed point P4r and extra '
            'points; added, criteria introduced afterwards. Non-empty shells: the nine salts (the four cocrystals have no '
            'molecule hydrogen-bonded to the transfer site). Core 9: the crystals fixed in the plan. All counts and their '
            'Clopper–Pearson intervals are in Table S2.']
    (HERE / 'table2.md').write_text('\n'.join(out) + '\n')


def table3():
    s = (NUM / 'tables' / 'table3_energies.md').read_text()
    for k, v in NAME.items():
        s = re.sub(rf'^\| {k} \|', f'| {v} |', s, flags=re.M)
    (HERE / 'table3.md').write_text(s)


if __name__ == '__main__':
    table1(); table2(); table3()
    print('wrote', ', '.join(p.name for p in sorted(HERE.glob('table*.md'))))
