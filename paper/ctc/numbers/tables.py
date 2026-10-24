#!/usr/bin/env python3
"""Main-text tables of the CTC series paper, from data/facts.json only (run facts.py first).

    tools/paper-env/bin/python paper/ctc/numbers/tables.py

tables/table1_set, table2_counts, table3_energies, table4_cocrystal_margins (.md pandoc pipe table + .csv), and
tables/facts_used.txt (every fact key printed).  Energies in kJ/mol per transferring pair; dE_PT = E(ion pair) -
E(neutral pair) except where a column says margin (other state - observed; for a cocrystal the two coincide).
"""
from _ctc import F, HERE, KEYS, COC, ORDER, LEVN, v, t, n_of, env, bond, write, used_report

T = HERE / 'tables'
CORR = ['PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ']

# ---------------------------------------------------------------------------------------------- Table 1: the set
MARK = {'compilation': 'c', 'off-axis': 'o', 'site value': 's', 'alternative': 'a'}
rows = []
for k in ORDER:
    p = f'x.{k}'
    ids = ' / '.join(x for x in [t(f'{p}.ccdc', ''), t(f'{p}.cod', '')] if x) if v(f'{p}.ccdc') else f'– / {t(p + ".cod")}^n^'
    fl = ''.join(MARK[f] for f in v(f'{p}.dpka_flags'))
    sgn = lambda key: ('+' if v(key) > 0 else '') + t(key)
    dpk = f'{sgn(p + ".dpka")} [{sgn(p + ".dpka_lo")}, {sgn(p + ".dpka_hi")}]'
    rows.append([k, v(f'{p}.name'), ids, v(f'{p}.set'), v(f'{p}.role'),
                 dpk + (f'^{fl}^' if fl else ''), v(f'{p}.observed'), v(f'{p}.proton_evidence'),
                 bond(v(f'{p}.bond')) if k != 'brcoc' else '– (ranking only)',
                 t(f'{p}.DA.xray') if k != 'brcoc' else '–', f'{t(p + ".Z")} / {t(p + ".n_fu")}', t(f'{p}.nat')])
notes = (f'ΔpKa = pKa(BH⁺) − pKa(HA), aqueous, 25 °C, with the range of the values found in brackets; fixed in the dated plan '
         f'(commit {v("s.plan_commit")}) for every member, pcplut and nbaetpy added later (set "extension"). '
         f'^c^ Compilation value, no primary measurement located. ^o^ Off-axis: {v("x.sulfamate.dpka_off_axis_reason")}. '
         f'^s^ Macroscopic value; the site constant gives {t("x.carboxy.dpka_site")}. '
         f'^a^ Alternative +{t("x.perchlorate.dpka_alt")} [{t("x.perchlorate.dpka_alt_lo")}, {t("x.perchlorate.dpka_alt_hi")}]. '
         '^n^ No CCDC deposition number found. D···A: X-ray, transferring bond. Z of the CIF / transferring pairs in the cell '
         '(formula units); atoms: the periodic cell. brcoc has no acid–base bond (polymorph ranking only); the optional '
         'nitrate salt has no results and is not listed.')
write(T / 'table1_set', ['Key', 'Crystal', 'CCDC / COD', 'Set', 'Role', 'ΔpKa', 'Observed', 'Evidence for the observed state',
                         'Transferring bond', 'D···A (Å)', 'Z / n', 'Atoms'],
      'lllllrllllrr', rows, 'The crystal set.', 'set', notes)

# ---------------------------------------------------------------------------------------------- Table 2: counts by rung
RUNG = lambda s: s.replace('pre-registered', 'plan').replace(' (P4 - P0 > 0)', '').replace('P0-P4', 'P0–P4')
rows, i = [], 0
while f's.count.{i:02d}.rung' in F:
    p = f's.count.{i:02d}'
    miss = sorted(set(F[f'{p}.all']['missed']))
    rows.append([RUNG(v(f'{p}.rung')), RUNG(v(f'{p}.criterion')), LEVN.get(v(f'{p}.level'), v(f'{p}.level')), v(f'{p}.status'),
                 n_of(f'{p}.all'), n_of(f'{p}.salt'), n_of(f'{p}.cocrystal'), n_of(f'{p}.core'), ', '.join(miss) or '–'])
    i += 1
notes = ('Status: plan = fixed in the dated plan before the production calculations; extension = the relaxed other-state point '
         'P4r and the extra profile points (19–20 Oct 2026); added = criteria added afterwards. The molecular rungs are scored '
         'by their sign at P4 (two configurations), R5 by the minimum of its profile. Non-trivial: crystals with an empty '
         'transfer-site shell (R3 = R1f, R4 = R4a) left out; symmetric site: such a crystal counted with its symmetric-site '
         'cluster where that shell is not empty. Resolved: class observed at PBE0 + MP2 and the error-budget envelope excludes 0 '
         '(the margin at P4r where an other-side minimum exists, else ΔE_PT at P4). Not counted in any row: '
         + '; '.join(v('s.not_counted')) + '. Core: the 9 crystals of the core set with a transferring bond.')
write(T / 'table2_counts', ['Rung', 'Criterion', 'Level', 'Status', 'All', 'Salts', 'Cocrystals', 'Core', 'Missed'],
      'llllrrrrl', rows, 'Observed states reproduced, by rung.', 'counts', notes)

# ---------------------------------------------------------------------------------------------- Table 3: energies
rows = []
for k in KEYS:
    p = f'x.{k}'
    triv = v(f'{p}.site_shell_empty')
    res = 'resolved' if v(f'{p}.resolved') else 'unresolved^b^'
    rows.append([k, v(f'{p}.set'), v(f'{p}.observed'), t(f'{p}.R1f.PBE0'), t(f'{p}.R4a_single.repeat'),
                 '(= R1f)' if triv else t(f'{p}.R3'), '(= R4a)' if triv else t(f'{p}.R4'),
                 t(f'{p}.R5.P4.PBE'), t(f'{p}.R5.P4.PBE0'),
                 f'{t(p + ".R5.P4.PBE0+MP2")} {env(p + ".budget.R5_P4.lo", p + ".budget.R5_P4.hi")}',
                 f'{t(p + ".dE_env.PBE0")} ± {t(p + ".budget.dE_env.rss")}', res])
unres = [k for k in KEYS if not v(f'x.{k}.resolved')]
mtxt = '; '.join(f'{k} {t(f"x.{k}.budget.margin.central")} {env(f"x.{k}.budget.margin.lo", f"x.{k}.budget.margin.hi")}' for k in unres)
notes = ('ΔE_PT = E(ion pair) − E(neutral pair), kJ mol⁻¹ per transferring pair, PBE0 unless stated; > 0: the neutral pair is '
         'lower. R1f: the isolated pair at the crystal geometry; R4a: the pair in the lattice field (REPEAT charges, observed '
         'environment, single site); R3: the pair with its first hydrogen-bonded shell; R4: R3 in the field; (= R1f), (= R4a): '
         'empty transfer-site shell. R5: the periodic crystal at P4 (fixed X-ray cell) with PBE, PBE0 and PBE0 + MP2 pair '
         'correction, the last with its error-budget envelope (central value and each check one at a time plus the computed '
         'combinations; an estimate: the checks were run at different settings and are added). ΔE_env = R5 − R1f (PBE0) ± the '
         'root sum of squares of its budget; R5 = R1f + ΔE_env by definition. '
         f'^b^ The budget envelope of the classification margin at P4r (other state − observed, PBE0 + MP2) spans 0: {mtxt}.')
write(T / 'table3_energies', ['Crystal', 'Set', 'Observed', 'R1f', 'R4a', 'R3', 'R4', 'R5 PBE', 'R5 PBE0', 'R5 PBE0 + MP2 [envelope]',
                              'ΔE_env ± rss', 'Resolution'],
      'lllrrrrrrrrl', rows, 'Proton-transfer energies ΔE_PT on every rung (kJ mol⁻¹).', 'energies', notes)

# ---------------------------------------------------------------------------------------------- Table 4: cocrystal checks
COLS = [('delta_T', 'δ(T)'), ('in_field', 'In field'), ('relaxed_vacuum', 'Relaxed cells'), ('relaxed_in_field', 'Relaxed cells, in field'),
        ('k', 'k mesh'), ('q', 'q mesh'), ('QZVP', 'def2-QZVP'), ('TZVPD', 'def2-TZVPD')]
rows, foot = [], []
for k in COC:
    for lev in CORR:
        p, x = f't4.{k}.{lev}', f'x.{k}'
        zk = f'{p}.value.ZPE_frozen'
        if f'{p}.ZPE_frozen.bound' in F:
            zpe = f'≥ {t(p + ".ZPE_frozen.bound")}'
        else:
            zpe = t(zk)
        verdict, budget = v(f'{p}.verdict'), ''
        if lev == 'PBE0+MP2':
            q = 'margin' if f'{x}.budget.margin.central' in F else 'P4r'
            budget = f'{env(f"{x}.budget.{q}.lo", f"{x}.budget.{q}.hi")}; {v(x + ".resolution.verdict")}'
            if v(f'{x}.resolution.quantity') != 'margin':
                budget += f' (rule applied to ΔE_PT at P4: {t(x + ".budget.R5_P4.central")} {env(x + ".budget.R5_P4.lo", x + ".budget.R5_P4.hi")})'
        mk = f'{x}.margin.{lev}'
        if mk not in F:
            foot.append(f'{k} {LEVN[lev]}: no other-side minimum (P4r − P0 is not a margin)')
        elif v(mk + '.point') != 'P4r':
            foot.append(f'{k} {LEVN[lev]}: the margin is at {v(mk + ".point")} ({t(mk)})')
        rows.append([k, LEVN[lev], t(f'{p}.xray')] + [t(f'{p}.value.{g}') for g, _ in COLS] + [zpe, env(f'{p}.env_lo', f'{p}.env_hi'), verdict, budget or '–'])
h = lambda k: ', '.join(f'{c} {t(f"t4.{c}.PBE0+MP2.shift.ZPE_harmonic_{k}")}' for c in COC)
notes = ('Each entry: P4r − P0 (the relaxed other-state point minus the observed state, kJ mol⁻¹ per transferring pair, other state '
         '− observed; for a cocrystal this is also ΔE_PT) with one check applied. X-ray cell: the production value. δ(T): CCSD(T) − '
         'MP2 of the pair (def2-SVP; MP2 level only). In field: the pair correction computed in the lattice field (REPEAT), X-ray '
         'cell. Relaxed cells: both states in their own variable-cell relaxed cells, vacuum or in-field pair correction. k, q: '
         'denser k mesh (PBE) and the exact-exchange q mesh equal to the k mesh (PBE0). def2-QZVP (MP2 only) and def2-TZVPD: the '
         'pair correction in the larger basis. ZPE: frozen-frame proton zero-point correction (H); ≥: a lower bound (no localized '
         'other-state level), not used in the envelope. Envelope: the X-ray-cell value, each check one at a time and the combined '
         'relaxed-cell checks (in field; + δ(T) + k + q), at that level; the checks at PBE0 + SCS-MP2 and PBE0 + ωB97X-D3(BJ) are '
         'formed with the rule of the PBE0 + MP2 budget (asserted against it in facts.py). Under every check: the sign of the entries '
         'of the row. Budget: the full error-budget envelope, which also holds the level spread and the PBE / PBE50 anchors, and the '
         'resolution verdict (mppcp and nbaetpy: P4r − P0 is not a margin, the resolution rule uses ΔE_PT at P4). '
         f'Harmonic partial-Hessian ZPE (PBE, H D A), shift: {h("HDA")}. – : not computed. ' + '; '.join(foot) + '.')
write(T / 'table4_cocrystal_margins', ['Crystal', 'Level', 'X-ray cell'] + [c for _, c in COLS] + ['ZPE', 'Envelope', 'Under every check', 'Budget (PBE0 + MP2)'],
      'llrrrrrrrrrrlll', rows, 'Cocrystal margins at every level under every check (kJ mol⁻¹).', 'cocrystal-margins', notes)

used_report(T / 'facts_used.txt')
print('wrote', *sorted(p.name for p in T.glob('table*')))
