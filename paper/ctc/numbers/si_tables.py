#!/usr/bin/env python3
"""SI profile tables of the CTC series paper (all 13 crystals), from data/facts.json only (run facts.py first).

    tools/paper-env/bin/python paper/ctc/numbers/si_tables.py

si/profile_<key>.md + .csv per crystal and si/profiles_all.md (all 13 in dpKa order, for the SI include).  Every point of
the periodic path: held distance, delta, D...A (raw merged results), E(point) - E(P0) at every level (P0 = observed
state; a profile energy, not a dE_PT); under each table dE_PT at P4, the class and the margin per level.
"""
from _ctc import F, HERE, KEYS, LEVN, v, t, env, bond, write, used_report

SI = HERE / 'si'
LEV = ['PBE', 'PBE0', 'PBE50', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ']
parts = []
for k in KEYS:
    x = f'x.{k}'
    pts = v(f'p.{k}.points')
    salt_back = ['P4r'] if 'P4r' in pts and f'p.{k}.P4r.PBE0' not in F else []     # salts: P4r relaxed back, PBE only
    main = sorted([p for p in pts if p not in salt_back], key=lambda p: v(f'p.{k}.{p}.delta'))
    rows = []
    for p in main + salt_back:
        q = f'p.{k}.{p}'
        tg = F[f'{q}.target']
        held = f'{bond(tg["target_type"])} {tg["text"]}' if tg['value'] is not None else '–'
        rows.append([p, tg['kind'], held, t(f'{q}.delta'), t(f'{q}.DA')] + [t(f'{q}.{l}') for l in LEV])
    lev_lines = []
    for l in LEV:
        if f'{x}.R5.P4.{l}' not in F or v(f'{x}.R5.P4.{l}') is None:
            continue
        cls = f'{v(x + ".class.full." + l)} state (full profile) / {v(x + ".class.plan." + l)} state (P0–P4)' if f'{x}.class.full.{l}' in F else '–'
        mg = (f'margin {t(x + ".margin." + l)} at {v(x + ".margin." + l + ".point")}' if f'{x}.margin.{l}' in F
              else (f'no other-side minimum, crossing energy {t(x + ".crossing." + l)}' if f'{x}.crossing.{l}' in F else ''))
        lev_lines.append(f'{LEVN[l]}: ΔE_PT(P4) {t(x + ".R5.P4." + l)}' + (f'; class {cls}' if cls != '–' else '') + (f'; {mg}' if mg else ''))
    back = (' P4r: the unconstrained relax started from P4 returned to the observed side (PBE only).' if salt_back else '')
    notes = (f'{v(x + ".name")}; observed state: {v(x + ".observed")}; transferring bond {bond(v(x + ".bond"))}; '
             f'D···A (X-ray) {t(x + ".DA.xray")} Å; k mesh shift at P4 {t(x + ".shift.k.P4")} kJ mol⁻¹ (ΔE_PT). '
             'Energies: E(point) − E(P0) in kJ mol⁻¹ per transferring pair (P0 = observed-state end of the path, fixed X-ray '
             'cell); δ = d(D–H) − d(H···A), < 0 on the observed side; held: the distance constrained at the point (D–H on the '
             'path, A–H at P4); kind: path P0–P4, extra (X: d(D–H) held at the value in the name; r: re-seeded from P4r), '
             f'relaxed (P4r, unconstrained); D···A: mean over the copies in the cell.{back} Per level: '
             + '; '.join(lev_lines) + '.')
    md = write(SI / f'profile_{k}', ['Point', 'Kind', 'Held (Å)', 'δ (Å)', 'D···A (Å)'] + [LEVN[l] for l in LEV], 'lllrr' + 'r' * len(LEV),
               rows, f'Periodic proton-transfer profile of {k} (kJ mol⁻¹).', f'profile-{k}', notes)
    parts.append(md.read_text())
(SI / 'profiles_all.md').write_text('\n'.join(parts))
used_report(SI / 'facts_used.txt')
print('wrote', len(parts), 'profile tables ->', SI)
