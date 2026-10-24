# Numbers module of the CTC series paper

Every number the text, tables and SI print comes from `data/facts.json`, generated from the pinned
`optc/results/summary.json` (md5 `411d4c2ad29499c8961de1bae2fc5b02`; `facts.py` refuses any other file),
`optc/crystals.json` and `optc/structures/<key>/cell.json`. The merged raw results are read only for D···A along the
periodic profiles. Nothing is typed by hand, and no sign is converted by hand.

    P=tools/paper-env/bin/python                       # from the repository root
    $P paper/ctc/numbers/facts.py                      # -> data/facts.json (runs about 540 assertions)
    $P paper/ctc/numbers/tables.py                     # -> tables/table1_set ... table4_cocrystal_margins (.md + .csv)
    $P paper/ctc/numbers/si_tables.py                  # -> si/profile_<key>.md/.csv, si/profiles_all.md

## Citing a fact

1. Find the key, for example with
   `$P -c "import json; F=json.load(open('paper/ctc/numbers/data/facts.json')); print({k: F[k]['text'] for k in F if k.startswith('s.R1f.PBE0.all')})"`.
2. Print the fact's `text` field, which is the value at its `precision`, rounded half to even, with a true minus sign
   and `+` on signed energies. For a count, print `text_of` (for example `9/9`). Do not re-round `value` yourself.
3. Check the `sign` field before you write a sentence around the number (see `_meta.signs`):
   - `dE_PT`: E(ion pair) − E(neutral pair), kJ mol⁻¹ per transferring pair. This is the paper's convention.
   - `margin`: other state − observed state. Use it only for a classification margin and its checks. For a cocrystal
     it equals ΔE_PT.
   - `lowering`: −ΔE_env, so "the crystal lowers the ion pair by …".
   - `toward_observed`: the two closure ranges, which the plan quotes in that sense.
   - `profile`: E(point) − E(P0) for the SI profiles. This is not a ΔE_PT.
4. Optionally leave `<!-- fact: key -->` after the number in the draft. The numbers audit matches printed numbers by
   value and unit:
   `python3 paper/_shared/tools/numbers_audit.py <dir> --facts paper/ctc/numbers/data/facts.json`.

## Key layout

| prefix | content |
|---|---|
| `x.<key>.` | per crystal: metadata, ΔpKa and its flags, R1f / R4a (single, collective E_int/2 and LR, three charge models) / R3 / R4 / R5 at every level, class, margin, ΔE_env, compensation (−ΔE_env/R1f), budget (central, lo, hi, rss, every component, combined), resolution, reverse gaps, closure cycle, k / q / basis / δ(T) / relaxed-cell checks, ZPE, continuum, field/shell split, decomposition |
| `p.<key>.<point>.` | periodic profile points: delta, held distance, D···A, E − E(P0) at every level |
| `t4.<key>.<level>.` | Table 4: P4r − P0 under each check at each corrected level, envelope, verdict |
| `s.` | series: counts, ranges (`.min`, `.max` with `at`, `.n` with `members`), counts by rung (`s.count.NN.*` = summary.json counts_table row NN, core column added; named shortcuts `s.count.R5_full_MP2.core` …), reverse-gap counts, closure ranges, ΔpKa slopes and correlations, G2b, polymorph pair |

`_meta.plan_check` lists the series statements of `optc/CTC_REWRITE_PLAN.md` section 1 next to the computed values.
`tables/facts_used.txt` and `si/facts_used.txt` list every key the tables print.
