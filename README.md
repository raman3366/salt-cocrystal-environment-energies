# Environment energies of proton transfer in thirteen acid–base crystals: data and code

Data and code of the study "Quantifying the crystal-environment contribution to proton transfer in acid–base salts
and cocrystals" (C. Muthuselvi, Anantha Raman Lakshmipathi, D. Saravanakkumar, Venkatramanan Varadharajan; manuscript
submitted to Computational and Theoretical Chemistry).

For thirteen acid–base crystals with ΔpKa between −1.4 and +2.4 (nine salts, four cocrystals) and a polymorph pair,
every crystal was taken through one ladder of models: the acid–base pair alone at the crystal geometry (R1f), the pair
in the electrostatic field of the lattice (R4a), the pair with its first hydrogen-bonded shell (R3, R4) and the periodic
crystal at PBE0 with MP2, SCS-MP2 and ωB97X-D3(BJ) corrections on the pair (R5). The environment term is
ΔE_env = R5 − R1f.

## Contents

| Path | What it holds |
|---|---|
| `optc/crystals.json` | the set: CIF, CCDC/COD numbers, ΔpKa with sources and quality flags, transferring bond, observed state and its evidence |
| `optc/cif/` | the CIFs (COD) with `MD5SUMS` |
| `optc/structures/` | the cells as built (`cell.json`, `start.extxyz`) |
| `optc/src/` | calculation and analysis code: `cell.py` (structures), `qe.py` (periodic, Quantum ESPRESSO), `mol.py` (molecular rungs, PySCF), `embed.py` and `charges.py` (lattice field), `analyse.py` (classes, counts, budgets) |
| `dft/periodic_qe.py` | Quantum ESPRESSO input and output helpers used by `qe.py` and `analyse.py` |
| `data/raw/<crystal>/` | inputs and outputs of every calculation (743 tasks): `runs/<tag>/` for Quantum ESPRESSO, `mol/` for PySCF, the JSON summaries; host names are replaced by machine A–G; the ESP cubes are not included (`qe.py esp` regenerates them) |
| `data/tasks.json` | every task with its command line |
| `optc/results/summary.json`, `summary.md` | the analysis of `data/raw` (`analyse.py`), the source of every number of the paper |
| `paper/ctc/numbers/` | `facts.py` (every number of the paper with its source key), the table scripts and their outputs |
| `paper/ctc/figures/scripts/`, `paper/_shared/figkit/` | the figure scripts |
| `paper/ctc/si/make_si_tables.py` | the tables of the Supplementary material |
| `plan/` | the plan as committed on 18 Oct 2026, before the production calculations, and the list of its later additions |

## Reproducing the numbers

    python3 optc/src/analyse.py --results data/raw --out optc/results/summary      # summary.json / summary.md
    python3 paper/ctc/numbers/facts.py                                              # facts.json (checks the summary md5)

The calculations themselves need Quantum ESPRESSO 7.5 (conda-forge `qe=7.5=h19104ac_2`), PySCF 2.14.0 with geomeTRIC
1.1.1 and the SSSP 1.3.0 efficiency and SG15 ONCV 1.2 pseudopotentials whose file names and md5 sums are in
`optc/src/qe.py`; see `ENVIRONMENT.md`.

## Licences

Code: MIT (`LICENSE`). Data, structures derived from the CIFs and results: CC BY 4.0 (`LICENSE-data`). The CIFs are
redistributed from the Crystallography Open Database; cite the original structure papers (listed in the article).

## Citation

Please cite the article and this deposit (`CITATION.cff`): https://doi.org/10.5281/zenodo.23608941 (v1.0.0; all
versions: https://doi.org/10.5281/zenodo.23608940). The earlier single-crystal study of 4-sulfamoylanilinium dihydrogen
phosphate is deposited at https://doi.org/10.5281/zenodo.23335995.

The code and analysis scripts were written and run with an AI tool (Claude, Anthropic) under the authors' direction.
