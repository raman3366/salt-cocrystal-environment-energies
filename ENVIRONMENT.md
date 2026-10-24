# Software environment

Periodic calculations: Quantum ESPRESSO 7.5, conda-forge build `qe=7.5=h19104ac_2` with `openmpi=5.0.11`, Python 3.12.
Molecular calculations: Python 3.11 with `pyscf=2.14.0`, `geometric=1.1.1`, `simple-dftd3`, `dftd3-python` (conda-forge)
and `pyscf-dispersion==1.5.0` (pip). Analysis: Python 3 with NumPy and SciPy; figures: Matplotlib 3.11 and RDKit.

Pseudopotentials: SSSP PBE efficiency 1.3.0 (Materials Cloud) for the PBE-D3(BJ) relaxations at 70/560 Ry, SG15 ONCV
1.2 for the PBE / PBE0 single points at 100/400 Ry. File names and md5 sums are in `optc/src/qe.py` (`SSSP`, `SG15`).

Settings (details in the article and in `plan/`): PBE-D3(BJ) two-body, fixed X-ray cells, Monkhorst–Pack meshes with
n_i = ceil(9.0 Å / |a_i|), forces below 5e-4 Ry/bohr at P0; PBE0-D3(BJ) single points with an exchange q mesh of
1 x 1 x 1 (q = k mesh as a check); PySCF PBE0-D3(BJ), MP2, SCS-MP2 and wB97X-D3(BJ) with def2-TZVP (density fitting,
frozen core), CCSD(T) with def2-SVP.

Machines: Linux cloud instances with 12-48 virtual CPUs; the run logs in `data/raw/*/runs.jsonl` name them machine A-G.
