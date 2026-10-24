#!/usr/bin/env python3
"""
Periodic DFT of the SAPH crystal (model M9: the Pc cell, Z = 2, 54 atoms) with Quantum ESPRESSO.

Why: the molecular models (gas, IEF-PCM) ask where the N1-H1A...O5 proton sits in an isolated ion pair, and
with Delta-pKa ~ 0 the answer there is expected to depend on the environment.  The crystal is the environment
the X-ray actually measured, so the decisive DFT test is the full lattice: every hydrogen bond, the Madelung
field and the neighbouring cations and anions at once.  We relax the atoms at the X-ray cell from two starts,
    (i)  as deposited (proton on N1: anilinium + H2PO4-, the salt), and
    (ii) H1A moved onto O5 (0.98 A from O5 on the N1->O5 line, in both formula units so Pc is kept):
         neutral sulfanilamide + H3PO4, the cocrystal alternative,
plus two supporting starts:
    (iir)   the same H1A move made on the coarsely relaxed salt (relax_i_pre) instead of the X-ray structure
            (review addition, 28 Sep 2026).  Start (ii) keeps every other H at its X-ray position (X-H 0.78-0.93 A),
            so its first BFGS steps are dominated by X-H stretching (the sandbox moved O4-H4A 0.80 -> 1.27 A in
            step 1, with H1A still on O5); (iir) tests the O5-H basin with only the one proton out of place.
    (iifix) start (ii) with H1A of both formula units fixed in space (if_pos 0 0 0) where (ii) puts it: the
            constrained neutral form, which gives an energy for it when (ii) falls back to the salt.  H1A is fixed
            in space, not held 0.98 A from O5 (pw.x allows distance constraints only with damp/FIRE dynamics, not
            BFGS), so N1 and O5 still move: read d(N1-H1A), d(H1A-O5) of the result from its 'proton' block.
            RESULT (VM, 28 Sep 2026): the constraint did not hold.  Pc has a floating origin along a and c, so BFGS
            slid every other atom by (0.09, 0, -0.67) A until N1 sat at the fixed H1A: proton on N1 (N1-H1A 1.068,
            H1A...O5 1.660 A, 165.7 deg), = relax_i translated (RMSD 0.10 A) with H1A 0.23 A off along b (the one
            direction the glide pins).  E(iifix) - E(i) = +80 meV/cell is a bent N-H...O in a salt, NOT a neutral form.
    (iirfix) the constrained neutral form done properly (review fix, 28 Sep 2026): the production relax_i geometry
            with H1A moved 0.98 A from O5 on the N1->O5 line (as iir, both formula units), then H1A AND O5 of both
            formula units fixed in space (if_pos 0 0 0).  Holding O5 as well keeps O5-H1A at 0.98 A and pins the
            floating origin; N1 cannot take the proton back without N1...O5 ~2.0 A.  Everything else relaxes at the
            production setting (no coarse pre-stage: the base is already relaxed).  An upper bound to the energy of
            the neutral (O5-H) form in the lattice; 'analyse' checks that the proton stayed on O5.
and report E(ii) - E(i), whether (ii) falls back to (i), the proton metric delta = d(N1-H1A) - d(H1A...O5), all
nine CIF hydrogen bonds (D...A) including the very short O4-H4A...O5 (2.500 A), and the heavy-atom RMSD against
the X-ray.  The X-ray places the O-H hydrogens with DFIX 0.82 A restraints, so DFT supplies the O-H positions.

Referee additions (29 Sep 2026; paper/main/review/ref_chem.md M2, M3):
    (profile) constrained proton-transfer profile ('profile' stage).  d(N1-H1A) of BOTH formula units is held at
            1.10, 1.25, 1.40, 1.55, 1.70 A with pw.x CONSTRAINTS (type 'distance'); every other coordinate, O5 and
            H1A's two other degrees of freedom included, relaxes at the production setting (SSSP 70/560 Ry, k 2x1x1,
            fixed X-ray cell, PBE-D3(BJ)).  This replaces the single frozen point iirfix (O5 and H1A fixed in space,
            O5-H1A 0.98 A) that the referee objected to.
            * Optimiser.  QE 7.5 refuses CONSTRAINTS with BFGS: PW/src/input.f90 (iosys, tag qe-7.5, line 1299)
              "IF ( lconstrain .AND. ( lbfgs .OR. lmovecell ) ) CALL errore( 'iosys', 'constraints only with
              fixed-cell dynamics', 1 )" (checked again on the Pi with pw.x 7.5, 29 Sep 2026: pw.x stops with that
              message).  INPUT_PW.def documents CONSTRAINTS for ion_dynamics 'damp' (and MD); 'fire' also applies the
              constraints in the code (dynamics_module.f90 fire(): remove_constr_force / remove_constr_vec /
              check_constraint) but is not documented for it.  Default here: ion_dynamics = 'damp' (projected-Verlet
              quick-min, dynamics_module.f90 proj_verlet, with the BFGS-preconditioned force of force_precond),
              because (a) it is the documented route, (b) it removes the centre-of-mass displacement of each step
              when no atom is fixed, i.e. the floating origin of Pc (along a and c) that defeated iifix cannot drift,
              which FIRE does not do, and (c) its BFGS preconditioning.  SHAKE (check_constraint, real masses,
              tolerance 1e-6 bohr) restores the distances after every step.  --dyn fire is available.  Convergence
              in both: (E_prev - E) < etot_conv_thr AND max |constrained force| < forc_conv_thr (the force with its
              component along the constraint gradients removed).  QE's conv_thr "upscale" exists only for BFGS, so
              the SCF threshold is fixed at 1e-9 Ry (force noise ~1e-5 Ry/bohr).  Thresholds: forc_conv_thr 1e-3
              Ry/bohr, etot_conv_thr 1e-4 Ry (the QE defaults; relax_i used 5e-4 / 1e-5): the profile spans
              hundreds of meV per cell and the residual-force energy error is ~1 meV/cell or less, an upper-bound bias.
            * Starts (--seed; chosen so that all points can run at once): 'interp' (default) = linear interpolation of
              every atom between the relaxed salt (relax_i, d(N1-H1A) 1.063 A) and the relaxed constrained neutral
              form (relax_iirfix, 1.616 A) at w = (d - 1.063)/(1.616 - 1.063), clipped to [0, 1], with H1A then put
              exactly d from N1 along the interpolated N1->H1A direction.  Both end structures are fully relaxed in
              everything but the proton (and, for iirfix, O5), so each start is already relaxed to first order in
              the rest of the lattice; the points are independent (no chain that serialises them or carries errors
              from one point to the next).  Other seeds: 'i' (relax_i with H1A moved on the N1->O5 line, the
              "linearly moved proton") and 'prof<ddd>' (the converged geometry of another profile point with H1A
              moved along its N1->H1A direction: chains and the hysteresis check, e.g. 1.40 seeded from 1.55).
            * Reported per point ('profile_analyse' -> profile.json): E - E(relax_i) in meV/cell and kJ/mol per
              formula unit (both protons move: per f.u. = per cell / 2), delta, N1...O5, O5-H1A, the N-H...O angle,
              O4-H4A...O5, P1-O5 and the pressure (fixed cell), for both formula units; and the slope dE/dd from the
              constraint force: at a constrained minimum the raw forces are F_H = -lambda u, F_N = +lambda u
              (u = unit vector N1->H1A), and dE/dd = lambda = (F_N - F_H).u / 2 per formula unit (envelope theorem).
              The sign of the slope at the largest d says whether the energy still falls there (a neutral minimum
              further out) without another point.
            PBE0-D3(BJ) at every point: the 'hybrid' stage with --start prof<ddd> (SG15 100/400 Ry, the same as
            relax_i and iirfix): E_PBE0(d) - E_PBE0(i) = [E_PBE-D3/SSSP(d) - E(i)] + [correction(d) - correction(i)],
            correction = E(PBE0-D3) - E(PBE-D3) with SG15 at the same geometry (single points at the PBE geometries).
    (iir1)  single-formula-unit transfer control ('relax --start iir1'): relax_i with H1A moved 0.98 A from O5 on the
            N1->O5 line in formula unit 1 ONLY (breaks the glide: pw.x finds no symmetry, 1 operation), then an
            unconstrained BFGS relaxation at the production setting, no pre-stage.  Does the proton return to N1
            when its neighbour along the glide keeps its salt proton?  Reported per formula unit in profile.json.

Level of theory and the choices behind it:
  * PBE + Grimme D3 with Becke-Johnson damping (vdw_corr='dft-d3', dftd3_version=4).  Two-body only
    (dftd3_threebody=.false.): QE switches the three-body ATM term ON by default, whereas the molecular
    B3LYP-D3(BJ) of this project (pyscf-dispersion 'd3bj', atm=False) and Grimme's standard D3(BJ) leave it out
    unless asked for.  (Review check, 28 Sep 2026: QE's D3 energy of the sandbox relax_ii geometry equals
    simple-dftd3 PBE-D3(BJ) two-body to 2.5e-7 Ry; with ATM it would differ by 0.018 Ry.)  PBE, not a hybrid,
    because plane-wave exact exchange costs one to two orders of magnitude more (see the 'estimate' stage).
    PBE's delocalisation error over-stabilises shared protons AND biases acid-base crystals towards the
    proton-transferred (salt) form: GGAs predict spurious proton transfer in organic acid-base co-crystals
    (LeBlanc et al., Angew. Chem. Int. Ed. 57 (2018) 14906, doi:10.1002/anie.201809381, Crossref-checked).  A
    PBE result "the salt is the only minimum" is therefore biased in the direction of its own conclusion; the
    manuscript must say so, and a hybrid single point on relax_i vs relax_iifix is the check ('hybrid' stage).
  * Hybrid check ('hybrid' stage, added 28 Sep 2026): PBE0-D3(BJ) SCF single points (no forces, no re-relaxation)
    at the PBE-D3(BJ) final geometries relax_i and relax_iirfix (the salt and the constrained neutral form; relax_iifix
    is a salt, see above, and relax_ii is structurally = relax_i: both only as controls), on the k grid
    of those relaxes (2x1x1), input_dft='pbe0' (exx_fraction 0.25).  Production = a hybrid CORRECTION with SG15
    norm-conserving potentials: E(PBE0-D3) - E(PBE-D3), both with SG15 at 100/400 Ry (80/320 Ry as the cutoff
    check), added to the PBE-D3(BJ)/SSSP difference of the relaxes (best_estimate in periodic_qe.json).  Why not
    PBE0 directly with the SSSP (US/PAW) potentials of the relaxes (possible with --pp sssp, but not in a burst):
      - Cost.  With US/PAW every pair density of every ACE build is augmented in G space (PW/src/us_exx.f90
        addusxx_g + newdxx_g): a loop over all atoms x nh^2 projector pairs x all EXX-grid G vectors per pair;
        sum(nh^2) = 3176 here (P, S have 18 projectors), 283847 G at ecutfock 280 Ry, 16928 pairs (92 x 92 bands
        x 2 k) -> ~2.4e14 flop per ACE build.  On the Pi (28 Sep 2026) the first ACE build at 20/160 Ry, Gamma,
        2 ranks had not finished after 49 CPU-min per rank (67 min wall); scaled to the VM that is >= 25 min per
        ACE build on 14 ranks, i.e. >= 1.6 h per single point with 4 outer loops.  Norm-conserving potentials have
        no augmentation: the exact exchange is FFTs only.
      - ecutfock cannot be raised for US/PAW: PW/src/setup.f90 resets it to MIN(4 ecutwfc, ecutrho) ("ecutfock
        not valid for US/PAW, ignored"), so the Fock grid (280 Ry) is always coarser than the density grid.
      - Real-space augmentation (tqr=.true.) would be ~10x faster (Pi: 5 CPU-min per ACE build instead of > 49)
        but is broken for EXX in QE 7.5 when the Fock grid differs from the dense grid (always the case for
        US/PAW, above): realus.f90 real_space_q normalises qq_at with the DENSE-grid point count (line 600,
        omega/(dfftp%nr1*nr2*nr3)) also when it builds the EXX-grid tables, and qq_at enters the overlap S.  Pi
        test (20/160 Ry): "integrated real space q too different ... 2.396 for atom type 6" on the EXX grid
        (0.002 on a dense grid of the same size), density charge 185.15 instead of 184, pw.x stops ("charge is
        wrong").  Not used.
    SG15 ONCV v1.2 (Schlipf & Gygi, Comput. Phys. Commun. 196 (2015) 36) were made for hybrid-functional plane-wave
    work and have no nonlinear core correction (pw.x warns that NLCC is inconsistent with hybrid XC; PseudoDojo
    has NLCC on C, N, O, P, S); PBE-generated, as usual for PBE0.  Norm-conserving: ecutrho = 4 ecutwfc = ecutfock,
    so neither the density nor the pair densities are truncated.  The PBE-D3 companion SCF uses the same
    potentials, cutoff and k grid, so the correction is free of pseudopotential and basis differences to first
    order; the SG15-vs-SSSP difference of the PBE-D3 Delta E itself is reported as a diagnostic.
    What QE 7.5 does, read from its source at tag qe-7.5 (gitlab.com/QEF/q-e, 28 Sep 2026):
      - D3 parameters follow the functional: PW/src/input.f90 dftd3_iosys() passes get_dft_short() ('PBE0')
        through dft-d3/dftd3_qe.f90 dftd3_xc() (lower case; only bp/blyp/b3lyp/hse/pw86pbe/olyp are renamed) to
        dft-d3/core.f90 setfuncpar(), whose version-4 (BJ) entry "pbe0" is s6 = 1, a1 = 0.4145, s8 = 1.2177,
        a2 = 4.8593 (Grimme, Ehrlich & Goerigk, J. Comput. Chem. 32 (2011) 1456), vs PBE s6 = 1, a1 = 0.4289,
        s8 = 0.7875, a2 = 4.4407.  dftd3_threebody=.false. is kept, so it is two-body D3(BJ) as before.  Checked
        numerically on the Pi (28 Sep 2026): iteration 1 of a PBE0 run is a PBE iteration, and its energy minus
        that of a PBE input at relax_i / relax_iifix (20/160 Ry) is 0.01362242 / 0.01358873 Ry, vs simple-dftd3
        D3(pbe0) - D3(pbe) two-body 0.01362241 / 0.01358871 Ry (differences 1e-8, 2e-8 Ry); QE's pbe0 set equals
        simple-dftd3's named 'pbe0'.  'analyse' repeats the check for every production run (d3_check).
      - ACE (adaptively compressed exchange, Lin, J. Chem. Theory Comput. 12 (2016) 2242) is the default
        (ace=.true.); it is written into the input explicitly.
      - ecutfock = MIN(ecutrho, 4 ecutwfc), the code default (exx_iosys; INPUT_PW.def still says "ecutrho"),
        written explicitly: = ecutrho for SG15 (no truncation), forced for US/PAW (above).
      - q mesh for the Fock operator: nqx = 1x1x1 (the documented default is the k grid; 1x1x1 is the usual
        choice for an insulator and ~2x cheaper here); --nq 2x1x1 is the optional check (a = 4.80 A is short).
      - exxdiv_treatment 'gygi-baldereschi' with x_gamma_extrapolation (defaults, written explicitly).
      - conv_thr 1e-6 Ry (the QE default) is also the EXX outer-loop threshold (dexx < conv_thr): 0.014 meV per
        cell, three orders below the ~1 meV scale of the differences discussed; each extra decade of dexx costs
        one more ACE build, so it is not tightened further.  The PBE-D3 companion SCF uses 1e-8 Ry.
    E(iirfix) - E(i) is reported in periodic_qe.json['hybrid']: PBE-D3(BJ)/SSSP (relaxes), PBE-D3 and PBE0-D3 with
    SG15, the hybrid correction and its split into the D3-parameter and exact-exchange parts, and the best
    estimate PBE-D3/SSSP + correction (hybrid_summary).
  * SSSP PBE efficiency v1.3.0 pseudopotentials (Materials Cloud, doi:10.24435/materialscloud:f3-ym); the
    largest recommended cutoffs of H C N O P S there are 60/480 Ry (H, N), so the cutoff test spans 50/400,
    60/480, 70/560 Ry (dual 8 throughout, as SSSP recommends for these US/PAW potentials).
  * Cell fixed at the X-ray values (294 K): only ionic positions relax (BFGS).  The residual PBE-D3 pressure of
    the X-ray cell is printed (tstress) in the convergence runs, as a diagnostic.
  * Fixed occupations (the crystal is a colourless insulator), spin-restricted, charge 0.
  * Symmetry is kept (QE detects Pc from the coordinates): both formula units carry the same proton.  A proton
    moved in one formula unit only is the iir1 control (29 Sep 2026): pw.x then finds no symmetry by itself (the
    glide maps H1A onto a different position), so nothing has to be switched off.

Stages (each is resumable: a finished pw.x run is never repeated; an interrupted relax restarts from the last
geometry it printed):
    build         write the 54-atom cell and both starting structures, check them against the CIF (no pw.x)
    converge      SCF+forces+stress at the X-ray geometry: cutoffs 50/400, 60/480, 70/560 Ry (starts i and ii,
                  k 3x2x2) and k grids 2x1x1, 3x2x2, 4x2x2, 5x2x3 (start i, 50/400 Ry); picks the cheapest setting
                  within 1 meV/atom and 1e-3 Ry/bohr of the most converged one, with the cutoff never below the
                  SSSP-recommended 60/480 Ry -> convergence.json
    relax         fixed-cell BFGS from one start (--start i|ii|iir|iifix|iirfix, or both = i and ii; iir needs
                  relax_i_pre, iirfix needs relax_i and has no pre-stage):
                  a coarse pre-relaxation (50/400 Ry, 2x1x1, QE default thresholds; --pre-only runs just that,
                  before 'converge' has finished), then the production relax at the chosen setting (forces
                  < 5e-4 Ry/bohr)
    analyse       energies, E(ii)-E(i) (and iir, iifix, iirfix), delta, H-bonds, O4-H4A...O5, heavy-atom RMSD, key bonds,
                  where each coarse pre-relaxation ended -> periodic_qe.json
    estimate      cost of a hybrid (PBE0/HSE) single point at the production setting (nothing is run)
    hybrid_probe  optional timing probe for 'estimate': PBE0 at Gamma, stopped by max_seconds
    hybrid        PBE0-D3(BJ) SCF single point at the final geometry of relax_<start> (--start i|ii|iifix|iirfix;
                  'both' = i then iirfix) on that relax's k grid, with a PBE-D3(BJ) companion SCF; --pp sg15 (default,
                  --ecut default 100/400) or sssp; --nq, --ecutfock, --conv-thr, --max-seconds, --exx-maxstep.
                  Run once per structure (two structures side by side in a burst); 'analyse' then writes
                  E(iifix) - E(i) at PBE0-D3(BJ) next to PBE-D3(BJ) into periodic_qe.json.
                  Resumable per structure: a converged run is reused; an unfinished one is kept as .out.partN
                  and re-run from scratch (pw.x's own EXX restart files live on the NVMe scratch, which is wiped
                  whenever the VM stops, so they are not relied on).
                  --start prof<ddd>[_<seed>] / iir1: the same at a converged profile point or at the iir1 control.
    profile       constrained N1-H1A profile (see "Referee additions"): --d 1.40 [1.55 ...] (default all five),
                  --seed interp|i|prof<ddd>, --dyn damp|fire, --forc-thr, --etot-thr, --max-seconds (pw.x stops
                  cleanly, the next call resumes from the last geometry), --dry-run (writes inputs/relax_prof*.in and
                  the seed report, runs nothing).  One run per point: runs/relax_prof<ddd>[_<seed>]/.
    profile_analyse  profile.json: PBE-D3(BJ) profile, PBE0-D3(BJ) profile (where the hybrid points exist), slopes,
                  the hysteresis check (same d from different seeds), the iir1 control and a verdict on a neutral
                  minimum; periodic_qe.json is not touched.

Usage (VM, from ~/cqc-sfa; one MPI rank per physical core, one OpenMP thread per rank; see
ops/vm/bursts/burst_periodic.json and burst2.json for the burst layouts):
    tools/qe/bin/python dft/periodic_qe.py build
    tools/qe/bin/python dft/periodic_qe.py converge --np 24
    tools/qe/bin/python dft/periodic_qe.py relax --start i --np 12      # and --start ii, concurrently
    tools/qe/bin/python dft/periodic_qe.py analyse && tools/qe/bin/python dft/periodic_qe.py estimate
    tools/qe/bin/python dft/periodic_qe.py hybrid --start i --np 14     # and --start iirfix, concurrently
    tools/qe/bin/python dft/periodic_qe.py profile --d 1.40 --np 12 --ecut 70/560 --kgrid 2x1x1   # one task per point
    tools/qe/bin/python dft/periodic_qe.py hybrid --start prof140 --np 12                          # after that point
    tools/qe/bin/python dft/periodic_qe.py relax --start iir1 --np 12 --ecut 70/560 --kgrid 2x1x1  # control
    tools/env/bin/python dft/periodic_qe.py profile_analyse                                        # -> profile.json
pw.x is launched as  mpirun -np N --bind-to none pw.x -nk <pools> -nd 1  with OMP_NUM_THREADS=1 and BLAS pinned
to one thread; the pools are the largest divisor of N not above the irreducible k count (8 for 3x2x2).
MPI layout of the hybrid single points (2 irreducible k points at 2x1x1, so -nk 2 for any even N; the rest of
the ranks split the plane waves and the EXX grid, which for SG15 is the dense grid, nr3 = 128 planes at 400 Ry):
    one run alone         N = 24 -> 2 pools x 12 ranks (one rank per physical core of the 24-core VM)
    two runs side by side N = 14 each (28) next to a 12-thread PySCF job (40 threads in all), or N = 20-22 each
                          (40-44) when nothing else runs: 2 pools x 10-11 ranks; ranks beyond 24 in total sit on
                          hyperthreads and add little.  -nd 1 (92 bands: parallel diagonalisation does not pay).
An odd N falls back to fewer pools (choose_pools), so keep N even.
Options --out, --scratch, --pseudo, --pw, --mpirun override the defaults below (used by the sandbox tests).
The environment variables QE_NP / QE_NK / QE_OMP set the defaults of --np / --nk / --omp (QE_NSTEP caps the BFGS
steps, for smoke tests only).  --geom-from DIR takes the relax_* runs for 'hybrid' from another results directory
(sandbox tests: tiny-setting hybrids in a scratch --out on the production geometries).
"""
import argparse, datetime, hashlib, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CIF = ROOT / 'e-70-0o997-sup1.cif'
OUT = ROOT / 'molecules/sfa/results/periodic_qe'
QE = ROOT / 'tools/qe'
SCRATCH = Path('/mnt/scratch/periodic_qe') if os.path.isdir('/mnt/scratch') else ROOT / 'scratch/periodic_qe'

RY_EV = 13.605693122994

# SSSP PBE efficiency 1.3.0: file name and md5 (both from SSSP_1.3.0_PBE_efficiency.json, record rcyfm-68h65).
PSEUDOS = {
    'H': ('H.pbe-rrkjus_psl.1.0.0.UPF', 'f52b6d4d1c606e5624b1dc7b2218f220'),
    'C': ('C.pbe-n-kjpaw_psl.1.0.0.UPF', '5d2aebdfa2cae82b50a7e79e9516da0f'),
    'N': ('N.pbe-n-radius_5.UPF', '16739722b17309cd8fe442a2ace49922'),
    'O': ('O.pbe-n-kjpaw_psl.0.1.UPF', '0234752ac141de4415c5fc33072bef88'),
    'P': ('P.pbe-n-rrkjus_psl.1.0.0.UPF', '8b930f418f0a4573dca56cd030ffe088'),
    'S': ('s_pbe_v1.4.uspp.F.UPF', '88d86576ff6df21479756cfb9bdac1df'),
}
MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.06}   # IUPAC conventional

# Convergence series.  The cutoff series runs at the smallest k grid of the k series (cutoff and k-point errors
# are nearly independent for an insulator); the k series at the smallest cutoff.  2x1x1 is an extra cheap point
# that shows how fast the k error falls off.  Targets: 1 meV/atom on the total energy, 1e-3 Ry/bohr on forces.
ECUTS = [(50, 400), (60, 480), (70, 560)]
KGRIDS = [(2, 1, 1), (3, 2, 2), (4, 2, 2), (5, 2, 3)]
K_FOR_ECUT = (3, 2, 2)
TARGET_E_MEV_ATOM = 1.0
TARGET_F_RYB = 1e-3
# Production cutoff floor (review fix, 28 Sep 2026): never below the largest SSSP-recommended cutoff of these six
# potentials (SSSP_1.3.0_PBE_efficiency.json: H 60/480, C 45/360, N 60/480, O 50/400, P 30/240, S 35/280 Ry),
# whatever the 1 meV/atom test says.  The library's recommendation also covers stress and phonons, and the
# observable here is a hydrogen position (H is the hardest potential).  50/400 stays the coarse pre-relax cutoff.
SSSP_MIN_ECUT = (60, 480)

# Proton transfer start (ii): H1A put 0.98 A from O5 on the N1->O5 line.
D_OH_START = 0.98

# Constrained N1-H1A profile (referee M2/M3, 29 Sep 2026; see "Referee additions" in the docstring).
PROFILE_D = (1.10, 1.25, 1.40, 1.55, 1.70)      # d(N1-H1A) targets in A, both formula units
PROFILE_FORC_THR = 1e-3                          # Ry/bohr, on the constrained forces (QE default)
PROFILE_ETOT_THR = 1e-4                          # Ry (QE default)
PROFILE_CONV_THR = 1e-9                          # Ry; fixed (no BFGS upscale in damped dynamics)
BOHR_A = 0.529177210903
CORE_STARTS = ('i', 'ii', 'iir', 'iifix', 'iirfix')   # the starts of periodic_qe.json (analyse / hybrid_summary)
_PROF = re.compile(r'^prof(\d{3})(?:_(i|prof\d{3}))?$')

# Hybrid single points ('hybrid' stage; see the level-of-theory notes above).  The D3(BJ) parameters are those of
# QE 7.5's dft-d3/core.f90 setfuncpar (version 4) for 'pbe' and 'pbe0', recorded in periodic_qe.json.
HYBRID_CONV_THR = 1e-6
# SG15 ONCV norm-conserving PBE potentials v1.2 (Schlipf & Gygi, Comput. Phys. Commun. 196 (2015) 36; made for hybrid
# functionals, no nonlinear core correction; valence as SSSP here: H 1, C 4, N 5, O 6, P 5, S 6 -> 184 electrons), from
# sg15_oncv_upf_2020-02-06.tar.gz (www.quantum-simulation.org, tarball md5 0d72299c3962fb9de17a22d81cb79734); file md5s
# recorded on the Pi, 28 Sep 2026.  Installed on the VM by ops/vm/setup_qe.sh into tools/qe/pseudo_sg15.
PSEUDOS_SG15 = {
    'H': ('H_ONCV_PBE-1.2.upf', 'bf21e08b90ac4ce7d0149ac4e8138190'),
    'C': ('C_ONCV_PBE-1.2.upf', '1a5f83a7b1f58d24996abe00ed223ac6'),
    'N': ('N_ONCV_PBE-1.2.upf', 'c233fa8df51455e992258b7c3dee8a9f'),
    'O': ('O_ONCV_PBE-1.2.upf', 'e36c48deb68f4e355fffe4602be2c42c'),
    'P': ('P_ONCV_PBE-1.2.upf', '0688a67b2b263c6b42d3d97c42745426'),
    'S': ('S_ONCV_PBE-1.2.upf', 'f9d020296ee6110346977bfd66d424f6'),
}
HYBRID_SG15_ECUT = (100, 400)       # production cutoff of the SG15 hybrid correction; 80/320 is the convergence check
D3BJ_QE = {'pbe': {'s6': 1.0, 'a1': 0.4289, 's8': 0.7875, 'a2': 4.4407},
           'pbe0': {'s6': 1.0, 'a1': 0.4145, 's8': 1.2177, 'a2': 4.8593}}
KJMOL_PER_EV = 96.485332

# Key covalent bonds reported for X-ray vs DFT: C1-N1 and the P-O set are the structural evidence for the salt
# (C1-N1 1.463 A; P-O 1.498/1.503 vs 1.558/1.577 A in the X-ray).
KEY_BONDS = [('C1', 'N1'), ('S1', 'N2'), ('S1', 'O1'), ('S1', 'O2'), ('S1', 'C4'),
             ('P1', 'O3'), ('P1', 'O4'), ('P1', 'O5'), ('P1', 'O6'), ('O3', 'H3A'), ('O4', 'H4A')]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def note(out, text):
    """Timestamped line in <out>/STAGES.md (human-readable log, like the molecular pipeline's)."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'STAGES.md', 'a') as f:
        f.write(f'- {datetime.datetime.now():%Y-%m-%d %H:%M}  {text}\n')
    print(f'[periodic_qe] {text}', flush=True)


def atomic_json(path, obj):
    tmp = Path(f'{path}.{os.getpid()}.tmp')          # per-process name: two analyse tasks may overlap in a burst
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------------------------
# Crystal: the deposited CIF expanded to the full cell, with every geometric quantity defined by CIF symmetry codes
# ----------------------------------------------------------------------------------------------------------------
class Crystal:
    """The Pc cell from the CIF.  Atom order: the 27 asymmetric-unit sites under symop 1 (x,y,z), then the same
    27 under symop 2 (x,-y,z+1/2), each wrapped into [0,1).  Structures are arrays of fractional coordinates in
    this order.  Distances are taken between explicit lattice images fixed once on the X-ray structure from the
    CIF symmetry codes, and a relaxed structure is first unwrapped against the X-ray one, so an atom that QE
    moves across a cell face is still measured to the same partner."""

    def __init__(self, cif=CIF):
        import gemmi
        self.block = gemmi.cif.read(str(cif)).sole_block()
        ss = gemmi.make_small_structure_from_block(self.block)
        # rows = lattice vectors a, b, c in Angstrom (gemmi orthogonalisation: a along x, b in the xy plane)
        self.lat = np.array(ss.cell.orth.mat.tolist()).T
        self.cellpar = [ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma]
        # symmetry operators in the CIF's own order, so that CIF symmetry codes ("2_554") index them correctly
        xyz_ops = [s.strip("'\" ") for s in self.block.find_values('_space_group_symop_operation_xyz')]
        self.ops = [gemmi.Op(t) for t in xyz_ops]
        self.op_str = xyz_ops
        self.asym = [(s.label, s.element.name, np.array([s.fract.x, s.fract.y, s.fract.z])) for s in ss.sites]
        self.nasym = len(self.asym)
        self.labels, self.syms, f0 = [], [], []
        for k, op in enumerate(self.ops):
            for lab, el, x in self.asym:
                y = np.array(op.apply_to_xyz(list(x)))
                f0.append(y - np.floor(y))
                self.labels.append(lab if k == 0 else f'{lab}_{k + 1}')
                self.syms.append(el)
        self.F0 = np.array(f0)
        self.nat = len(self.syms)
        self.index = {lab: i for i, lab in enumerate(self.labels)}

    # geometry primitives ---------------------------------------------------------------------------------------
    def unwrap(self, F):
        d = F - self.F0
        return self.F0 + d - np.round(d)

    def cart(self, F):
        return np.asarray(F) @ self.lat

    def sym_frac(self, label, op_index=0, t=(0, 0, 0)):
        """Fractional position of asym-unit site `label` under symop op_index plus lattice translation t (X-ray)."""
        x = dict((l, x) for l, _, x in self.asym)[label]
        return np.array(self.ops[op_index].apply_to_xyz(list(x))) + np.array(t, float)

    def locate(self, label, frac):
        """(cell atom index, lattice image) of the atom of that asym label sitting at `frac` in the X-ray cell."""
        for i, lab in enumerate(self.labels):
            if lab.split('_')[0] != label:
                continue
            n = frac - self.F0[i]
            if np.allclose(n, np.round(n), atol=1e-4):
                return i, np.round(n).astype(int)
        raise KeyError(f'{label} at {frac} not found in the cell')

    def point(self, F, site):
        i, n = site
        return self.cart(self.unwrap(F)[i] + n)

    def dist(self, F, s1, s2):
        return float(np.linalg.norm(self.point(F, s1) - self.point(F, s2)))

    def angle(self, F, s1, s2, s3):
        v1, v2 = self.point(F, s1) - self.point(F, s2), self.point(F, s3) - self.point(F, s2)
        return float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))

    @staticmethod
    def parse_code(code):
        """CIF symmetry code 'n_klm' -> (0-based symop, translation (k-5, l-5, m-5)); '.' or '?' = identity."""
        if code in ('.', '?', '', None):
            return 0, (0, 0, 0)
        op, tr = (code.split('_') + ['555'])[:2]
        return int(op) - 1, tuple(int(c) - 5 for c in tr)

    def compose(self, copy, label, code=None):
        """Site (index, image) of `label` transformed by the CIF code, then by symop `copy` (formula unit copy)."""
        k, t = self.parse_code(code)
        x = self.sym_frac(label, k, t)
        y = np.array(self.ops[copy].apply_to_xyz(list(x)))
        return self.locate(label, y)

    # the CIF's own tables ------------------------------------------------------------------------------------------
    def hbonds(self):
        """The nine H-bonds of the deposited _geom_hbond loop (Table 1 of the 2014 paper)."""
        import gemmi
        tab = self.block.find('_geom_hbond_', ['atom_site_label_D', 'atom_site_label_H', 'atom_site_label_A',
                                               'site_symmetry_A', 'distance_DH', 'distance_HA', 'distance_DA',
                                               'angle_DHA'])
        rows = []
        for r in tab:
            rows.append({'D': r[0], 'H': r[1], 'A': r[2], 'code_A': r[3],
                         'cif_DH': gemmi.cif.as_number(r[4]), 'cif_HA': gemmi.cif.as_number(r[5]),
                         'cif_DA': gemmi.cif.as_number(r[6]), 'cif_DHA': gemmi.cif.as_number(r[7])})
        return rows

    def hbond_sites(self, hb, copy=0):
        return (self.compose(copy, hb['D']), self.compose(copy, hb['H']), self.compose(copy, hb['A'], hb['code_A']))

    # starting structures -------------------------------------------------------------------------------------------
    def start_ii(self, d_oh=D_OH_START, base=None, copies=None):
        """H1A of both formula units put d_oh from its O5 acceptor on the N1->O5 line (keeps Pc), in the X-ray
        structure or, with `base`, in another structure of this cell (start iir: the relaxed salt).  copies=[0]:
        formula unit 1 only (start iir1, the single-transfer control; breaks the glide)."""
        base = self.F0 if base is None else self.unwrap(np.asarray(base, float))
        F = base.copy()
        hb = next(h for h in self.hbonds() if (h['D'], h['H'], h['A']) == ('N1', 'H1A', 'O5'))
        inv = np.linalg.inv(self.lat)
        for copy in (range(len(self.ops)) if copies is None else copies):
            sN, sH, sO = self.hbond_sites(hb, copy)
            pN, pO = self.point(base, sN), self.point(base, sO)
            u = (pO - pN) / np.linalg.norm(pO - pN)
            pH = pO - d_oh * u
            i, n = sH
            F[i] = pH @ inv - n            # store in the H atom's own (X-ray) image
        return F

    # constrained profile -------------------------------------------------------------------------------------------
    def nh_sites(self):
        """[(site N1, site H1A, site O5)] of the N1-H1A...O5 bond in each formula unit (copy 0 = x,y,z)."""
        hb = next(h for h in self.hbonds() if (h['D'], h['H'], h['A']) == ('N1', 'H1A', 'O5'))
        return [self.hbond_sites(hb, c) for c in range(len(self.ops))]

    def set_nh(self, F, d, line_to_o5=False):
        """H1A of both formula units put d (A) from its N1, along the current N1->H1A direction (or, with
        line_to_o5, on the N1->O5 line).  Returned unwrapped, with each H1A written in the lattice image of its N1,
        so that the raw coordinate difference pw.x takes for a 'distance' constraint is the N1-H1A bond itself
        (constraints_module.f90 applies the minimum image too, but the input does not rely on it)."""
        F = self.unwrap(np.asarray(F, float)).copy()
        inv = np.linalg.inv(self.lat)
        for sN, sH, sO in self.nh_sites():
            pN, pH, pO = self.point(F, sN), self.point(F, sH), self.point(F, sO)
            v = (pO - pN) if line_to_o5 else (pH - pN)
            u = v / np.linalg.norm(v)
            F[sH[0]] = (pN + d * u) @ inv - sN[1]
        return F

    def constraints_nh(self, d):
        """pw.x CONSTRAINTS rows (atom index N1, atom index H1A, target in bohr), 0-based indices, both f.u."""
        return [(sN[0], sH[0], d / BOHR_A) for sN, sH, _ in self.nh_sites()]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def irreducible_k(cr, grid):
    """Number of irreducible points of an unshifted Monkhorst-Pack grid under the point group of the CIF symops
    plus time reversal (what pw.x will use for a non-magnetic insulator); used to size the k-point pools."""
    n = np.array(grid)
    rots = []
    for op in cr.ops:
        W = np.array([[op.rot[i][j] / op.DEN for j in range(3)] for i in range(3)])
        R = np.linalg.inv(W).T          # action on reciprocal fractional coordinates
        rots += [R, -R]                 # with time reversal
    seen, count = set(), 0
    for idx in np.ndindex(*grid):
        k = np.array(idx) / n
        key = tuple(np.round((k % 1) * n).astype(int) % n)
        if key in seen:
            continue
        count += 1
        for R in rots:
            kk = (R @ k) % 1
            seen.add(tuple(np.round(kk * n).astype(int) % n))
    return count


def choose_pools(np_ranks, nirr):
    """Pools: the largest divisor of the rank count that is <= the irreducible k count, preferring divisors of
    the k count (equal load); k-point pools scale almost perfectly, plane-wave parallelism much less so."""
    divs = [d for d in range(1, np_ranks + 1) if np_ranks % d == 0 and d <= nirr]
    even = [d for d in divs if nirr % d == 0]
    return max(even) if even and max(even) * 2 > max(divs) else max(divs)


# ----------------------------------------------------------------------------------------------------------------
# pw.x input / run / output
# ----------------------------------------------------------------------------------------------------------------
def fortran(x):
    """1e-8 -> '1.0d-08' (Fortran double-precision literal)."""
    return f'{x:.1e}'.replace('e', 'd')


def pw_input(cr, F, tag, calc, ecut, kgrid, outdir, pseudo_dir, conv_thr=1e-9, tstress=False, extra_ctrl=(), extra_sys=(),
             forc_thr=5e-4, etot_thr=1e-5, fixed=(), tprnfor=True, extra_elec=(), pseudos=None, ion_dynamics='bfgs',
             constraints=()):
    """constraints: [(i, j, target_bohr)] -> CONSTRAINTS card, type 'distance', 0-based atom indices (written
    1-based); QE 7.5 accepts it only with ion_dynamics 'damp' / 'fire' (not 'bfgs', see the docstring)."""
    ecutwfc, ecutrho = ecut
    pseudos = pseudos or PSEUDOS
    species = [s for s in pseudos if s in cr.syms]
    ctrl = [f"calculation = '{calc}'", f"prefix = '{tag}'", f"outdir = '{outdir}'", f"pseudo_dir = '{pseudo_dir}'",
            f"tprnfor = {'.true.' if tprnfor else '.false.'}", f"tstress = {'.true.' if tstress else '.false.'}",
            "disk_io = 'low'", "verbosity = 'low'"] + list(extra_ctrl)
    if calc == 'relax':
        # forc_conv_thr 5e-4 Ry/bohr (0.013 eV/A), half the QE default, so H-bond geometries are converged well
        # below the differences we discuss; etot_conv_thr is extensive (54 atoms) -> 1e-5 Ry = 2.5 ueV/atom.
        # QE_NSTEP only for smoke tests (a relax cut short is reported as not converged)
        ctrl += [f"forc_conv_thr = {fortran(forc_thr)}", f"etot_conv_thr = {fortran(etot_thr)}",
                 f"nstep = {int(os.environ.get('QE_NSTEP', '300'))}"]
    # ibrav -12 = monoclinic P, unique axis b: v1 = (a,0,0), v2 = (0,b,0), v3 = (c cos(beta), 0, c sin(beta)), the
    # same frame as gemmi's orthogonalisation, so our fractional coordinates carry over unchanged.  (ibrav 0 works
    # too, and finds the same 2 symmetry operations, but pw.x warns that symmetry detection is less reliable.)
    a, b, c, _, beta, _ = cr.cellpar
    sysn = ["ibrav = -12", f"A = {a:.6f}", f"B = {b:.6f}", f"C = {c:.6f}", f"cosAC = {math.cos(math.radians(beta)):.10f}",
            f"nat = {cr.nat}", f"ntyp = {len(species)}", f"ecutwfc = {ecutwfc}", f"ecutrho = {ecutrho}",
            "occupations = 'fixed'", "vdw_corr = 'dft-d3'", "dftd3_version = 4", "dftd3_threebody = .false."] \
        + list(extra_sys)
    # conv_thr is extensive; 1e-9 Ry for 54 atoms keeps SCF force noise ~1e-5 Ry/bohr.  In a relax QE tightens it
    # further by itself as the forces fall (upscale, default 100).
    elec = [f"conv_thr = {fortran(conv_thr)}", "mixing_beta = 0.4", "electron_maxstep = 200"] + list(extra_elec)
    lines = ['&CONTROL'] + [f'  {x}' for x in ctrl] + ['/', '&SYSTEM'] + [f'  {x}' for x in sysn] + ['/', '&ELECTRONS'] \
        + [f'  {x}' for x in elec] + ['/']
    if calc == 'relax':
        lines += ['&IONS', f"  ion_dynamics = '{ion_dynamics}'", '/']
    lines += ['ATOMIC_SPECIES'] + [f'  {s} {MASS[s]:.4f} {pseudos[s][0]}' for s in species]
    # atoms in `fixed` get if_pos = 0 0 0 (held in place by BFGS)
    lines += ['ATOMIC_POSITIONS crystal'] + [f'  {s:2s} ' + ' '.join(f'{v:15.10f}' for v in f)
                                              + ('  0 0 0' if i in fixed else '')
                                              for i, (s, f) in enumerate(zip(cr.syms, F))]
    lines += ['K_POINTS automatic', '  {} {} {} 0 0 0'.format(*kgrid)]
    if constraints:
        if calc != 'relax' or ion_dynamics == 'bfgs':
            raise ValueError('pw.x 7.5 applies CONSTRAINTS only in damped dynamics / FIRE (iosys: "constraints only '
                             'with fixed-cell dynamics")')
        # nconstr, then one row per constraint: type, atom indices (1-based, ATOMIC_POSITIONS order), target (bohr);
        # the tolerance is left at its default (1e-6 bohr, Modules/input_parameters.f90 constr_tol_inp)
        lines += ['CONSTRAINTS', f'  {len(constraints)}'] + [f"  'distance' {i + 1} {j + 1} {t:.10f}"
                                                             for i, j, t in constraints]
    lines += ['']
    return '\n'.join(lines)


_TIME = re.compile(r'(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?')


def _seconds(s):
    m = _TIME.fullmatch(s.strip())
    if not m:
        return None
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(se or 0)


def parse_pw(path, nat):
    """What we need from a pw.x output: energies, forces, stress, positions, timings, k points, convergence."""
    txt = Path(path).read_text(errors='replace')
    r = {'file': str(path), 'job_done': 'JOB DONE.' in txt}
    m = re.search(r'Program PWSCF v\.(\S+)', txt); r['qe_version'] = m.group(1) if m else None
    e = re.findall(r'^!\s+total energy\s+=\s+(-?[\d.]+) Ry', txt, re.M)
    r['energies_Ry'] = [float(x) for x in e]
    r['energy_Ry'] = float(e[-1]) if e else None
    m = re.findall(r'Final energy\s+=\s+(-?[\d.]+) Ry', txt)
    r['final_energy_Ry'] = float(m[-1]) if m else None
    m = re.findall(r'(?:DFT-D3|Dispersion) (?:Dispersion|Correction)\s*=\s*(-?[\d.]+) Ry', txt)
    r['dispersion_Ry'] = float(m[-1]) if m else None
    m = re.search(r'number of k points=\s*(\d+)', txt); r['nk_irr'] = int(m.group(1)) if m else None
    m = re.search(r'(\d+) Sym\. Ops\.', txt)
    r['n_symops'] = int(m.group(1)) if m else (1 if 'No symmetry found' in txt else None)
    m = re.search(r'number of Kohn-Sham states=\s*(\d+)', txt); r['nbnd'] = int(m.group(1)) if m else None
    m = re.search(r'number of electrons\s*=\s*([\d.]+)', txt); r['nelec'] = float(m.group(1)) if m else None
    m = re.search(r'Dense  grid:\s*(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)', txt)
    r['fft_dense'] = [int(m.group(i)) for i in (2, 3, 4)] if m else None
    m = re.search(r'Number of MPI processes:\s*(\d+)', txt); r['mpi'] = int(m.group(1)) if m else None
    m = re.search(r'Threads/MPI process:\s*(\d+)', txt); r['omp'] = int(m.group(1)) if m else None
    m = re.search(r'K-points division:\s+npool\s*=\s*(\d+)', txt); r['npool'] = int(m.group(1)) if m else None
    r['scf_iterations'] = [int(x) for x in re.findall(r'convergence has been achieved in\s+(\d+) iterations', txt)]
    r['scf_failed'] = 'convergence NOT achieved' in txt
    m = re.findall(r'PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL', txt)
    r['wall_s'] = _seconds(m[-1][1]) if m else None
    # wall time spent in the SCF (the 'electrons' clock), without the setup and the forces/stress
    m = re.findall(r'^\s+electrons\s*:\s*(.+?)CPU\s+(.+?)WALL', txt, re.M)
    r['electrons_wall_s'] = _seconds(m[-1][1]) if m else None
    blocks = txt.split('Forces acting on atoms (cartesian axes, Ry/au):')
    if len(blocks) > 1:
        fl = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force =\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', blocks[-1])[:nat]
        r['forces_Ry_bohr'] = [[float(v) for v in f] for f in fl] if len(fl) == nat else None
    else:
        r['forces_Ry_bohr'] = None
    m = re.findall(r'Total force =\s+([\d.]+)', txt); r['total_force_Ry_bohr'] = float(m[-1]) if m else None
    m = re.findall(r'total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?[\d.]+)', txt)
    r['pressure_kbar'] = float(m[-1]) if m else None
    pos = re.findall(r'ATOMIC_POSITIONS \(crystal\)\n((?:\s*[A-Z][a-z]?\s+-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+.*\n){%d})' % nat,
                     txt)
    r['positions_crystal'] = ([[float(v) for v in l.split()[1:4]] for l in pos[-1].strip().splitlines()]
                              if pos else None)
    r['n_positions'] = len(pos)
    r['bfgs_converged'] = bool(re.search(r'bfgs converged in', txt))
    m = re.search(r'bfgs converged in\s+(\d+) scf cycles and\s+(\d+) bfgs steps', txt)
    r['bfgs_steps'] = [int(m.group(1)), int(m.group(2))] if m else None
    r['final_coordinates'] = 'End final coordinates' in txt
    # damped dynamics / FIRE (constrained profile): their own convergence line and the forces with the constraint
    # components removed ("Constrained forces", printed after the raw forces of each step)
    m = re.search(r'(?:Damped Dynamics|FIRE): convergence achieved in\s+(\d+) steps', txt)
    r['md_converged'] = bool(m)
    r['md_steps'] = int(m.group(1)) if m else None
    r['relax_converged'] = r['bfgs_converged'] or r['md_converged']
    r['n_ion_steps'] = len(re.findall(r'Entering (?:Dynamics|FIRE)\s*:', txt))
    r['max_steps_reached'] = 'The maximum number of steps has been reached' in txt
    r['stopped_by_time'] = bool(re.search(r'Maximum CPU time exceeded', txt))
    cb = txt.split('Constrained forces (Ry/au):')
    fl = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force =\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', cb[-1])[:nat] \
        if len(cb) > 1 else []
    r['constrained_forces_Ry_bohr'] = [[float(v) for v in f] for f in fl] if len(fl) == nat else None
    return r


_OUTER = re.compile(r'^(!!|  )   total energy\s+=\s+(-?[\d.]+) Ry[ \t]*\n\s+Harris-Foulkes estimate\s+=\s+(-?[\d.]+) Ry'
                    r'[ \t]*\n\s+est\. exchange err \(dexx\)\s+=\s+(-?[\d.]+(?:[Ee][-+]?\d+)?) Ry', re.M)
_CLOCK = re.compile(r'^\s+(\w+)\s+:\s+(.+?)CPU\s+(.+?)WALL\s+\(\s*(\d+) calls\)', re.M)


def parse_hybrid(path):
    """The exact-exchange part of a pw.x hybrid (PBE0) output (QE 7.5 PW/src/electrons.f90 formats).

    The run first converges a semilocal SCF ('!    total energy', printed before exact exchange is switched on:
    PBE electrons, with the D3 term of the hybrid's parameters), then alternates ACE builds (outer loops) with
    inner SCFs.  After every outer loop from the second on it prints
        '!!' or '  ' + '   total energy = E Ry', 'Harris-Foulkes estimate', 'est. exchange err (dexx)'
    with '!!' once dexx < conv_thr, followed by 'EXX self-consistency reached'.  The hybrid total energy is that
    '!!' line (it includes the D3 term); the '!' lines of the inner SCFs are not hybrid energies."""
    txt = Path(path).read_text(errors='replace')
    h = {}
    m = re.search(r'Exchange-correlation\s*=\s*(\S+)', txt); h['xc'] = m.group(1) if m else None
    m = re.search(r'EXX-fraction\s*=\s*([\d.]+)', txt); h['exx_fraction'] = float(m.group(1)) if m else None
    m = re.search(r'cutoff for Fock operator\s*=\s*([\d.]+)\s*Ry', txt); h['ecutfock_Ry'] = float(m.group(1)) if m else None
    m = re.search(r'EXX: q-point mesh:\s+(\d+)\s+(\d+)\s+(\d+)', txt); h['nq'] = [int(x) for x in m.groups()] if m else None
    m = re.search(r'EXX grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)', txt)
    h['exx_grid'] = {'G_vectors': int(m.group(1)), 'fft': [int(m.group(i)) for i in (2, 3, 4)]} if m else None
    h['ace'] = 'Using ACE for calculation of exact exchange' in txt
    h['tqr'] = 'Real space treatment of Q(r)' in txt
    # realus.f90 real_space_q: printed when the integral of Q(r) on the real-space grid misses the target by > 1e-5
    m = re.findall(r'integrated real space q too different from target q\s*\n\s*Largest difference:\s*([\d.Ee+-]+) for atom '
                   r'type #\s*(\d+)', txt)
    h['tqr_q_integration_warnings'] = [{'largest_difference': float(d), 'atom_type': int(n)} for d, n in m]
    m = re.search(r'^!\s+total energy\s+=\s+(-?[\d.]+) Ry', txt, re.M)
    h['first_scf_energy_Ry'] = float(m.group(1)) if m else None
    loops = [{'mark': g[0].strip(), 'energy_Ry': float(g[1]), 'harris_foulkes_Ry': float(g[2]), 'dexx_Ry': float(g[3])}
             for g in _OUTER.findall(txt)]
    fock = [float(x) for x in re.findall(r'\+ Fock energy \((?:ACE|full)\)\s*=\s*(-?[\d.]+) Ry', txt)]
    for lp, f in zip(loops, fock):
        lp['fock_energy_Ry'] = f
    h['outer_loops'] = loops
    h['n_outer_loops_printed'] = len(loops)
    h['exx_converged'] = 'EXX self-consistency reached' in txt and bool(loops) and loops[-1]['mark'] == '!!'
    h['energy_Ry'] = loops[-1]['energy_Ry'] if h['exx_converged'] else None
    h['last_outer_energy_Ry'] = loops[-1]['energy_Ry'] if loops else None
    h['last_dexx_Ry'] = loops[-1]['dexx_Ry'] if loops else None
    h['stopped_early'] = bool(re.search(r'Calculation \(EXX\) stopped|Maximum CPU time exceeded|'
                                        r'EXX convergence NOT achieved', txt))
    h['inner_scf_iterations'] = [int(x) for x in re.findall(r'convergence has been achieved in\s+(\d+) iterations', txt)]
    h['clocks_wall_s'] = {name: {'wall_s': _seconds(wall), 'calls': int(n)} for name, _, wall, n in _CLOCK.findall(txt)
                          if name in ('init_run', 'electrons', 'exxinit', 'aceinit', 'aceinit0', 'vexx', 'exxen2',
                                      'c_bands', 'h_psi', 'newd', 'fftc', 'fftcw')}
    h['timing_marks'] = parse_exx_timing(path)['marks_cpu_s']
    return h


def run_pw(args, cr, tag, calc, F, ecut, kgrid, conv_thr=1e-9, tstress=False, extra_ctrl=(), extra_sys=(), probe=False,
           forc_thr=5e-4, etot_thr=1e-5, fixed=(), tprnfor=True, extra_elec=(), hybrid=False, log_extra=None,
           pseudos=None, pseudo_dir=None, ion_dynamics='bfgs', constraints=()):
    """Run (or reuse) one pw.x calculation; returns the parsed output.  Runs live in <out>/runs/<tag>/.
    hybrid=True: success also needs EXX self-consistency (parse_hybrid), and the result carries p['hybrid']."""
    rd = Path(args.out) / 'runs' / tag
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / f'{tag}.out'

    def finished(p):
        return (p['job_done'] and not p['scf_failed'] and (calc != 'relax' or p['relax_converged'])
                and (not hybrid or p['hybrid']['exx_converged']))

    if out.exists():
        p = parse_pw(out, cr.nat)
        if hybrid:
            p['hybrid'] = parse_hybrid(out)
        if finished(p):
            print(f'[periodic_qe] {tag}: finished earlier, reused', flush=True)
            return p
        if calc == 'relax' and p['positions_crystal']:
            # interrupted relax: keep the old output, restart BFGS from the last printed geometry
            n = len(list(rd.glob(f'{tag}.out.part*')))
            out.rename(rd / f'{tag}.out.part{n + 1}')
            F = np.array(p['positions_crystal'])
            note(Path(args.out), f'{tag}: restarting from the last geometry of part {n + 1}')
        elif hybrid:
            # unfinished hybrid SCF (stopped by max_seconds, killed, ...): keep its output and timings, start again
            n = len(list(rd.glob(f'{tag}.out.part*')))
            out.rename(rd / f'{tag}.out.part{n + 1}')
            note(Path(args.out), f'{tag}: unfinished earlier run kept as part {n + 1}; starting again from scratch')
    scr = Path(args.scratch) / tag
    shutil.rmtree(scr, ignore_errors=True); scr.mkdir(parents=True, exist_ok=True)
    inp = rd / f'{tag}.in'
    inp.write_text(pw_input(cr, F, tag, calc, ecut, kgrid, scr, pseudo_dir or args.pseudo, conv_thr=conv_thr,
                            tstress=tstress, extra_ctrl=extra_ctrl, extra_sys=extra_sys, forc_thr=forc_thr,
                            etot_thr=etot_thr, fixed=fixed, tprnfor=tprnfor, extra_elec=extra_elec, pseudos=pseudos,
                            ion_dynamics=ion_dynamics, constraints=constraints))
    nirr = irreducible_k(cr, kgrid)
    nk = args.nk or choose_pools(args.np, nirr)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.omp), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_STACKSIZE=os.environ.get('OMP_STACKSIZE', '64M'))
    # --bind-to none: the VM is shared with other burst tasks that are not pinned; binding MPI ranks to cores
    # that PySCF threads also use would serialise both.  -nd 1: 92 bands is far too few for parallel
    # (ScaLAPACK/ELPA) subspace diagonalisation to pay off.
    cmd = [str(args.mpirun), '-np', str(args.np), '--bind-to', 'none', '--oversubscribe',
           str(args.pw), '-nk', str(nk), '-nd', '1', '-in', str(inp)]
    note(Path(args.out), f'{tag}: start {calc} ecut {ecut[0]}/{ecut[1]} Ry k {kgrid} ({nirr} irr) '
         f'np {args.np} nk {nk} omp {args.omp}')
    t0 = time.time()
    with open(out, 'w') as fo:
        # own process group, so that a killed driver (burst task stopped, SIGTERM) takes mpirun and its ranks with it
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
    rec = {'tag': tag, 'calc': calc, 'ecut': ecut, 'kgrid': kgrid, 'nk_irr': p['nk_irr'],
           'np': args.np, 'nk': nk, 'omp': args.omp, 'rc': rc, 'wall_s': round(wall, 1),
           'scf_iterations': p['scf_iterations'][:3] + (['...'] if len(p['scf_iterations']) > 3 else []),
           'n_scf': len(p['scf_iterations']), 'energy_Ry': p['energy_Ry'], 'utc': utc(),
           'host': os.uname().nodename}
    if hybrid:
        p['hybrid'] = h = parse_hybrid(out)
        # the hybrid total energy is the '!!' line, not the last inner-SCF '!' line that parse_pw returns
        rec.update({'energy_Ry': h['energy_Ry'], 'first_scf_energy_Ry': h['first_scf_energy_Ry'],
                    'exx_converged': h['exx_converged'], 'n_outer_loops': h['n_outer_loops_printed'],
                    'dexx_Ry': [lp['dexx_Ry'] for lp in h['outer_loops']],
                    'aceinit_wall_s': (h['clocks_wall_s'].get('aceinit') or {}).get('wall_s'),
                    'aceinit_calls': (h['clocks_wall_s'].get('aceinit') or {}).get('calls')})
    rec.update(log_extra or {})
    with open(Path(args.out) / 'runs.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    shutil.rmtree(scr, ignore_errors=True)          # wavefunctions are not needed afterwards
    ok = rc == 0 and finished(p)
    if probe:                        # a timing probe stops itself (max_seconds) before convergence
        note(Path(args.out), f'{tag}: probe ended rc={rc} after {wall/60:.1f} min')
        return p
    extra = ''
    if hybrid:
        extra = (f", EXX {'converged' if h['exx_converged'] else 'NOT converged'} after {h['n_outer_loops_printed']} "
                 f"printed outer loops (dexx {rec['dexx_Ry']}), ACE builds {rec['aceinit_calls']} "
                 f"({rec['aceinit_wall_s']} s)")
    note(Path(args.out), f'{tag}: {"done" if ok else "FAILED"} rc={rc} wall {wall/60:.1f} min, '
         f'E = {rec["energy_Ry"]} Ry, {len(p["scf_iterations"])} SCF, nk_irr {p["nk_irr"]}{extra}')
    if not ok:
        raise RuntimeError(f'pw.x {tag} failed (rc={rc}); see {out}')
    return p


# ----------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------
def geometry_report(cr, F):
    """Everything reported for one structure (Angstrom, degrees), measured on formula unit 1 (symop x,y,z);
    'sym_dev_A' is the largest difference between the two formula units (0 when Pc is kept)."""
    rep = {'hbonds': []}
    dev = 0.0
    for hb in cr.hbonds():
        vals = []
        for copy in range(len(cr.ops)):
            sD, sH, sA = cr.hbond_sites(hb, copy)
            vals.append((cr.dist(F, sD, sH), cr.dist(F, sH, sA), cr.dist(F, sD, sA), cr.angle(F, sD, sH, sA)))
        dev = max(dev, max(abs(a - b) for a, b in zip(vals[0][:3], vals[1][:3])))
        dh, ha, da, ang = vals[0]
        rep['hbonds'].append({'bond': f"{hb['D']}-{hb['H']}...{hb['A']} ({hb['code_A']})", 'D_H_A': round(dh, 4),
                              'H_A_A': round(ha, 4), 'D_A_A': round(da, 4), 'DHA_deg': round(ang, 2),
                              'xray_D_A_A': hb['cif_DA'], 'dD_A_vs_xray_A': round(da - hb['cif_DA'], 4)})
    by = {(h['D'], h['H'], h['A']): h for h in cr.hbonds()}
    n1 = by[('N1', 'H1A', 'O5')]
    sN, sH, sO = cr.hbond_sites(n1)
    dNH, dHO = cr.dist(F, sN, sH), cr.dist(F, sH, sO)
    sP = cr.compose(0, 'P1', n1['code_A'])
    po = {f'P1-{o}': round(cr.dist(F, sP, cr.compose(0, o, n1['code_A'])), 4) for o in ('O3', 'O4', 'O5', 'O6')}
    rep['proton'] = {'d_N1_H1A_A': round(dNH, 4), 'd_H1A_O5_A': round(dHO, 4), 'delta_A': round(dNH - dHO, 4),
                     'd_N1_O5_A': round(cr.dist(F, sN, sO), 4), 'angle_N1_H1A_O5_deg': round(cr.angle(F, sN, sH, sO), 2),
                     'P_O_A': po, 'proton_on': 'N' if dNH < dHO else 'O'}
    o4 = by[('O4', 'H4A', 'O5')]
    sD, sH4, sA = cr.hbond_sites(o4)
    dOH, dHA, dOO = cr.dist(F, sD, sH4), cr.dist(F, sH4, sA), cr.dist(F, sD, sA)
    rep['O4_H4A_O5'] = {'d_O4_H4A_A': round(dOH, 4), 'd_H4A_O5_A': round(dHA, 4), 'd_O4_O5_A': round(dOO, 4),
                        'delta_OHO_A': round(dOH - dHA, 4), 'angle_deg': round(cr.angle(F, sD, sH4, sA), 2),
                        # 0.5 = proton centred between O4 and O5; the X-ray has 0.80/1.76 A with DFIX 0.82 on O-H
                        'fraction_along_O4_O5': round(float((cr.point(F, sH4) - cr.point(F, sD))
                                                            @ (cr.point(F, sA) - cr.point(F, sD))) / dOO ** 2, 4),
                        'xray_d_O4_O5_A': o4['cif_DA']}
    rep['bonds'] = {f'{a}-{b}': round(cr.dist(F, cr.compose(0, a), cr.compose(0, b)), 4) for a, b in KEY_BONDS}
    rep['h_contacts'] = h_contacts(cr, F)
    rep['sym_dev_A'] = round(dev, 5)
    return rep


def h_contacts(cr, F, cutoff=2.6):
    """Independent of the CIF table: for every N-H / O-H hydrogen of formula unit 1, its covalent partner (the
    nearest N/O) and its nearest N/O acceptor in any lattice image.  Catches an H-bond that a relaxation has
    re-routed (e.g. after the proton moved to O5), which the fixed CIF assignment above would not show."""
    X = cr.cart(cr.unwrap(F))
    shifts = np.array([s for s in np.ndindex(5, 5, 5)]) - 2
    T = shifts @ cr.lat
    NO = [j for j, e in enumerate(cr.syms) if e in ('N', 'O')]
    out = []
    for i in range(cr.nasym):
        if cr.syms[i] != 'H':
            continue
        cand = []
        for j in NO:
            d = np.linalg.norm(X[j] + T - X[i], axis=1)
            k = int(np.argmin(d))
            cand.append((float(d[k]), j, k))
        cand.sort()
        (d1, j1, k1), (d2, j2, k2) = cand[0], cand[1]
        if d1 > 1.3:                   # a C-H hydrogen
            continue
        pD, pH, pA = X[j1] + T[k1], X[i], X[j2] + T[k2]
        v1, v2 = pD - pH, pA - pH
        ang = float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
        out.append({'H': cr.labels[i], 'bonded_to': cr.labels[j1], 'd_bond_A': round(d1, 4),
                    'nearest_acceptor': cr.labels[j2], 'image': shifts[k2].tolist(), 'H_A_A': round(d2, 4),
                    'D_A_A': round(float(np.linalg.norm(pA - pD)), 4), 'DHA_deg': round(ang, 2),
                    'hbond': d2 <= cutoff})
    return out


def rmsd_vs(cr, F, Fref=None, heavy_only=True):
    """RMSD (A) after removing the mean displacement: Pc is a polar group, so the origin is free along a and c
    and a uniform shift of the whole crystal is not a structural change.  Also the raw value and the maximum."""
    Fref = cr.F0 if Fref is None else Fref
    d = cr.unwrap(F) - Fref
    d = d - np.round(d)
    D = cr.cart(d)
    mask = np.array([s != 'H' for s in cr.syms]) if heavy_only else np.ones(cr.nat, bool)
    Dm = D[mask]
    shift = Dm.mean(axis=0)
    per = np.linalg.norm(Dm - shift, axis=1)
    worst = int(np.argmax(per))
    labs = [l for l, m in zip(cr.labels, mask) if m]
    return {'rmsd_A': round(float(np.sqrt((per ** 2).mean())), 4),
            'rmsd_raw_A': round(float(np.sqrt((np.linalg.norm(Dm, axis=1) ** 2).mean())), 4),
            'max_A': round(float(per[worst]), 4), 'max_atom': labs[worst],
            'mean_shift_A': [round(float(v), 4) for v in shift],
            'n_atoms': int(mask.sum())}


def write_extxyz(cr, F, path, comment):
    X = cr.cart(F)
    lat = ' '.join(f'{v:.8f}' for v in cr.lat.ravel())
    with open(path, 'w') as f:
        f.write(f'{cr.nat}\nLattice="{lat}" Properties=species:S:1:pos:R:3:label:S:1 pbc="T T T" comment="{comment}"\n')
        for s, (x, y, z), l in zip(cr.syms, X, cr.labels):
            f.write(f'{s:2s} {x:14.8f} {y:14.8f} {z:14.8f} {l}\n')


def conv_row(p, nat, ref=None):
    row = {'E_Ry': p['energy_Ry'], 'wall_s': p['wall_s'], 'nk_irr': p['nk_irr'],
           'n_scf_iter': (p['scf_iterations'] or [None])[0], 'pressure_kbar': p['pressure_kbar'], 'max_force_Ry_bohr': None}
    Fo = np.array(p['forces_Ry_bohr']) if p['forces_Ry_bohr'] else None
    if Fo is not None:
        row['max_force_Ry_bohr'] = round(float(np.abs(Fo).max()), 6)
    if ref is not None and Fo is not None and ref['forces_Ry_bohr']:
        row['dE_meV_atom'] = round((p['energy_Ry'] - ref['energy_Ry']) * RY_EV * 1000 / nat, 4)
        row['max_dF_Ry_bohr'] = round(float(np.abs(Fo - np.array(ref['forces_Ry_bohr'])).max()), 6)
        if ref.get('pressure_kbar') is not None and p.get('pressure_kbar') is not None:
            row['dP_kbar'] = round(p['pressure_kbar'] - ref['pressure_kbar'], 3)
    return row


# ----------------------------------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------------------------------
def stage_build(args, cr):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # sanity: 54 atoms, no two atoms closer than 0.7 A anywhere in the periodic crystal
    assert cr.nat == 54, cr.nat
    X = cr.cart(cr.F0)
    dmin = 9.0
    for sh in np.ndindex(3, 3, 3):
        t = (np.array(sh) - 1) @ cr.lat
        dd = np.linalg.norm(X[:, None, :] - (X[None, :, :] + t), axis=2)
        if sh == (1, 1, 1):
            np.fill_diagonal(dd, 9.0)
        dmin = min(dmin, float(dd.min()))
    assert dmin > 0.7, f'atoms overlap: {dmin:.3f} A'
    rep = geometry_report(cr, cr.F0)
    bad = [h for h in rep['hbonds'] if abs(h['dD_A_vs_xray_A']) > 0.002]
    assert not bad, f'D...A disagrees with the CIF table: {bad}'
    Fii = cr.start_ii()
    rep_ii = geometry_report(cr, Fii)
    assert abs(rep_ii['proton']['d_H1A_O5_A'] - D_OH_START) < 1e-6, rep_ii['proton']
    assert rep_ii['sym_dev_A'] < 1e-6 and rep['sym_dev_A'] < 1e-6
    write_extxyz(cr, cr.F0, out / 'start_i_xray.extxyz', 'start (i): X-ray, deposited H (proton on N1)')
    write_extxyz(cr, Fii, out / 'start_ii_OH.extxyz',
                 f'start (ii): H1A {D_OH_START} A from O5 on N1->O5 (both formula units)')
    info = {'cif': CIF.name, 'cif_md5': md5(CIF), 'cell_A_deg': cr.cellpar, 'lattice_rows_A': cr.lat.round(8).tolist(),
            'volume_A3': round(float(abs(np.linalg.det(cr.lat))), 3), 'symops': cr.op_str, 'nat': cr.nat,
            'min_interatomic_A': round(dmin, 4),
            'irreducible_k': {'x'.join(map(str, g)): irreducible_k(cr, g) for g in KGRIDS},
            'start_i': rep, 'start_ii': rep_ii}
    atomic_json(out / 'build.json', info)
    note(out, f'build: 54-atom Pc cell, V = {info["volume_A3"]} A^3, min distance {dmin:.3f} A, CIF H-bonds reproduced '
         f'(max |dD...A| {max(abs(h["dD_A_vs_xray_A"]) for h in rep["hbonds"]):.4f} A); start ii delta = '
         f'{rep_ii["proton"]["delta_A"]:+.3f} A')
    if args.dry_run:
        (out / 'inputs').mkdir(exist_ok=True)
        (out / 'inputs' / 'scf_i_example.in').write_text(
            pw_input(cr, cr.F0, 'scf_i', 'scf', ECUTS[0], K_FOR_ECUT, '/SCRATCH', args.pseudo, tstress=True))
        (out / 'inputs' / 'relax_ii_example.in').write_text(
            pw_input(cr, Fii, 'relax_ii', 'relax', ECUTS[1], K_FOR_ECUT, '/SCRATCH', args.pseudo, conv_thr=1e-8))
    return info


def check_pseudos(args, pseudos=PSEUDOS, pdir=None):
    pdir = pdir or args.pseudo
    bad = []
    for el, (fn, h) in pseudos.items():
        p = Path(pdir) / fn
        if not p.exists() or md5(p) != h:
            bad.append(fn)
    if bad:
        raise SystemExit(f'pseudopotentials missing or with a wrong md5 in {pdir}: {bad}')


def tag_scf(start, ecut, k):
    return f'scf_{start}_e{ecut[0]}_k{"".join(map(str, k))}'


def stage_converge(args, cr):
    check_pseudos(args)
    Fii = cr.start_ii()
    ecuts = [tuple(map(int, e.split('/'))) for e in args.ecuts] if args.ecuts else ECUTS
    kgrids = [tuple(map(int, k.split('x'))) for k in args.kgrids] if args.kgrids else KGRIDS
    kfe = tuple(map(int, args.k_for_ecut.split('x'))) if args.k_for_ecut else K_FOR_ECUT
    runs = {}
    # cheapest first, so a sandbox test with --max-runs gets the most information per core-hour
    jobs = [('i', ecuts[0], k) for k in kgrids] + [('ii', ecuts[0], kfe)] \
        + [(s, e, kfe) for e in ecuts[1:] for s in ('i', 'ii')]
    seen = set()
    for n, (s, e, k) in enumerate([j for j in jobs if not (j in seen or seen.add(j))]):
        if args.max_runs and n >= args.max_runs:
            break
        F = cr.F0 if s == 'i' else Fii
        runs[(s, e, k)] = run_pw(args, cr, tag_scf(s, e, k), 'scf', F, e, k, conv_thr=1e-9, tstress=True)
    summarise_convergence(args, cr, runs, ecuts, kgrids, kfe)


def summarise_convergence(args, cr, runs, ecuts, kgrids, kfe):
    out = Path(args.out)
    res = {'targets': {'dE_meV_atom': TARGET_E_MEV_ATOM, 'dF_Ry_bohr': TARGET_F_RYB}, 'ecut': [], 'kgrid': []}
    ref_e = max((e for e in ecuts if ('i', e, kfe) in runs), default=None)
    for e in ecuts:
        if ('i', e, kfe) not in runs:
            continue
        row = {'ecut_Ry': list(e), 'k': 'x'.join(map(str, kfe))}
        row.update(conv_row(runs[('i', e, kfe)], cr.nat, runs[('i', ref_e, kfe)]))
        if ('ii', e, kfe) in runs:
            dE = (runs[('ii', e, kfe)]['energy_Ry'] - runs[('i', e, kfe)]['energy_Ry']) * RY_EV * 1000
            row['E_ii_minus_i_meV_cell'] = round(dE, 2)
        res['ecut'].append(row)
    if ref_e and ('ii', ref_e, kfe) in runs:
        for row in res['ecut']:
            if 'E_ii_minus_i_meV_cell' in row:
                ref = next(r for r in res['ecut'] if tuple(r['ecut_Ry']) == ref_e)['E_ii_minus_i_meV_cell']
                row['d_E_ii_minus_i_meV_cell'] = round(row['E_ii_minus_i_meV_cell'] - ref, 2)
    ref_k = max((k for k in kgrids if ('i', ecuts[0], k) in runs), key=lambda k: np.prod(k), default=None)
    for k in kgrids:
        if ('i', ecuts[0], k) not in runs:
            continue
        row = {'k': 'x'.join(map(str, k)), 'ecut_Ry': list(ecuts[0])}
        row.update(conv_row(runs[('i', ecuts[0], k)], cr.nat, runs[('i', ecuts[0], ref_k)]))
        res['kgrid'].append(row)

    def first_ok(rows):
        for r in rows:
            if r.get('dE_meV_atom') is not None and abs(r['dE_meV_atom']) <= TARGET_E_MEV_ATOM \
                    and r['max_dF_Ry_bohr'] <= TARGET_F_RYB:
                return r
        return rows[-1] if rows else None
    # rows are ordered cheapest first; the reference itself always passes, so a series that never converges
    # below the reference falls back to the reference (the most expensive setting tested)
    e_test = first_ok(res['ecut']); k_ok = first_ok(res['kgrid'])
    # SSSP floor (see SSSP_MIN_ECUT); a series with no row at or above the floor (sandbox) keeps the plain choice
    e_ok = first_ok([r for r in res['ecut'] if r['ecut_Ry'][0] >= SSSP_MIN_ECUT[0]]) or e_test
    res['reference'] = {'ecut_Ry': list(ref_e) if ref_e else None, 'k': 'x'.join(map(str, ref_k)) if ref_k else None}
    res['chosen'] = {'ecut_Ry': e_ok['ecut_Ry'] if e_ok else None, 'k': k_ok['k'] if k_ok else None,
                     'complete': len(res['ecut']) == len(ecuts) and len(res['kgrid']) == len(kgrids),
                     'ecut_by_energy_test_Ry': e_test['ecut_Ry'] if e_test else None,
                     'sssp_floor_Ry': list(SSSP_MIN_ECUT),
                     'rule': 'cheapest setting within 1 meV/atom and 1e-3 Ry/bohr of the most converged one tested; '
                             'cutoff not below the SSSP-recommended 60/480 Ry'}
    res['E_ii_minus_i_note'] = ('E(ii) - E(i) at the unrelaxed starts (X-ray N-H 0.86-0.90 A vs O5-H 0.98 A): used only '
                                'for its convergence with cutoff; the physical energy difference is in periodic_qe.json')
    res['pressure_note'] = ('pressure_kbar: PBE-D3(BJ) stress of the X-ray (294 K) cell with the deposited, '
                            'unrelaxed H positions (X-H 0.78-0.93 A, too short), so it is large and positive and '
                            'says nothing about the cell; only its convergence (dP_kbar) is used.  The meaningful '
                            'pressure is the one at the relaxed ions (periodic_qe.json, pressure_kbar_final)')
    atomic_json(out / 'convergence.json', res)
    note(out, f'convergence: chosen {res["chosen"]}')
    return res


def chosen_setting(args):
    if args.ecut and args.kgrid:
        return tuple(map(int, args.ecut.split('/'))), tuple(map(int, args.kgrid.split('x')))
    cj = Path(args.out) / 'convergence.json'
    if not cj.exists():
        raise SystemExit('no convergence.json: run the converge stage first or pass --ecut and --kgrid')
    c = json.loads(cj.read_text())['chosen']
    if not c.get('complete') and not (args.ecut or args.kgrid):
        # a partial series (e.g. a sandbox run with --max-runs) must not silently pick the production setting
        raise SystemExit(f'convergence.json is incomplete ({c}); finish converge or pass --ecut/--kgrid')
    return (tuple(map(int, args.ecut.split('/'))) if args.ecut else tuple(c['ecut_Ry']),
            tuple(map(int, (args.kgrid or c['k']).split('x'))))


def relaxed_salt(args, cr, start='iir'):
    """Final geometry of a converged relax from start (i), the base of start iir: the coarse relax_i_pre (so that
    iir's pre-relaxation is at the same setting as its base), or relax_i with --no-pre.  Start iirfix: relax_i only."""
    tags = ['relax_i'] if start in ('iirfix', 'iir1') else (['relax_i', 'relax_i_pre'] if args.no_pre else ['relax_i_pre', 'relax_i'])
    # iir1: also --geom-from (sandbox tests); the other starts keep reading --out only, as before
    dirs = geom_dirs(args)[::-1] if start == 'iir1' else [args.out]
    for tag in tags:
        f = next((Path(g) / 'runs' / tag / f'{tag}.out' for g in dirs
                  if (Path(g) / 'runs' / tag / f'{tag}.out').exists()), Path(args.out) / 'runs' / tag / f'{tag}.out')
        if f.exists():
            p = parse_pw(f, cr.nat)
            if p['job_done'] and p['bfgs_converged'] and p['positions_crystal']:
                note(Path(args.out), f'start {start}: H1A moved {D_OH_START} A from O5 in the {tag} geometry'
                     + (' (formula unit 1 only)' if start == 'iir1' else ''))
                return np.array(p['positions_crystal'])
    raise SystemExit(f'start {start} needs a converged {" or ".join(tags)} (run relax --start i first)')


def stage_relax(args, cr):
    """Fixed-cell BFGS from start (i), (ii), (iir), (iifix) or (iirfix; no pre-stage).  From the X-ray hydrogens (X-H 0.78-0.93 A) the first steps
    only lengthen X-H bonds (-6 eV in the first step in the sandbox), so by default each start is first relaxed at
    the cheapest setting (--pre-ecut/--pre-kgrid, 50/400 Ry, 2x1x1: within 0.5 meV/atom and 8e-4 Ry/bohr of 3x2x2
    in the sandbox test) with the QE default thresholds, and the production relax continues from that geometry
    with a fresh BFGS Hessian.  Nothing about the final minimum depends on the pre-stage except which basin it
    starts in; a proton that crosses in the coarse stage is reported (relax_<s>_pre in runs.jsonl / STAGES.md)."""
    check_pseudos(args)
    pre_e = tuple(map(int, args.pre_ecut.split('/'))); pre_k = tuple(map(int, args.pre_kgrid.split('x')))
    # --pre-only needs no convergence result, so the pre-relaxations can run while 'converge' is still running
    ecut, k = (None, None) if args.pre_only else chosen_setting(args)
    if _PROF.match(args.start):
        raise SystemExit('relax: profile points are run by the profile stage (--d), not relax --start')
    starts = ['i', 'ii'] if args.start == 'both' else [args.start]
    for s in starts:
        if s in ('iirfix', 'iir1') and args.pre_only:
            note(Path(args.out), f'relax_{s} has no pre-stage (--pre-only): skipped')
            continue
        # iir1 (single-transfer control, 29 Sep 2026): relax_i with H1A moved in formula unit 1 only, unconstrained
        F = cr.F0 if s == 'i' else cr.start_ii(base=relaxed_salt(args, cr, s) if s in ('iir', 'iirfix', 'iir1') else None,
                                               copies=[0] if s == 'iir1' else None)
        # 'iifix': start (ii) with H1A of both formula units held where start (ii) puts it (0.98 A from the X-ray
        # O5 on the N1->O5 line); everything else relaxes.  (It failed: the floating origin let the lattice slide
        # until N1 re-bonded to the fixed H1A; see the docstring.)  'iirfix': H1A and its O5 acceptor of both
        # formula units fixed, on the relaxed salt, no pre-stage: the constrained neutral (O5-H) form.
        fixed = (cr.index['H1A'], cr.index['H1A_2']) if s == 'iifix' else ()
        if s == 'iirfix':
            hb = next(h for h in cr.hbonds() if (h['D'], h['H'], h['A']) == ('N1', 'H1A', 'O5'))
            fixed = tuple(sorted({cr.index['H1A'], cr.index['H1A_2']}
                                 | {cr.hbond_sites(hb, c)[2][0] for c in range(len(cr.ops))}))
        # (when the chosen setting is the pre-setting itself, the final stage just tightens the thresholds)
        if not args.no_pre and s not in ('iirfix', 'iir1'):
            p = run_pw(args, cr, f'relax_{s}_pre', 'relax', F, pre_e, pre_k, conv_thr=1e-8, forc_thr=1e-3, etot_thr=1e-4,
                       fixed=fixed)
            F = np.array(p['positions_crystal'])
            g = geometry_report(cr, F)['proton']
            note(Path(args.out), f'relax_{s}_pre: proton on {g["proton_on"]} (delta {g["delta_A"]:+.3f} A)')
            if args.pre_only:
                continue
        # tstress: the stress at the relaxed ions says how far the X-ray (294 K) cell is from the PBE-D3(BJ)
        # equilibrium cell (~10% extra per step); at the unrelaxed X-ray H positions it is meaningless (X-H too short)
        p = run_pw(args, cr, f'relax_{s}', 'relax', F, ecut, k, conv_thr=1e-8, tstress=True, fixed=fixed)
        Fr = np.array(p['positions_crystal'])
        write_extxyz(cr, Fr, Path(args.out) / f'relax_{s}.extxyz',
                     f'PBE-D3(BJ) fixed-cell relax from start ({s}), ecut {ecut[0]}/{ecut[1]} Ry, k {k}')
        if s == 'iirfix':
            # the point of iirfix is a neutral (O5-H) structure: fail loudly if it is not one (as iifix was not)
            g = geometry_report(cr, Fr)['proton']
            note(Path(args.out), f'relax_iirfix: proton on {g["proton_on"]} (N1-H1A {g["d_N1_H1A_A"]}, H1A-O5 '
                 f'{g["d_H1A_O5_A"]} A); lattice shift vs start {rmsd_vs(cr, Fr, cr.unwrap(F))["mean_shift_A"]} A')
            if g['proton_on'] != 'O':
                raise SystemExit('relax_iirfix: the proton ended on N1, so it is not the neutral form')
        if s == 'iir1':
            on = [proton_fu(cr, Fr, c) for c in range(len(cr.ops))]
            note(Path(args.out), 'relax_iir1 (H1A moved in formula unit 1 only): ' + '; '.join(
                f"f.u. {c + 1}: proton on {g['proton_on']} (N1-H1A {g['d_N1_H1A_A']}, H1A-O5 {g['d_H1A_O5_A']} A)"
                for c, g in enumerate(on)) + f"; {p['n_symops']} symmetry operation(s) in pw.x")


def run_setting(inp):
    """(ecut [wfc, rho] in Ry, k grid 'AxBxC') of a pw.x input written by pw_input()."""
    txt = Path(inp).read_text() if Path(inp).exists() else ''
    m = re.search(r'ecutwfc\s*=\s*(\d+).*?ecutrho\s*=\s*(\d+)', txt, re.S)
    kk = re.search(r'K_POINTS automatic\n\s*(\d+) (\d+) (\d+)', txt)
    return ([int(m.group(1)), int(m.group(2))] if m else None), ('x'.join(kk.groups()) if kk else None)


def compare_to_salt(cr, rl, s):
    """Relaxed start s (ii or iir, the O5-protonated starts) against the relaxed salt (i)."""
    dE = rl[s]['energy_Ry'] - rl['i']['energy_Ry']
    same_setting = rl['i']['ecut_Ry'] == rl[s]['ecut_Ry'] and rl['i']['k'] == rl[s]['k']
    between = rmsd_vs(cr, rl[s]['_F'], cr.unwrap(rl['i']['_F']), heavy_only=False)
    c = {f'E_{s}_minus_i_Ry': round(dE, 8), f'E_{s}_minus_i_meV_cell': round(dE * RY_EV * 1000, 2),
         f'E_{s}_minus_i_kJ_mol_per_formula_unit': round(dE * RY_EV * 96.485332 / 2, 3),
         'same_setting': same_setting,
         f'rmsd_all_atoms_final_{s}_vs_final_i_A': between['rmsd_A'],
         'proton_on_final': {t: rl[t]['geometry']['proton']['proton_on'] for t in rl},
         'both_converged': rl['i']['converged'] and rl[s]['converged']}
    on = rl[s]['geometry']['proton']['proton_on']
    # the O5-protonated start "returned" if its proton ends on N1 and the two relaxed structures coincide
    c[f'{s}_returned_to_i'] = on == 'N' and between['rmsd_A'] < 0.02 and abs(dE * RY_EV * 1000) < 5
    c[f'{s}_is_separate_minimum'] = on == 'O' and c['both_converged'] and same_setting
    if not same_setting:
        # E(s) - E(i) between different cutoffs / k grids is meaningless (review fix, 28 Sep 2026)
        c['verdict'] = f'provisional: relax_{s} and relax_i were run at different settings'
    elif not c['both_converged']:
        c['verdict'] = 'provisional: at least one relaxation has not converged'
    elif c[f'{s}_returned_to_i']:
        c['verdict'] = f'the O5-protonated start ({s}) relaxes back to the salt (proton on N1): single minimum'
    elif on == 'O':
        c['verdict'] = (f"two minima: the neutral (O5-H) form ({s}) is a local minimum "
                        f"{c[f'E_{s}_minus_i_meV_cell']:+.1f} meV/cell "
                        f"({c[f'E_{s}_minus_i_kJ_mol_per_formula_unit']:+.2f} kJ/mol per formula unit) relative to the salt")
    else:
        c['verdict'] = (f'proton of ({s}) back on N1 but in a different local minimum (RMSD '
                        f"{between['rmsd_A']} A, dE {c[f'E_{s}_minus_i_meV_cell']:+.1f} meV/cell): inspect h_contacts")
    return c


def stage_analyse(args, cr):
    out = Path(args.out)
    res = {'method': {
        'code': 'Quantum ESPRESSO pw.x', 'functional': 'PBE', 'dispersion': 'Grimme D3, Becke-Johnson damping, two-body '
        '(vdw_corr=dft-d3, dftd3_version=4, dftd3_threebody=.false.)',
        'pseudopotentials': 'SSSP PBE efficiency 1.3.0 (doi:10.24435/materialscloud:f3-ym)',
        'pseudo_files': {el: {'file': fn, 'md5': h} for el, (fn, h) in PSEUDOS.items()},
        'cell': 'fixed at the X-ray values (294 K), Pc, Z = 2, 54 atoms', 'occupations': 'fixed (insulator)',
        'relax': 'BFGS, forc_conv_thr 5e-4 Ry/bohr, etot_conv_thr 1e-5 Ry, conv_thr 1e-8 Ry (upscale 100)'},
        'generated_utc': utc()}
    if (out / 'convergence.json').exists():
        res['convergence'] = json.loads((out / 'convergence.json').read_text())
    res['xray'] = geometry_report(cr, cr.F0)
    rl = {}
    for s in ('i', 'ii', 'iir', 'iifix', 'iirfix'):
        f = out / 'runs' / f'relax_{s}' / f'relax_{s}.out'
        if not f.exists():
            continue
        p = parse_pw(f, cr.nat)
        if not p['positions_crystal']:
            continue
        F = np.array(p['positions_crystal'])
        E = p['final_energy_Ry'] or p['energy_Ry']
        ecut_in, k_in = run_setting(f.parent / f'relax_{s}.in')
        parts = sorted(f.parent.glob(f'relax_{s}.out.part*'))
        rl[s] = {'converged': p['bfgs_converged'], 'job_done': p['job_done'], 'energy_Ry': E,
                 'pressure_kbar_final': p['pressure_kbar'],
                 'energy_eV': round(E * RY_EV, 6) if E else None, 'dispersion_Ry': p['dispersion_Ry'],
                 'bfgs_steps': p['bfgs_steps'], 'n_geometries': p['n_positions'], 'restarts': len(parts),
                 'wall_s_last_part': p['wall_s'], 'ecut_Ry': ecut_in, 'k': k_in,
                 'qe_version': p['qe_version'], 'mpi': p['mpi'],
                 'npool': p['npool'], 'nk_irr': p['nk_irr'], 'n_symops': p['n_symops'],
                 'max_force_Ry_bohr': (round(float(np.abs(np.array(p['forces_Ry_bohr'])).max()), 6)
                                       if p['forces_Ry_bohr'] else None),
                 'geometry': geometry_report(cr, F), 'rmsd_heavy_vs_xray': rmsd_vs(cr, F),
                 'rmsd_all_vs_xray': rmsd_vs(cr, F, heavy_only=False), '_F': F}
        write_extxyz(cr, F, out / f'relax_{s}.extxyz', f'relax from start ({s}), E = {E} Ry')
    # 'comparison' (start ii) keeps its original keys; start iir gets the same block as 'comparison_iir'
    if 'i' in rl and 'ii' in rl:
        res['comparison'] = compare_to_salt(cr, rl, 'ii')
    if 'i' in rl and 'iir' in rl:
        res['comparison_iir'] = compare_to_salt(cr, rl, 'iir')
    def constrained(s, what):
        dEf = rl[s]['energy_Ry'] - rl['i']['energy_Ry']
        same = rl['i']['ecut_Ry'] == rl[s]['ecut_Ry'] and rl['i']['k'] == rl[s]['k']
        g = rl[s]['geometry']['proton']
        # the whole-crystal slide that defeated iifix: mean displacement of every atom against relax_i
        d = cr.unwrap(rl[s]['_F']) - cr.unwrap(rl['i']['_F'])
        c = {'what': what, f'E_{s}_minus_i_meV_cell': round(dEf * RY_EV * 1000, 2),
             f'E_{s}_minus_i_kJ_mol_per_formula_unit': round(dEf * RY_EV * 96.485332 / 2, 3),
             'same_setting': same, 'converged': rl[s]['converged'] and rl['i']['converged'] and same,
             'proton': g, 'constraint_held': g['proton_on'] == 'O',
             'mean_shift_vs_relax_i_A': [round(float(v), 3) for v in cr.cart(d - np.round(d)).mean(axis=0)]}
        if not c['constraint_held']:
            c['WARNING'] = (f'the proton of relax_{s} ended on N1 (N1-H1A {g["d_N1_H1A_A"]} A): this is a salt '
                            f'structure, NOT the neutral form; E_{s}_minus_i is not a neutral-salt energy difference')
        return c
    if 'i' in rl and 'iifix' in rl:
        res.setdefault('comparison', {})['constrained_neutral'] = constrained('iifix', (
            'start ii relaxed with H1A (both formula units) fixed in space where start ii puts it (0.98 A '
            'from the X-ray O5, on the N1->O5 line); N1 and O5 relax, so read the final d(N1-H1A) and '
            'd(H1A-O5) in proton.  Result: the floating origin of Pc let every other atom slide (mean_shift) until N1 '
            're-bonded to the fixed H1A, so this is a salt (see constraint_held); superseded by iirfix.'))
    if 'i' in rl and 'iirfix' in rl:
        res.setdefault('comparison', {})['constrained_neutral_iirfix'] = constrained('iirfix', (
            'relax_i with H1A (both formula units) moved 0.98 A from O5 on the N1->O5 line, then relaxed with H1A '
            'AND O5 fixed in space (O5-H1A held at 0.98 A; pins the floating origin).  A constrained structure, not a '
            'minimum: an upper bound to the energy of the neutral (O5-H) form in the lattice.'))
    # coarse pre-relaxations (50/400 Ry, 2x1x1 by default): where each start went before the production relax
    pre = {}
    for s in ('i', 'ii', 'iir', 'iifix', 'iirfix'):
        f = out / 'runs' / f'relax_{s}_pre' / f'relax_{s}_pre.out'
        if f.exists():
            p = parse_pw(f, cr.nat)
            if p['positions_crystal']:
                g = geometry_report(cr, np.array(p['positions_crystal']))
                pre[s] = {'converged': p['bfgs_converged'], 'energy_Ry': p['final_energy_Ry'] or p['energy_Ry'],
                          'setting': run_setting(f.parent / f'relax_{s}_pre.in'),
                          'bfgs_steps': p['bfgs_steps'], 'proton': g['proton'], 'O4_H4A_O5': g['O4_H4A_O5']}
    if 'i' in pre:
        for s in pre:
            if s != 'i' and pre[s]['energy_Ry'] and pre['i']['energy_Ry'] and pre[s]['setting'] == pre['i']['setting']:
                pre[s]['E_minus_i_meV_cell'] = round((pre[s]['energy_Ry'] - pre['i']['energy_Ry']) * RY_EV * 1000, 2)
    if pre:
        res['pre_relax'] = pre
    hyb = hybrid_summary(out, cr, rl)
    if hyb:
        res['hybrid'] = hyb
    for s in rl:
        rl[s].pop('_F')
    res['relax'] = rl
    runs = [json.loads(l) for l in (out / 'runs.jsonl').read_text().splitlines()] if (out / 'runs.jsonl').exists() else []
    res['timings'] = runs
    atomic_json(out / 'periodic_qe.json', res)
    v = [res[k]['verdict'] for k in ('comparison', 'comparison_iir', 'hybrid') if 'verdict' in (res.get(k) or {})]
    note(out, 'analyse: wrote periodic_qe.json' + ''.join(f'; {x}' for x in v))
    return res


def stage_estimate(args, cr):
    """Cost of one PBE0/HSE single point at the production setting (ACE; nq = 1x1x1 and nq = k grid).

    With ACE, each EXX outer loop rebuilds the exchange projector from one pair density per (k, q, occupied i,
    occupied j), i.e. its cost is proportional to N_k(irr) x N_q x N_occ^2 x (FFT on the exchange grid).  The
    exchange grid is set by ecutfock, whose QE 7.5 default is MIN(ecutrho, 4*ecutwfc) (PW/src/input.f90,
    exx_iosys; INPUT_PW.def still says "ecutrho"), e.g. 240 Ry at 60/480; this was seen in the sandbox probe
    output ("cutoff for Fock operator = 200 Ry" at 50/400) with these US/PAW potentials.  Pair densities of
    US/PAW atoms are augmented as well (a factor 2 is assumed, UNVERIFIED).  When the hybrid_probe stage has
    run, its measured time per outer loop (1 k x 1 q, Gamma) is scaled; otherwise the FFT-count model is used.
    Both are estimates (the output says which); nothing is run here."""
    out = Path(args.out)
    runs = [json.loads(l) for l in (out / 'runs.jsonl').read_text().splitlines()] if (out / 'runs.jsonl').exists() else []
    scf = [r for r in runs if r['calc'] == 'scf' and r['rc'] == 0 and not r['tag'].startswith(('probe', 'hybrid', 'pbe_'))]
    if not scf:
        raise SystemExit('no finished SCF run to scale from')
    try:
        ecut, kg = chosen_setting(args)
    except SystemExit:
        ecut, kg = (60, 480), (3, 2, 2)
    nk_irr = irreducible_k(cr, kg)
    nq_full = int(np.prod(kg))
    # measured PBE SCF (+forces, stress) cost per irreducible k point, scaled to the production cutoff by the
    # number of dense-grid points (~ ecutrho^1.5)
    r = max(scf, key=lambda r: r['nk_irr'] or 0)
    pbe_core_s = r['wall_s'] * r['np'] * r['omp'] / r['nk_irr'] * nk_irr * (ecut[1] / r['ecut'][1]) ** 1.5
    n_outer = (4, 7)                # ACE outer loops to dexx ~1e-5 Ry (typical for insulators)
    pj = out / 'hybrid_probe.json'
    probe = json.loads(pj.read_text()) if pj.exists() else None
    if probe and probe.get('exx_outer_loop_s'):
        t_loop = max(probe['exx_outer_loop_s'])                  # the first loops are the slow ones
        pe = probe['ecut']          # scale by the number of Fock-grid points (default ecutfock, see above)
        per_pair_core_s = t_loop * probe['np'] * (min(ecut[1], 4 * ecut[0]) / min(pe[1], 4 * pe[0])) ** 1.5
        basis = f"measured: hybrid_probe {probe['tag']} ({t_loop} s per outer loop on {probe['np']} ranks)"
        status = 'scaled from a measured probe (Gamma, 1 k x 1 q); scaling to many k/q points is an assumption'
    else:
        p = parse_pw(out / 'runs' / r['tag'] / f"{r['tag']}.out", cr.nat)
        n_dense = int(np.prod(p['fft_dense'])) * (ecut[1] / r['ecut'][1]) ** 1.5
        n_fock = n_dense * (min(ecut[1], 4 * ecut[0]) / ecut[1]) ** 1.5      # default ecutfock grid
        nocc = int(round(p['nelec'] / 2))
        # 2 FFTs per pair, 5 N log2 N flops at ~2 GFLOP/s per core, times 2-4 for the US/PAW augmentation of the
        # pair densities.  The upper factor is set by a Pi probe (25/200 Ry, Gamma, 2 ranks, 27 Sep 2026) whose first
        # ACE build had not finished after ~1000 s, i.e. >= 5x the bare FFT count on that machine (UNVERIFIED).
        per_pair_core_s = nocc ** 2 * 2 * 5 * n_fock * math.log2(n_fock) / 2e9 * np.array([2.0, 4.0])
        basis = 'FFT-count model (no complete probe)'
        status = 'UNVERIFIED order-of-magnitude estimate'
    est = {}
    for label, nq in (('nq 1x1x1', 1), (f"nq {'x'.join(map(str, kg))} (QE 7.5 default = k grid)", nq_full)):
        pp = np.atleast_1d(per_pair_core_s)
        lo = float(pbe_core_s + n_outer[0] * (nk_irr * nq * pp.min() + pbe_core_s * 0.5))
        hi = float(pbe_core_s + n_outer[1] * (nk_irr * nq * pp.max() + pbe_core_s * 0.5))
        # 24 MPI ranks at ~75% parallel efficiency
        est[label] = {'core_h': [round(lo / 3600, 1), round(hi / 3600, 1)],
                      'wall_h_24_ranks': [round(lo / 3600 / 24 / 0.75, 2), round(hi / 3600 / 24 / 0.75, 2)]}
    res = {'setting': {'ecut_Ry': list(ecut), 'k': 'x'.join(map(str, kg)), 'nk_irr': nk_irr},
           'pbe_scf_core_h': round(pbe_core_s / 3600, 3), 'exx_per_kq_pair_per_loop_core_s': np.round(np.atleast_1d(per_pair_core_s), 1).tolist(),
           'outer_loops_assumed': list(n_outer), 'estimate_per_single_point': est, 'basis': basis, 'status': status,
           'ecutfock_Ry': min(ecut[1], 4 * ecut[0]),
           'notes': ['Two single points are needed for Delta E: relax_i and the neutral form (relax_ii if it stays '
                     'on O5, otherwise the constrained relax_iifix).  Because PBE biases acid-base crystals towards '
                     'the salt (LeBlanc et al. 2018, doi:10.1002/anie.201809381), this hybrid check is what tests '
                     'the PBE conclusion.  A hybrid relaxation would cost roughly the per-point figure times the '
                     'number of BFGS steps and is not proposed.',
                     'nq 1x1x1 is the usual cheap choice for an insulator with a 3x2x2 k grid; the QE 7.5 default '
                     '(nq = k grid) is ~12x dearer.']}
    atomic_json(out / 'hybrid_estimate.json', res)
    note(out, f'estimate: hybrid single point at {res["setting"]}: {est} ({basis})')
    return res


def parse_exx_timing(path):
    """Wall-clock marks of a hybrid run: end of the first (PBE) SCF and of each EXX outer loop, in CPU seconds of
    the printing rank (one OpenMP thread per rank, so CPU ~ wall)."""
    txt = Path(path).read_text(errors='replace')
    marks, t = [], None
    for line in txt.splitlines():
        m = re.search(r'total cpu time spent up to now is\s+([\d.]+) secs', line)
        if m:
            t = float(m.group(1))
        if 'EXX: now go back to refine exchange calculation' in line or 'EXX self-consistency reached' in line:
            marks.append(t)
    dexx = [float(x) for x in re.findall(r'est\. exchange err \(dexx\)\s*=\s*([-\d.E+]+) Ry', txt)]
    inner = [len(b.split('iteration #')) - 1 for b in txt.split('EXX: now go back to refine exchange calculation')]
    return {'marks_cpu_s': marks, 'dexx_Ry': dexx, 'scf_iterations_per_outer_loop': inner}


def stage_hybrid_probe(args, cr):
    """Measure, do not run to convergence: one PBE0 SCF at start i with the cheapest setting (50/400 Ry, k and
    q = Gamma via a 1x1x1 grid), stopped by max_seconds after the first ACE outer loops.  The measured time per
    ACE outer loop (per k,q pair) is what the 'estimate' stage scales to the production grids."""
    check_pseudos(args)
    ecut = tuple(map(int, (args.ecut or '50/400').split('/')))
    kg = tuple(map(int, (args.kgrid or '1x1x1').split('x')))
    tag = f'probe_pbe0_e{ecut[0]}_k{"".join(map(str, kg))}'
    p = run_pw(args, cr, tag, 'scf', cr.F0, ecut, kg, conv_thr=1e-7,
               extra_ctrl=[f'max_seconds = {args.probe_seconds}'],
               extra_sys=["input_dft = 'pbe0'", 'nqx1 = 1', 'nqx2 = 1', 'nqx3 = 1'], probe=True)
    t = parse_exx_timing(Path(args.out) / 'runs' / tag / f'{tag}.out')
    marks = t['marks_cpu_s']
    res = {'tag': tag, 'ecut': ecut, 'kgrid': kg, 'np': args.np, 'nk_irr': p['nk_irr'], 'fft_dense': p['fft_dense'],
           'nbnd': p['nbnd'], **t,
           'first_scf_and_ace_build_s': marks[0] if marks else None,
           'exx_outer_loop_s': [round(b - a, 1) for a, b in zip(marks, marks[1:])]}
    atomic_json(Path(args.out) / 'hybrid_probe.json', res)
    note(Path(args.out), f'hybrid probe: PBE SCF + first ACE build {res["first_scf_and_ace_build_s"]} s, '
         f'EXX outer loops {res["exx_outer_loop_s"]} s '
         f'on {args.np} ranks')
    return res


# ----------------------------------------------------------------------------------------------------------------
# hybrid (PBE0-D3(BJ)) single points at the PBE-D3(BJ) geometries
# ----------------------------------------------------------------------------------------------------------------
_HTAG = re.compile(r'^hybrid_(prof\d{3}(?:_i|_prof\d{3})?|iirfix|iifix|iir1|iir|ii|i)(?:_(.+))?$')


def input_positions(inp, nat):
    """Fractional positions written in a pw.x input by pw_input()."""
    txt = Path(inp).read_text()
    blk = txt.split('ATOMIC_POSITIONS crystal\n', 1)[1].splitlines()[:nat]
    return np.array([[float(v) for v in l.split()[1:4]] for l in blk])


def input_value(inp, key):
    m = re.search(rf'^\s*{key}\s*=\s*(\S+)', Path(inp).read_text(), re.M)
    return m.group(1).strip("'") if m else None


def _signature(txt):
    """A pw.x input without the lines that do not change the physics (paths, time and loop caps)."""
    skip = ('outdir', 'pseudo_dir', 'max_seconds', 'exx_maxstep', 'prefix')
    return [l.strip() for l in txt.splitlines() if l.strip() and l.split('=')[0].strip() not in skip]


def geom_dirs(args):
    """Results directories searched for runs/relax_<s>: --geom-from first (as before), then --out (profile points and
    iir1 live in --out when a sandbox test takes relax_i / relax_iirfix from another directory)."""
    return list(dict.fromkeys(str(Path(d)) for d in (args.geom_from, args.out) if d))


def relaxed_geometry(cr, gdir, s):
    """Final positions, energy and setting of a converged relax_<s> (the geometry of a hybrid single point); gdir is
    one results directory or a list searched in order."""
    dirs = [gdir] if isinstance(gdir, (str, Path)) else list(gdir)
    cands = [Path(g) / 'runs' / f'relax_{s}' / f'relax_{s}.out' for g in dirs]
    f = next((c for c in cands if c.exists()), cands[0])
    if not f.exists():
        raise SystemExit(f'hybrid: {f} not found (run relax --start {s} first, or pass --geom-from)')
    p = parse_pw(f, cr.nat)
    if not (p['job_done'] and p['relax_converged'] and p['positions_crystal']):
        raise SystemExit(f'hybrid: relax_{s} has not converged ({f})')
    ecut, k = run_setting(f.parent / f'relax_{s}.in')
    return (np.array(p['positions_crystal']), p['final_energy_Ry'] or p['energy_Ry'], ecut, k,
            f, hashlib.sha256(f.read_bytes()).hexdigest())


def hybrid_suffix(sg15, ecut, kstr, ecut_r, k_r, nq, ecutfock, ef_default, conv_thr):
    """Tag suffix of a hybrid variant: '' = SSSP at the relax setting with the defaults; 'sg15_e<ecutwfc>' = SG15 at the
    relax k grid with the defaults; further parts for anything else (k, nq, ecutfock, conv_thr)."""
    suf = f'_sg15_e{ecut[0]}' if sg15 else ('' if list(ecut) == list(ecut_r) else f'_e{ecut[0]}')
    if kstr != k_r:
        suf += '_k' + kstr.replace('x', '')
    if tuple(nq) != (1, 1, 1):
        suf += '_nq' + ''.join(map(str, nq))
    if ecutfock != ef_default:
        suf += f'_ef{ecutfock:g}'
    if conv_thr != HYBRID_CONV_THR:
        suf += f'_thr{conv_thr:.0e}'
    return suf


def stage_hybrid(args, cr):
    """PBE0-D3(BJ) SCF single point at the final geometry of relax_<start> (no forces: QE 7.5 has none for hybrid +
    US/PAW, and none are needed).  --pp sg15 (default): SG15 norm-conserving potentials at --ecut (default 100/400 Ry),
    plus a PBE-D3(BJ) companion SCF with the same potentials and setting, so that the hybrid correction
    E(PBE0-D3) - E(PBE-D3) is computed with everything else equal; --pp sssp: the relax potentials (G-space
    augmentation, ~50x dearer, see the docstring), at the relax setting unless --ecut/--kgrid say otherwise.
    Runs and parsed records land in runs/hybrid_<s><suffix>/ and runs/pbe_<s><suffix>/; 'analyse' builds
    periodic_qe.json['hybrid']."""
    sg15 = args.pp == 'sg15'
    pseudos, pdir = (PSEUDOS_SG15, args.pseudo_sg15) if sg15 else (PSEUDOS, args.pseudo)
    check_pseudos(args, pseudos, pdir)
    starts = ['i', 'iirfix'] if args.start == 'both' else [args.start]
    nq = tuple(int(x) for x in args.nq.split('x'))
    gdir = geom_dirs(args)
    failed = []
    for s in starts:
        F, e_relax, ecut_r, k_r, rout, rsha = relaxed_geometry(cr, gdir, s)
        ecut = tuple(map(int, args.ecut.split('/'))) if args.ecut else (HYBRID_SG15_ECUT if sg15 else tuple(ecut_r))
        kstr = args.kgrid or k_r
        k = tuple(map(int, kstr.split('x')))
        if sg15 and ecut[1] != 4 * ecut[0]:
            raise SystemExit(f'hybrid: norm-conserving SG15 needs ecutrho = 4 ecutwfc, got {ecut}')
        ef_default = min(ecut[1], 4 * ecut[0])         # QE 7.5 default; setup.f90 forces it for US/PAW
        ecutfock = args.ecutfock or ef_default
        if not ecut[0] <= ecutfock <= ecut[1]:
            raise SystemExit(f'hybrid: ecutfock {ecutfock} must lie between ecutwfc and ecutrho {ecut}')
        if not sg15 and ecutfock != ef_default:
            raise SystemExit('hybrid: with US/PAW potentials QE 7.5 resets ecutfock to MIN(4 ecutwfc, ecutrho) '
                             '(PW/src/setup.f90, "ecutfock not valid for US/PAW, ignored")')
        suf = hybrid_suffix(sg15, ecut, kstr, ecut_r, k_r, nq, ecutfock, ef_default, args.conv_thr)
        tag, ptag = f'hybrid_{s}{suf}', f'pbe_{s}{suf}'
        pp_name = 'SG15 ONCV v1.2 (norm-conserving, no NLCC)' if sg15 else 'SSSP PBE efficiency 1.3.0 (US/PAW)'
        # PBE-D3(BJ) with the same potentials and setting; for SSSP at the relax setting the relax energy is that number
        need_pbe = sg15 or [list(ecut), kstr] != [list(ecut_r), k_r]
        if need_pbe:
            try:
                run_pw(args, cr, ptag, 'scf', F, ecut, k, conv_thr=1e-8, tprnfor=False, pseudos=pseudos, pseudo_dir=pdir,
                       log_extra={'input_dft': 'pbe (from the potentials)', 'pseudopotentials': pp_name})
            except RuntimeError as e:
                print(f'[periodic_qe] {e}', flush=True)
                failed.append(ptag)
                continue
        extra_sys = ["input_dft = 'pbe0'", f'nqx1 = {nq[0]}', f'nqx2 = {nq[1]}', f'nqx3 = {nq[2]}',
                     f'ecutfock = {float(ecutfock):.1f}', 'ace = .true.',
                     "exxdiv_treatment = 'gygi-baldereschi'", 'x_gamma_extrapolation = .true.']
        extra_elec = [f'exx_maxstep = {args.exx_maxstep}'] if args.exx_maxstep else []
        extra_ctrl = [f'max_seconds = {args.max_seconds}'] if args.max_seconds else []
        rd = Path(args.out) / 'runs' / tag
        inp, outf, jf = rd / f'{tag}.in', rd / f'{tag}.out', rd / f'{tag}.json'
        if outf.exists() and inp.exists():
            # never reuse a finished run whose input differs (e.g. relax_<s> was redone since)
            new = pw_input(cr, F, tag, 'scf', ecut, k, 'X', 'X', conv_thr=args.conv_thr, extra_sys=extra_sys,
                           tprnfor=False, extra_elec=extra_elec, pseudos=pseudos)
            done = parse_hybrid(outf)['exx_converged']
            if done and _signature(new) != _signature(inp.read_text()):
                raise SystemExit(f'hybrid: {rd} holds a finished run with a different input (geometry or setting); '
                                 f'move it away to run again')
            if done and jf.exists():
                print(f'[periodic_qe] {tag}: finished earlier, reused (record kept: {jf})', flush=True)
                continue
        meta = {'tag': tag, 'start': s, 'what': f'PBE0-D3(BJ) SCF single point at the final geometry of relax_{s}',
                'pseudopotentials': pp_name, 'pbe_companion': ptag if need_pbe else f'relax_{s} (same setting)',
                'geometry_from': str(rout), 'geometry_out_sha256': rsha, 'pbe_d3_relax_energy_Ry': e_relax,
                'relax_setting': {'ecut_Ry': ecut_r, 'k': k_r}, 'ecut_Ry': list(ecut), 'k': kstr, 'nq': list(nq),
                'ecutfock_Ry': ecutfock, 'ace': True, 'conv_thr_Ry': args.conv_thr,
                'exx_maxstep': args.exx_maxstep or 100, 'max_seconds': args.max_seconds or None, 'np': args.np,
                'utc_start': utc()}
        try:
            run_pw(args, cr, tag, 'scf', F, ecut, k, conv_thr=args.conv_thr, tprnfor=False, extra_ctrl=extra_ctrl,
                   extra_sys=extra_sys, extra_elec=extra_elec, hybrid=True, pseudos=pseudos, pseudo_dir=pdir,
                   log_extra={'input_dft': 'pbe0', 'nq': list(nq), 'ecutfock_Ry': ecutfock, 'pseudopotentials': pp_name})
            ok = True
        except RuntimeError as e:
            ok = False
            print(f'[periodic_qe] {e}', flush=True)
        h = parse_hybrid(outf)
        meta.update({'utc_end': utc(), 'converged': ok, 'energy_Ry': h['energy_Ry'],
                     'parts_before': len(list(rd.glob(f'{tag}.out.part*'))), 'hybrid': h})
        atomic_json(jf, meta)
        if not ok:
            failed.append(tag)
    if failed:
        raise SystemExit(f'hybrid: EXX not converged / pw.x failed for {failed}; partial record in runs/<tag>/<tag>.json '
                         f"('analyse' reports the last outer-loop energies as provisional)")


def d3_reference(cr, F):
    """simple-dftd3 two-body D3(BJ) energies (Ry) of one structure with QE's 'pbe' and 'pbe0' parameters (explicit,
    D3BJ_QE) and with simple-dftd3's own named 'pbe0' set, or None if the dftd3 module is absent (it is in
    tools/env, not tools/qe).  Real-space cutoffs as QE's dft-d3 (rthr 9000 bohr^2 ~ 95 bohr, cn_thr 1600 bohr^2 =
    40 bohr); this reproduced QE's PBE-D3 energy of relax_i/ii/iifix to 2.5e-7 Ry (28 Sep 2026)."""
    try:
        from dftd3.interface import RationalDampingParam, DispersionModel
    except Exception:
        return None
    bohr = 0.529177210903
    Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16}
    num = np.array([Z[s] for s in cr.syms])
    sets = {name: RationalDampingParam(s6=p['s6'], a1=p['a1'], s8=p['s8'], a2=p['a2'], s9=0.0)
            for name, p in D3BJ_QE.items()}
    sets['pbe0_simple_dftd3_named'] = RationalDampingParam(method='pbe0', atm=False)
    out = {}
    for name, par in sets.items():
        m = DispersionModel(num, cr.cart(F) / bohr, lattice=cr.lat / bohr, periodic=np.array([True] * 3))
        m.set_realspace_cutoff(95.0, 40.0, 40.0)
        out[name] = round(float(m.get_dispersion(par, grad=False)['energy']) * 2, 9)
    return out


def _mev(dRy):
    return round(dRy * RY_EV * 1000, 2)


def _kj_fu(dRy):
    return round(dRy * RY_EV * KJMOL_PER_EV / 2, 3)          # Z = 2 formula units per cell


def hybrid_point(cr, d, rl):
    """One hybrid run directory (+ its PBE-D3 companion) -> record: energies, EXX convergence, timings, D3 checks."""
    tag = d.name
    f, inp = d / f'{tag}.out', d / f'{tag}.in'
    s = _HTAG.match(tag).group(1)
    h = parse_hybrid(f)
    p = parse_pw(f, cr.nat)
    ecut, k = run_setting(inp)
    F = input_positions(inp, cr.nat)
    sg15 = 'ONCV' in inp.read_text()
    pt = {'tag': tag, 'start': s, 'pseudopotentials': 'SG15' if sg15 else 'SSSP', 'ecut_Ry': ecut, 'k': k,
          'nq': h['nq'], 'ecutfock_Ry': h['ecutfock_Ry'],
          'conv_thr_Ry': float(input_value(inp, 'conv_thr').replace('d', 'e')), 'xc': h['xc'],
          'exx_fraction': h['exx_fraction'], 'ace': h['ace'], 'exx_converged': h['exx_converged'],
          'energy_Ry': h['energy_Ry'], 'last_outer_energy_Ry': h['last_outer_energy_Ry'],
          'dexx_Ry': [lp['dexx_Ry'] for lp in h['outer_loops']],
          'fock_energy_Ry': (h['outer_loops'][-1].get('fock_energy_Ry') if h['outer_loops'] else None),
          'first_scf_energy_Ry': h['first_scf_energy_Ry'], 'stopped_early': h['stopped_early'],
          'restarts': len(list(d.glob(f'{tag}.out.part*'))), 'wall_s': p['wall_s'], 'mpi': p['mpi'],
          'npool': p['npool'], 'nk_irr': p['nk_irr'], 'n_symops': p['n_symops'], 'qe_version': p['qe_version'],
          'exx_grid': h['exx_grid'], 'clocks_wall_s': h['clocks_wall_s'], 'inner_scf_iterations': h['inner_scf_iterations']}
    # PBE-D3(BJ) at the same potentials and setting: the companion SCF, or the relax itself (SSSP at the relax setting)
    comp = d.parent / ('pbe_' + tag[len('hybrid_'):])
    if (comp / f'{comp.name}.out').exists():
        c = parse_pw(comp / f'{comp.name}.out', cr.nat)
        if c['job_done'] and not c['scf_failed'] and c['energy_Ry'] is not None:
            pt['pbe_d3_same_setting'] = {'source': comp.name, 'energy_Ry': c['energy_Ry'], 'dispersion_Ry': c['dispersion_Ry'],
                                         'wall_s': c['wall_s']}
    if s in rl:
        pt['geometry_max_dev_from_relax_frac'] = float(np.abs(cr.unwrap(F) - cr.unwrap(rl[s]['_F'])).max())
        pt['pbe_d3_relax_sssp'] = {'energy_Ry': rl[s]['energy_Ry'], 'dispersion_Ry': rl[s]['dispersion_Ry'],
                                   'ecut_Ry': rl[s]['ecut_Ry'], 'k': rl[s]['k']}
        if 'pbe_d3_same_setting' not in pt and not sg15 and rl[s]['ecut_Ry'] == ecut and rl[s]['k'] == k:
            pt['pbe_d3_same_setting'] = {'source': f'relax_{s}', 'energy_Ry': rl[s]['energy_Ry'],
                                         'dispersion_Ry': rl[s]['dispersion_Ry']}
    if 'pbe_d3_same_setting' in pt and h['first_scf_energy_Ry'] is not None:
        # the first SCF of a PBE0 run is PBE with the pbe0 D3 parameters: minus the PBE-D3 energy at the same geometry,
        # potentials and setting it is D3(pbe0 params) - D3(pbe params), up to SCF convergence
        pt['first_scf_minus_pbe_d3_Ry'] = round(h['first_scf_energy_Ry'] - pt['pbe_d3_same_setting']['energy_Ry'], 8)
    ref = d3_reference(cr, F)
    if ref:
        dc = pt['d3_check'] = {'simple_dftd3_two_body_Ry': ref,
                               'expected_shift_pbe0_minus_pbe_params_Ry': round(ref['pbe0'] - ref['pbe'], 8)}
        disp = pt.get('pbe_d3_same_setting', {}).get('dispersion_Ry')
        if disp is not None:
            dc['qe_pbe_dispersion_minus_simple_dftd3_Ry'] = round(disp - ref['pbe'], 8)
        if 'first_scf_minus_pbe_d3_Ry' in pt:
            dc['residual_Ry'] = round(pt['first_scf_minus_pbe_d3_Ry'] - dc['expected_shift_pbe0_minus_pbe_params_Ry'], 8)
            dc['residual_means'] = 'SCF noise (<~1e-6 Ry) if QE used the pbe0 D3(BJ) parameters in the PBE0 run'
    return pt


def compare_hybrid(a, b):
    """b (start s) against a (start i) in one variant.  PBE0-D3(BJ) difference; hybrid correction = PBE0-D3(BJ)
    minus PBE-D3(BJ) with the same potentials/setting; best estimate = PBE-D3(BJ)/SSSP relax difference + correction;
    split of the correction into the D3-parameter part (first SCF minus PBE-D3) and the exact-exchange part."""
    same = all(a[x] == b[x] for x in ('ecut_Ry', 'k', 'nq', 'ecutfock_Ry', 'conv_thr_Ry', 'pseudopotentials'))
    conv = a['exx_converged'] and b['exx_converged']
    Ea = a['energy_Ry'] if conv else a['last_outer_energy_Ry']
    Eb = b['energy_Ry'] if conv else b['last_outer_energy_Ry']
    c = {'same_setting': same, 'converged': conv, 'pseudopotentials': a['pseudopotentials'], 'ecut_Ry': a['ecut_Ry'],
         'k': a['k'], 'nq': a['nq']}
    if Ea is None or Eb is None or not same:
        c['status'] = 'no comparison: ' + ('settings differ' if not same else 'a hybrid energy is missing')
        return c
    dE = Eb - Ea
    c.update({'PBE0_D3BJ_meV_cell': _mev(dE), 'PBE0_D3BJ_kJ_mol_per_formula_unit': _kj_fu(dE)})
    corr = None
    if 'pbe_d3_same_setting' in a and 'pbe_d3_same_setting' in b:
        dS = b['pbe_d3_same_setting']['energy_Ry'] - a['pbe_d3_same_setting']['energy_Ry']
        corr = dE - dS
        c.update({'PBE_D3BJ_same_setting_meV_cell': _mev(dS),
                  'hybrid_correction_meV_cell': _mev(corr), 'hybrid_correction_kJ_mol_per_formula_unit': _kj_fu(corr)})
        if a['first_scf_energy_Ry'] is not None and b['first_scf_energy_Ry'] is not None:
            d1 = b['first_scf_energy_Ry'] - a['first_scf_energy_Ry']
            split = {'exact_exchange_25pct': _mev(dE - d1), 'D3_parameters_pbe0_vs_pbe': _mev(d1 - dS)}
            ea = a.get('d3_check', {}).get('expected_shift_pbe0_minus_pbe_params_Ry')
            eb = b.get('d3_check', {}).get('expected_shift_pbe0_minus_pbe_params_Ry')
            if ea is not None and eb is not None:
                split['D3_parameters_simple_dftd3'] = _mev(eb - ea)
            split['how'] = ('first SCF of each PBE0 run = PBE electrons with the pbe0 D3(BJ) parameters; D3 part = '
                            'first-SCF difference minus PBE-D3 difference; exact-exchange part = the rest')
            c['split_of_the_correction_meV_cell'] = split
    if 'pbe_d3_relax_sssp' in a and 'pbe_d3_relax_sssp' in b:
        dP = b['pbe_d3_relax_sssp']['energy_Ry'] - a['pbe_d3_relax_sssp']['energy_Ry']
        c.update({'PBE_D3BJ_meV_cell': _mev(dP), 'PBE_D3BJ_kJ_mol_per_formula_unit': _kj_fu(dP),
                  'PBE_D3BJ_note': 'SSSP relaxes (70/560 Ry, 2x1x1): the production PBE-D3(BJ) value'})
        if corr is not None:
            c.update({'best_estimate_PBE0_D3BJ_meV_cell': _mev(dP + corr),
                      'best_estimate_PBE0_D3BJ_kJ_mol_per_formula_unit': _kj_fu(dP + corr),
                      'best_estimate_how': 'PBE-D3(BJ)/SSSP relax difference + hybrid correction (PBE0-D3(BJ) minus '
                                           'PBE-D3(BJ) with the same potentials, cutoff and k grid)'})
            if 'PBE_D3BJ_same_setting_meV_cell' in c and c['pseudopotentials'] == 'SG15':
                c['PBE_D3BJ_SG15_minus_SSSP_meV_cell'] = round(c['PBE_D3BJ_same_setting_meV_cell'] - _mev(dP), 2)
    if conv:
        c['status'] = 'converged (EXX dexx < conv_thr in both runs)'
    else:
        dx = [x['dexx_Ry'][-1] for x in (a, b) if x['dexx_Ry']]
        c['status'] = ('PROVISIONAL: EXX not converged in ' + ', '.join(x['tag'] for x in (a, b) if not x['exx_converged'])
                       + f'; last printed outer-loop energies used, largest last dexx {max(dx) if dx else None} Ry')
    return c


def primary_variant(pts):
    """The variant reported at the top of periodic_qe.json['hybrid']: SSSP at the relax setting if it was run, else the
    SG15 variant with the defaults (nq 1x1x1, relax k grid, default ecutfock / conv_thr) at the highest cutoff."""
    if 'sssp' in pts:
        return 'sssp'
    sg = [(int(m.group(1)), v) for v in pts for m in [re.fullmatch(r'sg15_e(\d+)', v)] if m]
    return max(sg)[1] if sg else None


def hybrid_summary(out, cr, rl):
    """periodic_qe.json['hybrid']: every hybrid_* run, grouped by variant (the tag suffix; 'sssp' = SSSP at the relax
    setting, 'sg15_e<ecutwfc>' = SG15 defaults), E(s) - E(i) per variant, and the primary result on top."""
    pts = {}
    for d in sorted((Path(out) / 'runs').glob('hybrid_*')):
        m = _HTAG.match(d.name)
        # profile points and iir1 are summarised in profile.json (profile_analyse), not here
        if not m or m.group(1) not in CORE_STARTS or not (d / f'{d.name}.out').exists() \
                or not (d / f'{d.name}.in').exists():
            continue
        pts.setdefault(m.group(2) or 'sssp', {})[m.group(1)] = hybrid_point(cr, d, rl)
    if not pts:
        return None
    comps = {v: {f'E_{s}_minus_i': compare_hybrid(byS['i'], byS[s]) for s in ('iirfix', 'iifix', 'ii', 'iir')
                 if s in byS and 'i' in byS}
             for v, byS in pts.items()}
    res = {'method': {
        'functional': 'PBE0 (input_dft=pbe0, 25% exact exchange), SCF single points at the PBE-D3(BJ) relaxed '
                      'geometries (fixed X-ray cell, no re-relaxation), k grid of the relaxes (2x1x1)',
        'dispersion': 'D3(BJ) two-body (dftd3_version=4, dftd3_threebody=.false.); QE 7.5 takes the parameters from '
                      'the functional (get_dft_short -> dftd3_xc -> setfuncpar): pbe0 ' + json.dumps(D3BJ_QE['pbe0'])
                      + ', pbe ' + json.dumps(D3BJ_QE['pbe']) + ' (dft-d3/core.f90, tag qe-7.5; verified numerically '
                      'against simple-dftd3, d3_check)',
        'pseudopotentials': 'variant sg15_*: SG15 ONCV v1.2 norm-conserving PBE (Schlipf & Gygi, Comput. Phys. Commun. '
                            '196 (2015) 36; no nonlinear core correction), ecutrho = ecutfock = 4 ecutwfc; the hybrid '
                            'correction is added to the SSSP PBE-D3(BJ) difference.  Variant sssp: the relax potentials '
                            '(G-space augmentation).  Why SG15: see the periodic_qe.py docstring (cost of US/PAW exact '
                            'exchange; QE 7.5 tqr bug)',
        'exact_exchange': 'ACE (adaptively compressed exchange, QE default ace=.true.); q mesh nqx per variant '
                          "(default 1x1x1); exxdiv_treatment 'gygi-baldereschi' + x_gamma_extrapolation; EXX converged "
                          'when dexx < conv_thr',
        'units': 'meV per cell (54 atoms, Z = 2); kJ/mol per formula unit = per cell / 2'},
        'points': pts, 'comparisons': comps}
    pv = primary_variant(pts)
    # the neutral-form comparison: iirfix (H1A and O5 fixed); iifix only if iirfix is absent (iifix slid back to a salt)
    s = next((x for x in ('iirfix', 'iifix') if f'E_{x}_minus_i' in comps.get(pv, {})), None) if pv else None
    prod = comps[pv][f'E_{s}_minus_i'] if s else None
    if prod:
        key = f'E_{s}_minus_i'
        res['primary_variant'] = pv
        res['primary_comparison'] = key
        res[key] = prod
        others = {v: comps[v][key].get('hybrid_correction_meV_cell') for v in comps if v != pv and key in comps[v]}
        if others:
            res[f'{key}_correction_other_variants_meV_cell'] = others
        on = ((rl.get(s) or {}).get('geometry') or {}).get('proton', {}).get('proton_on')
        warn = '' if on == 'O' else (f'; WARNING: relax_{s} has its proton on {on or "?"}, so this is NOT a salt-vs-neutral '
                                     f'comparison')
        best = prod.get('best_estimate_PBE0_D3BJ_meV_cell', prod.get('PBE0_D3BJ_meV_cell'))
        if best is not None and 'PBE_D3BJ_meV_cell' in prod:
            bkj = prod.get('best_estimate_PBE0_D3BJ_kJ_mol_per_formula_unit', prod.get('PBE0_D3BJ_kJ_mol_per_formula_unit'))
            corr = (f"; hybrid correction {prod['hybrid_correction_meV_cell']:+.1f} meV/cell ({pv})"
                    if 'hybrid_correction_meV_cell' in prod else '')
            res['verdict'] = (f"{'PROVISIONAL ' if not prod['converged'] else ''}E({s}) - E(i): PBE0-D3(BJ) "
                              f"{best:+.1f} meV/cell ({bkj:+.2f} kJ/mol per f.u.) vs PBE-D3(BJ) "
                              f"{prod['PBE_D3BJ_meV_cell']:+.1f} meV/cell ({prod['PBE_D3BJ_kJ_mol_per_formula_unit']:+.2f})"
                              f"{corr}; {'salt (i) lower' if best > 0 else f'constrained neutral form ({s}) lower or equal'} "
                              f"at PBE0-D3(BJ){warn}")
    return res


# ----------------------------------------------------------------------------------------------------------------
# constrained N1-H1A proton-transfer profile and the single-formula-unit control (referee M2/M3, 29 Sep 2026)
# ----------------------------------------------------------------------------------------------------------------
def min_distance(cr, F):
    """Shortest interatomic distance (A) in the periodic crystal (overlap check of a constructed start)."""
    X = cr.cart(cr.unwrap(np.asarray(F, float)))
    dmin = 9.0
    for sh in np.ndindex(3, 3, 3):
        t = (np.array(sh) - 1) @ cr.lat
        dd = np.linalg.norm(X[:, None, :] - (X[None, :, :] + t), axis=2)
        if sh == (1, 1, 1):
            np.fill_diagonal(dd, 9.0)
        dmin = min(dmin, float(dd.min()))
    return dmin


def proton_fu(cr, F, copy):
    """The N1-H1A...O5 proton, P1-O5 and the O4-H4A...O5 bond of one formula unit (copy 0 = x,y,z; 1 = the glide)."""
    by = {(h['D'], h['H'], h['A']): h for h in cr.hbonds()}
    n1, o4 = by[('N1', 'H1A', 'O5')], by[('O4', 'H4A', 'O5')]
    sN, sH, sO = cr.hbond_sites(n1, copy)
    dNH, dHO = cr.dist(F, sN, sH), cr.dist(F, sH, sO)
    sP = cr.compose(copy, 'P1', n1['code_A'])
    s4, s4H, s4A = cr.hbond_sites(o4, copy)
    return {'d_N1_H1A_A': round(dNH, 4), 'd_H1A_O5_A': round(dHO, 4), 'delta_A': round(dNH - dHO, 4),
            'd_N1_O5_A': round(cr.dist(F, sN, sO), 4), 'angle_N1_H1A_O5_deg': round(cr.angle(F, sN, sH, sO), 2),
            'd_P1_O5_A': round(cr.dist(F, sP, sO), 4), 'proton_on': 'N' if dNH < dHO else 'O',
            'O4_H4A_O5': {'d_O4_H4A_A': round(cr.dist(F, s4, s4H), 4), 'd_H4A_O5_A': round(cr.dist(F, s4H, s4A), 4),
                          'd_O4_O5_A': round(cr.dist(F, s4, s4A), 4)}}


def profile_name(d, seed='interp'):
    """prof110 = d(N1-H1A) 1.10 A from the default (interpolated) start; prof140_prof155 = 1.40 A seeded from the
    converged 1.55 A point; prof140_i = 1.40 A from relax_i with the proton moved on the N1->O5 line."""
    base = f'prof{int(round(d * 100)):03d}'
    return base if seed == 'interp' else f'{base}_{seed}'


def profile_seed(args, cr, d):
    """Start geometry of one profile point (see --seed and the docstring) and a record of how it was made."""
    dirs = geom_dirs(args)
    Fi, _, _, _, fi, shai = relaxed_geometry(cr, dirs, 'i')
    rec = {'seed': args.seed, 'relax_i': str(fi), 'relax_i_out_sha256': shai}
    if args.seed == 'interp':
        Ff, _, _, _, ff, shaf = relaxed_geometry(cr, dirs, 'iirfix')
        Ui, Uf = cr.unwrap(Fi), cr.unwrap(Ff)
        di, df = proton_fu(cr, Ui, 0)['d_N1_H1A_A'], proton_fu(cr, Uf, 0)['d_N1_H1A_A']
        w = float(np.clip((d - di) / (df - di), 0.0, 1.0))
        F = cr.set_nh(Ui + w * (Uf - Ui), d)
        rec.update({'relax_iirfix': str(ff), 'relax_iirfix_out_sha256': shaf, 'd_N1_H1A_relax_i_A': di,
                    'd_N1_H1A_relax_iirfix_A': df, 'w': round(w, 4),
                    'how': 'every atom interpolated relax_i -> relax_iirfix at w, then H1A put d from N1 along the '
                           'interpolated N1->H1A direction (both formula units)'})
    elif args.seed == 'i':
        F = cr.set_nh(Fi, d, line_to_o5=True)
        rec['how'] = 'relax_i with H1A put d from N1 on the N1->O5 line (both formula units)'
    else:
        Fp, _, _, _, fp, shap = relaxed_geometry(cr, dirs, args.seed)
        F = cr.set_nh(Fp, d)
        rec.update({'from': str(fp), 'from_out_sha256': shap,
                    'how': f'converged relax_{args.seed} with H1A put d from N1 along its N1->H1A direction'})
    rec['start_geometry'] = [proton_fu(cr, F, c) for c in range(len(cr.ops))]
    rec['min_interatomic_A'] = round(min_distance(cr, F), 4)
    return F, rec


def profile_falls_before(args, cr, d):
    """--if-falling: True if the profile still falls at the largest converged default-start point below d (slope
    from the constraint force < 0, or its energy below that of the point before it); otherwise the reason (str)."""
    runs = Path(args.out) / 'runs'
    got = []
    for rd in sorted(runs.glob('relax_prof*')):
        m = _PROF.match(rd.name[len('relax_'):])
        f = rd / f'{rd.name}.out'
        if m and m.group(2) is None and int(m.group(1)) < round(d * 100) and f.exists():
            r = profile_point(cr, f, None)
            if r and r['converged']:
                got.append((int(m.group(1)), r))
    if not got:
        return 'no converged profile point below it'
    got.sort(key=lambda x: x[0])
    dd, last = got[-1]
    slope = (last.get('slope') or {}).get('dE_dd_cell_meV_A')
    if slope is not None and slope < 0:
        return True
    if len(got) > 1 and last['energy_Ry'] < got[-2][1]['energy_Ry']:
        return True
    return f'the profile rises at d = {dd / 100:.2f} A (slope {slope} meV/A per cell): no extension needed'


def stage_profile(args, cr):
    """Constrained relaxation at d(N1-H1A) = each --d (both formula units), everything else free: damped dynamics
    (or FIRE) with pw.x CONSTRAINTS, fixed cell, the production setting (normally --ecut 70/560 --kgrid 2x1x1, the
    setting of relax_i).  Resumable: a finished point is reused, an interrupted one restarts from its last geometry
    (kept as .out.partN)."""
    check_pseudos(args)
    if args.seed not in ('interp', 'i') and not _PROF.match(args.seed):
        raise SystemExit(f'profile: --seed must be interp, i or prof<ddd>, not {args.seed}')
    ecut, k = chosen_setting(args)
    out = Path(args.out)
    _, _, ecut_i, k_i, _, _ = relaxed_geometry(cr, geom_dirs(args), 'i')
    same = [list(ecut), 'x'.join(map(str, k))] == [ecut_i, k_i]
    if not same:
        note(out, f'profile: WARNING setting {ecut} {k} differs from relax_i ({ecut_i}, {k_i}): energies relative to '
                  f'relax_i will be flagged as not comparable (sandbox tests only)')
    failed = []
    for d in (args.d or PROFILE_D):
        if not 0.95 <= d <= 2.3:
            raise SystemExit(f'profile: d(N1-H1A) = {d} A is outside 0.95-2.3 A')
        name = profile_name(d, args.seed)
        tag = f'relax_{name}'
        if args.if_falling and not (out / 'runs' / tag / f'{tag}.out').exists():
            why = profile_falls_before(args, cr, d)
            if why is not True:
                note(out, f'{tag}: not run (--if-falling: {why})')
                continue
        F, seed = profile_seed(args, cr, d)
        cons = cr.constraints_nh(d)
        sg = seed['start_geometry']
        msg = (f'{tag}: seed {args.seed}' + (f" w={seed['w']}" if 'w' in seed else '') + ': start ' +
               '; '.join(f"f.u.{c + 1} N1-H1A {g['d_N1_H1A_A']} H1A-O5 {g['d_H1A_O5_A']} N1...O5 {g['d_N1_O5_A']} A"
                         for c, g in enumerate(sg)) + f"; min distance {seed['min_interatomic_A']} A")
        if seed['min_interatomic_A'] < 0.75:
            raise SystemExit(f'profile: {msg}: atoms overlap in the start geometry')
        if args.dry_run:
            (out / 'inputs').mkdir(parents=True, exist_ok=True)
            (out / 'inputs' / f'{tag}.in').write_text(pw_input(
                cr, F, tag, 'relax', ecut, k, '/SCRATCH', args.pseudo, conv_thr=PROFILE_CONV_THR, tstress=True,
                forc_thr=args.forc_thr, etot_thr=args.etot_thr, ion_dynamics=args.dyn, constraints=cons,
                extra_ctrl=[f'dt = {args.dt}'] if args.dt else []))
            atomic_json(out / 'inputs' / f'{tag}.seed.json', dict(seed, target_A=d, constraints_bohr=cons))
            note(out, f'dry run: inputs/{tag}.in  ({msg})')
            continue
        rd = out / 'runs' / tag
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / f'{tag}.seed.json').exists():       # the seed of the first attempt; restarts reuse positions
            atomic_json(rd / f'{tag}.seed.json', dict(seed, target_A=d, utc=utc()))
        if (rd / f'{tag}.out').exists():
            msg += ' [an existing run is reused if finished, else continued from its last geometry]'
        note(out, msg)
        extra_ctrl = ([f'dt = {args.dt}'] if args.dt else []) + \
                     ([f'max_seconds = {args.max_seconds}'] if args.max_seconds else [])
        try:
            p = run_pw(args, cr, tag, 'relax', F, ecut, k, conv_thr=PROFILE_CONV_THR, tstress=True,
                       forc_thr=args.forc_thr, etot_thr=args.etot_thr, ion_dynamics=args.dyn, constraints=cons,
                       extra_ctrl=extra_ctrl, log_extra={'ion_dynamics': args.dyn, 'target_A': d, 'seed': args.seed})
        except RuntimeError as e:
            print(f'[periodic_qe] {e}', flush=True)
            failed.append(tag)
            continue
        Fr = np.array(p['positions_crystal'])
        got = [proton_fu(cr, Fr, c) for c in range(len(cr.ops))]
        write_extxyz(cr, Fr, out / f'{tag}.extxyz', f'PBE-D3(BJ) fixed-cell {args.dyn} relax, d(N1-H1A) = {d} A '
                     f'(both f.u.), ecut {ecut[0]}/{ecut[1]} Ry, k {k}, E = {p["final_energy_Ry"]} Ry')
        note(out, f'{tag}: ' + '; '.join(f"f.u.{c + 1} N1-H1A {g['d_N1_H1A_A']} H1A-O5 {g['d_H1A_O5_A']} "
                                         f"N1...O5 {g['d_N1_O5_A']} A" for c, g in enumerate(got))
             + f", {p['md_steps']} steps")
        if max(abs(g['d_N1_H1A_A'] - d) for g in got) > 2e-3:
            print(f'[periodic_qe] {tag}: the constraint did not hold ({[g["d_N1_H1A_A"] for g in got]} vs {d})', flush=True)
            failed.append(tag)
    if failed:
        raise SystemExit(f'profile: not converged / failed: {failed} (rerun the same command to continue)')


def constraint_slopes(cr, F, forces):
    """dE/dd (Ry/bohr) of each N1-H1A constraint from the raw forces at a constrained minimum: lambda =
    (F_N - F_H).u / 2 with u the unit vector N1->H1A (F_H = -lambda u, F_N = +lambda u at a constrained stationary
    point); also the part of the H1A force perpendicular to u (should be ~0) as a check."""
    Fo = np.array(forces)
    lam, perp = [], []
    for sN, sH, _ in cr.nh_sites():
        pN, pH = cr.point(F, sN), cr.point(F, sH)
        u = (pH - pN) / np.linalg.norm(pH - pN)
        fN, fH = Fo[sN[0]], Fo[sH[0]]
        lam.append(float((fN - fH) @ u / 2))
        perp.append(float(np.linalg.norm(fH - (fH @ u) * u)))
    return lam, perp


def profile_point(cr, f, ref):
    """One relax_prof* (or relax_iir1) output -> record."""
    p = parse_pw(f, cr.nat)
    if not p['positions_crystal']:
        return None
    F = np.array(p['positions_crystal'])
    inp = f.with_suffix('.in')
    ecut, k = run_setting(inp)
    txt = inp.read_text() if inp.exists() else ''
    targets = [float(t) * BOHR_A for t in re.findall(r"^\s*'distance'\s+\d+\s+\d+\s+([-\d.]+)", txt, re.M)]
    dyn = input_value(inp, 'ion_dynamics') if inp.exists() else None
    E = p['final_energy_Ry'] if p['relax_converged'] else p['energy_Ry']
    conv = p['relax_converged'] and p['job_done']
    r = {'tag': f.parent.name, 'converged': conv, 'ion_dynamics': dyn, 'ecut_Ry': ecut, 'k': k,
         'targets_A': [round(t, 5) for t in targets], 'energy_Ry': E, 'n_symops': p['n_symops'],
         'steps': p['md_steps'] if p['md_converged'] else (p['bfgs_steps'] or p['n_ion_steps']),
         'n_scf': len(p['scf_iterations']), 'restarts': len(list(f.parent.glob(f'{f.name}.part*'))),
         'wall_s_last_part': p['wall_s'], 'pressure_kbar_final': p['pressure_kbar'],
         'max_force_Ry_bohr': round(float(np.abs(np.array(p['forces_Ry_bohr'])).max()), 6) if p['forces_Ry_bohr'] else None,
         'max_constrained_force_Ry_bohr': (round(float(np.abs(np.array(p['constrained_forces_Ry_bohr'])).max()), 6)
                                           if p['constrained_forces_Ry_bohr'] else None),
         'formula_units': [proton_fu(cr, F, c) for c in range(len(cr.ops))],
         'sym_dev_A': geometry_report(cr, F)['sym_dev_A'], '_F': F}
    if not conv:
        r['status'] = ('PROVISIONAL: not converged' + (' (stopped by max_seconds)' if p['stopped_by_time'] else '')
                       + '; energy of the last SCF, geometry one step later')
    if ref is not None:
        dE = E - ref['energy_Ry'] if E is not None else None
        same = [ecut, k] == [ref['ecut_Ry'], ref['k']]
        r['same_setting_as_relax_i'] = same
        if dE is not None:
            r.update({'E_minus_i_meV_cell': _mev(dE), 'E_minus_i_kJ_mol_per_formula_unit': _kj_fu(dE)})
        r['rmsd_all_atoms_vs_relax_i_A'] = rmsd_vs(cr, F, cr.unwrap(ref['_F']), heavy_only=False)['rmsd_A']
    if targets and p['forces_Ry_bohr'] and conv:
        lam, perp = constraint_slopes(cr, F, p['forces_Ry_bohr'])
        k_ev = RY_EV / BOHR_A
        r['slope'] = {'dE_dd_per_fu_eV_A': [round(x * k_ev, 4) for x in lam],
                      'dE_dd_cell_meV_A': round(sum(lam) * k_ev * 1000, 1),
                      'dE_dd_kJ_mol_A_per_formula_unit': round(sum(lam) * k_ev * KJMOL_PER_EV / len(lam), 2),
                      'H1A_force_perpendicular_to_NH_Ry_bohr': [round(x, 5) for x in perp],
                      'how': 'constraint force at the constrained minimum: dE/dd = (F_N1 - F_H1A).u / 2 per formula '
                             'unit (u = N1->H1A); per cell = sum over both formula units (both distances move)'}
    return r


def _neutral_verdict(rows, level, have_slopes):
    """Does the profile (list of (d, E meV/cell, slope meV/A or None), relax_i first) have a neutral minimum?"""
    if len(rows) < 3:
        return {'answer': 'undetermined', 'text': f'{level}: fewer than two profile points'}
    ds = [r[0] for r in rows]; es = [r[1] for r in rows]
    mins = [i for i in range(1, len(es) - 1) if es[i] < es[i - 1] and es[i] < es[i + 1]]
    last_slope = rows[-1][2] if have_slopes else None
    falls_at_end = es[-1] < es[-2] or (last_slope is not None and last_slope < 0)
    if mins:
        i = mins[0]
        top = max(es[:i])
        return {'answer': 'yes', 'd_min_A': ds[i], 'E_min_meV_cell': es[i], 'barrier_back_meV_cell': round(top - es[i], 1),
                'text': f'{level}: local minimum on the neutral side near d(N1-H1A) = {ds[i]} A, {es[i]:+.1f} meV/cell '
                        f'above the salt, {top - es[i]:.1f} meV/cell below the highest point before it (point grid 0.15 A)'}
    if falls_at_end:
        return {'answer': 'possibly beyond the scan', 'text': f'{level}: the energy falls at the end of the scan '
                f'(d = {ds[-1]} A' + (f', slope {last_slope:+.0f} meV/A per cell' if last_slope is not None else '')
                + '): a neutral minimum may lie at larger d; extend the profile (e.g. 1.85 A)'}
    rising = all(b > a for a, b in zip(es, es[1:]))
    if rising and (not have_slopes or all(r[2] is None or r[2] > 0 for r in rows[1:])):
        return {'answer': 'no', 'text': f'{level}: no neutral minimum between d(N1-H1A) = {ds[0]} and {ds[-1]} A; the '
                f'energy rises monotonically to {es[-1]:+.1f} meV/cell at {ds[-1]} A'
                + (' and the constraint force pushes the proton back to N1 at every point' if have_slopes else '')}
    return {'answer': 'unclear', 'text': f'{level}: not monotonic but no clear minimum; inspect the points'}


def stage_profile_analyse(args, cr):
    """profile.json: the PBE-D3(BJ) and PBE0-D3(BJ) profiles, slopes, hysteresis, the iir1 control.  Reads runs/ in
    --out (and relax_i / relax_iirfix / hybrid_i from --geom-from if given); never rewrites periodic_qe.json."""
    out = Path(args.out)
    dirs = geom_dirs(args)
    Fi, Ei, ecut_i, k_i, fi, _ = relaxed_geometry(cr, dirs, 'i')
    ref = {'energy_Ry': Ei, 'ecut_Ry': ecut_i, 'k': k_i, '_F': Fi}
    gi = [proton_fu(cr, Fi, c) for c in range(len(cr.ops))]
    res = {'generated_utc': utc(), 'method': {
        'what': 'constrained N1-H1A proton-transfer profile in the crystal (referee M2/M3): d(N1-H1A) of both formula '
                'units held by pw.x CONSTRAINTS (distance), every other coordinate relaxed (O5, H1A across the bond '
                'included), fixed X-ray cell, PBE-D3(BJ)/SSSP; PBE0-D3(BJ) single points at each point (SG15 hybrid '
                'correction, as for relax_i vs relax_iirfix)',
        'optimiser': "pw.x damped dynamics (ion_dynamics 'damp', BFGS-preconditioned quick-min) or FIRE, SHAKE "
                     "constraints; QE 7.5 refuses CONSTRAINTS with BFGS (PW/src/input.f90: 'constraints only with "
                     "fixed-cell dynamics'); per point: ion_dynamics",
        'thresholds': f'forc_conv_thr {PROFILE_FORC_THR} Ry/bohr on the constrained forces, etot_conv_thr '
                      f'{PROFILE_ETOT_THR} Ry, conv_thr {PROFILE_CONV_THR} Ry (defaults; see the run inputs)',
        'reference': f'relax_i (the salt, BFGS, {ecut_i} Ry, k {k_i}): E = {Ei} Ry, d(N1-H1A) {gi[0]["d_N1_H1A_A"]} A',
        'units': 'meV per cell (54 atoms, Z = 2: both protons move) and kJ/mol per formula unit (= per cell / 2); '
                 'slopes dE/dd in meV/A per cell and kJ/mol/A per formula unit'},
        'relax_i': {'energy_Ry': Ei, 'formula_units': gi, 'file': str(fi)}}
    try:
        Ff, Ef, _, _, _, _ = relaxed_geometry(cr, dirs, 'iirfix')
        res['context_iirfix'] = {'what': 'the earlier single constrained point (H1A and O5 fixed in space, O5-H1A 0.98 '
                                         'A); the profile replaces it', 'E_minus_i_meV_cell': _mev(Ef - Ei),
                                 'E_minus_i_kJ_mol_per_formula_unit': _kj_fu(Ef - Ei),
                                 'formula_unit_1': proton_fu(cr, Ff, 0)}
    except SystemExit:
        pass
    pts = {}
    for d in sorted((out / 'runs').glob('relax_prof*')):
        name = d.name[len('relax_'):]
        f = d / f'{d.name}.out'
        if not _PROF.match(name) or not f.exists():
            continue
        r = profile_point(cr, f, ref)
        if r:
            sj = d / f'{d.name}.seed.json'
            if sj.exists():
                s = json.loads(sj.read_text())
                r['seed'] = {kk: s.get(kk) for kk in ('seed', 'w', 'how', 'from')}
            pts[name] = r
    # PBE0-D3(BJ): SG15 hybrid correction at each point relative to relax_i, per variant (tag suffix)
    rl = {'i': {'_F': Fi, 'energy_Ry': Ei, 'dispersion_Ry': parse_pw(fi, cr.nat)['dispersion_Ry'], 'ecut_Ry': ecut_i,
                'k': k_i}}
    for name, r in pts.items():
        if r['converged']:
            rl[name] = {'_F': r['_F'], 'energy_Ry': r['energy_Ry'], 'dispersion_Ry': None, 'ecut_Ry': r['ecut_Ry'],
                        'k': r['k']}
    hyb = {}
    hdirs = [Path(x) / 'runs' for x in [args.out] + [g for g in dirs if g != str(Path(args.out))]]
    for name in pts:
        for hd in sorted((out / 'runs').glob(f'hybrid_{name}_*')):
            m = _HTAG.match(hd.name)
            if not m or m.group(1) != name or not (hd / f'{hd.name}.out').exists():
                continue
            var = m.group(2)
            ri = next((h / f'hybrid_i_{var}' for h in hdirs if (h / f'hybrid_i_{var}' / f'hybrid_i_{var}.out').exists()),
                      None)
            if ri is None:
                hyb.setdefault(var, {})[name] = {'status': f'no hybrid_i_{var} reference run'}
                continue
            a, b = hybrid_point(cr, ri, rl), hybrid_point(cr, hd, rl)
            c = compare_hybrid(a, b)
            c['tag'] = hd.name
            c['pbe0_wall_s'] = b['wall_s']
            hyb.setdefault(var, {})[name] = c
    res['hybrid_variants'] = sorted(hyb)
    primary = 'sg15_e100' if 'sg15_e100' in hyb else (sorted(hyb)[0] if hyb else None)
    # the profile: default-seed points (prof<ddd>), else whatever seed exists for that d
    byd = {}
    for name, r in pts.items():
        dd = int(_PROF.match(name).group(1))
        if dd not in byd or name == f'prof{dd:03d}':
            byd[dd] = name
    rows, rows0 = [(gi[0]['d_N1_H1A_A'], 0.0, 0.0)], [(gi[0]['d_N1_H1A_A'], 0.0, None)]
    table = []
    for dd in sorted(byd):
        name = byd[dd]; r = pts[name]
        g = r['formula_units'][0]
        row = {'point': name, 'd_N1_H1A_target_A': dd / 100, 'converged': r['converged'],
               'E_PBE_D3BJ_meV_cell': r.get('E_minus_i_meV_cell'),
               'E_PBE_D3BJ_kJ_mol_per_fu': r.get('E_minus_i_kJ_mol_per_formula_unit'),
               'slope_meV_A_cell': (r.get('slope') or {}).get('dE_dd_cell_meV_A'),
               'delta_A': g['delta_A'], 'd_N1_O5_A': g['d_N1_O5_A'], 'd_H1A_O5_A': g['d_H1A_O5_A'],
               'angle_N1_H1A_O5_deg': g['angle_N1_H1A_O5_deg'], 'd_P1_O5_A': g['d_P1_O5_A'],
               'd_O4_O5_A': g['O4_H4A_O5']['d_O4_O5_A'], 'pressure_kbar': r['pressure_kbar_final'],
               'same_setting_as_relax_i': r.get('same_setting_as_relax_i')}
        if not r.get('same_setting_as_relax_i'):
            # a sandbox point at another cutoff / k grid: its energy relative to relax_i means nothing
            row.update({'E_PBE_D3BJ_meV_cell': None, 'E_PBE_D3BJ_kJ_mol_per_fu': None,
                        'note': 'setting differs from relax_i: no energy relative to the salt'})
        h = (hyb.get(primary) or {}).get(name) or {}
        if 'best_estimate_PBE0_D3BJ_meV_cell' in h:
            row.update({'E_PBE0_D3BJ_meV_cell': h['best_estimate_PBE0_D3BJ_meV_cell'],
                        'E_PBE0_D3BJ_kJ_mol_per_fu': h['best_estimate_PBE0_D3BJ_kJ_mol_per_formula_unit'],
                        'hybrid_correction_meV_cell': h.get('hybrid_correction_meV_cell'),
                        'PBE0_converged': h.get('converged')})
        table.append(row)
        if r['converged'] and r.get('same_setting_as_relax_i') and row['E_PBE_D3BJ_meV_cell'] is not None:
            rows.append((dd / 100, row['E_PBE_D3BJ_meV_cell'], row['slope_meV_A_cell']))
            if row.get('E_PBE0_D3BJ_meV_cell') is not None and row.get('PBE0_converged'):
                rows0.append((dd / 100, row['E_PBE0_D3BJ_meV_cell'], None))
    res['profile'] = table
    res['relax_i_row'] = {'d_N1_H1A_A': gi[0]['d_N1_H1A_A'], 'E_meV_cell': 0.0, 'delta_A': gi[0]['delta_A'],
                          'd_N1_O5_A': gi[0]['d_N1_O5_A'], 'd_H1A_O5_A': gi[0]['d_H1A_O5_A']}
    n_exp = len(PROFILE_D)
    res['neutral_minimum'] = {
        'PBE_D3BJ': _neutral_verdict(rows, 'PBE-D3(BJ)', True),
        'PBE0_D3BJ': _neutral_verdict(rows0, 'PBE0-D3(BJ) single points at the PBE-D3(BJ) geometries', False),
        'complete': {'PBE_points': len(rows) - 1, 'PBE0_points': len(rows0) - 1, 'expected': n_exp}}
    # hysteresis: the same d from different starts
    hyst = {}
    for name, r in pts.items():
        dd = int(_PROF.match(name).group(1))
        base = f'prof{dd:03d}'
        if name != base and base in pts and r['converged'] and pts[base]['converged']:
            a = pts[base]
            hyst[name] = {'vs': base, 'dE_meV_cell': _mev(r['energy_Ry'] - a['energy_Ry']),
                          'rmsd_all_atoms_A': rmsd_vs(cr, r['_F'], cr.unwrap(a['_F']), heavy_only=False)['rmsd_A'],
                          'same_minimum': abs(r['energy_Ry'] - a['energy_Ry']) * RY_EV * 1000 < 5}
    if hyst:
        res['hysteresis'] = hyst
    # single-formula-unit control
    fc = next((Path(g) / 'runs' / 'relax_iir1' / 'relax_iir1.out' for g in [args.out] + dirs
               if (Path(g) / 'runs' / 'relax_iir1' / 'relax_iir1.out').exists()), None)
    if fc:
        c = profile_point(cr, fc, ref)
        if c:
            on = [g['proton_on'] for g in c['formula_units']]
            c['what'] = ('relax_i with H1A moved 0.98 A from O5 in formula unit 1 only, unconstrained BFGS (no symmetry): '
                         'does the proton return?')
            c['E_minus_i_note'] = 'one proton moved: the per-cell value is the cost of one transferred proton'
            if not c['converged']:
                c['verdict'] = 'provisional: the relaxation has not converged'
            elif on == ['N'] * len(on) and c['rmsd_all_atoms_vs_relax_i_A'] < 0.02 and abs(c['E_minus_i_meV_cell']) < 5:
                c['verdict'] = 'the proton returned to N1: the salt is recovered (single minimum)'
            elif on[0] == 'O':
                c['verdict'] = (f"the proton stayed on O5 in formula unit 1: a mixed salt/neutral local minimum "
                                f"{c['E_minus_i_meV_cell']:+.1f} meV/cell above the salt")
            else:
                c['verdict'] = (f"proton back on N1 but a different structure (RMSD {c['rmsd_all_atoms_vs_relax_i_A']} "
                                f"A, {c['E_minus_i_meV_cell']:+.1f} meV/cell): inspect")
            c.pop('_F')
            res['single_transfer_control'] = c
    for r in pts.values():
        r.pop('_F', None)
    res['points'] = pts
    res['hybrid'] = hyb
    atomic_json(out / 'profile.json', res)
    lines = [f"{'point':16s} {'d_NH':>5s} {'E_PBE':>8s} {'kJ/fu':>7s} {'E_PBE0':>8s} {'kJ/fu':>7s} {'slope':>7s} "
             f"{'delta':>7s} {'N..O5':>6s} {'O5-H':>6s}",
             f"{'relax_i':16s} {gi[0]['d_N1_H1A_A']:5.3f} {0:8.1f} {0:7.2f} {0:8.1f} {0:7.2f} {0:7.0f} "
             f"{gi[0]['delta_A']:7.3f} {gi[0]['d_N1_O5_A']:6.3f} {gi[0]['d_H1A_O5_A']:6.3f}"]
    fmt = lambda v, w, p: f'{v:{w}.{p}f}' if isinstance(v, (int, float)) else f'{"-":>{w}s}'
    for r in table:
        lines.append(f"{r['point']:16s} {r['d_N1_H1A_target_A']:5.2f} {fmt(r['E_PBE_D3BJ_meV_cell'], 8, 1)} "
                     f"{fmt(r['E_PBE_D3BJ_kJ_mol_per_fu'], 7, 2)} {fmt(r.get('E_PBE0_D3BJ_meV_cell'), 8, 1)} "
                     f"{fmt(r.get('E_PBE0_D3BJ_kJ_mol_per_fu'), 7, 2)} {fmt(r['slope_meV_A_cell'], 7, 0)} "
                     f"{r['delta_A']:7.3f} {r['d_N1_O5_A']:6.3f} {r['d_H1A_O5_A']:6.3f}"
                     + ('' if r['converged'] else '  NOT CONVERGED'))
    print('\n'.join(lines), flush=True)
    v = res['neutral_minimum']
    note(out, f"profile_analyse: wrote profile.json ({len(table)} points; {v['PBE_D3BJ']['text']}; "
              f"{v['PBE0_D3BJ']['text']}"
         + (f"; control iir1: {res['single_transfer_control']['verdict']}" if 'single_transfer_control' in res else '')
         + ')')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['build', 'converge', 'relax', 'analyse', 'estimate', 'hybrid_probe', 'hybrid', 'profile',
                                      'profile_analyse', 'all'])
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--scratch', default=str(SCRATCH))
    ap.add_argument('--pseudo', default=str(QE / 'pseudo'))
    ap.add_argument('--pw', default=str(QE / 'bin/pw.x'))
    ap.add_argument('--mpirun', default=str(QE / 'bin/mpirun'))
    ap.add_argument('--np', type=int, default=int(os.environ.get('QE_NP', '4')), help='MPI ranks')
    ap.add_argument('--nk', type=int, default=int(os.environ.get('QE_NK', '0')), help='k-point pools (0 = auto)')
    ap.add_argument('--omp', type=int, default=int(os.environ.get('QE_OMP', '1')), help='OpenMP threads per rank')
    def start_arg(v):
        if v in ('i', 'ii', 'iir', 'iifix', 'iirfix', 'iir1', 'both') or _PROF.match(v):
            return v
        raise argparse.ArgumentTypeError(f'unknown start {v!r}')
    ap.add_argument('--start', type=start_arg, default='both',
                    help="relax: 'iir' = H1A moved to O5 in the relaxed salt (needs relax_i_pre); 'iifix' = start ii "
                         "with H1A fixed in space (slid back to the salt); 'iirfix' = H1A moved to O5 in relax_i with H1A "
                         "and O5 fixed (constrained neutral form); 'iir1' = H1A moved to O5 in relax_i in formula unit 1 "
                         "only, unconstrained (single-transfer control); 'both' = i and ii.  hybrid: the relax "
                         "whose final geometry is used (also prof<ddd>[_<seed>], a converged profile point); 'both' = i "
                         "then iirfix")
    ap.add_argument('--ecut', help='override, e.g. 60/480 (hybrid --pp sg15: default 100/400)')
    ap.add_argument('--kgrid', help='override, e.g. 3x2x2')
    ap.add_argument('--ecuts', nargs='+', help='converge: cutoff series, e.g. 50/400 60/480 70/560')
    ap.add_argument('--kgrids', nargs='+', help='converge: k series, e.g. 2x1x1 3x2x2')
    ap.add_argument('--k-for-ecut', help='converge: k grid of the cutoff series (default 3x2x2)')
    ap.add_argument('--max-runs', type=int, default=0, help='converge: stop after this many runs (sandbox tests)')
    ap.add_argument('--dry-run', action='store_true',
                    help='build: also write example pw.x inputs; profile: write inputs/relax_prof*.in, run nothing')
    ap.add_argument('--probe-seconds', type=int, default=1200, help='hybrid_probe: pw.x max_seconds')
    ap.add_argument('--pre-ecut', default='50/400', help='relax: cutoff of the coarse pre-relaxation')
    ap.add_argument('--pre-kgrid', default='2x1x1', help='relax: k grid of the coarse pre-relaxation')
    ap.add_argument('--no-pre', action='store_true', help='relax: no coarse pre-relaxation')
    ap.add_argument('--pre-only', action='store_true', help='relax: only the coarse pre-relaxation')
    ap.add_argument('--nq', default='1x1x1', help='hybrid: q mesh of the Fock operator (nqx1-3), e.g. 2x1x1')
    ap.add_argument('--ecutfock', type=float, default=0,
                    help='hybrid: Fock-operator cutoff in Ry (0 = the QE 7.5 default MIN(ecutrho, 4*ecutwfc), written '
                         'explicitly; = ecutrho for SG15; QE forces it for US/PAW)')
    ap.add_argument('--conv-thr', type=float, default=HYBRID_CONV_THR,
                    help='hybrid: conv_thr in Ry, also the EXX outer-loop threshold on dexx')
    ap.add_argument('--max-seconds', type=int, default=0,
                    help='hybrid, profile: pw.x max_seconds (0 = none); a run stopped by it fails; hybrid: kept as '
                         'partial and re-run from scratch; profile: the next call continues from the last geometry')
    ap.add_argument('--d', type=float, nargs='+', help=f'profile: d(N1-H1A) targets in A (default {PROFILE_D})')
    ap.add_argument('--seed', default='interp',
                    help="profile: start geometry: interp (relax_i -> relax_iirfix interpolation, default), i (relax_i, "
                         "proton moved on the N1->O5 line) or prof<ddd> (another converged profile point)")
    ap.add_argument('--dyn', choices=['damp', 'fire'], default='damp',
                    help="profile: pw.x ion_dynamics with CONSTRAINTS (BFGS is refused by QE 7.5)")
    ap.add_argument('--forc-thr', type=float, default=PROFILE_FORC_THR, help='profile: forc_conv_thr (Ry/bohr)')
    ap.add_argument('--etot-thr', type=float, default=PROFILE_ETOT_THR, help='profile: etot_conv_thr (Ry)')
    ap.add_argument('--dt', type=float, default=0, help='profile: pw.x dt (Rydberg a.u.; 0 = the QE default 20)')
    ap.add_argument('--if-falling', action='store_true',
                    help='profile: run a point only if the profile still falls at the largest converged smaller d '
                         '(constraint-force slope < 0 or energy decreasing): the conditional 1.85 A extension')
    ap.add_argument('--exx-maxstep', type=int, default=0, help='hybrid: pw.x exx_maxstep (0 = QE default 100)')
    ap.add_argument('--geom-from', help='hybrid: results directory holding runs/relax_* (default: --out)')
    ap.add_argument('--pp', choices=['sg15', 'sssp'], default='sg15',
                    help='hybrid: sg15 = SG15 norm-conserving potentials at --ecut (default 100/400) + a PBE-D3 companion '
                         'SCF (hybrid correction); sssp = the relax potentials (US/PAW, G-space: very slow, see docs)')
    ap.add_argument('--pseudo-sg15', default=str(QE / 'pseudo_sg15'), help='hybrid: directory of the SG15 files')
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # -> SystemExit, so run_pw stops its pw.x ranks
    cr = Crystal()
    stages = ['build', 'converge', 'relax', 'analyse'] if args.stage == 'all' else [args.stage]
    for st in stages:
        globals()[f'stage_{st}'](args, cr)


if __name__ == '__main__':
    main()
