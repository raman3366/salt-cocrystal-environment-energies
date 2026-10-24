"""Fig. 1: the set (13 acid-base pairs + brcoc) by dpKa, the model ladder and the cycle R5 = R1f + dE_env (DOUBLE).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/fig1_set.py

(a) 2D structures of the two partners of every crystal, ordered by dpKa (summary.json crystals[].dpka; brcoc next to
    its polymorph brsalt), tile tint = observed class (blue salt, orange cocrystal), border and tag = set (core solid,
    optional dashed, extension dotted; brcoc grey).  Chemistry: the species formulas of optc/crystals.json alone do not
    fix connectivity, so each partner is cut from the crystal (optc/structures/<key>/start.extxyz, molecules made whole
    across the cell by minimum image, covalent radii x 1.15), matched to its Hill formula in crystals.json and given
    bond orders by RDKit rdDetermineBonds with the crystals.json charge; formula, net charge and sanitisation are
    asserted, the SMILES are written to the facts file.  Perchlorate is redrawn from RDKit's [Cl+3]([O-])4 form to
    O=Cl(=O)(=O)[O-] (same species, conventional drawing).  2D layout: RDKit CoordGen, principal axis horizontal,
    turned by up to 60 deg where an NH2 / NH3+ bond would be steep (turn); the tetrahedral oxo-anions as a square cross;
    nitro groups drawn as NO2 (drawn); bonds 2.6 mm, 2.3 mm where a pair does not fit its tile (dapsone).
    The proton is drawn where the crystal has it (salts: cation and anion; cocrystals: neutral partners).
(b) the model ladder R1f -> R4a -> R3 -> R4 -> R5 (definitions as optc/G4_RESULTS.md section 2); name and
    description centred in each box as one block, measured from the rendered text.
(c) the cycle: R5 = R1f + dE_env, all terms E(ion pair) - E(neutral pair).
nitrate (optional, no results) is not shown.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdAbbreviations, rdDepictor, rdDetermineBonds  # noqa: E402

import _ctc as S  # noqa: E402
from figkit import style, rdkit2mpl  # noqa: E402

STEM = 'fig1_set'
W = 175.0
FS = style.SMALL_PT
CR = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'Cl': 1.02, 'S': 1.05, 'Br': 1.20, 'P': 1.07}
TINT = {'salt': '#E3ECF8', 'cocrystal': '#FCEBD8'}
BORDER = {'core': '-', 'optional': (0, (3, 1.6)), 'extension': (0, (1, 1.2))}
SET_NAME = {'core': 'core', 'optional': 'optional', 'extension': 'extension'}


# ------------------------------------------------------------------------------------------------ chemistry
def hill(sym):
    c = Counter(sym)
    return ''.join(e + (str(c[e]) if c[e] > 1 else '') for e in ['C', 'H'] + sorted(k for k in c if k not in 'CH')
                   if e in c)


def fragments(key):
    """Whole molecules of the cell keyed by Hill formula (first copy), as XYZ blocks in sorted atom order."""
    L = (S.STRUCT / key / 'start.extxyz').read_text().splitlines()
    n = int(L[0])
    lat = np.array([float(x) for x in L[1].split('Lattice="')[1].split('"')[0].split()]).reshape(3, 3)
    sym = [ln.split()[0] for ln in L[2:2 + n]]
    X = np.array([[float(v) for v in ln.split()[1:4]] for ln in L[2:2 + n]])
    inv, imgs = np.linalg.inv(lat), np.array(list(np.ndindex(3, 3, 3))) - 1

    def mic(dv):
        f = dv @ inv
        f -= np.round(f)
        return min(((f + s) @ lat for s in imgs), key=np.linalg.norm)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if 'H' == sym[i] == sym[j]:
                continue
            if np.linalg.norm(mic(X[j] - X[i])) < 1.15 * (CR[sym[i]] + CR[sym[j]]):
                adj[i].append(j)
                adj[j].append(i)
    seen, out = set(), {}
    for i in range(n):
        if i in seen:
            continue
        pos, todo = {i: X[i]}, [i]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in pos:
                    pos[b] = pos[a] + mic(X[b] - pos[a])
                    todo.append(b)
        seen |= set(pos)
        idx = sorted(pos)
        out.setdefault(hill([sym[a] for a in idx]), f'{len(idx)}\n\n' + '\n'.join(
            f'{sym[a]} {pos[a][0]:.5f} {pos[a][1]:.5f} {pos[a][2]:.5f}' for a in idx))
    return out


def species_mol(xyz, formula, charge):
    m = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(m, charge=charge)
    assert hill([a.GetSymbol() for a in m.GetAtoms()]) == formula
    assert Chem.GetFormalCharge(m) == charge
    m = Chem.RemoveHs(m)
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():                     # perchlorate: [Cl+3]([O-])4 -> O=Cl(=O)(=O)[O-]
        if a.GetSymbol() == 'Cl' and a.GetFormalCharge() == 3:
            i = a.GetIdx()                      # by index: the bond objects of GetBonds() are copies here
            for j in [n.GetIdx() for n in a.GetNeighbors()][:3]:
                rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)
                rw.GetAtomWithIdx(j).SetFormalCharge(0)
            rw.GetAtomWithIdx(i).SetFormalCharge(0)
    m = rw.GetMol()
    # the hypervalent halogen (perchlorate): no clean-up (it restores [Cl+3]([O-])4) and no valence check
    hyper = any(a.GetSymbol() == 'Cl' and a.GetDegree() == 4 for a in m.GetAtoms())
    F_ = Chem.SanitizeFlags
    Chem.SanitizeMol(m, F_.SANITIZE_ALL ^ F_.SANITIZE_CLEANUP ^ F_.SANITIZE_PROPERTIES if hyper else F_.SANITIZE_ALL)
    if hyper:
        m.UpdatePropertyCache(strict=False)
    assert Chem.GetFormalCharge(m) == charge
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(m)
    conf = m.GetConformer()
    P = conf.GetPositions()[:, :2]
    P -= P.mean(0)
    centre = [a for a in m.GetAtoms() if a.GetDegree() == m.GetNumAtoms() - 1 and a.GetSymbol() in ('P', 'Cl', 'S')]
    if m.GetNumAtoms() <= 5 and centre:
        # a tetrahedral oxo-anion: a square cross (PCA has no preferred axis here and tilts it); anionic O on the
        # right, a single NH2 / OH on the left (sulfamate), else a neutral O; the rest above and below
        c = centre[0]
        bare = sorted((n for n in c.GetNeighbors() if not n.GetTotalNumHs()), key=lambda n: -n.GetFormalCharge())
        hyd = [n for n in c.GetNeighbors() if n.GetTotalNumHs()]
        left = hyd[0] if len(hyd) == 1 else bare[0]
        right = bare[-1]
        rest = [n for n in bare + hyd if n.GetIdx() not in (left.GetIdx(), right.GetIdx())]
        P = np.zeros((m.GetNumAtoms(), 2))
        for n, xy in zip([left, right] + rest, [(-1.5, 0.0), (1.5, 0.0), (0.0, 1.5), (0.0, -1.5)]):
            P[n.GetIdx()] = xy
    else:
        _, v = np.linalg.eigh(P.T @ P)
        P = P @ v[:, ::-1]
        P = turn(m, P)
    for i, p in enumerate(P):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), 0.0))
    return m


def turn(m, P):
    """Smallest turn of the PCA layout (0, +-30, +-60 deg) that leaves every terminal NH2 / NH3+ bond within 40 deg of
    horizontal: steeper, RDKit stacks the hydrogens over the N (as 'H3+' above 'N'), hard to read at this size."""
    term = [(a.GetIdx(), a.GetNeighbors()[0].GetIdx()) for a in m.GetAtoms()
            if a.GetSymbol() == 'N' and a.GetDegree() == 1 and a.GetTotalNumHs() >= 2]
    for deg in (0, 30, -30, 60, -60):
        t = np.radians(deg)
        Q = P @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        if not any(40 < abs(np.degrees(np.arctan2(*(Q[i] - Q[j])[::-1]))) % 180 < 140 for i, j in term):
            return Q
    return P


def drawn(m):
    """The molecule as drawn: nitro groups condensed to NO2 (RDKit abbreviation), whose charge-separated form puts the
    '+' of N on the N=O bond at this size; the species itself (formula, charge, SMILES) is the full m."""
    no2 = [a for a in rdAbbreviations.GetDefaultAbbreviations() if a.label == 'NO2']
    return rdAbbreviations.CondenseMolAbbreviations(m, no2, maxCoverage=0.9)


def pairs():
    keys = S.by_dpka(S.KEYS + ['brcoc'])
    keys.remove('brcoc')
    keys.insert(keys.index('brsalt') + 1, 'brcoc')
    out = []
    for k in keys:
        meta = S.META[k]
        assert meta['set'] == S.CRY[k]['set'] and meta['observed'] == S.CRY[k]['observed'], k
        fr = fragments(k)
        mols = {}
        for f, info in meta['species'].items():
            mols[info['role']] = (f, species_mol(fr[f], f, info['charge']))
        out.append((k, mols))
    return out


# ------------------------------------------------------------------------------------------------ drawing
def box(ax, x, y, w, h, fc, ec, ls='-', lw=0.6, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def arrow(ax, p0, p1, col='0.2', lw=0.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6, lw=lw, color=col, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main():
    S.setup()
    P = pairs()
    ncol, th, g = 5, 29.0, 1.5
    tw = (W - 1.2 - (ncol - 1) * g) / ncol             # five tiles and 0.6 mm margins fill the width
    nrow = -(-(len(P) + 1) // ncol)
    hb = 37.2                                       # schematic strip
    top = 6.0
    H = top + nrow * (th + g) + hb
    fig = S.figure_mm(W, H)
    ax = S.axes_mm(fig, 0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    smiles = {}
    for n, (k, mols) in enumerate(P):
        r, c = divmod(n, ncol)
        x0 = 0.6 + c * (tw + g)
        y0 = H - top - (r + 1) * (th + g) + g
        cls, st = S.META[k]['observed'], S.META[k]['set']
        is_poly = k == 'brcoc'
        box(ax, x0, y0, tw, th, '#F2F2F2' if is_poly else TINT[cls], '0.45' if is_poly else S.CLASS_DARK[cls],
            ls=(0, (3, 1.6)) if is_poly else BORDER[st], lw=0.8)
        col = '0.25' if is_poly else S.CLASS_DARK[cls]
        ax.text(x0 + 1.4, y0 + th - 1.2, S.NAME[k], ha='left', va='top', fontsize=style.TICK_PT, fontweight='bold',
                color=col)
        dp = S.META[k]['dpka'] if is_poly else S.V(k, 'dpka')
        sdp = 'ΔpK$_a$ ' + S.num(dp, 2, sign=True)
        ax.text(x0 + tw - 1.4, y0 + th - 1.2, sdp, ha='right', va='top', fontsize=FS, color='0.15')
        if not is_poly:
            S.mark(k, 'dpka', printed=sdp)
        tag = 'polymorph; ranking only' if is_poly else f"{cls} · {SET_NAME[st]}"
        ax.text(x0 + 1.4, y0 + 1.0, tag, ha='left', va='bottom', fontsize=FS, color=col, style='italic')
        # the two partners: side by side if they fit, else stacked
        # bonds of 2.6 mm where the pair fits (side by side or stacked), else 2.3 mm (dapsone)
        for bond in (2.6, 2.3):
            deps = [rdkit2mpl.Depiction(drawn(mols[role][1]), bond_mm=bond, font_pt=6.0, line_pt=0.5)
                    for role in ('base', 'acid')]
            (wa, ha_), (wb, hb_) = deps[0].size_mm, deps[1].size_mm
            if wa + wb + 2.5 <= tw - 1.0 or (ha_ + hb_ + 0.5 <= th - 8.0 and max(wa, wb) <= tw - 1.0):
                break
        smiles[k] = {mols[role][0]: Chem.MolToSmiles(mols[role][1]) for role in ('base', 'acid')}
        ymid = y0 + 4.2 + (th - 9.6) / 2
        if wa + wb + 2.5 <= tw - 1.0:
            gap = (tw - wa - wb) / 3
            deps[0].draw(ax, (x0 + gap, ymid - ha_ / 2), zorder=3)
            deps[1].draw(ax, (x0 + 2 * gap + wa, ymid - hb_ / 2), zorder=3)
        else:
            assert ha_ + hb_ + 0.5 <= th - 8.0 and max(wa, wb) <= tw - 1.0, (k, deps[0].size_mm, deps[1].size_mm)
            sp = (th - 8.4 - ha_ - hb_) / 3
            deps[1].draw(ax, (x0 + (tw - wb) / 2, y0 + 3.8 + sp), zorder=3)
            deps[0].draw(ax, (x0 + (tw - wa) / 2, y0 + 3.8 + 2 * sp + hb_), zorder=3)
    # key in the last slot
    n = len(P)
    r, c = divmod(n, ncol)
    x0, y0 = 0.6 + c * (tw + g), H - top - (r + 1) * (th + g) + g
    rows = [('salt (ion pair)', TINT['salt'], S.CLASS_DARK['salt'], '-'),
            ('cocrystal (neutral)', TINT['cocrystal'], S.CLASS_DARK['cocrystal'], '-'),
            ('core set', 'white', '0.3', BORDER['core']), ('optional', 'white', '0.3', BORDER['optional']),
            ('extension', 'white', '0.3', BORDER['extension'])]
    for i, (lab, fc, ec, ls) in enumerate(rows):
        yy = y0 + th - 3.6 - i * 4.4
        box(ax, x0 + 1.0, yy - 1.3, 4.5, 2.8, fc, ec, ls=ls, lw=0.8, r=0.5)
        ax.text(x0 + 6.8, yy + 0.1, lab, ha='left', va='center', fontsize=FS)
    ax.text(x0 + 1.0, y0 + 1.0, 'ordered by ΔpK$_a$ (aqueous)', ha='left', va='bottom', fontsize=FS,
            color='0.3', style='italic')
    ax.text(0.6, H - 0.6, '(a)', ha='left', va='top', fontsize=style.PANEL_PT, fontweight='bold')
    ax.text(7.0, H - 0.9, 'the set: base (left / top) and acid partner as found in the crystal', ha='left',
            va='top', fontsize=style.BASE_PT)

    # (b) ladder
    yb = hb - 8.0
    ax.text(0.6, yb + 5.5, '(b)', ha='left', va='top', fontsize=style.PANEL_PT, fontweight='bold')
    ax.text(7.0, yb + 5.2, 'model ladder (one transferring pair; observed packing and X-ray cell)', ha='left',
            va='top', fontsize=style.BASE_PT)
    rungs = [('R1f', 'pair in vacuum,\ncrystal geometry'), ('R4a', 'pair in the\nlattice field'),
             ('R3', 'pair + H-bonded\nshell, vacuum'), ('R4', 'R3 in the\nlattice field'),
             ('R5', 'periodic crystal\n+ pair correction')]
    bw, bh, bg = 19.4, 15.0, 3.0
    rend = fig.canvas.get_renderer()
    for i, (name, desc) in enumerate(rungs):
        x = 1.0 + i * (bw + bg)
        last = name == 'R5'
        box(ax, x, yb - bh - 1.5, bw, bh, '#EAF4EF' if last else 'white', '#00664A' if last else '0.35', lw=0.7)
        t1 = ax.text(x + bw / 2, 0, name, ha='center', va='top', fontsize=style.BASE_PT, fontweight='bold')
        t2 = ax.text(x + bw / 2, 0, desc, ha='center', va='top', fontsize=FS, linespacing=1.1)
        # the name and the two-line description as one block, centred in the box (measured, not guessed)
        h1, h2 = (np.ptp(ax.transData.inverted().transform(t.get_window_extent(rend)), 0)[1] for t in (t1, t2))
        top = yb - 1.5 - (bh - (h1 + 1.6 + h2)) / 2
        t1.set_y(top)
        t2.set_y(top - h1 - 1.6)
        if i:
            arrow(ax, (x - bg + 0.4, yb - bh / 2 - 1.5), (x - 0.4, yb - bh / 2 - 1.5))
    ax.text(1.0, yb - bh - 3.2, 'R4a, R4: point charges (REPEAT; in-crystal and gas-phase CHELPG as checks).\n'
            'R5: PBE0 + DF-MP2, SCS-MP2 or ωB97X-D3(BJ) for the pair.', ha='left', va='top', fontsize=FS,
            color='0.3', linespacing=1.2)

    # (c) cycle
    xc = 112.0
    ax.text(xc, yb + 5.5, '(c)', ha='left', va='top', fontsize=style.PANEL_PT, fontweight='bold')
    ax.text(xc + 6.4, yb + 5.2, 'the cycle', ha='left', va='top', fontsize=style.BASE_PT)
    nodes = {'nv': (xc + 3.0, yb - 1.0, 'neutral pair, vacuum'), 'iv': (xc + 38.0, yb - 1.0, 'ion pair, vacuum'),
             'nc': (xc + 3.0, yb - 15.5, 'neutral pair, crystal'), 'ic': (xc + 38.0, yb - 15.5, 'ion pair, crystal')}
    nw, nh = 23.5, 4.6
    for x, y, lab in nodes.values():
        box(ax, x, y - nh, nw, nh, 'white', '0.35', lw=0.6, r=0.8)
        ax.text(x + nw / 2, y - nh / 2, lab, ha='center', va='center', fontsize=FS)
    yv, yc_ = nodes['nv'][1] - nh / 2, nodes['nc'][1] - nh / 2
    arrow(ax, (xc + 3.0 + nw + 0.5, yv), (xc + 38.0 - 0.5, yv))
    arrow(ax, (xc + 3.0 + nw + 0.5, yc_), (xc + 38.0 - 0.5, yc_), col='#00664A', lw=0.9)
    ax.text((xc + 3.0 + nw + xc + 38.0) / 2, yv + 0.7, 'R1f', ha='center', va='bottom', fontsize=FS,
            fontweight='bold')
    ax.text((xc + 3.0 + nw + xc + 38.0) / 2, yc_ + 0.7, 'R5', ha='center', va='bottom', fontsize=FS,
            fontweight='bold', color='#00664A')
    for x in (xc + 3.0 + nw / 2, xc + 38.0 + nw / 2):
        arrow(ax, (x, nodes['nv'][1] - nh - 0.5), (x, nodes['nc'][1] + 0.5), col='0.45')
    ax.text(xc + 3.0 + nw / 2 + 0.8, (nodes['nv'][1] - nh + nodes['nc'][1]) / 2, 'environment', ha='left',
            va='center', fontsize=FS, color='0.35')
    ax.text(xc + 38.0 + nw / 2 - 0.8, (nodes['nv'][1] - nh + nodes['nc'][1]) / 2, 'environment', ha='right',
            va='center', fontsize=FS, color='0.35')
    ax.text(xc + 3.0, yc_ - 4.6, 'R5 = R1f + ΔE$_\\mathrm{env}$\neach term: E(ion pair) − E(neutral pair)',
            ha='left', va='top', fontsize=FS, linespacing=1.2)
    S.save(fig, STEM, W, extra={'species_smiles_from_crystal': smiles,
                                'not_shown': {'nitrate': 'optional, no results'}})


if __name__ == '__main__':
    main()
