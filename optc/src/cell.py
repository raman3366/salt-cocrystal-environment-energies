#!/usr/bin/env python3
"""Crystal builder of the option C series (optc/PLAN.md, gate G1): CIF -> full P1 cell, molecules, charges,
hydrogen bonds and the transferring D-H...A bond, written as optc/structures/<key>/cell.json + start.extxyz.

    python -I optc/src/cell.py build [--only KEY ...]     # every crystal of optc/crystals.json
    python -I optc/src/cell.py check                      # G1: rebuild, compare, round trip through a pw.x input

Rules (fixed in optc/PLAN.md before any run):
  * Symmetry operations come from the CIF's own loop (_space_group_symop_operation_xyz, or the old
    _symmetry_equiv_pos_as_xyz); sites are wrapped into [0, 1) and merged when closer than 0.05 A.
  * Covalent bonds: d < r_i + r_j + 0.40 A (Cordero 2008 radii), each H bonded to its nearest heavy atom only.
    Molecules are the connected components under the minimum image; each is made whole for later cluster work.
  * Charges come from the molecular formula (crystals.json "species": Hill formula -> charge); a molecule whose
    formula is not listed stops the build.
  * Hydrogen bonds: H...A < 2.60 A and D-H...A > 110 deg, A in (N, O, Cl), A not within two bonds of D.  The
    transferring bond: donor atom D in the species marked "donor" with D_nH hydrogens (3 = the anilinium NH3+,
    1 = the acid O-H), acceptor A in the species marked "acceptor", A of the element given and (if
    "acceptor_free" is set) carrying no H, and a near-linear bond (D-H...A >= 140 deg, the proton must be able to
    move along it).  Shortest D...A wins; ties (< 0.005 A) go to the larger angle, then to CIF order.  The
    runner-up within 0.10 A is recorded (near-tie rule of the plan).
  * Collective transfer: every copy of the chosen bond in the cell (same CIF labels of D, H, A).
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import gemmi

OPTC = Path(__file__).resolve().parent.parent
REG = OPTC / 'crystals.json'
OUTD = OPTC / 'structures'

# Cordero et al., Dalton Trans. (2008) 2832, single-bond covalent radii (A)
RCOV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20}
BOND_TOL = 0.40
MERGE_A = 0.05
HB_HA_MAX, HB_ANG_MIN = 2.60, 110.0
TRANSFER_ANG_MIN = 140.0


def symops(block):
    xyz = list(block.find_values('_space_group_symop_operation_xyz')) or \
        list(block.find_values('_symmetry_equiv_pos_as_xyz'))
    if not xyz:
        raise SystemExit('no symmetry operations in the CIF')
    return [gemmi.Op(gemmi.cif.as_string(s).replace(' ', '')) for s in xyz]


def hill(counts):
    keys = sorted(counts, key=lambda e: (e != 'C', e != 'H' if 'C' in counts else False, e))
    return ''.join(f'{e}{counts[e] if counts[e] > 1 else ""}' for e in keys)


class Cell:
    """Lattice (rows = a, b, c in A), species, fractional coordinates, CIF labels; minimum-image helpers."""

    def __init__(self, lat, syms, frac, labels):
        self.lat = np.array(lat, float)
        self.syms = list(syms)
        self.frac = np.array(frac, float)
        self.labels = list(labels)
        self.nat = len(self.syms)

    @property
    def cart(self):
        return self.frac @ self.lat

    def mic(self, i, j):
        """Cartesian vector j - i under the minimum image (robust for skewed cells: 27 images checked)."""
        df = self.frac[j] - self.frac[i]
        df -= np.round(df)
        best = None
        for s in np.ndindex(3, 3, 3):
            v = (df + np.array(s) - 1) @ self.lat
            if best is None or v @ v < best @ best:
                best = v
        return best

    def dist(self, i, j):
        return float(np.linalg.norm(self.mic(i, j)))

    def all_pairs_within(self, rmax):
        """(i, j, vector) for every pair (and periodic image) closer than rmax, i < j or image != 0."""
        cart = self.cart
        inv = np.linalg.inv(self.lat)
        out = []
        reach = [int(math.ceil(rmax * np.linalg.norm(inv[:, k]))) + 1 for k in range(3)]
        shifts = [np.array(s) for s in np.ndindex(*(2 * r + 1 for r in reach))]
        shifts = [s - np.array(reach) for s in shifts]
        for s in shifts:
            t = s @ self.lat
            d = cart[None, :, :] + t[None, None, :] - cart[:, None, :]
            r = np.linalg.norm(d, axis=2)
            ii, jj = np.where((r < rmax) & (r > 1e-6))
            for i, j in zip(ii, jj):
                if (s == 0).all() and j <= i:
                    continue
                out.append((int(i), int(j), d[i, j].copy(), tuple(int(x) for x in s)))
        return out


def expand(cif_path):
    block = gemmi.cif.read(str(cif_path)).sole_block()
    small = gemmi.make_small_structure_from_block(block)
    ops = symops(block)
    lat = np.array(small.cell.orth.mat.tolist()).T     # rows a, b, c in gemmi's frame (a along x, b in xy)
    syms, frac, labels = [], [], []
    for site in small.sites:
        if site.occ < 0.999:
            raise SystemExit(f'{cif_path.name}: partial occupancy at {site.label}')
        f0 = np.array([site.fract.x, site.fract.y, site.fract.z])
        for op in ops:
            f = np.array(op.apply_to_xyz(f0.tolist())) % 1.0
            dup = False
            for k, fk in enumerate(frac):
                if syms[k] != site.element.name:
                    continue
                df = f - fk
                df -= np.round(df)
                if np.linalg.norm(df @ lat) < MERGE_A:
                    dup = True
                    break
            if not dup:
                syms.append(site.element.name)
                frac.append(f)
                labels.append(site.label)
    meta = {k: (block.find_value(t) or '').strip("'\"") for k, t in (
        ('space_group', '_symmetry_space_group_name_H-M'), ('space_group_alt', '_space_group_name_H-M_alt'),
        ('Z', '_cell_formula_units_Z'), ('T', '_diffrn_ambient_temperature'),
        ('radiation', '_diffrn_radiation_type'), ('probe', '_diffrn_radiation_probe'),
        ('formula_sum', '_chemical_formula_sum'), ('formula_moiety', '_chemical_formula_moiety'))}
    meta['cell'] = list(small.cell.parameters)
    meta['n_symops'] = len(ops)
    meta['symops'] = [op.triplet() for op in ops]
    return Cell(lat, syms, frac, labels), meta


def bonds(cell):
    """Covalent bonds (i, j, shift) with each H on its nearest heavy atom only."""
    pairs = cell.all_pairs_within(2.4)
    heavy = []
    hbest = {}
    for i, j, v, s in pairs:
        a, b = cell.syms[i], cell.syms[j]
        r = float(np.linalg.norm(v))
        if r > RCOV[a] + RCOV[b] + BOND_TOL:
            continue
        if a == 'H' and b == 'H':
            continue
        if a == 'H' or b == 'H':
            h, x = (i, j) if a == 'H' else (j, i)
            if h not in hbest or r < hbest[h][0]:
                hbest[h] = (r, x)
            continue
        heavy.append((i, j, r))
    hb = [(h, x, r) for h, (r, x) in hbest.items()]
    return heavy, hb


def molecules(cell, heavy, hb):
    parent = list(range(cell.nat))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = [(i, j) for i, j, _ in heavy] + [(h, x) for h, x, _ in hb]
    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(cell.nat):
        groups.setdefault(find(i), []).append(i)
    adj = {i: set() for i in range(cell.nat)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    mols = []
    for g in sorted(groups.values(), key=min):
        # whole molecule: breadth-first unwrap from the lowest index
        whole = {g[0]: cell.cart[g[0]]}
        todo = [g[0]]
        while todo:
            i = todo.pop()
            for j in adj[i]:
                if j not in whole:
                    whole[j] = whole[i] + cell.mic(i, j)
                    todo.append(j)
        counts = {}
        for i in g:
            counts[cell.syms[i]] = counts.get(cell.syms[i], 0) + 1
        mols.append({'atoms': sorted(g), 'formula': hill(counts),
                     'whole_cart': {int(i): [round(float(x), 6) for x in whole[i]] for i in sorted(g)}})
    return mols, adj


def angle(cell, d, h, a):
    v1 = cell.mic(h, d)
    v2 = cell.mic(h, a)
    c = v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def hbonds(cell, hb, mol_of, adj):
    hpar = {h: x for h, x, _ in hb}
    out = []
    for h, d in hpar.items():
        near = {d} | adj[d] | {k for j in adj[d] for k in adj[j]}
        for a in range(cell.nat):
            if a in near or cell.syms[a] not in ('N', 'O', 'Cl'):
                continue
            rha = cell.dist(h, a)
            if rha > HB_HA_MAX:
                continue
            ang = angle(cell, d, h, a)
            if ang < HB_ANG_MIN:
                continue
            # D...A of the image bonded through H (not the nearest image of A: in sulfamate, 20 Oct 2026, the nearest
            # O21 to N4 is bonded by H43 and the O21 bonded by H41 is one lattice vector further, 3.0 A)
            da = float(np.linalg.norm(cell.mic(d, h) + cell.mic(h, a)))
            out.append({'D': d, 'H': h, 'A': a, 'labels': f'{cell.labels[d]}-{cell.labels[h]}...{cell.labels[a]}',
                        'DA': round(da, 4), 'HA': round(rha, 4), 'DH': round(cell.dist(d, h), 4),
                        'angle': round(ang, 2), 'inter': mol_of[d] != mol_of[a]})
    out.sort(key=lambda r: (r['DA'], r['D'], r['A']))
    return out


def build(key, spec):
    cif = OPTC / 'cif' / spec['cif']
    cell, meta = expand(cif)
    heavy, hb = bonds(cell)
    mols, adj = molecules(cell, heavy, hb)
    mol_of = {}
    for k, m in enumerate(mols):
        for i in m['atoms']:
            mol_of[i] = k
    species = spec['species']
    for m in mols:
        if m['formula'] not in species:
            raise SystemExit(f'{key}: molecule {m["formula"]} is not in crystals.json species {sorted(species)}')
        m['charge'] = species[m['formula']]['charge']
        m['role'] = species[m['formula']].get('role', '')
    hbs = hbonds(cell, hb, mol_of, adj)
    nh = {i: sum(1 for j in adj[i] if cell.syms[j] == 'H') for i in range(cell.nat)}
    t = spec.get('transfer')
    chosen, runner, copies = None, None, []
    if t:
        cands = []
        for r in hbs:
            md, ma = mols[mol_of[r['D']]], mols[mol_of[r['A']]]
            if md['role'] != t['donor'] or ma['role'] != t['acceptor']:
                continue
            if cell.syms[r['D']] != t['D'] or cell.syms[r['A']] != t['A']:
                continue
            if t.get('acceptor_free') and nh[r['A']] > 0:
                continue
            if nh[r['D']] != t['D_nH'] or r['angle'] < TRANSFER_ANG_MIN or not r['inter']:
                continue
            cands.append(r)
        if not cands:
            raise SystemExit(f'{key}: no transferring hydrogen bond matches {t}')
        # one per label triple (copies are symmetry images), best first
        by_lab = {}
        for r in cands:
            by_lab.setdefault(r['labels'], []).append(r)
        reps = [min(v, key=lambda r: r['D']) for v in by_lab.values()]
        reps.sort(key=lambda r: (round(r['DA'] / 0.005), -r['angle'], r['D']))
        chosen = reps[0]
        if len(reps) > 1 and reps[1]['DA'] - chosen['DA'] < 0.10:
            runner = reps[1]
        copies = [{'D': r['D'], 'H': r['H'], 'A': r['A']} for r in by_lab[chosen['labels']]]
    zmeta = int(meta['Z']) if meta['Z'].isdigit() else None
    rec = {'key': key, 'cif': spec['cif'], 'cif_md5': __import__('hashlib').md5(cif.read_bytes()).hexdigest(),
           'meta': meta, 'nat': cell.nat, 'lattice_A': cell.lat.round(8).tolist(), 'species': cell.syms,
           'labels': cell.labels, 'frac': cell.frac.round(10).tolist(),
           'molecules': [{k: v for k, v in m.items() if k != 'whole_cart'} for m in mols],
           'whole_cart': {str(i): xyz for m in mols for i, xyz in m['whole_cart'].items()},
           'cell_charge': sum(m['charge'] for m in mols),
           'hbonds': hbs, 'transfer': chosen, 'runner_up': runner, 'copies': copies, 'Z_cif': zmeta,
           'n_formula_units': len(copies) if copies else None,
           'markers': markers(cell, adj)}
    if rec['cell_charge'] != 0:
        raise SystemExit(f'{key}: cell charge {rec["cell_charge"]}')
    return cell, rec


def markers(cell, adj):
    """Protonation-state evidence from heavy atoms: C-N, P-O, C-O, Cl-O and S-O lengths (A), first copy of each label."""
    out, seen = [], set()
    for i in range(cell.nat):
        for j in sorted(adj[i]):
            a, b = cell.syms[i], cell.syms[j]
            pair = {a, b}
            if pair in ({'C', 'N'}, {'P', 'O'}, {'C', 'O'}, {'Cl', 'O'}, {'S', 'O'}, {'S', 'N'}) and i < j:
                lab = tuple(sorted((cell.labels[i], cell.labels[j])))
                if lab in seen:
                    continue
                seen.add(lab)
                hs = [cell.labels[k] for k in adj[i] | adj[j] if cell.syms[k] == 'H']
                out.append({'bond': '-'.join(lab), 'd': round(cell.dist(i, j), 4), 'H_on_ends': hs})
    return out


def write_extxyz(cell, path, comment=''):
    lat = cell.lat
    lines = [str(cell.nat), 'Lattice="' + ' '.join(f'{x:.8f}' for x in lat.ravel()) + '" Properties=species:S:1:pos:R:3 '
             + f'pbc="T T T" {comment}']
    for s, x in zip(cell.syms, cell.cart):
        lines.append(f'{s:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}')
    Path(path).write_text('\n'.join(lines) + '\n')


def cell_from_rec(rec):
    return Cell(rec['lattice_A'], rec['species'], rec['frac'], rec['labels'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'check'])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    reg = json.loads(REG.read_text())
    keys = [k for k in reg['crystals'] if not a.only or k in a.only]
    for key in keys:
        spec = reg['crystals'][key]
        cell, rec = build(key, spec)
        d = OUTD / key
        if a.cmd == 'build':
            d.mkdir(parents=True, exist_ok=True)
            (d / 'cell.json').write_text(json.dumps(rec, indent=1))
            write_extxyz(cell, d / 'start.extxyz', f'key={key} source=COD{spec["cif"]}')
        else:
            old = json.loads((d / 'cell.json').read_text())
            same = old['frac'] == rec['frac'] and old['species'] == rec['species'] and old['copies'] == rec['copies']
            print(f"{key}: rebuild identical: {same}, runner-up identical: {old.get('runner_up') == rec['runner_up']}")
        t = rec['transfer']
        print(f"{key:10s} nat {rec['nat']:3d} mols {len(rec['molecules'])} "
              f"({', '.join(sorted(set(m['formula'] + ('%+d' % m['charge'] if m['charge'] else '') for m in rec['molecules'])))}) "
              + (f"transfer {t['labels']} DA {t['DA']:.3f} HA {t['HA']:.3f} ang {t['angle']:.1f} copies {len(rec['copies'])}"
                 if t else 'no transfer bond (ranking only)')
              + (f" | runner-up {rec['runner_up']['labels']} DA {rec['runner_up']['DA']:.3f}" if rec['runner_up'] else ''))


if __name__ == '__main__':
    main()
