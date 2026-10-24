"""Data plots for the manuscripts: geometry parity, per-bond deviations, descriptor bars, MO levels, DOS,
MEP surface-area histogram, optimisation trajectory.

Data helpers
    bond_table(symbols, ref_xyz, models, labels, bonds=None, exclude_h=True)   -> list[dict]
    load_mo_energies(case_dir)                  -> dict(energies_eV, n_occ, source)
    esp_surface_areas(case_dir, isovalue=0.001) -> dict(esp_kcal, area_A2, total_A2, check)
    trajectory_series(frames, distances)        -> dict(step, energy_kcal, <label>: values)
Plot helpers (ax-level; each figure-level *_figure helper creates the figure at print size)
    parity_plot(ax, ref, models, band=0.02, ...)
    deviation_dots(ax, names, devs, band=0.02, ...)            / deviation_figure(...)
    grouped_bars(ax, categories, series, ylabel, ...)
    mo_levels(ax, models, n_below=4, n_above=4, ...)           / mo_level_figure(...)
    dos(ax, energies, n_occ, fwhm=0.3, ...)                    / dos_figure(models, ...)
    esp_histogram(ax, data, bins, ...)                         / esp_histogram_figure(...)
    trajectory_figure(frames, distances, groups='auto', ...)  energy panel + distance panels, direct end labels
    end_labels(ax, items)                                      right-end line labels spread without overlap
`models` arguments are dicts keyed by model name ('gas', 'pcm' or any label); colours default to
style.MODEL_COLOURS and legend names to style.MODEL_LABELS when the key is 'gas' / 'pcm'.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch

from . import style, geometry as geo, qa

EXTRA_COLOURS = ['#6A3D9A', '#33A02C', '#B15928', '#E31A1C']


def _colour(name, k=0):
    return style.MODEL_COLOURS.get(name, EXTRA_COLOURS[k % len(EXTRA_COLOURS)])


def _label(name):
    return style.MODEL_LABELS.get(name, name)


MINUS = '−'


def _fmt(v, nd=2):
    s = f'{v:.{nd}f}'
    return s.replace('-', MINUS)


# ---------------------------------------------------------------------------------------------------------------
# geometry comparison
# ---------------------------------------------------------------------------------------------------------------


def bond_table(symbols, ref_xyz, models, labels, bonds=None, exclude_h=True):
    """Bond lengths of the reference (X-ray) and each model for the bonds of the reference structure.

    models: dict name -> xyz (same atom order). Returns rows dict(name, i, j, ref, <model>, d_<model>)."""
    ref = np.asarray(ref_xyz, float)
    bonds = geo.detect_bonds(symbols, ref) if bonds is None else bonds
    rows = []
    for i, j in bonds:
        if exclude_h and 'H' in (symbols[i], symbols[j]):
            continue
        r = dict(name=geo.bond_label(labels, i, j), i=i, j=j, ref=geo.distance(ref, i, j))
        for m, X in models.items():
            r[m] = geo.distance(X, i, j)
            r['d_' + m] = r[m] - r['ref']
        rows.append(r)
    return rows


def parity_plot(ax, ref, models, band=0.02, colours=None, names=None, units='Å', markers=('o', 's', '^', 'D'),
                lim=None, xlabel='X-ray bond length (Å)', ylabel='DFT bond length (Å)', legend_loc='upper left',
                ms=3.2, marker_kw=None, rmsd_fmt='{:.3f}'):
    """Parity plot: model vs reference with y = x and a ±band shaded strip; RMSD of each model in the legend.

    marker_kw  optional dict model -> Line2D kwargs (marker, ms, mfc, mec, mew, ...) overriding the default
               'first model filled, later models open' styling
    rmsd_fmt   format of the RMSD in the legend (default '{:.3f}')"""
    ref = np.asarray(ref, float)
    allv = np.concatenate([ref] + [np.asarray(v, float) for v in models.values()])
    if lim is None:
        lo, hi = allv.min(), allv.max()
        pad = 0.04 * (hi - lo) + 0.01
        lim = (lo - pad, hi + pad)
    xx = np.array(lim)
    ax.fill_between(xx, xx - band, xx + band, color='0.88', lw=0, zorder=0,
                    label=f'±{band:.2f} {units}')
    ax.plot(xx, xx, color='0.35', lw=0.7, zorder=1)
    handles = []
    for k, (m, v) in enumerate(models.items()):
        v = np.asarray(v, float)
        rmsd = float(np.sqrt(np.mean((v - ref) ** 2)))
        c = (colours or {}).get(m, _colour(m, k))
        nm = (names or {}).get(m, _label(m))
        filled = k == 0                          # later models as open markers so overlapping points stay visible
        kw = dict(marker=markers[k % len(markers)], ms=ms if filled else ms * 1.05, mfc=c if filled else 'none',
                  mec='white' if filled else c, mew=0.35 if filled else 0.8, color=c, zorder=3 + k)
        kw.update((marker_kw or {}).get(m, {}))
        h = ax.plot(ref, v, label=f'{nm}, RMSD {rmsd_fmt.format(rmsd)} {units}', ls='none', **kw)[0]
        handles.append(h)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    band_h = Patch(color='0.88', label=f'±{band:.2f} {units}')
    ax.legend(handles=handles + [band_h], loc=legend_loc, handletextpad=0.3, borderaxespad=0.4)
    return handles


def parity_figure(ref, models, width=None, **kw):
    style.apply()
    width = style.SINGLE if width is None else width
    fig = plt.figure(figsize=(width, width * 0.92), layout='constrained')
    ax = fig.add_subplot()
    parity_plot(ax, ref, models, **kw)
    return fig, ax


def deviation_dots(ax, names, devs, band=0.02, colours=None, labels=None, markers=('o', 's', '^', 'D'),
                   xlabel=f'DFT {MINUS} X-ray (Å)', ms=3.0, legend=True, marker_kw=None):
    """Horizontal dot chart: one row per bond (names), one marker per model (devs: dict name -> array).

    marker_kw  optional dict model -> Line2D kwargs overriding the default 'first filled, later open' styling"""
    n = len(names)
    y = np.arange(n)[::-1]
    ax.axvspan(-band, band, color='0.9', lw=0, zorder=0)
    ax.axvline(0, color='0.4', lw=0.6, zorder=1)
    for yy in y:
        ax.axhline(yy, color='0.93', lw=0.4, zorder=0)
    hs = []
    for k, (m, d) in enumerate(devs.items()):
        c = (colours or {}).get(m, _colour(m, k))
        filled = k == 0
        kw = dict(marker=markers[k % len(markers)], ms=ms, mfc=c if filled else 'none',
                  mec='white' if filled else c, mew=0.3 if filled else 0.8, color=c, zorder=3 + k)
        kw.update((marker_kw or {}).get(m, {}))
        hs.append(ax.plot(np.asarray(d), y, ls='none', label=(labels or {}).get(m, _label(m)), **kw)[0])
    allv = np.concatenate([np.asarray(d, float) for d in devs.values()])
    lo, hi = min(allv.min(), 0.0), max(allv.max(), 0.0)
    pad = 0.06 * (hi - lo) + 1e-4
    ax.set_xlim(lo - pad, hi + pad)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=style.SMALL_PT)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='y', length=0)
    if legend:
        ax.legend(handles=hs + [Patch(color='0.9', label=f'±{band:.2f} Å')], loc='lower center',
                  bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.3)
    return hs


def deviation_figure(rows, models, top=None, width=None, band=0.02, row_mm=3.0, **kw):
    """Dot chart of per-bond deviations from bond_table rows. top=N keeps the N largest |deviation| bonds."""
    style.apply()
    width = style.SINGLE if width is None else width
    rows = list(rows)
    key = lambda r: max(abs(r['d_' + m]) for m in models)
    if top:
        rows = sorted(rows, key=key, reverse=True)[:top]
    rows = sorted(rows, key=lambda r: r[f'd_{models[0]}'])
    names = [r['name'] for r in rows]
    devs = {m: [r['d_' + m] for r in rows] for m in models}
    h = len(rows) * row_mm * style.MM + 22 * style.MM
    fig = plt.figure(figsize=(width, min(h, style.MAX_HEIGHT)), layout='constrained')
    ax = fig.add_subplot()
    deviation_dots(ax, names, devs, band=band, **kw)
    return fig, ax


# ---------------------------------------------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------------------------------------------


def grouped_bars(ax, categories, series, ylabel='', colours=None, labels=None, value_labels=True, fmt='{:.2f}',
                 width=0.8, legend=True, legend_loc='best'):
    """Grouped bar chart: categories on x, one bar per series (dict name -> values). Value labels above positive
    and below negative bars; the y range is padded so they stay inside the axes."""
    names = list(series)
    k = len(names)
    x = np.arange(len(categories))
    bw = width / k
    allv = np.concatenate([np.asarray(series[n], float) for n in names])
    for j, n in enumerate(names):
        v = np.asarray(series[n], float)
        c = (colours or {}).get(n, _colour(n, j))
        xs = x - width / 2 + bw * (j + 0.5)
        ax.bar(xs, v, bw * 0.92, color=c, label=(labels or {}).get(n, _label(n)), zorder=2, lw=0)
        if value_labels:
            for xi, vi in zip(xs, v):
                ax.annotate(fmt.format(vi).replace('-', MINUS), (xi, vi), xytext=(0, 1.5 if vi >= 0 else -1.5),
                            textcoords='offset points', ha='center', va='bottom' if vi >= 0 else 'top',
                            fontsize=style.SMALL_PT - 0.5 if style.SMALL_PT - 0.5 >= 6 else 6, rotation=90,
                            zorder=3)
    ax.axhline(0, color='0.3', lw=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    lo, hi = min(allv.min(), 0), max(allv.max(), 0)
    span = hi - lo
    ax.set_ylim(lo - (0.2 * span if lo < 0 else 0), hi + 0.2 * span)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend_loc)


# ---------------------------------------------------------------------------------------------------------------
# molecular orbitals
# ---------------------------------------------------------------------------------------------------------------


def read_molden_mos(path):
    """(energies_eV, occupations) of the alpha (or restricted) MOs in a Molden file, in file order."""
    ene, occ, spin = [], [], []
    with open(path) as f:
        in_mo = False
        for line in f:
            if line.startswith('[MO]'):
                in_mo = True
                continue
            if not in_mo:
                continue
            if line.startswith('['):
                break
            t = line.strip()
            if t.startswith('Ene='):
                ene.append(float(t.split('=')[1]))
            elif t.startswith('Spin='):
                spin.append(t.split('=')[1].strip().lower())
            elif t.startswith('Occup='):
                occ.append(float(t.split('=')[1]))
    ene = np.array(ene) * geo.HARTREE_EV
    occ = np.array(occ)
    if spin and len(spin) == len(ene):
        a = np.array([s.startswith('alpha') for s in spin])
        ene, occ = ene[a], occ[a]
    order = np.argsort(ene)
    return ene[order], occ[order]


def load_mo_energies(case_dir):
    """MO energies (eV, ascending) and number of occupied orbitals: fmo.json if present, else final.molden."""
    d = Path(case_dir)
    f = d / 'fmo.json'
    if f.exists():
        j = json.loads(f.read_text())
        if 'mo_energies_eV' in j and 'n_occ' in j:
            e = np.sort(np.asarray(j['mo_energies_eV'], float))
            return dict(energies_eV=e, n_occ=int(j['n_occ']), source=str(f),
                        homo=float(e[j['n_occ'] - 1]), lumo=float(e[j['n_occ']]))
    m = d / 'final.molden'
    if m.exists():
        e, occ = read_molden_mos(m)
        n_occ = int((occ > 0.5).sum())
        return dict(energies_eV=e, n_occ=n_occ, source=str(m), homo=float(e[n_occ - 1]), lumo=float(e[n_occ]))
    raise FileNotFoundError(f'no fmo.json or final.molden in {d}')


def _level_names(n_below, n_above):
    names = {}
    for k in range(n_below + 1):
        names[-k] = 'HOMO' if k == 0 else f'HOMO{MINUS}{k}'
    for k in range(n_above + 1):
        names[k + 1] = 'LUMO' if k == 0 else f'LUMO+{k}'
    return names


def mo_levels(ax, models, n_below=4, n_above=4, colours=None, labels=None, level_width=0.62, spacing=1.0,
              gap_fmt='{:.2f} eV', label_frontier=True, ylabel='Orbital energy (eV)', label_pos='level_end'):
    """MO energy-level diagram: one column per model with HOMO−n … LUMO+n, occupied levels solid, virtual
    dashed, HOMO and LUMO drawn heavier. The frontier labels sit inside the gap, directly against their own level
    (the gap is empty by definition, so near-degenerate neighbours can never be mistaken for them), and a
    double-headed arrow spans the gap with its value. models: dict name -> load_mo_energies() dict.

    label_pos  'level_end' (default, as before): frontier labels right-aligned at the right end of the level;
               'arrow': left-aligned just right of the gap arrow, like the gap value (for narrow panels, where a
               right-aligned label would run into the arrow; it may then extend past the level end)"""
    xt, xl = [], []
    hw = level_width / 2
    for k, (m, d) in enumerate(models.items()):
        e, no = np.asarray(d['energies_eV']), int(d['n_occ'])
        occ = e[max(0, no - 1 - n_below):no]
        vir = e[no:no + 1 + n_above]
        x0 = k * spacing
        c = (colours or {}).get(m, _colour(m, k))
        homo, lumo = e[no - 1], e[no]
        for v in occ:
            ax.plot([x0 - hw, x0 + hw], [v, v], color=c, lw=1.6 if v == homo else 0.9, solid_capstyle='butt')
        for v in vir:
            ax.plot([x0 - hw, x0 + hw], [v, v], color=c, lw=1.6 if v == lumo else 0.9, ls=(0, (2.5, 1.2)),
                    dash_capstyle='butt')
        xa = x0 - hw + 0.12 * level_width
        ax.add_patch(FancyArrowPatch((xa, homo), (xa, lumo), arrowstyle='<|-|>', mutation_scale=5, lw=0.6,
                                     color='0.2', shrinkA=1.2, shrinkB=1.2, zorder=5))
        ax.text(xa + 0.04, (homo + lumo) / 2, gap_fmt.format(lumo - homo), ha='left', va='center',
                fontsize=style.SMALL_PT, zorder=6)
        if label_frontier:
            if label_pos == 'arrow':
                xt_, ha_ = xa + 0.04, 'left'
            else:
                xt_, ha_ = x0 + hw, 'right'
            ax.annotate(f'HOMO {_fmt(homo)}', (xt_, homo), xytext=(0, 1.2), textcoords='offset points',
                        ha=ha_, va='bottom', fontsize=style.SMALL_PT)
            ax.annotate(f'LUMO {_fmt(lumo)}', (xt_, lumo), xytext=(0, -1.6), textcoords='offset points',
                        ha=ha_, va='top', fontsize=style.SMALL_PT)
        xt.append(x0)
        xl.append((labels or {}).get(m, _label(m)))
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-hw - 0.12, (len(models) - 1) * spacing + hw + 0.12)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def mo_level_figure(models, width=None, height=None, **kw):
    style.apply()
    width = style.SINGLE if width is None else width
    height = width * 0.9 if height is None else height
    fig = plt.figure(figsize=(width, height), layout='constrained')
    ax = fig.add_subplot()
    mo_levels(ax, models, **kw)
    return fig, ax


def broaden(energies, grid, fwhm=0.3):
    """Gaussian-broadened density of states (states per eV) on `grid` for level energies (eV)."""
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    e = np.asarray(energies)[:, None]
    return np.exp(-0.5 * ((grid[None] - e) / s) ** 2).sum(0) / (s * np.sqrt(2 * np.pi))


def dos(ax, energies, n_occ, fwhm=0.3, erange=(-12.0, 0.0), colour='0.2', label=None, shade=True, npts=3000,
        mark_frontier=True, gap_label=True, degeneracy=2):
    """Total DOS for one model: occupied part shaded, virtual part as a line only; HOMO/LUMO dotted lines."""
    e = np.asarray(energies)
    g = np.linspace(erange[0], erange[1], npts)
    y = degeneracy * broaden(e, g, fwhm)
    homo, lumo = e[n_occ - 1], e[n_occ]
    mid = (homo + lumo) / 2
    ax.plot(g, y, color=colour, lw=0.9, label=label)
    if shade:
        m = g <= mid
        ax.fill_between(g[m], 0, y[m], color=colour, alpha=0.25, lw=0)
    ymax = y.max()
    if mark_frontier:
        for v in (homo, lumo):
            ax.axvline(v, color=colour, lw=0.5, ls=(0, (1.5, 1.5)))
        if gap_label:
            yy = 0.9 * ymax * 1.18
            ax.add_patch(FancyArrowPatch((homo, yy), (lumo, yy), arrowstyle='<|-|>', mutation_scale=5, lw=0.6,
                                         color='0.2', shrinkA=0, shrinkB=0))
            ax.text(lumo + 0.12, yy, f'{lumo - homo:.2f} eV', ha='left', va='center', fontsize=style.SMALL_PT)
    ax.set_xlim(*erange)
    ax.set_ylim(0, ymax * 1.35)
    return dict(homo=homo, lumo=lumo, ymax=ymax)


def dos_figure(models, fwhm=0.3, erange=(-12.0, 0.0), width=None, panel_h_mm=24, colours=None, labels=None,
               note=False):
    """Stacked DOS panels (one per model), shared energy axis. models: dict name -> load_mo_energies() dict.
    The broadening belongs in the caption; note=True also writes 'FWHM … eV' in the first panel."""
    style.apply()
    width = style.SINGLE if width is None else width
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(width, (panel_h_mm * n + 14) * style.MM), sharex=True,
                             layout='constrained')
    axes = np.atleast_1d(axes)
    for k, (ax, (m, d)) in enumerate(zip(axes, models.items())):
        c = (colours or {}).get(m, _colour(m, k))
        dos(ax, d['energies_eV'], d['n_occ'], fwhm=fwhm, erange=erange, colour=c)
        ax.text(0.015, 0.93, (labels or {}).get(m, _label(m)), transform=ax.transAxes, ha='left', va='top',
                fontsize=style.TICK_PT, color=c, fontweight='bold')
        ax.set_ylabel('DOS (eV$^{-1}$)')
    axes[-1].set_xlabel('Energy (eV)')
    if note:
        axes[0].text(0.985, 0.93, f'FWHM {fwhm:.2f} eV', transform=axes[0].transAxes, ha='right', va='top',
                     fontsize=style.SMALL_PT, color='0.35')
    return fig, axes


# ---------------------------------------------------------------------------------------------------------------
# MEP surface
# ---------------------------------------------------------------------------------------------------------------


def read_cube(path):
    """Gaussian cube -> (data (nx, ny, nz), origin (Bohr), axes (3x3, Bohr per step), atoms (Z, xyz Bohr))."""
    with open(path) as f:
        f.readline(), f.readline()
        t = f.readline().split()
        nat = int(t[0])
        origin = np.array([float(v) for v in t[1:4]])
        n, ax = [], []
        for _ in range(3):
            t = f.readline().split()
            n.append(int(t[0]))
            ax.append([float(v) for v in t[1:4]])
        atoms = [f.readline().split() for _ in range(abs(nat))]
        if nat < 0:
            f.readline()
        data = np.array(f.read().split(), dtype=float)
    return data.reshape(n), origin, np.array(ax), atoms


def read_vtx(path):
    """Multiwfn vtx.txt: first line = count; then X Y Z (Bohr), ESP in a.u., eV, kcal/mol."""
    a = np.loadtxt(path, skiprows=1)
    return dict(xyz_bohr=a[:, :3], esp_au=a[:, 3], esp_kcal=a[:, 5])


def esp_surface_areas(case_dir, isovalue=0.001, rescale=False):
    """Area-weighted ESP samples on the ρ = isovalue surface.

    The ESP values are Multiwfn's own (vtx.txt, evaluated exactly at its surface vertices). vtx.txt carries no
    facets, so the area is taken from a marching-cubes triangulation of density.cube at the same isovalue: each
    triangle gets the ESP of the nearest Multiwfn vertex. The result is checked against Multiwfn's total,
    positive and negative areas (mep_surface.json); rescale=True scales the areas so the total matches exactly.
    Returns dict(esp_kcal, area_A2, total_A2, check={...})."""
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree
    d = Path(case_dir)
    vtx = read_vtx(d / 'vtx.txt')
    rho, origin, axes, _ = read_cube(d / 'density.cube')
    if not np.allclose(axes, np.diag(np.diag(axes))):
        raise ValueError('non-orthogonal cube axes are not supported')
    verts, faces, _, _ = marching_cubes(rho, level=isovalue, spacing=tuple(np.diag(axes)))
    verts = verts + origin
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * geo.BOHR ** 2
    cen = tri.mean(1)
    dist, idx = cKDTree(vtx['xyz_bohr']).query(cen)
    esp = vtx['esp_kcal'][idx]
    total = float(area.sum())
    chk = dict(n_vertices=len(vtx['esp_kcal']), n_triangles=len(faces), total_A2=total,
               max_nn_dist_A=float(dist.max() * geo.BOHR), mean_nn_dist_A=float(dist.mean() * geo.BOHR),
               positive_A2=float(area[esp > 0].sum()), negative_A2=float(area[esp < 0].sum()))
    ms = d / 'mep_surface.json'
    if ms.exists():
        j = json.loads(ms.read_text())
        chk.update(mwfn_total_A2=j.get('area_A2'), mwfn_positive_A2=j.get('positive_area_A2'),
                   mwfn_negative_A2=j.get('negative_area_A2'))
        if j.get('area_A2'):
            chk['total_ratio'] = total / j['area_A2']
            if rescale:
                area = area * j['area_A2'] / total
                total = float(area.sum())
    return dict(esp_kcal=esp, area_A2=area, total_A2=total, check=chk)


def esp_histogram(ax, data, bins=None, colours=None, labels=None, mode='step', xlabel='ESP on the surface (kcal mol$^{-1}$)',
                  ylabel='Surface area (Å$^2$)', legend=True):
    """Surface area per ESP bin. data: dict name -> esp_surface_areas() dict. mode 'step' (outlines, several
    models) or 'bars' (bars coloured red for negative, blue for positive ESP, for a single model)."""
    allv = np.concatenate([d['esp_kcal'] for d in data.values()])
    if bins is None:
        lo = 5 * np.floor(allv.min() / 5)
        hi = 5 * np.ceil(allv.max() / 5)
        bins = np.arange(lo, hi + 2.5, 2.5)
    for k, (m, d) in enumerate(data.items()):
        h, e = np.histogram(d['esp_kcal'], bins=bins, weights=d['area_A2'])
        c = (colours or {}).get(m, _colour(m, k))
        nm = (labels or {}).get(m, _label(m))
        if mode == 'bars':
            cen = 0.5 * (e[1:] + e[:-1])
            col = np.where(cen < 0, '#C0392B', '#2E6FBF')
            ax.bar(cen, h, width=np.diff(e) * 0.95, color=col, lw=0, zorder=2)
        else:
            ax.stairs(h, e, color=c, lw=1.0, label=nm, zorder=3 + k)
            ax.stairs(h, e, color=c, alpha=0.08, fill=True, zorder=2)
    ax.axvline(0, color='0.4', lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and mode != 'bars':
        ax.legend(loc='upper right')


def esp_histogram_figure(data, width=None, **kw):
    style.apply()
    width = style.SINGLE if width is None else width
    fig = plt.figure(figsize=(width, width * 0.66), layout='constrained')
    ax = fig.add_subplot()
    esp_histogram(ax, data, **kw)
    return fig, ax


# ---------------------------------------------------------------------------------------------------------------
# optimisation trajectory
# ---------------------------------------------------------------------------------------------------------------


def trajectory_series(frames, distances):
    """Energies (kcal/mol relative to the last frame) and chosen distances along an optimisation.

    frames: list of geometry.Frame (read_xyz_frames); distances: dict label -> (i, j)."""
    E = np.array([f.energy if f.energy is not None else np.nan for f in frames])
    out = dict(step=np.arange(len(frames)))
    if np.isfinite(E).any():
        out['energy_kcal'] = (E - E[np.isfinite(E)][-1]) * geo.HARTREE_KCAL
    for lab, (i, j) in distances.items():
        out[lab] = np.array([geo.distance(f.xyz, i, j) for f in frames])
    return out


def end_labels(ax, items, pad_pt=3.0, gap_pt=0.8, fontsize=None):
    """Direct labels at the right end of lines, spread vertically so they never overlap (1D repulsion in display
    space, after the layout is frozen). items: list of (text, x_last, y_last, colour). Returns the Text artists."""
    from .labels import freeze_layout
    fontsize = style.SMALL_PT if fontsize is None else fontsize
    fig = ax.figure
    freeze_layout(fig)
    r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    arts, want = [], []
    for text, x, y, c in items:
        t = ax.annotate(text, (x, y), xytext=(pad_pt, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=fontsize, color=c, annotation_clip=False)
        arts.append(t)
        want.append(ax.transData.transform((x, y))[1])
    h = np.array([a.get_window_extent(r).height for a in arts]) + gap_pt * ppt
    order = np.argsort(want)
    y = np.array(want, float)[order]
    hh = h[order]
    bb = ax.bbox
    for _ in range(200):                     # alternate pushes until no neighbours overlap
        moved = False
        for k in range(1, len(y)):
            need = (hh[k] + hh[k - 1]) / 2
            if y[k] - y[k - 1] < need - 1e-6:
                d = (need - (y[k] - y[k - 1])) / 2
                y[k] += d
                y[k - 1] -= d
                moved = True
        lo, hi = bb.y0 + hh[0] / 2, bb.y1 - hh[-1] / 2
        if len(y) and y[0] < lo:
            y += lo - y[0]
        if len(y) and y[-1] > hi:
            y -= y[-1] - hi
        if not moved:
            break
    for k, i in enumerate(order):
        dy = (y[k] - want[i]) / ppt
        arts[i].xyann = (pad_pt + (4.0 if abs(dy) > 1.0 else 0.0), dy)
        if abs(dy) > 1.0:                    # displaced label: short connector from the line end
            x0, y0 = arts[i].xy
            p0 = ax.transData.transform((x0, y0))
            p1 = p0 + np.array([(pad_pt + 3.5) * ppt, dy * ppt])
            ax.plot(*ax.transData.inverted().transform([p0 + np.array([1.2 * ppt, 0]), p1]).T,
                    color=arts[i].get_color(), lw=0.4, clip_on=False, zorder=1)
    return arts


def trajectory_figure(frames, distances, width=None, colours=None, energy=True, groups='auto',
                      dist_ylabel='Distance (Å)', energy_ylabel=r'$E - E_\mathrm{final}$ (kcal mol$^{-1}$)',
                      markers=True, label_space_mm=14.0):
    """Energy (top, when the XYZ comment lines carry energies) and selected distances versus optimisation step.

    distances: dict label -> (i, j). groups: 'auto' puts covalent X–H-like distances (< 1.35 Å on average) and
    longer contacts in separate panels so each keeps a readable scale; or a list of label lists; or None (one
    panel). Lines are labelled directly at their right end (no legend over the data)."""
    style.apply()
    width = style.SINGLE if width is None else width
    ser = trajectory_series(frames, distances)
    has_e = energy and 'energy_kcal' in ser
    labs = list(distances)
    if groups == 'auto':
        short = [l for l in labs if np.nanmean(ser[l]) < 1.35]
        long_ = [l for l in labs if l not in short]
        groups = [g for g in (long_, short) if g]
    elif groups is None:
        groups = [labs]
    n = len(groups) + (1 if has_e else 0)
    hr = ([1.0] if has_e else []) + [1.0] * len(groups)
    H = width * (0.34 * n + 0.18)
    fig, axes = plt.subplots(n, 1, figsize=(width, H), sharex=True, layout='constrained',
                             gridspec_kw=dict(height_ratios=hr))
    axes = np.atleast_1d(axes)
    s = ser['step']
    k0 = 0
    if has_e:
        ax = axes[0]
        ax.plot(s, ser['energy_kcal'], '-o' if markers else '-', color='0.2', ms=2.0, lw=0.8)
        ax.set_ylabel(energy_ylabel)
        ax.axhline(0, color='0.6', lw=0.4, ls=(0, (3, 2)))
        k0 = 1
    pal = ['#1F5AA6', '#E07B00', '#2CA02C', '#8E44AD', '#C0392B', '#555555']
    cmap = {l: (colours or {}).get(l, pal[k % len(pal)]) for k, l in enumerate(labs)}
    for g, ax in zip(groups, axes[k0:]):
        for lab in g:
            ax.plot(s, ser[lab], '-o' if markers else '-', color=cmap[lab], ms=2.0, lw=0.9)
        lo = min(np.nanmin(ser[l]) for l in g)
        hi = max(np.nanmax(ser[l]) for l in g)
        pad = max(0.08 * (hi - lo), 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(dist_ylabel)
    axes[-1].set_xlabel('Optimisation step')
    from matplotlib.ticker import MaxNLocator
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # room for the end labels on the right, then freeze and place them
    fig.get_layout_engine().set(rect=(0, 0, 1 - label_space_mm * style.MM / width, 1))
    for g, ax in zip(groups, axes[k0:]):
        end_labels(ax, [(l, s[-1], ser[l][-1], cmap[l]) for l in g])
    return fig, axes, ser
