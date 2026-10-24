"""Shared helpers for the CTC series figures (paper/ctc/figures; plan optc/CTC_REWRITE_PLAN.md section 3).

    cd /root/workspace/cqc-4-sulfamoylanilinium-dihydrogen-phosphate
    nice -n 10 tools/paper-env/bin/python paper/ctc/figures/scripts/<stem>.py

Data   every number from optc/results/summary.json (md5 asserted: the copy pinned in git, b143860); crystal metadata
       from optc/crystals.json.  V(key, *path) reads crystals[key].<path> and records it; PT(key, *path) reads a
       quantity stored as "other state - observed state" (the margin sign of analyse.py) and returns it as
       dE_PT = E(ion pair) - E(neutral pair), with the factor taken from summary.json itself
       (fixed_ion_minus_neutral.R1f / R1f.PBE0 = +1 for cocrystals, -1 for salts; asserted to be exactly +-1).
Output save(fig, stem, width_mm): figkit.qa strict save to paper/ctc/figures/<stem>.png/.pdf and <stem>.facts.json
       (every summary.json key drawn or printed, with the strings printed).  Nothing else is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'paper' / '_shared'))

from figkit import style, qa  # noqa: E402

SUMMARY = REPO / 'optc' / 'results' / 'summary.json'
SUMMARY_MD5 = '411d4c2ad29499c8961de1bae2fc5b02'
CRYSTALS = REPO / 'optc' / 'crystals.json'
STRUCT = REPO / 'optc' / 'structures'
MM = 1 / 25.4
MINUS = '−'

assert hashlib.md5(SUMMARY.read_bytes()).hexdigest() == SUMMARY_MD5, 'summary.json is not the pinned copy'
SUM = json.loads(SUMMARY.read_text())
CRY = {c['key']: c for c in SUM['crystals']}
META = json.loads(CRYSTALS.read_text())['crystals']
USED: dict = {}

# the 13 crystals with a transferring bond and results (brcoc: polymorph ranking only; nitrate: no results)
KEYS = [k for k, c in CRY.items() if c['fixed_ion_minus_neutral'].get('R1f') is not None]
assert len(KEYS) == 13 and 'brcoc' not in KEYS and 'nitrate' not in CRY, KEYS
SALTS = [k for k in KEYS if CRY[k]['observed'] == 'salt']
COCRYSTALS = [k for k in KEYS if CRY[k]['observed'] == 'cocrystal']

# house colours: blue = ion pair / salt, orange = neutral pair / cocrystal (figkit _saph SIDE_COLOURS)
CLASS_COLOURS = {'salt': '#1F5AA6', 'cocrystal': '#E07B00'}
CLASS_DARK = {'salt': '#123A70', 'cocrystal': '#9A5300'}
SET_TAG = {'core': 'core', 'optional': 'opt.', 'extension': 'ext.'}
# display names (Table 1 of the paper defines them): dihydrogen phosphate salts on the SAPH pattern, 4-sulfamoylanilinium
# perchlorate / sulfamate, the other pairs base and acid partner with '·' for a cocrystal and an en dash for a salt
NAME = {'cyano': 'CNAPH', 'saph': 'SAPH', 'carboxy': 'CXAPH', 'chloro': 'CLAPH', 'anis': 'MOAPH', 'perchlorate': 'SAPC',
        'sulfamate': 'SASF', 'dapsone': 'DDS·DNBA', 'mppcp': 'MP·PCP', 'pcplut': 'LUT·PCP', 'nbaetpy': 'EPY·NBA',
        'brsalt': 'BMA–DNB', 'clsalt': 'CMA–DNB', 'brcoc': 'BMA·DNBA'}
LEVELS = ['PBE', 'PBE0', 'PBE0+MP2', 'PBE0+SCS_MP2', 'PBE0+wB97X_D3BJ']
LEVEL_LABELS = {'PBE': 'PBE', 'PBE0': 'PBE0', 'PBE0+MP2': 'PBE0+MP2', 'PBE0+SCS_MP2': 'PBE0+SCS-MP2',
                'PBE0+wB97X_D3BJ': 'PBE0+ωB97X-D3(BJ)'}
# Okabe-Ito based, distinct under colour-vision deficiency; the corrected levels darker than the DFT ones
LEVEL_COLOURS = {'PBE': '#999999', 'PBE0': '#56B4E9', 'PBE0+MP2': '#0B4F8A', 'PBE0+SCS_MP2': '#009E73',
                 'PBE0+wB97X_D3BJ': '#D55E00'}
LEVEL_MARKERS = {'PBE': 'v', 'PBE0': '^', 'PBE0+MP2': 'o', 'PBE0+SCS_MP2': 's', 'PBE0+wB97X_D3BJ': 'D'}


def _key(k, path):
    return f'crystals[{k}].' + '.'.join(str(p) for p in path)


def _get(k, path):
    v = CRY[k]
    for p in path:
        v = v[p]
    return v


def V(k, *path, note=None):
    """Raw summary.json value crystals[k].<path> (recorded)."""
    v = _get(k, path)
    USED.setdefault(_key(k, path), {'value': v, 'printed': []})
    if note:
        USED[_key(k, path)].setdefault('note', note)
    return v


def G(*path):
    """Raw top-level summary.json value (recorded)."""
    v = SUM
    for p in path:
        v = v[p]
    USED.setdefault('.'.join(str(p) for p in path), {'value': v, 'printed': []})
    return v


def sign_pt(k):
    """Factor from 'other - observed' to E(ion pair) - E(neutral pair), from summary.json itself."""
    f = _get(k, ('fixed_ion_minus_neutral', 'R1f')) / _get(k, ('R1f', 'PBE0'))
    assert abs(abs(f) - 1) < 1e-9, (k, f)
    assert (f > 0) == (CRY[k]['observed'] == 'cocrystal'), k
    return round(f)


def PT(k, *path):
    """A margin-sign quantity of summary.json returned as dE_PT (recorded with the factor used)."""
    s = sign_pt(k)
    v = _get(k, path)
    out = [s * x for x in v] if isinstance(v, (list, tuple)) else s * v
    USED.setdefault(_key(k, path), {'value': v, 'printed': [],
                                    'note': f'stored as other - observed; drawn x{s:+d} as E(ion) - E(neutral)'})
    return out


def mark(key_or_k, *path, printed='plotted'):
    """Record a printed string (or 'plotted') against a key already read."""
    key = _key(key_or_k, path) if path else key_or_k
    USED[key]['printed'].append(printed)


def rnd(x, nd):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_EVEN))


def num(x, nd=1, sign=False):
    v = rnd(x, nd)
    v = 0.0 if v == 0 else v
    s = f'{abs(v):.{nd}f}'
    return (MINUS + s) if v < 0 else (('+' + s) if sign else s)


def by_dpka(keys):
    return sorted(keys, key=lambda k: (CRY[k]['dpka'], k))


def setup():
    style.apply()
    import matplotlib.pyplot as plt
    plt.rcParams['savefig.bbox'] = 'tight'
    return plt


def figure_mm(w_mm, h_mm):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig._mm = (w_mm, h_mm)
    return fig


def axes_mm(fig, x, y, w, h, **kw):
    W, H = fig._mm
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def panel_letter(fig, x_mm, y_mm, letter, title=None):
    W, H = fig._mm
    fig.text(x_mm / W, y_mm / H, f'({letter})', fontsize=style.PANEL_PT, fontweight='bold', ha='left',
             va='baseline')
    if title:
        fig.text((x_mm + 5.2) / W, y_mm / H, title, fontsize=style.BASE_PT, ha='left', va='baseline')


def save(fig, stem, width_mm, extra=None, strict=True, **kw):
    """figkit.qa strict save + <stem>.facts.json (summary.json keys -> printed strings or 'checked')."""
    if os.environ.get('CTC_QA_LAX'):
        strict = False
    out = qa.save(fig, FIG / stem, strict=strict, max_width_mm=width_mm, glyphs=True, **kw)
    from PIL import Image
    with Image.open(FIG / f'{stem}.png') as im:
        w, h = im.size
        dpi = im.info.get('dpi', (600, 600))[0]
    facts = {'script': f'paper/ctc/figures/scripts/{stem}.py', 'source': 'optc/results/summary.json',
             'source_md5': SUMMARY_MD5, 'sign': 'dE_PT = E(ion pair) - E(neutral pair), kJ/mol per transferring pair '
             'unless a key note says otherwise',
             'used': {k: ({'value': v['value'], 'printed': v['printed'] or 'checked', **({'note': v['note']} if
                                                                                         'note' in v else {})})
                      for k, v in sorted(USED.items())}}
    if extra:
        facts.update(extra)
    (FIG / f'{stem}.facts.json').write_text(json.dumps(facts, indent=1, ensure_ascii=False, default=float) + '\n')
    print(f'{stem}: {w / dpi * 25.4:.1f} x {h / dpi * 25.4:.1f} mm ({w} x {h} px); wrote',
          ', '.join(p.name for p in out), f'+ {stem}.facts.json')
    return out
