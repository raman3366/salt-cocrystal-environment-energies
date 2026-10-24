"""House style for every figure of the SAPH manuscript and SI (figkit adapted from cqc-dft).

Import this first:  from figkit import style; style.apply()
Sizes follow common journal column widths; text sizes are chosen for the PRINTED size, so figures must be
created at their final width (style.SINGLE / ONE_HALF / DOUBLE) and never scaled up or down afterwards.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]
FONT_DIR = REPO / 'tools' / 'fonts' / 'extracted' / 'usr' / 'share' / 'fonts' / 'truetype' / 'liberation'
# Bulky raw results (cubes, Molden files, Multiwfn grids) are not in git: on the Pi they are the VM pulls in
# /root/cqc-sfa-archive/vm/cqc-sfa (same molecules/<id>/results/<case>/ layout). Override with SAPH_RAW_DATA.
DATA = Path(os.environ.get('SAPH_RAW_DATA', '/root/cqc-sfa-archive/vm/cqc-sfa'))
MOLECULES = REPO / 'molecules'                  # git-tracked inputs and curated results

MM = 1 / 25.4
SINGLE = 85 * MM          # one journal column
ONE_HALF = 120 * MM
DOUBLE = 175 * MM         # full page width
MAX_HEIGHT = 230 * MM

BASE_PT = 8               # axis labels
TICK_PT = 7
SMALL_PT = 6.5            # atom labels on maps; never go below 6 pt at print size
PANEL_PT = 9              # (a), (b) panel letters
DPI = 600

# Element colours: close to the Jmol/CPK convention readers expect, with H and C separated for print.
ELEMENT_COLOURS = {
    'H': '#E8E8E8', 'C': '#4A4A4A', 'N': '#2F5FD0', 'O': '#D62728', 'Cl': '#2CA02C',
    'Zn': '#7D80B0', 'S': '#E6C229', 'Br': '#A52A2A', 'P': '#FF8000',
}
# Covalent radii (Å), Cordero et al. 2008, for bond detection.
COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'Cl': 1.02, 'Zn': 1.22, 'S': 1.05, 'Br': 1.20,
                  'P': 1.07}
# Drawing radii (Å) for ball-and-stick pictures.
BALL_RADII = {'H': 0.22, 'C': 0.34, 'N': 0.34, 'O': 0.34, 'Cl': 0.42, 'Zn': 0.45, 'S': 0.42, 'Br': 0.45,
              'P': 0.42}

# Model names used in every figure and caption (STYLE.md); periodic = plane-wave PBE-D3 cell.
MODEL_LABELS = {'gas': 'Gas phase', 'pcm': 'IEF-PCM (water)', 'xray': 'X-ray', 'periodic': 'Periodic (PBE-D3)'}
MODEL_COLOURS = {'gas': '#E07B00', 'pcm': '#1F5AA6', 'xray': '#333333',     # orange / blue / X-ray black
                 'periodic': '#009E73'}                                     # Okabe-Ito bluish green
# periodic was '#2CA02C' until 28 Sep 2026: it failed a colour-vision check against the gas-phase orange
# (protan ΔE 3.2, OKLab x100); '#009E73' passes against both blue and orange (worst ΔE 8.9). See CHANGES.md.

_applied = False


def apply():
    """Register the Arial-metric font and set rcParams. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    family = 'DejaVu Sans'
    if FONT_DIR.exists():
        for f in FONT_DIR.glob('LiberationSans-*.ttf'):
            font_manager.fontManager.addfont(str(f))
        family = 'Liberation Sans'
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [family, 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': family, 'mathtext.it': f'{family}:italic', 'mathtext.bf': f'{family}:bold',
        'mathtext.cal': family, 'mathtext.sf': family, 'mathtext.tt': 'DejaVu Sans Mono',
        'font.size': BASE_PT, 'axes.labelsize': BASE_PT, 'axes.titlesize': BASE_PT,
        'xtick.labelsize': TICK_PT, 'ytick.labelsize': TICK_PT, 'legend.fontsize': TICK_PT,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.0, 'lines.markersize': 4,
        'legend.frameon': False, 'axes.spines.top': True, 'axes.spines.right': True,
        'savefig.dpi': DPI, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': True,
    })
    _applied = True


def panel_label(ax, letter, x=-0.02, y=1.02, **kw):
    """Bold (a), (b) ... in the upper-left corner outside the axes."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=PANEL_PT, fontweight='bold',
            ha='right', va='bottom', **kw)


def save(fig, path, formats=('png', 'pdf')):
    """Save at 600 dpi PNG plus a vector PDF next to it. `path` is given without extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix('.' + fmt)
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)
    return out
