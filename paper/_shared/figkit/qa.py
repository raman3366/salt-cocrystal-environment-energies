"""Automatic figure checks: run before and after saving every figure.

    check_figure(fig, min_pt=6.0, strict=False, name='')  -> list[str]
        text–text overlaps (from the bounding boxes of the text actually rendered), text outside the canvas,
        text clipped by its clip box, font size below `min_pt` (points at print size), plus any checks registered
        on the figure with register_check (draw2d uses this to verify that atom labels clear atoms and bonds).
    check_png(path, min_dpi=600, max_border_mm=2.5, max_width_mm=None, strict=False) -> list[str]
        dpi metadata, excessive white border, printed width.
    save(fig, path, strict=True, max_width_mm=None, formats=('png', 'pdf')) -> list[Path]
        check_figure -> style.save -> check_png. Raises FigureQAError in strict mode, otherwise warns loudly.
    register_check(fig, fn, name)      fn(fig, renderer) -> list[str]
    rendered_texts(fig)                -> list[(Text, Bbox)]   texts that were really drawn, with display bboxes
    missing_glyphs(fig)                -> list[str]  characters the text's font has no glyph for (they print as boxes,
                                          e.g. U+207B '⁻' in Liberation Sans); opt-in via check_figure/save(glyphs=True)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.text as mtext
import matplotlib.pyplot as plt

from . import style


class FigureQAError(RuntimeError):
    pass


class FigureQAWarning(UserWarning):
    pass


def register_check(fig, fn, name='custom'):
    """Attach an extra check to a figure; it runs inside check_figure after the figure is drawn."""
    if not hasattr(fig, '_figkit_checks'):
        fig._figkit_checks = []
    fig._figkit_checks.append((name, fn))


def rendered_texts(fig):
    """Draw the figure and return every Text that was actually rendered (visible, non-empty) with its display bbox.

    Implemented by wrapping Text.draw during one canvas draw, so tick labels outside the view, empty titles and
    hidden artists are excluded automatically."""
    rec = {}
    orig = mtext.Text.draw

    def wrapped(self, renderer):
        out = orig(self, renderer)
        try:
            s = self.get_text()
            a = self.get_alpha()
            if self.get_visible() and s and s.strip() and (a is None or a > 0):
                bb = self.get_window_extent(renderer)
                if np.all(np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1])):
                    rec[id(self)] = (self, bb)
        except Exception:                                        # pragma: no cover - never break a draw
            pass
        return out

    mtext.Text.draw = wrapped
    try:
        fig.canvas.draw()
    finally:
        mtext.Text.draw = orig
    return list(rec.values())


def _overlap(a, b, tol):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w > tol and h > tol, max(w, 0) * max(h, 0)


def _short(t):
    s = t.get_text().replace('\n', ' ')
    return s if len(s) <= 28 else s[:25] + '...'


def check_figure(fig, min_pt=6.0, strict=False, name='', tol_pt=0.3, quiet=False, glyphs=False):
    """Check a figure before saving. Returns the list of issues (empty when clean).

    glyphs  also report characters missing from the font of their text (missing_glyphs); off by default"""
    texts = rendered_texts(fig)
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    tol = tol_pt * px_per_pt
    issues = []
    fb = fig.bbox
    for t, bb in texts:
        fs = t.get_fontsize()
        if fs < min_pt - 1e-6:
            issues.append(f'font {fs:.2f} pt < {min_pt} pt: "{_short(t)}"')
        if bb.x0 < fb.x0 - tol or bb.y0 < fb.y0 - tol or bb.x1 > fb.x1 + tol or bb.y1 > fb.y1 + tol:
            issues.append(f'text outside the canvas: "{_short(t)}"')
        if t.get_clip_on():
            cb = t.get_clip_box()
            if cb is not None and (bb.x0 < cb.x0 - tol or bb.x1 > cb.x1 + tol or bb.y0 < cb.y0 - tol or bb.y1 > cb.y1 + tol):
                issues.append(f'text clipped by its clip box: "{_short(t)}"')
    # savefig(bbox_inches='tight') keeps only artists that take part in the layout: anything outside that box
    # would be cut from the saved file
    if str(plt.rcParams.get('savefig.bbox')) == 'tight':
        from matplotlib.transforms import Bbox
        tb = fig.get_tightbbox(renderer)
        pad = float(plt.rcParams.get('savefig.pad_inches', 0.0))
        tb = Bbox.from_extents(tb.x0 - pad, tb.y0 - pad, tb.x1 + pad, tb.y1 + pad).transformed(fig.dpi_scale_trans)
        for t, bb in texts:
            if bb.x0 < tb.x0 - tol or bb.y0 < tb.y0 - tol or bb.x1 > tb.x1 + tol or bb.y1 > tb.y1 + tol:
                issues.append(f'text would be cut by the tight bounding box: "{_short(t)}"')
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            ok, area = _overlap(texts[i][1], texts[j][1], tol)
            if ok:
                issues.append(f'text overlap: "{_short(texts[i][0])}" / "{_short(texts[j][0])}"')
    issues += legend_data_overlaps(fig, renderer)
    if glyphs:
        issues += missing_glyphs(fig, texts)
    for cname, fn in getattr(fig, '_figkit_checks', []):
        try:
            issues += [f'[{cname}] {m}' for m in fn(fig, renderer)]
        except Exception as e:                                   # a broken check is itself an issue
            issues.append(f'[{cname}] check failed: {e!r}')
    _report(issues, f'figure {name}'.strip(), strict, quiet)
    return issues


def legend_data_overlaps(fig, renderer):
    """Legends drawn over data: lines/markers (Line2D), scatter points and bar/step outlines of the same axes."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Rectangle, StepPatch
    from .labels import _seg_box_hit
    issues = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        lb = leg.get_window_extent(renderer)
        box = np.array([[lb.x0, lb.y0, lb.x1, lb.y1]])
        ab = ax.bbox
        hit = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xy = ln.get_xydata()
            if len(xy) == 0:
                continue
            P = ln.get_transform().transform(xy)
            P = P[np.all(np.isfinite(P), axis=1)]
            if len(P) == 0:
                continue
            ms = (ln.get_markersize() if ln.get_marker() not in (None, 'None', '', ' ') else 0) * fig.dpi / 72 / 2
            inside = (P[:, 0] > lb.x0 - ms) & (P[:, 0] < lb.x1 + ms) & (P[:, 1] > lb.y0 - ms) & (P[:, 1] < lb.y1 + ms)
            seg_hit = False
            if ln.get_linestyle() not in ('None', '', ' ') and len(P) > 1:
                S = np.c_[P[:-1], P[1:], np.full(len(P) - 1, ln.get_linewidth() * fig.dpi / 72 / 2)]
                seg_hit = _seg_box_hit(box, S).any()
            if inside.any() or seg_hit:
                hit.append(ln.get_label() if not ln.get_label().startswith('_') else 'a line')
        for col in ax.collections:
            if isinstance(col, PathCollection) and col.get_visible():
                P = col.get_offset_transform().transform(col.get_offsets())
                if len(P) and ((P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)).any():
                    hit.append('scatter points')
        for pa in ax.patches:
            if isinstance(pa, (Rectangle, StepPatch)) and pa.get_visible() and pa.get_alpha() in (None, 1, 1.0):
                V = pa.get_transform().transform(pa.get_path().vertices)
                if len(V) > 1:
                    S = np.c_[V[:-1], V[1:], np.zeros(len(V) - 1)]
                    if _seg_box_hit(box, S).any():
                        hit.append('a bar/step outline')
        if hit:
            issues.append(f'legend overlaps data ({", ".join(sorted(set(hit))[:3])})')
    return issues


def _strip_math(s):
    """Text outside $...$ mathtext segments (mathtext uses its own glyph lookup)."""
    parts = s.replace(r'\$', '\x00').split('$')
    if len(parts) % 2 == 0:                                  # odd number of $: not mathtext
        return s
    return ''.join(parts[0::2]).replace('\x00', '$')


def missing_glyphs(fig, texts=None):
    """Characters of rendered texts that the text's own font (font_manager.findfont of its FontProperties) has no
    glyph for. Such characters print as empty boxes; e.g. Liberation Sans has no U+207A/U+207B (superscript plus
    and minus), so write mol$^{-1}$ instead of mol⁻¹. Mathtext segments are skipped."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    texts = rendered_texts(fig) if texts is None else texts
    cmaps = {}
    issues = []
    for t, _ in texts:
        s = _strip_math(t.get_text())
        if not s.strip():
            continue
        path = font_manager.findfont(t.get_fontproperties())
        if path not in cmaps:
            try:
                cmaps[path] = set(FT2Font(path).get_charmap())
            except Exception:                                # pragma: no cover - unreadable font: skip the check
                cmaps[path] = None
        cm = cmaps[path]
        if cm is None:
            continue
        bad = sorted({ch for ch in s if not ch.isspace() and ord(ch) not in cm})
        if bad:
            codes = ', '.join(f"U+{ord(ch):04X} '{ch}'" for ch in bad)
            issues.append(f'missing glyph ({codes}) in {Path(path).name}: "{_short(t)}"')
    return issues


def check_png(path, min_dpi=600, max_border_mm=2.5, max_width_mm=None, strict=False, quiet=False):
    """Check a saved PNG: dpi metadata, white border, printed width."""
    from PIL import Image
    path = Path(path)
    issues = []
    with Image.open(path) as im:
        dpi = im.info.get('dpi', (0, 0))
        dpi = float(dpi[0]) if isinstance(dpi, (tuple, list)) else float(dpi)
        if dpi < min_dpi - 0.5:
            issues.append(f'dpi {dpi:.0f} < {min_dpi}')
        a = np.asarray(im.convert('RGBA'))
    rgb = a[..., :3].astype(np.int16)
    alpha = a[..., 3]
    ink = (alpha > 8) & (rgb.min(-1) < 245)
    H, W = ink.shape
    eff = dpi if dpi > 0 else style.DPI
    px_per_mm = eff / 25.4
    if not ink.any():
        issues.append('image is blank')
    else:
        rows = np.nonzero(ink.any(1))[0]
        cols = np.nonzero(ink.any(0))[0]
        borders = {'top': rows[0], 'bottom': H - 1 - rows[-1], 'left': cols[0], 'right': W - 1 - cols[-1]}
        for side, b in borders.items():
            if b / px_per_mm > max_border_mm:
                issues.append(f'white border {side} {b / px_per_mm:.1f} mm > {max_border_mm} mm')
    width_mm = W / px_per_mm
    if max_width_mm is not None and width_mm > max_width_mm + 1.0:
        issues.append(f'printed width {width_mm:.1f} mm > {max_width_mm:.1f} mm')
    _report(issues, f'png {path.name}', strict, quiet)
    return issues


def save(fig, path, strict=True, max_width_mm=None, formats=('png', 'pdf'), min_pt=6.0, max_border_mm=2.5,
         glyphs=False):
    """QA-checked save: check_figure -> style.save (600 dpi PNG + vector PDF) -> check_png.
    glyphs=True adds the missing-glyph check (see missing_glyphs)."""
    name = Path(path).name
    issues = check_figure(fig, min_pt=min_pt, strict=strict, name=name, glyphs=glyphs)
    out = style.save(fig, path, formats=formats)
    for p in out:
        if p.suffix == '.png':
            issues += check_png(p, max_width_mm=max_width_mm, strict=strict, max_border_mm=max_border_mm)
    return out


def _report(issues, what, strict, quiet=False):
    if not issues:
        return
    msg = f'FIGURE QA: {len(issues)} issue(s) in {what}:\n  - ' + '\n  - '.join(issues)
    if strict:
        raise FigureQAError(msg)
    if not quiet:
        print('\n' + '!' * 78 + '\n' + msg + '\n' + '!' * 78, file=sys.stderr)
        warnings.warn(msg, FigureQAWarning, stacklevel=3)
