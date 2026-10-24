# figkit changes

## 2026-09-28 · imported into the SAPH project
- Copied from cqc-dft `papers/_shared/figkit` (library modules only; `demo_2d.py`/`demo_3d.py` and their
  outputs were left behind because they draw the sister compounds). Changes: `style.DATA` points at the VM
  pulls (`/root/cqc-sfa-archive/vm/cqc-sfa`, override with `SAPH_RAW_DATA`); phosphorus added to the
  element colour (Jmol orange), covalent-radius (1.07 Å, Cordero 2008) and ball-radius tables, without
  which P–O bonds were not detected (default radius 0.8 Å); model labels/colours for X-ray and the periodic
  PBE-D3 model. Everything else unchanged.

Backwards-compatible additions made while building the paper figures (newest last). Existing signatures and
defaults are unchanged.

## 2026-09-23 · btpc structure/vibration figures
- `draw2d.draw_molecule`: new optional keyword arguments `hbond_fmt` (format of the H···A distance labels,
  default `'{:.2f} Å'` as before), `hbond_colours` (dict `(d, h, a) -> colour` for individual hydrogen bonds),
  `hbond_lw` (default 0.9 as before), `extra_segments` / `extra_discs` (further label obstacles in data
  coordinates, e.g. normal-mode arrows drawn on the molecule) and `label_kw` (extra text properties for atom
  labels). Defaults reproduce the previous output exactly.

## 2026-09-23 · 3atb structure/crystal/vibration figures
- `draw2d.draw_overlay`: `bonds` and `hbonds` also accept a dict `name -> list` so that each overlaid geometry
  can have its own covalent bonds and hydrogen bonds (needed when a proton has moved between the geometries, e.g.
  X-ray N4–H4···O11 against gas-phase O11–H4···N4); geometries missing from a `bonds` dict get their own detected
  bonds. New keyword `hbond_obstacles=False`: when True the dashed hydrogen bonds are label obstacles too.
  A list (or None) behaves exactly as before.

## 2026-09-23 · btp structure/vibration figures
- `draw2d.displacement_arrows(ax, xy, vec, atoms=None, scale=1.0, min_len=0.0, colour='black', lw=0.7,
  head_pt=3.0, halo=True, ...)`: new function. Arrows from atom positions along projected normal-mode
  displacements (white halo so they stay visible over atoms); returns label-obstacle segments for
  `draw_molecule(extra_segments=...)`, and with `draw=False` only reserves them. No existing function changed.
- New module `rdkit2mpl.py` (nothing existing changed): `Depiction(mol, bond_mm, font_pt, line_pt, ...)` draws an
  RDKit 2D depiction with MolDraw2DSVG in Liberation Sans and re-draws the SVG paths as matplotlib patches in a
  millimetre axes (vector output, house font), with atom positions, label-glyph boxes and bond segments returned
  for hydrogen-bond lines and `LabelPlacer` obstacles; negative charges are redrawn as a true minus sign
  (`true_minus=True`), per-atom recolouring (`recolour={atom: colour}`) for e.g. a transferred proton. Used by the
  BTP scheme.
- `plots.parity_plot`: optional `marker_kw` (dict model -> Line2D kwargs, e.g. open squares for one model) and
  `rmsd_fmt` (legend RMSD format, default `'{:.3f}'` as before); `plots.deviation_dots`: optional `marker_kw`.
  Defaults reproduce the previous output. (Added by the earlier BTP structure run; recorded here.)

## 2026-09-23 · pzc structure/crystal/vibration figures
- `draw2d.draw_molecule`: `hbond_distances` also accepts an iterable of (d, h, a) triples, so that only those
  hydrogen bonds get an H···A distance label (e.g. weak C–H···Cl contacts drawn but not labelled); True / False
  behave as before. New optional keyword `atom_colours` (dict index -> colour, or a sequence; None entries keep
  the element colour) overrides the ball and bond-half colours per atom, e.g. to draw a crystal environment
  lightened around the model in one depth-sorted call. Defaults reproduce the previous output exactly.

## 2026-09-23 · btp 3D renders (fig_fmo, fig_nci, fig_mep, SI renders)
- `render3d.shared_hbond_view(mols, ref=None, atoms=None, tilts=range(25, 41, 5), phis=range(0, 360, 15), ...)`
  (new function, exported in `__all__`): one camera for several models of the same cluster, chosen so that every
  hydrogen-bond NCI disc of every model is seen at least ~25° from edge-on and not hidden behind atoms, with no
  ring seen more edge-on than a limit; returns ({name: View}, info). The views of the other models are
  `matched_view` copies of the reference view. Nothing else in render3d changed.
- `rdg.plot_rdg`: new optional keyword `spike_prefer=(0.3, -1)` (preferred direction of spike labels; the
  default is the previous hard-wired value) and an optional per-spike `'prefer'` key in the spike dicts.
  Defaults reproduce the previous output exactly.
- `callouts.place`: two optional per-item keys, `'prefer_weight'` (weight of the preferred direction in the
  placement cost, default 0.6 as before) and `'min_leader_mm'` (per-item minimum leader length, default the
  call's `min_leader_mm`). Items without these keys are placed exactly as before.

## 2026-09-23 · btpc scheme / crystal / trajectory / vibration figures (second agent)
- New module `figkit/scheme.py` (no existing code changed): 2D chemical-structure schemes from RDKit drawn as
  vector matplotlib, so `qa.save` checks every symbol. `Structure.from_smiles(smiles, charge=...)` sanitises,
  asserts the total formal charge, adds explicit H on heteroatoms, computes CoordGen coordinates (bond length 1),
  kekulises and condenses nitro groups to NO2 / O2N (so only ionic charges are drawn); atom-map numbers tag atoms
  (`s.idx(map)`); `orient`, `mirror_to`, `reflect_atom`, `rotate_atom` set the orientation. `mm_axes(fig)` gives
  one axes in millimetres; `draw_structure(ax, s, origin, bond, fontsize, atom_colours=..., atom_weights=...)`
  draws skeletal bonds (ring double bonds inside the ring, shortened), heteroatom symbols with bonds clipped at
  the glyphs, and formal charges in the least crowded raised position; `add_obstacles(placer, drawn)` declares
  everything drawn to a `LabelPlacer` (e.g. for CIF labels). Used by `papers/btpc/figures/scripts/fig_scheme.py`.
## 2026-09-23 · btpc 3D renders (fig_fmo, fig_nci, fig_mep, SI renders)
- New module `figkit/callouts.py` (added in the first btpc 3D-render session, recorded here now; nothing in
  render3d changed): `expand_canvas(render, left, top, right, bottom)` (white room round a render, projection
  kept valid), `content_mask(image)`, `patch_target(render, points)` (visible vertex of a surface patch nearest
  its projected centre) and `place(ax, items, ...)`, which puts labels on the white background AROUND a render
  with short leaders (never on the picture) and registers a save-time qa check.
- `compose.grid`: new optional keyword `extra_bottom_mm=0.0` (an empty band under the grid for ordinary
  matplotlib panels; its rectangle is `fig._figkit_grid['bottom_band']`) (same first session), and
  `colourbar_linear=False` (passed to `colourbar`); the grid's Colorbar is stored in
  `fig._figkit_grid['colourbar']`.
- `compose.colourbar`: new optional keyword `linear=False`. True draws the long axis on a linear value scale
  for a non-linear norm (the two-slope MEP norm with green at 0 and an asymmetric range, e.g. −55…+110 kcal
  mol⁻¹), so the ticks are evenly spaced. Defaults reproduce the previous output exactly.

## 2026-09-23 · 3atb 3D renders (FMO / MEP / NCI)
- `compose.label_atoms`: new optional keyword `placer_kw` (dict of extra `labels.LabelPlacer` arguments, e.g.
  `n_rings=6` so that a label in a crowded spot may search further out, with a leader). Default `None` builds the
  placer exactly as before.
## 2026-09-23 · pzc 3D-render figures (FMO / MEP / NCI)
- `rdg.plot_rdg`: new optional keyword `number_rings` (default 5 = previous behaviour): number of candidate
  rings tried for the numbered spike labels (`spike_style='numbered'`), so crowded panels can use longer leaders.
- `rdg.plot_rdg`: a spike dict may carry `'mark_only': True` (optional `'marker'`, default open square `'s'`,
  `'ms'`, `'mec'`): the feature gets a marker but no label and counts as an obstacle for the labels of the
  other spikes. Spike dicts without the key behave exactly as before.
- `qa.missing_glyphs(fig)` (new): reports characters of rendered texts that the text's own font has no glyph for
  (they print as empty boxes, e.g. U+207B '⁻' in Liberation Sans: write `mol$^{-1}$`, not `mol⁻¹`); mathtext
  segments are skipped. Opt-in through the new keyword `glyphs=False` of `qa.check_figure` and `qa.save`; the
  default leaves both functions exactly as before.

## 2026-09-23 · btp 2D analysis figures (ELF/LOL, levels/PDOS, gas vs PCM)
- `maps.map_geometry`: two optional keywords. `centre=None` (default: mean of the in-plane heavy atoms, as
  before) also accepts graph coordinates (u, v) or a Cartesian point (x, y, z), so the maps of two geometries
  (gas phase, IEF-PCM) whose in-plane atom sets differ keep the defining ring at the same place; `mirror=False`
  (True reflects the graph y axis before the rotation, i.e. the plane seen from the other side, to match the
  handedness of a 3D camera). The returned dict has a new key `'mirror'`. Defaults reproduce the previous output
  exactly (the rotation matrix is multiplied by the identity).

## 2026-09-23 · btpc 2D electronic-structure figures (levels/PDOS, descriptors, ELF/LOL)
- `maps.plot_plane_map`: new optional keyword `label_atoms=None` (iterable of atom indices). When given, only these
  atoms get a label (e.g. the in-plane heavy atoms plus the two N–H hydrogens); every drawn atom still counts as a
  label obstacle. `None` labels every drawn atom subject to `label_h`, exactly as before.

## 2026-09-23 · pzc electronic-structure plots (levels / DOS / descriptors, ELF / LOL, RDG)
- `plots.mo_levels`: new optional keyword `label_pos='level_end'` (default = previous behaviour). `'arrow'` puts
  the HOMO/LUMO labels left-aligned just right of the gap arrow (like the gap value), for narrow panels where the
  right-aligned labels ran into the arrow.

## 2026-09-23 · 3atb 2D analysis figures (levels/DOS, RDG, ELF/LOL, gas vs PCM)
- `maps.plot_plane_map`: new optional keyword `label_prefer=None` (dict atom index -> (dx, dy), the preferred label
  direction in map coordinates for those atoms, e.g. away from a ring centre so that ring-atom labels sit outside
  the ring instead of between two ring atoms). Atoms not in the dict, and every call without the keyword, keep the
  previous automatic direction (away from the bonded in-plane neighbours); default output unchanged.

## 2026-09-24 · pzc QA round 1 (figures)
- `maps.plot_plane_map`: new optional keyword `line_smooth=0.0` (Gaussian sigma in grid points applied to the field
  for the black contour lines only; the colour fill is unchanged). Removes pixel-size contour islands where a line
  level runs along a plateau (gas-phase LOL maps of the PZC paper). The default 0 reproduces the previous output
  exactly.
- `maps.plot_plane_map`: new optional keyword `line_min_extent=0.0` (map units, Å): contour-line pieces (sub-paths)
  smaller than this, and thin slivers (enclosed area < 0.15 × extent²) smaller than three times this, are not drawn
  (broken dashes of a level that runs along a plateau, e.g. round the Cl cores of the LOL maps; small round core
  contours are kept). The default 0 keeps every piece, exactly as before.

## 2026-09-24 · callouts.place keywords (recorded afterwards; btp QA round 2, minor 17)
- `callouts.place`: three new optional keywords, added on 2026-09-24 at 00:01 without a CHANGES entry (they are used
  by `papers/btpc/figures/scripts/fig_nci.py`; recorded here by the btp QA-round-2 fixer from a diff against the
  previous file): `avoid_boxes=None` (extra rectangles x0, y0, x1, y1 in image px that no label may overlap and no
  leader may cross), `avoid_segments=None` ((N, 4) segments in image px that no leader may cross, e.g. projected
  bonds) and `patch_extent=False` (measure a label by its bbox patch instead of its glyphs). The defaults reproduce the
  previous behaviour: the btp `fig_nci` renders and `figS8_nci_views`, which place labels through `callouts.place`
  without the new keywords, re-rendered fresh are pixel-identical to their renders from before the change.

## 2026-09-24 · btp QA round 2
- `render3d.View.flipped`: docstring only. It said "mirror-image view"; the method reverses the viewing direction and
  keeps screen-up, i.e. turns the camera by 180° about the screen vertical, which is a rotation (left and right
  exchanged), not a reflection. No code changed.
- Note for the other papers (no code change): figkit has two blue–green–red NCI maps, `rdg.nci_cmap()` (#0000FF →
  #00D000 → #FF0000, the default of `rdg.plot_rdg`) and `render3d.NCI_CMAP` (#1E3CD2 → #23B43A → #D7261E, isosurfaces
  and `compose.colourbar`). A figure that shows a render colour bar as the key of an RDG scatter must pass
  `cmap=render3d.NCI_CMAP` to `plot_rdg` (btp Fig. 7 did not; 3atb, btpc and pzc already do).

## 2026-09-24 · btpc figure QA round 2 (fig_nci)
- `callouts.place`: three optional keywords. `avoid_boxes=None` (extra rectangles in image px that no label may
  overlap and no leader may pass through, e.g. atom labels grown by a clearance), `avoid_segments=None` ((N, 4)
  segments in image px that no leader may cross, e.g. the projected bonds, so that a leader cannot read as a bond)
  and `patch_extent=False` (True measures a label by its bbox patch, e.g. a circled number, instead of its
  glyphs). The defaults reproduce the previous behaviour exactly.

## 2026-09-24 · 3atb QA round 3 (fig_rdg)
- `rdg.plot_rdg`: new optional keyword `obstacle_vlines=()` (x values of vertical guide lines that the caller draws in
  the plot, e.g. the region bounds): the spike labels and numbers keep clear of them as they do of the zero line
  (3atb Fig. 10b: the number '4' sat on the +0.01 a.u. guide and its white halo cut it). The default adds no
  obstacle, so every existing call gives the same output as before.
- Recorded afterwards (no code change by 3atb): `draw2d.draw_molecule` gained the optional keyword
  `text_obstacles=()` on 2026-09-24 at 03:01 without a CHANGES entry (used by `papers/pzc/figures/scripts/
  fig_structures.py` for its distance labels): Text artists already on the axes that no atom label may overlap. The
  default () adds no obstacle; the 3atb figures drawn with draw2d do not pass it.

## 2026-09-24 · pzc figure QA round 2
- `draw2d.draw_molecule`: new optional keyword `text_obstacles=()` (Text artists already on the axes that no atom
  label may overlap; passed to `LabelPlacer.add_text_obstacle`). The PZC Fig. 3 places its H···A distance labels
  itself, beside the middle of their own dashed segment, before the atom labels. The default (empty) adds nothing,
  so every existing call draws exactly as before.

## 2026-09-28 · SAPH figure group A (structure and proton story)
- `style.MODEL_COLOURS['periodic']`: '#2CA02C' -> '#009E73' (Okabe-Ito bluish green). The old green and the
  gas-phase orange '#E07B00' are nearly indistinguishable for protanopes (simulated OKLab ΔE ×100 = 3.2, validator
  floor 8); the new value gives ΔE 8.9 against the orange and 24 against the IEF-PCM blue under every simulated
  deficiency. Only the periodic model uses this key. Nothing else changed.
