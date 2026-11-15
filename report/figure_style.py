"""figure_style.py — size and typeset a figure for the medium it lands in.

A project-agnostic starting point, not a finished house style. Copy it into a project, then:

  1. MEASURE your destinations and call ``measure(...)`` — see SKILL.md section 1. Every
     width below is a class default, which is to say a guess about your document.
  2. Freeze the layouts you actually use into ``SIZES`` once they settle, so a figure's
     size is a name rather than a number typed in a notebook.
  3. Put this project's labels and colours HERE, not in the notebooks.

The one rule the whole module exists to enforce: a figure is built at the width it will be
rendered at, so a point in the figure is a point on the page. Everything else follows.

Requires matplotlib. Nothing else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "Medium", "MEDIA", "measure", "poster_medium", "set_style", "active_medium",
    "figure_size", "new_figure", "check_panel_floor", "panel_label", "panel_labels",
    "save_figure", "effective_pt", "min_nominal_pt", "min_pt_for_distance", "report",
    "SIZES", "LABELS_SHORT", "CB_COLORS", "CB_CYCLE",
]

# --------------------------------------------------------------------------------------
# Units.  These two are not the same and conflating them is a silent 0.37 % error.
# --------------------------------------------------------------------------------------
TEX_PT_PER_IN = 72.27  # \the\textwidth prints TeX points
PS_PT_PER_IN = 72.0    # matplotlib font sizes are PostScript points
MM_PER_IN = 25.4


def in_from_texpt(pt: float) -> float:
    """Inches from a TeX dimension, i.e. anything \\the\\textwidth printed."""
    return pt / TEX_PT_PER_IN


def in_from_mm(mm: float) -> float:
    return mm / MM_PER_IN


# --------------------------------------------------------------------------------------
# The panel floor.
#
# Below this a panel cannot carry its own axis furniture and still show a shape: the
# y-label plus its tick labels take ~5 em of width, the x-label plus its ticks ~4 em of
# height, and what remains stops being a plot.
#
# Derived, validated at one point: 27 x 19 em at 8 pt is 3.00 x 2.11 in, which is the
# floor a working plasma manuscript arrived at by hand. The slide and poster rows are this
# model's predictions, not measurements — though they independently reproduce the usual
# "one figure per slide" advice, which is some evidence they are right.
# --------------------------------------------------------------------------------------
PANEL_FLOOR_EM = (27.0, 19.0)

# Font sizes are ratios of the base, rounded to 0.5 pt. At base 8 this yields exactly
# 8 / 9 / 7.5 — the sizes that manuscript settled on independently.
FONT_RATIOS = {
    "base": 1.0, "tick_label": 1.0, "axis_label": 1.125,
    "title": 1.125, "panel_label": 1.125, "legend": 0.94,
}

# Point-valued rcParams do NOT scale with the canvas (SKILL.md trap 3), so derive them
# from the base font instead of inheriting them across media.
RC_RATIOS = {"lines.linewidth": 0.16, "lines.markersize": 0.50, "axes.linewidth": 0.10}

# Okabe-Ito: colour-vision-safe, universal, no project meaning attached.
CB_COLORS = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
CB_CYCLE = [CB_COLORS[k] for k in
            ("blue", "vermillion", "green", "purple", "orange", "sky", "black")]

# Freeze this project's layouts here: {"NAME": (width_in, height_in)}. Ships empty on
# purpose — a generic heuristic cannot reproduce a table you tuned by hand, and pretending
# otherwise would be worse than leaving it to you.
SIZES: dict[str, tuple[float, float]] = {}

# Short axis labels per medium (SKILL.md trap 2): {"medium": {"key": "short text"}}.
# The mechanism is general; the content is yours.
LABELS_SHORT: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class Medium:
    """A destination, and everything that follows from it."""

    name: str
    column_width_in: float   # the narrow placement (equal to full for one-column media)
    full_width_in: float     # the widest a figure may be
    max_height_in: float
    base_pt: float
    frame_factor: float = 1.0  # nominal pt -> the frame this medium's floor is quoted in
    floor_pt: float = 6.0      # lowest legible EFFECTIVE pt, in that frame
    savefig_dpi: int = 300
    measured: bool = False     # True once measure() has replaced the class default
    note: str = ""

    @property
    def em_in(self) -> float:
        return self.base_pt / PS_PT_PER_IN

    @property
    def panel_floor_in(self) -> tuple[float, float]:
        return (PANEL_FLOOR_EM[0] * self.em_in, PANEL_FLOOR_EM[1] * self.em_in)

    @property
    def fonts(self) -> dict[str, float]:
        return {k: round(self.base_pt * r * 2) / 2 for k, r in FONT_RATIOS.items()}

    @property
    def min_nominal_pt(self) -> float:
        """Smallest nominal pt that still clears this medium's floor, at 1:1."""
        return self.floor_pt / self.frame_factor

    def width(self, span: str = "column") -> float:
        if span not in ("column", "full"):
            raise ValueError(f"span must be 'column' or 'full', not {span!r}")
        return self.full_width_in if span == "full" else self.column_width_in


# --------------------------------------------------------------------------------------
# MEASURE ME.  These are class defaults, not your document.  See SKILL.md section 1.
# --------------------------------------------------------------------------------------
MEDIA: dict[str, Medium] = {
    # REVTeX 4.2, 10 pt, two-column reprint. Read from aps10pt4-2.rtx lines 113-116:
    #   \textwidth  42.5pc = 510 pt = 7.0569 in
    #   \columnsep   1.5pc =  18 pt
    #   \columnwidth       = (510 - 18) / 2 = 246 pt = 3.4039 in   [246+18+246 = 510 exactly]
    # APS and AIP two-column reprints both load this file, so it covers PRL/PRE/PRX and
    # PoP/JAP/RSI alike. The 11pt and 12pt options load a different file -- measure.
    "paper": Medium(
        name="paper",
        column_width_in=in_from_texpt(246.0),
        full_width_in=in_from_texpt(510.0),
        max_height_in=8.5,  # \textheight is 9.30 in; leave room for a caption
        base_pt=8, floor_pt=6.0,
        note="REVTeX 4.2 10pt two-column (APS + AIP).",
    ),

    # There is no defensible default here. LaTeX article+a4paper is only 4.77 in, but
    # nearly every thesis loads geometry and lands somewhere else entirely.
    "thesis": Medium(
        name="thesis",
        column_width_in=6.0, full_width_in=6.0, max_height_in=8.0,
        base_pt=10, floor_pt=7.0,
        note="PLACEHOLDER -- measure \\the\\textwidth and call measure('thesis', ...).",
    ),

    # beamer, aspectratio=169, default 1 cm margins on a 160 x 90 mm canvas:
    #   \paperwidth 455.24 pt = 160 mm ; \textwidth 398.34 pt = 140 mm = 5.5118 in
    #   \textheight 227.62 pt = 80 mm -- BEFORE the frame title takes its share.
    # frame_factor 2.1169 = 338.7 mm (PowerPoint 16:9) / 160 mm. It is not a physical
    # magnification; it converts nominal beamer pt into the frame the 18/24/30 pt slide
    # targets are quoted in. Those targets belong to the scientific-talk-slides skill.
    "slide": Medium(
        name="slide",
        column_width_in=in_from_texpt(398.3386),
        full_width_in=in_from_texpt(398.3386),
        max_height_in=2.8,
        base_pt=12, frame_factor=2.1169, floor_pt=18.0, savefig_dpi=200,
        note="beamer aspectratio=169; floor_pt imported from scientific-talk-slides.",
    ),
}


def min_pt_for_distance(distance_m: float) -> float:
    """Minimum comfortable point size for text read from ``distance_m`` metres.

    Derivation, so you can move it:
      * threshold acuity is ~1 arcmin per stroke, ~5 arcmin for a whole letter; nobody
        reads at threshold. Comfortable continuous reading sits well above it -- take cap
        height subtending 15 arcmin (1/229.2 rad). This is the one judgement call, chosen
        from a defensible 5-29 arcmin range.
      * cap height is ~0.70 em in common sans faces (Helvetica .717, Arial .716,
        DejaVu Sans .729) -- convention, but a stable one.

    Gives pt = 0.449 x distance_in, i.e. about 18 pt per metre. It reproduces four
    conventions it was not fitted to: 0.4 m -> 7 pt (journal body), 1.5 m -> 26 pt (poster
    body 24-32), 4 m -> 71 pt (poster title 72-100), and 10 m -> the 24 pt slide body
    target. One model matching both the poster and the slide convention is the reason to
    trust it.
    """
    cap_fraction_of_em = 0.70
    radians_per_cap = 1.0 / 229.2  # 15 arcmin
    distance_in = distance_m * 1000.0 / MM_PER_IN
    return distance_in * radians_per_cap / cap_fraction_of_em * PS_PT_PER_IN


SHEETS_MM = {
    "A0": (841.0, 1189.0), "A1": (594.0, 841.0), "A2": (420.0, 594.0),
    "A3": (297.0, 420.0),
    # Many US meetings specify boards in feet, not A-series.
    "4x8ft": (1219.2, 2438.4), "4x4ft": (1219.2, 1219.2),
}


def poster_medium(sheet: str = "A0", orientation: str = "portrait", ncols: int = 3,
                  margin_mm: float = 25.0, gutter_mm: float = 20.0,
                  viewing_distance_m: float = 1.35, register: bool = True) -> Medium:
    """Build a poster Medium from sheet size, column count and viewing distance.

    ``viewing_distance_m`` defaults to 1.35 m: the distance someone stands to read *a
    figure*, having stepped in, not the distance they read the title from. Margins,
    gutters and the column count are convention -- the real constraint is the board size
    the conference states, which is frequently not A-series at all.
    """
    if sheet not in SHEETS_MM:
        raise ValueError(f"unknown sheet {sheet!r}; choose from {sorted(SHEETS_MM)}")
    short_mm, long_mm = SHEETS_MM[sheet]
    width_mm = short_mm if orientation == "portrait" else long_mm
    height_mm = long_mm if orientation == "portrait" else short_mm

    usable_mm = width_mm - 2 * margin_mm - (ncols - 1) * gutter_mm
    column_mm = usable_mm / ncols
    if column_mm <= 0:
        raise ValueError("margins and gutters leave no room; reduce ncols or the margins")

    floor = min_pt_for_distance(viewing_distance_m)
    base = round(floor * 1.08)  # a little headroom above the floor, rounded to whole pt

    m = Medium(
        name="poster",
        column_width_in=in_from_mm(column_mm),
        full_width_in=in_from_mm(width_mm - 2 * margin_mm),
        max_height_in=in_from_mm(height_mm) * 0.35,
        base_pt=base, floor_pt=round(floor),
        savefig_dpi=150,  # vector at final size; dpi only matters for embedded rasters
        measured=True,
        note=(f"{sheet} {orientation}, {ncols} columns, {margin_mm:g} mm margins, "
              f"{gutter_mm:g} mm gutters, read at {viewing_distance_m:g} m."),
    )
    if register:
        MEDIA["poster"] = m
    return m


# Register a default poster so MEDIA["poster"] exists; override with poster_medium(...).
poster_medium()


def measure(name: str, *, column_pt: float | None = None, full_pt: float | None = None,
            column_in: float | None = None, full_in: float | None = None,
            height_pt: float | None = None, height_in: float | None = None,
            base_pt: float | None = None, register: bool = True) -> Medium:
    """Replace a medium's class default with what you measured out of the LaTeX log.

    ``*_pt`` arguments are TeX points, exactly as ``\\the\\textwidth`` prints them.

        measure("slide", full_pt=398.3386)            # from a \\typeout inside a frame
        measure("thesis", full_pt=455.24, height_pt=650.0)
        measure("paper", column_pt=246.0, full_pt=510.0)

    A one-column medium may be given only ``full_*``; the column width follows it.
    """
    base = MEDIA.get(name)
    if base is None:
        raise KeyError(f"unknown medium {name!r}; known: {sorted(MEDIA)}")

    col = column_in if column_in is not None else (
        in_from_texpt(column_pt) if column_pt is not None else None)
    full = full_in if full_in is not None else (
        in_from_texpt(full_pt) if full_pt is not None else None)
    hgt = height_in if height_in is not None else (
        in_from_texpt(height_pt) if height_pt is not None else None)

    if col is None and full is not None:
        col = full if base.column_width_in == base.full_width_in else base.column_width_in
    if full is None and col is not None:
        full = col if base.column_width_in == base.full_width_in else base.full_width_in

    m = replace(
        base,
        column_width_in=col if col is not None else base.column_width_in,
        full_width_in=full if full is not None else base.full_width_in,
        max_height_in=hgt if hgt is not None else base.max_height_in,
        base_pt=base_pt if base_pt is not None else base.base_pt,
        measured=True,
        note=(base.note.replace("PLACEHOLDER -- ", "") + " [measured]").strip(),
    )
    if register:
        MEDIA[name] = m
    return m


# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------
_ACTIVE: Medium = MEDIA["paper"]


def active_medium() -> Medium:
    """The medium set by the last ``set_style()``.

    Read fonts from here for anything drawn with an explicit size. An ``ax.text(...,
    fontsize=9)`` literal cannot be moved by rcParams, so it survives the medium switch
    and lands at manuscript size on a slide -- which is how a panel label becomes
    invisible rather than merely small.
    """
    return _ACTIVE


def set_style(medium: str | Medium = "paper", **rc_overrides) -> Medium:
    """Apply a medium's rcParams globally. The only entry point that touches rcParams."""
    global _ACTIVE
    m = MEDIA[medium] if isinstance(medium, str) else medium
    if m.name == "thesis" and not m.measured:
        warnings.warn(
            "medium 'thesis' is still a placeholder (6.0 in). Measure \\the\\textwidth in "
            "your thesis class and call measure('thesis', full_pt=...) -- the LaTeX "
            "article+a4 default is 4.77 in, so guessing here is usually wrong.",
            stacklevel=2,
        )
    f = m.fonts
    rc = {
        "font.size": f["base"],
        "axes.labelsize": f["axis_label"],
        "axes.titlesize": f["title"],
        "xtick.labelsize": f["tick_label"],
        "ytick.labelsize": f["tick_label"],
        "legend.fontsize": f["legend"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.prop_cycle": mpl.cycler(color=CB_CYCLE),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": m.savefig_dpi,
        # NOT "tight": it crops to the ink, so the file is not `figsize` wide and the 1:1
        # arithmetic silently stops holding. constrained layout already reserves room for
        # outside legends and colorbars.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    rc.update({k: round(m.base_pt * r, 2) for k, r in RC_RATIOS.items()})
    rc.update(rc_overrides)
    mpl.rcParams.update(rc)
    _ACTIVE = m
    return m


# --------------------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------------------
def effective_pt(nominal_pt: float, figure_width_in: float, rendered_width_in: float,
                 magnification: float = 1.0) -> float:
    """What a nominal point size actually renders as in the destination."""
    return nominal_pt * (rendered_width_in / figure_width_in) * magnification


def min_nominal_pt(floor_pt: float, figure_width_in: float, rendered_width_in: float,
                   magnification: float = 1.0) -> float:
    """The smallest nominal size that still clears ``floor_pt`` in the destination."""
    return floor_pt * figure_width_in / (rendered_width_in * magnification)


def check_panel_floor(figsize: tuple[float, float], nrows: int, ncols: int,
                      medium: Medium | None = None) -> list[str]:
    """Return a list of human-readable panel-floor violations (empty if the layout fits)."""
    m = medium or _ACTIVE
    fw, fh = figsize
    floor_w, floor_h = m.panel_floor_in
    panel_w, panel_h = fw / ncols, fh / nrows
    out = []
    if panel_w < floor_w:
        out.append(f"panel width {panel_w:.2f} in < {floor_w:.2f} in floor "
                   f"({m.name}, {m.base_pt:g} pt)")
    if panel_h < floor_h:
        out.append(f"panel height {panel_h:.2f} in < {floor_h:.2f} in floor "
                   f"({m.name}, {m.base_pt:g} pt)")
    return out


def figure_size(nrows: int = 1, ncols: int = 1, span: str = "column",
                aspect: float = 0.75, medium: Medium | None = None) -> tuple[float, float]:
    """A starting size for a layout. ``aspect`` is panel height / panel width.

    A heuristic, deliberately: it cannot reproduce a table you tuned by hand. Use it to
    find a size, then freeze the good ones in ``SIZES``. The floor check, not this
    function, is the authority on whether a layout is viable.
    """
    m = medium or _ACTIVE
    width = m.width(span)
    height = min((width / ncols) * aspect * nrows, m.max_height_in)
    return (round(width, 4), round(height, 4))


def new_figure(nrows: int = 1, ncols: int = 1, size=None, span: str = "column",
               aspect: float = 0.75, allow_small: bool = False, **subplot_kw):
    """``plt.subplots`` with the medium's geometry and a panel-floor check.

    ``size`` may be a key in ``SIZES``, an explicit ``(w, h)``, or None to derive one.
    """
    m = _ACTIVE
    if size is None:
        figsize = figure_size(nrows, ncols, span, aspect, m)
    elif isinstance(size, str):
        if size not in SIZES:
            raise KeyError(f"unknown size key {size!r}; SIZES has {sorted(SIZES)}")
        figsize = SIZES[size]
    else:
        figsize = tuple(size)

    if figsize[0] > m.full_width_in + 1e-6:
        warnings.warn(
            f"figure is {figsize[0]:.2f} in wide but {m.name} is at most "
            f"{m.full_width_in:.2f} in. It will be scaled down on the way in, shrinking "
            f"every label by {m.full_width_in / figsize[0]:.2f}x.",
            stacklevel=2,
        )
    if not allow_small:
        for msg in check_panel_floor(figsize, nrows, ncols, m):
            warnings.warn(
                f"{msg}. Cut content, re-shape (a {nrows}x{ncols} stack buys panel width "
                f"over a row), promote to span='full', or split the figure. Lowering the "
                f"font is not on the list. Pass allow_small=True only with a written "
                f"reason next to the call.",
                stacklevel=2,
            )
    subplot_kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=figsize, **subplot_kw)


# --------------------------------------------------------------------------------------
# Drawing helpers that must not escape the medium switch
# --------------------------------------------------------------------------------------
def panel_label(ax, text: str, x: float = -0.02, y: float = 1.02, **kw):
    """Panel letter, at the ACTIVE medium's size rather than a literal."""
    kw.setdefault("fontsize", _ACTIVE.fonts["panel_label"])
    kw.setdefault("fontweight", "bold")
    kw.setdefault("va", "bottom")
    kw.setdefault("ha", "right")
    return ax.text(x, y, text, transform=ax.transAxes, **kw)


def panel_labels(axes, start: str = "a", fmt: str = "({})", **kw):
    flat = getattr(axes, "flat", axes)
    return [panel_label(ax, fmt.format(chr(ord(start) + i)), **kw)
            for i, ax in enumerate(flat)]


def short_label(key: str, default: str = "", medium: Medium | None = None) -> str:
    """This medium's short form of a label, falling back to ``default``.

    Narrow panels need the label *text* to get shorter, not just the type to get smaller;
    a 22-character axis label collides with its neighbour at any legible size.
    """
    m = medium or _ACTIVE
    return LABELS_SHORT.get(m.name, {}).get(key, default or key)


def save_figure(fig, out_dir, stem: str, formats=("pdf", "png"),
                dpi: int | None = None, close: bool = True) -> list[Path]:
    """Write vector + raster, with metadata stripped so unchanged reruns are identical."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = dpi or _ACTIVE.savefig_dpi
    written = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        meta = {"CreationDate": None} if ext == "pdf" else (
            {"Software": None} if ext == "png" else {})
        try:
            fig.savefig(path, dpi=dpi, metadata=meta)
        except (TypeError, ValueError):
            fig.savefig(path, dpi=dpi)
        written.append(path)
    if close:
        plt.close(fig)
    return written


def report(medium: str | Medium | None = None) -> str:
    """One-screen summary of a medium: what it is, and what follows from it."""
    m = MEDIA[medium] if isinstance(medium, str) else (medium or _ACTIVE)
    fw, fh = m.panel_floor_in
    f = m.fonts
    across = max(1, int(m.full_width_in // fw))
    lines = [
        f"{m.name}{'' if m.measured else '  [CLASS DEFAULT -- measure it]'}",
        f"  {m.note}" if m.note else "",
        f"  column {m.column_width_in:.3f} in    full {m.full_width_in:.3f} in"
        f"    max height {m.max_height_in:.2f} in",
        f"  base {m.base_pt:g} pt   axis {f['axis_label']:g}   legend {f['legend']:g}"
        f"   panel label {f['panel_label']:g}",
        f"  floor {m.floor_pt:g} pt effective; frame factor {m.frame_factor:g}"
        f"  ->  nominal must be >= {m.min_nominal_pt:.1f} pt at 1:1",
        f"  panel floor {fw:.2f} x {fh:.2f} in  ->  at most {across} panel(s) across the"
        f" full width",
    ]
    return "\n".join(ln for ln in lines if ln)


if __name__ == "__main__":
    for name in ("paper", "thesis", "slide", "poster"):
        print(report(name))
        print()
