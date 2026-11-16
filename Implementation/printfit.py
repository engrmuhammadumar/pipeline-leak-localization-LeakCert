r"""
printfit.py - one IEEE house style for every figure of the paper, applied at save time.

For each figure listed in PRINT_WIDTH it
  1. resizes the figure to its printed width in the paper (IEEE: column 3.5 in, text 7.16 in),
  2. sets one font for everything (Times New Roman, STIX for the mathematics, as in the IEEE text):
       panel titles 8.5 pt bold, axis labels 8.5 pt bold, tick labels 7.5 pt, legends 7 pt bold, other text 6.5-8 pt,
  3. scales lines and markers so that signals stay clear in print (0.8-1.8 pt lines),
  4. moves every legend to the place where it covers the fewest data points (white, opaque frame); a legend that
     still covers data is moved above its panel,
  5. replaces wording that does not match the manuscript,
  6. re-runs the layout so that nothing overlaps or is clipped.
Figures drawn already at print size by exp54 (fig_83x) only get steps 2, 5 and 6.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

TW, CW = 7.16, 3.5                                     # IEEE text width / column width [in]
FONT = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]

# printed width of each figure, as used in paper\main.tex
PRINT_WIDTH = {
    "fig_704_opening_zoom": TW,
    "fig_708_coherence": TW,
    "fig_800_uncertainty_budget": TW,
    "fig_804_robustness": TW,
    "fig_813_design_chart": TW,
}
NATIVE = {"fig_830_two_principles", "fig_831_calibration", "fig_832_evidence_examples", "fig_820_error_histogram",
          "fig_840_multipath", "fig_841_methods", "fig_842_control_chart",
          "fig_700_setup_testbed"}   # drawn at print size
MAX_HEIGHT = {"fig_708_coherence": 4.3, "fig_804_robustness": 4.5, "fig_704_opening_zoom": 2.9,
              "fig_700_setup_testbed": 2.6, "fig_800_uncertainty_budget": 2.7, "fig_813_design_chart": 2.5,
              "fig_820_error_histogram": 2.5}

WORDING = [
    ("two branches", "two principles"), ("branches", "principles"), ("branch", "principle"),
    ("Coherence alone", "Principle A alone"), ("coherence alone", "Principle A alone"),
    ("Wavefront alone", "Principle B alone"), ("wavefront alone", "Principle B alone"),
    ("Sound-speed", "Propagation-speed"), ("sound-speed", "propagation-speed"),
    ("Sound speed", "Propagation speed"), ("sound speed", "propagation speed"),
    ("(open end)", "(pipe end)"), ("open end", "pipe end"), ("without P", "no P"),
]
SIZE = {"title": 8.5, "label": 8.5, "tick": 7.5, "legend": 7.0}


def use_font():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "mathtext.fontset": "dejavusans",
                         "axes.titleweight": "bold", "axes.labelweight": "bold", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "legend.framealpha": 0.92, "legend.edgecolor": "0.35"})


use_font()


def _legend_points(ax):
    """display coordinates of the data drawn in an axes (line vertices, scatter offsets, bar corners)"""
    pts = []
    for ln in ax.get_lines():
        if ln.get_label().startswith("_") and len(ln.get_xdata()) < 3:
            continue
        xy = np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float)
        if len(xy) > 400:
            xy = xy[np.linspace(0, len(xy) - 1, 400).astype(int)]
        pts.append(ln.get_transform().transform(xy))
    for c in ax.collections:
        off = c.get_offsets()
        if len(off) and not isinstance(c, PolyCollection):
            pts.append(c.get_offset_transform().transform(np.asarray(off, float)))
    for p in ax.patches:
        try:
            ext = p.get_window_extent()
            if ext.width * ext.height > 0:
                pts.append(np.array([[ext.x0, ext.y1], [ext.x1, ext.y1], [(ext.x0 + ext.x1) / 2, ext.y1]]))
        except Exception:
            pass
    pts = [p for p in pts if p.size]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def place_legends(fig):
    """legend where it covers the fewest data points; above the panel if every position covers data"""
    fig.canvas.draw()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        pts = _legend_points(ax)
        pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
        best, best_n = None, None
        for loc in (2, 1, 3, 4, 9, 8, 6, 7, 10):           # upper left/right, lower left/right, centres
            lg._loc = loc
            fig.canvas.draw()
            bb = lg.get_window_extent()
            n = int(np.sum((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))) if len(pts) else 0
            if best_n is None or n < best_n:
                best, best_n = loc, n
            if n == 0:
                break
        if best_n:                                        # still on data: make head-room above the data, legend on top
            if ax.get_yscale() == "linear":
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.45 * (hi - lo))
                lg._loc = 2
                lg.set_ncols(min(3, len(lg.get_texts()))) if hasattr(lg, "set_ncols") else None
            else:
                lg._loc = best
            for t in lg.get_texts():
                t.set_fontsize(6.3)
        else:
            lg._loc = best


def style_text(fig, k=1.0):
    roles = {}
    for ax in fig.axes:
        roles[id(ax.title)] = "title"
        for t in (ax._left_title, ax._right_title):
            roles[id(t)] = "title"
        roles[id(ax.xaxis.label)] = roles[id(ax.yaxis.label)] = "label"
        for t in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
            roles[id(t)] = "tick"
        ax.tick_params(labelsize=SIZE["tick"], length=3, width=0.6, direction="in", top=False, right=False)
        ax.xaxis.offsetText.set_fontsize(7); ax.yaxis.offsetText.set_fontsize(7)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                roles[id(t)] = "legend"
            if lg.get_title() is not None:
                roles[id(lg.get_title())] = "legend"
    for lg in fig.legends:
        for t in lg.get_texts():
            roles[id(t)] = "legend"
    for attr in ("_suptitle", "_supxlabel", "_supylabel"):
        t = getattr(fig, attr, None)
        if t is not None:
            roles[id(t)] = "title" if attr == "_suptitle" else "label"
    for t in fig.findobj(Text):
        s = t.get_text()
        if s:
            for a, b in WORDING:
                if a in s:
                    s = s.replace(a, b)
            t.set_text(s)
        role = roles.get(id(t), "other")
        t.set_fontfamily("sans-serif")
        if role in SIZE:
            t.set_fontsize(SIZE[role])
        else:
            t.set_fontsize(max(6.5, min(8.0, t.get_fontsize() * k * 1.45)) if k != 1.0 else max(6.5, min(8.0, t.get_fontsize())))
        if role in ("title", "label", "legend"):
            t.set_fontweight("bold")


def fit(fig, name):
    if name not in PRINT_WIDTH and name not in NATIVE:
        return
    use_font()
    if name in NATIVE:
        style_text(fig)
        for ax in fig.axes:
            lg = ax.get_legend()
            if lg is not None:
                lg.get_frame().set_linewidth(0.5); lg.get_frame().set_alpha(0.92)
        return
    W, H = fig.get_size_inches()
    k = PRINT_WIDTH[name] / W
    h = H * k * 1.15
    if name in MAX_HEIGHT:
        h = min(h, MAX_HEIGHT[name])
    fig.set_size_inches(PRINT_WIDTH[name], h)
    style_text(fig, k)
    lk = max(k * 1.5, 0.5)
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(max(0.8, min(1.8, ln.get_linewidth() * lk)))
        if ln.get_markersize():
            ln.set_markersize(max(2.5, min(5.0, ln.get_markersize() * lk)))
    for pc in fig.findobj(PathCollection):
        sz = pc.get_sizes()
        if len(sz):
            pc.set_sizes([max(6.0, min(30.0, v * lk ** 2)) for v in sz])
        lw = pc.get_linewidths()
        if len(lw):
            pc.set_linewidths([max(0.3, min(1.0, v * lk)) for v in lw])
    for p in fig.findobj(Patch):
        if p.get_linewidth() > 0:
            p.set_linewidth(max(0.3, min(1.2, p.get_linewidth() * lk)))
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            fr = lg.get_frame(); fr.set_linewidth(0.5); fr.set_alpha(0.92); fr.set_facecolor("white")
            lg.handlelength = 1.6; lg.borderpad = 0.3; lg.labelspacing = 0.25
    top, left = 1.0, (0.035 if getattr(fig, "_supylabel", None) is not None else 0.0)
    if fig.legends:                                    # figure-level legend: keep it above the panels
        for lg in fig.legends:
            lg._loc = 9                                # upper center
            lg.set_bbox_to_anchor((0.5, 1.0), transform=fig.transFigure)
            lg.get_frame().set_linewidth(0.0)
        fig.canvas.draw()
        hl = max(lg.get_window_extent().height for lg in fig.legends) / fig.dpi
        top = max(0.6, 1.0 - (hl + 0.04) / h)
    try:
        fig.tight_layout(pad=0.3, h_pad=0.7, w_pad=0.7, rect=(left, 0, 1, top))
    except Exception:
        pass
    place_legends(fig)
    fig.canvas.draw()
    for ax in fig.axes:                                  # overlapping tick labels (e.g. sensor names) -> rotate
        tl = [t for t in ax.get_xticklabels() if t.get_text()]
        bb = [t.get_window_extent() for t in tl]
        if any(b1.x1 > b2.x0 - 2 for b1, b2 in zip(bb[:-1], bb[1:])):
            for t in tl:
                t.set_rotation(30); t.set_ha("right"); t.set_rotation_mode("anchor")
    try:
        fig.tight_layout(pad=0.3, h_pad=0.7, w_pad=0.7, rect=(left, 0, 1, top))
    except Exception:
        pass
