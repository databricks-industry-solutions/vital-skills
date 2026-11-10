"""Figures for nearest-neighbour annotation of embeddings: maps, confidence, accuracy, triage.

Follows the repo's `scientific-figures` skill (palette from its `references/color.md`, white background,
equal aspect for 2-D maps, `viridis` for sequential values, PDF + PNG export at 300 dots per inch), with
screen/slide preferences on top:

- **Text at least 11 pt** (the skill's default is 8 pt for print).
- **Notes and legends sit in their own rows below the panels**, so they never overlap the data.
- **Helpers only draw.** Every number they show (thresholds, counts, accuracies) is computed by the
  caller and passed in, so notes are always data-driven.
- **Figures are returned detached from pyplot**, so a notebook shows each one once, where you call
  `display(fig)` (an open pyplot figure would be drawn again at the end of the cell). `save_figure` still works.

The style is applied inside each function (`matplotlib.rc_context`), so importing this module changes no
global settings. `map_2d` needs `umap-learn` (Uniform Manifold Approximation and Projection, UMAP) for
10 or more reference cells: `%pip install umap-learn`, then restart Python.
"""
import os
import textwrap
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Palette: scientific-figures/references/color.md
PALETTE = {
    "black": "#000000",
    "navy_900": "#0B1F3A", "navy_700": "#1F3F6E", "navy_500": "#3A6EA5", "navy_300": "#8FB3DA",
    "red_900": "#7F1414", "red_700": "#A81E1F", "red_500": "#D62728", "red_300": "#F2A49C",
    "orange_900": "#8A4300", "orange_700": "#C25E00", "orange_500": "#F28C28", "orange_300": "#FBC98E",
    "teal_900": "#00474B", "teal_700": "#00737A", "teal_500": "#1BA7A6", "teal_300": "#93D9D6",
    "grey_700": "#4D4D4D", "grey_500": "#8C8C8C", "grey_300": "#C8C8C8",
    "navy_50": "#E8EEF6", "orange_50": "#FEF3E6", "light_grey": "#F2F4F7",
}
CATEGORICAL = [PALETTE[k] for k in ("black", "red_500", "navy_500", "orange_500", "teal_500", "grey_500")]

MIN_PT = 11
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": MIN_PT, "axes.labelsize": MIN_PT, "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
    "legend.fontsize": MIN_PT, "legend.title_fontsize": MIN_PT, "axes.titlesize": MIN_PT + 1,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": "white", "legend.framealpha": 1.0, "legend.edgecolor": PALETTE["grey_300"],
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}
SINGLE_W, DOUBLE_W = 3.15, 5.85   # inches; scientific-figures width rule
UMAP_MIN_CELLS = 10               # umap-learn 0.5.12 fails below 4 reference cells; 10 leaves headroom


def styled(fn):
    """Draw inside the figure style without changing global matplotlib settings; detach returned figures."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(STYLE):
            out = fn(*args, **kwargs)
        if isinstance(out, Figure):
            plt.close(out)          # pyplot forgets it; display(fig) and fig.savefig still work
        return out
    return wrapper


def categorical_colors(categories):
    """Colour-by-count order from color.md; at most 6 hues."""
    cats = list(categories)
    if len(cats) > len(CATEGORICAL):
        raise ValueError(f"{len(cats)} categories > {len(CATEGORICAL)}; group the rest as 'other'")
    return dict(zip(cats, CATEGORICAL))


def short(label, n=30):
    label = str(label)
    return label if len(label) <= n else label[: n - 1] + "…"


def layout(width, panel_h, ncols=1, nrows=1, legend_lines=0, note_text="", left_in=0.75, right_in=0.15,
           wspace_in=0.9, xlabel_in=0.75, top_in=0.35):
    """Panels on top, then an optional legend block, then a note block — all placed in inches so nothing overlaps.

    left_in: room for y tick labels (raise it for long category names); xlabel_in: room under each panel row
    for x tick labels and the x label (raise it for rotated tick labels).
    Returns (fig, axes, legend_ax, note_ax); legend_ax / note_ax are blank axes (None if not requested).
    """
    line_h = MIN_PT * 1.4 / 72
    note_lines = textwrap.wrap(note_text, width=int(width * 11)) if note_text else []
    legend_h = legend_lines * line_h + 0.2 if legend_lines else 0.0
    note_h = len(note_lines) * line_h + 0.1 if note_lines else 0.0
    height = top_in + nrows * (panel_h + xlabel_in) + legend_h + (0.15 if legend_h else 0) + note_h + 0.1
    fig = plt.figure(figsize=(width, height))
    panel_w = (width - left_in - right_in - (ncols - 1) * wspace_in) / ncols

    def box(x_in, y_top_in, w_in, h_in):        # inches from the top-left corner -> figure fractions
        return [x_in / width, 1 - (y_top_in + h_in) / height, w_in / width, h_in / height]

    axes = []
    for r in range(nrows):
        y = top_in + r * (panel_h + xlabel_in)
        for c in range(ncols):
            axes.append(fig.add_axes(box(left_in + c * (panel_w + wspace_in), y, panel_w, panel_h)))
    y = top_in + nrows * (panel_h + xlabel_in)
    legend_ax = note_ax = None
    if legend_h:
        legend_ax = fig.add_axes(box(0.05, y, width - 0.1, legend_h)); legend_ax.axis("off")
        y += legend_h + 0.15
    if note_h:
        note_ax = fig.add_axes(box(0.05, y, width - 0.1, note_h)); note_ax.axis("off")
        note_ax.text(0, 1, "\n".join(note_lines), ha="left", va="top", fontsize=MIN_PT,
                     color=PALETTE["grey_700"], transform=note_ax.transAxes)
    return fig, axes, legend_ax, note_ax


def put_legend(legend_ax, handles, labels, ncol=2, title=None):
    legend_ax.legend(handles, labels, loc="upper center", ncol=ncol, frameon=True, title=title,
                     facecolor="white", framealpha=1.0, edgecolor=PALETTE["grey_300"], borderaxespad=0)


def dot(color, size=8):
    return Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor=color, markersize=size)


@styled
def save_figure(fig, out_dir, name, formats=("pdf", "png")):
    """Save PDF (vector, for reports) and PNG (300 dpi, for slides); returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    w_in, h_in = fig.get_size_inches()
    print(f"saved {', '.join(paths)} (figsize {w_in:.2f} x {h_in:.2f} in)")
    return paths


# ---------------------------------------------------------------------------------------------------
# 2-D map of the embeddings
# ---------------------------------------------------------------------------------------------------
def map_2d(ref, query, seed=42):
    """2-D coordinates (reference, query, method name): UMAP, or PCA when the reference is very small.

    The map is fitted on the reference only and the query is placed onto it (`transform`), so new cells
    never move the reference. PCA = principal component analysis.
    """
    ref = np.asarray(ref, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1, ref.shape[1])
    if len(ref) >= UMAP_MIN_CELLS:
        import umap
        reducer = umap.UMAP(n_neighbors=min(15, len(ref) - 1), random_state=seed).fit(ref)
        return reducer.embedding_, reducer.transform(query), "UMAP"
    print(f"Only {len(ref)} reference cells: drawing a PCA map instead of UMAP (needs {UMAP_MIN_CELLS})")
    centre = ref.mean(axis=0)
    axes = np.linalg.svd(ref - centre, full_matrices=False)[2][:2].T
    axes = np.pad(axes, ((0, 0), (0, 2 - axes.shape[1])))           # a 1-cell reference has one axis
    return (ref - centre) @ axes, (query - centre) @ axes, "PCA"


def _map_axes(ax, method):
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2"); ax.set_xticks([]); ax.set_yticks([])


@styled
def plot_reference_map(ref_xy, ref_groups, group_order, query_xy=None, note="", method="UMAP",
                       query_label="new sample", families=None):
    """Reference cells coloured by cell type, others grey; query cells as black rings with a white halo.

    The colour limit is the scientific-figures colour rule (color.md), not anything model-specific:
    Rule 1 allows at most 6 hues and grey is kept for "other", so ``group_order`` colours at most 5 types.
    Pass a meaningful ``group_order`` (e.g. the query cells' types); with a balanced reference, "the most
    common types" is an arbitrary tie-break. Name the grey types in ``note``.

    To colour more types, use Rule 2: ``families={"navy": [4 related types], "red": [...], ...}`` gives each
    group of related types (e.g. interneuron subtypes) shades of one family (at most 4 families of at most
    4 types; family names: navy, red, orange, teal). ``group_order`` is then ignored.
    """
    if families:
        if len(families) > 4 or any(f not in ("navy", "red", "orange", "teal") for f in families):
            raise ValueError("families: at most 4 of navy, red, orange, teal")
        colors, group_order = {}, []
        for fam, members in families.items():
            members = list(members)
            if len(members) > 4:
                raise ValueError(f"{fam}: {len(members)} members > 4 shades; facet or label on-plot")
            for m, s in zip(members, (500, 300, 700, 900)):    # unordered subgroups: 500 core, then 300/700/900
                colors[m] = PALETTE[f"{fam}_{s}"]
                group_order.append(m)
    else:
        group_order = list(group_order)[:5]
        colors = categorical_colors(group_order + ["other"])
    colors["other"] = PALETTE["grey_300"]
    groups = np.where(pd.Series(ref_groups).isin(group_order), np.asarray(ref_groups, dtype=object), "other")
    labels = [textwrap.fill(str(g), 34) for g in group_order]          # wrap, do not cut type names
    n_lines = sum(l.count("\n") + 1 for l in labels) + 1 + (query_xy is not None)
    fig, (ax,), leg_ax, _ = layout(DOUBLE_W, 4.2, legend_lines=int(np.ceil(n_lines / 2)) + 1, note_text=note)
    for g in ["other"] + group_order:
        m = groups == g
        ax.scatter(ref_xy[m, 0], ref_xy[m, 1], s=4, c=colors[g], linewidths=0, rasterized=True, zorder=0)
    if query_xy is not None:    # white halo under the black ring, so rings stay visible on a black cluster
        ax.scatter(query_xy[:, 0], query_xy[:, 1], s=22, facecolors="none", edgecolors="white",
                   linewidths=1.8, rasterized=True, zorder=1)
        ax.scatter(query_xy[:, 0], query_xy[:, 1], s=22, facecolors="none", edgecolors=PALETTE["black"],
                   linewidths=0.8, rasterized=True, zorder=2)
    _map_axes(ax, method)
    handles = [dot(colors[g]) for g in group_order]
    if (groups == "other").any():                  # only list "other" when some cells are grey
        n_other = len(set(map(str, ref_groups)) - set(map(str, group_order)))
        labels.append(f"other reference types ({n_other})")
        handles.append(dot(colors["other"]))
    if query_xy is not None:
        labels.append(query_label)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black",
                              markersize=8))
    put_legend(leg_ax, handles, labels, ncol=2)
    return fig


@styled
def plot_flagged_map(ref_xy, query_xy, query_category, category_order, note="", method="UMAP"):
    """Reference in light grey; new-sample cells coloured by triage category ("not flagged" is always grey)."""
    flagged = [c for c in category_order if c != "not flagged"]
    colors = categorical_colors(flagged)
    colors["not flagged"] = PALETTE["grey_500"]
    fig, (ax,), leg_ax, _ = layout(DOUBLE_W, 4.2, legend_lines=int(np.ceil((len(category_order) + 1) / 2)) + 1,
                                   note_text=note)
    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], s=3, c=PALETTE["grey_300"], linewidths=0, rasterized=True, zorder=0)
    cat = np.asarray(query_category)
    for c in list(category_order)[::-1]:
        m = cat == c
        ax.scatter(query_xy[m, 0], query_xy[m, 1], s=12, c=colors[c], linewidths=0, rasterized=True, zorder=0)
    _map_axes(ax, method)
    put_legend(leg_ax, [dot(PALETTE["grey_300"])] + [dot(colors[c]) for c in category_order],
               ["reference cells"] + [short(c, 36) for c in category_order], ncol=2)
    return fig


# ---------------------------------------------------------------------------------------------------
# Confidence and accuracy
# ---------------------------------------------------------------------------------------------------
@styled
def plot_vote_confidence(confidence, correct, k, threshold=None, note=""):
    """Held-out vote share, stacked correct vs wrong, with the calibrated threshold (if any)."""
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=bool)
    fig, (ax,), leg_ax, _ = layout(DOUBLE_W, 2.6, legend_lines=1, note_text=note)
    bins = np.linspace(0, 1, k + 2)
    ax.hist([confidence[correct], confidence[~correct]], bins=bins, stacked=True,
            color=[PALETTE["navy_500"], PALETTE["red_500"]])
    handles, labels = [dot(PALETTE["navy_500"]), dot(PALETTE["red_500"])], ["correct", "wrong"]
    if threshold is not None:
        ax.axvline(threshold, color=PALETTE["grey_700"], linestyle="--", linewidth=1.5)
        handles.append(Line2D([0], [0], color=PALETTE["grey_700"], linestyle="--"))
        labels.append(f"threshold {threshold:.2f}")
    ax.set_xlabel("Vote share (confidence)"); ax.set_ylabel("Held-out cells")
    put_legend(leg_ax, handles, labels, ncol=3)
    return fig


@styled
def plot_confidence_hist(heldout_vals, query_vals, thresholds, xlabels, note=""):
    """Two panels (neighbour agreement, best neighbour score): held-out reference vs new sample, with thresholds."""
    fig, axes, leg_ax, _ = layout(DOUBLE_W, 2.4, ncols=2, legend_lines=1, note_text=note)
    for ax, h, q, t, xl in zip(axes, heldout_vals, query_vals, thresholds, xlabels):
        lo, hi = float(min(np.min(h), np.min(q))), float(max(np.max(h), np.max(q)))
        bins = np.linspace(lo, hi, 21)
        ax.hist(h, bins=bins, color=PALETTE["grey_500"], alpha=0.7, density=True)
        ax.hist(q, bins=bins, histtype="step", color=PALETTE["black"], linewidth=1.5, density=True)
        if t is not None:
            ax.axvline(t, color=PALETTE["red_500"], linestyle="--", linewidth=1.5)
        ax.set_xlabel(xl); ax.set_ylabel("Density")
    put_legend(leg_ax, [dot(PALETTE["grey_500"]), Line2D([0], [0], color="black"),
                        Line2D([0], [0], color=PALETTE["red_500"], linestyle="--")],
               ["held-out reference cells", "new sample", "calibrated threshold"], ncol=3)
    return fig


def confusion_fraction(true, predicted, labels=None):
    """Row-normalised confusion matrix (rows: true label, columns: predicted) and its label order."""
    labels = list(labels) if labels is not None else sorted(set(map(str, true)) | set(map(str, predicted)))
    cm = pd.crosstab(pd.Series(true, dtype=str), pd.Series(predicted, dtype=str), normalize="index")
    cm = cm.reindex(index=labels, columns=labels, fill_value=0.0).fillna(0.0)
    return cm.to_numpy(), labels


@styled
def plot_confusion(cm_frac, labels, note="", ylabel="True label"):
    """Row-normalised confusion matrix: the diagonal is per-type accuracy."""
    n = len(labels)
    fig, (ax,), _, _ = layout(DOUBLE_W, 4.0, note_text=note, left_in=1.9, xlabel_in=2.0, top_in=0.9)
    im = ax.imshow(cm_frac, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(l, 22) for l in labels], rotation=60, ha="right")
    ax.set_yticklabels([short(l, 22) for l in labels])
    ax.set_xlabel("Predicted (neighbour vote)"); ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax, location="top", fraction=0.05, pad=0.04)
    cb.set_label("Share of the row's cells")
    return fig


@styled
def plot_per_type_accuracy(types, accuracy, n_cells, overall, note=""):
    """Horizontal bars of accuracy per cell type (a ranked list, so taller than wide is fine)."""
    h = max(2.5, 0.24 * len(types))
    fig, (ax,), _, _ = layout(DOUBLE_W, h, note_text=note, left_in=3.1)
    y = np.arange(len(types))
    ax.barh(y, accuracy, color=PALETTE["navy_500"], height=0.7)
    ax.axvline(overall, color=PALETTE["grey_700"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels([f"{short(t, 30)} (n={n})" for t, n in zip(types, n_cells)])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Held-out accuracy (share correct)")
    return fig


@styled
def plot_accuracy_comparison(table, note=""):
    """Held-out accuracy per method with 95% Wilson intervals (rows of nn_annotate.accuracy_row)."""
    t = pd.DataFrame(table)
    h = max(1.6, 0.45 * len(t))
    fig, (ax,), _, _ = layout(DOUBLE_W, h, note_text=note, left_in=2.9)
    y = np.arange(len(t))
    colors = [PALETTE["grey_500"], PALETTE["navy_500"], PALETTE["red_500"], PALETTE["teal_500"]]
    ax.barh(y, t["accuracy"], color=[colors[i % len(colors)] for i in range(len(t))], height=0.6)
    err = np.vstack([t["accuracy"] - t["accuracy_lo"], t["accuracy_hi"] - t["accuracy"]])
    ax.errorbar(t["accuracy"], y, xerr=err, fmt="none", ecolor=PALETTE["black"], capsize=3, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{short(m, 34)} (n={n})" for m, n in zip(t["method"], t["cells"])])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Held-out accuracy (95% interval)")
    return fig


# ---------------------------------------------------------------------------------------------------
# Triage and compute
# ---------------------------------------------------------------------------------------------------
@styled
def plot_markers_and_pathways(genes, logfc, terms, neg_log10_padj, padj_line, note=""):
    """Top: marker genes (log fold change). Bottom: enriched pathways (-log10 adjusted p)."""
    n = max(len(genes), len(terms), 1)
    fig, axes, _, _ = layout(DOUBLE_W, max(2.2, 0.24 * n), nrows=2, note_text=note, left_in=3.3, top_in=0.45)
    ax1, ax2 = axes
    y = np.arange(len(genes))
    ax1.barh(y, logfc, color=PALETTE["red_500"], height=0.7)
    ax1.set_yticks(y); ax1.set_yticklabels(genes); ax1.invert_yaxis()
    ax1.set_xlabel("log2 fold change"); ax1.set_title("Top marker genes", loc="left")
    y = np.arange(len(terms))
    ax2.barh(y, neg_log10_padj, color=PALETTE["navy_500"], height=0.7)
    ax2.axvline(padj_line, color=PALETTE["grey_700"], linestyle="--", linewidth=1.2)
    ax2.set_yticks(y); ax2.set_yticklabels([short(t, 30) for t in terms]); ax2.invert_yaxis()
    ax2.set_xlabel("-log10 adjusted p-value"); ax2.set_title("Enriched pathways", loc="left")
    return fig


@styled
def plot_throughput(labels, cells_per_s, note=""):
    """Bar per compute option: cells embedded per second (y starts at zero)."""
    fig, (ax,), _, _ = layout(SINGLE_W, 2.6, note_text=note)
    x = np.arange(len(labels))
    ax.bar(x, cells_per_s, color=[PALETTE["grey_500"], PALETTE["black"], PALETTE["red_500"]][: len(labels)], width=0.6)
    for xi, v in zip(x, cells_per_s):          # value labels sit above each bar, so small bars stay readable
        ax.text(xi, v + max(cells_per_s) * 0.02, f"{v:.3g}", ha="center", va="bottom", fontsize=MIN_PT)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Cells per second"); ax.set_ylim(0, max(cells_per_s) * 1.2)
    return fig


# ---------------------------------------------------------------------------------------------------
# Whole-sample checks (scripts/sample_checks.py)
# ---------------------------------------------------------------------------------------------------
@styled
def plot_donor_composition(check, note=""):
    """Per broad cell type: the healthy donors' range (grey bar, median tick) and each new donor's share (dot).

    check: rows of sample_checks.donor_composition_check. Dots outside the grey range ("unusual") get a black
    ring. Types are ordered by the largest new-donor share.
    """
    c = pd.DataFrame(check)
    donors = sorted(c["donor_id"].unique())
    hues = [h for h in CATEGORICAL if h != PALETTE["black"]]          # black is kept for the "unusual" ring
    if len(donors) > len(hues):
        raise ValueError(f"{len(donors)} donors > {len(hues)}; plot a subset")
    colors = dict(zip(donors, hues))
    order = c.groupby("broad_type")["fraction"].max().sort_values(ascending=False).index.tolist()
    rng = c.drop_duplicates("broad_type").set_index("broad_type").loc[order]
    h = max(2.0, 0.34 * len(order))
    fig, (ax,), leg_ax, _ = layout(DOUBLE_W, h, legend_lines=int(np.ceil((len(donors) + 1) / 3)) + 1,
                                   note_text=note, left_in=2.4)
    y = np.arange(len(order))
    ax.barh(y, rng["healthy_max"] - rng["healthy_min"], left=rng["healthy_min"], height=0.5,
            color=PALETTE["grey_300"], zorder=0)
    ax.scatter(rng["healthy_median"], y, marker="|", s=120, color=PALETTE["grey_700"], zorder=1)
    offsets = np.linspace(-0.2, 0.2, len(donors)) if len(donors) > 1 else [0.0]
    for d, off in zip(donors, offsets):
        rows = c[c["donor_id"] == d].set_index("broad_type").reindex(order)
        unusual = rows["unusual"].isin(["above healthy range", "below healthy range"]).values
        ax.scatter(rows["fraction"], y + off, s=36, color=colors[d], zorder=2,
                   edgecolors=np.where(unusual, PALETTE["black"], colors[d]), linewidths=np.where(unusual, 1.5, 0))
    ax.set_yticks(y); ax.set_yticklabels([short(t, 28) for t in order]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Share of the donor's cells")
    n_h = int(c["healthy_donors"].iloc[0]) if "healthy_donors" in c else None
    put_legend(leg_ax, [dot(PALETTE["grey_300"], 10)] + [dot(colors[d]) for d in donors],
               [f"healthy donors' range{f' (n={n_h})' if n_h else ''}"] + [short(d, 20) for d in donors], ncol=3)
    return fig


@styled
def plot_antibody_usage(usage, highlight=(), note=""):
    """Two panels per group (rows of sample_checks.antibody_usage): share of cells using the most common
    heavy-chain V gene, and share of light-chain calls that are kappa. Groups in `highlight` are drawn in red."""
    u = pd.DataFrame(usage)
    u = u[u["cells_with_heavy_v"] > 0].reset_index(drop=True)
    h = max(1.6, 0.42 * len(u))
    fig, (ax1, ax2), _, _ = layout(DOUBLE_W, h, ncols=2, note_text=note, left_in=2.0, wspace_in=0.3)
    y = np.arange(len(u))
    colors = [PALETTE["red_500"] if g in set(highlight) else PALETTE["navy_500"] for g in u["group"]]
    ax1.barh(y, u["top_heavy_v_share"], color=colors, height=0.6)
    for yi, (share, gene) in enumerate(zip(u["top_heavy_v_share"], u["top_heavy_v"])):
        ax1.text(min(share, 1) + 0.02, yi, str(gene), va="center", fontsize=MIN_PT - 1, color=PALETTE["grey_700"])
    ax1.set_yticks(y); ax1.set_yticklabels([f"{short(g, 20)} (n={n})" for g, n in zip(u["group"], u["cells_with_heavy_v"])])
    ax1.invert_yaxis(); ax1.set_xlim(0, 1.45); ax1.set_xticks([0, 0.5, 1])
    ax1.set_xlabel("Top heavy V gene (share)"); ax1.set_title("One heavy chain?", loc="left")
    ax2.barh(y, u["kappa_share"], color=colors, height=0.6)
    ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.invert_yaxis(); ax2.set_xlim(0, 1)
    ax2.set_xlabel("Kappa light chain (share)"); ax2.set_title("One light-chain type?", loc="left")
    return fig
