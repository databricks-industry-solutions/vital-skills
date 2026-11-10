"""scripts/visualize.py: every figure draws and saves on made-up data; the map falls back to PCA when small."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import visualize as viz  # noqa: E402

RNG = np.random.default_rng(0)


def test_import_changes_no_global_style():
    assert mpl.rcParams["font.size"] == mpl.rcParamsDefault["font.size"]
    with mpl.rc_context(viz.STYLE):
        assert mpl.rcParams["font.size"] == viz.MIN_PT


@pytest.mark.parametrize("n_ref", [1, 2, 3, 4, 9])
def test_map_2d_small_reference_uses_pca(n_ref):
    ref_xy, new_xy, method = viz.map_2d(RNG.normal(size=(n_ref, 512)), RNG.normal(size=(3, 512)))
    assert method == "PCA" and ref_xy.shape == (n_ref, 2) and new_xy.shape == (3, 2)
    assert np.isfinite(ref_xy).all() and np.isfinite(new_xy).all()


@pytest.mark.parametrize("n_ref", [10, 11, 200])
def test_map_2d_umap(n_ref):
    pytest.importorskip("umap")
    ref_xy, new_xy, method = viz.map_2d(RNG.normal(size=(n_ref, 512)), RNG.normal(size=(1, 512)))
    assert method == "UMAP" and ref_xy.shape == (n_ref, 2) and new_xy.shape == (1, 2)


COMPOSITION = pd.DataFrame([
    {"donor_id": d, "broad_type": bt, "fraction": f, "healthy_min": lo, "healthy_median": md, "healthy_max": hi,
     "unusual": u, "healthy_donors": 29}
    for d, bt, f, lo, md, hi, u in [("d1", "B", 0.79, 0.02, 0.1, 0.6, "above healthy range"),
                                    ("d1", "mono", 0.1, 0.0, 0.2, 0.35, ""),
                                    ("d2", "B", 0.0, 0.02, 0.1, 0.6, "below healthy range"),
                                    ("d2", "mono", 0.56, 0.0, 0.2, 0.35, "above healthy range")]])
USAGE = pd.DataFrame([
    {"group": "d1", "cells_with_heavy_v": 55, "top_heavy_v": "IGHV1-2", "top_heavy_v_share": 0.96, "kappa_share": 1.0},
    {"group": "d2", "cells_with_heavy_v": 43, "top_heavy_v": "IGHV3-23", "top_heavy_v_share": 0.12, "kappa_share": 0.52},
    {"group": "d3", "cells_with_heavy_v": 0, "top_heavy_v": None, "top_heavy_v_share": float("nan"),
     "kappa_share": float("nan")}])


def test_figures_are_detached_from_pyplot():
    """Returned figures are closed in pyplot, so a notebook shows them once (via display) — and still save."""
    before = set(viz.plt.get_fignums())
    fig = viz.plot_throughput(["CPU", "GPU"], [1.0, 2.0])
    assert set(viz.plt.get_fignums()) == before
    assert fig.axes                                         # still a complete figure


def test_every_figure_draws_and_saves(tmp_path):
    xy, qxy = RNG.normal(size=(60, 2)), RNG.normal(size=(8, 2))
    groups = RNG.choice(list("ABCDEFG"), size=60)
    cm, labels = viz.confusion_fraction(list("AABBC"), list("ABBBC"))
    assert labels == ["A", "B", "C"] and np.allclose(cm.sum(axis=1), 1)
    table = pd.DataFrame([{"method": "vote", "cells": 50, "correct": 45, "accuracy": 0.9, "accuracy_lo": 0.79,
                           "accuracy_hi": 0.96},
                          {"method": "classifier", "cells": 50, "correct": 47, "accuracy": 0.94,
                           "accuracy_lo": 0.84, "accuracy_hi": 0.98}])
    figs = {
        "map": viz.plot_reference_map(xy, groups, ["A", "B", "C", "D", "E", "F"], qxy, note="n=60", method="PCA"),
        "flagged": viz.plot_flagged_map(xy, qxy, ["not flagged"] * 4 + ["low confidence"] * 4,
                                        ["low confidence", "not flagged"], note="x"),
        "vote": viz.plot_vote_confidence(RNG.uniform(size=40), RNG.uniform(size=40) > 0.3, k=10, threshold=0.7),
        "hist": viz.plot_confidence_hist([RNG.uniform(size=40)] * 2, [RNG.uniform(size=8)] * 2, [0.5, None],
                                         ["agreement", "score"]),
        "confusion": viz.plot_confusion(cm, labels, note="rows sum to 1"),
        "per_type": viz.plot_per_type_accuracy(["A", "B"], [0.9, 0.5], [10, 4], 0.8),
        "compare": viz.plot_accuracy_comparison(table, note="same held-out cells"),
        "markers": viz.plot_markers_and_pathways(["G1", "G2"], [2.0, 1.5], ["P1"], [3.0], 1.3),
        "throughput": viz.plot_throughput(["CPU", "GPU"], [0.07, 4.2]),
        "composition": viz.plot_donor_composition(COMPOSITION, note="healthy range = min to max"),
        "antibodies": viz.plot_antibody_usage(USAGE, highlight=["d1"], note="x"),
    }
    for name, fig in figs.items():
        paths = viz.save_figure(fig, str(tmp_path), name)
        assert all(Path(p).stat().st_size > 0 for p in paths)
        viz.plt.close(fig)


def test_reference_map_lists_other_only_when_present():
    xy = RNG.normal(size=(20, 2))
    two = viz.plot_reference_map(xy, ["a"] * 10 + ["b"] * 10, ["a", "b"], method="PCA")
    three = viz.plot_reference_map(xy, ["a"] * 10 + ["b"] * 5 + ["c"] * 5, ["a", "b"], method="PCA")
    legend_text = lambda fig: [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    assert legend_text(two) == ["a", "b"]
    assert legend_text(three) == ["a", "b", "other reference types (1)"]


def test_reference_map_families_colour_related_types_as_shades():
    types = [f"t{i}" for i in range(9)]
    xy = RNG.normal(size=(90, 2))
    groups = [t for t in types for _ in range(10)]
    families = {"red": types[:1], "navy": types[1:5], "orange": types[5:7], "teal": types[7:9]}
    fig = viz.plot_reference_map(xy, groups, None, RNG.normal(size=(5, 2)), families=families,
                                 query_label="5 query cells", method="PCA")
    legend = [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    assert legend == types + ["5 query cells"]                  # every type named, no "other"
    colours = {tuple(c) for ax in fig.axes for coll in ax.collections for c in coll.get_facecolors()}
    assert len(colours) >= 9                                     # 9 distinct shades for 9 types
    viz.plt.close(fig)
    default = viz.plot_reference_map(xy, groups, types, method="PCA")
    assert "other reference types (4)" in [t.get_text() for ax in default.axes if ax.get_legend()
                                           for t in ax.get_legend().get_texts()]
    viz.plt.close(default)


def test_reference_map_families_are_checked():
    xy = RNG.normal(size=(10, 2))
    with pytest.raises(ValueError, match="at most 4 of"):
        viz.plot_reference_map(xy, ["a"] * 10, None, families={"purple": ["a"]}, method="PCA")
    with pytest.raises(ValueError, match="members > 4 shades"):
        viz.plot_reference_map(xy, ["a"] * 10, None, families={"navy": list("abcde")}, method="PCA")


def test_too_many_categories_is_refused():
    with pytest.raises(ValueError, match="group the rest"):
        viz.categorical_colors(list("ABCDEFG"))
