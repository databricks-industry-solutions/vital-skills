"""scripts/sample_checks.py on made-up data: broad labels, donor composition, antibody-gene (clone) usage."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sample_checks as sx  # noqa: E402

RULES = [("NK T cell", ["nk t cell"]), ("dendritic cell", ["dendritic"]), ("B / plasma cell", ["b cell", "plasma"]),
         ("other T cell", ["t cell"])]


# ---- broad labels ------------------------------------------------------------------------------------
@pytest.mark.parametrize("label, expected", [
    ("naive B cell", "B / plasma cell"),
    ("IgG plasma cell", "B / plasma cell"),
    ("plasmablast", "B / plasma cell"),                    # word START, so "plasma" matches "plasmablast"
    ("plasmacytoid dendritic cell", "dendritic cell"),     # earlier rule wins over "plasma"
    ("mature NK T cell", "NK T cell"),
    ("gamma-delta T cell", "other T cell"),
    ("malignant cell", "other"),                           # "t cell" inside "malignant cell" is not a word start
    ("platelet", "other"),
])
def test_broad_label_matches_at_word_start_in_rule_order(label, expected):
    assert sx.broad_label(label, RULES) == expected


def test_is_b_lineage():
    labels = ["naive B cell", "IgG plasma cell", "plasmablast", "plasmacytoid dendritic cell", "malignant cell",
              "CD14-positive monocyte", "B cell"]
    assert list(sx.is_b_lineage(labels)) == [True, True, True, False, False, False, True]


def test_broad_label_without_rules_keeps_the_label():
    assert sx.broad_label("naive B cell", None) == "naive B cell"


# ---- composition -------------------------------------------------------------------------------------
def _long(mix):
    """{donor: {type: cells}} -> composition table."""
    donors, labels = zip(*[(d, t) for d, types in mix.items() for t, n in types.items() for _ in range(n)])
    return sx.composition(donors, labels)


HEALTHY = _long({f"h{i}": {"B": 10 + i, "T": 60, "mono": 30 - i} for i in range(10)})


def test_composition_fractions_sum_to_one_per_donor():
    c = _long({"d1": {"B": 3, "T": 1}, "d2": {"T": 4}})
    assert c.groupby("donor_id")["fraction"].sum().round(12).eq(1).all()
    assert c.set_index(["donor_id", "broad_type"]).loc[("d1", "B"), "fraction"] == 0.75


def test_composition_check_flags_only_outside_every_healthy_donor():
    new = _long({"clone": {"B": 80, "T": 15, "mono": 5}, "typical": {"B": 15, "T": 60, "mono": 25}})
    out = sx.donor_composition_check(new, HEALTHY, min_cells=50).set_index(["donor_id", "broad_type"])
    assert out.loc[("clone", "B"), "unusual"] == "above healthy range"
    assert out.loc[("clone", "B"), "healthy_percentile"] == 100
    assert out.loc[("clone", "mono"), "unusual"] == "below healthy range"
    assert (out.loc["typical", "unusual"] == "").all()
    assert (out["healthy_donors"] == 10).all()


def test_composition_check_does_not_judge_small_donors():
    new = _long({"small": {"B": 20}})
    out = sx.donor_composition_check(new, HEALTHY, min_cells=50)
    assert set(out["unusual"]) == {"too few cells"}


def test_types_missing_on_one_side_count_as_zero():
    new = _long({"d": {"B": 10, "T": 50, "mono": 20, "other": 20}})      # "other" never seen in healthy donors
    out = sx.donor_composition_check(new, HEALTHY, min_cells=50).set_index("broad_type")
    assert out.loc["other", "healthy_max"] == 0 and out.loc["other", "unusual"] == "above healthy range"


# ---- antibody genes ----------------------------------------------------------------------------------
SYMBOLS = np.array(["IGHV1-2", "IGHV3-23", "IGHV4-34", "IGKC", "IGKV1-39", "IGLC2", "IGLV2-14", "IGLL1",
                    "IGLON5", "CD79A", "RPL41", "MT-CO1", "RPS4Y1", "XIST"])


def test_gene_classes():
    assert list(SYMBOLS[sx.is_antibody_gene(SYMBOLS)]) == ["IGHV1-2", "IGHV3-23", "IGHV4-34", "IGKC", "IGKV1-39",
                                                            "IGLC2", "IGLV2-14"]       # not IGLL1 / IGLON5
    assert list(SYMBOLS[sx.donor_effect_genes(SYMBOLS)]) == ["RPL41", "MT-CO1", "RPS4Y1", "XIST"]


def _cell(heavy=None, kappa=0, lam=0):
    row = np.zeros(len(SYMBOLS))
    if heavy:
        row[list(SYMBOLS).index(heavy)] = 5
    row[list(SYMBOLS).index("IGKC")] = kappa
    row[list(SYMBOLS).index("IGLC2")] = lam
    row[list(SYMBOLS).index("CD79A")] = 3
    return row


def test_antibody_calls_dominant_heavy_v_and_light_chain_type():
    X = np.array([_cell("IGHV1-2", kappa=4), _cell("IGHV3-23", lam=2), _cell(None), _cell("IGHV4-34", 1, 1)])
    calls = sx.antibody_calls(sp.csr_matrix(X), SYMBOLS)
    assert list(calls["heavy_v"]) == ["IGHV1-2", "IGHV3-23", None, "IGHV4-34"]
    assert list(calls["light_chain"]) == ["kappa", "lambda", None, None]          # a tie is not called


def test_antibody_usage_separates_a_clone_from_a_mixed_population():
    clone = [_cell("IGHV1-2", kappa=6) for _ in range(20)]
    mixed = [_cell(h, kappa=k, lam=3 - k) for h, k in zip(["IGHV1-2", "IGHV3-23", "IGHV4-34"] * 7, [3, 0] * 11)]
    X = np.array(clone + mixed)
    out = sx.antibody_usage(X, SYMBOLS, ["clone"] * 20 + ["mixed"] * len(mixed), min_cells=10).set_index("group")
    assert out.loc["clone", "top_heavy_v"] == "IGHV1-2" and out.loc["clone", "top_heavy_v_share"] == 1.0
    assert out.loc["clone", "kappa_share"] == 1.0 and out.loc["clone", "distinct_heavy_v"] == 1
    assert out.loc["mixed", "top_heavy_v_share"] == pytest.approx(1 / 3)
    assert 0.4 < out.loc["mixed", "kappa_share"] < 0.6
    assert set(out["status"]) == {""}


def test_antibody_usage_marks_small_groups():
    X = np.array([_cell("IGHV1-2", kappa=2) for _ in range(5)] + [_cell(None) for _ in range(20)])
    out = sx.antibody_usage(X, SYMBOLS, ["a"] * 5 + ["b"] * 20, min_cells=10).set_index("group")
    assert out.loc["a", "status"] == "too few cells"
    assert out.loc["b", "cells_with_heavy_v"] == 0 and pd.isna(out.loc["b", "top_heavy_v_share"])


def test_antibody_usage_without_any_heavy_v_gene_in_the_data():
    symbols = np.array(["CD79A", "IGKC"])
    out = sx.antibody_usage(np.array([[1, 2], [3, 0]]), symbols, ["a", "a"])
    assert out.loc[0, "cells_with_heavy_v"] == 0 and out.loc[0, "kappa_share"] == 1.0
