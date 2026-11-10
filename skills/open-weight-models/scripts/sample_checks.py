"""Checks on a whole new sample, not just its cells: donor cell-type mix and antibody-gene (clone) usage.

The neighbour vote (`nn_annotate`) asks "does this cell look like some reference cell?". Two kinds of unusual
pass that test:

- **An unusual mix.** Every cell looks normal, but a donor has far more (or fewer) cells of one type than any
  healthy donor. `donor_composition_check` compares each donor's mix of broad cell types with healthy donors at
  their natural proportions (not a per-type-capped reference).
- **One clone of B cells.** Cells that all make the same antibody look like ordinary B cells to a model whose
  gene vocabulary leaves out the antibody genes (TEDDY-70M's vocabulary holds 3 of the 330 antibody variable
  genes and none of the 17 antibody constant genes in the TEDDY workshop's PBMC dataset, checked 2026-10-28).
  `antibody_usage` reads those genes from the raw counts:
  a clone shows one heavy-chain variable gene in most cells and only one light-chain type (kappa or lambda).

Plain pandas / NumPy / SciPy; no Databricks calls, so every function is unit-tested on made-up data.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

HEAVY_V = re.compile(r"^IGHV\d")
KAPPA = re.compile(r"^(IGKC$|IGKV\d)")
LAMBDA = re.compile(r"^(IGLC\d|IGLV\d)")          # not IGLL1 (surrogate light chain) or IGLON5 (unrelated)
ANTIBODY = re.compile(r"^((IGHV|IGHD|IGHJ|IGKV|IGKJ|IGLV|IGLJ)\d|IGKC$|IGLC\d|IGH[MDE]$|IGHG\d|IGHA\d)")
RIBOSOMAL = re.compile(r"^RP[SL]\d|^RPLP\d|^RPSA$")
MITOCHONDRIAL = re.compile(r"^MT-")
# Genes expressed from only one sex chromosome: they separate donors by sex, not by biology of interest.
SEX_LINKED = {"XIST", "TSIX", "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY", "NLGN4Y", "TXLNGY",
              "PRKY"}


# ---------------------------------------------------------------------------------------------------
# Broad cell types
# ---------------------------------------------------------------------------------------------------
def broad_label(label, rules, default="other"):
    """Map a fine cell-type label to a broad type with ordered keyword rules [(broad name, [keywords]), ...].

    Keywords match at the START of a word, case-insensitively ("t cell" does not match "malignant cell"); the
    first rule that matches wins, so put specific rules first ("dendritic" before "plasma" keeps plasmacytoid
    dendritic cells out of the B lineage). No rules (None) returns the label unchanged; no match returns `default`.
    """
    if rules is None:
        return str(label)
    s = str(label).lower()
    for name, keys in rules:
        if any(re.search(r"\b" + re.escape(k.lower()), s) for k in keys):
            return name
    return default


# Any dataset's B-lineage labels ("naive B cell", "IgG plasma cell", "plasmablast"), not plasmacytoid dendritic cells
B_LINEAGE_RULES = [("not B lineage", ["dendritic"]), ("B lineage", ["b cell", "plasma"])]


def is_b_lineage(labels):
    """Boolean mask: which labels are B cells or plasma cells (the cells that make antibodies)."""
    return np.array([broad_label(l, B_LINEAGE_RULES) == "B lineage" for l in labels])


# ---------------------------------------------------------------------------------------------------
# Donor composition
# ---------------------------------------------------------------------------------------------------
def composition(donor_ids, labels):
    """Long table, one row per donor and label: donor_id, broad_type, cells, fraction (of the donor's cells)."""
    df = pd.DataFrame({"donor_id": np.asarray(donor_ids, dtype=str), "broad_type": np.asarray(labels, dtype=str)})
    out = df.groupby(["donor_id", "broad_type"]).size().rename("cells").reset_index()
    out["fraction"] = out["cells"] / out.groupby("donor_id")["cells"].transform("sum")
    return out


def wide_fractions(long):
    """Donors x broad types (fractions, 0 where a donor has none) from a `composition` long table."""
    return long.pivot_table(index="donor_id", columns="broad_type", values="fraction", fill_value=0.0)


def donor_composition_check(new_long, healthy_long, min_cells=50):
    """Compare each new-sample donor's mix with healthy donors, one row per donor and broad type.

    new_long / healthy_long: `composition` tables (healthy donors at their natural proportions).
    A fraction outside the range of EVERY healthy donor is "above healthy range" / "below healthy range";
    `healthy_percentile` is the share of healthy donors with a smaller fraction (100 = above all of them).
    Donors with fewer than `min_cells` cells are "too few cells" (one cell is 2% of 50), not judged.
    Types seen only in the new sample count as 0 in every healthy donor, and the reverse.
    """
    new_w, healthy_w = wide_fractions(new_long), wide_fractions(healthy_long)
    types = sorted(set(new_w.columns) | set(healthy_w.columns))
    new_w = new_w.reindex(columns=types, fill_value=0.0)
    healthy_w = healthy_w.reindex(columns=types, fill_value=0.0)
    n_cells = new_long.groupby("donor_id")["cells"].sum()
    rows = []
    for donor, mix in new_w.iterrows():
        for bt in types:
            f, h = float(mix[bt]), healthy_w[bt]
            status = ("too few cells" if n_cells[donor] < min_cells else
                      "above healthy range" if f > h.max() else
                      "below healthy range" if f < h.min() else "")
            rows.append({"donor_id": donor, "n_cells": int(n_cells[donor]), "broad_type": bt, "fraction": f,
                         "healthy_min": float(h.min()), "healthy_median": float(h.median()),
                         "healthy_max": float(h.max()), "healthy_percentile": float(100 * (h < f).mean()),
                         "unusual": status})
    out = pd.DataFrame(rows)
    out["healthy_donors"] = len(healthy_w)
    return out


# ---------------------------------------------------------------------------------------------------
# Antibody genes (clone check)
# ---------------------------------------------------------------------------------------------------
def gene_mask(symbols, pattern):
    return np.array([bool(pattern.search(str(s))) for s in symbols])


def is_antibody_gene(symbols):
    """Boolean mask: antibody (immunoglobulin) gene segments and constant genes."""
    return gene_mask(symbols, ANTIBODY)


def donor_effect_genes(symbols):
    """Boolean mask of genes that mostly separate donors or capture quality, not cell biology: ribosomal,
    mitochondrial and common sex-linked genes. Use it to set them aside when reading a donor-vs-donor marker list."""
    s = np.asarray(symbols, dtype=str)
    return gene_mask(s, RIBOSOMAL) | gene_mask(s, MITOCHONDRIAL) | np.isin(s, sorted(SEX_LINKED))


def antibody_calls(X, symbols):
    """Per cell: the dominant heavy-chain variable gene (or None) and the light-chain type ("kappa", "lambda" or None).

    X: raw counts, cells x genes (dense or sparse); symbols: gene symbol per column. The dominant heavy V gene is
    the one with the most counts; the light-chain type is the one (kappa or lambda genes summed) with more counts.
    """
    X = sp.csr_matrix(X)
    symbols = np.asarray(symbols, dtype=str)
    hv = gene_mask(symbols, HEAVY_V)
    heavy = X[:, hv].toarray()
    hv_names = symbols[hv]
    has_heavy = heavy.sum(axis=1) > 0
    dominant = np.where(has_heavy, hv_names[heavy.argmax(axis=1)] if hv.any() else None, None)
    kappa = np.asarray(X[:, gene_mask(symbols, KAPPA)].sum(axis=1)).ravel()
    lam = np.asarray(X[:, gene_mask(symbols, LAMBDA)].sum(axis=1)).ravel()
    light = np.where(kappa > lam, "kappa", np.where(lam > kappa, "lambda", None))
    return pd.DataFrame({"heavy_v": dominant, "light_chain": light})


def antibody_usage(X, symbols, groups, min_cells=10):
    """One row per group (e.g. donor): how concentrated its B cells' antibody genes are.

    top_heavy_v_share: share of cells with heavy-V reads whose dominant heavy V gene is the group's most common
    one. Many different B cells (polyclonal) give a small share; one clone gives most of the cells.
    kappa_share: share of cells with light-chain reads that are kappa. A mixed population has both types; a
    clone has only one. Groups with fewer than `min_cells` cells with heavy-V reads are "too few cells".
    """
    calls = antibody_calls(X, symbols).assign(group=np.asarray(groups, dtype=str))
    rows = []
    for g, c in calls.groupby("group", sort=True):
        hv = c["heavy_v"].dropna()
        lc = c["light_chain"].dropna()
        top = hv.value_counts()
        rows.append({
            "group": g, "cells": len(c), "cells_with_heavy_v": len(hv),
            "top_heavy_v": top.index[0] if len(top) else None,
            "top_heavy_v_share": float(top.iloc[0] / len(hv)) if len(hv) else np.nan,
            "distinct_heavy_v": int(hv.nunique()),
            "cells_with_light_chain": len(lc),
            "kappa_share": float((lc == "kappa").mean()) if len(lc) else np.nan,
            "status": "too few cells" if len(hv) < min_cells else "",
        })
    return pd.DataFrame(rows)
