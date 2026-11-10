"""Embed cells through a serving endpoint, label them by nearest-neighbour vote, and say how far to trust it.

The tested code behind `references/workflows/embeddings-to-vector-search.md` steps 1, 1c, 5 and
"Interpret and visualise the results". Plain-language terms:

- **Vote label**: the most common label among a cell's `k` nearest reference cells.
- **Agreement** (confidence): the share of those neighbours that agree — a vote share, not a probability.
- **Top-1 score**: how close the nearest reference cell is (AI Search scores: higher is closer).
- **Wilson lower bound**: the accuracy you can be 95% sure is not an overstatement, given how many cells
  it rests on.
- **Held-out cells**: labelled cells kept out of the index, used to measure accuracy and calibrate flags.

Inference controls go to custom PyFunc (Python function model) endpoints as optional columns on the record
(default, `route="column"`; `references/integration-contract.md` §Standard), or under the REST request-body
key "params" (`route="params"`, via `invoke`) for endpoints logged before that standard.
`w.serving_endpoints.query(extra_params=...)` sends them under "extra_params", which the MLflow scoring server
drops without an error (`references/models/teddy.md` §Sending inference controls). When the response echoes
a control, the echo is checked against what was sent.

Importable without Spark, PyTorch or a GPU. scikit-learn is imported only by `compare_with_classifier`.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from vs_index import same_uc_name

# Cells per request by GPU tier for TEDDY 70M at 2,048 genes (teddy.md §Compute requirements). GPU_MEDIUM is
# 10, not 50: 50 cells ran out of memory on an A10G (24 GB) at 2,048 genes (2026-10-29); 10 per request ran
# cleanly on GPU_MEDIUM (2026-10-30). Use 8 for 400M.
MAX_BATCH = {"GPU_SMALL": 10, "GPU_MEDIUM": 10}
ROUTES = ("column", "params")


class ControlNotApplied(RuntimeError):
    """The response echo shows a setting other than the one sent; retrying cannot fix it."""

HOLDOUT_FRACTION = 0.2    # share of each cell type held out
MIN_HELD_PER_TYPE = 2     # a type gives up at least this many cells ...
MIN_REF_PER_TYPE = 3      # ... and keeps at least this many in the reference
K = 10                    # neighbours per vote (the validated runs used 10)
TARGET = 0.90             # accuracy the accepted labels must reach, as a 95% lower bound
MIN_ACCEPTED = 30         # never calibrate on fewer held-out cells than this


# ---------------------------------------------------------------------------------------------------
# Step 1a — embeddings from a Model Serving endpoint
# ---------------------------------------------------------------------------------------------------
def invoke(w, endpoint_name, records, params=None):
    """POST to the endpoint with inference controls under the body "params" key."""
    body = {"dataframe_records": records}
    if params:
        body["params"] = params
    return w.api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body=body)


def serving_endpoint_check(w, name, *, uc_model, uc_model_version):
    """Return (endpoint, workload type string) only if it serves exactly the approved version, 100% of traffic.

    An endpoint splitting traffic between versions would mix two embedding spaces in one index.
    """
    info = w.serving_endpoints.get(name)
    cfg = info.config
    if cfg is None:
        raise RuntimeError(f"{name} has no active configuration")
    # served_entities is the current field; legacy endpoints report served_models instead
    served = [(e.name, e.entity_name, e.entity_version, e.workload_type) for e in (cfg.served_entities or [])]
    if not served:
        served = [(m.name, m.model_name, m.model_version, m.workload_type) for m in (cfg.served_models or [])]
    routes = [(r.served_entity_name or r.served_model_name, r.traffic_percentage)
              for r in ((cfg.traffic_config.routes or []) if cfg.traffic_config else [])]
    if len(served) != 1:
        raise RuntimeError(f"{name} serves {len(served)} entities; this workflow needs exactly one model version "
                           "taking 100% of traffic (mixed versions mix embedding spaces).")
    served_name, model, version, workload = served[0]
    if not same_uc_name(model, uc_model) or str(version) != str(uc_model_version):
        raise RuntimeError(f"{name} serves {model} v{version}, not the approved {uc_model} v{uc_model_version}; "
                           "confirm the model and version with the user")
    if routes and routes != [(served_name, 100)]:
        raise RuntimeError(f"{name} routes traffic as {routes}; this workflow needs 100% to {served_name}")
    return info, getattr(workload, "value", workload)


def _records(X, ids, var_json, controls=None):
    rec = {
        "adata_sparsematrix": np.asarray(X, dtype=np.float32).tolist(),
        "adata_obs": pd.DataFrame({"cell_id": list(ids)}).to_json(orient="split"),
        "adata_var": var_json,
    }
    rec.update({k: str(v) for k, v in (controls or {}).items()})   # optional control columns, as strings
    return [rec]


def embed_via_endpoint(w, endpoint_name, X, cell_ids, genes, *, params, tier="GPU_SMALL", batch_size=None,
                       route="column"):
    """Embed cells through a TEDDY-style endpoint (shared single-cell request format).

    X: dense counts (cells x genes), columns in the order of `genes` (Ensembl IDs in the model vocabulary).
    params: inference controls, e.g. {"pooling": "mean", "max_seq_len": "2048"}; sent as columns on the record
    (route="column", the standard) or under the REST body "params" key (route="params", older endpoints).
    Returns an array (cells x dimension). The response holds one {"embedding": [...]} per cell, in order
    (teddy.md §Output); if a prediction also carries "cell_id", it must match the cell sent, and if it echoes
    a control (e.g. "pooling"), the echo must match the value sent.
    A failed batch is retried one cell at a time; a cell that still fails raises. No zero vectors are
    substituted: they would be indexed and returned as neighbours of every other zero vector.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if tier not in MAX_BATCH:
        raise ValueError(f"No batch limit documented for {tier}; use one of {sorted(MAX_BATCH)}")
    batch_size = batch_size or MAX_BATCH[tier]
    if batch_size > MAX_BATCH[tier]:
        raise ValueError(f"batch_size {batch_size} exceeds the {tier} limit of {MAX_BATCH[tier]}")
    X, cell_ids = np.asarray(X), list(cell_ids)
    if len(X) != len(cell_ids):
        raise ValueError(f"{len(X)} rows of counts but {len(cell_ids)} cell IDs")
    var_json = pd.DataFrame({"index": list(genes)}).to_json(orient="split")

    sent = {k: str(v) for k, v in (params or {}).items()}

    def query(rows, ids):
        if route == "column":
            records, body_params = _records(rows, ids, var_json, sent), None
        else:
            records, body_params = _records(rows, ids, var_json), sent or None
        preds = invoke(w, endpoint_name, records, params=body_params)["predictions"]
        if len(preds) != len(ids):
            raise RuntimeError(f"Endpoint returned {len(preds)} predictions for {len(ids)} cells")
        for p, cid in zip(preds, ids):
            if "cell_id" in p and str(p["cell_id"]) != str(cid):
                raise RuntimeError(f"Endpoint returned cell {p['cell_id']!r} where {cid!r} was sent")
            wrong = {k: (v, p[k]) for k, v in sent.items() if k in p and str(p[k]) != v}
            if wrong:   # the echo shows a setting other than the one sent: it was dropped or overridden
                raise ControlNotApplied(f"Endpoint used {wrong} (sent, used) via route={route!r}; the control did "
                                        "not reach predict(). Try the other route or check the signature.")
        return [[float(v) for v in p["embedding"]] for p in preds]

    out = []
    for start in range(0, len(cell_ids), batch_size):
        ids, rows = cell_ids[start:start + batch_size], X[start:start + batch_size]
        try:
            out.extend(query(rows, ids))
        except ControlNotApplied:
            raise                                   # a settings problem, not a batch-size one: don't retry
        except Exception as e:                      # e.g. CUDA out of memory on a large batch
            print(f"Batch {start} failed ({e}); retrying one cell at a time")
            for i, cid in enumerate(ids):
                out.extend(query(rows[i:i + 1], [cid]))   # let a persistent failure raise
    emb = np.asarray(out, dtype=np.float64)
    if emb.shape[0] != len(cell_ids) or not np.isfinite(emb).all() or (np.abs(emb).sum(axis=1) == 0).any():
        raise RuntimeError("Endpoint returned missing, non-finite or all-zero embeddings")
    return emb


def check_controls_applied(w, endpoint_name, X, cell_ids, genes, *, params, control="max_seq_len", value="32",
                           tier="GPU_SMALL", route="column"):
    """Prove inference controls reach predict(): a non-default value must change the embedding.

    A successful call does not prove a control was applied — extra_params also "succeeds" while ignored.
    Run once per route ("column" and "params") to test both, as integration-contract.md asks.
    Uses the cell with the most expressed genes (truncation only matters with more than `value` genes).
    Returns the cosine similarity between the default and the changed embedding.
    """
    X = np.asarray(X)
    i = int(np.argmax((X > 0).sum(axis=1)))
    if (X[i] > 0).sum() <= int(value):
        raise ValueError(f"Need a cell with more than {value} expressed genes for this check")
    one = X[i:i + 1], [cell_ids[i]]
    a = embed_via_endpoint(w, endpoint_name, *one, genes, params=params, tier=tier, route=route)[0]
    b = embed_via_endpoint(w, endpoint_name, *one, genes, params={**params, control: value}, tier=tier,
                           route=route)[0]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if cos >= 0.99:
        raise RuntimeError(f"{control}={value} gave the same embedding (cosine {cos:.4f}) via route={route!r}: "
                           "controls are not reaching predict(). Declare them in the signature (optional columns "
                           "and params) and send them as columns or under the body 'params' key.")
    print(f"Inference controls applied via route={route!r} (cosine default vs {control}={value}: {cos:.3f})")
    return cos


def unit(v):
    """Scale vectors to length 1, so AI Search's distance ranking equals cosine-similarity ranking."""
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------------
# Step 1c — hold out labelled cells
# ---------------------------------------------------------------------------------------------------
def split_holdout(cell_types, fraction=HOLDOUT_FRACTION, min_held=MIN_HELD_PER_TYPE,
                  min_ref=MIN_REF_PER_TYPE, seed=42):
    """Seeded stratified hold-out: returns (mask of held-out cells, [(rare type, cell count)])."""
    cell_types = pd.Series(cell_types).astype(str).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    held, rare = np.zeros(len(cell_types), dtype=bool), []
    for cell_type, idx in sorted(cell_types.groupby(cell_types).indices.items()):
        if len(idx) < min_held + min_ref:
            rare.append((cell_type, len(idx)))           # too few cells to measure: reference only
            continue
        n_held = min(max(min_held, round(fraction * len(idx))), len(idx) - min_ref)
        held[rng.choice(idx, size=n_held, replace=False)] = True
    return held, rare


# ---------------------------------------------------------------------------------------------------
# Step 5 — neighbours and the vote
# ---------------------------------------------------------------------------------------------------
def query_neighbours(w, index_name, vector, *, columns, k=K, exclude_ids=None, filters_json=None):
    """Top-k neighbours of one vector as dicts keyed by `columns` plus "score" (AI Search appends it last).

    Ask only for columns the index syncs. exclude_ids: cell IDs that must not count as neighbours —
    normally just the query cell, when the index may contain it. One extra result is requested per
    excluded ID, so k neighbours remain. `columns` must include "cell_id" when excluding.
    """
    exclude_ids = set(exclude_ids or ())
    if len(exclude_ids) > k:
        raise ValueError(f"exclude at most k={k} IDs per query (got {len(exclude_ids)})")
    if exclude_ids and "cell_id" not in columns:
        raise ValueError("columns must include 'cell_id' to exclude IDs")
    kwargs = {"filters_json": filters_json} if filters_json else {}
    r = w.vector_search_indexes.query_index(index_name=index_name, columns=list(columns),
                                            query_vector=[float(x) for x in vector],
                                            num_results=k + len(exclude_ids), **kwargs)
    rows = (r.result.data_array or []) if r.result else []
    hits = [dict(zip(list(columns) + ["score"], row)) for row in rows]
    hits = [h for h in hits if h.get("cell_id") not in exclude_ids]
    if exclude_ids and len(hits) < k:
        raise RuntimeError(f"only {len(hits)} neighbours left after excluding {sorted(exclude_ids)}")
    return hits[:k]


def vote(hits, label="cell_type"):
    """Vote label, agreement (share agreeing), best score and mean score for one query cell."""
    if not hits:
        raise RuntimeError("No neighbours returned — is the index ready?")
    counts = Counter(str(h[label]) for h in hits)
    predicted, n = counts.most_common(1)[0]
    scores = np.array([float(h["score"]) for h in hits])
    return {"predicted": predicted, "agreement": n / len(hits), "top1_score": float(scores.max()),
            "mean_score": float(scores.mean())}


def annotate(w, index_name, vectors, *, columns, k=K, label="cell_type", ids=None, exclude_self=False):
    """Vote for every vector; returns a DataFrame (cell_id if given, predicted, agreement, top1_score, mean_score)."""
    ids = list(ids) if ids is not None else [None] * len(vectors)
    if exclude_self and None in ids:
        raise ValueError("exclude_self needs the cell IDs")
    rows = []
    for cid, vec in zip(ids, vectors):
        hits = query_neighbours(w, index_name, vec, columns=columns, k=k,
                                exclude_ids={cid} if exclude_self else None)
        rows.append({**({"cell_id": cid} if cid is not None else {}), **vote(hits, label)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Calibration — how far to trust a label, measured on held-out cells
# ---------------------------------------------------------------------------------------------------
def wilson_lower(n_correct, n, z=1.96):
    """Lower end of the 95% Wilson interval for accuracy n_correct / n (0 when n == 0)."""
    if n == 0:
        return 0.0
    p = n_correct / n
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p + z * z / (2 * n) - margin) / (1 + z * z / n)


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k / n (returns (nan, nan) when n == 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def calibrate_vote_threshold(confidence, correct, target=TARGET, min_accepted=MIN_ACCEPTED):
    """Lowest vote share whose accepted held-out cells reach `target` as a Wilson lower bound.

    Returns (threshold, or None if no cutoff qualifies, and a table of every candidate cutoff). Even with
    every cell correct, a 0.90 lower bound needs 35 accepted cells, so small runs end with no threshold.
    """
    df = pd.DataFrame({"confidence": confidence, "correct": correct})
    rows = []
    for t in sorted(df["confidence"].unique()):
        accepted = df[df["confidence"] >= t]
        n, k = len(accepted), int(accepted["correct"].sum())
        rows.append({"threshold": t, "accepted": n, "correct": k, "accuracy": k / n,
                     "lower_bound": wilson_lower(k, n)})
    table = pd.DataFrame(rows, columns=["threshold", "accepted", "correct", "accuracy", "lower_bound"])
    ok = table[(table["accepted"] >= min_accepted) & (table["lower_bound"] >= target)]
    return (float(ok["threshold"].iloc[0]) if len(ok) else None), table


def score_floor(top1_scores, min_cells=MIN_ACCEPTED, quantile=0.05):
    """"Far" cut-off: the top-1 score that 95% of held-out cells reach (None below `min_cells` cells)."""
    s = np.asarray(top1_scores, dtype=float)
    return float(np.quantile(s, quantile)) if len(s) >= min_cells else None


def flag_cells(annotations, threshold, floor):
    """Add a "flag" column: "far from the reference", "low confidence", "not calibrated" or "" (trusted)."""
    out = annotations.copy()
    far = out["top1_score"] < floor if floor is not None else pd.Series(False, index=out.index)
    low = out["agreement"] < threshold if threshold is not None else pd.Series(True, index=out.index)
    out["flag"] = np.select([far, low], ["far from the reference",
                                         "low confidence" if threshold is not None else "not calibrated"], default="")
    return out


def calibrate_flag_rule(agreement, top1, coverage=0.95, min_cells=MIN_ACCEPTED):
    """Thresholds for the COMBINED rule "agreement < a OR best score < s", fitted on held-out healthy cells.

    Both thresholds sit at the same percentile r of their held-out distributions; r is the largest value whose
    combined rule still leaves at least `coverage` of the held-out cells unflagged (measured jointly, not per
    measure). Returns None when there are fewer than `min_cells` held-out cells: no honest threshold exists.
    """
    a, s = np.asarray(agreement, dtype=float), np.asarray(top1, dtype=float)
    if len(a) < min_cells:
        return None
    best = None
    for r in np.linspace(0.0, 1.0 - coverage, 201):          # the flag rate never falls as r rises
        at, st = float(np.quantile(a, r)), float(np.quantile(s, r))
        n_flag = int(((a < at) | (s < st)).sum())
        if n_flag <= (1.0 - coverage) * len(a):
            best = (float(r), at, st, n_flag)
    r, at, st, n_flag = best                                 # r = 0 always qualifies: nothing is below the minimum
    lo, hi = wilson_interval(n_flag, len(a))
    return {"percentile": r, "agreement_threshold": at, "score_threshold": st, "heldout_cells": len(a),
            "heldout_flagged": n_flag, "heldout_flag_rate": n_flag / len(a),
            "heldout_flag_rate_lo": lo, "heldout_flag_rate_hi": hi}


# ---------------------------------------------------------------------------------------------------
# Light adaptation — a small classifier on the frozen embeddings, compared with the vote
# ---------------------------------------------------------------------------------------------------
def accuracy_row(method, correct):
    """One comparison row: method, cells, correct, accuracy and its 95% Wilson interval."""
    correct = np.asarray(correct, dtype=bool)
    n, k = len(correct), int(correct.sum())
    lo, hi = wilson_interval(k, n)
    return {"method": method, "cells": n, "correct": k, "accuracy": k / n if n else float("nan"),
            "accuracy_lo": lo, "accuracy_hi": hi}


def compare_with_classifier(ref_emb, ref_labels, held_emb, held_labels, held_vote, *, seed=42, C=1.0, max_iter=1000):
    """Fit a logistic-regression classifier on the reference embeddings (CPU, seconds) and compare it with
    the neighbour vote on the same held-out cells.

    The encoder is not changed ("frozen"); only a linear layer is learned. Returns (comparison DataFrame,
    the classifier's held-out predictions, the fitted classifier). Held-out labels the reference never had
    count as wrong for both methods.
    """
    from sklearn.linear_model import LogisticRegression

    ref_labels, held_labels = np.asarray(ref_labels).astype(str), np.asarray(held_labels).astype(str)
    if len(np.unique(ref_labels)) < 2:
        raise ValueError("The reference needs at least two labels to train a classifier")
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(unit(ref_emb), ref_labels)
    pred = clf.predict(unit(held_emb))
    table = pd.DataFrame([
        accuracy_row(f"neighbour vote (k={K})", np.asarray(held_vote).astype(str) == held_labels),
        accuracy_row("classifier on frozen embeddings", pred == held_labels),
    ])
    return table, pred, clf
