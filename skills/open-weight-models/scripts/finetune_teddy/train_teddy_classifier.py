"""Fine-tune TEDDY-70M for cell-type classification on one GPU (Databricks AI Runtime), and compare it with
the frozen model on held-out donors.

**Not upstream-validated.** The TEDDY repository (Hugging Face `Merck/TEDDY`, revision
8c880aa67674042ccdc254bccb5de081092a48bf) says its models can be fine-tuned (`README.md:83`, `:284-286`) and
ships a classification head (`teddy/models/classification_heads.py:70-88`, attached with
`TeddyGModel.add_classification_head`, `teddy/models/teddy_g/model.py:158-163`), but it ships **no fine-tuning
script**. The training loop below is ours. What it does take from upstream, unchanged:

- preprocessing (`normalize_data_inplace`, `scale_columns_by_median_dict`) and tokenization
  (`_check_genes_in_tokenizer`, `_build_batch_tensors`) with the pretraining settings the model card lists
  (normalise to 10,000 counts, divide by gene medians, no log, top 2,048 genes, zeros dropped);
- the model class, weights and the classification head.

Before training it checks that the frozen encoder gives the same cell embedding for a cell whether it is run
alone or padded inside a batch (cosine >= 0.9999, the threshold of the skill's upstream-parity test), so the
batching cannot be what changes the result. It then reports three held-out accuracies on the SAME cells:
neighbour vote and logistic regression on the frozen embeddings, and the fine-tuned model.

A fine-tuned model is a different model: give it its own Unity Catalog name, and never write its embeddings
into an index built from the original model.

Run on AI Runtime with `workload.yaml` (next to this file), or on CPU for a smoke test:

    python train_teddy_classifier.py --params params.yaml --smoke --no-mlflow

Hyperparameters come from `$HYPERPARAMETERS_PATH` (written by AI Runtime from the workload's `parameters:`),
or `--params`. Metrics and the checkpoint are logged to the run in `$MLFLOW_RUN_ID` when AI Runtime sets it.
Both variable names come from the databricks-ai-runtime skill; confirm them with `databricks air run --dry-run`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

UPSTREAM_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"
PARITY_COSINE = 0.9999

DEFAULTS = {
    "snapshot_dir": None,          # Volume folder holding the TEDDY snapshot (contains teddy/ and data/)
    "variant": "70M",
    "reference_h5ad": None,        # raw counts, var_names = Ensembl IDs (or var["feature_id"]), obs[label_column]
    "heldout_h5ad": None,          # cells from donors NOT in the reference
    "label_column": "cell_type",
    "max_seq_len": 2048,           # pretraining setting (README.md:171-190)
    "unfreeze_layers": 2,          # last N of the 12 encoder layers are trained; the rest stay frozen
    "head_layers": 1,              # hidden layers in upstream's ClassificationHead
    "epochs": 3,
    "learning_rate": 5.0e-5,       # encoder layers
    "head_learning_rate": 1.0e-3,  # classification head
    "weight_decay": 0.01,
    "batch_size": 16,
    "max_cells": None,             # cap on reference cells (stratified); None = all
    "min_cells_per_label": 5,      # labels with fewer reference cells are dropped from training
    "knn_k": 10,
    "parity_cells": 5,
    "seed": 42,
    "output_dir": "/tmp/teddy_finetune",
}
SMOKE = {"max_cells": 64, "epochs": 1, "batch_size": 4, "max_seq_len": 256, "parity_cells": 2,
         "min_cells_per_label": 2, "knn_k": 3}


# ---------------------------------------------------------------------------------------------------
# Plain helpers (no PyTorch needed; unit-tested on CPU)
# ---------------------------------------------------------------------------------------------------
def load_hyperparameters(path=None, overrides=None, smoke=False):
    """DEFAULTS, then the YAML file at `path` (or $HYPERPARAMETERS_PATH), then smoke settings, then overrides."""
    hp = dict(DEFAULTS)
    path = path or os.environ.get("HYPERPARAMETERS_PATH")
    if path:
        import yaml
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown hyperparameters {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        hp.update(loaded)
    if smoke:
        hp.update(SMOKE)
    hp.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("snapshot_dir", "reference_h5ad", "heldout_h5ad"):
        if not hp[key]:
            raise ValueError(f"Set {key} (in the parameters file or on the command line)")
    if not 0 <= int(hp["unfreeze_layers"]):
        raise ValueError("unfreeze_layers must be >= 0")
    return hp


def stratified_subsample(labels, max_cells, seed):
    """Indices of at most `max_cells` cells, the same share from every label (at least one each)."""
    labels = np.asarray(labels).astype(str)
    idx = np.arange(len(labels))
    if max_cells is None or max_cells >= len(labels):
        return idx
    rng = np.random.default_rng(seed)
    groups = {lab: idx[labels == lab] for lab in sorted(set(labels))}
    if max_cells < len(groups):
        raise ValueError(f"max_cells={max_cells} is below the number of labels ({len(groups)})")
    size = {lab: max(1, int(round(max_cells * len(m) / len(labels)))) for lab, m in groups.items()}
    while sum(size.values()) > max_cells:                    # trim the largest share first; never below one
        size[max(size, key=lambda lab: (size[lab], lab))] -= 1
    keep = np.concatenate([rng.choice(m, size=size[lab], replace=False) for lab, m in groups.items()])
    return np.sort(keep)


def encode_labels(ref_labels, held_labels, min_cells):
    """Classes trained on, reference/held-out codes, and which held-out cells have a label the model never saw.

    Reference labels with fewer than `min_cells` cells are dropped (code -1). Held-out cells whose label is
    not a trained class get code -1 and count as wrong for every method.
    """
    ref_labels, held_labels = np.asarray(ref_labels).astype(str), np.asarray(held_labels).astype(str)
    values, counts = np.unique(ref_labels, return_counts=True)
    classes = [str(v) for v, c in zip(values, counts) if c >= min_cells]   # plain str: loads with weights_only
    if len(classes) < 2:
        raise ValueError(f"Need at least two labels with >= {min_cells} reference cells, got {classes}")
    code = {c: i for i, c in enumerate(classes)}
    ref_codes = np.array([code.get(x, -1) for x in ref_labels])
    held_codes = np.array([code.get(x, -1) for x in held_labels])
    return classes, ref_codes, held_codes


def trainable_parameter_names(names, nlayers, unfreeze_layers):
    """Parameters to train: the classification head plus the last `unfreeze_layers` encoder layers."""
    first = nlayers - min(int(unfreeze_layers), nlayers)
    prefixes = ["cls_decoder."] + [f"encoder.layers.{i}." for i in range(first, nlayers)]
    return [n for n in names if any(n.startswith(p) for p in prefixes)]


def knn_vote(ref_emb, ref_codes, query_emb, k):
    """Majority code among the k most cosine-similar reference cells (exact, NumPy)."""
    ref = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    q = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    top = np.argsort(-(q @ ref.T), axis=1)[:, :k]
    return np.array([np.bincount(ref_codes[row]).argmax() for row in top])


def accuracy(pred_codes, true_codes):
    """Share correct; held-out cells with an untrained label (code -1) always count as wrong."""
    true_codes = np.asarray(true_codes)
    return float(np.mean((np.asarray(pred_codes) == true_codes) & (true_codes >= 0))) if len(true_codes) else float("nan")


# ---------------------------------------------------------------------------------------------------
# Upstream TEDDY code (imported from the snapshot; needs torch, transformers==4.41.0, anndata, scanpy)
# ---------------------------------------------------------------------------------------------------
def import_upstream(snapshot_dir):
    """Import TEDDY's own modules from the snapshot, flushing any cached copies first."""
    for k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
        del sys.modules[k]
    sys.path.insert(0, snapshot_dir)
    from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
    from teddy.data_processing.tokenization.tokenization import _build_batch_tensors, _check_genes_in_tokenizer
    from teddy.models.model_directory import get_architecture, model_dict
    from teddy.tokenizer.gene_tokenizer import GeneTokenizer
    return {"normalize": normalize_data_inplace, "scale_by_median": scale_columns_by_median_dict,
            "build_batch": _build_batch_tensors, "check_genes": _check_genes_in_tokenizer,
            "get_architecture": get_architecture, "model_dict": model_dict, "GeneTokenizer": GeneTokenizer}


def read_cells(path):
    """Read an .h5ad of raw counts; use Ensembl IDs as gene names."""
    import anndata as ad
    a = ad.read_h5ad(path)
    a = a.raw.to_adata() if a.raw is not None else a
    if not str(a.var_names[0]).startswith("ENSG") and "feature_id" in a.var:
        a.var_names = a.var["feature_id"].astype(str).values
        a.var_names_make_unique()
    return a


def tokenize_cells(a, up, tokenizer, medians_path, max_seq_len, chunk=512):
    """Upstream preprocessing + tokenization -> (gene_ids, attention_mask) LongTensors, zeros padded and masked."""
    import types

    import torch
    from scipy.sparse import csr_matrix

    in_vocab, coding_genes, ratio = up["check_genes"](a, "index", tokenizer)
    token_array = torch.tensor(tokenizer.encode(coding_genes.tolist(), add_special_tokens=False))
    args = types.SimpleNamespace(max_seq_len=max_seq_len, add_cls=False, random_genes=False)
    ids_out, mask_out = [], []
    for start in range(0, a.n_obs, chunk):
        part = a[start:start + chunk]
        layer = csr_matrix(part.X, dtype=np.float32)
        up["normalize"](layer, 10000)
        up["scale_by_median"](layer, part, medians_path, "index")
        X = torch.tensor(layer.toarray()[:, in_vocab], dtype=torch.float)
        ids, vals, _, _ = up["build_batch"](X, token_array, args)
        expressed = vals != 0                                   # include_zero_genes=False
        ids = ids.clone()
        ids[~expressed] = tokenizer.pad_token_id
        ids_out.append(ids)
        mask_out.append(expressed.long())                       # 1 = attend; the model inverts it (model.py:236-238)
    width = max(t.shape[1] for t in ids_out)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, width - t.shape[1]), value=v)   # noqa: E731
    return (torch.cat([pad(t, tokenizer.pad_token_id) for t in ids_out]),
            torch.cat([pad(m, 0) for m in mask_out]), ratio)


def cell_embeddings(model, gene_ids, attention_mask, batch_size, device):
    """Frozen first-position embeddings ("cell_emb", model.py:261-264) for every cell."""
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(gene_ids), batch_size):
            ids, m = gene_ids[s:s + batch_size].to(device), attention_mask[s:s + batch_size].to(device)
            out.append(model(gene_ids=ids, attention_mask=m)["cell_emb"].float().cpu().numpy())
    return np.concatenate(out)


def batching_parity_check(model, gene_ids, attention_mask, n, device):
    """Each cell alone (only its expressed tokens) vs padded in a batch: cosine must be >= PARITY_COSINE."""
    import torch
    n = min(n, len(gene_ids))
    batched = cell_embeddings(model, gene_ids[:n], attention_mask[:n], n, device)
    worst = 1.0
    with torch.no_grad():
        for i in range(n):
            ids = gene_ids[i][attention_mask[i].bool()].unsqueeze(0).to(device)
            alone = model(gene_ids=ids, attention_mask=torch.ones_like(ids))["cell_emb"][0].float().cpu().numpy()
            cos = float(alone @ batched[i] / (np.linalg.norm(alone) * np.linalg.norm(batched[i])))
            worst = min(worst, cos)
    if worst < PARITY_COSINE:
        raise RuntimeError(f"Batched embeddings differ from per-cell ones (worst cosine {worst:.6f}); "
                           "fix the padding/mask before training")
    print(f"Batching parity: worst cosine over {n} cells = {worst:.8f} (>= {PARITY_COSINE})")
    return worst


def train(hp, use_mlflow=True):
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression

    torch.manual_seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    up = import_upstream(hp["snapshot_dir"])
    model_dir = os.path.join(hp["snapshot_dir"], "teddy", "models", "teddy_g", hp["variant"])
    medians = os.path.join(hp["snapshot_dir"], "teddy", "data_processing", "utils", "medians", "data",
                           "teddy_gene_medians.json")
    tokenizer = up["GeneTokenizer"].from_pretrained(model_dir)

    ref, held = read_cells(hp["reference_h5ad"]), read_cells(hp["heldout_h5ad"])
    keep = stratified_subsample(ref.obs[hp["label_column"]], hp["max_cells"], hp["seed"])
    ref = ref[keep].copy()
    if hp["max_cells"]:
        held = held[stratified_subsample(held.obs[hp["label_column"]], hp["max_cells"], hp["seed"])].copy()
    classes, ref_codes, held_codes = encode_labels(ref.obs[hp["label_column"]], held.obs[hp["label_column"]],
                                                   hp["min_cells_per_label"])
    train_rows = np.where(ref_codes >= 0)[0]
    print(f"{len(train_rows)} reference cells, {len(classes)} labels; {len(held_codes)} held-out cells "
          f"({int((held_codes < 0).sum())} with a label the model never saw)")

    t0 = time.time()
    ref_ids, ref_mask, ref_ratio = tokenize_cells(ref, up, tokenizer, medians, hp["max_seq_len"])
    held_ids, held_mask, held_ratio = tokenize_cells(held, up, tokenizer, medians, hp["max_seq_len"])
    print(f"Tokenized in {time.time() - t0:.0f} s; genes in vocabulary: reference {ref_ratio:.1%}, held-out {held_ratio:.1%}")

    arch = up["get_architecture"](model_dir)
    cfg_cls, model_cls = up["model_dict"][arch]["config_cls"], up["model_dict"][arch]["model_cls"]
    config = cfg_cls.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir, config=config).to(device)

    # 1. Frozen model: batching parity, then the two frozen baselines on the same held-out cells
    parity = batching_parity_check(model, ref_ids, ref_mask, hp["parity_cells"], device)
    rows_t = torch.as_tensor(train_rows)
    ref_emb = cell_embeddings(model, ref_ids[rows_t], ref_mask[rows_t], hp["batch_size"], device)
    held_emb = cell_embeddings(model, held_ids, held_mask, hp["batch_size"], device)
    acc_knn = accuracy(knn_vote(ref_emb, ref_codes[train_rows], held_emb, hp["knn_k"]), held_codes)
    clf = LogisticRegression(max_iter=1000, random_state=hp["seed"]).fit(
        ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True), ref_codes[train_rows])
    acc_lr = accuracy(clf.predict(held_emb / np.linalg.norm(held_emb, axis=1, keepdims=True)), held_codes)
    print(f"Frozen: neighbour vote {acc_knn:.3f}, logistic regression {acc_lr:.3f}")

    # 2. Attach upstream's head and choose what trains
    model.add_classification_head(config.d_model, len(classes), hp["head_layers"])
    model.to(device)
    names = trainable_parameter_names([n for n, _ in model.named_parameters()], config.nlayers, hp["unfreeze_layers"])
    for n, p in model.named_parameters():
        p.requires_grad = n in names
    head = [p for n, p in model.named_parameters() if n.startswith("cls_decoder.")]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls_decoder.")]
    groups = [{"params": head, "lr": hp["head_learning_rate"]}] + ([{"params": enc, "lr": hp["learning_rate"]}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
    n_train = sum(p.numel() for p in head + enc)
    print(f"Training {n_train:,} of {sum(p.numel() for p in model.parameters()):,} parameters "
          f"(head + last {hp['unfreeze_layers']} encoder layers) on {device}")

    def predict(ids, mask):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(ids), hp["batch_size"]):
                o = model(gene_ids=ids[s:s + hp["batch_size"]].to(device), attention_mask=mask[s:s + hp["batch_size"]].to(device))
                out.append(o["cls_output"]["output"].argmax(dim=-1).cpu().numpy())
        return np.concatenate(out)

    history = []
    y = torch.tensor(ref_codes, dtype=torch.long)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    for epoch in range(int(hp["epochs"])):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for s in range(0, len(order), hp["batch_size"]):
            b = torch.as_tensor(order[s:s + hp["batch_size"]])
            with autocast:
                logits = model(gene_ids=ref_ids[b].to(device), attention_mask=ref_mask[b].to(device))["cls_output"]["output"]
            loss = F.cross_entropy(logits.float(), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        acc = accuracy(predict(held_ids, held_mask), held_codes)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "heldout_accuracy": acc})
        print(f"epoch {epoch + 1}: train loss {np.mean(losses):.4f}, held-out accuracy {acc:.3f}")
        if use_mlflow:
            import mlflow
            mlflow.log_metrics({"train_loss": float(np.mean(losses)), "heldout_accuracy_finetuned": acc}, step=epoch + 1)

    # 3. Save the trained parameters (not the whole model: the frozen part is the upstream checkpoint)
    os.makedirs(hp["output_dir"], exist_ok=True)
    ckpt = os.path.join(hp["output_dir"], "teddy_cell_type_head.pt")
    torch.save({"trainable_state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                "classes": classes, "head_layers": hp["head_layers"], "unfreeze_layers": hp["unfreeze_layers"],
                "upstream_revision": UPSTREAM_REVISION, "variant": hp["variant"]}, ckpt)
    summary = {
        "not_upstream_validated": "TEDDY ships no fine-tuning script; this training loop is ours.",
        "upstream_revision": UPSTREAM_REVISION, "variant": hp["variant"], "device": device,
        "classes": classes, "reference_cells": int(len(train_rows)), "heldout_cells": int(len(held_codes)),
        "heldout_cells_unseen_label": int((held_codes < 0).sum()), "batching_parity_worst_cosine": parity,
        "heldout_accuracy_frozen_knn": acc_knn, "heldout_accuracy_frozen_logreg": acc_lr,
        "heldout_accuracy_finetuned": history[-1]["heldout_accuracy"] if history else None,
        "history": history, "hyperparameters": hp,
    }
    with open(os.path.join(hp["output_dir"], "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if use_mlflow:
        import mlflow
        mlflow.log_params({k: v for k, v in hp.items() if k not in ("snapshot_dir", "reference_h5ad", "heldout_h5ad", "output_dir")})
        mlflow.log_metrics({"heldout_accuracy_frozen_knn": acc_knn, "heldout_accuracy_frozen_logreg": acc_lr,
                            "batching_parity_worst_cosine": parity})
        mlflow.set_tags({"teddy.upstream_revision": UPSTREAM_REVISION, "teddy.not_upstream_validated": "true"})
        mlflow.log_artifacts(hp["output_dir"], artifact_path="finetune")
    print(json.dumps({k: summary[k] for k in ("heldout_accuracy_frozen_knn", "heldout_accuracy_frozen_logreg",
                                              "heldout_accuracy_finetuned")}, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--params", help="YAML hyperparameters (default: $HYPERPARAMETERS_PATH)")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run: a few cells, one pass")
    p.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow (local smoke tests)")
    for key in ("snapshot_dir", "reference_h5ad", "heldout_h5ad", "output_dir"):
        p.add_argument(f"--{key.replace('_', '-')}", dest=key)
    args = p.parse_args(argv)
    hp = load_hyperparameters(args.params, {k: getattr(args, k) for k in ("snapshot_dir", "reference_h5ad",
                                                                          "heldout_h5ad", "output_dir")}, args.smoke)
    if args.no_mlflow:
        return train(hp, use_mlflow=False)
    import mlflow
    run_id = os.environ.get("MLFLOW_RUN_ID")          # set by AI Runtime for its run
    with (mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name="teddy-70m-cell-type-finetune")):
        return train(hp, use_mlflow=True)


if __name__ == "__main__":
    main()
