"""scripts/finetune_teddy: the plain helpers and the workload file (no PyTorch, no TEDDY weights).

The training loop itself needs a GPU-capable environment with TEDDY's code and weights; it has not been run
in CI. The workload YAML is checked for the fields `databricks air run -h config` documents (CLI v1.20.0).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parents[1] / "scripts" / "finetune_teddy"
sys.path.insert(0, str(HERE))
import train_teddy_classifier as ft  # noqa: E402

WORKLOAD = yaml.safe_load((HERE / "workload.yaml").read_text())


def test_module_imports_without_torch():
    assert not hasattr(ft, "torch")                     # torch is imported inside the training functions only
    assert ft.UPSTREAM_REVISION == "8c880aa67674042ccdc254bccb5de081092a48bf"


def test_workload_fields_are_documented_and_safe():
    documented = {"experiment_name", "compute", "environment", "command", "containers", "env_variables", "secrets",
                  "code_source", "max_retries", "timeout_minutes", "idempotency_token", "parameters",
                  "mlflow_run_name", "mlflow_experiment_directory", "mlflow_artifact_location", "permissions",
                  "usage_policy_name", "usage_policy_id"}
    assert set(WORKLOAD) <= documented
    assert WORKLOAD["compute"] == {"num_accelerators": 1, "accelerator_type": "GPU_1xA10"}
    assert WORKLOAD["max_retries"] == 0
    deps = WORKLOAD["environment"]["dependencies"]
    assert "transformers==4.41.0" in deps and not any(d.startswith("torch") for d in deps)
    assert set(WORKLOAD["parameters"]) <= set(ft.DEFAULTS)
    assert "train_teddy_classifier.py" in WORKLOAD["command"]


def test_hyperparameters_layering(tmp_path, monkeypatch):
    f = tmp_path / "hp.yaml"
    f.write_text(yaml.safe_dump({**WORKLOAD["parameters"], "epochs": 7}))
    hp = ft.load_hyperparameters(str(f))
    assert hp["epochs"] == 7 and hp["unfreeze_layers"] == 2
    monkeypatch.setenv("HYPERPARAMETERS_PATH", str(f))
    assert ft.load_hyperparameters(smoke=True)["epochs"] == ft.SMOKE["epochs"]
    assert ft.load_hyperparameters(overrides={"epochs": 2})["epochs"] == 2
    f.write_text(yaml.safe_dump({"epoch": 1}))
    with pytest.raises(ValueError, match="Unknown"):
        ft.load_hyperparameters(str(f))
    monkeypatch.delenv("HYPERPARAMETERS_PATH")
    with pytest.raises(ValueError, match="snapshot_dir"):
        ft.load_hyperparameters()


def test_stratified_subsample_keeps_every_label():
    labels = ["a"] * 90 + ["b"] * 9 + ["c"]
    keep = ft.stratified_subsample(labels, 20, seed=0)
    kept = {labels[i] for i in keep}
    assert len(keep) <= 20 and kept == {"a", "b", "c"}
    assert len(ft.stratified_subsample(labels, None, 0)) == 100


def test_encode_labels_drops_rare_and_marks_unseen():
    classes, ref, held = ft.encode_labels(["a"] * 5 + ["b"] * 5 + ["c"], ["a", "c", "d"], min_cells=5)
    assert classes == ["a", "b"] and ref[-1] == -1 and list(held) == [0, -1, -1]
    with pytest.raises(ValueError):
        ft.encode_labels(["a"] * 5, ["a"], min_cells=5)


def test_trainable_parameter_names():
    names = [f"encoder.layers.{i}.linear1.weight" for i in range(12)] + ["embeddings.weight", "cls_decoder._decoder.0.weight"]
    picked = ft.trainable_parameter_names(names, nlayers=12, unfreeze_layers=2)
    assert picked == ["encoder.layers.10.linear1.weight", "encoder.layers.11.linear1.weight",
                      "cls_decoder._decoder.0.weight"]
    assert ft.trainable_parameter_names(names, 12, 0) == ["cls_decoder._decoder.0.weight"]


def test_knn_vote_and_accuracy():
    ref = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    pred = ft.knn_vote(ref, np.array([0, 0, 1, 1]), np.array([[1.0, 0.05], [0.05, 1.0]]), k=2)
    assert list(pred) == [0, 1]
    assert ft.accuracy([0, 1, 0], [0, 1, -1]) == pytest.approx(2 / 3)


def test_training_loop_on_a_stand_in_model(tmp_path, monkeypatch):
    """The loop (parity check, frozen baselines, freezing, optimiser, evaluation, checkpoint) on a tiny stand-in
    with TeddyGModel's interface (model.py:103-163, 197-264). Skipped without torch (CI does not install it)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    nn = torch.nn

    class Config:
        d_model, nlayers, layer_activation, dropout = 8, 3, "gelu", 0.0

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class Head(nn.Module):                               # same shape as upstream ClassificationHead
        def __init__(self, config, n_cls, nlayers):
            super().__init__()
            self._decoder = nn.ModuleList([nn.Linear(config.d_model, n_cls)])

        def forward(self, x):
            for m in self._decoder:
                x = m(x)
            return {"output": x}

    class StandIn(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embeddings = nn.Embedding(50, config.d_model)
            layer = nn.TransformerEncoderLayer(config.d_model, 2, 16, 0.0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, config.nlayers, enable_nested_tensor=False)
            self.n_cls = 0

        @classmethod
        def from_pretrained(cls, path, config):
            torch.manual_seed(0)
            return cls(config)

        def add_classification_head(self, d_model, n_cls, nlayers):
            self.cls_decoder = Head(self.config, n_cls, nlayers)
            self.n_cls = n_cls

        def forward(self, gene_ids, attention_mask=None, **kw):
            x = self.embeddings(gene_ids.long())
            pad = ~attention_mask.bool()
            for layer in self.encoder.layers:
                x = layer(src=x, src_key_padding_mask=pad)
            out = {"cell_emb": x[:, 0, :]}
            if self.n_cls > 1:
                out["cls_output"] = self.cls_decoder(out["cell_emb"])
            return out

    class Tok:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    up = {"get_architecture": lambda d: "StandIn", "GeneTokenizer": Tok,
          "model_dict": {"StandIn": {"config_cls": Config, "model_cls": StandIn}}}
    labels = {"ref": ["A"] * 12 + ["B"] * 12, "held": ["A"] * 4 + ["B"] * 4 + ["C"]}

    class Cells:
        def __init__(self, split):
            import pandas as pd
            self.split, self.obs = split, pd.DataFrame({"cell_type": labels[split]})
            self.n_obs = len(labels[split])

        def __getitem__(self, rows):
            c = Cells(self.split)
            c.obs = self.obs.iloc[rows].reset_index(drop=True)
            c.n_obs = len(c.obs)
            return c

        def copy(self):
            return self

    def fake_tokenize(a, up_, tok, medians, max_seq_len, chunk=512):
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(1, 50, (a.n_obs, 6), generator=g)
        ids[:, 0] = torch.tensor([1 if t == "A" else 2 for t in a.obs["cell_type"]])   # label signal in token 0
        mask = torch.ones_like(ids)
        mask[:, 4:] = 0
        ids[mask == 0] = 0
        return ids, mask, 1.0

    monkeypatch.setattr(ft, "import_upstream", lambda d: up)
    monkeypatch.setattr(ft, "read_cells", lambda p: Cells("ref" if "ref" in p else "held"))
    monkeypatch.setattr(ft, "tokenize_cells", fake_tokenize)
    hp = ft.load_hyperparameters(overrides={"snapshot_dir": str(tmp_path), "reference_h5ad": "ref.h5ad",
                                            "heldout_h5ad": "held.h5ad", "output_dir": str(tmp_path / "out"),
                                            "epochs": 2, "batch_size": 4, "min_cells_per_label": 2, "knn_k": 3,
                                            "unfreeze_layers": 1, "parity_cells": 3})
    summary = ft.train(hp, use_mlflow=False)
    assert summary["batching_parity_worst_cosine"] >= ft.PARITY_COSINE
    assert summary["classes"] == ["A", "B"] and summary["heldout_cells_unseen_label"] == 1
    assert len(summary["history"]) == 2 and 0 <= summary["heldout_accuracy_finetuned"] <= 8 / 9
    saved = torch.load(tmp_path / "out" / "teddy_cell_type_head.pt")
    trained = set(saved["trainable_state"])
    assert trained and all(n.startswith(("cls_decoder.", "encoder.layers.2.")) for n in trained)
    assert (tmp_path / "out" / "run_summary.json").exists()
