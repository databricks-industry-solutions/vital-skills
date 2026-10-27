"""Offline tests for the practical checks TRIAL (eval/practical_checks.py), not adopted.

One passing and one failing case per check, the notebook line numbers the reasons cite, and
the trial results on the five committed 2026-10-27 exports (models/teddy/runs_20261026/),
which PRACTICAL_TRIAL.md reports. Needs mlflow-skinny (tests/requirements.txt); no
workspace, no network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = SKILL_DIR / "eval"
ROUND = EVAL_DIR / "models" / "teddy" / "runs_20261026"


@pytest.fixture(scope="module")
def pc():
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location("owm_practical_checks", EVAL_DIR / "practical_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nb(*cells):
    """Exported-notebook text from Python cells."""
    return "# Databricks notebook source\n" + "\n\n# COMMAND ----------\n\n".join(cells) + "\n"


def _record(endpoint_config=None, **outcome):
    """A run record with outcome checks given as keyword arguments (outcome_endpoint_ready=True)."""
    record = {"outcome": {k.replace("outcome_", "outcome/", 1): {"passed": v} for k, v in outcome.items()}}
    if endpoint_config is not None:
        record["endpoint_config"] = endpoint_config
    return record


READY = _record(outcome_endpoint_ready=True)


def _check(pc, name, source, record=READY, task_id="teddy-001"):
    task = pc.scorers.TASKS[task_id]
    return pc.CHECK_FUNCTIONS[name](pc.PracticalCode(source), source, record, task)


# ---- metadata -----------------------------------------------------------------------------

def test_every_check_has_goal_severity_and_alternatives(pc):
    assert set(pc.CHECKS) == set(pc.CHECK_FUNCTIONS)
    for name, meta in pc.CHECKS.items():
        assert meta["severity"] in pc.SEVERITIES, name
        assert meta["goal"] and meta["accepts"] and meta["fails"] and meta["evidence"], name


def test_similarity_search_only_for_search_task(pc):
    assert "similarity_search_works" not in pc.checks_for(pc.scorers.TASKS["teddy-001"])
    assert "similarity_search_works" in pc.checks_for(pc.scorers.TASKS["teddy-002"])


def test_trial_is_not_wired_into_the_scoring(pc):
    assert "practical" not in (EVAL_DIR / "score_open_weight_models.py").read_text()
    names = {s.name for s in pc.scorers.scorers}
    assert not any("practical" in n for n in names)
    assert pc.scorers.metric_group("practical/gpu_serving") == "replicated"  # no group: not adopted


def test_reason_line_numbers_are_notebook_lines(pc):
    source = _nb("# MAGIC %md\n# MAGIC # Title", "%pip install x\nimport os",
                 "from huggingface_hub import snapshot_download\n\nsnapshot_download('r', local_dir='/tmp/x')")
    line = next(i for i, t in enumerate(source.splitlines(), 1) if t.startswith("snapshot_download("))
    ok, reason = _check(pc, "download_location_ok", source)
    assert ok and f"line {line} " in reason


# ---- download_location_ok -------------------------------------------------------------------

@pytest.mark.parametrize("cell", [
    "snapshot_download('r', local_dir='/tmp/teddy')",
    "import tempfile\nd = tempfile.mkdtemp()\nsnapshot_download('r', local_dir=f'{d}/TEDDY')",
    "import tempfile\nwith tempfile.TemporaryDirectory() as d:\n    snapshot_download('r', local_dir=d)",
    "V = '/Volumes/c/s/v/files'\nsnapshot_download(repo_id='r', local_dir=V)",
    "import subprocess, sys\nTMP = '/tmp/t'\nscript = '''\nimport sys\nfrom huggingface_hub import snapshot_download\n"
    "snapshot_download('r', local_dir=sys.argv[1])\n'''\nsubprocess.run([sys.executable, '-c', script, f'{TMP}/snap'])",
])
def test_download_location_accepted(pc, cell):
    assert _check(pc, "download_location_ok", _nb("from huggingface_hub import snapshot_download", cell))[0]


def test_shell_download_cell_counts_but_markdown_mention_does_not(pc):
    sh = _nb("# MAGIC %sh\n# MAGIC hf download Merck/TEDDY --local-dir /tmp/teddy --revision 8c880aa")
    md = _nb("# MAGIC %md\n# MAGIC Run `hf download Merck/TEDDY --local-dir /tmp/teddy`.")
    assert _check(pc, "download_location_ok", sh)[0] and _check(pc, "xet_download_ok", sh)[0]
    assert not _check(pc, "download_location_ok", md)[0]


@pytest.mark.parametrize("cell", [
    "snapshot_download('r', local_dir='/local_disk0/teddy')",
    "snapshot_download('r', local_dir='/tmp/t')\nSTAGE = '/local_disk0/stage'",
    "snapshot_download('r')",                                  # default HF cache: not stated
    "import os",                                               # no download
])
def test_download_location_rejected(pc, cell):
    assert not _check(pc, "download_location_ok", _nb("from huggingface_hub import snapshot_download", cell))[0]


# ---- xet_download_ok ----------------------------------------------------------------------

DOWNLOAD = "from huggingface_hub import snapshot_download\nsnapshot_download('r', local_dir='/tmp/t')"
LATER_LOAD = "model = Model.from_pretrained('/tmp/t/70M')"


@pytest.mark.parametrize("source,record", [
    (_nb("import os\nos.environ['HF_HUB_DISABLE_XET'] = '1'", DOWNLOAD), _record(outcome_endpoint_ready=False)),
    (_nb("import subprocess, sys\nsubprocess.check_call([sys.executable, '-c', "
         "\"from huggingface_hub import snapshot_download; snapshot_download('r', local_dir='/tmp/t')\"])"),
     _record(outcome_endpoint_ready=False)),
    (_nb(DOWNLOAD, LATER_LOAD), READY),
])
def test_xet_download_accepted(pc, source, record):
    assert _check(pc, "xet_download_ok", source, record)[0]


@pytest.mark.parametrize("source,record", [
    (_nb(DOWNLOAD, LATER_LOAD), _record(outcome_endpoint_ready=False)),   # files used, endpoint never READY
    (_nb(DOWNLOAD), READY),                                                # nothing uses the files
    (_nb("# os.environ['HF_HUB_DISABLE_XET'] = '1'", DOWNLOAD), _record(outcome_endpoint_ready=False)),  # comment
])
def test_xet_download_rejected(pc, source, record):
    assert not _check(pc, "xet_download_ok", source, record)[0]


# ---- model_loads_with_pinned_deps -----------------------------------------------------------

def _logged(reqs: str) -> str:
    return f"import mlflow\nmlflow.pyfunc.log_model(name='m', python_model=W(), pip_requirements={reqs})"


@pytest.mark.parametrize("source", [
    _nb(_logged("['torch', 'transformers==4.41.0']")),
    _nb('%pip install "transformers>=4.41,<5"',
        "import transformers\n" + _logged("[f'transformers=={transformers.__version__}']")),
    _nb("# MAGIC %pip install transformers~=4.41", "import transformers\nmlflow.pyfunc.log_model(name='m')"),
    _nb("import torch\nfrom torch import nn\nclass Enc(nn.Module):\n    pass", _logged("['torch==2.3.1']")),
])
def test_pinned_deps_accepted(pc, source):
    assert _check(pc, "model_loads_with_pinned_deps", source)[0]


@pytest.mark.parametrize("source,record", [
    (_nb(_logged("['transformers>=4.41']")), READY),
    (_nb(_logged("['transformers']")), READY),
    (_nb("import transformers\n" + _logged("[f'transformers=={transformers.__version__}']")), READY),  # no %pip limit
    (_nb("from teddy.models.model_directory import model_dict", _logged("['torch']")), READY),          # never pinned
    (_nb(_logged("['transformers==4.41.0']")), _record(outcome_endpoint_ready=False)),
])
def test_pinned_deps_rejected(pc, source, record):
    assert not _check(pc, "model_loads_with_pinned_deps", source, record)[0]


# ---- fidelity_verified ----------------------------------------------------------------------

@pytest.mark.parametrize("cells", [
    ("from teddy.data_processing.tokenization.tokenization import _build_batch_tensors",
     "ids = _build_batch_tensors(X, toks, args)\nref = run(ids)\n"
     "cos = float(ref @ E[0] / (np.linalg.norm(ref) * np.linalg.norm(E[0])))\nassert cos > 0.9999"),
    ("class W(mlflow.pyfunc.PythonModel):\n    def load_context(self, c):\n"
     "        from teddy.models.model_directory import model_dict\n        self.model = model_dict['g']['model_cls']()",
     "w = W()\nref = w.model(gene_ids=ids)\nd = np.abs(pred - ref['cell_emb']).max()\nassert d < 1e-3"),
])
def test_fidelity_accepted(pc, cells):
    assert _check(pc, "fidelity_verified", _nb(*cells))[0]


@pytest.mark.parametrize("cells", [
    # re-implementation compared only with itself
    ("from torch import nn\nclass Enc(nn.Module):\n    pass\nmodel = Enc()",
     "out = W().predict(None, x)\nd = float(np.abs(np.stack(out.e) - emb[:2]).max())"),
    # endpoint vs the same wrapper in the notebook
    ("from teddy.models.model_directory import model_dict",
     "cos = (E_ep * E_loc).sum(1) / (np.linalg.norm(E_ep, axis=1) * np.linalg.norm(E_loc, axis=1))"),
    # a raw forward without any comparison
    ("from teddy.models.model_directory import model_dict", "o = embedder.model(gene_ids=ids)\nprint(o.keys())"),
])
def test_fidelity_rejected(pc, cells):
    assert not _check(pc, "fidelity_verified", _nb(*cells))[0]


# ---- signature_and_example ------------------------------------------------------------------

@pytest.mark.parametrize("build", ["infer_signature(x, y)", "ModelSignature(inputs=Schema([]))"])
def test_signature_and_example_accepted(pc, build):
    source = _nb(f"import mlflow\nsig = {build}\nmlflow.pyfunc.log_model(name='m', signature=sig, input_example=x)")
    assert _check(pc, "signature_and_example", source)[0]


@pytest.mark.parametrize("kwargs", ["signature=sig", "input_example=x", "signature=None, input_example=x", ""])
def test_signature_and_example_rejected(pc, kwargs):
    source = _nb(f"import mlflow\nsig = infer_signature(x, y)\nmlflow.pyfunc.log_model(name='m', {kwargs})")
    assert not _check(pc, "signature_and_example", source)[0]


# ---- registered_in_uc (reuses the new check) --------------------------------------------------

def test_registered_in_uc(pc):
    three = _nb("import mlflow\nmlflow.register_model('runs:/x/m', 'c.s.m')")
    two = _nb("import mlflow\nmlflow.register_model('runs:/x/m', 's.m')")
    assert _check(pc, "registered_in_uc", three)[0]
    assert not _check(pc, "registered_in_uc", two)[0]


# ---- record-based checks ----------------------------------------------------------------------

def test_serves_real_cell_data(pc):
    ok = _record(outcome_accepts_expression_input=True, outcome_returns_valid_embeddings=True)
    tokens = _record(outcome_accepts_expression_input=False, outcome_returns_valid_embeddings=True)
    assert _check(pc, "serves_real_cell_data", "", ok)[0]
    assert not _check(pc, "serves_real_cell_data", "", tokens)[0]
    assert not _check(pc, "serves_real_cell_data", "", READY)[0]          # not measured


def test_similarity_search_works(pc):
    ok = _record(outcome_index_ready=True, outcome_nn_returns_cell_type=True)
    no_type = _record(outcome_index_ready=True, outcome_nn_returns_cell_type=False)
    assert _check(pc, "similarity_search_works", "", ok, "teddy-002")[0]
    assert not _check(pc, "similarity_search_works", "", no_type, "teddy-002")[0]


def test_gpu_serving(pc):
    gpu = _record({"served_entities": [{"workload_type": "GPU_SMALL"}]})
    cpu = _record({"served_entities": [{"workload_type": "CPU"}]})
    assert _check(pc, "gpu_serving", "", gpu)[0]
    ok, reason = _check(pc, "gpu_serving", "", cpu)
    assert not ok and "15 s per cell" in reason
    code = _nb("from databricks.sdk.service.serving import ServingModelWorkloadType",
               "e = ServedEntityInput(workload_type=ServingModelWorkloadType.GPU_SMALL)")
    assert _check(pc, "gpu_serving", code, {})[0]                         # no endpoint_config: code fallback
    assert not _check(pc, "gpu_serving", _nb("t = 'CPU'"), {})[0]


# ---- governance_logging (reuses the new checks) -------------------------------------------------

def test_governance_logging(pc):
    gateway = ("from databricks.sdk import WorkspaceClient\nw = WorkspaceClient()\n"
               "g = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True){usage})\n"
               "w.serving_endpoints.create(name='e', ai_gateway=g)")
    both = gateway.format(usage=", usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True)")
    assert _check(pc, "governance_logging", _nb(both))[0]
    assert not _check(pc, "governance_logging", _nb(gateway.format(usage="")))[0]


# ---- provenance_recorded ----------------------------------------------------------------------

PINNED = "from huggingface_hub import snapshot_download\nsnapshot_download('r', revision='8c880aa', local_dir='/tmp/t')"


@pytest.mark.parametrize("cells", [
    (PINNED, "import mlflow\nmlflow.set_tags({'license': 'apache-2.0'})"),
    (PINNED, "import mlflow\nmlflow.log_params({'hf_revision': R, 'license': 'apache-2.0'})"),
    (PINNED, "import json\nm = {'model': {}, 'license': {'code': 'Apache-2.0'}}\njson.dump(m, open(P, 'w'))"),
    (PINNED, "c = MlflowClient()\nfor k, v in {'license': 'Apache-2.0', 'variant': '70M'}.items():\n"
             "    c.set_model_version_tag(NAME, '1', k, v)"),
    (PINNED, "c = MlflowClient()\nc.update_registered_model(NAME, description='TEDDY (Apache-2.0)')"),
    ("import subprocess, sys\nscript = '''\nfrom huggingface_hub import HfApi, snapshot_download\n"
     "sha = HfApi().model_info('r').sha\nsnapshot_download('r', revision=sha, local_dir='/tmp/t')\n'''\n"
     "subprocess.run([sys.executable, '-c', script])", "import mlflow\nmlflow.set_tag('license', 'MIT')"),
])
def test_provenance_accepted(pc, cells):
    assert _check(pc, "provenance_recorded", _nb(*cells))[0]


@pytest.mark.parametrize("cells", [
    (PINNED.replace("revision='8c880aa', ", ""), "import mlflow\nmlflow.set_tags({'license': 'apache-2.0'})"),
    (PINNED.replace("'8c880aa'", "'main'"), "import mlflow\nmlflow.set_tags({'license': 'apache-2.0'})"),
    (PINNED, "# MAGIC %md\n# MAGIC Apache-2.0 licence", "# license: apache-2.0\nx = 1"),
])
def test_provenance_rejected(pc, cells):
    assert not _check(pc, "provenance_recorded", _nb(*cells))[0]


# ---- the committed 2026-10-27 exports ------------------------------------------------------------

EXPECTED_FAILS = {   # label, task -> practical checks that fail (PRACTICAL_TRIAL.md)
    ("baseline", "teddy-001"): {"serves_real_cell_data", "gpu_serving", "governance_logging"},
    ("baseline attempt 1", "teddy-001"): {"fidelity_verified", "serves_real_cell_data", "gpu_serving",
                                          "governance_logging", "provenance_recorded"},
    ("baseline", "teddy-002"): {"fidelity_verified", "gpu_serving", "governance_logging", "provenance_recorded"},
    ("with_skill", "teddy-001"): set(),
    ("with_skill", "teddy-002"): {"fidelity_verified"},
}


def test_trial_results_on_committed_exports(pc):
    samples = pc.round_samples(ROUND)
    assert {(label, tid) for label, tid, _, _ in samples} == set(EXPECTED_FAILS)
    for label, task_id, record, source in samples:
        scores = pc.practical_scores(pc.scorers.TASKS[task_id], source, record)
        failed = {name[len(pc.PRACTICAL_PREFIX):] for name, ok in scores.items() if not ok}
        assert failed == EXPECTED_FAILS[(label, task_id)], (label, task_id)


def test_failed_by_severity(pc):
    failed = pc.failed_by_severity({"practical/gpu_serving": False, "practical/xet_download_ok": False,
                                    "practical/registered_in_uc": True})
    assert failed == {"critical": ["practical/xet_download_ok"], "major": ["practical/gpu_serving"], "minor": []}
