"""Offline tests for the practical checks TRIAL (eval/practical_checks.py), not adopted.

Passing and failing cases per check (including the false positives / negatives found in
review of ff3a64b), the notebook line numbers the reasons cite, the separately reported
guideline, and the trial results on the five committed 2026-10-27 exports
(models/teddy/runs_20261026/), which PRACTICAL_TRIAL.md reports. Needs mlflow-skinny
(tests/requirements.txt); no workspace, no network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = SKILL_DIR / "eval"
ROUND = EVAL_DIR / "models" / "teddy" / "runs_20261026"
# Run folders (exported notebooks, run records, per-run scores) are not kept in the repository: they are
# git-ignored and stay with whoever ran the round. Tests that replay them run where the folders are
# present and are skipped elsewhere, e.g. on CI.
NEEDS_ROUND = pytest.mark.skipif(not ROUND.is_dir(),
                                 reason=f"run records not in this checkout ({ROUND.name} is kept outside the repository)")
SHA = "8c880aa67674042ccdc254bccb5de081092a48bf"


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


READY = _record(outcome_endpoint_ready=True, outcome_serves_recorded_version=True)
NOT_READY = _record(outcome_endpoint_ready=False, outcome_serves_recorded_version=False)


def _check(pc, name, source, record=READY, task_id="teddy-001"):
    task = pc.scorers.TASKS[task_id]
    return pc.CHECK_FUNCTIONS[name](pc.PracticalCode(source), source, record, task)


# ---- metadata -----------------------------------------------------------------------------

def test_every_check_has_goal_severity_and_alternatives(pc):
    assert set(pc.CHECKS) == set(pc.CHECK_FUNCTIONS)
    for name, meta in pc.CHECKS.items():
        assert meta["severity"] in pc.SEVERITIES, name
        assert meta["goal"] and meta["accepts"] and meta["fails"] and meta["evidence"], name
    assert set(pc.GUIDELINES) == set(pc.GUIDELINE_FUNCTIONS) == {"gpu_serving"}
    assert "gpu_serving" not in pc.CHECKS


def test_severities_follow_the_answer_key(pc):
    """The split outcome checks keep expectations.json's severities."""
    key = pc.scorers.TASKS["teddy-001"]["checks"]
    assert pc.CHECKS["serves_valid_embeddings"]["severity"] == key["outcome/returns_valid_embeddings"]["severity"]
    assert pc.CHECKS["serves_expression_input"]["severity"] == key["outcome/accepts_expression_input"]["severity"]


def test_similarity_search_only_for_search_task(pc):
    assert "similarity_search_works" not in pc.checks_for(pc.scorers.TASKS["teddy-001"])
    assert "similarity_search_works" in pc.checks_for(pc.scorers.TASKS["teddy-002"])


def test_trial_is_not_wired_into_the_scoring(pc):
    assert "practical" not in (EVAL_DIR / "score_open_weight_models.py").read_text()
    names = {s.name for s in pc.scorers.scorers}
    assert not any("practical" in n or "guideline" in n for n in names)


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
    # an unrelated /local_disk0 string is not a destination
    "snapshot_download('r', local_dir='/tmp/t')\nprint('not using /local_disk0 on serverless')\nNOTE = '/local_disk0'",
])
def test_download_location_accepted(pc, cell):
    assert _check(pc, "download_location_ok", _nb("from huggingface_hub import snapshot_download", cell))[0]


@pytest.mark.parametrize("cell", [
    "snapshot_download('r', local_dir='/local_disk0/teddy')",
    "import shutil\nsnapshot_download('r', local_dir='/tmp/t')\nshutil.copytree('/tmp/t', '/local_disk0/stage')",
    "snapshot_download('r')",                                  # default HF cache: not stated
    "import os",                                               # no download
])
def test_download_location_rejected(pc, cell):
    assert not _check(pc, "download_location_ok", _nb("from huggingface_hub import snapshot_download", cell))[0]


def test_shell_download_cell_counts_but_markdown_mention_does_not(pc):
    sh = _nb(f"# MAGIC %sh\n# MAGIC export HF_HUB_DISABLE_XET=1\n# MAGIC hf download Merck/TEDDY --local-dir /tmp/teddy --revision {SHA}")
    md = _nb("# MAGIC %md\n# MAGIC Run `hf download Merck/TEDDY --local-dir /tmp/teddy`.")
    assert _check(pc, "download_location_ok", sh)[0] and _check(pc, "xet_download_ok", sh, NOT_READY)[0]
    assert not _check(pc, "download_location_ok", md)[0]


# ---- xet_download_ok ----------------------------------------------------------------------

DOWNLOAD = "snapshot_download('r', local_dir='/tmp/t')"
IMPORT = "from huggingface_hub import snapshot_download"
LATER_LOAD = "model = Model.from_pretrained('/tmp/t/70M')"
TEDDY_MD = ("import subprocess, sys, os\nenv = {**os.environ, 'HF_HUB_DISABLE_XET': '1'}\n"
            "subprocess.check_call([sys.executable, '-c', \"from huggingface_hub import snapshot_download; "
            "snapshot_download('r', local_dir='/tmp/t')\"], env=env)")


@pytest.mark.parametrize("source", [
    _nb(TEDDY_MD),                                                                  # teddy.md pattern: env dict
    _nb("import subprocess, sys\nscript = '''\nimport os\nos.environ['HF_HUB_DISABLE_XET'] = '1'\n"
        "from huggingface_hub import snapshot_download\nsnapshot_download('r', local_dir='/tmp/t')\n'''\n"
        "subprocess.run([sys.executable, '-c', script])"),                         # set inside the script first
    _nb("import os\nenv = dict(os.environ)\nenv.update({'HF_HUB_DISABLE_XET': '1'})",
        "import subprocess, sys\nsubprocess.run([sys.executable, '-c', \"from huggingface_hub import "
        "snapshot_download; snapshot_download('r', local_dir='/tmp/t')\"], env=env)"),  # env updated by name
    _nb("import os\nos.environ['HF_HUB_DISABLE_XET'] = '1'", IMPORT + "\n" + DOWNLOAD),  # in-process, before import
])
def test_xet_setting_accepted(pc, source):
    assert _check(pc, "xet_download_ok", source, NOT_READY)[0]


@pytest.mark.parametrize("source", [
    _nb(IMPORT, DOWNLOAD, "import os\nos.environ['HF_HUB_DISABLE_XET'] = '1'"),        # set after the download
    _nb(IMPORT, "import os\nos.environ['HF_HUB_DISABLE_XET'] = '1'", DOWNLOAD),        # set after the import
    _nb("import subprocess, sys\nsubprocess.check_call([sys.executable, '-c', \"from huggingface_hub import "
        "snapshot_download; snapshot_download('r', local_dir='/tmp/t')\"])"),         # bare subprocess
    _nb("# os.environ['HF_HUB_DISABLE_XET'] = '1'", IMPORT + "\n" + DOWNLOAD),         # only a comment
])
def test_xet_setting_rejected_without_recorded_evidence(pc, source):
    assert not _check(pc, "xet_download_ok", source, NOT_READY)[0]


@pytest.mark.parametrize("source", [
    _nb("import os\nos.environ['HF_HUB_DISABLE_XET'] = 'TRUE'", IMPORT + "\n" + DOWNLOAD),
    _nb("import os\nos.environ.setdefault('HF_HUB_DISABLE_XET', 'on')", IMPORT + "\n" + DOWNLOAD),
    _nb("# MAGIC %sh\n# MAGIC HF_HUB_DISABLE_XET=yes hf download Merck/TEDDY --local-dir /tmp/t"),
])
def test_xet_disabling_values_accepted(pc, source):
    assert _check(pc, "xet_download_ok", source, NOT_READY)[0]


@pytest.mark.parametrize("source", [   # set, but to a value that does not disable Xet (review case: '0')
    _nb("import os\nos.environ['HF_HUB_DISABLE_XET'] = '0'", IMPORT + "\n" + DOWNLOAD),
    _nb("import os\nos.environ.setdefault('HF_HUB_DISABLE_XET', 'false')", IMPORT + "\n" + DOWNLOAD),
    _nb(TEDDY_MD.replace("'HF_HUB_DISABLE_XET': '1'", "'HF_HUB_DISABLE_XET': '0'")),
    _nb("import os\nenv = dict(os.environ)\nenv.update({'HF_HUB_DISABLE_XET': 'no'})",
        "import subprocess, sys\nsubprocess.run([sys.executable, '-c', \"from huggingface_hub import "
        "snapshot_download; snapshot_download('r', local_dir='/tmp/t')\"], env=env)"),
    _nb("# MAGIC %sh\n# MAGIC export HF_HUB_DISABLE_XET=0\n# MAGIC hf download Merck/TEDDY --local-dir /tmp/t"),
    _nb("# MAGIC %env HF_HUB_DISABLE_XET=0", IMPORT + "\n" + DOWNLOAD),
])
def test_xet_non_disabling_values_rejected(pc, source):
    assert not _check(pc, "xet_download_ok", source, NOT_READY)[0]


def test_xet_recorded_evidence(pc):
    source = _nb(IMPORT, DOWNLOAD, LATER_LOAD)
    ok, reason = _check(pc, "xet_download_ok", source, READY)
    assert ok and "recorded evidence" in reason
    assert not _check(pc, "xet_download_ok", source, NOT_READY)[0]                  # files used, never READY
    assert not _check(pc, "xet_download_ok", _nb(IMPORT, DOWNLOAD), READY)[0]       # nothing uses the files
    # Codex review of 117ac56: a later load of files from elsewhere is not the download's files
    other = _nb(IMPORT, DOWNLOAD, "model = Model.from_pretrained('/Volumes/c/s/v/elsewhere')")
    assert not _check(pc, "xet_download_ok", other, READY)[0]
    # ... but a copy of the download, loaded through a function parameter, is
    staged = _nb(IMPORT, "import shutil\nlocal = snapshot_download('r', local_dir='/tmp/t')\nshutil.copytree(local, '/Volumes/c/s/v/r')",
                 "def load(d):\n    return Model.from_pretrained(f'{d}/70M')\nm = load('/Volumes/c/s/v/r')")
    assert _check(pc, "xet_download_ok", staged, READY)[0]


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
    (_nb(_logged("['transformers==4.41.0']")), NOT_READY),
])
def test_pinned_deps_rejected(pc, source, record):
    assert not _check(pc, "model_loads_with_pinned_deps", source, record)[0]


# ---- fidelity_verified ----------------------------------------------------------------------

UPSTREAM_TOKENIZE = "from teddy.data_processing.tokenization.tokenization import _build_batch_tensors"


@pytest.mark.parametrize("cells", [
    # cosine of the wrapper's predict vs a forward on upstream token ids, in one cell
    (UPSTREAM_TOKENIZE,
     "E = wrapper.predict(None, x)\nids = _build_batch_tensors(X, toks, args)\nref = run(ids)\n"
     "cos = float(ref @ E[0] / (np.linalg.norm(ref) * np.linalg.norm(E[0])))"),
    # the same split across cells: reference in one cell, comparison two cells later
    (UPSTREAM_TOKENIZE, "ref = run(_build_batch_tensors(X, toks, args))", "pred = wrapper.predict(None, x)",
     "assert np.allclose(pred, ref, atol=1e-4)"),
    # wrapper vs a direct forward of the upstream model, across cells, through a loop
    ("class W(mlflow.pyfunc.PythonModel):\n    def load_context(self, c):\n"
     "        from teddy.models.model_directory import model_dict\n        self.model = model_dict['g']['model_cls']()",
     "w = W()\npred = w.predict(None, df)",
     "for j, s in enumerate(cells):\n    ref = w.model(gene_ids=s)\n    d = np.abs(pred[j] - ref['cell_emb']).max()"),
    # endpoint output vs an upstream reference computed by a notebook function
    (UPSTREAM_TOKENIZE, "def reference(X):\n    return run(_build_batch_tensors(X, toks, args))",
     "E_ep = w.serving_endpoints.query(name=N, dataframe_records=r).predictions\nref = reference(X)\n"
     "ok = np.allclose(np.array(E_ep), ref)"),
])
def test_fidelity_accepted(pc, cells):
    assert _check(pc, "fidelity_verified", _nb(*cells))[0]


@pytest.mark.parametrize("cells", [
    # an upstream call next to a comparison of two local values (review false positive)
    ("import teddy", "v = teddy.version()\nassert np.allclose(local_a, local_b)"),
    # an upstream call that does not feed the comparison
    (UPSTREAM_TOKENIZE, "ids = _build_batch_tensors(X, toks, args)\npred = wrapper.predict(None, x)\n"
     "assert np.allclose(pred, other)"),
    # re-implementation compared only with itself
    ("from torch import nn\nclass Enc(nn.Module):\n    pass\nmodel = Enc()",
     "emb = model(ids)\nout = W().predict(None, x)\nd = float(np.abs(np.stack(out.e) - emb[:2]).max())"),
    # endpoint vs the same wrapper in the notebook
    ("from teddy.models.model_directory import model_dict",
     "E_ep = w.serving_endpoints.query(name=N).predictions\nE_loc = wrapper.predict(None, x)\n"
     "cos = (E_ep * E_loc).sum(1) / (np.linalg.norm(E_ep, axis=1) * np.linalg.norm(E_loc, axis=1))"),
    # a raw forward without any comparison
    ("from teddy.models.model_directory import model_dict", "o = embedder.model(gene_ids=ids)\nprint(o.keys())"),
])
def test_fidelity_rejected(pc, cells):
    assert not _check(pc, "fidelity_verified", _nb(*cells))[0]


# ---- signature_and_example ------------------------------------------------------------------

@pytest.mark.parametrize("build", ["infer_signature(x, y)", "ModelSignature(inputs=Schema([]))",
                                   "mlflow.models.infer_signature(x, y)"])
def test_signature_and_example_accepted(pc, build):
    source = _nb(f"import mlflow\nsig = {build}\nmlflow.pyfunc.log_model(name='m', signature=sig, input_example=x)")
    assert _check(pc, "signature_and_example", source, NOT_READY)[0]


def test_unrecognised_signature_needs_recorded_evidence(pc):
    source = _nb("import mlflow\nsig = make_sig()\nmlflow.pyfunc.log_model(name='m', signature=sig, input_example=x)")
    assert not _check(pc, "signature_and_example", source, NOT_READY)[0]
    assert _check(pc, "signature_and_example", source, READY)[0]


@pytest.mark.parametrize("kwargs", ["signature=sig", "input_example=x", "signature=None, input_example=x", ""])
def test_signature_and_example_rejected(pc, kwargs):
    source = _nb(f"import mlflow\nsig = infer_signature(x, y)\nmlflow.pyfunc.log_model(name='m', {kwargs})")
    assert not _check(pc, "signature_and_example", source, READY)[0]


# ---- registered_in_uc / existing_resources_checked (reuse the new checks) ------------------------

def test_registered_in_uc(pc):
    three = _nb("import mlflow\nmlflow.register_model('runs:/x/m', 'c.s.m')")
    two = _nb("import mlflow\nmlflow.register_model('runs:/x/m', 's.m')")
    assert _check(pc, "registered_in_uc", three)[0]
    assert not _check(pc, "registered_in_uc", two)[0]


def test_existing_resources_checked(pc):
    looked = _nb("try:\n    w.serving_endpoints.get(N)\nexcept NotFound:\n    w.serving_endpoints.create(name=N)")
    blind = _nb("w.serving_endpoints.create(name=N)")
    assert _check(pc, "existing_resources_checked", looked)[0]
    assert not _check(pc, "existing_resources_checked", blind)[0]
    # teddy-002 also lists table and index: an endpoint check alone is not enough
    assert not _check(pc, "existing_resources_checked", looked + "\ndf.write.saveAsTable(T)\n", task_id="teddy-002")[0]


# ---- endpoint_smoke_tested ----------------------------------------------------------------------

@pytest.mark.parametrize("cells", [
    ("resp = w.serving_endpoints.query(name=N, dataframe_records=r)\nE = np.array(resp.predictions)",
     "assert E.shape == (2, 512)"),
    ("def embed(x):\n    out = []\n    out += w.serving_endpoints.query(name=N, dataframe_records=x).predictions\n"
     "    return out", "preds = embed(q)\nprint('accuracy', (preds == truth).mean())"),
    ("r = w.api_client.do('POST', f'/serving-endpoints/{N}/invocations', body=b)",
     "d = np.abs(np.array(r['predictions']) - local).max()"),
])
def test_endpoint_smoke_accepted(pc, cells):
    assert _check(pc, "endpoint_smoke_tested", _nb(*cells))[0]


@pytest.mark.parametrize("cells", [
    ("print(f'ready at /serving-endpoints/{N}/invocations')",),                      # a URL in a message
    ("resp = w.serving_endpoints.query(name=N, dataframe_records=r)\ndisplay(resp.predictions)",),  # only shown
    ("resp = w.serving_endpoints.query(name=N, dataframe_records=r)", "x = other()\nassert x > 0"),   # unrelated assert
])
def test_endpoint_smoke_rejected(pc, cells):
    assert not _check(pc, "endpoint_smoke_tested", _nb(*cells))[0]


# ---- record-based checks ----------------------------------------------------------------------

def test_serving_outcomes_are_split(pc):
    tokens = _record(outcome_accepts_expression_input=False, outcome_returns_valid_embeddings=True)
    broken = _record(outcome_accepts_expression_input=True, outcome_returns_valid_embeddings=False)
    assert _check(pc, "serves_valid_embeddings", "", tokens)[0]
    assert not _check(pc, "serves_expression_input", "", tokens)[0]
    assert not _check(pc, "serves_valid_embeddings", "", broken)[0]
    assert _check(pc, "serves_expression_input", "", broken)[0]
    assert not _check(pc, "serves_valid_embeddings", "", READY)[0]          # not measured


def test_similarity_search_works(pc):
    ok = _record(outcome_index_ready=True, outcome_nn_returns_cell_type=True)
    no_type = _record(outcome_index_ready=True, outcome_nn_returns_cell_type=False)
    assert _check(pc, "similarity_search_works", "", ok, "teddy-002")[0]
    assert not _check(pc, "similarity_search_works", "", no_type, "teddy-002")[0]


def test_gpu_is_a_recorded_guideline_not_a_check(pc):
    gpu = pc.guideline_details(_record({"served_entities": [{"workload_type": "GPU_SMALL"}]}))
    cpu = pc.guideline_details(_record({"served_entities": [{"workload_type": "CPU"}]}))
    assert gpu["guideline/gpu_serving"][0] is True
    assert cpu["guideline/gpu_serving"][0] is False and "15 s per cell" in cpu["guideline/gpu_serving"][1]
    assert pc.guideline_details({})["guideline/gpu_serving"][0] is None     # not recorded: code never read
    scores = pc.practical_scores(pc.scorers.TASKS["teddy-001"],
                                 _nb("e = ServedEntityInput(workload_type=ServingModelWorkloadType.GPU_SMALL)"), {})
    assert not any("gpu" in name for name in scores)


# ---- governance_logging (reuses the new checks) -------------------------------------------------

def test_governance_logging(pc):
    gateway = ("from databricks.sdk import WorkspaceClient\nw = WorkspaceClient()\n"
               "g = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True){usage})\n"
               "w.serving_endpoints.create(name='e', ai_gateway=g)")
    both = gateway.format(usage=", usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True)")
    assert _check(pc, "governance_logging", _nb(both))[0]
    assert not _check(pc, "governance_logging", _nb(gateway.format(usage="")))[0]


# ---- provenance_recorded ----------------------------------------------------------------------

PINNED = f"from huggingface_hub import snapshot_download\nsnapshot_download('r', revision='{SHA}', local_dir='/tmp/t')"
RESOLVED = ("import subprocess, sys, json\nscript = '''\nimport json\nfrom huggingface_hub import HfApi, snapshot_download\n"
            "sha = HfApi().model_info('r').sha\nsnapshot_download('r', revision=sha, local_dir='/tmp/t')\n"
            "print(json.dumps({'sha': sha}))\n'''\n"
            "res = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)\n"
            "REV = json.loads(res.stdout)['sha']")
RESOLVED_FILE = ("import subprocess, sys, json, shutil\nscript = '''\nimport json\n"
                 "from huggingface_hub import HfApi, snapshot_download\nsha = HfApi().model_info('r').sha\n"
                 "snapshot_download('r', revision=sha, local_dir='/tmp/t')\n"
                 "json.dump({'sha': sha}, open('/tmp/rev.json', 'w'))\n'''\n"
                 "subprocess.check_call([sys.executable, '-c', script])\n"
                 "shutil.copy2('/tmp/rev.json', '/Volumes/c/s/v/.rev.json')\n"
                 "REV = json.load(open('/Volumes/c/s/v/.rev.json'))['sha']")
RESOLVED_HERE = ("from huggingface_hub import HfApi, snapshot_download\nsha = HfApi().model_info('r').sha\n"
                 "snapshot_download('r', revision=sha, local_dir='/tmp/t')")
TAGS = "import mlflow\nmlflow.set_tags({'license': 'apache-2.0'})"


@pytest.mark.parametrize("cells", [
    (PINNED, TAGS),
    (PINNED, "import mlflow\nmlflow.log_params({'hf_revision': R, 'license': 'apache-2.0'})"),
    (PINNED, "import json\nm = {'model': {}, 'license': {'code': 'Apache-2.0'}}\njson.dump(m, open(P, 'w'))"),
    (PINNED, "c = MlflowClient()\nfor k, v in {'license': 'Apache-2.0', 'variant': '70M'}.items():\n"
             "    c.set_model_version_tag(NAME, '1', k, v)"),
    (RESOLVED, "import mlflow\nmlflow.set_tags({'license': 'MIT', 'hf_revision': REV})"),   # resolved, via stdout
    (RESOLVED_FILE, "import mlflow\nmlflow.set_tags({'license': 'MIT', 'hf_revision': REV})"),  # via a copied file
    (RESOLVED_HERE, "import mlflow\nmlflow.set_tag('hf_revision', sha)\nmlflow.set_tag('license', 'MIT')"),
    (PINNED, "import json\nfrom pathlib import Path\nPath(P).write_text(json.dumps({'license': 'MIT'}))"),
])
def test_provenance_accepted(pc, cells):
    assert _check(pc, "provenance_recorded", _nb(*cells))[0]


@pytest.mark.parametrize("cells", [
    (PINNED.replace(f"revision='{SHA}', ", ""), TAGS),                       # no revision
    (PINNED.replace(SHA, "main"), TAGS),                                      # branch
    (PINNED.replace(SHA, "dev"), TAGS),                                       # branch (review case)
    (PINNED.replace(f"'{SHA}'", "get_rev()"), TAGS),                          # unresolved expression
    (RESOLVED, TAGS),                                                         # resolved, never recorded
    (RESOLVED, "import mlflow\nmlflow.set_tags({'license': 'MIT', 'hf_revision': 'main'})"),  # review case
    (RESOLVED_HERE, "import mlflow\nmlflow.set_tags({'license': 'MIT', 'hf_revision': get_rev()})"),  # not that commit
    (RESOLVED_FILE.replace("open('/Volumes/c/s/v/.rev.json')", "open('/Volumes/c/s/v/other.json')"),
     "import mlflow\nmlflow.set_tags({'license': 'MIT', 'hf_revision': REV})"),                # a different file
    (PINNED, "import io, json\njson.dump({'license': 'MIT'}, io.StringIO())"),                # not a file (review case)
    (PINNED, "import json\njson.dump({'license': 'MIT'}, open(P))"),                         # opened for reading
    (PINNED, "import mlflow\nmlflow.set_tags({'license': None})"),            # null licence
    (PINNED, "import mlflow\nmlflow.set_tag('license', 'unknown')"),          # unknown licence
    (PINNED, "import mlflow\nmlflow.log_params({'license': ''})"),            # empty licence
    (PINNED, "c = MlflowClient()\nc.update_registered_model(NAME, description='TEDDY (Apache-2.0)')"),  # description
    (PINNED, "# MAGIC %md\n# MAGIC Apache-2.0 licence", "# license: apache-2.0\nx = 1"),
])
def test_provenance_rejected(pc, cells):
    assert not _check(pc, "provenance_recorded", _nb(*cells))[0]


# ---- the committed 2026-10-27 exports ------------------------------------------------------------

EXPECTED_FAILS = {   # label, task -> practical checks that fail (PRACTICAL_TRIAL.md)
    ("baseline", "teddy-001"): {"serves_expression_input", "governance_logging"},
    ("baseline attempt 1", "teddy-001"): {"fidelity_verified", "serves_valid_embeddings", "serves_expression_input",
                                          "endpoint_smoke_tested", "governance_logging", "provenance_recorded"},
    ("baseline", "teddy-002"): {"fidelity_verified", "existing_resources_checked", "governance_logging",
                                "provenance_recorded"},
    ("with_skill", "teddy-001"): set(),
    ("with_skill", "teddy-002"): {"fidelity_verified"},
}
EXPECTED_GPU = {("baseline", "teddy-001"): False, ("baseline attempt 1", "teddy-001"): False,
                ("baseline", "teddy-002"): False, ("with_skill", "teddy-001"): True, ("with_skill", "teddy-002"): True}


@NEEDS_ROUND
def test_trial_results_on_committed_exports(pc):
    samples = pc.round_samples(ROUND)
    assert {(label, tid) for label, tid, _, _ in samples} == set(EXPECTED_FAILS)
    for label, task_id, record, source in samples:
        scores = pc.practical_scores(pc.scorers.TASKS[task_id], source, record)
        failed = {name[len(pc.PRACTICAL_PREFIX):] for name, ok in scores.items() if not ok}
        assert failed == EXPECTED_FAILS[(label, task_id)], (label, task_id)
        assert pc.guideline_details(record)["guideline/gpu_serving"][0] is EXPECTED_GPU[(label, task_id)]


def test_failed_by_severity(pc):
    failed = pc.failed_by_severity({"practical/serves_expression_input": False, "practical/xet_download_ok": False,
                                    "practical/registered_in_uc": True, "guideline/gpu_serving": False})
    assert failed == {"critical": ["practical/xet_download_ok"], "major": ["practical/serves_expression_input"],
                      "minor": []}
