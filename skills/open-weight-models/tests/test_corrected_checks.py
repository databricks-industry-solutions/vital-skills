"""Offline tests for the corrected replicated checks (eval/corrected_checks.py), the scorer of record.

For each corrected check: the valid alternatives the verbatim (Sep–Oct 2026) check got wrong,
and the failures it must still catch. Also: the verbatim logic and its score files are unchanged,
the committed corrected score files (2026-10-27 round and the Sep–Oct 2026 re-score) match a
fresh re-score, and the recorded Sep–Oct 2026 files are never written to. Needs mlflow-skinny
and packaging (tests/requirements.txt); no workspace, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = SKILL_DIR / "eval"
ROUND = EVAL_DIR / "models" / "teddy" / "runs_20261026"
SEP_OCT = EVAL_DIR / "models" / "teddy" / "runs_20260924"
# Run folders (exported notebooks, run records, per-run scores) are not kept in the repository: they are
# git-ignored and stay with whoever ran the round. Tests that replay them run where the folders are
# present and are skipped elsewhere, e.g. on CI.
NEEDS_ROUND = pytest.mark.skipif(not ROUND.is_dir(),
                                 reason=f"run records not in this checkout ({ROUND.name} is kept outside the repository)")
NEEDS_SEP_OCT = pytest.mark.skipif(not SEP_OCT.is_dir(),
                                 reason=f"run records not in this checkout ({SEP_OCT.name} is kept outside the repository)")


@pytest.fixture(scope="module")
def cc():
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location("owm_corrected_checks", EVAL_DIR / "corrected_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nb(*cells):
    return "# Databricks notebook source\n" + "\n\n# COMMAND ----------\n\n".join(cells) + "\n"


READY = {"outcome": {"outcome/endpoint_ready": {"passed": True}, "outcome/serves_recorded_version": {"passed": True}}}


def _check(cc, key, source, record=None, task_id="teddy-001"):
    task = cc.scorers.TASKS[task_id]
    return cc.CORRECTIONS[key]["check"](cc.pc.PracticalCode(source), source, record, task)


LOG = ("import mlflow\nmlflow.pyfunc.log_model(name='m', python_model=W(), signature=sig, input_example=ex,\n"
       "    pip_requirements=REQS)")


# ---- registry ------------------------------------------------------------------------------

EP_POST = 'w.api_client.do("POST", f"/serving-endpoints/{EP}/invocations", body=b)'


@pytest.mark.parametrize("cell, ok", [
    ("r = w.serving_endpoints.query(name=EP, dataframe_records=rows)", True),
    (EP_POST, True),
    # round 2 (with-skill teddy-001): the REST call inside a helper the notebook calls
    (f"def invoke(w, b):\n    return {EP_POST}\nout = invoke(w, {{'dataframe_records': rows}})", True),
    (f"def invoke(w, b):\n    return {EP_POST}", False),                   # helper never called
    ("# test the endpoint later\nprint('verify the endpoint in the UI')", False),   # text only
    ('print("/serving-endpoints/x/invocations")', False),                  # a print is not a request
])
def test_endpoint_query(cc, cell, ok):
    assert _check(cc, "smoke_tests/endpoint_query", _nb(cell))[0] is ok


def test_every_correction_states_flaw_and_rule(cc):
    expected = cc.scorers.TASKS["teddy-002"]["checks"]
    for key, meta in cc.CORRECTIONS.items():
        assert meta["flaw"] and meta["accepts"] and callable(meta["check"]), key
        assert cc.scorers.mlflow_metric_name(key) in expected, key    # a real replicated sub-check
    assert "unchanged" in cc.SEVERITY_NOTE


@NEEDS_ROUND
def test_unflawed_keys_keep_the_verbatim_result(cc):
    source = (ROUND / "baseline" / "teddy-002.py").read_text()
    task = cc.scorers.TASKS["teddy-002"]
    verbatim = cc.scorers.sub_check_scores(task, source)
    details = cc.corrected_details(task, source, None)
    assert list(details) == list(verbatim)
    for key, (ok, reason, v) in details.items():
        assert v == verbatim[key]
        if key not in cc.CORRECTIONS:
            assert ok == v and "same logic" in reason


def test_edge_task_phase_without_sub_checks_is_left_alone(cc):
    edge = {**cc.scorers.TASKS["teddy-001"], "difficulty": "edge"}
    details = cc.corrected_details(edge, "print(1)\n")
    assert "download_staging" in details and not any(k.startswith("download_staging/") for k in details)


# ---- each corrected check: alternatives the verbatim check got wrong, and real failures --------

@pytest.mark.parametrize("cell, ok", [
    ("import tempfile\nfrom huggingface_hub import snapshot_download\nd = tempfile.mkdtemp()\n"
     "snapshot_download('Merck/TEDDY', local_dir=d)", True),
    ("from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/Volumes/c/s/v/t')", True),
    ("from huggingface_hub import snapshot_download\nprint('/tmp is used')\nsnapshot_download('Merck/TEDDY')", False),
    ("from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/local_disk0/t')", False),
    # round 2 (with-skill teddy-001): probe /local_disk0, fall back to /tmp, download in a subprocess
    # script whose local_dir is formatted from that variable, then stage a copy under the same variable
    ("import os, subprocess, sys\ntry:\n    os.makedirs('/local_disk0/tmp')\n    TMP = '/local_disk0/tmp'\nexcept OSError:\n    TMP = '/tmp'\nd = f'{TMP}/snap'\n"
     "script = f'''\nfrom huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir=\"{d}\")\n'''\n"
     "subprocess.check_call([sys.executable, '-c', script])" + "\nimport shutil\nshutil.copytree(d, f'{TMP}/code')", True),
    ("import subprocess, sys\nd = '/local_disk0/snap'\n"
     "script = f'''\nfrom huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir=\"{d}\")\n'''\n"
     "subprocess.check_call([sys.executable, '-c', script])", False),                                      # /local_disk0 with no fallback
    ("import shutil\nshutil.copytree('/tmp/t', '/local_disk0/t')\n" + "from huggingface_hub import snapshot_download\n"
     "snapshot_download('Merck/TEDDY', local_dir='/tmp/t')", False),    # staging to /local_disk0, no fallback
])
def test_uses_tmp(cc, cell, ok):
    assert _check(cc, "download_staging/uses_tmp", _nb(cell))[0] is ok


def test_hf_xet_disabled(cc):
    marked_only = _nb("# MAGIC %md\n# MAGIC HF_HUB_DISABLE_XET was not set",
                      "from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/tmp/t')")
    assert _check(cc, "download_staging/hf_xet_disabled", marked_only)[0] is False   # verbatim passes this
    early = _nb("import os\nos.environ['HF_HUB_DISABLE_XET'] = '1'",
                "from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/tmp/t')")
    assert _check(cc, "download_staging/hf_xet_disabled", early)[0] is True
    completed = _nb("from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/tmp/t')",
                    "from transformers import AutoModel\nm = AutoModel.from_pretrained('/tmp/t')")
    assert _check(cc, "download_staging/hf_xet_disabled", completed, READY)[0] is True
    assert _check(cc, "download_staging/hf_xet_disabled", completed, None)[0] is False   # no run record
    # Codex review of 117ac56: a later load of OTHER files is no evidence the download completed
    unrelated = _nb("from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/tmp/t')",
                    "from transformers import AutoModel\nm = AutoModel.from_pretrained('/Volumes/c/s/v/other')")
    assert _check(cc, "download_staging/hf_xet_disabled", unrelated, READY)[0] is False
    joined = _nb("import os\nfrom huggingface_hub import snapshot_download\nV = '/Volumes/c/s/v'\n"
                 "snapshot_download('Merck/TEDDY', local_dir=V)",
                 "from transformers import AutoModel\nm = AutoModel.from_pretrained(os.path.join(V, '70M'))")
    assert _check(cc, "download_staging/hf_xet_disabled", joined, READY)[0] is True
    other = {**cc.scorers.TASKS["teddy-001"], "model_family": "scgpt"}
    assert cc.hf_xet_disabled(cc.pc.PracticalCode(""), "", None, other)[0] is True
    # round 2 (baseline teddy-002): download to a temp folder, copy file by file into a Volume via
    # os.walk, then load from the Volume; a per-file copy from the download folder carries the evidence
    walked = _nb("import os, shutil, tempfile\nfrom huggingface_hub import snapshot_download\nV = '/Volumes/c/s/v/teddy'\n"
                 "tmp = tempfile.mkdtemp()\nsnapshot_download('Merck/TEDDY', local_dir=tmp)\n"
                 "for root, _, files in os.walk(tmp):\n    for fn in files:\n        src = os.path.join(root, fn)\n"
                 "        shutil.copyfile(src, os.path.join(V, os.path.relpath(src, tmp)))",
                 "from safetensors.torch import load_file\nw = load_file(f'{V}/70M/model.safetensors')")
    assert _check(cc, "download_staging/hf_xet_disabled", walked, READY)[0] is True
    # the same copy loop, but walking a folder that is not the download destination
    elsewhere = _nb("import os, shutil, tempfile\nfrom huggingface_hub import snapshot_download\nV = '/Volumes/c/s/v/teddy'\n"
                    "tmp = tempfile.mkdtemp()\nsnapshot_download('Merck/TEDDY', local_dir=tmp)\nother = '/tmp/x'\n"
                    "for root, _, files in os.walk(other):\n    for fn in files:\n        src = os.path.join(root, fn)\n"
                    "        shutil.copyfile(src, os.path.join(V, fn))",
                    "from safetensors.torch import load_file\nw = load_file(f'{V}/70M/model.safetensors')")
    assert _check(cc, "download_staging/hf_xet_disabled", elsewhere, READY)[0] is False


@pytest.mark.parametrize("cells, ok", [
    (("%pip install 'transformers>=4.41,<5'", "import transformers"), True),
    (("REQS = ['transformers==4.41.0']", LOG), True),
    (("import torch\nREQS = ['torch==2.5.1', 'safetensors==0.4.5']", LOG), True),   # no transformers at all
    (("REQS = ['transformers>=4.40']", LOG), False),
    (("%pip install transformers", "import transformers"), False),
    # round 2 baselines: the served model is plain PyTorch; the upstream package is imported only to
    # compare against upstream, so no transformers pin is needed
    (("import torch, mlflow\nclass Enc(torch.nn.Module):\n    pass\n"
      "class W(mlflow.pyfunc.PythonModel):\n    def load_context(self, c):\n        self.m = Enc()",
      "from teddy.models.teddy_g.model import TeddyGModel\nref = TeddyGModel",
      "REQS = ['torch==2.5.1']", LOG), True),
    # the served class uses an upstream / transformers import made at notebook level -> pin needed
    (("import mlflow\nfrom transformers import AutoModel\n"
      "class W(mlflow.pyfunc.PythonModel):\n    def load_context(self, c):\n        self.m = AutoModel.from_pretrained(c)",
      "REQS = ['torch==2.5.1']", LOG), False),
    (("import mlflow\nclass W(mlflow.pyfunc.PythonModel):\n    def load_context(self, c):\n"
      "        from teddy.models.teddy_g.model import TeddyGModel\n        self.m = TeddyGModel",
      "REQS = ['torch==2.5.1']", LOG), False),
    # the upstream package shipped with the model via code_paths -> pin needed
    (("import mlflow\nclass W(mlflow.pyfunc.PythonModel):\n    pass",
      "mlflow.pyfunc.log_model(name='m', python_model=W(), code_paths=['/tmp/code/teddy'], pip_requirements=['torch'])"), False),
])
def test_deps_pinned(cc, cells, ok):
    assert _check(cc, "download_staging/deps_pinned", _nb(*cells))[0] is ok


@pytest.mark.parametrize("cells, ok", [
    (("%pip install 'transformers>=4.41,<5'",), True),
    (("print('no transformers here')",), True),
    (("%pip install 'transformers>=4.40'",), False),
    (("%pip install transformers", "import transformers\nREQS = [f'transformers=={transformers.__version__}']", LOG), False),
    (("%pip install 'transformers<5'", "import transformers\nREQS = [f'transformers=={transformers.__version__}']", LOG), True),
])
def test_forbidden_transformers_open_range(cc, cells, ok):
    assert _check(cc, "forbidden_patterns/transformers>=", _nb(*cells))[0] is ok


@pytest.mark.parametrize("cells, record, ok", [
    (("from mlflow.models import ModelSignature\nsig = ModelSignature(inputs=s)\nex = x\nREQS = []", LOG), None, True),
    (("from mlflow.models import infer_signature\nsig = infer_signature(x, y)\nex = x\nREQS = []", LOG), None, True),
    (("# we use infer_signature\nsig = None\nex = x\nREQS = []", LOG), None, False),
    (("sig = load_sig()\nex = x\nREQS = []", LOG), READY, True),
    (("sig = load_sig()\nex = x\nREQS = []", LOG), None, False),
])
def test_infer_signature(cc, cells, record, ok):
    assert _check(cc, "model_registration/infer_signature", _nb(*cells), record)[0] is ok


TOKENIZE = ("from teddy.data_processing.tokenization.tokenization import tokenize\n"
            "tokenize(data_path=f'{W}/in.h5ad', tokenization_args={'save_dir': f'{W}/tokenized'})")


@pytest.mark.parametrize("cells, ok", [
    (("vocab = open(f'{D}/vocab.txt').read().split()\nex = {'genes': vocab[:10]}\nsig = s\nREQS = []", LOG), True),
    (("import anndata\na = anndata.read_h5ad(P)\nex = a.var_names[:10]\nsig = s\nREQS = []", LOG), True),
    (("from teddy.tokenizer.gene_tokenizer import GeneTokenizer\nt = GeneTokenizer.from_pretrained(D)",
      "ex = {'gene_ids': [t.vocab[g] for g in GENES]}\nsig = s\nREQS = []", LOG), True),
    # files the upstream tokenizer wrote, read back from its save_dir (artifact relationship)
    (("W = '/tmp/w'", TOKENIZE, "import glob\nfiles = glob.glob(f'{W}/tokenized/**/*.arrow')\nex = load(files)\nsig = s\nREQS = []",
      LOG), True),
    # Codex review of 117ac56: an unrelated upstream-tokenizer call + made-up (non-random) IDs
    (("W = '/tmp/w'", TOKENIZE, "ex = {'gene_ids': [7, 14, 21, 28]}\nsig = s\nREQS = []", LOG), False),
    (("W = '/tmp/w'", TOKENIZE, "import glob\nfiles = glob.glob('/tmp/other/*.arrow')\nex = load(files)\nsig = s\nREQS = []",
      LOG), False),
    (("import torch\nex = torch.randint(2, 100, (2, 8))\nsig = s\nREQS = []", LOG), False),
    (("# see vocab.txt\nex = {'gene_ids': [7, 14, 21]}\nsig = s\nREQS = []", LOG), False),
    # a literal gene ID list does not derive from the model's vocabulary or real data
    (("ex = {'genes': ['ENSG00000000003', 'ENSG00000000005']}\nsig = s\nREQS = []", LOG), False),
])
def test_real_gene_ids(cc, cells, ok):
    assert _check(cc, "pyfunc_quality/real_gene_ids", _nb(*cells))[0] is ok


SNAPSHOT = "from huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir='/tmp/t')"


@pytest.mark.parametrize("cells, ok", [
    # the model's own pyproject.toml, parsed, flowing into pip_requirements
    ((SNAPSHOT, "import tomllib\nwith open('/tmp/t/pyproject.toml', 'rb') as f:\n    deps = tomllib.load(f)",
      "pins = deps['tool']['poetry']['dependencies']\nREQS = [f'{k}=={v}' for k, v in pins.items()]\nsig = s\nex = x", LOG), True),
    (("REQS = ['torch>=2.3.0', 'transformers==4.41.0', 'numpy>=1.26.4,<2.0']\nsig = s\nex = x", LOG), True),
    # Codex review of 117ac56: reading some file named requirements.txt / pyproject.toml is not enough
    ((SNAPSHOT, "reqs = open('requirements.txt').read().split()\nREQS = reqs\nsig = s\nex = x", LOG), False),
    ((SNAPSHOT, "import tomllib\nwith open('/tmp/t/pyproject.toml', 'rb') as f:\n    deps = tomllib.load(f)",
      "REQS = ['torch']\nsig = s\nex = x", LOG), False),                       # read, but not used
    (("REQS = ['transformers==4.57.6']\nsig = s\nex = x", LOG), False),
    (("import transformers\nREQS = [f'transformers=={transformers.__version__}']\nsig = s\nex = x", LOG), False),
    (("REQS = ['numpy', 'pandas']\nsig = s\nex = x", LOG), False),              # nothing pinned
    (("snapshot_download('Merck/TEDDY', allow_patterns=['pyproject.toml'])\nREQS = ['torch']\nsig = s\nex = x", LOG), False),
])
def test_pip_reqs_from_source(cc, cells, ok):
    assert _check(cc, "model_registration/pip_reqs_from_source", _nb(*cells))[0] is ok


def test_upstream_pins_are_recorded_with_their_source(cc):
    upstream = cc.family_settings.load_family("teddy")["upstream_dependencies"]
    assert upstream["revision"] == "8c880aa67674042ccdc254bccb5de081092a48bf"
    assert upstream["pins"]["transformers"] == "4.41.0"


HF = "from huggingface_hub import snapshot_download\n"


@pytest.mark.parametrize("cell, ok", [
    # SKILL.md #16: probe /local_disk0, fall back to /tmp, then use the variable
    ("import os\ntry:\n    os.makedirs('/local_disk0/tmp')\n    TMP = '/local_disk0/tmp'\nexcept OSError:\n    TMP = '/tmp'\n"
     + HF + "snapshot_download('Merck/TEDDY', local_dir=f'{TMP}/teddy')", True),
    ("print('no /local_disk0 on serverless')\nimport os\nos.makedirs('/tmp/x')", True),
    ("import os\nos.makedirs('/local_disk0/teddy')", False),
    (HF + "snapshot_download('Merck/TEDDY', local_dir='/local_disk0/t')", False),
    ("import shutil\nshutil.copytree('/tmp/t', '/local_disk0/t')", False),
    # Codex review of 117ac56: a try with any handler is not a fallback
    ("import os\ntry:\n    os.makedirs('/local_disk0/teddy')\nexcept Exception:\n    pass", False),
    (HF + "try:\n    snapshot_download('Merck/TEDDY', local_dir='/local_disk0/t')\nexcept Exception:\n    pass", False),
    # a fallback is assigned, but the download does not use it
    ("try:\n    TMP = '/local_disk0/tmp'\nexcept OSError:\n    TMP = '/tmp'\n"
     + HF + "snapshot_download('Merck/TEDDY', local_dir='/local_disk0/t')", False),
    # round 2: a subprocess download whose script formats in the fallback variable
    ("import os, subprocess, sys\ntry:\n    os.makedirs('/local_disk0/tmp')\n    TMP = '/local_disk0/tmp'\nexcept OSError:\n    TMP = '/tmp'\nd = f'{TMP}/snap'\n"
     "script = f'''\nfrom huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir=\"{d}\")\n'''\n"
     "subprocess.check_call([sys.executable, '-c', script])", True),
    ("import subprocess, sys\nd = '/local_disk0/snap'\n"
     "script = f'''\nfrom huggingface_hub import snapshot_download\nsnapshot_download('Merck/TEDDY', local_dir=\"{d}\")\n'''\n"
     "subprocess.check_call([sys.executable, '-c', script])", False),
])
def test_local_disk0(cc, cell, ok):
    assert _check(cc, "forbidden_patterns//local_disk0", _nb(cell))[0] is ok


REST = "w.api_client.do('POST', '/api/2.0/serving-endpoints', body=body)"


@pytest.mark.parametrize("cell, ok", [
    ("body = {'name': N, 'config': {'served_entities': [{'workload_type': 'GPU_SMALL'}]}}\n" + REST, True),
    ("from databricks.sdk.service.serving import ServedEntityInput, ServingModelWorkloadType\n"
     "e = ServedEntityInput(entity_name=M, workload_type=ServingModelWorkloadType.GPU_SMALL)", True),
    ("e = ServedEntityInput(entity_name=M, workload_type='GPU_SMALL')", False),
    ("WT = 'GPU_SMALL'\ne = ServedEntityInput(entity_name=M, workload_type=WT)", False),
    # Codex review of 117ac56: a dict is not REST just because it has a workload_type key
    ("e = ServedEntityInput(**{'entity_name': M, 'workload_type': 'GPU_SMALL'})", False),
    ("kw = {'entity_name': M, 'workload_type': 'GPU_SMALL'}\ne = ServedEntityInput(**kw)", False),
])
def test_sdk_enums(cc, cell, ok):
    assert _check(cc, "endpoint_deployment/sdk_enums", _nb(cell))[0] is ok


@pytest.mark.parametrize("cell, ok", [
    ("import requests\nrequests.put(f'{HOST}/api/2.0/serving-endpoints/{N}/ai-gateway',\n"
     "    json={'inference_table_config': {'enabled': True}})", True),
    ("from databricks.sdk.service.serving import AiGatewayInferenceTableConfig", False),   # import only
    ("from databricks.sdk.service.serving import AutoCaptureConfigInput\n"
     "w.serving_endpoints.create(name=N, ai_gateway=AiGatewayConfig(\n"
     "    inference_table_config=AiGatewayInferenceTableConfig(enabled=True)))\nc = AutoCaptureConfigInput(enabled=True)", False),
])
def test_ai_gateway_config(cc, cell, ok):
    assert _check(cc, "endpoint_deployment/ai_gateway_config", _nb(cell))[0] is ok


INDEX = ("vsc.create_delta_sync_index(endpoint_name=E, index_name=I, source_table_name=T,\n"
         "    primary_key='id', embedding_dimension=512, embedding_vector_column='emb')")


@pytest.mark.parametrize("cell, ok", [
    (f"T = 'c.s.cells'\nif spark.catalog.tableExists(T):\n    {INDEX}\nelse:\n    print('skip search')", True),
    (f"T = 'c.s.cells'\nif not spark.catalog.tableExists(T):\n    print('skip')\nelse:\n    {INDEX}", True),
    ("from pyspark.errors import AnalysisException\nT = 'c.s.cells'\ntry:\n    df = spark.table(T)\n"
     f"except AnalysisException:\n    df = None\n{INDEX}", True),
    (f"T = 'c.s.cells'\ntry:\n    spark.sql(f'DROP TABLE IF EXISTS {{T}}')\nexcept Exception:\n    pass\n{INDEX}", False),
    (f"T = 'c.s.cells'\nspark.sql(f'CREATE TABLE IF NOT EXISTS {{T}} (id INT)')\ndf = spark.table(T)\n{INDEX}", False),
    (f"T = 'c.s.cells'\n# skip VS if the table does not exist\ndf = spark.table(T)\n{INDEX}", False),
    # Codex review of 117ac56: an unrelated table check, and a handler that re-raises
    (f"T = 'c.s.cells'\nif spark.catalog.tableExists('c.s.other'):\n    print('x')\n{INDEX}", False),
    (f"T = 'c.s.cells'\ntry:\n    df = spark.table(T)\nexcept Exception as e:\n    raise RuntimeError('missing') from e\n{INDEX}", False),
    (f"T = 'c.s.cells'\nif not spark.catalog.tableExists(T):\n    raise ValueError('run the embed step first')\n{INDEX}", False),
    ("if spark.catalog.tableExists('c.s.cells'):\n    print('ok')", False),     # no index at all
])
def test_graceful_skip(cc, cell, ok):
    assert _check(cc, "ai_search/graceful_skip", _nb(cell), task_id="teddy-002")[0] is ok


DEPLOY = "w.serving_endpoints.create(name=N, config=c)"


@pytest.mark.parametrize("cells, ok", [
    # widget gate, default off -> pass; default on -> fail
    (("dbutils.widgets.dropdown('run_go', 'false', ['false', 'true'])",
      "RUN_GO = dbutils.widgets.get('run_go') == 'true'", f"if RUN_GO:\n    {DEPLOY}"), True),
    (("dbutils.widgets.dropdown('run_go', 'true', ['false', 'true'])",
      "RUN_GO = dbutils.widgets.get('run_go') == 'true'", f"if RUN_GO:\n    {DEPLOY}"), False),
    # the deploy in the else branch of `if not RUN_GO`
    (("dbutils.widgets.dropdown('run_go', 'false', ['false', 'true'])",
      "RUN_GO = dbutils.widgets.get('run_go') == 'true'",
      f"if not RUN_GO:\n    print('off')\nelse:\n    {DEPLOY}"), True),
    # any widget name; text widget with defaultValue=
    (("dbutils.widgets.text(name='deploy', defaultValue='no')",
      "if dbutils.widgets.get('deploy').lower() == 'yes':\n    " + DEPLOY), True),
    # plain constants
    (("GO = False", f"if GO:\n    {DEPLOY}"), True),
    (("GO = True", f"if GO:\n    {DEPLOY}"), False),
    # environment / notebook parameters with a default
    (("import os\nGO = os.environ.get('RUN_GO', 'false') == 'true'", f"if GO:\n    {DEPLOY}"), True),
    (("import os\nGO = os.getenv('RUN_GO', '1') == '1'", f"if GO:\n    {DEPLOY}"), False),
    # default not determinable: the widget is never defined in the notebook
    (("RUN_GO = dbutils.widgets.get('run_go') == 'true'", f"if RUN_GO:\n    {DEPLOY}"), False),
    # early exit before the deploy when the flag is off
    (("GO = False", "if not GO:\n    dbutils.notebook.exit('gate off')", DEPLOY), True),
    # Codex review of 117ac56: an exit with no deploy call at all is not a gate
    (("GO = False", "if not GO:\n    dbutils.notebook.exit('gate off')"), False),
    # an exit AFTER the deploy does not gate it
    (("GO = False", DEPLOY, "if not GO:\n    dbutils.notebook.exit('gate off')"), False),
    # one deploy gated, another not
    (("GO = False", f"if GO:\n    {DEPLOY}", "w.serving_endpoints.update_config(name=N, served_entities=e)"), False),
    # a flag the notebook sets from endpoint state is idempotency, not a gate
    (("SKIP_DEPLOY = False\ntry:\n    SKIP_DEPLOY = w.serving_endpoints.get(N).state.ready == 'READY'\nexcept Exception:\n    pass",
      f"if not SKIP_DEPLOY:\n    {DEPLOY}"), False),
    (("# run_go gate\n" + DEPLOY,), False),
    (("NAME = 'ep'", "if NAME in existing:\n    w.serving_endpoints.update_config(name=NAME, served_entities=e)"), False),
    # round 2: a top-level `assert FLAG` before the deploy stops the notebook while the flag is off
    (("CONFIRM = False", "assert CONFIRM, 'set CONFIRM = True'", DEPLOY), True),
    (("CONFIRM = True", "assert CONFIRM, 'set CONFIRM = True'", DEPLOY), False),
    (("import os", "assert os.path.exists('/tmp')", DEPLOY), False),                # not a user flag
    (("CONFIRM = False", DEPLOY, "assert CONFIRM, 'set CONFIRM = True'"), False),    # after the deploy
])
def test_run_go_gate(cc, cells, ok):
    assert _check(cc, "endpoint_deployment/run_go_gate", _nb(*cells))[0] is ok


def test_run_go_gate_reasons(cc):
    on = _nb("dbutils.widgets.dropdown('run_go', 'true', ['false', 'true'])",
             "RUN_GO = dbutils.widgets.get('run_go') == 'true'", f"if RUN_GO:\n    {DEPLOY}")
    assert "defaults to on" in _check(cc, "endpoint_deployment/run_go_gate", on)[1]
    unknown = _nb("RUN_GO = dbutils.widgets.get('run_go') == 'true'", f"if RUN_GO:\n    {DEPLOY}")
    assert "cannot be determined" in _check(cc, "endpoint_deployment/run_go_gate", unknown)[1]
    assert "no serving-endpoint deploy call" in _check(cc, "endpoint_deployment/run_go_gate", _nb("GO = False"))[1]
    saved_on = _nb("CONFIRM = True", "assert CONFIRM, 'set CONFIRM = True'", DEPLOY)
    assert "assert at line" in _check(cc, "endpoint_deployment/run_go_gate", saved_on)[1]


# ---- verbatim logic and recorded files unchanged; committed corrected files reproduce ---------

def _round_samples(cc):
    return cc.pc.round_samples(ROUND)


@NEEDS_ROUND
def test_verbatim_round_scores_unchanged(cc):
    """The verbatim replicated score files of the 2026-10-27 round still come from scorers.py as is."""
    for label, task_id, record, source in _round_samples(cc):
        recorded = json.loads((ROUND / "scores" / f"{label.replace(' ', '_')}_replicated.json").read_text())[task_id]
        fresh = {cc.scorers.mlflow_metric_name(k): v
                 for k, v in cc.scorers.sub_check_scores(cc.scorers.TASKS[task_id], source).items()}
        assert fresh == recorded, label


@NEEDS_ROUND
def test_committed_round_corrected_scores_reproduce(cc):
    files = {}
    for label, task_id, record, source in _round_samples(cc):
        files.setdefault(f"{label.replace(' ', '_')}_corrected.json", {})[task_id] = cc.corrected_scores(
            cc.scorers.TASKS[task_id], source, record)
    for name, scores in files.items():
        assert json.loads((ROUND / "scores" / name).read_text()) == scores, name


@NEEDS_SEP_OCT
def test_committed_sep_oct_corrected_scores_reproduce(cc):
    files = {}
    for arm, old, new, source in cc.sep_oct_samples():
        files.setdefault(f"{arm}_corrected.json", {})[old] = cc.corrected_scores(cc.scorers.TASKS[new], source, None)
    for name, scores in files.items():
        assert json.loads((SEP_OCT / "scores_corrected" / name).read_text()) == scores, name


@NEEDS_SEP_OCT
def test_sep_oct_rescore_writes_only_new_files(cc, tmp_path, monkeypatch):
    """--sep-oct --write touches nothing but models/teddy/runs_20260924/scores_corrected/."""
    monkeypatch.setattr(cc, "SEP_OCT_DIR", tmp_path)
    for arm in cc.SEP_OCT_ARMS:
        (tmp_path / "results" / arm).mkdir(parents=True)
        for old in cc.SEP_OCT_TASKS:
            (tmp_path / "results" / arm / f"{old}.txt").write_text((SEP_OCT / "results" / arm / f"{old}.txt").read_text())
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    cc.main(["--sep-oct", "--write"])
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert {p.relative_to(tmp_path).as_posix() for p in after - before} == {
        "scores_corrected/baseline_corrected.json", "scores_corrected/with_skill_corrected.json"}


@NEEDS_ROUND
def test_scorer_feedbacks_flag_verbatim_disagreement(cc):
    source = (ROUND / "baseline" / "teddy-002.py").read_text()
    record = next(r for label, t, r, _ in _round_samples(cc) if label == "baseline" and t == "teddy-002")
    outcome = {k: v["passed"] for k, v in record["outcome"].items()}
    feedbacks = cc.corrected_replicated_checks(inputs={"task_id": "teddy-002"},
                                               outputs={"notebook_source": source, "outcome": outcome})
    by_name = {f.name: f for f in feedbacks}
    assert by_name["corrected/download_staging/uses_tmp"].value is True
    assert "⚑ verbatim: fail" in by_name["corrected/download_staging/uses_tmp"].rationale
    assert all(f.name.startswith(cc.CORRECTED_PREFIX) for f in feedbacks)
    assert cc.scorers.metric_group("corrected/download_staging/uses_tmp") == "corrected"


def test_scoring_notebook_logs_corrected_as_the_primary_group(cc):
    text = (EVAL_DIR / "score_open_weight_models.py").read_text()
    assert "corrected_scorers" in text and "run_scorers = corrected_scorers + all_scorers" in text
    assert cc.scorers.COUNTED_GROUPS[0] == "corrected" and "replicated" not in cc.scorers.COUNTED_GROUPS
