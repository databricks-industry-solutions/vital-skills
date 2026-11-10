"""The workflow page and the check notebook call the skill's scripts instead of carrying their own copies.

references/workflows/embeddings-to-vector-search.md and tests/check_embeddings_to_vector_search.py used to
define split_holdout, calibrate, map_2d and the AI Search waits inline (and a test kept the two copies equal).
That code now lives once in scripts/ (vs_index.py, nn_annotate.py, sample_checks.py, visualize.py), tested by
tests/test_scripts_*.py. This test keeps it that way: neither file may redefine a script function, and every
`vx.` / `nn.` / `sx.` / `viz.` name they use must exist in the scripts.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
WORKFLOW = SKILL / "references" / "workflows" / "embeddings-to-vector-search.md"
NOTEBOOK = SKILL / "tests" / "check_embeddings_to_vector_search.py"
sys.path.insert(0, str(SKILL / "scripts"))
pytest.importorskip("matplotlib")
import nn_annotate  # noqa: E402
import sample_checks  # noqa: E402
import visualize  # noqa: E402
import vs_index  # noqa: E402

MODULES = {"vx": vs_index, "nn": nn_annotate, "sx": sample_checks, "viz": visualize}
SCRIPT_FUNCTIONS = {name for m in MODULES.values() for name, obj in vars(m).items()
                    if callable(obj) and getattr(obj, "__module__", None) == m.__name__}
# Thin bindings the check notebook keeps on purpose: they call vx.* with this run's widget values.
NOTEBOOK_BINDINGS = {"on_existing", "index_mismatches"}


def _workflow_code():
    return "\n".join(re.findall(r"```python\n(.*?)```", WORKFLOW.read_text(), re.S))


def _defined_functions(code):
    return {n.name for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef)}


def _attribute_uses(code):
    uses = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in MODULES:
            uses.add((node.value.id, node.attr))
    return uses


@pytest.mark.parametrize("source", ["workflow", "notebook"])
def test_no_copies_of_script_functions(source):
    code = _workflow_code() if source == "workflow" else NOTEBOOK.read_text()
    copies = _defined_functions(code) & SCRIPT_FUNCTIONS
    if source == "notebook":
        copies -= NOTEBOOK_BINDINGS
    assert not copies, f"{source} redefines {sorted(copies)}; call the scripts/ version instead"


@pytest.mark.parametrize("source", ["workflow", "notebook"])
def test_every_script_name_used_exists(source):
    code = _workflow_code() if source == "workflow" else NOTEBOOK.read_text()
    uses = _attribute_uses(code)
    assert uses, f"{source} does not call the scripts"
    missing = sorted(f"{alias}.{attr}" for alias, attr in uses if not hasattr(MODULES[alias], attr))
    assert not missing, f"{source} uses names the scripts do not define: {missing}"


def test_notebook_bindings_only_delegate():
    tree = ast.parse(NOTEBOOK.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NOTEBOOK_BINDINGS:
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
            assert any(isinstance(c.func, ast.Attribute) and getattr(c.func.value, "id", None) == "vx"
                       and c.func.attr == node.name for c in calls), f"{node.name} must call vx.{node.name}"


def test_workflow_sets_up_the_scripts_path():
    code = _workflow_code()
    assert 'sys.path.insert(0, f"{SKILL_DIR}/scripts")' in code
    assert "import vs_index as vx, nn_annotate as nn, visualize as viz" in code
