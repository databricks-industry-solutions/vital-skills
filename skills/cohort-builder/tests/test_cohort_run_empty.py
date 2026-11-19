"""Regression test: Q returns [] (never None) for a SUCCEEDED statement with zero rows.

The Statement Execution API sends `result.data_array = None` (not []) for a statement that
succeeds with no rows. Before the fix, Q returned that None, so every `for r in q(...)` /
set-comprehension call site — e.g. the ascertainment `GROUP BY` over an empty cohort table in
build_cohort — crashed with "TypeError: 'NoneType' object is not iterable". Reported by Kayla
for an empty cohort (valid condition codes, 0 matching patients).

Offline: the SDK is imported lazily inside Q, so we bypass Q.__init__ and stub the SDK only if
it is not installed — no warehouse, no network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# Q.__call__ does `from databricks.sdk.service.sql import StatementState`. Use the real enum
# when the SDK is present (local), otherwise install a minimal stand-in (CI has no SDK).
try:
    from databricks.sdk.service.sql import StatementState
except Exception:  # pragma: no cover - exercised only in SDK-less envs
    class StatementState:  # minimal stand-in with the states Q compares against
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        SUCCEEDED = "SUCCEEDED"

    for _name in ("databricks", "databricks.sdk", "databricks.sdk.service"):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    _mod = types.ModuleType("databricks.sdk.service.sql")
    _mod.StatementState = StatementState
    sys.modules["databricks.sdk.service.sql"] = _mod

import cohort_run  # noqa: E402


class _Status:
    def __init__(self):
        self.state = StatementState.SUCCEEDED
        self.error = None


class _Resp:
    def __init__(self, data_array):
        self.status = _Status()
        self.result = type("R", (), {"data_array": data_array})()
        self.statement_id = "stmt-1"


class _FakeExec:
    def __init__(self, data_array):
        self._da = data_array

    def execute_statement(self, **kw):
        return _Resp(self._da)

    def get_statement(self, sid):
        return _Resp(self._da)


def _q(data_array):
    q = cohort_run.Q.__new__(cohort_run.Q)  # bypass __init__ (no WorkspaceClient / no auth)
    q.w = type("W", (), {"statement_execution": _FakeExec(data_array)})()
    q.wid, q.catalog, q.schema = "wid", "cat", "sch"
    return q


def test_zero_row_result_returns_empty_list_not_none():
    # A SUCCEEDED GROUP BY with no rows -> data_array is None; Q must coalesce to [].
    q = _q(None)
    rows = q("SELECT ascertainment, COUNT(*) FROM empty_cohort GROUP BY ascertainment")
    assert rows == []
    # the real crash was downstream: iterating the return value
    assert {r[0]: r[1] for r in rows} == {}


def test_populated_result_passthrough():
    q = _q([["both", "5"], ["code", "3"]])
    assert q("SELECT ...") == [["both", "5"], ["code", "3"]]


def test_empty_list_result_passthrough():
    q = _q([])
    assert q("SELECT ...") == []
