"""
OpenEvolve Decomposition-Workflow Engine — Run Persistence Layer

Durable, restart-surviving storage for workflow runs and their audit logs.

Design notes
============
* Raw ``sqlite3`` with WAL journaling (mirrors ``sovereign_persistence.py``).
* The full :class:`WorkflowState` is stored as a **pickled blob**.  Pickle is
  used on purpose: ``WorkflowState`` is a deeply nested dataclass graph that
  contains ``set`` (``solved_sub_problem_ids``), ``Enum`` (stages/types),
  ``datetime``-like floats, and nested dataclasses (``DecompositionPlan``,
  ``Team``, ``SubProblem`` ...).  A bespoke JSON caster would be fragile; the
  single-process engine can round-trip its own data reliably with pickle.
* The scalar columns (status, tenant, etc.) are duplicated as indexed columns
  so historical runs are queryable without unpickling every blob.
* Every public method may raise on a hard DB error; callers in ``api_server``
  wrap store calls in best-effort try/except so persistence never breaks a
  request.

The schema is owned by ``migrations.MIGRATIONS``; this module applies them.
"""

from __future__ import annotations

import os
import sys
import pickle
import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional

# Make the flat sibling module importable (mirrors ``api_server`` boot fix).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from migrations import MIGRATIONS  # noqa: E402

logger = logging.getLogger("workflow_persistence")

DEFAULT_DB_PATH = os.path.join(_THIS_DIR, "data", "workflow_runs.db")


class WorkflowRunStore:
    """Persist workflow runs (full state blobs) and audit events to SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        # Ensure the data directory exists.
        parent = os.path.dirname(self.db_path)
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                self.logger.warning("Could not create DB dir %s: %s", parent, exc)
        # Defer actual schema work to ``init_database``/``apply_migrations`` so
        # construction itself stays side-effect free for callers that wrap it.

    # ------------------------------------------------------------------
    # Connection / schema management
    # ------------------------------------------------------------------

    def get_connection(self):
        """Open a SQLite connection with WAL mode (mirrors sovereign_persistence)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=-32000")
        return conn

    def init_database(self) -> None:
        """Create the base schema (tables + schema_version)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    status TEXT,
                    current_stage TEXT,
                    problem_statement TEXT,
                    workflow_type TEXT,
                    start_time REAL,
                    end_time REAL,
                    progress REAL,
                    state_blob BLOB,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user TEXT,
                    role TEXT,
                    operation TEXT,
                    resource TEXT,
                    resource_id TEXT,
                    success INTEGER,
                    details_json TEXT
                )
                """
            )
            self._reset_version_table(cursor)
            # Indexes for common historical queries.
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_wr_tenant ON workflow_runs(tenant_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_wr_status ON workflow_runs(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_wr_updated ON workflow_runs(updated_at)"
            )

    def get_current_schema_version(self) -> int:
        """Return the applied schema version (0 if none)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
            )
            cursor.execute("INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 0)")
            cursor.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()
            return row[0] if row else 0

    def _reset_version_table(self, cursor) -> None:
        """Ensure a single-row ``schema_version`` (id=1) tracking the version.

        The table is dropped and recreated so migration application is fully
        idempotent regardless of any prior schema shape (an earlier version
        used ``version`` as the PRIMARY KEY, which caused a UNIQUE collision on
        the second boot when the unqualified ``UPDATE`` re-targeted a freshly
        inserted ``version=0`` row).
        """
        cursor.execute("DROP TABLE IF EXISTS schema_version")
        cursor.execute(
            "CREATE TABLE schema_version "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        cursor.execute("INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 0)")

    def apply_migrations(self) -> None:
        """Apply any pending migrations from ``migrations.MIGRATIONS``."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Self-sufficient + idempotent: drop/recreate the version table even
            # if ``init_database`` has not committed it in this connection.
            self._reset_version_table(cursor)
            cursor.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()
            current = row[0] if row else 0
            for version in sorted(MIGRATIONS.keys()):
                if version > current:
                    self.logger.info("Applying workflow persistence migration v%d", version)
                    for statement in MIGRATIONS[version]:
                        cursor.execute(statement)
                    cursor.execute(
                        "UPDATE schema_version SET version = ? WHERE id = 1", (version,)
                    )
                    current = version

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    def upsert_run(self, state: Any, scalars: Optional[Dict[str, Any]] = None) -> None:
        """Insert or update a workflow run.

        The full ``state`` is pickled into ``state_blob``.  ``scalars`` may
        supply ``created_at`` / ``updated_at`` (ISO strings); the indexed
        columns are otherwise derived from the state object.
        """
        scalars = scalars or {}
        now = _now_iso()

        # Capture the ad-hoc ``error`` attribute the engine attaches at runtime
        # (not a declared dataclass field) so it survives the round-trip.
        state.error = getattr(state, "error", None)

        blob = pickle.dumps(state)
        workflow_id = str(getattr(state, "workflow_id"))
        workflow_type = getattr(state, "workflow_type", None)
        params = {
            "workflow_id": workflow_id,
            "tenant_id": getattr(state, "tenant_id", None),
            "status": getattr(state, "status", None),
            "current_stage": getattr(state, "current_stage", None),
            "problem_statement": getattr(state, "problem_statement", None),
            "workflow_type": str(workflow_type) if workflow_type is not None else None,
            "start_time": _to_float(getattr(state, "start_time", None)),
            "end_time": _to_float(getattr(state, "end_time", None)),
            "progress": getattr(state, "progress", 0.0),
            "state_blob": blob,
            "created_at": scalars.get("created_at", now),
            "updated_at": scalars.get("updated_at", now),
        }
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO workflow_runs (
                    workflow_id, tenant_id, status, current_stage,
                    problem_statement, workflow_type, start_time, end_time,
                    progress, state_blob, created_at, updated_at
                ) VALUES (
                    :workflow_id, :tenant_id, :status, :current_stage,
                    :problem_statement, :workflow_type, :start_time, :end_time,
                    :progress, :state_blob, :created_at, :updated_at
                )
                ON CONFLICT(workflow_id) DO UPDATE SET
                    tenant_id = excluded.tenant_id,
                    status = excluded.status,
                    current_stage = excluded.current_stage,
                    problem_statement = excluded.problem_statement,
                    workflow_type = excluded.workflow_type,
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    progress = excluded.progress,
                    state_blob = excluded.state_blob,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                params,
            )

    def get_run(self, workflow_id: str) -> Any:
        """Return the :class:`WorkflowState` for ``workflow_id`` or ``None``."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state_blob FROM workflow_runs WHERE workflow_id = ?",
                (workflow_id,),
            )
            row = cursor.fetchone()
        if not row or row["state_blob"] is None:
            return None
        return pickle.loads(bytes(row["state_blob"]))

    def delete_run(self, workflow_id: str) -> bool:
        """Delete a run by id. Returns True if a row was removed."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM workflow_runs WHERE workflow_id = ?", (workflow_id,)
            )
            return cursor.rowcount > 0

    def list_runs(self, tenant_id: Optional[str] = None) -> List[Any]:
        """Return all (optionally tenant-scoped) runs as WorkflowStates.

        Rows whose blob fails to unpickle are skipped (and logged) so one bad
        record cannot break listing.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id:
                cursor.execute(
                    "SELECT state_blob FROM workflow_runs WHERE tenant_id = ? "
                    "ORDER BY updated_at DESC",
                    (tenant_id,),
                )
            else:
                cursor.execute(
                    "SELECT state_blob FROM workflow_runs ORDER BY updated_at DESC"
                )
            rows = cursor.fetchall()

        states: List[Any] = []
        for row in rows:
            blob = row["state_blob"]
            if blob is None:
                continue
            try:
                states.append(pickle.loads(bytes(blob)))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Skipping unpickleable workflow blob: %s", exc)
        return states

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def append_audit(self, event: Dict[str, Any]) -> None:
        """Append an audit event. ``event`` keys: timestamp, user, role,
        operation, resource, resource_id, success, details."""
        details = event.get("details") or {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO workflow_audit_logs (
                    timestamp, user, role, operation, resource,
                    resource_id, success, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("timestamp"),
                    event.get("user"),
                    event.get("role"),
                    event.get("operation"),
                    event.get("resource"),
                    event.get("resource_id"),
                    int(bool(event.get("success", True))),
                    json.dumps(details),
                ),
            )

    def get_audit_logs(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return the latest ``limit`` audit events as dicts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM workflow_audit_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = cursor.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                details = json.loads(row["details_json"]) if row["details_json"] else {}
            except (ValueError, TypeError):
                details = {}
            out.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "user": row["user"],
                    "role": row["role"],
                    "operation": row["operation"],
                    "resource": row["resource"],
                    "resource_id": row["resource_id"],
                    "success": bool(row["success"]),
                    "details": details,
                }
            )
        return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _to_float(value: Any) -> Optional[float]:
    """Coerce start/end times (which may be floats or None) to float-or-None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
