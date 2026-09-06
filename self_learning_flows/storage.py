"""SQLite persistence for immutable episodes and versioned workflow state."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .models import TaskEpisode, WorkflowDefinition, WorkflowFeedback


class SQLiteStore:
    """A dependency-free store suitable for local experiments and small services.

    A new SQLite connection is used per operation, making the store safe to share
    between worker threads. WAL mode allows readers while the learner writes.
    """

    def __init__(self, path: str | Path = ".self_learning_flows/workflows.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _setup(self) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    task_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    verified INTEGER NOT NULL,
                    structural_signature TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_scope
                    ON episodes(scope, verified, success);
                CREATE INDEX IF NOT EXISTS idx_episodes_signature
                    ON episodes(structural_signature);

                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    structural_signature TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_workflows_scope_status
                    ON workflows(scope, status);
                CREATE INDEX IF NOT EXISTS idx_workflows_signature
                    ON workflows(structural_signature);

                CREATE TABLE IF NOT EXISTS workflow_versions (
                    workflow_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (workflow_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_workflow_versions
                    ON workflow_versions(workflow_id, version);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    condition TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, condition);
                """
            )

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def add_episode(self, episode: TaskEpisode) -> None:
        signature = episode.metadata.get("structural_signature", "")
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO episodes
                    (task_id, scope, success, verified, structural_signature, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.task_id,
                    episode.scope,
                    int(episode.success),
                    int(episode.verified),
                    signature,
                    episode.started_at,
                    self._dump(episode.to_dict()),
                ),
            )

    def get_episode(self, task_id: str) -> TaskEpisode | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM episodes WHERE task_id = ?", (task_id,)
            ).fetchone()
        return TaskEpisode.from_dict(json.loads(row["payload"])) if row else None

    def list_episodes(self, *, scope: str | None = None) -> list[TaskEpisode]:
        query = "SELECT payload FROM episodes"
        params: tuple[Any, ...] = ()
        if scope is not None:
            query += " WHERE scope IN (?, '*')"
            params = (scope,)
        query += " ORDER BY created_at"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [TaskEpisode.from_dict(json.loads(row["payload"])) for row in rows]

    def episodes_by_signature(self, signature: str, *, scope: str) -> list[TaskEpisode]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload FROM episodes
                WHERE structural_signature = ? AND scope IN (?, '*')
                    AND success = 1 AND verified = 1
                ORDER BY created_at
                """,
                (signature, scope),
            ).fetchall()
        return [TaskEpisode.from_dict(json.loads(row["payload"])) for row in rows]

    def upsert_workflow(self, workflow: WorkflowDefinition) -> None:
        workflow.updated_at = time.time()
        payload = self._dump(workflow.to_dict())
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO workflows
                    (workflow_id, scope, status, structural_signature, version, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    scope = excluded.scope,
                    status = excluded.status,
                    structural_signature = excluded.structural_signature,
                    version = excluded.version,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    workflow.workflow_id,
                    workflow.scope,
                    str(workflow.status),
                    workflow.structural_signature,
                    workflow.version,
                    workflow.updated_at,
                    payload,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO workflow_versions
                    (workflow_id, version, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (workflow.workflow_id, workflow.version, workflow.updated_at, payload),
            )

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
        return WorkflowDefinition.from_dict(json.loads(row["payload"])) if row else None

    def list_workflow_versions(self, workflow_id: str) -> list[WorkflowDefinition]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload FROM workflow_versions
                WHERE workflow_id = ? ORDER BY version
                """,
                (workflow_id,),
            ).fetchall()
        return [WorkflowDefinition.from_dict(json.loads(row["payload"])) for row in rows]

    def workflow_by_signature(self, signature: str, *, scope: str) -> WorkflowDefinition | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT payload FROM workflows
                WHERE structural_signature = ? AND scope IN (?, '*')
                ORDER BY updated_at DESC LIMIT 1
                """,
                (signature, scope),
            ).fetchone()
        return WorkflowDefinition.from_dict(json.loads(row["payload"])) if row else None

    def list_workflows(
        self,
        *,
        scope: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[WorkflowDefinition]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope IN (?, '*')")
            params.append(scope)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(sorted(statuses))
        query = "SELECT payload FROM workflows"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [WorkflowDefinition.from_dict(json.loads(row["payload"])) for row in rows]

    def add_feedback(self, feedback: WorkflowFeedback) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO feedback(workflow_id, decision, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    feedback.workflow_id,
                    str(feedback.decision),
                    feedback.created_at,
                    self._dump(feedback.to_dict()),
                ),
            )

    def add_metric(self, run_id: str, condition: str, payload: dict[str, Any]) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO metrics(run_id, task_id, condition, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload.get("task_id"),
                    condition,
                    time.time(),
                    self._dump(payload),
                ),
            )

    def close(self) -> None:
        """Kept for symmetry with remote stores; connections are operation-scoped."""
