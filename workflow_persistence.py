"""
Workflow Persistence Module

Handles persistence of workflow states to disk/database with support for:
- Multiple storage backends (file, SQLite, PostgreSQL)
- Automatic state versioning
- Compression for large states
- Integrity checking
- Concurrent access handling
"""

from __future__ import annotations

import os
import json
import gzip
import hashlib
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict
import threading
import platform

# Windows compatibility for file locking
if platform.system() != 'Windows':
    import fcntl  # Unix file locking
else:
    import msvcrt  # Windows file locking

from sovereign_data_models import WorkflowState, CheckpointInfo, AuditTrail, AuditEvent
import logging

logger = logging.getLogger(__name__)


def generate_workflow_id() -> str:
    """Generate a unique workflow ID."""
    import uuid
    return f"workflow_{uuid.uuid4().hex[:12]}"


def generate_state_id() -> str:
    """Generate a unique state ID."""
    import uuid
    return f"state_{uuid.uuid4().hex[:12]}"


def compute_checksum(data: str) -> str:
    """Compute SHA-256 checksum for integrity checking."""
    return hashlib.sha256(data.encode()).hexdigest()


class WorkflowPersistence:
    """
    Handles persistence of workflow states to disk/database.

    Features:
    - Automatic state versioning
    - Compression for large states
    - Integrity checking
    - Concurrent access handling
    """

    def __init__(self, storage_backend: str = "file", storage_path: str = "workflow_states"):
        """
        Initialize with storage backend.

        Args:
            storage_backend: "file", "sqlite", or "postgres"
            storage_path: Path for file-based storage or database directory
        """
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path)
        self.lock = threading.Lock()

        # Initialize storage backend
        if storage_backend == "file":
            self._init_file_backend()
        elif storage_backend == "sqlite":
            self._init_sqlite_backend()
        elif storage_backend == "postgres":
            self._init_postgres_backend()
        else:
            raise ValueError(f"Unknown storage backend: {storage_backend}")

    def _init_file_backend(self):
        """Initialize file-based storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.workflows_dir = self.storage_path / "workflows"
        self.checkpoints_dir = self.storage_path / "checkpoints"
        self.audit_dir = self.storage_path / "audit"
        self.workflows_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized file-based storage at {self.storage_path}")

    def _init_sqlite_backend(self):
        """Initialize SQLite backend."""
        self.db_path = self.storage_path / "workflow_states.db"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create tables
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_states (
                    workflow_id TEXT NOT NULL,
                    state_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_data TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    parent_state_id TEXT,
                    branch_name TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    checkpoint_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    state_size INTEGER NOT NULL,
                    parent_checkpoint_id TEXT,
                    branch_name TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_states(workflow_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trails (
                    workflow_id TEXT NOT NULL,
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_states(workflow_id)
                )
            """)

            conn.commit()

        logger.info(f"Initialized SQLite backend at {self.db_path}")

    def _init_postgres_backend(self):
        """Initialize PostgreSQL backend for production use."""
        try:
            import psycopg2
            from psycopg2.extras import Json
        except ImportError:
            logger.warning("psycopg2 not available. Falling back to file-based storage.")
            self.storage_backend = "file"
            self._init_file_backend()
            return

        # Get database connection parameters from environment or config
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "openevolve")
        db_user = os.getenv("POSTGRES_USER", "openevolve_user")
        db_password = os.getenv("POSTGRES_PASSWORD", "openevolve_pass")

        try:
            # Establish connection
            self.conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password
            )

            # Create tables if they don't exist
            cursor = self.conn.cursor()

            # Create workflow_states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_states (
                    state_id VARCHAR(255) PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active'
                );
            """)

            # Create workflow_checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    checkpoint_id VARCHAR(255) PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL,
                    state_id VARCHAR(255),
                    checkpoint_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parent_checkpoint_id VARCHAR(255),
                    branch_name VARCHAR(255)
                );
            """)

            # Create workflow_audit_trail table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_audit_trail (
                    event_id VARCHAR(255) PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255)
                );
            """)

            self.conn.commit()
            cursor.close()

            logger.info(f"PostgreSQL backend initialized successfully at {db_host}:{db_port}/{db_name}")
            self.storage_backend = "postgres"

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to initialize PostgreSQL backend: {e}. Falling back to file-based.")
            self.storage_backend = "file"
            self._init_file_backend()

    def persist_state(self, state: WorkflowState) -> str:
        """
        Persist state to storage.

        Args:
            state: WorkflowState to persist

        Returns:
            state_id of persisted state
        """
        with self.lock:
            try:
                state.updated_at = datetime.now()

                if self.storage_backend == "file":
                    return self._persist_state_file(state)
                elif self.storage_backend == "sqlite":
                    return self._persist_state_sqlite(state)
                else:
                    raise ValueError(f"Unknown storage backend: {self.storage_backend}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to persist state: {e}", exc_info=True)
                raise

    def _persist_state_file(self, state: WorkflowState) -> str:
        """Persist state to file."""
        state_dict = state.to_dict()
        state_json = json.dumps(state_dict, indent=2, default=str)

        # Compress if large (> 100KB)
        if len(state_json) > 100_000:
            state_json = gzip.compress(state_json.encode()).decode('latin1')
            is_compressed = True
        else:
            is_compressed = False

        # Compute checksum
        checksum = compute_checksum(state_json)

        # Create metadata
        metadata = {
            'state_id': state.state_id,
            'workflow_id': state.workflow_id,
            'version': state.version,
            'checksum': checksum,
            'compressed': is_compressed,
            'created_at': state.created_at.isoformat(),
            'updated_at': state.updated_at.isoformat()
        }

        # Write state file
        workflow_dir = self.workflows_dir / state.workflow_id
        workflow_dir.mkdir(exist_ok=True)

        state_file = workflow_dir / f"{state.state_id}.json"
        meta_file = workflow_dir / f"{state.state_id}.meta"

        # Write with file locking for concurrent access (cross-platform)
        with open(state_file, 'w') as f:
            if platform.system() != 'Windows':
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(state_json)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                # Windows file locking using msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    f.write(state_json)
                finally:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Persisted state {state.state_id} for workflow {state.workflow_id}")
        return state.state_id

    def _persist_state_sqlite(self, state: WorkflowState) -> str:
        """Persist state to SQLite."""
        state_dict = state.to_dict()
        state_json = json.dumps(state_dict, indent=2, default=str)
        checksum = compute_checksum(state_json)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workflow_states
                (workflow_id, state_id, version, state_data, checksum, created_at, updated_at, parent_state_id, branch_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.workflow_id,
                    state.state_id,
                    state.version,
                    state_json,
                    checksum,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    state.parent_state_id,
                    state.branch_name
                )
            )
            conn.commit()

        logger.debug(f"Persisted state {state.state_id} for workflow {state.workflow_id} to SQLite")
        return state.state_id

    def retrieve_state(
        self,
        workflow_id: str,
        state_id: str = None
    ) -> Optional[WorkflowState]:
        """
        Retrieve state from storage.

        Args:
            workflow_id: Workflow ID
            state_id: State ID (if None, returns latest state)

        Returns:
            WorkflowState or None if not found
        """
        with self.lock:
            try:
                if self.storage_backend == "file":
                    return self._retrieve_state_file(workflow_id, state_id)
                elif self.storage_backend == "sqlite":
                    return self._retrieve_state_sqlite(workflow_id, state_id)
                else:
                    raise ValueError(f"Unknown storage backend: {self.storage_backend}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to retrieve state: {e}", exc_info=True)
                return None

    def _retrieve_state_file(
        self,
        workflow_id: str,
        state_id: str = None
    ) -> Optional[WorkflowState]:
        """Retrieve state from file."""
        workflow_dir = self.workflows_dir / workflow_id

        if not workflow_dir.exists():
            return None

        # If state_id not specified, get latest
        if state_id is None:
            state_files = list(workflow_dir.glob("*.json"))
            if not state_files:
                return None
            # Get most recently modified
            state_file = max(state_files, key=lambda f: f.stat().st_mtime)
        else:
            state_file = workflow_dir / f"{state_id}.json"

        if not state_file.exists():
            return None

        # Read state
        with open(state_file, 'r') as f:
            state_json = f.read()

        # Verify checksum
        meta_file = state_file.with_suffix('.meta')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            expected_checksum = metadata.get('checksum')
            actual_checksum = compute_checksum(state_json)
            if expected_checksum and expected_checksum != actual_checksum:
                logger.error(f"Checksum mismatch for state {state_file}")
                return None

        # Decompress if needed
        if metadata.get('compressed'):
            state_json = gzip.decompress(state_json.encode('latin1')).decode()

        # Deserialize
        state_dict = json.loads(state_json)
        return WorkflowState.from_dict(state_dict)

    def _retrieve_state_sqlite(
        self,
        workflow_id: str,
        state_id: str = None
    ) -> Optional[WorkflowState]:
        """Retrieve state from SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if state_id is None:
                # Get latest state
                cursor = conn.execute(
                    """
                    SELECT state_data FROM workflow_states
                    WHERE workflow_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (workflow_id,)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT state_data FROM workflow_states
                    WHERE workflow_id = ? AND state_id = ?
                    """,
                    (workflow_id, state_id)
                )

            row = cursor.fetchone()
            if not row:
                return None

            state_json = row[0]

            # Verify checksum (simplified for SQLite)
            state_dict = json.loads(state_json)
            return WorkflowState.from_dict(state_dict)

    def list_workflow_states(self, workflow_id: str) -> List[WorkflowState]:
        """
        List all states for workflow (chronological).

        Args:
            workflow_id: Workflow ID

        Returns:
            List of WorkflowStates in chronological order
        """
        with self.lock:
            try:
                if self.storage_backend == "file":
                    return self._list_states_file(workflow_id)
                elif self.storage_backend == "sqlite":
                    return self._list_states_sqlite(workflow_id)
                else:
                    raise ValueError(f"Unknown storage backend: {self.storage_backend}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to list states: {e}", exc_info=True)
                return []

    def _list_states_file(self, workflow_id: str) -> List[WorkflowState]:
        """List states from file storage."""
        workflow_dir = self.workflows_dir / workflow_id

        if not workflow_dir.exists():
            return []

        states = []
        for state_file in workflow_dir.glob("*.json"):
            if state_file.with_suffix('.meta').exists():
                continue  # Skip metadata files

            state = self._retrieve_state_file(workflow_id, state_file.stem)
            if state:
                states.append(state)

        # Sort by creation time
        states.sort(key=lambda s: s.created_at)
        return states

    def _list_states_sqlite(self, workflow_id: str) -> List[WorkflowState]:
        """List states from SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT state_data FROM workflow_states
                WHERE workflow_id = ?
                ORDER BY created_at ASC
                """,
                (workflow_id,)
            )

            states = []
            for row in cursor.fetchall():
                state_dict = json.loads(row[0])
                states.append(WorkflowState.from_dict(state_dict))

            return states

    def delete_state(self, workflow_id: str, state_id: str):
        """
        Delete a state (for cleanup).

        Args:
            workflow_id: Workflow ID
            state_id: State ID to delete
        """
        with self.lock:
            try:
                if self.storage_backend == "file":
                    self._delete_state_file(workflow_id, state_id)
                elif self.storage_backend == "sqlite":
                    self._delete_state_sqlite(workflow_id, state_id)
                else:
                    raise ValueError(f"Unknown storage backend: {self.storage_backend}")

                logger.info(f"Deleted state {state_id} for workflow {workflow_id}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to delete state: {e}", exc_info=True)
                raise

    def _delete_state_file(self, workflow_id: str, state_id: str):
        """Delete state from file storage."""
        workflow_dir = self.workflows_dir / workflow_id
        state_file = workflow_dir / f"{state_id}.json"
        meta_file = workflow_dir / f"{state_id}.meta"

        if state_file.exists():
            state_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

    def _delete_state_sqlite(self, workflow_id: str, state_id: str):
        """Delete state from SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                DELETE FROM workflow_states
                WHERE workflow_id = ? AND state_id = ?
                """,
                (workflow_id, state_id)
            )
            conn.commit()

    def cleanup_old_states(self, workflow_id: str, keep_latest_n: int = 10):
        """
        Clean up old states, keeping only recent ones.

        Args:
            workflow_id: Workflow ID
            keep_latest_n: Number of recent states to keep
        """
        with self.lock:
            try:
                states = self.list_workflow_states(workflow_id)

                if len(states) <= keep_latest_n:
                    return

                # Delete oldest states
                states_to_delete = states[:-keep_latest_n]

                for state in states_to_delete:
                    self.delete_state(workflow_id, state.state_id)

                logger.info(f"Cleaned up {len(states_to_delete)} old states for workflow {workflow_id}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to cleanup old states: {e}", exc_info=True)
                raise

    def export_workflow(self, workflow_id: str, output_path: str):
        """
        Export complete workflow to archive.

        Creates a package with all states, checkpoints, and metadata
        for long-term storage.

        Args:
            workflow_id: Workflow ID to export
            output_path: Path for output archive (without extension)
        """
        with self.lock:
            try:
                output_path = Path(output_path)

                # Create temporary directory for export
                export_dir = output_path / "temp_export"
                export_dir.mkdir(parents=True, exist_ok=True)

                # Copy workflow states
                if self.storage_backend == "file":
                    workflow_dir = self.workflows_dir / workflow_id
                    if workflow_dir.exists():
                        shutil.copytree(workflow_dir, export_dir / "states")

                # Copy checkpoints
                checkpoint_dir = self.checkpoints_dir / workflow_id
                if checkpoint_dir.exists():
                    shutil.copytree(checkpoint_dir, export_dir / "checkpoints")

                # Copy audit trail
                audit_file = self.audit_dir / f"{workflow_id}.json"
                if audit_file.exists():
                    shutil.copy2(audit_file, export_dir / "audit_trail.json")

                # Create archive
                archive_path = str(output_path) + ".tar.gz"
                shutil.make_archive(str(output_path), "gztar", export_dir)

                # Cleanup temp directory
                shutil.rmtree(export_dir)

                logger.info(f"Exported workflow {workflow_id} to {archive_path}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to export workflow: {e}", exc_info=True)
                raise

    def import_workflow(self, archive_path: str) -> str:
        """
        Import workflow from archive.

        Args:
            archive_path: Path to workflow archive

        Returns:
            workflow_id of imported workflow
        """
        with self.lock:
            try:
                archive_path = Path(archive_path)

                # Extract archive
                temp_dir = self.storage_path / "temp_import"
                temp_dir.mkdir(exist_ok=True)
                shutil.unpack_archive(str(archive_path), temp_dir)

                # Find workflow ID from audit trail or states
                audit_file = temp_dir / "audit_trail.json"
                if audit_file.exists():
                    with open(audit_file, 'r') as f:
                        audit_data = json.load(f)
                    workflow_id = audit_data.get('workflow_id')
                else:
                    # Try to get from first state file
                    states_dir = temp_dir / "states"
                    if states_dir.exists():
                        state_files = list(states_dir.glob("*.json"))
                        if state_files:
                            with open(state_files[0], 'r') as f:
                                state_data = json.load(f)
                            workflow_id = state_data.get('workflow_id')
                        else:
                            raise ValueError("No states found in archive")
                    else:
                        raise ValueError("Invalid workflow archive")

                # Copy to appropriate locations
                if (temp_dir / "states").exists():
                    dest_dir = self.workflows_dir / workflow_id
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(temp_dir / "states", dest_dir)

                if (temp_dir / "checkpoints").exists():
                    dest_dir = self.checkpoints_dir / workflow_id
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(temp_dir / "checkpoints", dest_dir)

                if audit_file.exists():
                    shutil.copy2(audit_file, self.audit_dir / f"{workflow_id}.json")

                # Cleanup
                shutil.rmtree(temp_dir)

                logger.info(f"Imported workflow {workflow_id} from {archive_path}")
                return workflow_id
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to import workflow: {e}", exc_info=True)
                raise

    def save_checkpoint(self, checkpoint: CheckpointInfo):
        """Save checkpoint metadata."""
        with self.lock:
            try:
                if self.storage_backend == "file":
                    self._save_checkpoint_file(checkpoint)
                elif self.storage_backend == "sqlite":
                    self._save_checkpoint_sqlite(checkpoint)

                logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
                raise

    def _save_checkpoint_file(self, checkpoint: CheckpointInfo):
        """Save checkpoint to file."""
        workflow_dir = self.checkpoints_dir / checkpoint.workflow_id
        workflow_dir.mkdir(exist_ok=True)

        checkpoint_file = workflow_dir / f"{checkpoint.checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

    def _save_checkpoint_sqlite(self, checkpoint: CheckpointInfo):
        """Save checkpoint to SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (checkpoint_id, workflow_id, checkpoint_name, created_at, stage, progress, state_size, parent_checkpoint_id, branch_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.workflow_id,
                    checkpoint.checkpoint_name,
                    checkpoint.created_at.isoformat(),
                    checkpoint.stage,
                    checkpoint.progress,
                    checkpoint.state_size,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.branch_name
                )
            )
            conn.commit()

    def list_checkpoints(self, workflow_id: str) -> List[CheckpointInfo]:
        """List all checkpoints for a workflow."""
        with self.lock:
            try:
                if self.storage_backend == "file":
                    return self._list_checkpoints_file(workflow_id)
                elif self.storage_backend == "sqlite":
                    return self._list_checkpoints_sqlite(workflow_id)
                else:
                    return []
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to list checkpoints: {e}", exc_info=True)
                return []

    def _list_checkpoints_file(self, workflow_id: str) -> List[CheckpointInfo]:
        """List checkpoints from file storage."""
        workflow_dir = self.checkpoints_dir / workflow_id

        if not workflow_dir.exists():
            return []

        checkpoints = []
        for checkpoint_file in workflow_dir.glob("*.json"):
            with open(checkpoint_file, 'r') as f:
                checkpoint_dict = json.load(f)
            checkpoints.append(CheckpointInfo.from_dict(checkpoint_dict))

        # Sort by creation time
        checkpoints.sort(key=lambda c: c.created_at)
        return checkpoints

    def _list_checkpoints_sqlite(self, workflow_id: str) -> List[CheckpointInfo]:
        """List checkpoints from SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT checkpoint_id, workflow_id, checkpoint_name, created_at, stage, progress, state_size, parent_checkpoint_id, branch_name
                FROM checkpoints
                WHERE workflow_id = ?
                ORDER BY created_at ASC
                """,
                (workflow_id,)
            )

            checkpoints = []
            for row in cursor.fetchall():
                checkpoint_dict = {
                    'checkpoint_id': row[0],
                    'workflow_id': row[1],
                    'checkpoint_name': row[2],
                    'created_at': datetime.fromisoformat(row[3]),
                    'stage': row[4],
                    'progress': row[5],
                    'state_size': row[6],
                    'parent_checkpoint_id': row[7],
                    'branch_name': row[8]
                }
                checkpoints.append(CheckpointInfo(**checkpoint_dict))

            return checkpoints

    def save_audit_trail(self, audit_trail: AuditTrail):
        """Save audit trail."""
        with self.lock:
            try:
                audit_file = self.audit_dir / f"{audit_trail.workflow_id}.json"
                with open(audit_file, 'w') as f:
                    json.dump(audit_trail.to_dict(), f, indent=2, default=str)

                logger.debug(f"Saved audit trail for workflow {audit_trail.workflow_id}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to save audit trail: {e}", exc_info=True)
                raise

    def load_audit_trail(self, workflow_id: str) -> Optional[AuditTrail]:
        """Load audit trail."""
        with self.lock:
            try:
                audit_file = self.audit_dir / f"{workflow_id}.json"

                if not audit_file.exists():
                    return None

                with open(audit_file, 'r') as f:
                    audit_dict = json.load(f)

                return AuditTrail.from_dict(audit_dict)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to load audit trail: {e}", exc_info=True)
                return None
