"""
Traceability Storage Backend

Provides persistent storage for traceability data with support for
multiple backends (in-memory, SQLite, PostgreSQL).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from .traceability import Change, ChangeTrace, Modification

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Storage backend configuration"""
    backend_type: str = "memory"  # memory, sqlite, postgresql
    connection_string: Optional[str] = None
    table_prefix: str = "traceability"
    create_tables: bool = True


class TraceabilityRepository:
    """
    Repository for storing and retrieving traceability data.
    """

    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self._backend = self._create_backend()

    def _create_backend(self):
        """Create storage backend based on configuration"""
        if self.config.backend_type == "memory":
            return InMemoryBackend()
        elif self.config.backend_type == "sqlite":
            return SQLiteBackend(self.config)
        elif self.config.backend_type == "postgresql":
            return PostgreSQLBackend(self.config)
        else:
            raise ValueError(f"Unknown backend type: {self.config.backend_type}")

    def save_change(self, change: Change, problem_id: str) -> bool:
        """Save a change to storage"""
        return self._backend.save_change(change, problem_id)

    def save_modification(self, modification: Modification, problem_id: str) -> bool:
        """Save a modification to storage"""
        return self._backend.save_modification(modification, problem_id)

    def get_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get complete trace for a problem"""
        return self._backend.get_trace(problem_id)

    def get_changes_by_problem(self, problem_id: str) -> List[Change]:
        """Get all changes for a problem"""
        return self._backend.get_changes_by_problem(problem_id)

    def get_changes_by_team(self, problem_id: str, team: str) -> List[Change]:
        """Get changes by team"""
        return self._backend.get_changes_by_team(problem_id, team)

    def get_changes_by_time_range(
        self,
        problem_id: str,
        start: datetime,
        end: datetime
    ) -> List[Change]:
        """Get changes within time range"""
        return self._backend.get_changes_by_time_range(problem_id, start, end)

    def get_all_traces(self) -> Dict[str, ChangeTrace]:
        """Get all traces"""
        return self._backend.get_all_traces()

    def delete_trace(self, problem_id: str) -> bool:
        """Delete trace for a problem"""
        return self._backend.delete_trace(problem_id)

    def clear_all(self) -> bool:
        """Clear all traces"""
        return self._backend.clear_all()

    @contextmanager
    def transaction(self):
        """Execute operations in a transaction"""
        yield self._backend.transaction()


class InMemoryBackend:
    """In-memory storage backend for testing"""

    def __init__(self):
        self.traces: Dict[str, ChangeTrace] = {}
        self.modifications: Dict[str, List[Modification]] = {}

    def save_change(self, change: Change, problem_id: str) -> bool:
        """Save change to memory"""
        if problem_id not in self.traces:
            self.traces[problem_id] = ChangeTrace(problem_id=problem_id)

        self.traces[problem_id].changes.append(change)
        self.traces[problem_id].updated_at = datetime.utcnow()
        return True

    def save_modification(self, modification: Modification, problem_id: str) -> bool:
        """Save modification to memory"""
        if problem_id not in self.modifications:
            self.modifications[problem_id] = []

        self.modifications[problem_id].append(modification)
        return True

    def get_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get trace from memory"""
        if problem_id in self.traces:
            trace = self.traces[problem_id]
            # Add modifications
            if problem_id in self.modifications:
                trace.modifications = self.modifications[problem_id]
            return trace
        return None

    def get_changes_by_problem(self, problem_id: str) -> List[Change]:
        """Get changes for problem"""
        if problem_id in self.traces:
            return self.traces[problem_id].changes
        return []

    def get_changes_by_team(self, problem_id: str, team: str) -> List[Change]:
        """Get changes by team"""
        changes = self.get_changes_by_problem(problem_id)
        return [c for c in changes if c.team == team]

    def get_changes_by_time_range(
        self,
        problem_id: str,
        start: datetime,
        end: datetime
    ) -> List[Change]:
        """Get changes in time range"""
        changes = self.get_changes_by_problem(problem_id)
        return [c for c in changes if start <= c.timestamp <= end]

    def get_all_traces(self) -> Dict[str, ChangeTrace]:
        """Get all traces"""
        return self.traces.copy()

    def delete_trace(self, problem_id: str) -> bool:
        """Delete trace"""
        if problem_id in self.traces:
            del self.traces[problem_id]
            if problem_id in self.modifications:
                del self.modifications[problem_id]
            return True
        return False

    def clear_all(self) -> bool:
        """Clear all traces"""
        self.traces.clear()
        self.modifications.clear()
        return True

    @contextmanager
    def transaction(self):
        """No-op transaction for in-memory"""
        yield self


class SQLiteBackend:
    """SQLite storage backend for persistent storage"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = config.connection_string or ":memory:"
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS changes (
                    change_id TEXT PRIMARY KEY,
                    problem_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    team TEXT NOT NULL,
                    author TEXT,
                    change_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    before TEXT,
                    after TEXT,
                    diff TEXT,
                    metadata TEXT,
                    INDEX idx_problem (problem_id),
                    INDEX idx_team (team),
                    INDEX idx_timestamp (timestamp)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS modifications (
                    modification_id TEXT PRIMARY KEY,
                    change_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    section TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    before_value TEXT,
                    after_value TEXT,
                    line_number INTEGER,
                    reason TEXT,
                    FOREIGN KEY (change_id) REFERENCES changes(change_id),
                    INDEX idx_change (change_id),
                    INDEX idx_problem (problem_id)
                )
            """)

            conn.commit()

    def save_change(self, change: Change, problem_id: str) -> bool:
        """Save change to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO changes (
                        change_id, problem_id, timestamp, team, author,
                        change_type, description, before, after, diff, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    change.change_id,
                    problem_id,
                    change.timestamp.isoformat(),
                    change.team,
                    change.author,
                    change.change_type,
                    change.description,
                    json.dumps(change.before) if change.before else None,
                    json.dumps(change.after) if change.after else None,
                    change.diff,
                    json.dumps(change.metadata) if change.metadata else None,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save change: {e}")
            return False

    def save_modification(self, modification: Modification, problem_id: str) -> bool:
        """Save modification to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO modifications (
                        modification_id, change_id, problem_id, section,
                        operation, before_value, after_value, line_number, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    modification.modification_id,
                    modification.change_id,
                    problem_id,
                    modification.section,
                    modification.operation,
                    json.dumps(modification.before_value) if modification.before_value else None,
                    json.dumps(modification.after_value) if modification.after_value else None,
                    modification.line_number,
                    modification.reason,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save modification: {e}")
            return False

    def get_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get trace from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get changes
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT change_id, timestamp, team, author, change_type,
                           description, before, after, diff, metadata
                    FROM changes
                    WHERE problem_id = ?
                    ORDER BY timestamp ASC
                """, (problem_id,))

                changes = []
                for row in cursor.fetchall():
                    change = Change(
                        change_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        team=row[2],
                        author=row[3],
                        change_type=row[4],
                        description=row[5],
                        before=json.loads(row[6]) if row[6] else None,
                        after=json.loads(row[7]) if row[7] else None,
                        diff=row[8],
                        metadata=json.loads(row[9]) if row[9] else None,
                    )
                    changes.append(change)

                # Get modifications
                cursor.execute("""
                    SELECT modification_id, change_id, section, operation,
                           before_value, after_value, line_number, reason
                    FROM modifications
                    WHERE problem_id = ?
                """, (problem_id,))

                modifications = []
                for row in cursor.fetchall():
                    mod = Modification(
                        modification_id=row[0],
                        change_id=row[1],
                        section=row[2],
                        operation=row[3],
                        before_value=json.loads(row[4]) if row[4] else None,
                        after_value=json.loads(row[5]) if row[5] else None,
                        line_number=row[6],
                        reason=row[7],
                    )
                    modifications.append(mod)

                if changes:
                    return ChangeTrace(
                        problem_id=problem_id,
                        changes=changes,
                        modifications=modifications,
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get trace: {e}")
            return None

    def get_changes_by_problem(self, problem_id: str) -> List[Change]:
        """Get changes for problem"""
        trace = self.get_trace(problem_id)
        return trace.changes if trace else []

    def get_changes_by_team(self, problem_id: str, team: str) -> List[Change]:
        """Get changes by team"""
        changes = self.get_changes_by_problem(problem_id)
        return [c for c in changes if c.team == team]

    def get_changes_by_time_range(
        self,
        problem_id: str,
        start: datetime,
        end: datetime
    ) -> List[Change]:
        """Get changes in time range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT change_id, timestamp, team, author, change_type,
                           description, before, after, diff, metadata
                    FROM changes
                    WHERE problem_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, (problem_id, start.isoformat(), end.isoformat()))

                changes = []
                for row in cursor.fetchall():
                    change = Change(
                        change_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        team=row[2],
                        author=row[3],
                        change_type=row[4],
                        description=row[5],
                        before=json.loads(row[6]) if row[6] else None,
                        after=json.loads(row[7]) if row[7] else None,
                        diff=row[8],
                        metadata=json.loads(row[9]) if row[9] else None,
                    )
                    changes.append(change)

                return changes
        except Exception as e:
            logger.error(f"Failed to get changes by time range: {e}")
            return []

    def get_all_traces(self) -> Dict[str, ChangeTrace]:
        """Get all traces"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT problem_id FROM changes")

                traces = {}
                for (problem_id,) in cursor.fetchall():
                    trace = self.get_trace(problem_id)
                    if trace:
                        traces[problem_id] = trace

                return traces
        except Exception as e:
            logger.error(f"Failed to get all traces: {e}")
            return {}

    def delete_trace(self, problem_id: str) -> bool:
        """Delete trace"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM modifications WHERE problem_id = ?", (problem_id,))
                conn.execute("DELETE FROM changes WHERE problem_id = ?", (problem_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete trace: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all traces"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM modifications")
                conn.execute("DELETE FROM changes")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to clear all: {e}")
            return False

    @contextmanager
    def transaction(self):
        """SQLite transaction context"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class PostgreSQLBackend:
    """PostgreSQL storage backend"""

    def __init__(self, config: StorageConfig):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL backend. Install with: pip install psycopg2-binary")
        
        self.config = config
        self.connection_string = config.connection_string
        self.table_prefix = config.table_prefix
        self._conn_pool = None
        
        # Initialize database tables if requested
        if config.create_tables:
            self._create_tables()

    def _get_connection(self):
        """Get a database connection"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)

    def _create_tables(self):
        """Create required tables if they don't exist"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create changes table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_changes (
                        id SERIAL PRIMARY KEY,
                        problem_id VARCHAR(255) NOT NULL,
                        change_id VARCHAR(255) NOT NULL UNIQUE,
                        team VARCHAR(100),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        change_type VARCHAR(50),
                        content TEXT,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create modifications table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_modifications (
                        id SERIAL PRIMARY KEY,
                        problem_id VARCHAR(255) NOT NULL,
                        modification_id VARCHAR(255) NOT NULL UNIQUE,
                        change_id VARCHAR(255),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        modification_type VARCHAR(50),
                        content TEXT,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_changes_problem_id ON {self.table_prefix}_changes(problem_id);
                    CREATE INDEX IF NOT EXISTS idx_changes_team ON {self.table_prefix}_changes(team);
                    CREATE INDEX IF NOT EXISTS idx_changes_timestamp ON {self.table_prefix}_changes(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_modifications_problem_id ON {self.table_prefix}_modifications(problem_id);
                    CREATE INDEX IF NOT EXISTS idx_modifications_change_id ON {self.table_prefix}_modifications(change_id);
                """)
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def save_change(self, change: Change, problem_id: str) -> bool:
        """Save a change to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_prefix}_changes 
                    (problem_id, change_id, team, timestamp, change_type, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (change_id) DO UPDATE SET
                        team = EXCLUDED.team,
                        timestamp = EXCLUDED.timestamp,
                        change_type = EXCLUDED.change_type,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """, (
                    problem_id,
                    change.id,
                    getattr(change, 'team', ''),
                    getattr(change, 'timestamp', datetime.now()),
                    getattr(change, 'type', ''),
                    json.dumps(asdict(change)) if hasattr(change, '__dataclass_fields__') else str(change),
                    json.dumps(getattr(change, 'metadata', {}))
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save change: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def save_modification(self, modification: Modification, problem_id: str) -> bool:
        """Save a modification to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_prefix}_modifications 
                    (problem_id, modification_id, change_id, timestamp, modification_type, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (modification_id) DO UPDATE SET
                        change_id = EXCLUDED.change_id,
                        timestamp = EXCLUDED.timestamp,
                        modification_type = EXCLUDED.modification_type,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """, (
                    problem_id,
                    modification.id,
                    getattr(modification, 'change_id', ''),
                    getattr(modification, 'timestamp', datetime.now()),
                    getattr(modification, 'type', ''),
                    json.dumps(asdict(modification)) if hasattr(modification, '__dataclass_fields__') else str(modification),
                    json.dumps(getattr(modification, 'metadata', {}))
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save modification: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get trace for a specific problem"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_prefix}_changes 
                    WHERE problem_id = %s 
                    ORDER BY timestamp ASC;
                """, (problem_id,))
                
                rows = cur.fetchall()
                if not rows:
                    return None
                
                # Reconstruct ChangeTrace object from stored changes
                changes = []
                for row in rows:
                    # Parse the stored JSON back to a Change object
                    change_data = json.loads(row['content'])
                    # Create a basic Change-like object from the stored data
                    change_obj = Change(
                        id=row['change_id'],
                        timestamp=row['timestamp'],
                        type=row['change_type'],
                        content=change_data.get('content', ''),
                        metadata=row['metadata']
                    )
                    changes.append(change_obj)
                
                # Create a basic ChangeTrace object
                trace = ChangeTrace(changes=changes)
                return trace
        except Exception as e:
            logger.error(f"Failed to get trace: {e}")
            return None
        finally:
            conn.close()

    def get_changes_by_problem(self, problem_id: str) -> List[Change]:
        """Get all changes for a specific problem"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_prefix}_changes 
                    WHERE problem_id = %s 
                    ORDER BY timestamp ASC;
                """, (problem_id,))
                
                rows = cur.fetchall()
                changes = []
                for row in rows:
                    # Parse the stored JSON back to a Change object
                    change_data = json.loads(row['content'])
                    change_obj = Change(
                        id=row['change_id'],
                        timestamp=row['timestamp'],
                        type=row['change_type'],
                        content=change_data.get('content', ''),
                        metadata=row['metadata']
                    )
                    changes.append(change_obj)
                
                return changes
        except Exception as e:
            logger.error(f"Failed to get changes by problem: {e}")
            return []
        finally:
            conn.close()

    def get_changes_by_team(self, problem_id: str, team: str) -> List[Change]:
        """Get all changes for a specific problem and team"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_prefix}_changes 
                    WHERE problem_id = %s AND team = %s
                    ORDER BY timestamp ASC;
                """, (problem_id, team))
                
                rows = cur.fetchall()
                changes = []
                for row in rows:
                    # Parse the stored JSON back to a Change object
                    change_data = json.loads(row['content'])
                    change_obj = Change(
                        id=row['change_id'],
                        timestamp=row['timestamp'],
                        type=row['change_type'],
                        content=change_data.get('content', ''),
                        metadata=row['metadata']
                    )
                    changes.append(change_obj)
                
                return changes
        except Exception as e:
            logger.error(f"Failed to get changes by team: {e}")
            return []
        finally:
            conn.close()

    def get_changes_by_time_range(
        self,
        problem_id: str,
        start: datetime,
        end: datetime
    ) -> List[Change]:
        """Get changes for a problem within a time range"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_prefix}_changes 
                    WHERE problem_id = %s 
                      AND timestamp >= %s 
                      AND timestamp <= %s
                    ORDER BY timestamp ASC;
                """, (problem_id, start, end))
                
                rows = cur.fetchall()
                changes = []
                for row in rows:
                    # Parse the stored JSON back to a Change object
                    change_data = json.loads(row['content'])
                    change_obj = Change(
                        id=row['change_id'],
                        timestamp=row['timestamp'],
                        type=row['change_type'],
                        content=change_data.get('content', ''),
                        metadata=row['metadata']
                    )
                    changes.append(change_obj)
                
                return changes
        except Exception as e:
            logger.error(f"Failed to get changes by time range: {e}")
            return []
        finally:
            conn.close()

    def get_all_traces(self) -> Dict[str, ChangeTrace]:
        """Get all traces from the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT DISTINCT problem_id FROM {self.table_prefix}_changes;
                """)
                
                problem_ids = [row['problem_id'] for row in cur.fetchall()]
                all_traces = {}
                
                for problem_id in problem_ids:
                    trace = self.get_trace(problem_id)
                    if trace:
                        all_traces[problem_id] = trace
                
                return all_traces
        except Exception as e:
            logger.error(f"Failed to get all traces: {e}")
            return {}
        finally:
            conn.close()

    def delete_trace(self, problem_id: str) -> bool:
        """Delete trace for a specific problem"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM {self.table_prefix}_changes 
                    WHERE problem_id = %s;
                """, (problem_id,))
                
                deleted_rows = cur.rowcount
                conn.commit()
                
                # Also delete associated modifications
                cur.execute(f"""
                    DELETE FROM {self.table_prefix}_modifications 
                    WHERE problem_id = %s;
                """, (problem_id,))
                
                conn.commit()
                return deleted_rows > 0
        except Exception as e:
            logger.error(f"Failed to delete trace: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def clear_all(self) -> bool:
        """Clear all traces from the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_prefix}_changes;")
                changes_deleted = cur.rowcount
                cur.execute(f"DELETE FROM {self.table_prefix}_modifications;")
                modifications_deleted = cur.rowcount
                conn.commit()
                
                logger.info(f"Cleared {changes_deleted} changes and {modifications_deleted} modifications")
                return True
        except Exception as e:
            logger.error(f"Failed to clear all: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """PostgreSQL transaction context"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# Convenience functions
def create_repository(
    backend_type: str = "memory",
    connection_string: Optional[str] = None
) -> TraceabilityRepository:
    """Create traceability repository"""
    config = StorageConfig(
        backend_type=backend_type,
        connection_string=connection_string
    )
    return TraceabilityRepository(config)


def get_in_memory_repository() -> TraceabilityRepository:
    """Get in-memory repository"""
    return create_repository("memory")


def get_sqlite_repository(db_path: str = ":memory:") -> TraceabilityRepository:
    """Get SQLite repository"""
    return create_repository("sqlite", db_path)
