"""
Phi 1.5 Failure Database

Persistent storage for failure patterns, inferred assumptions, and
historical paradigm shift data.

Author: Agent B1 (Phi 1/Phi 1.5 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
"""

import sys
import sqlite3
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))
from tacit_assumption_miner import (
    NullResult, FailureFeatures, TacitAssumption,
    ParadigmShiftRecommendation, ErrorType, AssumptionType, PatternType
)


class FailureDatabase:
    """
    SQLite database for storing failures, assumptions, and paradigm history.

    Provides:
    - Persistent storage of failure patterns
    - Historical paradigm shift database
    - Efficient querying and indexing
    - Caching for performance
    """

    def __init__(self, db_path: str = "rese/data/phi15_failures.db",
                 cache_size: int = 1000):
        """
        Initialize failure database.

        Args:
            db_path: Path to SQLite database file
            cache_size: Maximum number of items to cache per category
        """
        self.db_path = db_path
        self.conn = None
        self.cache_size = cache_size
        # Use OrderedDict for LRU cache
        self.cache = {
            'failures': OrderedDict(),
            'assumptions': OrderedDict(),
            'paradigms': OrderedDict()
        }

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect and create tables
        self._connect()
        self._create_tables()

    def __del__(self):
        """Cleanup on deletion"""
        # Only close if we still have a connection and haven't been explicitly closed
        if hasattr(self, 'conn') and self.conn is not None:
            try:
                # Check if connection is still open
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
            except:
                # Connection is already closed or invalid
                pass
            else:
                # Connection is still open, close it
                try:
                    self.conn.close()
                except:
                    pass

    def _connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Failures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                attempt_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                approach_type TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                state_json TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                resources_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                constraints_json TEXT NOT NULL,
                feature_vector_json TEXT,
                keywords_json TEXT,
                failure_cluster INTEGER,
                anomaly_score REAL,
                processed BOOLEAN DEFAULT FALSE
            )
        """)

        # Failure features table (denormalized for performance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_features (
                attempt_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                approach_type TEXT NOT NULL,
                error_type TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                time_to_failure REAL NOT NULL,
                error_magnitude REAL,
                resource_consumption REAL NOT NULL,
                constraint_violation_count INTEGER NOT NULL,
                feature_vector_json TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                failure_cluster INTEGER,
                anomaly_score REAL,
                FOREIGN KEY (attempt_id) REFERENCES failures(attempt_id)
            )
        """)

        # Assumptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assumptions (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                formalization TEXT NOT NULL,
                assumption_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                support INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                constraint_relaxation TEXT NOT NULL,
                paradigm_implication BOOLEAN NOT NULL,
                alternative_paradigm TEXT,
                timestamp TEXT NOT NULL,
                verified BOOLEAN DEFAULT FALSE
            )
        """)

        # Paradigm shifts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paradigm_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                assumptions_json TEXT NOT NULL,
                alternatives_json TEXT NOT NULL,
                explanation TEXT NOT NULL
            )
        """)

        # Historical paradigm shifts (reference data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_paradigm_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                year INTEGER NOT NULL,
                domain TEXT NOT NULL,
                old_paradigm TEXT NOT NULL,
                new_paradigm TEXT NOT NULL,
                tacit_assumption TEXT NOT NULL,
                description TEXT NOT NULL,
                failure_pattern_json TEXT NOT NULL
            )
        """)

        # Indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failures_timestamp
            ON failures(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failures_problem_type
            ON failures(problem_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failures_error_type
            ON failures(error_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assumptions_confidence
            ON assumptions(confidence)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assumptions_timestamp
            ON assumptions(timestamp)
        """)

        self.conn.commit()

    # ========================================================================
    # Failure Operations
    # ========================================================================

    def add_failure(self, null_result: NullResult,
                   features: Optional[FailureFeatures] = None) -> None:
        """
        Add a failure to the database.

        Args:
            null_result: Null result from Stage 6
            features: Extracted features (optional, computed if None)
        """
        cursor = self.conn.cursor()

        # Convert to JSON
        state_json = json.dumps(null_result.state)
        resources_json = json.dumps(null_result.resources_used)
        metadata_json = json.dumps(null_result.metadata)
        constraints_json = json.dumps(null_result.constraints)

        # Insert null result
        cursor.execute("""
            INSERT OR REPLACE INTO failures
            (attempt_id, timestamp, problem_type, approach_type, error_type,
             error_message, state_json, iteration, resources_json, metadata_json,
             constraints_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            null_result.attempt_id,
            null_result.timestamp.isoformat(),
            null_result.problem_type,
            null_result.approach_type,
            null_result.error_type.value,
            null_result.error_message,
            state_json,
            null_result.iteration,
            resources_json,
            metadata_json,
            constraints_json
        ))

        # Add features if provided
        if features:
            self.add_failure_features(features)

        self.conn.commit()

        # Update cache with LRU eviction
        cache = self.cache['failures']
        cache[null_result.attempt_id] = null_result
        # Move to end (most recently used)
        cache.move_to_end(null_result.attempt_id)
        # Evict oldest if over limit
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

    def add_failure_features(self, features: FailureFeatures) -> None:
        """Add failure features to database"""
        cursor = self.conn.cursor()

        feature_vector_json = json.dumps(features.feature_vector.tolist())
        keywords_json = json.dumps(features.keywords)

        cursor.execute("""
            INSERT OR REPLACE INTO failure_features
            (attempt_id, timestamp, problem_type, approach_type, error_type,
             iteration, time_to_failure, error_magnitude, resource_consumption,
             constraint_violation_count, feature_vector_json, keywords_json,
             failure_cluster, anomaly_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            features.attempt_id,
            features.timestamp.isoformat(),
            features.problem_type,
            features.approach_type,
            features.error_type.value,
            features.iteration,
            features.time_to_failure,
            features.error_magnitude,
            features.resource_consumption,
            features.constraint_violation_count,
            feature_vector_json,
            keywords_json,
            features.failure_cluster,
            features.anomaly_score
        ))

        self.conn.commit()

    def get_failure(self, attempt_id: str) -> Optional[NullResult]:
        """Get a failure by attempt ID"""
        # Check cache first (with LRU update)
        cache = self.cache['failures']
        if attempt_id in cache:
            # Move to end (most recently used)
            cache.move_to_end(attempt_id)
            return cache[attempt_id]

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM failures WHERE attempt_id = ?
        """, (attempt_id,))

        row = cursor.fetchone()
        if not row:
            return None

        null_result = self._row_to_null_result(row)
        # Add to cache with LRU eviction
        cache[attempt_id] = null_result
        cache.move_to_end(attempt_id)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

        return null_result

    def get_failures_since(self, timestamp: datetime) -> List[NullResult]:
        """Get all failures since a given timestamp"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM failures
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (timestamp.isoformat(),))

        rows = cursor.fetchall()
        return [self._row_to_null_result(row) for row in rows]

    def get_recent_failures(self, hours: int = 24) -> List[NullResult]:
        """Get failures from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return self.get_failures_since(cutoff)

    def get_unprocessed_failures(self) -> List[NullResult]:
        """Get failures that haven't been processed yet"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM failures
            WHERE processed = FALSE
            ORDER BY timestamp ASC
        """)

        rows = cursor.fetchall()
        return [self._row_to_null_result(row) for row in rows]

    def mark_as_processed(self, attempt_id: str) -> None:
        """Mark a failure as processed"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE failures
            SET processed = TRUE
            WHERE attempt_id = ?
        """, (attempt_id,))
        self.conn.commit()

    def get_failure_count(self) -> int:
        """Get total number of failures in database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM failures")
        return cursor.fetchone()[0]

    # ========================================================================
    # Assumption Operations
    # ========================================================================

    def add_assumption(self, assumption: TacitAssumption) -> None:
        """Add an assumption to the database"""
        cursor = self.conn.cursor()

        evidence_json = json.dumps(assumption.evidence)

        cursor.execute("""
            INSERT OR REPLACE INTO assumptions
            (id, description, formalization, assumption_type, confidence,
             support, evidence_json, pattern_type, constraint_relaxation,
             paradigm_implication, alternative_paradigm, timestamp, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assumption.id,
            assumption.description,
            assumption.formalization,
            assumption.assumption_type.value,
            assumption.confidence,
            assumption.support,
            evidence_json,
            assumption.pattern_type.value,
            assumption.constraint_relaxation,
            assumption.paradigm_implication,
            assumption.alternative_paradigm,
            assumption.timestamp.isoformat(),
            assumption.verified
        ))

        self.conn.commit()

        # Update cache with LRU eviction
        cache = self.cache['assumptions']
        cache[assumption.id] = assumption
        cache.move_to_end(assumption.id)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

    def get_assumption(self, assumption_id: str) -> Optional[TacitAssumption]:
        """Get an assumption by ID"""
        # Check cache first (with LRU update)
        cache = self.cache['assumptions']
        if assumption_id in cache:
            cache.move_to_end(assumption_id)
            return cache[assumption_id]

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM assumptions WHERE id = ?
        """, (assumption_id,))

        row = cursor.fetchone()
        if not row:
            return None

        assumption = self._row_to_assumption(row)
        # Add to cache with LRU eviction
        cache[assumption_id] = assumption
        cache.move_to_end(assumption_id)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

        return assumption

    def get_recent_assumptions(self, days: int = 30) -> List[TacitAssumption]:
        """Get assumptions from last N days"""
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT * FROM assumptions
            WHERE timestamp >= ?
            ORDER BY confidence DESC
        """, (cutoff.isoformat(),))

        rows = cursor.fetchall()
        return [self._row_to_assumption(row) for row in rows]

    def get_high_confidence_assumptions(self,
                                      min_confidence: float = 0.7) -> List[TacitAssumption]:
        """Get assumptions with confidence above threshold"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM assumptions
            WHERE confidence >= ?
            ORDER BY confidence DESC
        """, (min_confidence,))

        rows = cursor.fetchall()
        return [self._row_to_assumption(row) for row in rows]

    def update_assumption_confidence(self, assumption_id: str,
                                    new_confidence: float) -> None:
        """Update assumption confidence (e.g., after validation)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE assumptions
            SET confidence = ?, verified = TRUE
            WHERE id = ?
        """, (new_confidence, assumption_id))

        self.conn.commit()

        # Invalidate cache
        if assumption_id in self.cache['assumptions']:
            del self.cache['assumptions'][assumption_id]

    # ========================================================================
    # Paradigm Shift Operations
    # ========================================================================

    def add_paradigm_shift(self,
                          recommendation: ParadigmShiftRecommendation) -> None:
        """Add a paradigm shift recommendation to database"""
        cursor = self.conn.cursor()

        assumptions_json = json.dumps([a.id for a in recommendation.primary_assumptions])
        alternatives_json = json.dumps(recommendation.suggested_alternatives)

        cursor.execute("""
            INSERT INTO paradigm_shifts
            (timestamp, trigger, confidence, assumptions_json,
             alternatives_json, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            recommendation.timestamp.isoformat(),
            recommendation.trigger,
            recommendation.confidence,
            assumptions_json,
            alternatives_json,
            recommendation.explanation
        ))

        self.conn.commit()

    def get_recent_paradigm_shifts(self, days: int = 365) -> List[Dict]:
        """Get paradigm shift recommendations from last N days"""
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT * FROM paradigm_shifts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff.isoformat(),))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # Historical Paradigm Shifts (Reference Data)
    # ========================================================================

    def load_historical_paradigm_shifts(self, filepath: str) -> None:
        """Load historical paradigm shifts from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        cursor = self.conn.cursor()

        for shift in data['paradigm_shifts']:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO historical_paradigm_shifts
                    (name, year, domain, old_paradigm, new_paradigm,
                     tacit_assumption, description, failure_pattern_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    shift['name'],
                    shift['year'],
                    shift['domain'],
                    shift['old_paradigm'],
                    shift['new_paradigm'],
                    shift['tacit_assumption'],
                    shift['description'],
                    json.dumps(shift.get('failure_pattern', {}))
                ))
            except sqlite3.IntegrityError:
                # Duplicate entry, skip
                pass

        self.conn.commit()

    def find_similar_historical_shifts(self, pattern: Dict,
                                      limit: int = 5) -> List[Dict]:
        """Find similar paradigm shifts in history"""
        cursor = self.conn.cursor()

        # Simple query by domain (can be enhanced with similarity search)
        domain = pattern.get('domain', '')

        cursor.execute("""
            SELECT * FROM historical_paradigm_shifts
            WHERE domain LIKE ?
            ORDER BY year DESC
            LIMIT ?
        """, (f"%{domain}%", limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _row_to_null_result(self, row: sqlite3.Row) -> NullResult:
        """Convert database row to NullResult"""
        return NullResult(
            attempt_id=row['attempt_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            problem_type=row['problem_type'],
            approach_type=row['approach_type'],
            error_type=ErrorType(row['error_type']),
            error_message=row['error_message'],
            state=json.loads(row['state_json']),
            iteration=row['iteration'],
            resources_used=json.loads(row['resources_json']),
            metadata=json.loads(row['metadata_json']),
            constraints=json.loads(row['constraints_json'])
        )

    def _row_to_assumption(self, row: sqlite3.Row) -> TacitAssumption:
        """Convert database row to TacitAssumption"""
        from tacit_assumption_miner import AssumptionType, PatternType

        return TacitAssumption(
            id=row['id'],
            description=row['description'],
            formalization=row['formalization'],
            assumption_type=AssumptionType(row['assumption_type']),
            confidence=row['confidence'],
            support=row['support'],
            evidence=json.loads(row['evidence_json']),
            pattern_type=PatternType(row['pattern_type']),
            constraint_relaxation=row['constraint_relaxation'],
            paradigm_implication=row['paradigm_implication'],
            alternative_paradigm=row['alternative_paradigm'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            verified=row['verified']
        )

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.cache = {
            'failures': {},
            'assumptions': {},
            'paradigms': {}
        }

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """
    High-level manager for failure database operations.

    Provides convenient methods for common database operations
    and handles connection pooling, caching, etc.
    """

    def __init__(self, db_path: str = "rese/data/phi15_failures.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to database file
        """
        self.db = FailureDatabase(db_path)

    def add_null_results(self, null_results: List[NullResult]) -> int:
        """Add multiple null results to database"""
        count = 0
        for nr in null_results:
            try:
                self.db.add_failure(nr)
                count += 1
            except sqlite3.Error as e:
                print(f"Error adding failure {nr.attempt_id}: {e}")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_failures': self.db.get_failure_count(),
            'total_assumptions': len(self.db.get_recent_assumptions(days=36500)),
            'recent_failures_24h': len(self.db.get_recent_failures(hours=24)),
            'recent_assumptions_30d': len(self.db.get_recent_assumptions(days=30)),
            'high_confidence_assumptions': len(
                self.db.get_high_confidence_assumptions(min_confidence=0.7)
            ),
            'paradigm_shifts_1y': len(self.db.get_recent_paradigm_shifts(days=365))
        }
        return stats

    def cleanup_old_data(self, days: int = 365) -> int:
        """
        Remove failures older than specified days (keep assumptions).

        Args:
            days: Number of days to keep data

        Returns:
            Number of failures removed
        """
        cursor = self.db.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            DELETE FROM failures
            WHERE timestamp < ?
        """, (cutoff.isoformat(),))

        deleted = cursor.rowcount
        self.db.conn.commit()

        return deleted

    def export_to_json(self, filepath: str,
                      include_failures: bool = True,
                      include_assumptions: bool = True,
                      include_paradigms: bool = True) -> None:
        """Export database to JSON file"""
        data = {}

        if include_failures:
            failures = self.db.get_failures_since(
                datetime.now() - timedelta(days=36500)
            )
            data['failures'] = [f.to_dict() if hasattr(f, 'to_dict') else asdict(f)
                               for f in failures]

        if include_assumptions:
            assumptions = self.db.get_recent_assumptions(days=36500)
            data['assumptions'] = [a.to_dict() for a in assumptions]

        if include_paradigms:
            paradigms = self.db.get_recent_paradigm_shifts(days=36500)
            data['paradigm_shifts'] = paradigms

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def close(self) -> None:
        """Close database connection"""
        self.db.close()


if __name__ == "__main__":
    # Quick test
    print("Φ₁.₅ Failure Database - Agent B1")
    print("=" * 50)

    # Create database
    with DatabaseManager() as db:
        # Get statistics
        stats = db.get_statistics()
        print(f"\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\nDatabase ready at: {db.db.db_path}")
