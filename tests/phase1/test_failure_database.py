"""
Unit Tests for Φ₁.₅ Failure Database

Tests database operations for failures, assumptions, and paradigm shifts.

Author: Agent B1 (Φ₁/Φ₁.₅ Specialist)
Created: 2025-12-31
Status: 🟢 Active
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase1"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from failure_database import (
    FailureDatabase,
    DatabaseManager,
    NullResult,
    ErrorType,
    TacitAssumption,
    AssumptionType,
    PatternType,
    ParadigmShiftRecommendation,
    FailureFeatures
)


@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.remove(path)
    except:
        pass


@pytest.fixture
def sample_null_result():
    """Create sample null result"""
    return NullResult(
        attempt_id="test_001",
        timestamp=datetime.now(),
        problem_type="optimization",
        approach_type="gradient_descent",
        constraints=["x > 0", "y < 10"],
        error_type=ErrorType.OPTIMIZATION_FAILED,
        error_message="Failed to converge",
        state={"iteration": 100},
        iteration=100,
        resources_used={"cpu": 50.0, "memory": 100.0},
        metadata={"test": "data"}
    )


@pytest.fixture
def sample_assumption():
    """Create sample assumption"""
    return TacitAssumption(
        id="assump_001",
        description="Test assumption",
        formalization="test_formalization",
        assumption_type=AssumptionType.CONSTRAINT,
        confidence=0.8,
        support=10,
        evidence=["fail1", "fail2"],
        pattern_type=PatternType.SYSTEMATIC_VIOLATION,
        constraint_relaxation="Relax test",
        paradigm_implication=False,
        alternative_paradigm=None
    )


@pytest.fixture
def sample_failure_features():
    """Create sample failure features"""
    import numpy as np
    return FailureFeatures(
        attempt_id="feat_001",
        timestamp=datetime.now(),
        problem_type="test",
        approach_type="test",
        error_type=ErrorType.NUMERICAL_INSTABILITY,
        iteration=50,
        time_to_failure=100.0,
        error_magnitude=0.5,
        resource_consumption=0.7,
        constraint_violation_count=2,
        feature_vector=np.array([1.0, 2.0, 3.0]),
        keywords=["test", "failure"]
    )


class TestFailureDatabase:
    """Test FailureDatabase operations"""

    def test_database_initialization(self, temp_db_path):
        """Test database initialization"""
        db = FailureDatabase(db_path=temp_db_path)

        assert db.db_path == temp_db_path
        assert db.conn is not None
        assert isinstance(db.cache, dict)

    def test_database_creates_tables(self, temp_db_path):
        """Test database creates required tables"""
        db = FailureDatabase(db_path=temp_db_path)

        cursor = db.conn.cursor()

        # Check failures table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failures'")
        assert cursor.fetchone() is not None

        # Check failure_features table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failure_features'")
        assert cursor.fetchone() is not None

        # Check assumptions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assumptions'")
        assert cursor.fetchone() is not None

        # Check paradigm_shifts table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paradigm_shifts'")
        assert cursor.fetchone() is not None

        # Check historical_paradigm_shifts table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_paradigm_shifts'")
        assert cursor.fetchone() is not None

    def test_add_failure(self, temp_db_path, sample_null_result):
        """Test adding a failure to database"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)

        # Verify it was added
        retrieved = db.get_failure(sample_null_result.attempt_id)

        assert retrieved is not None
        assert retrieved.attempt_id == sample_null_result.attempt_id
        assert retrieved.problem_type == sample_null_result.problem_type

    def test_add_failure_with_features(self, temp_db_path, sample_null_result, sample_failure_features):
        """Test adding failure with features"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result, features=sample_failure_features)

        # Verify failure was added
        retrieved = db.get_failure(sample_null_result.attempt_id)
        assert retrieved is not None

    def test_add_failure_features(self, temp_db_path, sample_failure_features):
        """Test adding failure features"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure_features(sample_failure_features)

        # Verify through direct query
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM failure_features WHERE attempt_id=?", (sample_failure_features.attempt_id,))
        row = cursor.fetchone()
        assert row is not None

    def test_get_failure(self, temp_db_path, sample_null_result):
        """Test retrieving a failure"""
        db = FailureDatabase(db_path=temp_db_path)

        # Get non-existent failure
        result = db.get_failure("nonexistent")
        assert result is None

        # Add and get existing failure
        db.add_failure(sample_null_result)
        result = db.get_failure(sample_null_result.attempt_id)

        assert result is not None
        assert result.attempt_id == sample_null_result.attempt_id

    def test_get_failures_since(self, temp_db_path):
        """Test getting failures since timestamp"""
        db = FailureDatabase(db_path=temp_db_path)

        # Add failures at different times
        now = datetime.now()
        old_time = now - timedelta(hours=48)

        old_failure = NullResult(
            attempt_id="old_001",
            timestamp=old_time,
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )

        new_failure = NullResult(
            attempt_id="new_001",
            timestamp=now,
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )

        db.add_failure(old_failure)
        db.add_failure(new_failure)

        # Get failures since yesterday
        cutoff = now - timedelta(hours=24)
        recent_failures = db.get_failures_since(cutoff)

        # Should only get the new one
        assert len(recent_failures) == 1
        assert recent_failures[0].attempt_id == "new_001"

    def test_get_recent_failures(self, temp_db_path, sample_null_result):
        """Test getting recent failures"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)

        recent = db.get_recent_failures(hours=24)

        assert len(recent) >= 1
        assert any(f.attempt_id == sample_null_result.attempt_id for f in recent)

    def test_get_unprocessed_failures(self, temp_db_path, sample_null_result):
        """Test getting unprocessed failures"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)

        unprocessed = db.get_unprocessed_failures()

        assert len(unprocessed) >= 1
        assert any(f.attempt_id == sample_null_result.attempt_id for f in unprocessed)

    def test_mark_as_processed(self, temp_db_path, sample_null_result):
        """Test marking failure as processed"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)

        # Should be in unprocessed
        unprocessed = db.get_unprocessed_failures()
        assert any(f.attempt_id == sample_null_result.attempt_id for f in unprocessed)

        # Mark as processed
        db.mark_as_processed(sample_null_result.attempt_id)

        # Should not be in unprocessed anymore
        unprocessed_after = db.get_unprocessed_failures()
        assert not any(f.attempt_id == sample_null_result.attempt_id for f in unprocessed_after)

    def test_get_failure_count(self, temp_db_path):
        """Test getting failure count"""
        db = FailureDatabase(db_path=temp_db_path)

        count_before = db.get_failure_count()

        # Add some failures
        for i in range(5):
            failure = NullResult(
                attempt_id=f"count_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                constraints=[],
                error_type=ErrorType.TIMEOUT,
                error_message="Test",
                state={},
                iteration=1,
                resources_used={}
            )
            db.add_failure(failure)

        count_after = db.get_failure_count()

        assert count_after == count_before + 5

    def test_add_assumption(self, temp_db_path, sample_assumption):
        """Test adding an assumption"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_assumption(sample_assumption)

        # Retrieve it
        retrieved = db.get_assumption(sample_assumption.id)

        assert retrieved is not None
        assert retrieved.id == sample_assumption.id
        assert retrieved.description == sample_assumption.description
        assert retrieved.confidence == sample_assumption.confidence

    def test_get_assumption(self, temp_db_path, sample_assumption):
        """Test retrieving an assumption"""
        db = FailureDatabase(db_path=temp_db_path)

        # Get non-existent
        result = db.get_assumption("nonexistent")
        assert result is None

        # Add and get existing
        db.add_assumption(sample_assumption)
        result = db.get_assumption(sample_assumption.id)

        assert result is not None
        assert result.id == sample_assumption.id

    def test_get_recent_assumptions(self, temp_db_path, sample_assumption):
        """Test getting recent assumptions"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_assumption(sample_assumption)

        recent = db.get_recent_assumptions(days=30)

        assert len(recent) >= 1
        assert any(a.id == sample_assumption.id for a in recent)

    def test_get_high_confidence_assumptions(self, temp_db_path):
        """Test getting high confidence assumptions"""
        db = FailureDatabase(db_path=temp_db_path)

        # Add assumptions with different confidences
        for i in range(5):
            assumption = TacitAssumption(
                id=f"conf_{i}",
                description=f"Assumption {i}",
                formalization=f"form_{i}",
                assumption_type=AssumptionType.METHODOLOGICAL,
                confidence=0.5 + i * 0.1,  # 0.5 to 0.9
                support=5,
                evidence=[],
                pattern_type=PatternType.REPEATED_FAILURE,
                constraint_relaxation="Relax",
                paradigm_implication=False,
                alternative_paradigm=None
            )
            db.add_assumption(assumption)

        # Get high confidence (>= 0.7)
        high_conf = db.get_high_confidence_assumptions(min_confidence=0.7)

        assert len(high_conf) == 3  # conf_2 (0.7), conf_3 (0.8), conf_4 (0.9)
        assert all(a.confidence >= 0.7 for a in high_conf)

    def test_update_assumption_confidence(self, temp_db_path, sample_assumption):
        """Test updating assumption confidence"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_assumption(sample_assumption)

        # Update confidence
        db.update_assumption_confidence(sample_assumption.id, 0.95)

        # Retrieve and verify
        retrieved = db.get_assumption(sample_assumption.id)

        assert retrieved.confidence == 0.95
        assert bool(retrieved.verified) is True

    def test_add_paradigm_shift(self, temp_db_path):
        """Test adding paradigm shift recommendation"""
        db = FailureDatabase(db_path=temp_db_path)

        rec = ParadigmShiftRecommendation(
            trigger=True,
            confidence=0.8,
            primary_assumptions=[],
            suggested_alternatives=["Alternative 1"],
            explanation="Test paradigm shift"
        )

        db.add_paradigm_shift(rec)

        # Get recent shifts
        recent = db.get_recent_paradigm_shifts(days=365)

        assert len(recent) >= 1

    def test_get_recent_paradigm_shifts(self, temp_db_path):
        """Test getting recent paradigm shifts"""
        db = FailureDatabase(db_path=temp_db_path)

        rec = ParadigmShiftRecommendation(
            trigger=False,
            confidence=0.3,
            primary_assumptions=[],
            suggested_alternatives=[],
            explanation="No crisis"
        )

        db.add_paradigm_shift(rec)

        recent = db.get_recent_paradigm_shifts(days=365)

        assert len(recent) >= 1

    def test_load_historical_paradigm_shifts(self, temp_db_path):
        """Test loading historical paradigm shifts from file"""
        db = FailureDatabase(db_path=temp_db_path)

        # Create temp file with historical data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_json = f.name
            data = {
                "paradigm_shifts": [
                    {
                        "name": "Copernican Revolution",
                        "year": 1543,
                        "domain": "Astronomy",
                        "old_paradigm": "Geocentric",
                        "new_paradigm": "Heliocentric",
                        "tacit_assumption": "Earth is stationary",
                        "description": "Earth moves around the Sun",
                        "failure_pattern": {}
                    }
                ]
            }
            json.dump(data, f)

        try:
            db.load_historical_paradigm_shifts(temp_json)

            # Verify it was loaded
            cursor = db.conn.cursor()
            cursor.execute("SELECT * FROM historical_paradigm_shifts WHERE name=?", ("Copernican Revolution",))
            row = cursor.fetchone()
            assert row is not None

        finally:
            os.remove(temp_json)

    def test_find_similar_historical_shifts(self, temp_db_path):
        """Test finding similar historical paradigm shifts"""
        db = FailureDatabase(db_path=temp_db_path)

        # Load historical data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_json = f.name
            data = {
                "paradigm_shifts": [
                    {
                        "name": "Quantum Revolution",
                        "year": 1900,
                        "domain": "Physics",
                        "old_paradigm": "Classical Mechanics",
                        "new_paradigm": "Quantum Mechanics",
                        "tacit_assumption": "Determinism",
                        "description": "Wave-particle duality",
                        "failure_pattern": {}
                    }
                ]
            }
            json.dump(data, f)

        try:
            db.load_historical_paradigm_shifts(temp_json)

            # Find similar
            similar = db.find_similar_historical_shifts({"domain": "Physics"})

            assert len(similar) >= 1
            assert similar[0]['name'] == "Quantum Revolution"

        finally:
            os.remove(temp_json)

    def test_cache_functionality(self, temp_db_path, sample_null_result):
        """Test database caching"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)

        # First access - loads from DB
        result1 = db.get_failure(sample_null_result.attempt_id)

        # Second access - loads from cache
        result2 = db.get_failure(sample_null_result.attempt_id)

        assert result1.attempt_id == result2.attempt_id
        assert sample_null_result.attempt_id in db.cache['failures']

    def test_clear_cache(self, temp_db_path, sample_null_result):
        """Test clearing cache"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)
        db.get_failure(sample_null_result.attempt_id)

        # Cache should have entry
        assert sample_null_result.attempt_id in db.cache['failures']

        # Clear cache
        db.clear_cache()

        # Cache should be empty
        assert len(db.cache['failures']) == 0

    def test_context_manager(self, temp_db_path):
        """Test database as context manager"""
        with FailureDatabase(db_path=temp_db_path) as db:
            assert db.conn is not None

            # Add some data
            failure = NullResult(
                attempt_id="ctx_test",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                constraints=[],
                error_type=ErrorType.TIMEOUT,
                error_message="Test",
                state={},
                iteration=1,
                resources_used={}
            )
            db.add_failure(failure)

        # Connection should be closed after context
        # We can't directly test this without accessing private members,
        # but we can verify no exception was raised


class TestDatabaseManager:
    """Test DatabaseManager high-level operations"""

    @pytest.fixture
    def manager(self, temp_db_path):
        return DatabaseManager(db_path=temp_db_path)

    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.db is not None
        assert isinstance(manager.db, FailureDatabase)

    def test_add_null_results(self, manager):
        """Test adding multiple null results"""
        results = []
        for i in range(10):
            nr = NullResult(
                attempt_id=f"manager_{i}",
                timestamp=datetime.now(),
                problem_type="test",
                approach_type="test",
                constraints=[],
                error_type=ErrorType.TIMEOUT,
                error_message="Test",
                state={},
                iteration=1,
                resources_used={}
            )
            results.append(nr)

        count = manager.add_null_results(results)

        assert count == 10

    def test_get_statistics(self, manager, temp_db_path):
        """Test getting database statistics"""
        # Add some data
        failure = NullResult(
            attempt_id="stats_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )
        manager.add_null_results([failure])

        stats = manager.get_statistics()

        assert "total_failures" in stats
        assert "total_assumptions" in stats
        assert "recent_failures_24h" in stats
        assert stats["total_failures"] >= 1

    def test_cleanup_old_data(self, manager):
        """Test cleaning up old data"""
        # Add old failure
        old_failure = NullResult(
            attempt_id="old_data",
            timestamp=datetime.now() - timedelta(days=400),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )
        manager.add_null_results([old_failure])

        # Add recent failure
        recent_failure = NullResult(
            attempt_id="recent_data",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )
        manager.add_null_results([recent_failure])

        # Cleanup data older than 300 days
        deleted = manager.cleanup_old_data(days=300)

        assert deleted == 1

        # Verify only recent remains
        stats = manager.get_statistics()
        assert stats["total_failures"] == 1

    def test_export_to_json(self, manager, temp_db_path):
        """Test exporting database to JSON"""
        # Add some data
        failure = NullResult(
            attempt_id="export_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={}
        )
        manager.add_null_results([failure])

        # Export
        export_path = temp_db_path.replace('.db', '_export.json')
        try:
            manager.export_to_json(export_path)

            # Verify file exists and has content
            assert os.path.exists(export_path)

            with open(export_path, 'r') as f:
                data = json.load(f)

            assert "failures" in data
            assert len(data["failures"]) >= 1

        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_manager_close(self, manager):
        """Test manager close"""
        # This should not raise an exception
        manager.close()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_database(self, temp_db_path):
        """Test operations on empty database"""
        db = FailureDatabase(db_path=temp_db_path)

        assert db.get_failure_count() == 0
        assert db.get_recent_failures() == []
        assert db.get_recent_assumptions() == []
        assert db.get_recent_paradigm_shifts() == []

    def test_duplicate_failure(self, temp_db_path, sample_null_result):
        """Test adding duplicate failure (should replace)"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_failure(sample_null_result)
        db.add_failure(sample_null_result)  # Add again

        # Should still have only one
        count = db.get_failure_count()
        assert count == 1

    def test_duplicate_assumption(self, temp_db_path, sample_assumption):
        """Test adding duplicate assumption (should replace)"""
        db = FailureDatabase(db_path=temp_db_path)

        db.add_assumption(sample_assumption)

        # Modify and add again
        sample_assumption.confidence = 0.95
        db.add_assumption(sample_assumption)

        # Should have updated value
        retrieved = db.get_assumption(sample_assumption.id)
        assert retrieved.confidence == 0.95

    def test_very_long_strings(self, temp_db_path):
        """Test handling very long strings"""
        db = FailureDatabase(db_path=temp_db_path)

        long_description = "A" * 10000

        failure = NullResult(
            attempt_id="long_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message=long_description,
            state={},
            iteration=1,
            resources_used={}
        )

        # Should not raise exception
        db.add_failure(failure)

        retrieved = db.get_failure("long_test")
        assert retrieved is not None

    def test_special_characters(self, temp_db_path):
        """Test handling special characters in strings"""
        db = FailureDatabase(db_path=temp_db_path)

        special_msg = "Error: 'test' with \"quotes\" and \n newlines \t tabs"

        failure = NullResult(
            attempt_id="special_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message=special_msg,
            state={},
            iteration=1,
            resources_used={}
        )

        db.add_failure(failure)

        retrieved = db.get_failure("special_test")
        assert retrieved.error_message == special_msg

    def test_unicode_characters(self, temp_db_path):
        """Test handling unicode characters"""
        db = FailureDatabase(db_path=temp_db_path)

        unicode_msg = "Error: 测试 error with emoji 🚀"

        failure = NullResult(
            attempt_id="unicode_test",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message=unicode_msg,
            state={},
            iteration=1,
            resources_used={}
        )

        db.add_failure(failure)

        retrieved = db.get_failure("unicode_test")
        assert retrieved.error_message == unicode_msg

    def test_none_metadata(self, temp_db_path):
        """Test handling None metadata"""
        db = FailureDatabase(db_path=temp_db_path)

        failure = NullResult(
            attempt_id="none_meta",
            timestamp=datetime.now(),
            problem_type="test",
            approach_type="test",
            constraints=[],
            error_type=ErrorType.TIMEOUT,
            error_message="Test",
            state={},
            iteration=1,
            resources_used={},
            metadata=None
        )

        # Should handle None metadata
        db.add_failure(failure)

        retrieved = db.get_failure("none_meta")
        assert retrieved is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
