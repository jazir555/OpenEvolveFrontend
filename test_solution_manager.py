"""
Unit Tests for Solution Manager
================================

Comprehensive test suite for the SolutionManager module.

Tests cover:
- Solution creation and validation
- Status updates and transitions
- Solution history tracking
- Archival and cleanup
- Custom validators
- Thread safety
- Edge cases and error handling

Run with: pytest test_solution_manager.py -v
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from solution_manager import (
    SolutionManager,
    SolutionAttempt,
    SolutionStatus,
    SolutionHistory,
    ValidationResult,
    ValidationLevel,
    SolutionManagerError,
    SolutionValidationError,
    SolutionNotFoundError,
    SolutionStorageError,
    compute_content_hash,
    format_solution_summary
)


class TestSolutionAttempt(unittest.TestCase):
    """Test SolutionAttempt data model."""

    def test_solution_attempt_creation(self):
        """Test creating a solution attempt."""
        attempt = SolutionAttempt(
            id="sol_001",
            sub_problem_id="sp_001",
            content="def solution(): pass",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        self.assertEqual(attempt.id, "sol_001")
        self.assertEqual(attempt.sub_problem_id, "sp_001")
        self.assertEqual(attempt.status, SolutionStatus.PENDING.value)
        self.assertEqual(attempt.version, 1)
        self.assertIsNotNone(attempt.created_at)

    def test_solution_attempt_serialization(self):
        """Test converting to and from dictionary."""
        original = SolutionAttempt(
            id="sol_002",
            sub_problem_id="sp_002",
            content="test content",
            generated_by_model="claude-3",
            timestamp=time.time(),
            status=SolutionStatus.COMPLETED.value,
            metadata={"key": "value"}
        )

        # Convert to dict
        data = original.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["id"], "sol_002")

        # Convert back from dict
        restored = SolutionAttempt.from_dict(data)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.metadata, original.metadata)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult data model."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            is_valid=True,
            score=0.95,
            issues=[],
            warnings=["Minor warning"],
            feedback="Good solution",
            validator_name="TestValidator",
            timestamp=time.time()
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(len(result.warnings), 1)

    def test_validation_result_serialization(self):
        """Test ValidationResult serialization."""
        original = ValidationResult(
            is_valid=False,
            score=0.5,
            issues=["Critical error"],
            warnings=["Warning"],
            feedback="Has errors",
            validator_name="TestValidator",
            timestamp=time.time(),
            level=ValidationLevel.STRICT
        )

        # To dict
        data = original.to_dict()
        self.assertEqual(data["level"], ValidationLevel.STRICT.value)

        # From dict
        restored = ValidationResult.from_dict(data)
        self.assertEqual(restored.level, ValidationLevel.STRICT)
        self.assertFalse(restored.is_valid)


class TestSolutionManager(unittest.TestCase):
    """Test SolutionManager main functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.test_dir) / "solutions"
        self.archive_dir = Path(self.test_dir) / "archive"

        # Create manager
        self.manager = SolutionManager(
            storage_dir=str(self.storage_dir),
            archive_dir=str(self.archive_dir),
            enable_persistence=True,
            validation_level=ValidationLevel.MODERATE
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        self.assertTrue(self.storage_dir.exists())
        self.assertTrue(self.archive_dir.exists())
        self.assertEqual(len(self.manager._solutions), 0)
        self.assertEqual(len(self.manager._sub_problem_index), 0)

    def test_create_solution_attempt(self):
        """Test creating a solution attempt."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_001",
            content="def solution(): return 42",
            model="gpt-4"
        )

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.id)
        self.assertEqual(solution.sub_problem_id, "sp_001")
        self.assertEqual(solution.generated_by_model, "gpt-4")
        self.assertEqual(solution.status, SolutionStatus.PENDING.value)

        # Verify it's stored
        retrieved = self.manager.get_solution_attempt(solution.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, solution.id)

    def test_create_solution_with_validation_disabled(self):
        """Test creating solution without validation."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_002",
            content="short",  # Would normally fail validation
            model="test",
            validate=False
        )

        self.assertIsNotNone(solution)
        self.assertEqual(solution.content, "short")

    def test_create_solution_with_parent(self):
        """Test creating solution as revision of parent."""
        parent = self.manager.create_solution_attempt(
            sub_problem_id="sp_003",
            content="original content",
            model="gpt-4"
        )

        child = self.manager.create_solution_attempt(
            sub_problem_id="sp_003",
            content="revised content",
            model="gpt-4",
            parent_attempt_id=parent.id
        )

        self.assertEqual(child.parent_attempt_id, parent.id)
        self.assertEqual(child.version, 1)

    def test_create_solution_invalid_content_raises_error(self):
        """Test that creating solution with invalid content raises error."""
        with self.assertRaises(SolutionValidationError):
            self.manager.create_solution_attempt(
                sub_problem_id="sp_004",
                content="",  # Empty content
                model="test",
                validate=True
            )

    def test_update_solution_content(self):
        """Test updating solution content."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_005",
            content="original",
            model="gpt-4",
            validate=False
        )

        updated = self.manager.update_solution_attempt(
            attempt=solution,
            content="updated content",
            status=SolutionStatus.IN_PROGRESS.value
        )

        self.assertEqual(updated.content, "updated content")
        self.assertEqual(updated.status, SolutionStatus.IN_PROGRESS.value)
        self.assertEqual(updated.version, 2)

    def test_update_solution_metadata(self):
        """Test updating solution metadata."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_006",
            content="content",
            model="gpt-4",
            validate=False
        )

        updated = self.manager.update_solution_attempt(
            attempt=solution,
            metadata={"test_key": "test_value"}
        )

        self.assertIn("test_key", updated.metadata)
        self.assertEqual(updated.metadata["test_key"], "test_value")

    def test_update_solution_with_verification_report(self):
        """Test attaching verification report."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_007",
            content="content",
            model="gpt-4",
            validate=False
        )

        report = {
            "gauntlet": "test_gauntlet",
            "passed": True,
            "score": 0.9
        }

        updated = self.manager.update_solution_attempt(
            attempt=solution,
            verification_report=report
        )

        self.assertEqual(len(updated.verification_reports), 1)
        self.assertEqual(updated.verification_reports[0]["gauntlet"], "test_gauntlet")

    def test_update_nonexistent_solution_raises_error(self):
        """Test updating non-existent solution raises error."""
        fake_solution = SolutionAttempt(
            id="fake_id",
            sub_problem_id="fake_sp",
            content="content",
            generated_by_model="test",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        with self.assertRaises(SolutionNotFoundError):
            self.manager.update_solution_attempt(
                attempt=fake_solution,
                status=SolutionStatus.COMPLETED.value
            )

    def test_validate_solution_attempt(self):
        """Test solution validation."""
        solution = SolutionAttempt(
            id="sol_001",
            sub_problem_id="sp_001",
            content="def solve():\n    return True",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.score, 0.0)

    def test_validate_solution_with_short_content(self):
        """Test validation fails for short content."""
        solution = SolutionAttempt(
            id="sol_002",
            sub_problem_id="sp_002",
            content="x",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)
        self.assertIn("too short", result.issues[0].lower())

    def test_validate_solution_with_missing_fields(self):
        """Test validation fails for missing required fields."""
        solution = SolutionAttempt(
            id="",  # Missing ID
            sub_problem_id="",
            content="content",
            generated_by_model="",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)

    def test_get_solution_history(self):
        """Test retrieving solution history."""
        # Create multiple solutions
        s1 = self.manager.create_solution_attempt(
            sub_problem_id="sp_hist",
            content="solution 1",
            model="gpt-4",
            validate=False
        )
        time.sleep(0.01)  # Ensure different timestamps

        s2 = self.manager.create_solution_attempt(
            sub_problem_id="sp_hist",
            content="solution 2",
            model="gpt-4",
            validate=False
        )
        time.sleep(0.01)

        s3 = self.manager.create_solution_attempt(
            sub_problem_id="sp_hist",
            content="solution 3",
            model="gpt-4",
            validate=False
        )

        history = self.manager.get_solution_history("sp_hist")

        self.assertEqual(history.sub_problem_id, "sp_hist")
        self.assertEqual(history.total_attempts, 3)
        self.assertEqual(len(history.attempts), 3)
        self.assertEqual(history.latest_attempt.id, s3.id)
        # Should be sorted by timestamp
        self.assertEqual(history.attempts[0].id, s1.id)
        self.assertEqual(history.attempts[2].id, s3.id)

    def test_get_solution_history_with_limit(self):
        """Test retrieving solution history with limit."""
        # Create multiple solutions
        for i in range(10):
            self.manager.create_solution_attempt(
                sub_problem_id="sp_limit",
                content=f"solution {i}",
                model="gpt-4",
                validate=False
            )

        history = self.manager.get_solution_history("sp_limit", limit=5)

        self.assertEqual(history.total_attempts, 5)
        self.assertEqual(len(history.attempts), 5)

    def test_get_solution_history_empty(self):
        """Test retrieving history for non-existent sub-problem."""
        history = self.manager.get_solution_history("nonexistent")

        self.assertEqual(history.total_attempts, 0)
        self.assertEqual(len(history.attempts), 0)
        self.assertIsNone(history.latest_attempt)

    def test_get_latest_solution(self):
        """Test retrieving latest solution."""
        s1 = self.manager.create_solution_attempt(
            sub_problem_id="sp_latest",
            content="first",
            model="gpt-4",
            validate=False
        )
        time.sleep(0.01)

        s2 = self.manager.create_solution_attempt(
            sub_problem_id="sp_latest",
            content="second",
            model="gpt-4",
            validate=False
        )

        latest = self.manager.get_latest_solution("sp_latest")

        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, s2.id)

    def test_get_latest_solution_with_status_filter(self):
        """Test retrieving latest solution with status filter."""
        s1 = self.manager.create_solution_attempt(
            sub_problem_id="sp_status",
            content="first",
            model="gpt-4",
            validate=False
        )

        self.manager.update_solution_attempt(
            attempt=s1,
            status=SolutionStatus.COMPLETED.value
        )

        s2 = self.manager.create_solution_attempt(
            sub_problem_id="sp_status",
            content="second",
            model="gpt-4",
            validate=False
        )

        # Get latest with completed status
        latest = self.manager.get_latest_solution(
            "sp_status",
            status_filter=[SolutionStatus.COMPLETED.value]
        )

        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, s1.id)

    def test_get_solutions_by_status(self):
        """Test filtering solutions by status."""
        s1 = self.manager.create_solution_attempt(
            sub_problem_id="sp_filter",
            content="solution 1",
            model="gpt-4",
            validate=False
        )
        self.manager.update_solution_attempt(s1, status=SolutionStatus.COMPLETED.value)

        s2 = self.manager.create_solution_attempt(
            sub_problem_id="sp_filter",
            content="solution 2",
            model="gpt-4",
            validate=False
        )
        self.manager.update_solution_attempt(s2, status=SolutionStatus.FAILED.value)

        s3 = self.manager.create_solution_attempt(
            sub_problem_id="sp_other",
            content="solution 3",
            model="gpt-4",
            validate=False
        )
        self.manager.update_solution_attempt(s3, status=SolutionStatus.COMPLETED.value)

        # Get all completed
        completed = self.manager.get_solutions_by_status(SolutionStatus.COMPLETED.value)
        self.assertEqual(len(completed), 2)

        # Get completed for specific sub-problem
        sp_completed = self.manager.get_solutions_by_status(
            SolutionStatus.COMPLETED.value,
            sub_problem_id="sp_filter"
        )
        self.assertEqual(len(sp_completed), 1)
        self.assertEqual(sp_completed[0].id, s1.id)

    def test_archive_solution(self):
        """Test archiving a solution."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_archive",
            content="to archive",
            model="gpt-4",
            validate=False
        )

        self.manager.archive_solution(solution)

        # Check status updated
        self.assertEqual(solution.status, SolutionStatus.ARCHIVED.value)

        # Check file moved
        storage_path = self.manager._get_storage_path(solution.id)
        archive_path = self.manager._get_archive_path(solution.id)

        self.assertFalse(storage_path.exists())
        self.assertTrue(archive_path.exists())

    def test_archive_old_solutions(self):
        """Test archiving old solutions while keeping latest."""
        sub_problem_id = "sp_bulk_archive"

        # Create 10 solutions
        for i in range(10):
            sol = self.manager.create_solution_attempt(
                sub_problem_id=sub_problem_id,
                content=f"solution {i}",
                model="gpt-4",
                validate=False
            )
            time.sleep(0.001)  # Ensure different timestamps

        # Archive all but latest 5
        archived_count = self.manager.archive_old_solutions(
            sub_problem_id=sub_problem_id,
            keep_latest=5
        )

        self.assertEqual(archived_count, 5)

        # Check history
        history = self.manager.get_solution_history(sub_problem_id)
        active_count = sum(
            1 for s in history.attempts
            if s.status != SolutionStatus.ARCHIVED.value
        )
        self.assertEqual(active_count, 5)

    def test_persist_and_load_solution(self):
        """Test persisting and loading solutions."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_persist",
            content="persist me",
            model="gpt-4",
            validate=False
        )

        # Verify file exists
        file_path = self.manager._get_storage_path(solution.id)
        self.assertTrue(file_path.exists())

        # Load from disk
        loaded = self.manager.load_solution_from_disk(solution.id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, solution.id)
        self.assertEqual(loaded.content, solution.content)

    def test_load_all_solutions(self):
        """Test loading all solutions from disk."""
        # Create multiple solutions
        for i in range(5):
            self.manager.create_solution_attempt(
                sub_problem_id=f"sp_load_{i}",
                content=f"content {i}",
                model="gpt-4",
                validate=False
            )

        # Clear in-memory storage
        self.manager.clear_all()
        self.assertEqual(len(self.manager._solutions), 0)

        # Load all from disk
        loaded_count = self.manager.load_all_solutions()

        self.assertEqual(loaded_count, 5)
        self.assertEqual(len(self.manager._solutions), 5)

    def test_get_statistics(self):
        """Test getting manager statistics."""
        # Create solutions with different statuses
        s1 = self.manager.create_solution_attempt(
            sub_problem_id="sp_stats1",
            content="solution 1",
            model="gpt-4",
            validate=False
        )
        self.manager.update_solution_attempt(s1, status=SolutionStatus.COMPLETED.value)

        s2 = self.manager.create_solution_attempt(
            sub_problem_id="sp_stats2",
            content="solution 2",
            model="gpt-4",
            validate=False
        )
        self.manager.update_solution_attempt(
            s2,
            status=SolutionStatus.COMPLETED.value,
            quality_score=0.8
        )

        stats = self.manager.get_statistics()

        self.assertEqual(stats["total_solutions"], 2)
        self.assertEqual(stats["total_sub_problems"], 2)
        self.assertIn(SolutionStatus.COMPLETED.value, stats["status_distribution"])
        self.assertEqual(stats["status_distribution"][SolutionStatus.COMPLETED.value], 2)
        self.assertGreater(stats["average_quality_score"], 0.0)

    def test_clear_all(self):
        """Test clearing all solutions."""
        # Create solutions
        self.manager.create_solution_attempt(
            sub_problem_id="sp_clear",
            content="solution",
            model="gpt-4",
            validate=False
        )

        self.assertGreater(len(self.manager._solutions), 0)

        # Clear
        self.manager.clear_all()

        self.assertEqual(len(self.manager._solutions), 0)
        self.assertEqual(len(self.manager._sub_problem_index), 0)


class TestCustomValidators(unittest.TestCase):
    """Test custom validator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = SolutionManager(
            storage_dir=f"{self.test_dir}/solutions",
            enable_persistence=False
        )

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_register_custom_validator(self):
        """Test registering a custom validator."""
        def custom_validator(attempt: SolutionAttempt) -> ValidationResult:
            has_keyword = "important" in attempt.content.lower()
            return ValidationResult(
                is_valid=has_keyword,
                score=1.0 if has_keyword else 0.0,
                issues=[] if has_keyword else ["Missing 'important' keyword"],
                warnings=[],
                feedback="Custom validation",
                validator_name="KeywordValidator",
                timestamp=time.time()
            )

        self.manager.register_validator("keyword", custom_validator)

        self.assertIn("keyword", self.manager._validators)

    def test_custom_validator_executes(self):
        """Test that custom validator executes during validation."""
        def python_validator(attempt: SolutionAttempt) -> ValidationResult:
            has_python = "def " in attempt.content
            return ValidationResult(
                is_valid=has_python,
                score=1.0 if has_python else 0.5,
                issues=[] if has_python else ["No Python function"],
                warnings=[],
                feedback="Python check",
                validator_name="PythonValidator",
                timestamp=time.time()
            )

        self.manager.register_validator("python", python_validator)

        # Create solution with Python
        solution = SolutionAttempt(
            id="sol_python",
            sub_problem_id="sp_py",
            content="def solve(): return True",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)

        self.assertTrue(result.is_valid)

        # Create solution without Python
        solution_no_py = SolutionAttempt(
            id="sol_no_py",
            sub_problem_id="sp_nopy",
            content="This is just text",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution_no_py)

        # Should have issue from custom validator
        self.assertTrue(any("Python" in issue for issue in result.issues))

    def test_unregister_validator(self):
        """Test unregistering a validator."""
        def dummy_validator(attempt: SolutionAttempt) -> ValidationResult:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                issues=[],
                warnings=[],
                feedback="dummy",
                validator_name="dummy",
                timestamp=time.time()
            )

        self.manager.register_validator("dummy", dummy_validator)
        self.assertIn("dummy", self.manager._validators)

        self.manager.unregister_validator("dummy")
        self.assertNotIn("dummy", self.manager._validators)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        content1 = "test content"
        content2 = "test content"
        content3 = "different content"

        hash1 = compute_content_hash(content1)
        hash2 = compute_content_hash(content2)
        hash3 = compute_content_hash(content3)

        # Same content should produce same hash
        self.assertEqual(hash1, hash2)
        # Different content should produce different hash
        self.assertNotEqual(hash1, hash3)
        # Hash should be hex string
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1))

    def test_format_solution_summary(self):
        """Test solution summary formatting."""
        attempt = SolutionAttempt(
            id="sol_summary",
            sub_problem_id="sp_summary",
            content="def solve(): return 'summary'",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.COMPLETED.value,
            quality_score=0.95
        )

        summary = format_solution_summary(attempt)

        self.assertIn("sol_summary", summary)
        self.assertIn("sp_summary", summary)
        self.assertIn("0.95", summary)
        self.assertIn("COMPLETED", summary)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of SolutionManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = SolutionManager(
            storage_dir=f"{self.test_dir}/solutions",
            enable_persistence=False
        )

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_concurrent_solution_creation(self):
        """Test creating solutions from multiple threads."""
        num_threads = 10
        solutions_per_thread = 5

        def create_solutions(thread_id: int):
            for i in range(solutions_per_thread):
                self.manager.create_solution_attempt(
                    sub_problem_id=f"sp_thread_{thread_id}",
                    content=f"Solution {i} from thread {thread_id}",
                    model="gpt-4",
                    validate=False
                )

        # Create threads
        threads = [
            threading.Thread(target=create_solutions, args=(i,))
            for i in range(num_threads)
        ]

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all solutions created
        stats = self.manager.get_statistics()
        self.assertEqual(stats["total_solutions"], num_threads * solutions_per_thread)

    def test_concurrent_updates(self):
        """Test updating solutions from multiple threads."""
        # Create a solution
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_concurrent",
            content="original",
            model="gpt-4",
            validate=False
        )

        num_updates = 10

        def update_solution(thread_id: int):
            for i in range(num_updates):
                try:
                    self.manager.update_solution_attempt(
                        attempt=solution,
                        metadata={"thread": thread_id, "update": i}
                    )
                except SolutionNotFoundError:
                    # Solution might be updated by another thread
                    pass

        # Create threads
        threads = [
            threading.Thread(target=update_solution, args=(i,))
            for i in range(5)
        ]

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify solution still exists
        retrieved = self.manager.get_solution_attempt(solution.id)
        self.assertIsNotNone(retrieved)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = SolutionManager(
            storage_dir=f"{self.test_dir}/solutions",
            enable_persistence=False
        )

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_empty_content_validation(self):
        """Test validating empty content."""
        solution = SolutionAttempt(
            id="sol_empty",
            sub_problem_id="sp_empty",
            content="",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("short" in i.lower() for i in result.issues))

    def test_very_long_content(self):
        """Test handling very long content."""
        # Create content exceeding max length
        long_content = "x" * (10_000_001)

        solution = SolutionAttempt(
            id="sol_long",
            sub_problem_id="sp_long",
            content=long_content,
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("long" in i.lower() for i in result.issues))

    def test_invalid_status_value(self):
        """Test handling invalid status value."""
        solution = SolutionAttempt(
            id="sol_invalid_status",
            sub_problem_id="sp_invalid",
            content="content",
            generated_by_model="gpt-4",
            timestamp=time.time(),
            status="invalid_status"
        )

        result = self.manager.validate_solution_attempt(solution)
        # Should have warning about unknown status
        self.assertTrue(any("unknown status" in w.lower() for w in result.warnings))

    def test_negative_timestamp(self):
        """Test handling negative timestamp."""
        solution = SolutionAttempt(
            id="sol_neg_time",
            sub_problem_id="sp_neg",
            content="content",
            generated_by_model="gpt-4",
            timestamp=-1.0,
            status=SolutionStatus.PENDING.value
        )

        result = self.manager.validate_solution_attempt(solution)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("timestamp" in i.lower() for i in result.issues))

    def test_update_with_invalid_status(self):
        """Test updating with invalid status raises error."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_invalid",
            content="content",
            model="gpt-4",
            validate=False
        )

        with self.assertRaises(ValueError):
            self.manager.update_solution_attempt(
                attempt=solution,
                status="not_a_real_status"
            )

    def test_quality_score_clamping(self):
        """Test that quality scores are clamped to [0.0, 1.0]."""
        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_clamp",
            content="content",
            model="gpt-4",
            validate=False
        )

        # Test high value
        updated = self.manager.update_solution_attempt(
            attempt=solution,
            quality_score=1.5
        )
        self.assertEqual(updated.quality_score, 1.0)

        # Test low value
        updated = self.manager.update_solution_attempt(
            attempt=updated,
            quality_score=-0.5
        )
        self.assertEqual(updated.quality_score, 0.0)

    def test_special_characters_in_content(self):
        """Test handling special characters in content."""
        special_content = '''
        def solve():
            # Unicode: café, naïve, 日本語
            return "result"
        '''

        solution = self.manager.create_solution_attempt(
            sub_problem_id="sp_special",
            content=special_content,
            model="gpt-4",
            validate=False
        )

        self.assertEqual(solution.content, special_content)

        # Should persist and load correctly
        retrieved = self.manager.get_solution_attempt(solution.id)
        self.assertEqual(retrieved.content, special_content)


if __name__ == "__main__":
    unittest.main()
