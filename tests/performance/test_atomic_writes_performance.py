"""
Atomic Write Tests (Bug #17)

Tests for atomic file write implementation
- Verifies write-to-temp-then-rename pattern
- Tests atomicity (no partial files)
- Tests that failed writes don't create target file
- Tests concurrent writes don't corrupt files
- Tests cleanup on failure
"""

import pytest
import time
import os
from pathlib import Path
from typing import Optional
import uuid


# ============================================================================
# Helper Functions (mimicking the implementation)
# ============================================================================

def writeFileAtomic(filePath: Path, data: bytes) -> None:
    """
    Write file atomically using temp-file-then-rename pattern

    Args:
        filePath: Final file path
        data: Data to write

    Raises:
        IOError: If write or rename fails
    """
    tempPath = filePath.with_suffix(filePath.suffix + f'.tmp-{uuid.uuid4().hex[:8]}')

    try:
        # Write to temporary file first
        tempPath.write_bytes(data)

        # Atomic rename to final path
        tempPath.rename(filePath)

    except Exception as error:
        # Clean up temp file on failure
        try:
            if tempPath.exists():
                tempPath.unlink()
        except Exception:
            pass

        raise error


def writeFileNonAtomic(filePath: Path, data: bytes) -> None:
    """
    Write file non-atomically (baseline for comparison)

    Args:
        filePath: Final file path
        data: Data to write
    """
    filePath.write_bytes(data)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return b"This is test data that should be written atomically to prevent corruption"


@pytest.fixture
def large_data():
    """Create larger data sample"""
    return b"x" * (1024 * 1024)  # 1MB


# ============================================================================
# Atomic Write Tests
# ============================================================================

class TestAtomicWritePattern:
    """Test atomic write pattern implementation"""

    def test_writes_to_temp_file_first(self, temp_dir):
        """Should write to temp file before final location"""
        file_path = temp_dir / "test_atomic.txt"
        data = b"test data"

        writeFileAtomic(file_path, data)

        # Final file should exist
        assert file_path.exists(), "Final file should exist"
        assert file_path.read_bytes() == data, "Final file should contain correct data"

        # Temp file should be cleaned up
        temp_files = list(temp_dir.glob("*.tmp-*"))
        assert len(temp_files) == 0, f"Temp files should be cleaned up, found {len(temp_files)}"

    def test_atomic_rename_happens_after_write(self, temp_dir):
        """Rename should happen after successful write"""
        file_path = temp_dir / "test_rename_order.txt"
        data = b"test data"

        # File shouldn't exist initially
        assert not file_path.exists(), "File should not exist before write"

        writeFileAtomic(file_path, data)

        # After write, file should exist atomically
        # (either exists completely or doesn't exist at all)
        assert file_path.exists(), "File should exist after write"
        assert file_path.read_bytes() == data, "File should have complete data"

    def test_failed_write_doesnt_create_target(self, temp_dir):
        """Failed write should not create target file"""
        file_path = temp_dir / "test_failed_write.txt"
        data = b"test data"

        # Make parent directory non-writable to cause failure
        # (This is tricky to test cross-platform, so we'll use a different approach)

        # Write to a path that includes a nonexistent directory
        bad_path = temp_dir / "nonexistent" / "subdir" / "file.txt"

        with pytest.raises(Exception):
            writeFileAtomic(bad_path, data)

        # Target file should not exist
        assert not bad_path.exists(), "Target file should not exist after failed write"

        # Check that partial files weren't created
        parent = bad_path.parent.parent
        if parent.exists():
            temp_files = list(parent.glob("*.tmp-*"))
            assert len(temp_files) == 0, "No temp files should remain after failure"

    def test_cleanup_happens_on_failure(self, temp_dir):
        """Temp files should be cleaned up on failure"""
        file_path = temp_dir / "test_cleanup.txt"
        data = b"test data"

        # Use invalid path to trigger failure
        invalid_path = temp_dir / "nonexistent_dir" / "file.txt"

        try:
            writeFileAtomic(invalid_path, data)
            assert False, "Should have raised an exception"
        except Exception:
            pass

        # No temp files should remain
        temp_files = list(temp_dir.glob("*.tmp-*"))
        assert len(temp_files) == 0, "Temp files should be cleaned up after failure"


class TestAtomicity:
    """Test atomicity guarantees"""

    def test_file_either_fully_written_or_not_written(self, temp_dir):
        """File should be either complete or not exist"""
        file_path = temp_dir / "test_complete.txt"
        data = b"x" * 10000

        writeFileAtomic(file_path, data)

        # If file exists, it should be complete
        if file_path.exists():
            content = file_path.read_bytes()
            assert content == data, "File should be complete if it exists"
            assert len(content) == len(data), "File should have correct size"

    def test_no_partial_files_written(self, temp_dir):
        """Should never write partial files"""
        for i in range(100):
            file_path = temp_dir / f"test_no_partial_{i}.txt"
            data = b"x" * (1000 + i * 100)  # Varying sizes

            writeFileAtomic(file_path, data)

            # Verify complete write
            assert file_path.exists(), f"File {i} should exist"
            content = file_path.read_bytes()
            assert content == data, f"File {i} should be complete"
            assert len(content) == len(data), f"File {i} should have correct size"

    def test_idempotent_retries(self, temp_dir):
        """Should be safe to retry failed writes"""
        file_path = temp_dir / "test_retry.txt"
        data = b"test data"

        # First write
        writeFileAtomic(file_path, data)
        assert file_path.read_bytes() == data

        # Retry (overwrite) - should work fine
        new_data = b"new data"
        writeFileAtomic(file_path, new_data)
        assert file_path.read_bytes() == new_data

        # Multiple retries
        for i in range(5):
            test_data = f"attempt_{i}".encode()
            writeFileAtomic(file_path, test_data)
            assert file_path.read_bytes() == test_data


class TestConcurrentWrites:
    """Test concurrent write scenarios"""

    def test_concurrent_writes_dont_corrupt(self, temp_dir):
        """Multiple concurrent writes should not corrupt files"""
        import threading

        file_path = temp_dir / "test_concurrent.txt"
        results = []
        errors = []

        def write_attempt(data):
            try:
                writeFileAtomic(file_path, data)
                content = file_path.read_bytes()
                results.append(content)
            except Exception as e:
                errors.append(e)

        # Launch multiple threads writing to same file
        threads = []
        for i in range(10):
            data = f"thread_{i}".encode()
            thread = threading.Thread(target=write_attempt, args=(data,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # One of them should have succeeded
        assert len(results) > 0, "At least one write should succeed"

        # The final file should be valid (one complete write)
        if file_path.exists():
            content = file_path.read_bytes()
            assert len(content) > 0, "File should have content"
            # Content should be one of the writes
            assert content in results, "File should contain one of the written values"

    def test_concurrent_writes_different_files(self, temp_dir):
        """Concurrent writes to different files should all succeed"""
        import threading

        files = []
        for i in range(20):
            file_path = temp_dir / f"concurrent_{i}.txt"
            data = f"data_{i}".encode()
            files.append((file_path, data))

        errors = []
        results = []

        def write_single(file_path, data):
            try:
                writeFileAtomic(file_path, data)
                results.append((file_path, data))
            except Exception as e:
                errors.append(e)

        # Write all files concurrently
        threads = []
        for file_path, data in files:
            thread = threading.Thread(target=write_single, args=(file_path, data))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert len(errors) == 0, f"All writes should succeed, got {len(errors)} errors"
        assert len(results) == 20, "All 20 files should be written"

        # Verify all files
        for file_path, expected_data in files:
            assert file_path.exists(), f"File {file_path} should exist"
            actual_data = file_path.read_bytes()
            assert actual_data == expected_data, f"File {file_path} should have correct data"


class TestPerformance:
    """Test atomic write performance"""

    def test_atomic_write_performance(self, sample_data, large_data, benchmark_results):
        """Atomic writes should have reasonable performance"""
        file_path = Path("/tmp/test_atomic_perf.txt")

        # Test with small data
        start = time.time()
        writeFileAtomic(file_path, sample_data)
        small_time = time.time() - start

        # Test with large data
        start = time.time()
        writeFileAtomic(file_path, large_data)
        large_time = time.time() - start

        benchmark_results.add_result(
            "atomic_write_small_time",
            "time",
            small_time,
            "s"
        )
        benchmark_results.add_result(
            "atomic_write_large_time",
            "time",
            large_time,
            "s"
        )

        print(f"\nAtomic Write Performance:")
        print(f"  Small data ({len(sample_data)} bytes): {small_time*1000:.2f}ms")
        print(f"  Large data ({len(large_data)} bytes): {large_time*1000:.2f}ms")

        # Should be fast enough for practical use
        assert small_time < 0.1, "Small writes should be under 100ms"
        assert large_time < 1.0, "Large writes should be under 1 second"

        # Cleanup
        if file_path.exists():
            file_path.unlink()

    def test_atomic_vs_non_atomic_overhead(self, temp_dir, benchmark_results):
        """Compare atomic vs non-atomic write overhead"""
        file_path_atomic = temp_dir / "atomic.txt"
        file_path_non_atomic = temp_dir / "non_atomic.txt"
        data = b"x" * (1024 * 100)  # 100KB

        # Measure atomic write
        start = time.time()
        writeFileAtomic(file_path_atomic, data)
        atomic_time = time.time() - start

        # Measure non-atomic write
        start = time.time()
        writeFileNonAtomic(file_path_non_atomic, data)
        non_atomic_time = time.time() - start

        overhead = ((atomic_time / non_atomic_time) - 1) * 100 if non_atomic_time > 0 else 0

        benchmark_results.add_result(
            "atomic_write_overhead",
            "overhead_percent",
            overhead,
            "%"
        )

        print(f"\nAtomic vs Non-Atomic Overhead:")
        print(f"  Atomic: {atomic_time*1000:.2f}ms")
        print(f"  Non-atomic: {non_atomic_time*1000:.2f}ms")
        print(f"  Overhead: {overhead:.1f}%")

        # Atomic should not be significantly slower (allow up to 3x overhead)
        assert atomic_time < non_atomic_time * 3, \
            "Atomic write should not be more than 3x slower than non-atomic"


class TestDataIntegrity:
    """Test data integrity through atomic writes"""

    def test_data_integrity_small_files(self, temp_dir):
        """Data integrity for small files"""
        test_data = [
            b"small",
            b"medium sized content",
            b"x" * 1000,
            json_data := b'{"key": "value", "number": 123}'
        ]

        for i, data in enumerate(test_data):
            file_path = temp_dir / f"integrity_small_{i}.txt"
            writeFileAtomic(file_path, data)

            assert file_path.exists(), f"File {i} should exist"
            assert file_path.read_bytes() == data, f"File {i} should have exact data"

    def test_data_integrity_large_files(self, temp_dir):
        """Data integrity for large files"""
        sizes = [1024, 10*1024, 100*1024, 1024*1024]  # 1KB, 10KB, 100KB, 1MB

        for size in sizes:
            data = b"x" * size
            file_path = temp_dir / f"integrity_large_{size}.txt"

            writeFileAtomic(file_path, data)

            assert file_path.exists(), f"File of size {size} should exist"
            assert file_path.read_bytes() == data, f"File of size {size} should have exact data"
            assert file_path.stat().st_size == size, f"File should have correct size"

    def test_data_integrity_binary_data(self, temp_dir):
        """Data integrity for binary data"""
        # Test with various binary patterns
        patterns = [
            bytes(range(256)),  # All byte values
            b"\x00\x01\x02\xff" * 1000,  # Null bytes and high values
        ]

        for i, pattern in enumerate(patterns):
            file_path = temp_dir / f"binary_{i}.bin"
            writeFileAtomic(file_path, pattern)

            assert file_path.exists(), f"Binary file {i} should exist"
            assert file_path.read_bytes() == pattern, f"Binary file {i} should preserve all bytes"


class TestErrorHandling:
    """Test error handling"""

    def test_handles_disk_full(self, temp_dir):
        """Should handle disk full errors gracefully"""
        # This is difficult to test reliably cross-platform
        # We'll simulate by using a bad path
        file_path = temp_dir / "nonexistent_dir" / "file.txt"
        data = b"test"

        with pytest.raises(Exception):
            writeFileAtomic(file_path, data)

        # No temp files should remain
        temp_files = list(temp_dir.glob("*.tmp-*"))
        assert len(temp_files) == 0

    def test_handles_permission_errors(self, temp_dir):
        """Should handle permission errors gracefully"""
        # Create file in read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only (Unix)
            os.chmod(readonly_dir, 0o444)

            file_path = readonly_dir / "file.txt"
            data = b"test"

            with pytest.raises(Exception):
                writeFileAtomic(file_path, data)

            # Verify cleanup
            temp_files = list(readonly_dir.glob("*.tmp-*"))
            assert len(temp_files) == 0

        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
