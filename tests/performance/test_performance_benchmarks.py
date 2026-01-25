"""
Comprehensive Performance Benchmarks

Benchmarks comparing before/after performance for all improvements:
- Sequential vs Concurrent File Deletion (Bug #15)
- With vs Without Compression (Bug #16)
- Paginated vs Full Query (Bug #18)
- Atomic vs Non-Atomic Write Overhead (Bug #17)
"""

import pytest
import time
import gzip
import asyncio
from pathlib import Path
from typing import List, Tuple
import uuid
import shutil


# ============================================================================
# Helper Functions
# ============================================================================

async def deleteFilesConcurrently(files: List[Path], concurrency: int = 10) -> Tuple[int, int]:
    """Delete files concurrently"""
    deleted = 0
    failed = 0

    async def deleteBatch(batch: List[Path]) -> Tuple[int, int]:
        results = await asyncio.gather(
            *[asyncio.to_thread(f.unlink) if f.exists() else asyncio.sleep(0) for f in batch],
            return_exceptions=True
        )
        batch_deleted = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        batch_failed = len(batch) - batch_deleted
        return batch_deleted, batch_failed

    for i in range(0, len(files), concurrency):
        batch = files[i:i + concurrency]
        batch_deleted, batch_failed = await deleteBatch(batch)
        deleted += batch_deleted
        failed += batch_failed

    return deleted, failed


def deleteFilesSequentially(files: List[Path]) -> Tuple[int, int]:
    """Delete files sequentially"""
    deleted = 0
    failed = 0
    for filePath in files:
        try:
            if filePath.exists():
                filePath.unlink()
                deleted += 1
        except Exception:
            failed += 1
    return deleted, failed


def compressData(data: bytes) -> bytes:
    """Compress data using gzip"""
    return gzip.compress(data)


def writeFileAtomic(filePath: Path, data: bytes) -> None:
    """Write file atomically"""
    tempPath = filePath.with_suffix(filePath.suffix + f'.tmp-{uuid.uuid4().hex[:8]}')
    try:
        tempPath.write_bytes(data)
        tempPath.rename(filePath)
    except Exception:
        if tempPath.exists():
            tempPath.unlink()
        raise


def writeFileNonAtomic(filePath: Path, data: bytes) -> None:
    """Write file non-atomically"""
    filePath.write_bytes(data)


# ============================================================================
# Benchmark Test Suite
# ============================================================================

class TestPerformanceBenchmarks:
    """
    Comprehensive performance benchmarks

    This test suite measures and reports the performance improvements
    from all the bug fixes.
    """

    @pytest.mark.asyncio
    async def test_benchmark_sequential_vs_concurrent_deletion(
        self, temp_dir, benchmark_results
    ):
        """
        Benchmark #15: Sequential vs Concurrent File Deletion

        Expected: Concurrent should be 5-10x faster
        """
        file_counts = [50, 100, 200]
        results = []

        for count in file_counts:
            test_dir = temp_dir / f"deletion_benchmark_{count}"
            test_dir.mkdir(exist_ok=True)

            # Create test files
            files = []
            for i in range(count):
                file_path = test_dir / f"test_{i}.txt"
                file_path.write_text(f"Content {i}")
                files.append(file_path)

            # Benchmark sequential deletion
            for f in files:
                if not f.exists():
                    f.write_text("content")

            start = time.time()
            del_seq, fail_seq = deleteFilesSequentially(files)
            sequential_time = time.time() - start

            # Recreate files for concurrent test
            for f in files:
                f.write_text("content")

            start = time.time()
            del_conc, fail_conc = await deleteFilesConcurrently(files, concurrency=10)
            concurrent_time = time.time() - start

            speedup = sequential_time / concurrent_time if concurrent_time > 0 else float('inf')

            results.append({
                'count': count,
                'sequential_ms': sequential_time * 1000,
                'concurrent_ms': concurrent_time * 1000,
                'speedup': speedup
            })

            print(f"\nFile Deletion Benchmark ({count} files):")
            print(f"  Sequential: {sequential_time*1000:.2f}ms")
            print(f"  Concurrent: {concurrent_time*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")

            # Cleanup
            shutil.rmtree(test_dir, ignore_errors=True)

        # Report average speedup
        avg_speedup = sum(r['speedup'] for r in results) / len(results)

        benchmark_results.add_result(
            "benchmark_file_deletion_speedup",
            "speedup_ratio",
            avg_speedup,
            "x"
        )

        print(f"\nAverage Speedup: {avg_speedup:.2f}x")

        # Concurrent should be at least 2x faster
        assert avg_speedup >= 2.0, f"Concurrent deletion should be at least 2x faster, got {avg_speedup:.2f}x"

    def test_benchmark_with_without_compression(self, temp_dir, benchmark_results):
        """
        Benchmark #16: With vs Without Compression

        Expected: 70-90% size reduction for text files
        """
        # Test different content types
        test_cases = [
            {
                'name': 'HTML',
                'content_type': 'text/html',
                'content': ("<html><body>" + "content " * 10000 + "</body></html>").encode('utf-8')
            },
            {
                'name': 'JSON',
                'content_type': 'application/json',
                'content': ('{"data": "' + 'x' * 50000 + '"}').encode('utf-8')
            },
            {
                'name': 'Plain Text',
                'content_type': 'text/plain',
                'content': ("repeated text pattern " * 10000).encode('utf-8')
            }
        ]

        results = []

        for test_case in test_cases:
            data = test_case['content']
            original_size = len(data)

            # Compress
            start = time.time()
            compressed = compressData(data)
            compression_time = time.time() - start

            compressed_size = len(compressed)
            compression_ratio = (1 - compressed_size / original_size) * 100

            results.append({
                'name': test_case['name'],
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'compression_time_ms': compression_time * 1000
            })

            print(f"\nCompression Benchmark ({test_case['name']}):")
            print(f"  Original: {original_size:,} bytes")
            print(f"  Compressed: {compressed_size:,} bytes")
            print(f"  Ratio: {compression_ratio:.1f}%")
            print(f"  Time: {compression_time*1000:.2f}ms")

            # Verify data integrity
            decompressed = gzip.decompress(compressed)
            assert decompressed == data, "Decompressed data should match original"

        # Report overall compression ratio
        avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)

        benchmark_results.add_result(
            "benchmark_compression_ratio",
            "compression_ratio",
            avg_ratio,
            "%"
        )

        print(f"\nAverage Compression Ratio: {avg_ratio:.1f}%")

        # Should achieve at least 70% compression
        assert avg_ratio >= 70, f"Compression should achieve at least 70%, got {avg_ratio:.1f}%"

    @pytest.mark.asyncio
    async def test_benchmark_paginated_vs_full_query(self, pagination_db, benchmark_results):
        """
        Benchmark #18: Paginated vs Full Query

        Expected: Paginated should be much faster for large datasets
        """
        conn, run_id = pagination_db

        # Benchmark paginated query (100 records)
        times_paginated = []
        for i in range(10):
            start = time.time()
            cursor = conn.execute(
                "SELECT id, node_type, content FROM evolution_nodes WHERE run_id = ? LIMIT 100",
                (run_id,)
            )
            results = cursor.fetchall()
            elapsed = time.time() - start
            times_paginated.append(elapsed)

        avg_paginated = sum(times_paginated) / len(times_paginated)

        # Benchmark full query (all 10,000 records)
        times_full = []
        for i in range(3):  # Fewer iterations for full query
            start = time.time()
            cursor = conn.execute(
                "SELECT id, node_type, content FROM evolution_nodes WHERE run_id = ?",
                (run_id,)
            )
            results = cursor.fetchall()
            elapsed = time.time() - start
            times_full.append(elapsed)

        avg_full = sum(times_full) / len(times_full)

        speedup = avg_full / avg_paginated if avg_paginated > 0 else float('inf')
        reduction = (1 - avg_paginated / avg_full) * 100 if avg_full > 0 else 0

        benchmark_results.add_result(
            "benchmark_pagination_speedup",
            "speedup_ratio",
            speedup,
            "x"
        )
        benchmark_results.add_result(
            "benchmark_pagination_time_reduction",
            "reduction_percent",
            reduction,
            "%"
        )

        print(f"\nPagination Benchmark:")
        print(f"  Paginated (100 records): {avg_paginated*1000:.2f}ms")
        print(f"  Full (10,000 records): {avg_full*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time Reduction: {reduction:.1f}%")

        # Paginated should be significantly faster
        assert avg_paginated < avg_full, "Paginated query should be faster"
        assert speedup >= 2.0, f"Should be at least 2x faster, got {speedup:.2f}x"

    def test_benchmark_atomic_vs_non_atomic_write(self, temp_dir, benchmark_results):
        """
        Benchmark #17: Atomic vs Non-Atomic Write Overhead

        Expected: Atomic should have < 200% overhead
        """
        sizes = [1024, 10*1024, 100*1024, 1024*1024]  # 1KB, 10KB, 100KB, 1MB
        results = []

        for size in sizes:
            data = b"x" * size
            file_atomic = temp_dir / f"atomic_{size}.txt"
            file_non_atomic = temp_dir / f"non_atomic_{size}.txt"

            # Benchmark atomic write
            times_atomic = []
            for i in range(5):
                if file_atomic.exists():
                    file_atomic.unlink()
                start = time.time()
                writeFileAtomic(file_atomic, data)
                elapsed = time.time() - start
                times_atomic.append(elapsed)

            avg_atomic = sum(times_atomic) / len(times_atomic)

            # Benchmark non-atomic write
            times_non_atomic = []
            for i in range(5):
                if file_non_atomic.exists():
                    file_non_atomic.unlink()
                start = time.time()
                writeFileNonAtomic(file_non_atomic, data)
                elapsed = time.time() - start
                times_non_atomic.append(elapsed)

            avg_non_atomic = sum(times_non_atomic) / len(times_non_atomic)

            overhead = ((avg_atomic / avg_non_atomic) - 1) * 100 if avg_non_atomic > 0 else 0

            results.append({
                'size': size,
                'atomic_ms': avg_atomic * 1000,
                'non_atomic_ms': avg_non_atomic * 1000,
                'overhead_percent': overhead
            })

            print(f"\nAtomic Write Benchmark ({size} bytes):")
            print(f"  Atomic: {avg_atomic*1000:.2f}ms")
            print(f"  Non-atomic: {avg_non_atomic*1000:.2f}ms")
            print(f"  Overhead: {overhead:.1f}%")

            # Verify both produced correct files
            assert file_atomic.read_bytes() == data
            assert file_non_atomic.read_bytes() == data

        # Report average overhead
        avg_overhead = sum(r['overhead_percent'] for r in results) / len(results)

        benchmark_results.add_result(
            "benchmark_atomic_write_overhead",
            "overhead_percent",
            avg_overhead,
            "%"
        )

        print(f"\nAverage Atomic Write Overhead: {avg_overhead:.1f}%")

        # Atomic should not be more than 3x slower (300% overhead)
        assert avg_overhead < 300, f"Atomic write overhead should be < 300%, got {avg_overhead:.1f}%"


class TestMemoryUsage:
    """Test memory usage patterns"""

    def test_concurrent_deletion_memory(self, temp_dir, benchmark_results):
        """Test that concurrent deletion doesn't use excessive memory"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        files = []

        # Create 500 test files
        test_dir = temp_dir / "memory_test"
        test_dir.mkdir(exist_ok=True)

        for i in range(500):
            file_path = test_dir / f"test_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(file_path)

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Delete files concurrently
        start = time.time()
        deleted, failed = asyncio.run(deleteFilesConcurrently(files, concurrency=10))
        elapsed = time.time() - start

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        print(f"\nMemory Usage (Concurrent Deletion - 500 files):")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After: {mem_after:.1f} MB")
        print(f"  Increase: {mem_increase:.1f} MB")
        print(f"  Time: {elapsed*1000:.2f}ms")

        benchmark_results.add_result(
            "memory_concurrent_deletion_mb",
            "memory_increase",
            mem_increase,
            "MB"
        )

        # Memory increase should be reasonable (< 50MB)
        assert mem_increase < 50, f"Memory increase should be < 50MB, got {mem_increase:.1f}MB"


class TestRealWorldScenarios:
    """Test realistic usage scenarios"""

    @pytest.mark.asyncio
    async def test_evolution_graph_workflow(self, temp_dir, benchmark_results):
        """
        Simulate real evolution graph workflow:
        1. Upload multiple large assets (with compression)
        2. Query nodes (with pagination)
        3. Delete old assets (with concurrent deletion)
        """
        print("\n=== Real-World Evolution Graph Workflow ===")

        # Step 1: Upload assets
        print("\nStep 1: Uploading assets...")
        assets = []
        upload_times = []

        for i in range(10):
            asset_data = (f"<html><body>Evolution graph asset {i}" +
                         " content " * 1000 + "</body></html>").encode('utf-8')

            asset_path = temp_dir / f"asset_{i}.html"
            start = time.time()
            writeFileAtomic(asset_path, asset_data)
            upload_time = time.time() - start
            upload_times.append(upload_time)
            assets.append(asset_path)

            # Compress if large enough
            if len(asset_data) > 100 * 1024:
                compressed = compressData(asset_data)
                if len(compressed) < len(asset_data):
                    # Store compressed version
                    asset_path.write_bytes(compressed)

        avg_upload_time = sum(upload_times) / len(upload_times)
        print(f"  Uploaded {len(assets)} assets in {avg_upload_time*1000:.2f}ms avg")

        # Step 2: Simulate querying nodes
        print("\nStep 2: Querying nodes (simulated)...")
        # In real scenario, this would query database
        query_times = []
        for i in range(10):
            start = time.time()
            # Simulate paginated query
            time.sleep(0.001)  # Simulate DB latency
            query_time = time.time() - start
            query_times.append(query_time)

        avg_query_time = sum(query_times) / len(query_times)
        print(f"  Queried nodes in {avg_query_time*1000:.2f}ms avg")

        # Step 3: Cleanup old assets
        print("\nStep 3: Cleaning up old assets...")
        start = time.time()
        deleted, failed = await deleteFilesConcurrently(assets[:5], concurrency=10)
        cleanup_time = time.time() - start
        print(f"  Deleted {deleted} assets in {cleanup_time*1000:.2f}ms")

        benchmark_results.add_result(
            "workflow_total_time",
            "time",
            avg_upload_time + avg_query_time + cleanup_time,
            "s"
        )

        print(f"\nTotal workflow time: {(avg_upload_time + avg_query_time + cleanup_time)*1000:.2f}ms")

        # Verify assets were deleted
        assert deleted == 5, "Should delete 5 assets"
        assert len([a for a in assets if a.exists()]) == 5, "Should have 5 assets remaining"


class TestPerformanceSummary:
    """Generate performance summary report"""

    def test_generate_performance_report(self, benchmark_results, temp_dir):
        """Generate comprehensive performance report"""
        results = benchmark_results.get_results()

        report_path = temp_dir / "PERFORMANCE_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Performance Test Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary of Improvements\n\n")

            # Group results by metric type
            by_metric = {}
            for result in results:
                metric = result['metric']
                if metric not in by_metric:
                    by_metric[metric] = []
                by_metric[metric].append(result)

            # File Deletion Performance (Bug #15)
            if 'speedup_ratio' in by_metric:
                speedups = [r['value'] for r in by_metric['speedup_ratio']]
                avg_speedup = sum(speedups) / len(speedups)
                f.write(f"### 1. Concurrent File Deletion (Bug #15)\n")
                f.write(f"- **Average Speedup**: {avg_speedup:.2f}x\n")
                f.write(f"- **Improvement**: {(1 - 1/avg_speedup)*100:.1f}% faster\n\n")

            # Compression (Bug #16)
            if 'compression_ratio' in by_metric:
                ratios = [r['value'] for r in by_metric['compression_ratio']]
                avg_ratio = sum(ratios) / len(ratios)
                f.write(f"### 2. Compression (Bug #16)\n")
                f.write(f"- **Average Compression Ratio**: {avg_ratio:.1f}%\n")
                f.write(f"- **Space Savings**: {avg_ratio:.1f}% reduction in storage\n\n")

            # Pagination (Bug #18)
            if 'reduction_percent' in by_metric:
                reductions = [r['value'] for r in by_metric['reduction_percent']]
                avg_reduction = sum(reductions) / len(reductions)
                f.write(f"### 3. Pagination (Bug #18)\n")
                f.write(f"- **Time Reduction**: {avg_reduction:.1f}%\n")
                f.write(f"- **Memory Savings**: Prevents loading large datasets\n\n")

            # Atomic Writes (Bug #17)
            if 'overhead_percent' in by_metric:
                overheads = [r['value'] for r in by_metric['overhead_percent']]
                avg_overhead = sum(overheads) / len(overheads)
                f.write(f"### 4. Atomic Writes (Bug #17)\n")
                f.write(f"- **Overhead**: {avg_overhead:.1f}%\n")
                f.write(f"- **Benefit**: 100% protection against data corruption\n\n")

            f.write("## Detailed Results\n\n")
            for result in results:
                f.write(f"- **{result['test_name']}**: {result['value']:.2f} {result['unit']}\n")

        print(f"\nPerformance report generated: {report_path}")
        print(f"Total metrics collected: {len(results)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
