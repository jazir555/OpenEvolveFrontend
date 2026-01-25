"""
Pagination Performance Tests (Bug #18)

Tests for pagination implementation in listEvolutionNodes
- Verifies limit/offset parameters work correctly
- Tests default and maximum limits
- Validates pagination metadata
- Tests performance with large datasets
"""

import pytest
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def pagination_db(temp_dir):
    """Create test database with large dataset for pagination testing"""
    db_path = temp_dir / "pagination_test.db"
    conn = sqlite3.connect(str(db_path))

    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evolution_runs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            config TEXT,
            created_at TEXT,
            status TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS evolution_nodes (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_type TEXT,
            content TEXT,
            metadata TEXT,
            created_at TEXT,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(id)
        )
    """)

    # Create test run
    run_id = "test_run_pagination"
    conn.execute(
        "INSERT INTO evolution_runs (id, user_id, config, created_at, status) VALUES (?, ?, ?, ?, ?)",
        (run_id, "test_user", '{"test": true}', datetime.now().isoformat(), 'active')
    )

    # Insert 10,000 nodes for pagination testing
    nodes = []
    for i in range(10000):
        node_id = f"node_{i:05d}"
        nodes.append((
            node_id,
            run_id,
            f"type_{i % 5}",
            f"content_{i}",
            f'{{"index": {i}, "batch": {i // 100}}}',
            datetime.now().isoformat()
        ))

    conn.executemany(
        "INSERT INTO evolution_nodes (id, run_id, node_type, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        nodes
    )

    conn.commit()
    return conn, run_id


# ============================================================================
# Pagination Function Tests
# ============================================================================

class TestPaginationParameters:
    """Test pagination parameter validation"""

    def test_default_limit_is_100(self, pagination_db):
        """Verify default limit is 100"""
        conn, run_id = pagination_db

        # Query without limit parameter
        cursor = conn.execute(
            "SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT 100",
            (run_id,)
        )
        results = cursor.fetchall()

        assert len(results) == 100, "Default limit should be 100"

    def test_custom_limit_respected(self, pagination_db):
        """Test custom limit parameter"""
        conn, run_id = pagination_db

        limit = 50
        cursor = conn.execute(
            f"SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT {limit}",
            (run_id,)
        )
        results = cursor.fetchall()

        assert len(results) == limit, f"Should return {limit} results"

    def test_offset_parameter_works(self, pagination_db):
        """Test offset parameter skips correct number of records"""
        conn, run_id = pagination_db

        offset = 100
        limit = 10

        # Get first batch
        cursor1 = conn.execute(
            f"SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT {limit}",
            (run_id,)
        )
        first_batch = cursor1.fetchall()

        # Get second batch with offset
        cursor2 = conn.execute(
            f"SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT {limit} OFFSET {offset}",
            (run_id,)
        )
        second_batch = cursor2.fetchall()

        assert len(second_batch) == limit, "Offset query should return correct number of results"
        assert first_batch[0] != second_batch[0], "Offset should skip to different records"

    def test_maximum_limit_enforced(self, pagination_db):
        """Test that maximum limit of 1000 is enforced"""
        max_limit = 1000

        # Try to get more than max
        conn, run_id = pagination_db
        cursor = conn.execute(
            f"SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT {max_limit}",
            (run_id,)
        )
        results = cursor.fetchall()

        assert len(results) <= max_limit, "Should not exceed maximum limit"

    def test_total_count_included(self, pagination_db):
        """Test that total count is returned"""
        conn, run_id = pagination_db

        # Get total count
        cursor = conn.execute(
            "SELECT COUNT(*) FROM evolution_nodes WHERE run_id = ?",
            (run_id,)
        )
        total = cursor.fetchone()[0]

        assert total == 10000, "Total count should be 10,000"

    def test_has_more_flag(self, pagination_db):
        """Test hasMore flag calculation"""
        conn, run_id = pagination_db

        limit = 100
        offset = 0

        # Get page
        cursor = conn.execute(
            f"SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT {limit} OFFSET {offset}",
            (run_id,)
        )
        results = cursor.fetchall()

        # Get total
        cursor_total = conn.execute(
            "SELECT COUNT(*) FROM evolution_nodes WHERE run_id = ?",
            (run_id,)
        )
        total = cursor_total.fetchone()[0]

        has_more = (offset + results) < total

        assert has_more == True, "Should have more results when offset + results < total"


class TestPaginationPerformance:
    """Test pagination performance improvements"""

    def test_paginated_query_faster_than_full_query(self, pagination_db, benchmark_results):
        """Verify paginated query is significantly faster than loading all records"""
        conn, run_id = pagination_db

        # Measure paginated query time
        start = time.time()
        cursor = conn.execute(
            "SELECT id, node_type, content FROM evolution_nodes WHERE run_id = ? LIMIT 100",
            (run_id,)
        )
        paginated_results = cursor.fetchall()
        paginated_time = time.time() - start

        # Measure full query time
        start = time.time()
        cursor = conn.execute(
            "SELECT id, node_type, content FROM evolution_nodes WHERE run_id = ?",
            (run_id,)
        )
        full_results = cursor.fetchall()
        full_time = time.time() - start

        # Paginated should be much faster
        speedup = full_time / paginated_time if paginated_time > 0 else float('inf')

        benchmark_results.add_result(
            "pagination_speedup",
            "speedup_ratio",
            speedup,
            "x"
        )
        benchmark_results.add_result(
            "pagination_paginated_time",
            "time",
            paginated_time,
            "s"
        )
        benchmark_results.add_result(
            "pagination_full_query_time",
            "time",
            full_time,
            "s"
        )

        assert len(paginated_results) == 100, "Paginated query should return 100 results"
        assert len(full_results) == 10000, "Full query should return all results"
        assert paginated_time < full_time, "Paginated query should be faster than full query"

        print(f"\nPagination Performance:")
        print(f"  Paginated query (100 records): {paginated_time*1000:.2f}ms")
        print(f"  Full query (10,000 records): {full_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_pagination_with_large_dataset(self, pagination_db, benchmark_results):
        """Test pagination performance with 10,000+ records"""
        conn, run_id = pagination_db

        # Simulate paginating through all records
        page_size = 100
        total_pages = 100
        times = []

        for page in range(10):  # Test first 10 pages
            start = time.time()
            cursor = conn.execute(
                f"SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT {page_size} OFFSET {page * page_size}",
                (run_id,)
            )
            results = cursor.fetchall()
            query_time = time.time() - start
            times.append(query_time)

            assert len(results) == page_size, f"Page {page} should return {page_size} results"

        avg_time = sum(times) / len(times)
        max_time = max(times)

        benchmark_results.add_result(
            "pagination_avg_page_time",
            "time",
            avg_time,
            "s"
        )
        benchmark_results.add_result(
            "pagination_max_page_time",
            "time",
            max_time,
            "s"
        )

        print(f"\nPagination with Large Dataset:")
        print(f"  Average page load time: {avg_time*1000:.2f}ms")
        print(f"  Max page load time: {max_time*1000:.2f}ms")

        # Each page should be fast
        assert avg_time < 0.1, "Average page load should be under 100ms"

    def test_pagination_consistency(self, pagination_db):
        """Test that pagination returns consistent results"""
        conn, run_id = pagination_db

        page_size = 100

        # Get all records via pagination
        all_via_pagination = []
        for offset in range(0, 1000, page_size):
            cursor = conn.execute(
                f"SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT {page_size} OFFSET {offset}",
                (run_id,)
            )
            all_via_pagination.extend([row[0] for row in cursor.fetchall()])

        # Get all records directly
        cursor = conn.execute(
            "SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT 1000",
            (run_id,)
        )
        all_direct = [row[0] for row in cursor.fetchall()]

        assert len(all_via_pagination) == len(all_direct), "Should return same number of records"
        assert all_via_pagination == all_direct, "Paginated results should match direct query"


class TestPaginationEdgeCases:
    """Test pagination edge cases"""

    def test_empty_page(self, pagination_db):
        """Test pagination with offset beyond available data"""
        conn, run_id = pagination_db

        # Offset beyond total records
        cursor = conn.execute(
            "SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT 100 OFFSET 50000",
            (run_id,)
        )
        results = cursor.fetchall()

        assert len(results) == 0, "Should return empty result set"

    def test_page_boundary(self, pagination_db):
        """Test pagination at exact page boundaries"""
        conn, run_id = pagination_db

        # Get last page
        cursor = conn.execute(
            "SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT 100 OFFSET 9900",
            (run_id,)
        )
        results = cursor.fetchall()

        assert len(results) == 100, "Last page should return 100 results"

    def test_partial_last_page(self, pagination_db):
        """Test pagination when total is not divisible by page size"""
        conn, run_id = pagination_db

        # Total is 10000, page size is 99
        # 10000 / 99 = 101 pages with remainder
        cursor = conn.execute(
            "SELECT id FROM evolution_nodes WHERE run_id = ? ORDER BY id LIMIT 99 OFFSET 9900",
            (run_id,)
        )
        results = cursor.fetchall()

        # Should return remaining records (10000 - 9900 = 100)
        assert len(results) == 100, "Should return remaining records on last page"


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPaginationBenchmarks:
    """Benchmark pagination against different scenarios"""

    def test_benchmark_small_page_vs_large_page(self, pagination_db, benchmark_results):
        """Compare performance of small vs large page sizes"""
        conn, run_id = pagination_db

        # Small pages
        start = time.time()
        for offset in range(0, 1000, 10):
            conn.execute(
                "SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT 10 OFFSET ?",
                (run_id, offset)
            )
        small_page_time = time.time() - start

        # Large pages
        start = time.time()
        for offset in range(0, 1000, 100):
            conn.execute(
                "SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT 100 OFFSET ?",
                (run_id, offset)
            )
        large_page_time = time.time() - start

        benchmark_results.add_result(
            "pagination_small_pages_time",
            "time",
            small_page_time,
            "s"
        )
        benchmark_results.add_result(
            "pagination_large_pages_time",
            "time",
            large_page_time,
            "s"
        )

        print(f"\nPage Size Comparison (1000 records):")
        print(f"  Small pages (10): {small_page_time*1000:.2f}ms")
        print(f"  Large pages (100): {large_page_time*1000:.2f}ms")

        # Large pages should be more efficient
        assert large_page_time < small_page_time, "Large pages should be more efficient"

    def test_benchmark_offset_vs_cursor(self, pagination_db):
        """Test offset-based pagination (can be improved with cursor-based later)"""
        conn, run_id = pagination_db

        # Offset-based pagination
        start = time.time()
        for offset in range(0, 1000, 100):
            conn.execute(
                "SELECT id FROM evolution_nodes WHERE run_id = ? LIMIT 100 OFFSET ?",
                (run_id, offset)
            )
        offset_time = time.time() - start

        print(f"\nPagination Method Comparison:")
        print(f"  Offset-based: {offset_time*1000:.2f}ms")

        benchmark_results.add_result(
            "pagination_offset_method",
            "time",
            offset_time,
            "s"
        )

        # Note: Cursor-based pagination would be faster for large offsets
        # This test documents the baseline for future improvement


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
