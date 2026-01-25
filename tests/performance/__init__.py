"""
Performance Test Suite

Comprehensive performance tests for bug fixes and improvements:
- Bug #15: Concurrent File Deletion
- Bug #16: Compression for Large Assets
- Bug #17: Atomic File Writes
- Bug #18: Pagination Implementation

Run all tests:
    pytest tests/performance/ -v

Run specific test suite:
    pytest tests/performance/test_pagination_performance.py -v
    pytest tests/performance/test_compression_performance.py -v
    pytest tests/performance/test_concurrent_files_performance.py -v
    pytest tests/performance/test_atomic_writes_performance.py -v

Run benchmarks:
    pytest tests/performance/test_performance_benchmarks.py -v -s
"""

__version__ = "1.0.0"
