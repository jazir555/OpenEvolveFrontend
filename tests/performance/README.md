# Performance Test Suite

Comprehensive performance tests for bug fixes and improvements in the Evolution Graph API.

## Test Files

### 1. `test_pagination_performance.py` (Bug #18)
Tests for pagination implementation in `listEvolutionNodes`

**Tests:**
- Default limit is 100
- Custom limit parameter works
- Offset parameter works correctly
- Maximum limit of 1000 is enforced
- Total count is included in response
- hasMore flag is calculated correctly
- Paginated queries are faster than full queries
- Pagination works with large datasets (10,000+ nodes)
- Consistent results across pages
- Edge cases (empty pages, boundaries, partial last page)

**Expected Results:**
- Paginated queries should be 2-10x faster than full queries
- Memory usage should be controlled and predictable
- Database load reduced by up to 98% for large queries

### 2. `test_compression_performance.py` (Bug #16)
Tests for compression implementation in `createEvolutionAssetRoute`

**Tests:**
- HTML files > 100KB are compressed
- JSON files > 100KB are compressed
- Plain text files > 100KB are compressed
- Compression ratio is 70-90%
- Compressed data can be decompressed correctly
- Data integrity survives compression/decompression
- Small files (< 100KB) are not compressed
- Binary files are not compressed
- Compression speed is acceptable (> 1 MB/s)
- Decompression speed is acceptable (> 5 MB/s)

**Expected Results:**
- 70-90% size reduction for text-based assets
- Compression throughput > 1 MB/s
- Decompression throughput > 5 MB/s
- Zero data corruption

### 3. `test_concurrent_files_performance.py` (Bug #15)
Tests for concurrent file deletion implementation

**Tests:**
- Deletes 100 files in batches of 10
- Custom concurrency levels work
- Failed deletions don't block other deletions
- Returns correct deleted/failed counts
- Concurrent deletion is faster than sequential
- Performance scales with file count
- Optimal concurrency level is around 10
- Doesn't overwhelm the filesystem
- Works with large files (1MB+)
- Handles nonexistent files gracefully

**Expected Results:**
- 5-10x faster than sequential deletion
- 100ms for 100 files (vs 1000ms sequential)
- Linear scaling with file count
- Zero file system overload

### 4. `test_atomic_writes_performance.py` (Bug #17)
Tests for atomic file write implementation

**Tests:**
- Writes to temp file first
- Atomic rename happens after successful write
- Failed writes don't create target file
- Cleanup happens on failure
- Concurrent writes don't corrupt files
- Data integrity for all file sizes
- Data integrity for binary data
- Atomic write overhead is acceptable
- Idempotent retries work correctly

**Expected Results:**
- Zero partial/corrupted files
- < 300% overhead vs non-atomic writes
- Safe to retry failed uploads
- All data intact after failures

### 5. `test_performance_benchmarks.py`
Comprehensive benchmarks comparing all improvements

**Benchmarks:**
- Sequential vs Concurrent file deletion (50, 100, 200 files)
- With vs Without compression (HTML, JSON, Plain text)
- Paginated vs Full query (100 vs 10,000 records)
- Atomic vs Non-atomic write overhead (1KB, 10KB, 100KB, 1MB)
- Memory usage patterns
- Real-world workflow simulation

**Expected Results:**
- Overall 5-10x performance improvement
- 70-90% storage savings from compression
- 98% reduction in database load from pagination
- Zero data corruption from atomic writes

## Running the Tests

### Run All Performance Tests
```bash
pytest tests/performance/ -v
```

### Run Specific Test Suite
```bash
# Pagination tests
pytest tests/performance/test_pagination_performance.py -v

# Compression tests
pytest tests/performance/test_compression_performance.py -v

# Concurrent deletion tests
pytest tests/performance/test_concurrent_files_performance.py -v

# Atomic write tests
pytest tests/performance/test_atomic_writes_performance.py -v
```

### Run Comprehensive Benchmarks
```bash
pytest tests/performance/test_performance_benchmarks.py -v -s
```

### Run with Detailed Output
```bash
pytest tests/performance/ -v -s --tb=short
```

### Generate Performance Report
```bash
pytest tests/performance/test_performance_benchmarks.py::TestPerformanceSummary::test_generate_performance_report -v -s
```

## Test Dependencies

Required packages:
```bash
pip install pytest pytest-asyncio psutil
```

Optional for database tests:
```bash
pip install sqlite3
```

## Understanding the Results

### Performance Improvement Metrics

1. **Speedup Ratio**: How many times faster the new implementation is
   - Example: 5.2x speedup = 5.2 times faster (81% reduction in time)

2. **Compression Ratio**: Percentage of space saved
   - Example: 85% compression = file is 85% smaller

3. **Overhead**: Additional cost of safety/atomicity
   - Example: 150% overhead = takes 2.5x longer than baseline
   - Still acceptable if < 300%

4. **Time Reduction**: Percentage of time saved
   - Example: 75% reduction = takes 25% of original time

### What to Look For

**Green Flags:**
- Speedup > 2x for concurrent operations
- Compression ratio > 70% for text files
- Atomic write overhead < 300%
- Zero data corruption
- Linear scaling with data size

**Red Flags:**
- Speedup < 1.5x (not worth the complexity)
- Compression ratio < 50% (ineffective)
- Atomic write overhead > 500% (too expensive)
- Data corruption or integrity failures
- Exponential scaling with data size

## Continuous Monitoring

These tests should be run:
1. Before deploying changes to production
2. After any database schema changes
3. After any file system changes
4. As part of CI/CD pipeline
5. When investigating performance issues

## Performance Baselines

Based on the implementation, these are expected baselines:

| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| 100 files deletion (sequential) | ~1000ms | N/A | N/A |
| 100 files deletion (concurrent) | N/A | ~100ms | N/A |
| 500KB HTML compression | N/A | ~50KB | N/A |
| 10,000 nodes query (full) | ~5000ms | N/A | N/A |
| 100 nodes query (paginated) | N/A | ~50ms | N/A |
| Atomic write overhead | N/A | <200% | N/A |

Run the tests to get actual values for your system.

## Troubleshooting

### Tests Fail on CI/CD
- Ensure temp directory is writable
- Check file system permissions
- Verify enough disk space for large file tests

### Performance is Worse Than Expected
- Check system load (other processes)
- Verify disk I/O performance (SSD vs HDD)
- Check database indexing
- Verify network latency (for database)

### Memory Usage is High
- Reduce batch size in concurrent tests
- Check for memory leaks in test fixtures
- Verify temp files are being cleaned up

## Contributing

When adding new performance tests:
1. Create a new test file in `tests/performance/`
2. Follow the existing naming convention: `test_*_performance.py`
3. Add benchmark results to `benchmark_results` fixture
4. Update this README with test descriptions
5. Add performance baselines for your tests

## Related Documentation

- [MEDIUM_PRIORITY_PERFORMANCE_FIXES_SUMMARY.md](../../MEDIUM_PRIORITY_PERFORMANCE_FIXES_SUMMARY.md) - Implementation details
- [CLAUDE.md](../../CLAUDE.md) - Project guidelines
- [BubbleLab evolution-graph.ts](../../BubbleLab/apps/bubblelab-api/src/routes/evolution-graph.ts) - Source code
