# Performance Fixes Testing Guide

**Date:** 2026-01-18
**Purpose:** Verify critical performance and resource management fixes

---

## Quick Start Testing

### 1. Memory Leak Detection Test

```typescript
/**
 * Test script: test-memory-leaks.ts
 * Purpose: Verify memory leaks are fixed
 */

import { AIAgentBubble } from './bubbles/service-bubble/ai-agent.js';

async function testMemoryLeaks() {
  console.log('=== Memory Leak Test ===');
  console.log('Testing 1000 conversations...');

  const initialMemory = process.memoryUsage().heapUsed;

  for (let i = 0; i < 1000; i++) {
    const agent = new AIAgentBubble({
      message: `Test message ${i}`,
      systemPrompt: 'You are a helpful assistant',
      model: { model: 'openai/gpt-4' },
    });

    await agent.action();

    if (i % 100 === 0) {
      const currentMemory = process.memoryUsage().heapUsed;
      const memoryMB = (currentMemory - initialMemory) / 1024 / 1024;
      console.log(`Iteration ${i}: Memory delta = ${memoryMB.toFixed(2)}MB`);
    }
  }

  const finalMemory = process.memoryUsage().heapUsed;
  const totalDelta = (finalMemory - initialMemory) / 1024 / 1024;

  console.log(`\nResults:`);
  console.log(`Total memory delta: ${totalDelta.toFixed(2)}MB`);
  console.log(`Status: ${totalDelta < 100 ? '✓ PASS' : '✗ FAIL (memory leak detected)'} `);
}

testMemoryLeaks().catch(console.error);
```

**Expected Output:**
```
Iteration 0: Memory delta = 5.23MB
Iteration 100: Memory delta = 12.45MB
Iteration 200: Memory delta = 14.12MB
Iteration 300: Memory delta = 15.01MB
Iteration 400: Memory delta = 15.23MB
Iteration 500: Memory delta = 15.45MB
...
Iteration 900: Memory delta = 15.67MB

Results:
Total memory delta: 15.89MB
Status: ✓ PASS
```

### 2. Timer Leak Detection Test

```typescript
/**
 * Test script: test-timer-leaks.ts
 * Purpose: Verify timer handles are properly cleaned up
 */

import { HttpBubble } from './bubbles/service-bubble/http.js';

async function testTimerLeaks() {
  console.log('=== Timer Leak Test ===');
  console.log('Testing 5000 HTTP requests with 50% failure rate...');

  const testUrls = [
    'https://httpbin.org/status/200',  // Success
    'https://httpbin.org/status/404',  // Client error
    'https://invalid-domain-12345.com', // Network error
  ];

  for (let i = 0; i < 5000; i++) {
    const url = testUrls[i % testUrls.length];
    const httpBubble = new HttpBubble({ url, method: 'GET' });

    try {
      await httpBubble.action();
    } catch (error) {
      // Expected to fail for some requests
    }

    if (i % 500 === 0) {
      const memory = process.memoryUsage();
      console.log(`Request ${i}: RSS = ${Math.round(memory.rss / 1024 / 1024)}MB`);
    }
  }

  const finalMemory = process.memoryUsage();
  console.log(`\nFinal RSS: ${Math.round(finalMemory.rss / 1024 / 1024)}MB`);
  console.log('Status: ✓ PASS (if RSS < 200MB)');
}

testTimerLeaks().catch(console.error);
```

**Expected Output:**
```
Request 0: RSS = 45MB
Request 500: RSS = 52MB
Request 1000: RSS = 58MB
Request 1500: RSS = 61MB
Request 2000: RSS = 63MB
...
Request 4500: RSS = 68MB

Final RSS: 69MB
Status: ✓ PASS
```

### 3. File Watcher Leak Test

```typescript
/**
 * Test script: test-watcher-leaks.ts
 * Purpose: Verify file watchers are limited and cleaned up
 */

import { FileProcessorTool } from './bubbles/tool-bubble/file-processor-tool.js';
import { mkdirSync, rmSync } from 'fs';
import { join } from 'path';

async function testWatcherLeaks() {
  console.log('=== File Watcher Leak Test ===');

  // Create test directories
  const testDir = join(process.cwd(), 'test-watchers');
  const maxWatchers = 150; // Try to create more than limit

  try {
    mkdirSync(testDir, { recursive: true });

    for (let i = 0; i < maxWatchers; i++) {
      const watchDir = join(testDir, `dir-${i}`);
      mkdirSync(watchDir, { recursive: true });

      const tool = new FileProcessorTool({
        operation: 'watch',
        directoryPath: watchDir,
        watchDuration: 100, // Watch for 100ms
      });

      await tool.action();

      if (i % 20 === 0) {
        const memory = process.memoryUsage();
        console.log(`Watchers created: ${i + 1}, RSS: ${Math.round(memory.rss / 1024 / 1024)}MB`);
      }
    }

    console.log(`\nAttempted to create ${maxWatchers} watchers`);
    console.log('Status: ✓ PASS (system should limit to 100 watchers)');
  } finally {
    // Cleanup
    try {
      rmSync(testDir, { recursive: true, force: true });
    } catch (error) {
      console.error('Cleanup failed:', error);
    }
  }
}

testWatcherLeaks().catch(console.error);
```

**Expected Output:**
```
Watchers created: 1, RSS: 42MB
Watchers created: 21, RSS: 45MB
Watchers created: 41, RSS: 47MB
Watchers created: 61, RSS: 49MB
Watchers created: 81, RSS: 51MB
Watchers created: 100, RSS: 52MB
[FileWatcher] Maximum watcher limit reached (100). Cannot watch ...
[FileWatcher] Maximum watcher limit reached (100). Cannot watch ...
...

Attempted to create 150 watchers
Status: ✓ PASS (system should limit to 100 watchers)
```

### 4. PostgreSQL Connection Leak Test

```typescript
/**
 * Test script: test-connection-leaks.ts
 * Purpose: Verify database connections are properly cleaned up
 */

import { PostgreSQLBubble } from './bubbles/service-bubble/postgresql.js';

async function testConnectionLeaks() {
  console.log('=== PostgreSQL Connection Leak Test ===');
  console.log('Testing 1000 queries with 10% error rate...');

  const queries = [
    'SELECT 1', // Success
    'SELECT * FROM nonexistent_table', // Error
    'SELECT invalid syntax', // Error
  ];

  for (let i = 0; i < 1000; i++) {
    const query = queries[i % queries.length];
    const pgBubble = new PostgreSQLBubble({
      query,
      allowedOperations: ['SELECT'],
      timeout: 5000,
    });

    try {
      await pgBubble.action();
    } catch (error) {
      // Expected to fail for some queries
    }

    if (i % 100 === 0) {
      const memory = process.memoryUsage();
      console.log(`Query ${i}: RSS = ${Math.round(memory.rss / 1024 / 1024)}MB`);
    }
  }

  const finalMemory = process.memoryUsage();
  console.log(`\nFinal RSS: ${Math.round(finalMemory.rss / 1024 / 1024)}MB`);
  console.log('Status: ✓ PASS (if RSS < 150MB)');
}

testConnectionLeaks().catch(console.error);
```

**Expected Output:**
```
Query 0: RSS = 38MB
Query 100: RSS = 42MB
Query 200: RSS = 45MB
Query 300: RSS = 47MB
Query 400: RSS = 48MB
...
Query 900: RSS = 52MB

Final RSS: 53MB
Status: ✓ PASS
```

---

## Load Testing Scenarios

### Scenario 1: Sustained Load (24 Hours)

```bash
# Run load test for 24 hours
npm run load-test:24h

# Expected: Memory usage stable, no leaks
# Acceptable: RSS < 200MB throughout test
```

### Scenario 2: Burst Traffic

```bash
# Simulate burst traffic pattern
npm run load-test:burst

# Pattern: 100 requests/sec for 1 minute, then 10 requests/sec for 5 minutes
# Repeat for 2 hours
# Expected: Memory spikes during bursts, then returns to baseline
```

### Scenario 3: Error Spike

```bash
# Test error handling
npm run load-test:errors

# 50% error rate for 10000 requests
# Expected: No resource leaks despite errors
```

---

## Monitoring During Tests

### 1. Memory Profiling

```bash
# Start with heap profiling
node --heap-prof test-memory-leaks.ts

# Analyze results
node --heap-prof-process --heap-out-file=./heap-profile-*.heapsnapshot
```

### 2. CPU Profiling

```bash
# Start with CPU profiling
node --prof test-performance.ts

# Analyze results
node --prof-process isolate-*.log > processed.txt
```

### 3. Real-time Monitoring

```bash
# Use clinic.js (if installed)
npm install -g clinic
clinic doctor -- node test-memory-leaks.ts

# Or use 0x for flame graphs
npm install -g 0x
0x test-memory-leaks.ts
```

---

## Success Criteria

### Memory Usage
- [ ] RSS < 200MB after 10000 operations
- [ ] Heap usage stable (not growing monotonically)
- [ ] No "out of memory" errors during 24-hour test
- [ ] Memory returns to baseline after load spike

### Resource Cleanup
- [ ] 0 timer leaks after 5000 requests
- [ ] 0 connection leaks after 1000 queries
- [ ] 0 file watcher leaks after 100 watches
- [ ] All finally blocks execute

### Performance
- [ ] Average response time < 500ms
- [ ] P95 response time < 1000ms
- [ ] Error rate < 5%
- [ ] No degradation over time

### Cache Behavior
- [ ] LRU eviction working (cache size stays at limit)
- [ ] TTL cleanup working (old entries removed)
- [ ] No cache misses for recently used items
- [ ] Cache hit rate > 80% for repeated operations

---

## Troubleshooting

### Issue: Memory Still Growing

**Diagnosis:**
```typescript
// Add detailed logging
PerformanceMonitor.logMemoryUsage('before-operation');
await operation();
PerformanceMonitor.logMemoryUsage('after-operation');

// Check for leaks
const leakReport = PerformanceMonitor.detectMemoryLeaks(50);
console.log('Leak Report:', leakReport);
```

**Possible Causes:**
1. Missing finally block
2. Event listener not removed
3. Global variable accumulation
4. Closure retaining large objects

### Issue: High CPU Usage

**Diagnosis:**
```bash
# Generate CPU profile
node --prof test-performance.ts
node --prof-process isolate-*.log > profile.txt
```

**Possible Causes:**
1. Infinite loop
2. Frequent garbage collection
3. Expensive regular expressions
4. Synchronous file operations

### Issue: Slow Response Times

**Diagnosis:**
```typescript
// Add timing instrumentation
const start = Date.now();
await operation();
console.log(`Operation took ${Date.now() - start}ms`);
```

**Possible Causes:**
1. Blocking operations
2. Large payload processing
3. Network latency
4. Database query performance

---

## Automated Testing

### Create Test Suite

```typescript
// test/performance.test.ts
import { describe, it, expect } from '@jest/globals';
import { PerformanceMonitor } from '../src/utils/performance-monitor.js';

describe('Performance Tests', () => {
  it('should not leak memory during 1000 AI agent conversations', async () => {
    const initialMemory = process.memoryUsage().heapUsed;

    for (let i = 0; i < 1000; i++) {
      const agent = new AIAgentBubble({
        message: `Test ${i}`,
        model: { model: 'openai/gpt-4' },
      });
      await agent.action();
    }

    const finalMemory = process.memoryUsage().heapUsed;
    const memoryDelta = (finalMemory - initialMemory) / 1024 / 1024;

    expect(memoryDelta).toBeLessThan(100); // Less than 100MB
  });

  it('should not leak timer handles', async () => {
    for (let i = 0; i < 1000; i++) {
      const http = new HttpBubble({
        url: 'https://httpbin.org/status/200',
      });
      await http.action();
    }

    const leakReport = PerformanceMonitor.detectMemoryLeaks(10);
    expect(leakReport.potentialLeak).toBe(false);
  });
});
```

### Run Tests

```bash
# Run all performance tests
npm test test/performance.test.ts

# Run with coverage
npm test -- --coverage

# Run with monitoring
node --experimental-modules test/performance.test.ts
```

---

## Continuous Monitoring

### Production Monitoring Setup

```typescript
// Add to main application startup
import { PerformanceMonitor } from './utils/performance-monitor.js';

// Log memory every 5 minutes
setInterval(() => {
  PerformanceMonitor.logMemoryUsage('periodic-check');

  // Check for leaks
  const leakReport = PerformanceMonitor.detectMemoryLeaks(100);
  if (leakReport.potentialLeak) {
    console.error('WARNING: Potential memory leak detected!', leakReport);
    // Send alert to monitoring system
  }
}, 5 * 60 * 1000);

// Subscribe to performance events
PerformanceMonitor.on('operation-complete', ({ metrics }) => {
  if (metrics.duration > 5000) {
    console.warn(`Slow operation detected: ${metrics.context} (${metrics.duration}ms)`);
  }
});

PerformanceMonitor.on('operation-failed', ({ metrics }) => {
  console.error(`Operation failed: ${metrics.context}`, metrics.error);
});
```

---

## Verification Checklist

Before deploying to production:

- [ ] All unit tests pass
- [ ] Memory leak tests pass (1000 iterations)
- [ ] Timer leak tests pass (5000 requests)
- [ ] File watcher tests pass (150 attempts)
- [ ] Connection leak tests pass (1000 queries)
- [ ] Load test completed (24 hours)
- [ ] No memory leaks detected in profiling
- [ ] Performance metrics within acceptable range
- [ ] Error handling verified (all error paths tested)
- [ ] Documentation updated
- [ ] Runbooks created
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set

---

## Next Steps After Deployment

1. **Monitor First 24 Hours**
   - Check memory usage every hour
   - Verify no leaks in production traffic
   - Validate performance metrics

2. **Analyze Performance Data**
   - Review response time distributions
   - Check error rates by operation type
   - Identify any unexpected patterns

3. **Tune if Necessary**
   - Adjust cache sizes based on usage
   - Modify timeout values if needed
   - Fine-tune resource limits

4. **Document Learnings**
   - Record any issues found
   - Document resolutions
   - Update runbooks

---

**Last Updated:** 2026-01-18
**Status:** READY FOR TESTING
