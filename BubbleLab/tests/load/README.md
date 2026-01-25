# BubbleLab Load Testing Suite

## Overview

This directory contains load testing scenarios for the BubbleLab system. The goal is to verify that the system can handle **1000 requests/minute** (16.67 requests/second) under normal operating conditions.

## Tools

We support two load testing tools:

1. **k6** (Recommended) - https://k6.io/
   - Modern, developer-friendly
   - JavaScript-based scenarios
   - Great CI/CD integration
   - Detailed metrics

2. **Artillery** (Alternative) - https://artillery.io/
   - YAML configuration
   - Good for simple scenarios
   - AWS integration

## Prerequisites

### For k6:

```bash
# Install k6
# macOS
brew install k6

# Linux
sudo apt-get install k6

# Windows
choco install k6

# Or download from https://k6.io/docs/getting-started/installation/
```

### For Artillery:

```bash
npm install -g artillery
```

### Environment Variables:

```bash
# Required
export API_BASE_URL="http://localhost:3000"
export API_KEY="your-api-key-here"

# Optional
export K6_PROMETHEUS_RW_SERVER_URL="http://localhost:9090/api/v1/write"
```

## Test Scenarios

### 1. Normal Load Test (k6)

**Target**: 100 req/s for 5 minutes

**Purpose**: Verify system stability under expected normal load

**Run**:
```bash
k6 run tests/load/k6-load-test.js
```

**Expected Results**:
- 95th percentile response time < 500ms
- Error rate < 1%
- 99% of requests successful

### 2. Peak Load Test (k6)

**Target**: 500 req/s for 2 minutes

**Purpose**: Verify system can handle peak traffic

**Run**:
```bash
k6 run --stage 'normal_load:5m,peak_load:2m' tests/load/k6-load-test.js
```

**Expected Results**:
- 95th percentile response time < 1000ms
- Error rate < 2%
- No system crashes

### 3. Stress Test (k6)

**Target**: 1000 req/s for 1 minute

**Purpose**: Find system breaking point

**Run**:
```bash
k6 run --stage 'normal_load:5m,peak_load:2m,stress_test:1m' tests/load/k6-load-test.js
```

**Expected Results**:
- System remains operational
- Graceful degradation (circuit breakers activate)
- No data corruption

### 4. Soak Test (k6)

**Target**: 50 req/s for 30 minutes

**Purpose**: Detect memory leaks and long-term issues

**Run**:
```bash
k6 run tests/load/k6-load-test.js --stage 'soak_test:30m'
```

**Expected Results**:
- Memory usage stable
- No performance degradation over time
- Connection pools healthy

## Service Bubble Tests

Each service bubble has dedicated tests:

### Qdrant Bubble Tests

- Create collection
- Insert points
- Search points
- Delete collection

**Target**: 100 operations/second
**Expected Latency**: p95 < 500ms

### Elasticsearch Bubble Tests

- Create index
- Index documents
- Search
- Delete index

**Target**: 100 operations/second
**Expected Latency**: p95 < 500ms

### Redis Bubble Tests

- Set value
- Get value
- Delete value

**Target**: 1000 operations/second
**Expected Latency**: p95 < 100ms

### PostgreSQL Bubble Tests

- Create table
- Insert rows
- Query
- Drop table

**Target**: 100 operations/second
**Expected Latency**: p95 < 500ms

## Workflow Tests

- Create workflow
- Execute workflow
- Check status
- Delete workflow

**Target**: 20 workflows/second
**Expected Latency**: p95 < 5000ms (workflows are slower)

## Connection Pooling Tests

- Rapid consecutive requests
- Test connection reuse
- Verify no connection leaks

**Target**: 10 concurrent requests
**Expected Latency**: p95 < 100ms

## Running Specific Tests

### Run only Redis tests:

```bash
k6 run --exec 'redisTests' tests/load/k6-load-test.js
```

### Run only Qdrant tests:

```bash
k6 run --exec 'qdrantTests' tests/load/k6-load-test.js
```

### Run with custom VUs:

```bash
k6 run --vus 50 --duration 5m tests/load/k6-load-test.js
```

## Performance Baselines

### Current Baselines (To be established after first run):

| Operation | Target | p50 | p95 | p99 | Max |
|-----------|--------|-----|-----|-----|-----|
| Redis Get | 1000 req/s | - | - | - | - |
| Redis Set | 1000 req/s | - | - | - | - |
| Qdrant Search | 100 req/s | - | - | - | - |
| Qdrant Insert | 100 req/s | - | - | - | - |
| Elasticsearch Search | 100 req/s | - | - | - | - |
| Elasticsearch Index | 100 req/s | - | - | - | - |
| PostgreSQL Query | 100 req/s | - | - | - | - |
| PostgreSQL Insert | 100 req/s | - | - | - | - |
| Workflow Execution | 20 req/s | - | - | - | - |

**After first run, fill in actual values.**

## Identifying Bottlenecks

### Common Bottlenecks:

1. **Database Connection Pool Exhausted**
   - Symptom: Response times increase, timeouts
   - Solution: Increase pool size

2. **Circuit Breaker Open**
   - Symptom: Requests failing fast
   - Solution: Fix underlying service issue

3. **Memory Leaks**
   - Symptom: Memory usage increases over time
   - Solution: Fix leak (common in long-running tests)

4. **CPU Saturation**
   - Symptom: Response times increase, CPU at 100%
   - Solution: Scale horizontally or optimize code

5. **Network Latency**
   - Symptom: High response times, low CPU
   - Solution: Optimize network calls, add caching

## Troubleshooting

### High Error Rate

1. Check service health: `curl http://localhost:3000/health`
2. Check logs: `tail -f logs/app.log`
3. Check circuit breakers: Look for "circuit breaker open" in logs

### Slow Response Times

1. Check database query performance
2. Check connection pool settings
3. Check for N+1 queries
4. Profile with: `npm run profile`

### Memory Leaks

1. Run soak test for 30+ minutes
2. Monitor memory usage: `docker stats`
3. Use heap snapshot: `node --heap-prof`
4. Check for: Event listeners not removed, caches not cleared

### Connection Pool Issues

1. Check pool size settings
2. Verify connections are released
3. Check for: "connection timeout" errors
4. Monitor: Active connections vs pool size

## CI/CD Integration

### GitHub Actions Example:

```yaml
name: Load Tests

on:
  schedule:
    - cron: '0 2 * * *' # Run daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install k6
        run: |
          sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Start services
        run: docker-compose up -d

      - name: Wait for services
        run: sleep 30

      - name: Run load tests
        run: k6 run tests/load/k6-load-test.js
        env:
          API_BASE_URL: http://localhost:3000
          API_KEY: ${{ secrets.API_KEY }}

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: |
            k6-report.json
            k6-report.html
```

## Prometheus Integration

### Export metrics to Prometheus:

```bash
export K6_PROMETHEUS_RW_SERVER_URL="http://localhost:9090/api/v1/write"
export K6_PROMETHEUS_RW_TLSCONFIG=""
k6 run --out json=test.json tests/load/k6-load-test.js
```

### Grafana Dashboard:

Import the k6 dashboard from:
https://k6.io/docs/results-output/real-time/#grafana-dashboard

## Results Analysis

### Generate HTML Report:

```bash
k6 run --out json=test.json tests/load/k6-load-test.js
k6-reporter test.json --output k6-report.html
```

### Key Metrics to Track:

1. **Throughput**: Requests per second
2. **Response Time**: p50, p95, p99 percentiles
3. **Error Rate**: Percentage of failed requests
4. **Virtual Users**: Active concurrent users
5. **Resource Usage**: CPU, Memory, Network

## Next Steps

1. **Baseline Establishment**: Run tests and record baseline metrics
2. **Performance Optimization**: Optimize based on test results
3. **Regression Testing**: Run tests before each deployment
4. **Continuous Monitoring**: Set up automated load testing in CI/CD

## Support

For issues or questions:
- Check: `FINAL_PRODUCTION_READINESS_REPORT.md`
- Check: `BubbleLab/docs/deployment/`
- File issue on GitHub

## Checklist

Before running load tests:

- [ ] All services running (`docker-compose up -d`)
- [ ] Database migrations run
- [ ] Environment variables set
- [ ] k6 installed
- [ ] API key available
- [ ] Sufficient disk space for logs
- [ ] Monitoring (Prometheus/Grafana) running

After running load tests:

- [ ] Review error rate
- [ ] Review response times (p50, p95, p99)
- [ ] Check for memory leaks
- [ ] Document bottlenecks
- [ ] Create optimization plan
- [ ] Upload results to documentation

---

**Target Achievement**: 1000 requests/minute ✅

**Current Status**: 🚧 Baselines not yet established

**Last Updated**: 2026-01-18
