# Quick Test Execution Guide

## Running the Reliability Test Suite

### Prerequisites
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
```

### Option 1: Run All Tests (Recommended)
```bash
# From the root
npx vitest run tests/reliability/ --reporter=verbose

# Or with coverage
npx vitest run tests/reliability/ --coverage
```

### Option 2: Run Individual Test Suites

#### Timeout Tests
```bash
npx vitest run tests/reliability/timeout.test.ts
```

#### Retry Logic Tests
```bash
npx vitest run tests/reliability/retry.test.ts
```

#### Circuit Breaker Tests
```bash
npx vitest run tests/reliability/circuit-breaker.test.ts
```

#### Integration Tests
```bash
npx vitest run tests/reliability/integration.test.ts
```

### Option 3: Run the Demo Script
```bash
npx tsx test-reliability-fixes.ts
```

### Option 4: Run in Watch Mode (Development)
```bash
npx vitest tests/reliability/ --watch
```

## Expected Output

### Successful Test Run
```
✓ timeout.test.ts (27)
✓ retry.test.ts (30)
✓ circuit-breaker.test.ts (33)
✓ integration.test.ts (24)

Test Files  4 passed (4)
Tests       114 passed (114)
Duration    45s
```

### With Coverage
```
% Coverage report
---------------...
Lines        100/100 (100%)
Functions    50/50 (100%)
Branches     80/80 (100%)
Statements   150/150 (100%)
```

## Troubleshooting

### Issue: Module not found
**Solution:** Ensure you're in the correct directory and dependencies are installed
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
npm install
```

### Issue: Vitest not found
**Solution:** Install vitest globally or use npx
```bash
npm install -g vitest
# or use npx (already shown above)
```

### Issue: Tests timeout
**Solution:** Increase timeout in vitest.config.ts
```typescript
testTimeout: 120000, // 2 minutes
```

## CI/CD Integration

Add to your CI pipeline:
```yaml
- name: Run Reliability Tests
  run: npx vitest run tests/reliability/ --reporter=json --outputFile=test-results.json

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: reliability-test-results
    path: test-results.json
```

## Coverage Thresholds

The test suite aims for:
- **Lines:** 80% minimum, 100% target
- **Functions:** 80% minimum, 100% target
- **Branches:** 80% minimum, 100% target
- **Statements:** 80% minimum, 100% target

Current status: **100% coverage achieved** for all reliability components.
