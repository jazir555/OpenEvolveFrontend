# Tests Directory

## Purpose

End-to-end (E2E) contract tests that verify the federation's integration contracts.

## Test Philosophy

**"The Proof of Work"**

Tests serve two critical functions:

1. **Discovery (Probes)**: Verify APIs actually work before writing adapters
2. **Protection (Contracts)**: Prevent upstream changes from breaking the federation

## Test Categories

### 1. Probe Tests (`glue/adapters/{project}/probes/`)

**Run BEFORE writing adapter code.**

Purpose: Verify the API exists and behaves as documented.

Example: `probes/check_api.sh`
```bash
#!/usr/bin/env bash
# This script proves the Z3 API exists and returns the expected fields
curl -f http://z3-core:8000/api/check || exit 1
```

**Rule**: If the probe fails, the feature does not exist. Do not write code.

### 2. Contract Tests (`glue/adapters/{project}/tests/`)

**Run on container startup.**

Purpose: Ensure the API still returns the fields we depend on.

Example: `tests/contract.test.ts`
```typescript
// This test ensures Z3 returns the exact fields we need
test('Z3 check API returns proof_status field', async () => {
  const response = await z3.check(query);
  expect(response).toHaveProperty('proof_status');
});
```

**Rule**: If contract test fails, the adapter MUST refuse to start to prevent data corruption.

### 3. E2E Integration Tests (`tests/`)

**Run in CI/CD pipeline.**

Purpose: Verify end-to-end workflows across multiple adapters.

Examples:
- User creates proof request → Z3 processes it → Lean4 verifies it
- Error scenarios (timeouts, bad data, service failures)
- Circuit breaker activation

## Test Execution Order

1. **Unit Tests**: Run on every commit (fast feedback)
2. **Contract Tests**: Run on adapter startup (prevents broken deployments)
3. **E2E Tests**: Run in CI before merge (catches integration issues)

## Test Data Management

- **Idempotent**: Tests should be safe to run 100 times
- **Isolated**: Each test should clean up after itself
- **Realistic**: Use production-like data, not toy examples
- **No Mocking**: Tests should hit real APIs (following Law #2: Runtime Truth)

## Coverage Requirements

- **Critical Path**: 100% coverage (happy path + all error scenarios)
- **Adapters**: Contract tests for every API endpoint used
- **Schemas**: Validation tests for all canonical models
- **Failure Modes**: Test timeout, retry, circuit breaker, DLQ

## Running Tests

```bash
# All tests
npm test

# Specific adapter contract tests
npm test -- adapters/z3-adapter/tests/contract.test.ts

# E2E tests (requires full stack running)
npm test -- tests/e2e/
```

## CI/CD Integration

Contract tests must block deployment:
```yaml
# .github/workflows/test.yml
- name: Run contract tests
  run: npm test -- --contract
  # If this fails, DO NOT deploy
```

## Test Philosophy Summary

> "We don't trust documentation. We trust execution."
> "If you can't get a 200 OK from the shell, you can't write the code."
> "If the contract is violated, the adapter refuses to start."
