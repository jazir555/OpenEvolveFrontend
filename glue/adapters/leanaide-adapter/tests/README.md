# LeanAide Adapter - Contract Tests

## Overview

These contract tests validate the API contracts between the Glue Layer and LeanAide server. Following the **Federation Constitution's** "Proof of Work" doctrine, these tests ensure that:

1. **FAIL FAST**: If contracts are violated, the adapter refuses to start
2. **RUNTIME TRUTH**: Tests validate actual API behavior, not documentation
3. **ZERO TRUST**: Every field and response type is explicitly validated

## Purpose

Prevent LeanAide API changes from breaking the integration silently by validating:
- Proof Verification API Contract (POST /verify)
- Lean Compilation Contract (Lake build)
- Package Manager Contract (Lake commands)
- Canonical schema compliance

## Test Requirements

The tests validate the following contracts:

### 1. Proof Verification API Contract

- **POST /verify** returns `{ verified: boolean, tactics_used: string[] }`
- Handles valid/invalid Lean 4 syntax
- Error responses include `correlation_id`
- Timeout scenarios are handled gracefully

### 2. Lean Compilation Contract

- Lake build returns expected status
- Mathlib imports resolve correctly
- Compilation errors are properly formatted with line/column information

### 3. Package Manager Contract

- Lake commands return expected structure
- Package metadata includes required fields (name, version, status, dependencies)

### 4. Edge Cases

- Malformed proofs
- Timeout handling
- Special characters and Unicode
- Invalid correlation IDs

## Installation

```bash
# Install dependencies
npm install --save-dev jest @types/jest ts-jest

# Or using yarn
yarn add --dev jest @types/jest ts-jest
```

## Configuration

Environment variables (following **Law of Configuration Explicitness**):

```bash
# LeanAide API URL (REQUIRED - no magic defaults)
export LEANAIDE_API_URL=http://localhost:7654

# Verification timeout in milliseconds (optional, default: 30000)
export LEANAIDE_TIMEOUT_MS=30000
```

**CRITICAL**: The adapter will crash immediately if `LEANAIDE_API_URL` is not configured.

## Usage

### Running Tests Manually

```bash
# Run all contract tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage

# Run specific test suite
npm test -- --testNamePattern="Proof Verification"
```

### Running Tests Programmatically

The tests can be executed during adapter startup to validate contracts before the adapter starts:

```typescript
import { runContractTests } from './tests/contract.test';

async function startAdapter() {
  // Run contract tests
  const testsPassed = await runContractTests();

  if (!testsPassed) {
    console.error('CRITICAL: Contract tests failed. Adapter cannot start.');
    process.exit(1);
  }

  // Start the adapter
  console.log('Contract tests passed. Starting adapter...');
  // ... adapter startup code
}
```

### Container Startup Integration

Add to your Dockerfile or container startup script:

```bash
#!/bin/bash
set -e

echo "Running LeanAide contract tests..."
npm test

if [ $? -ne 0 ]; then
  echo "CRITICAL: Contract tests failed. Container will not start."
  exit 1
fi

echo "Contract tests passed. Starting adapter..."
node dist/index.js
```

## Test Structure

```
tests/
├── contract.test.ts      # Main contract test suite
├── README.md             # This file
└── jest.config.js        # Jest configuration
```

## Test Categories

### 1. Proof Verification API Contract Tests

Validates that the LeanAide proof verification API returns expected responses:

```typescript
describe('Proof Verification API Contract', () => {
  it('should return verified: boolean in response');
  it('should return tactics_used: string[] in successful verification');
  it('should handle valid Lean 4 syntax');
  it('should handle invalid Lean 4 syntax with appropriate error');
  it('should include correlation_id in error responses');
  it('should handle timeout scenarios gracefully');
});
```

### 2. Lean Compilation Contract Tests

Validates Lake build and Lean compilation behavior:

```typescript
describe('Lean Compilation Contract', () => {
  it('should return compiled: boolean status');
  it('should resolve Mathlib imports correctly');
  it('should return properly formatted compilation errors');
  it('should include metadata with compilation results');
  it('should handle malformed proofs with specific error messages');
});
```

### 3. Package Manager Contract Tests

Validates Lake package manager commands:

```typescript
describe('Package Manager Contract', () => {
  it('should handle Lake build commands with expected structure');
  it('should include required fields in package metadata');
  it('should handle Lake command failures gracefully');
});
```

### 4. Canonical Schema Validation Tests

Validates that all data conforms to the canonical schema:

```typescript
describe('Canonical Schema Validation', () => {
  it('should validate example proof verification request');
  it('should validate example proof verification response');
  it('should validate example compilation request');
  it('should validate example compilation response');
  it('should reject invalid proof verification request');
  it('should reject response without required fields');
});
```

### 5. Edge Cases and Malformed Inputs

Tests behavior with edge cases:

```typescript
describe('Edge Cases and Malformed Inputs', () => {
  it('should handle empty proof code');
  it('should handle extremely long timeout values');
  it('should reject timeout exceeding maximum');
  it('should handle special characters in theorem statements');
  it('should handle correlation_id format validation');
});
```

### 6. Integration Contract Tests

Validates end-to-end contract compliance:

```typescript
describe('Integration Contract Tests', () => {
  it('should maintain UTC timestamps (Law of UTC)');
  it('should preserve correlation_id throughout request lifecycle');
  it('should handle structured logging format');
});
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: LeanAide Contract Tests

on: [push, pull_request]

jobs:
  contract-tests:
    runs-on: ubuntu-latest

    services:
      leanaide:
        image: leanaide:latest
        ports:
          - 7654:7654
        options: >-
          --health-cmd "curl -f http://localhost:7654 || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run contract tests
        run: npm test
        env:
          LEANAIDE_API_URL: http://localhost:7654
          LEANAIDE_TIMEOUT_MS: 30000
```

## Fail-Fast Behavior

Following the **Federation Constitution**, these tests implement fail-fast behavior:

1. **Missing Configuration**: If `LEANAIDE_API_URL` is not set, tests fail immediately
2. **Contract Violation**: If any contract test fails, adapter refuses to start
3. **Schema Validation**: If response doesn't match canonical schema, test fails

## Observability

Tests use structured logging (JSON Lines format) with required context:

```json
{
  "timestamp": "2025-02-03T12:34:56.789Z",
  "level": "info",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Contract validated: verified field exists and is boolean"
}
```

## Troubleshooting

### Tests Fail to Start

**Problem**: Tests fail with "LEANAIDE_API_URL is not configured"

**Solution**: Set the environment variable
```bash
export LEANAIDE_API_URL=http://localhost:7654
```

### Timeout Errors

**Problem**: Tests timeout waiting for LeanAide server

**Solution**: Ensure LeanAide server is running and accessible
```bash
# Check if server is running
curl http://localhost:7654

# Start LeanAide server if needed
lake exe leanaide_process
```

### Schema Validation Errors

**Problem**: "Schema validation failed" errors

**Solution**: Check that the canonical schema is up to date with the LeanAide API

## Maintenance

### When LeanAide API Updates

1. Run contract tests to identify breaking changes
2. Update canonical schema if needed
3. Update this test file to reflect new contracts
4. Run tests to validate compliance

### Adding New Tests

Follow this pattern:

```typescript
describe('New Contract Feature', () => {
  it('should validate required field', async () => {
    const request = { /* test data */ };
    const response = await apiClient.post('/', request);

    // CRITICAL: Assert required field exists
    expect(response).toHaveProperty('required_field');

    logger.log('info', 'Contract validated: required field exists');
  });
});
```

## References

- **Federation Constitution**: `/CLAUDE.md`
- **Canonical Schema**: `/glue/schemas/leanaide-canonical.ts`
- **Adapter Implementation**: `/glue/adapters/leanaide-adapter/src/`
- **LeanAide Documentation**: https://github.com/ms-jpq/lean-aide

## License

These tests are part of the OpenEvolve Federation Glue Layer.
