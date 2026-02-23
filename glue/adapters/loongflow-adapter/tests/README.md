# LoongFlow Adapter Contract Tests

This directory contains comprehensive contract tests that validate the API contracts between the LoongFlow adapter and the LoongFlow core system.

## Purpose

**Phase 2: The Contract (Defense)**

These tests protect the adapter against breaking changes from the LoongFlow core API. If any contract is violated, the adapter refuses to start, preventing data corruption.

### Following Federation Constitution

- **Law of Runtime Truth**: Tests execute against real LoongFlow API (no mocking in critical paths)
- **Law of Configuration Explicitness**: API URL and timeouts from environment variables
- **Law of UTC**: All timestamps validated as UTC ISO-8601 format
- **Law of Idempotency**: Validates operations can be safely retried
- **Law of Air Gap**: Verifies no imports from core-projects

## Test Files

### Core Contract Tests (`contract.test.ts`)

Comprehensive test suite covering all API contracts:

- **Test Suite 1**: Health and Connectivity
- **Test Suite 2**: Problem Submission Contracts
- **Test Suite 3**: Solution Data Structure Contracts
- **Test Suite 4**: Execution State Contracts
- **Test Suite 5**: Evolutionary Database Contracts
- **Test Suite 6**: Checkpoint Contracts
- **Test Suite 7**: Error Handling Contracts
- **Test Suite 8**: Response Format Contracts
- **Test Suite 9**: UTC Compliance
- **Test Suite 10**: Configuration Explicitness
- **Test Suite 11**: Idempotency Requirements
- **Test Suite 12**: Air Gap Compliance
- **Test Suite 13**: Structured Logging

### Test Runner (`contract-runner.ts`)

Standalone script for running contract tests and generating detailed reports:

- Runs all test suites
- Generates JSON report
- Provides detailed error messages
- Returns appropriate exit codes for CI/CD

### Test Fixtures (`fixtures/test-data.ts`)

Comprehensive test data for all scenarios:

- Valid requests and responses
- Invalid data for negative testing
- Edge cases and boundary conditions
- Utility functions for validation

### Contract Documentation (`CONTRACTS.md`)

Complete documentation of all API contracts:

- Endpoint specifications
- Request/response formats
- Validation rules
- Error handling
- Update procedures

## Running Tests

### Quick Start

```bash
# Set required environment variables
export LOONGFLOW_API_URL=http://localhost:8000
export LOONGFLOW_TIMEOUT_MS=30000

# Run all contract tests
npm run test:contract

# Run with Jest
jest tests/contract.test.ts

# Run standalone test runner
npm run test:contract:runner
```

### Test Options

```bash
# Run only fixture tests (offline, no API calls)
SKIP_CONTRACT_TESTS=true npm run test:contract

# Run with verbose output
VERBOSE=true npm run test:contract:runner

# Output JSON report
JSON_OUTPUT=true npm run test:contract:runner

# Run quick health check (for Docker)
npm run test:contract:quick
```

### CI/CD Integration

```bash
# Run contract tests in CI pipeline
npm run test:contract:ci
```

## Environment Variables

### Required

- `LOONGFLOW_API_URL` - Base URL of LoongFlow sidecar API
- `LOONGFLOW_TIMEOUT_MS` - Request timeout in milliseconds

### Optional

- `SKIP_CONTRACT_TESTS` - Skip integration tests if 'true' (fixture tests only)
- `VERBOSE` - Enable verbose output if 'true'
- `JSON_OUTPUT` - Output JSON report if 'true'

## Test Results

### Success

```
✅ All contract tests passed
✅ Environment Configuration: PASSED
✅ Fixture Contracts: PASSED
✅ API Connectivity: PASSED

Total Passed: 3
Duration: 245ms
```

### Failure

```
❌ Contract tests failed

Failed Tests:
- Health Check: API connection timeout
- Solution Contract: Invalid score range > 1.0

Action Required:
1. Verify LoongFlow API is accessible
2. Check environment variables are set correctly
3. Review fixture data matches API responses
4. Update contracts if API has changed
```

## Contract Violation Response

When a contract test fails:

1. **Stop the adapter** - Do not start with violated contracts
2. **Log the violation** - Full error details logged
3. **Alert the team** - Monitoring system notified
4. **Investigate** - Determine if API changed or test is wrong
5. **Resolve** - Update adapter or rollback API change
6. **Document** - Create ADR explaining the change

## Updating Contracts

When LoongFlow core API changes:

1. Update the contract schema in `CONTRACTS.md`
2. Update Zod validation in `contract.test.ts`
3. Update test fixtures in `fixtures/test-data.ts`
4. Run contract tests to verify
5. Update adapter code if needed
6. Create ADR documenting the change
7. Get approval before deploying

## Example: Adding New Field

```typescript
// 1. Update contract schema
const PESAgentStateContract = z.object({
  // ... existing fields
  new_field: z.string().optional(),
});

// 2. Update fixture
export const RUNNING_AGENT_STATE = {
  // ... existing fields
  new_field: 'new_value',
};

// 3. Update documentation in CONTRACTS.md
// Add field to agent state section
```

## Test Coverage

Current coverage:

- ✅ Health check endpoint
- ✅ Problem submission
- ✅ Agent state queries
- ✅ Agent interruption
- ✅ Execution results
- ✅ Solution data structures
- ✅ Database operations
- ✅ Checkpoint operations
- ✅ Error responses
- ✅ Configuration validation
- ✅ UTC timestamp compliance
- ✅ Idempotency requirements
- ✅ Air gap compliance

## Metrics

### Success Criteria

- All contracts pass before adapter starts
- Tests complete in under 30 seconds
- Zero false positives/negatives
- 100% contract test pass rate

### Monitoring

- Contract test pass rate (should be 100%)
- Test execution time (alert if > 30s)
- Contract violation frequency (should be zero)
- API availability percentage

## Troubleshooting

### Tests Fail with Connection Error

**Problem**: Cannot connect to LoongFlow API

**Solution**:
```bash
# Check API is running
curl http://localhost:8000/health

# Verify environment variables
echo $LOONGFLOW_API_URL
echo $LOONGFLOW_TIMEOUT_MS

# Check network connectivity
docker ps | grep loongflow
```

### Tests Fail with Validation Error

**Problem**: API response doesn't match expected schema

**Solution**:
```bash
# Run with verbose output to see actual vs expected
VERBOSE=true npm run test:contract

# Check if API changed
curl http://localhost:8000/health | jq .

# Update fixtures if API changed
# Edit fixtures/test-data.ts
```

### Tests Time Out

**Problem**: Tests take too long to complete

**Solution**:
```bash
# Increase timeout
export LOONGFLOW_TIMEOUT_MS=60000

# Skip slow integration tests
SKIP_CONTRACT_TESTS=true npm run test:contract

# Run quick check only
npm run test:contract:quick
```

## Additional Resources

- **Adapter Code**: `../src/adapter.ts`
- **Canonical Schemas**: `../../../schemas/loongflow-canonical.ts`
- **Contract Documentation**: `CONTRACTS.md`
- **Federation Constitution**: `../../../../CLAUDE.md`

---

**Maintained By**: OpenEvolve Federation
**Last Updated**: 2026-02-22
**Status**: Active
