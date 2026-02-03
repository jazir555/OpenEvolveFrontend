# Z3 Adapter Contract Tests

## Purpose

Following **CLAUDE.md Section 4: The Proof of Work (The Vibe Check)**

These contract tests validate the API contract between the Z3 Adapter and Z3 Core service. **If these tests fail, the adapter MUST refuse to start** to prevent data corruption and system instability.

### The Contract Defense Strategy

1. **PROBE FIRST** - Before writing adapter code, we probe the live Z3 API
2. **DEFINE CONTRACT** - We codify the expected API behavior in tests
3. **FAIL FAST** - Any contract violation triggers immediate shutdown
4. **ZERO TRUST** - We don't trust documentation, we trust execution

## Test Coverage

### 1. Z3 API Contract Tests

#### Health Endpoint (`GET /health`)
- Validates response structure for healthy status
- Validates response structure for degraded status
- Ensures transformation to `CanonicalService` schema
- Checks version string presence and format

#### Solve Endpoint (`POST /solve`)
- Validates `Z3SolveResponseSchema` conformance
- Tests satisfiable responses (`sat`) with model data
- Tests unsatisfiable responses (`unsat`) without model
- Validates timing information for performance tracking
- Checks statistics structure and types

#### Optimize Endpoint (`POST /optimize`)
- Validates `Z3OptimizeResponseSchema` conformance
- Tests optimal status with objective values
- Validates objective value types and structure

#### Simplify Endpoint (`POST /simplify`)
- Validates `Z3SimplifyResponseSchema` conformance
- Tests result field contains valid SMTLIB2 expression

#### Tactic Endpoint (`POST /tactic`)
- Validates `Z3TacticResponseSchema` conformance
- Tests status field and optional goals array

#### Fixedpoint Endpoint (`POST /fixedpoint`)
- Validates `Z3FixedpointResponseSchema` conformance
- Tests result field and optional answer

### 2. Correlation ID Contract Tests

- Ensures responses can accommodate correlation tracking
- Validates transformation to `CanonicalLogEntry` schema
- Tests correlation ID propagation through the system

### 3. Error Response Contract Tests

- Validates error responses conform to `CanonicalErrorSchema`
- Tests error fields in solve responses
- Ensures proper error classification (validation, internal, etc.)

### 4. Database Contract Tests

#### Knowledge Queries
- Validates entity structure (id, type, attributes, relations)
- Tests query results return arrays of entities
- Handles empty result sets gracefully

#### ORM Models
- Ensures models have required fields (id, createdAt, updatedAt)
- Validates UTC timestamp convention (ISO 8601 with Z suffix)

### 5. Knowledge Extraction Contract Tests

#### Graph Structure
- Validates nodes and edges structure
- Ensures node IDs are unique
- Validates edges reference valid node IDs

#### Edge Cases
- Handles empty results (no nodes/edges)
- Handles disconnected nodes (no edges)
- Handles complex nested attributes

## Installation

```bash
# Install dependencies
npm install

# or with pnpm
pnpm install

# or with yarn
yarn install
```

## Usage

### Run All Tests

```bash
npm test
```

### Run with Verbose Output

```bash
npm run test:verbose
```

### Run in Watch Mode (Development)

```bash
npm run test:watch
```

### Run with Coverage Report

```bash
npm run test:coverage
```

### Run Contract Tests Only

```bash
npm run test:contract
```

## Integration with Adapter Startup

**CRITICAL:** These tests must run before the adapter starts serving requests. Example integration:

```typescript
// adapter/src/index.ts
import { execSync } from 'child_process';
import { logger } from './lib/logger';

async function validateContractBeforeStartup(): Promise<void> {
  try {
    logger.info({ msg: 'Running Z3 contract validation...' });

    // Run contract tests
    execSync('npm run test:contract', {
      cwd: process.cwd(),
      stdio: 'inherit',
      timeout: 30000, // 30 second timeout
    });

    logger.info({ msg: '✅ Z3 contract validation passed' });
  } catch (error) {
    logger.error({
      msg: '❌ Z3 contract validation FAILED',
      error: error instanceof Error ? error.message : String(error),
    });

    // FAIL FAST - Do not start adapter if contract is violated
    process.exit(1);
  }
}

// Main startup
async function main() {
  await validateContractBeforeStartup();
  // ... continue with adapter initialization
}

main().catch(err => {
  logger.error({ msg: 'Adapter startup failed', error: err.message });
  process.exit(1);
});
```

## Environment Variables

The tests use the following environment variables (if needed):

```bash
# Optional: Override default timeout (default: 10000ms)
JEST_TIMEOUT=30000

# Optional: Enable debug logging
DEBUG=z3:contract:tests
```

## Mocking Strategy

**IMPORTANT:** These tests use MOCK DATA and do NOT require a running Z3 instance. This follows the principle of:

- Fast test execution
- Deterministic results
- No external dependencies
- Can run in CI/CD pipelines without infrastructure

## Fail-Fast Behavior

If any contract test fails:

1. **DO NOT** start the adapter
2. **DO NOT** attempt to handle the error gracefully
3. **DO** log the specific contract violation
4. **DO** exit with non-zero status code

### Why Fail Fast?

- Prevents data corruption from malformed responses
- Catches API changes immediately (not in production)
- Forces immediate resolution of contract violations
- Maintains system integrity

## Contract Violation Resolution

If tests fail:

1. **Check Z3 Core Version** - Did Z3 API change?
2. **Review Release Notes** - Are there breaking changes?
3. **Update Contract Tests** - If API change is intentional
4. **Update Adapter Code** - Adapt to new API structure
5. **Re-run Tests** - Ensure all tests pass
6. **Update ADR.md** - Document the change and rationale

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/z3-adapter-tests.yml
name: Z3 Adapter Contract Tests

on: [push, pull_request]

jobs:
  contract-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm install
        working-directory: ./glue/adapters/z3-adapter
      - name: Run contract tests
        run: npm run test:contract
        working-directory: ./glue/adapters/z3-adapter
```

## Contract Test Checklist

When adding new Z3 API endpoints or features:

- [ ] Add mock response data
- [ ] Create schema validation test
- [ ] Test successful responses
- [ ] Test error responses
- [ ] Test edge cases (empty, null, invalid)
- [ ] Validate correlation tracking
- [ ] Test transformation to canonical schemas
- [ ] Update this README
- [ ] Run all tests locally
- [ ] Ensure CI/CD passes

## Dependencies

- **jest** - Test framework
- **@jest/globals** - Jest global types
- **ts-jest** - TypeScript preprocessor
- **zod** - Schema validation (canonical models)
- **typescript** - TypeScript compiler

## Related Documentation

- `../../../CLAUDE.md` - Federation Constitution
- `../ADR.md` - Architecture Decision Records
- `../../../schemas/` - Canonical data models
- `../probes/check_api.sh` - API probe scripts

## Troubleshooting

### Tests Time Out

Increase timeout in `package.json`:

```json
"jest": {
  "testTimeout": 30000
}
```

### Import Path Errors

Ensure `baseUrl` and `paths` are correctly configured in `tsconfig.json`.

### Schema Validation Errors

Verify canonical schema import paths match your project structure.

### TypeScript Errors

Ensure `@types/node` and `@types/jest` are installed.

## License

MIT

## Authors

OpenEvolve Federation Architecture Team
