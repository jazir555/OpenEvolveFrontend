# LeanAide Contract Tests - Quick Start Guide

## Quick Start

### 1. Install Dependencies

```bash
cd glue/adapters/leanaide-adapter
npm install
```

### 2. Configure Environment

```bash
# Required: LeanAide API URL
export LEANAIDE_API_URL=http://localhost:7654

# Optional: Timeout (default: 30000ms)
export LEANAIDE_TIMEOUT_MS=30000
```

### 3. Run Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode (for development)
npm run test:watch

# Run contract tests via shell script (with server health check)
./tests/run-contract-tests.sh
```

## Container Startup Integration

### Option 1: Shell Script (Recommended)

Add to your Dockerfile or entrypoint:

```dockerfile
# Run contract tests before starting
COPY glue/adapters/leanaide-adapter /app/adapter
WORKDIR /app/adapter
RUN npm install

# Make test script executable
RUN chmod +x tests/run-contract-tests.sh

# Run tests, then start adapter
CMD ["tests/run-contract-tests.sh", "&&", "node", "dist/index.js"]
```

Or in docker-compose.yml:

```yaml
services:
  leanaide-adapter:
    build: ./glue/adapters/leanaide-adapter
    environment:
      - LEANAIDE_API_URL=http://leanaide:7654
      - LEANAIDE_TIMEOUT_MS=30000
      - FAIL_FAST=true
    depends_on:
      leanaide:
        condition: service_healthy
```

### Option 2: Direct NPM Script

```dockerfile
# In your Dockerfile
RUN npm install && npm test
CMD ["node", "dist/index.js"]
```

### Option 3: Programmatic Integration

In your adapter startup code:

```typescript
import { runContractTests } from './tests/contract.test';

async function startAdapter() {
  // 1. Validate contracts
  const testsPassed = await runContractTests();

  if (!testsPassed) {
    console.error('CRITICAL: Contract tests failed.');
    console.error('The adapter cannot start due to contract violations.');
    process.exit(1);
  }

  // 2. Start adapter
  console.log('✓ Contracts validated. Starting adapter...');
  // ... your adapter startup code
}
```

## What Gets Tested

### ✓ Proof Verification API
- POST /verify returns `{ verified: boolean, tactics_used: string[] }`
- Valid Lean 4 syntax is accepted
- Invalid syntax produces errors with correlation_id
- Timeouts are handled gracefully

### ✓ Lean Compilation
- Lake build returns expected status
- Mathlib imports resolve correctly
- Compilation errors include line/column info

### ✓ Package Manager
- Lake commands return proper structure
- Package metadata includes required fields

### ✓ Canonical Schema
- All requests/responses match canonical schema
- Invalid data is rejected
- Edge cases are handled

### ✓ Edge Cases
- Empty/malformed proofs
- Timeout handling
- Unicode characters
- Invalid correlation IDs
- Maximum timeout limits

## Troubleshooting

### "LEANAIDE_API_URL is not configured"

**Problem**: Missing required environment variable

**Solution**:
```bash
export LEANAIDE_API_URL=http://localhost:7654
```

### "LeanAide server is not reachable"

**Problem**: Tests can't connect to LeanAide server

**Solution**:
```bash
# Start LeanAide server
cd /path/to/LeanAide
lake exe leanaide_process

# Or check if Docker container is running
docker ps | grep leanaide
```

### "Contract tests failed"

**Problem**: One or more contract tests failed

**Solution**:
1. Check test output to see which tests failed
2. Review the failing contract in `contract.test.ts`
3. Update canonical schema or adapter implementation
4. Verify LeanAide API hasn't changed

### "Timeout waiting for server"

**Problem**: Server took too long to start

**Solution**:
```bash
# Increase timeout
export LEANAIDE_TIMEOUT_MS=60000

# Or wait for server to be healthy first
curl http://localhost:7654
npm test
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run contract tests
  run: |
    export LEANAIDE_API_URL=http://localhost:7654
    npm test
  env:
    LEANAIDE_TIMEOUT_MS: 30000
```

### GitLab CI

```yaml
test:contracts:
  script:
    - export LEANAIDE_API_URL=http://leanaide:7654
    - npm test
  services:
    - name: leanaide:latest
      alias: leanaide
```

### Jenkins Pipeline

```groovy
stage('Contract Tests') {
  environment {
    LEANAIDE_API_URL = 'http://localhost:7654'
  }
  steps {
    sh 'npm test'
  }
}
```

## Development Workflow

### 1. Write Tests First

```typescript
describe('New Feature', () => {
  it('should validate new contract', async () => {
    const response = await apiClient.post('/new-endpoint', data);
    expect(response).toHaveProperty('new_field');
  });
});
```

### 2. Implement Feature

### 3. Validate Contracts

```bash
npm test
```

### 4. Commit and Push

Tests will run automatically in CI/CD and reject commits with contract violations.

## Next Steps

- Read full documentation: `tests/README.md`
- Review canonical schema: `../../schemas/leanaide-canonical.ts`
- Check probe scripts: `../probes/`
- Review Federation Constitution: `/CLAUDE.md`
