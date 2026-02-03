# Z3 Adapter Contract Tests - Integration Examples

## Quick Start

### 1. Install and Run Tests

```bash
# Navigate to adapter directory
cd glue/adapters/z3-adapter

# Install dependencies
npm install

# Run contract tests
npm test

# Or use make
make test
```

### 2. Integrate with Adapter Startup

Create a startup validation script:

```typescript
// glue/adapters/z3-adapter/src/startup.ts
import { execSync } from 'child_process';
import path from 'path';

interface ContractValidationResult {
  success: boolean;
  output: string;
  error?: string;
}

export async function validateContract(): Promise<ContractValidationResult> {
  try {
    const adapterDir = path.dirname(__dirname);
    const testCommand = 'npm run test:contract';

    const output = execSync(testCommand, {
      cwd: adapterDir,
      encoding: 'utf-8',
      stdio: 'pipe',
      timeout: 30000,
    });

    return {
      success: true,
      output,
    };
  } catch (error: any) {
    return {
      success: false,
      output: error.stdout || '',
      error: error.stderr || error.message,
    };
  }
}

// Usage in main adapter
export async function startupWithValidation() {
  console.log('🔒 Validating Z3 API contracts...');

  const result = await validateContract();

  if (!result.success) {
    console.error('❌ CONTRACT VALIDATION FAILED');
    console.error(result.error);
    console.error('');
    console.error('The adapter CANNOT start until contracts are validated.');
    console.error('This prevents data corruption from API mismatches.');
    process.exit(1);
  }

  console.log('✅ Contract validation passed');
  console.log('🚀 Starting Z3 Adapter...');

  // Continue with normal startup
  // await startAdapter();
}
```

### 3. Docker Integration

Add to your Dockerfile:

```dockerfile
# glue/adapters/z3-adapter/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm ci --only=development

# Copy source and tests
COPY src/ ./src/
COPY tests/ ./tests/

# Run contract tests before starting
RUN npm run test:contract || \
    (echo "❌ Contract tests failed. Build aborted." && exit 1)

# Build adapter
RUN npm run build

# Start with contract validation
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
```

Docker entrypoint:

```bash
#!/bin/sh
# docker-entrypoint.sh

set -e

echo "🔒 Running contract validation in Docker..."
npm run test:contract

echo "✅ Contracts validated"
echo "🚀 Starting adapter..."

exec "$@"
```

### 4. Kubernetes Init Container

Use as init container in Kubernetes:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: z3-adapter
spec:
  containers:
    - name: z3-adapter
      image: openevolve/z3-adapter:latest
      # ... adapter config
  initContainers:
    - name: contract-validation
      image: openevolve/z3-adapter:latest
      command: ['npm', 'run', 'test:contract']
```

### 5. GitHub Actions Workflow

```yaml
# .github/workflows/z3-adapter-contract.yml
name: Z3 Adapter Contract Tests

on:
  push:
    paths:
      - 'glue/adapters/z3-adapter/**'
  pull_request:
    paths:
      - 'glue/adapters/z3-adapter/**'

jobs:
  contract-tests:
    name: Validate Contracts
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: glue/adapters/z3-adapter/package-lock.json

      - name: Install dependencies
        working-directory: glue/adapters/z3-adapter
        run: npm ci

      - name: Run contract tests
        working-directory: glue/adapters/z3-adapter
        run: npm run test:contract

      - name: Upload coverage
        if: always()
        working-directory: glue/adapters/z3-adapter
        run: npm run test:coverage
        # Upload to coverage service...

      - name: Fail if contracts violated
        if: failure()
        run: |
          echo "::error::Z3 contract tests failed!"
          echo "::error::The adapter will NOT start until contracts are fixed."
          exit 1
```

## Advanced Usage

### Custom Test Suites

Add specific test suites:

```typescript
// tests/custom/specific-contract.test.ts
import { test, expect } from '@jest/globals';
import { mockSolveResponse } from '../contract.test';

describe('Custom Contract Validation', () => {
  test('custom business logic constraint', () => {
    // Your custom validation
    expect(mockSolveResponse.timing).toBeLessThan(1000);
  });
});
```

### Contract Snapshots

Use Jest snapshots for API response regression testing:

```typescript
test('Z3 solve response matches snapshot', () => {
  const response = {
    result: 'sat',
    model: { x: 5, y: 10 },
    timing: 45,
  };

  expect(response).toMatchSnapshot();
});
```

### Performance Contract Testing

Add performance contracts:

```typescript
describe('Performance Contracts', () => {
  test('solve response time < 100ms', () => {
    const maxTime = 100;
    expect(mockSolveResponse.timing).toBeLessThan(maxTime);
  });

  test('memory usage within limits', () => {
    // Add memory profiling
    const memoryUsage = process.memoryUsage();
    expect(memoryUsage.heapUsed).toBeLessThan(100 * 1024 * 1024); // 100MB
  });
});
```

### Dynamic Contract Loading

Load contracts from external source:

```typescript
// tests/dynamic-contracts.test.ts
import { z } from 'zod';

async function loadContractFromAPI(): Promise<any> {
  const response = await fetch('https://api.z3.dev/contract');
  return response.json();
}

test('dynamic contract validation', async () => {
  const contract = await loadContractFromAPI();
  const schema = z.object({
    version: z.string(),
    endpoints: z.array(z.object({
      path: z.string(),
      method: z.string(),
      response: z.any(),
    })),
  });

  const result = schema.safeParse(contract);
  expect(result.success).toBe(true);
});
```

## Troubleshooting Examples

### Scenario: API Change Detected

```bash
# Tests fail with:
# Error: Property 'result' is missing in response

# Solution:
# 1. Check Z3 release notes
# 2. Update mock data in contract.test.ts
# 3. Update schema if change is permanent
# 4. Re-run tests
```

### Scenario: Timeout in CI

```bash
# Tests timeout in GitHub Actions

# Solution: Increase timeout in .github/workflows
- name: Run contract tests
  env:
    JEST_TIMEOUT: 60000
  run: npm run test:contract
```

### Scenario: Import Path Errors

```bash
# Error: Cannot find module '@/schemas/z3'

# Solution: Update tsconfig.json paths
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/schemas/*": ["../../../BubbleLab/apps/bubblelab-api/src/schemas/*"]
    }
  }
}
```

## Best Practices

1. **Always run tests before committing**
   ```bash
   # Install pre-commit hook
   npm install -D husky
   npx husky install .husky/pre-commit
   ```

2. **Keep tests fast**
   - Use mocks, not real API calls
   - Avoid heavy computations
   - Parallelize when possible

3. **Test edge cases**
   ```typescript
   test('handles null gracefully', () => {
     const result = parseResponse(null);
     expect(result).toBeNull();
   });
   ```

4. **Document violations**
   ```typescript
   test('known issue: legacy API returns snake_case', () => {
     // TODO: Remove when migrating to v2 API
     expect(response).toHaveProperty('result_type');
   });
   ```

5. **Version contracts**
   ```typescript
   const Z3_CONTRACT_VERSION = '1.0.0';

   test(`contract version ${Z3_CONTRACT_VERSION}`, () => {
     expect(mockResponse.contract_version).toBe(Z3_CONTRACT_VERSION);
   });
   ```

## Resources

- Main README: `tests/README.md`
- Project Constitution: `../../../CLAUDE.md`
- Jest Docs: https://jestjs.io/docs/getting-started
- Zod Docs: https://zod.dev/
