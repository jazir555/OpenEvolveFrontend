# BubbleLab Testing Guide

Comprehensive testing infrastructure for all BubbleLab bubbles.

## Test Structure

```
BubbleLab/packages/bubble-core/
├── src/
│   ├── tests/
│   │   ├── setup.ts                    # Global test setup
│   │   ├── test-utils.ts               # Reusable test utilities
│   │   ├── mocks/                      # Mock implementations
│   │   ├── templates/                  # Test templates
│   │   ├── integration/                # Integration tests
│   │   └── security/                   # Security tests
│   ├── bubbles/
│   │   ├── service-bubble/
│   │   │   ├── *.test.ts               # Existing unit tests
│   │   │   └── tests/
│   │   │       └── *.comprehensive.test.ts  # New comprehensive tests
│   │   └── tool-bubble/
│   │       ├── *.test.ts               # Existing unit tests
│   │       └── tests/
│   │           └── *.comprehensive.test.ts  # New comprehensive tests
│   └── vitest.setup.ts                 # Legacy setup (use src/tests/setup.ts)
├── coverage/                           # Coverage reports
└── test-results/                       # Test results
```

## Test Categories

### 1. Unit Tests
Tests for individual bubble functionality:
- **Validation Tests**: Input validation, schema validation
- **Operation Tests**: Successful operations, error handling
- **Security Tests**: SQL injection, XSS, SSRF prevention
- **Resilience Tests**: Circuit breakers, rate limiting, retries

### 2. Integration Tests
Tests for multi-bubble workflows:
- Service → Tool → Service workflows
- Error propagation across bubbles
- Data flow validation
- End-to-end operation tests

### 3. Security Tests
Comprehensive security testing:
- SQL injection prevention
- XSS attack prevention
- SSRF attack prevention
- Path traversal prevention
- Command injection prevention
- Authentication validation
- Rate limiting enforcement

## Running Tests

### Run All Unit Tests
```bash
cd BubbleLab/packages/bubble-core
pnpm test
```

### Run with Coverage
```bash
pnpm test:coverage
```

### Run Integration Tests
```bash
pnpm test:integration
```

### Run Security Tests
```bash
pnpm test:security
```

### Run Specific Test File
```bash
pnpm test postgresql.comprehensive.test.ts
```

### Run Tests in Watch Mode
```bash
pnpm test:watch
```

### Run Tests with Sharding (CI)
```bash
# Shard 1 of 4
pnpm test --shard=1/4

# Shard 2 of 4
pnpm test --shard=2/4

# etc.
```

## Coverage Goals

- **Lines Coverage**: 80%+
- **Branches Coverage**: 75%+
- **Functions Coverage**: 80%+
- **Statements Coverage**: 80%+

## View Coverage Reports

### HTML Report
```bash
pnpm test:coverage
# Open coverage/lcov-report/index.html in browser
```

### Terminal Summary
Coverage summary is printed to terminal after running `pnpm test:coverage`.

## CI/CD Integration

Tests run automatically on:
- Push to main, develop, or feature branches
- Pull requests to main or develop

### CI Workflow Stages
1. **Unit Tests**: Sharded across 4 runners
2. **Integration Tests**: With PostgreSQL and Redis services
3. **Security Tests**: Comprehensive security validation
4. **Coverage Report**: Aggregated coverage with Codecov upload
5. **Lint & Type Check**: ESLint and TypeScript validation
6. **Build Verification**: Ensures build succeeds

## Writing New Tests

### Using Test Templates

1. Copy the appropriate template:
   ```bash
   # Service bubble
   cp src/tests/templates/service-bubble-test.template.ts \
      src/bubbles/service-bubble/tests/your-bubble.comprehensive.test.ts

   # Tool bubble
   cp src/bubbles/tool-bubble/tests/tool-bubble-tests.template.ts \
      src/bubbles/tool-bubble/tests/your-tool.comprehensive.test.ts
   ```

2. Replace placeholders with actual implementation:
   - Replace `${BUBBLE_NAME}` with your bubble class name
   - Replace `${bubble_name}` with your bubble instance name
   - Implement test cases (remove placeholders)

3. Import necessary dependencies:
   ```typescript
   import { YourBubble } from '../your-bubble.js';
   import { CredentialType } from '@bubblelab/shared-schemas';
   import { securityPayloads, createTestContext } from '../../tests/test-utils.js';
   ```

### Test Structure Example

```typescript
describe('YourBubble - Comprehensive Tests', () => {
  let testContext: ReturnType<typeof createTestContext>;

  beforeEach(() => {
    testContext = createTestContext();
    vi.clearAllMocks();
  });

  describe('Unit Tests - Validation', () => {
    it('should validate required inputs', () => {
      expect(() => {
        new YourBubble({ valid: 'inputs' });
      }).not.toThrow();
    });
  });

  describe('Security Tests - SQL Injection', () => {
    it('should block UNION-based injection', () => {
      expect(() => {
        new YourBubble({ input: "1' UNION SELECT * FROM users--" });
      }).toThrow();
    });
  });

  // More test categories...
});
```

## Test Utilities

### Available Test Utilities

```typescript
import {
  createTestCredentials,
  createDatabaseCredentials,
  createApiCredentials,
  createOAuthCredentials,
  wait,
  createMockResponse,
  createMockErrorResponse,
  mockFetch,
  generateTestData,
  expectError,
  createTestContext,
  createMockFactory,
  securityPayloads,
  measurePerformance,
} from './src/tests/test-utils.js';
```

### Using Mocks

```typescript
import { setupMocks, clearMocks } from './src/tests/mocks/index.js';

beforeEach(() => {
  setupMocks();
});

afterEach(() => {
  clearMocks();
});
```

## Test Best Practices

### 1. Test Isolation
- Each test should be independent
- Use `beforeEach` to reset state
- Clean up mocks in `afterEach`

### 2. Descriptive Test Names
```typescript
// Good
it('should block UNION-based SQL injection', () => { ... });

// Bad
it('should work', () => { ... });
```

### 3. Arrange-Act-Assert Pattern
```typescript
it('should validate user input', () => {
  // Arrange
  const invalidInput = '<script>alert("xss")</script>';

  // Act
  const result = bubble.validate(invalidInput);

  // Assert
  expect(result.valid).toBe(false);
});
```

### 4. Test Edge Cases
- Empty inputs
- Null/undefined values
- Boundary values (max/min)
- Concurrent requests
- Error conditions

### 5. Mock External Dependencies
- Use mocks for external APIs
- Mock database connections
- Mock file system operations
- Reset mocks between tests

### 6. Security Testing
- Test all OWASP Top 10 vectors
- Verify input sanitization
- Test authentication/authorization
- Verify rate limiting

## Troubleshooting

### Tests Timing Out
- Increase timeout in test:
  ```typescript
  it('should complete', async () => { ... }, 10000); // 10s timeout
  ```
- Or globally in `vitest.config.ts`:
  ```typescript
  testTimeout: 120000, // 120s
  ```

### Flaky Tests
- Add retry logic in `vitest.config.ts`:
  ```typescript
  retry: 2, // Retry failed tests twice
  ```
- Ensure proper cleanup in `afterEach`

### Coverage Not Meeting Thresholds
- Identify uncovered code:
  ```bash
  pnpm test:coverage
  # Check coverage/lcov-report/index.html
  ```
- Add tests for missing branches
- Consider if threshold is too strict

### Mock Not Working
- Ensure `setupMocks()` is called in `beforeEach`
- Clear mocks in `afterEach`
- Verify mock configuration

## Continuous Improvement

### Regular Test Maintenance
1. Review and update tests when code changes
2. Remove obsolete tests
3. Add tests for new features
4. Monitor test execution time
5. Keep test data in fixtures

### Performance Monitoring
- Track test execution time
- Identify slow tests
- Optimize test setup/teardown
- Use test sharding for parallel execution

### Coverage Goals
- Aim for 80%+ coverage
- Focus on critical paths
- Test error conditions
- Document uncovered code

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Testing Best Practices](https://testingjavascript.com/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [BubbleLab Architecture](../ARCHITECTURE.md)

## Support

For questions or issues:
1. Check existing test files for examples
2. Review test utilities in `src/tests/test-utils.ts`
3. Consult Vitest documentation
4. Open an issue in the repository
