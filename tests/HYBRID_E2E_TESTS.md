# Hybrid PES Evolution E2E Tests (TypeScript/Jest)

## Overview

Comprehensive end-to-end tests for the hybrid OpenEvolve LoongFlow PES (Plan-Execute-Summarize) system using TypeScript and Jest. These tests validate the complete integration of evolutionary optimization and planning systems.

### Test Statistics

- **Total Test Suites**: 7
- **Total Test Functions**: 25+
- **Test Framework**: Jest with TypeScript
- **Coverage Areas**: E2E Integration, Workflows, Knowledge, Error Handling, Performance

## Test Architecture

```
tests/
├── test_hybrid_pes_evolution_e2e.test.ts  # Main E2E test suite (TypeScript)
├── mocks/                                  # Mock adapters
│   ├── loongflow-mock.ts                  # Mock LoongFlow adapter
│   └── openevolve-mock.ts                 # Mock OpenEvolve adapter
├── archive/                                # Archived Python tests
│   └── test_hybrid_pes_evolution_e2e.py.bak
├── jest.setup.ts                           # Jest setup file
├── run_hybrid_e2e_tests.sh                # Test runner script
└── HYBRID_E2E_TESTS.md                     # This documentation
```

## Prerequisites

### Required

- Node.js >= 18.0.0
- npm >= 9.0.0
- TypeScript >= 5.3.0
- Jest >= 29.7.0

### Optional (for real service testing)

- LoongFlow Core running on port 8050
- LoongFlow Adapter running on port 8040
- OpenEvolve Adapter running on port 8030
- Graphiti (for knowledge tests)
- VectorDB (for knowledge tests)

### Installation

```bash
# Install dependencies
npm install

# Or install specific dependencies
npm install --save-dev jest @types/jest ts-jest typescript
npm install uuid zod
```

## Running Tests

### Basic Usage

```bash
# Run all tests
npm test

# Run E2E tests
npm run test:e2e

# Run with watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run with verbose output
npm run test:verbose
```

### Advanced Usage

```bash
# Run specific test suite
npx jest tests/test_hybrid_pes_evolution_e2e.test.ts --testNamePattern="TestBasicPESExecution"

# Run specific test
npx jest tests/test_hybrid_pes_evolution_e2e.test.ts --testNamePattern="should submit and execute problem"

# Skip slow tests
SKIP_SLOW_TESTS=true npm test

# Include slow tests
SKIP_SLOW_TESTS=false npm test

# Enable knowledge tests
ENABLE_KNOWLEDGE_TESTS=true npm test
```

### Using the Test Runner Script

```bash
# Run all tests
./tests/run_hybrid_e2e_tests.sh

# Run with coverage
./tests/run_hybrid_e2e_tests.sh --coverage

# Run only PES Evolution tests
./tests/run_hybrid_e2e_tests.sh --filter 'PESEvolution'

# Run in watch mode
./tests/run_hybrid_e2e_tests.sh --watch

# Show help
./tests/run_hybrid_e2e_tests.sh --help
```

## Environment Variables

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOONGFLOW_API_URL` | `http://localhost:8050` | LoongFlow Core API endpoint |
| `LOONGFLOW_ADAPTER_URL` | `http://localhost:8040` | LoongFlow Adapter endpoint |
| `OPENEVOLVE_API_URL` | `http://localhost:8030` | OpenEvolve Adapter endpoint |
| `TEST_TIMEOUT` | `60000` | Test timeout in milliseconds |
| `SKIP_SLOW_TESTS` | `true` | Skip slow-running tests |
| `ENABLE_KNOWLEDGE_TESTS` | `false` | Enable knowledge extraction tests |
| `NODE_ENV` | `test` | Node environment |

### Jest Configuration

Jest is configured in `jest.config.js`:

- **Preset**: `ts-jest`
- **Test Environment**: Node
- **Test Match**: `**/*.test.ts`
- **Coverage**: Collects from `glue/**/*.{ts,tsx}`
- **Timeout**: 60 seconds (configurable per test)

## Test Coverage

### 1. Basic PES Execution (TestBasicPESExecution)

Tests the fundamental Plan-Execute-Summarize cycle:

- **test_submit_and_execute_problem**: Submit problem and retrieve results
- **test_retrieve_solution_results**: Validate solution structure
- **test_pes_cycle_complete**: Full P-E-S cycle validation
- **test_get_best_solutions**: Retrieve top solutions from database

**Validates:**
- Problem submission
- Status tracking
- Result retrieval
- Solution quality metrics

### 2. Evolutionary Optimization (TestEvolutionaryOptimization)

Tests OpenEvolve integration:

- **test_evolve_solution**: Basic evolution workflow
- **test_multi_generation_evolution**: Multi-generational optimization
- **test_evolutionary_parameters**: Different parameter configurations

**Validates:**
- Solution evolution
- Population management
- Fitness improvement
- Parameter tuning

### 3. Hybrid Workflows (TestHybridWorkflows)

Tests combined PES + Evolution workflows:

- **test_pes_evolution_workflow**: Sequential PES → Evolution
- **test_knowledge_extraction_workflow**: Extract and store knowledge
- **test_adaptive_execution_workflow**: Adaptive paradigm switching
- **test_multi_stage_reasoning_workflow**: Multi-stage reasoning pipeline

**Validates:**
- Workflow orchestration
- System integration
- Knowledge extraction
- Adaptive behavior
- Multi-stage processing

### 4. Knowledge Management (TestKnowledgeManagement)

Tests knowledge extraction and reuse:

- **test_extract_evolutionary_knowledge**: Extract patterns from solutions
- **test_reuse_knowledge_for_new_problems**: Apply knowledge to new problems

**Validates:**
- Knowledge extraction
- Pattern recognition
- Knowledge reuse
- Success rate tracking

### 5. Error Handling and Recovery (TestErrorHandlingAndRecovery)

Tests failure scenarios:

- **test_timeout_handling**: Timeout handling for long operations
- **test_retry_with_backoff**: Exponential backoff retry logic
- **test_invalid_input_handling**: Invalid input handling
- **test_missing_execution_id**: Missing resource handling

**Validates:**
- Graceful failure handling
- Retry mechanisms
- Error recovery
- Input validation

### 6. Performance and Scalability (TestPerformanceAndScalability)

Tests system performance:

- **test_concurrent_problem_execution**: Concurrent execution
- **test_workflow_execution_time**: Execution time validation
- **test_large_problem_handling**: Large-scale problem handling
- **test_resource_cleanup**: Resource management

**Validates:**
- Concurrency
- Performance
- Scalability
- Resource management

## Mock Adapters and Test Utilities

### Mock Adapter Pattern

The tests use mock adapters located in `tests/mocks/`:

```typescript
import { createMockLoongFlowAdapter } from './mocks/loongflow-mock';
import { createMockOpenEvolveAdapter } from './mocks/openevolve-mock';

// Create mock with custom configuration
const loongflow = createMockLoongFlowAdapter({
  mockSolution: {
    solution: 'def solve(x): return x * 2',
    score: 0.95,
  },
  mockLowConfidence: false,
});

const openevolve = createMockOpenEvolveAdapter({
  mockOptimized: {
    solution: 'optimized solution',
    fitness: 0.98,
  },
});
```

### Creating Test Problems

```typescript
import { Problem } from '../glue/schemas/pes-canonical';

const problem: Problem = {
  id: crypto.randomUUID(),
  type: 'optimization',
  description: 'Maximize f(x) = x^2',
  context: { domain: 'mathematics' },
  constraints: [],
  success_criteria: [],
  created_at: new Date().toISOString(),
};
```

### Using Global Test Utilities

```typescript
// Create test problem
const problem = global.testUtils.createTestProblem({
  type: 'reasoning',
  description: 'Test problem',
});

// Wait for async operations
await global.testUtils.wait(1000);

// Retry helper
const result = await global.testUtils.retry(
  async () => await flakyOperation(),
  3,  // max retries
  100 // delay
);
```

## Test Patterns

### Async Test Pattern

```typescript
it('should execute async operation', async () => {
  const loongflow = createMockLoongFlowAdapter();

  const result = await loongflow.submitProblem({
    task: problem.description,
    max_iterations: 10,
  });

  const solution = await loongflow.getExecutionResult(result.agent_id);

  expect(solution).toBeDefined();
  expect(solution.final_score).toBeGreaterThanOrEqual(0);
});
```

### Mock Adapter Pattern

```typescript
import { createMockLoongFlowAdapter, MockLoongFlowConfig } from './mocks/loongflow-mock';

const config: MockLoongFlowConfig = {
  mockLowConfidence: true,
  mockTimeout: false,
};

const loongflow = createMockLoongFlowAdapter(config);
```

### Retry Pattern with Exponential Backoff

```typescript
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delay = 100
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt)));
    }
  }
  throw new Error('Retry failed');
}

// Usage
const result = await retryWithBackoff(async () => {
  return await flakyOperation();
});
```

## CI/CD Integration

### GitLab CI

```yaml
e2e_tests:
  stage: test
  script:
    - npm install
    - npm run test:e2e
  only:
    - main
    - develop
  artifacts:
    when: always
    reports:
      junit: test-results.xml
    coverage:
      - coverage/
```

### GitHub Actions

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm install
      - name: Run E2E tests
        run: npm run test:e2e
        env:
          SKIP_SLOW_TESTS: ${{ github.event_name == 'pull_request' }}
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('E2E Tests') {
            steps {
                sh 'npm install'
                sh 'npm run test:e2e'
            }
        }
    }
    post {
        always {
            junit 'test-results.xml'
        }
    }
}
```

## Troubleshooting

### Tests Fail with Module Not Found

**Problem**: Cannot find module 'glue/...'

**Solution**: Ensure tsconfig.json paths are configured correctly and run `npm install`

```bash
# Verify TypeScript configuration
npm run typecheck

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Tests Timeout

**Problem**: Tests timeout after 60 seconds

**Solution**: Increase timeout in jest.config.js or use jest.setTimeout()

```typescript
// Increase timeout for specific test
jest.setTimeout(120000);

// Or in jest.config.js
module.exports = {
  testTimeout: 120000,
};
```

### Type Errors

**Problem**: TypeScript compilation errors

**Solution**: Run typecheck to identify issues

```bash
npm run typecheck
```

### Mock Adapter Not Working

**Problem**: Mock adapter returns unexpected values

**Solution**: Check mock configuration and ensure correct parameters are passed

```typescript
// Enable verbose logging
const loongflow = createMockLoongFlowAdapter({
  mockSolution: {
    solution: 'expected solution',
    score: 0.95,
  },
});

// Verify mock behavior
console.log(await loongflow.submitProblem({ task: 'test' }));
```

## Migration from Python

### Key Changes

1. **Test Framework**: pytest → Jest
2. **Language**: Python → TypeScript
3. **Mocking**: unittest.mock → Custom mock adapters
4. **Assertions**: assert → expect().toBe()
5. **Async/Await**: async/await (same syntax)

### Example Comparison

**Python (pytest)**:
```python
@pytest.mark.asyncio
async def test_submit_and_execute_problem(self, setup_adapters, sample_problem):
    loongflow_adapter = setup_adapters['loongflow']
    result = await loongflow_adapter.submit_problem(sample_problem)
    assert result['status'] == 'SUBMITTED'
```

**TypeScript (Jest)**:
```typescript
it('should submit and execute problem', async () => {
  const loongflow = createMockLoongFlowAdapter();
  const result = await loongflow.submitProblem({
    task: problem.description,
    max_iterations: 10,
  });
  expect(result.status).toBe('SUBMITTED');
});
```

## Test Statistics

- **Total Test Functions**: 25+
- **Test Suites**: 7
- **Mock Adapters**: 2
- **Lines of Code**: ~1200
- **Expected Runtime**: 30-60 seconds (without slow tests), 2-3 minutes (all tests)

## Success Criteria

All E2E tests should:

- ✓ Pass consistently (>95% success rate)
- ✓ Complete within timeout limits
- ✓ Provide clear failure messages
- ✓ Test real integration points
- ✓ Validate end-to-end workflows
- ✓ Cover error scenarios
- ✓ Test performance characteristics

## Future Enhancements

- [ ] Add visual performance regression tests
- [ ] Add load testing for concurrent workflows
- [ ] Add chaos engineering tests
- [ ] Add A/B testing for workflow strategies
- [ ] Add real-time monitoring integration
- [ ] Add automated test data generation
- [ ] Add cross-service integration tests

## Contributing

When adding new E2E tests:

1. Follow the existing test structure
2. Use async/await for async operations
3. Add appropriate test descriptions
4. Document the test purpose
5. Use mock adapters for isolation
6. Validate both success and failure cases
7. Ensure tests are idempotent
8. Add cleanup in afterEach if needed

### Test Checklist

- [ ] Test is isolated (no dependencies on other tests)
- [ ] Test has clear description following "should <behavior>" pattern
- [ ] Test uses mock adapters
- [ ] Test handles errors appropriately
- [ ] Test is fast (< 5 seconds) or marked with slow test pattern
- [ ] Test updates this documentation

## Version History

### 2.0.0 (TypeScript/Jest)
- Migrated from Python/pytest to TypeScript/Jest
- Added mock adapters in TypeScript
- Updated test runner script for Jest
- Added comprehensive documentation
- 25+ test functions across 7 test suites

### 1.0.0 (Python/pytest)
- Initial implementation
- 35+ test functions
- 7 test classes

## References

- [Jest Documentation](https://jestjs.io/)
- [TypeScript Documentation](https://www.typescriptlang.org/)
- [Main E2E Test Suite](test_hybrid_pes_evolution_e2e.test.ts)
- [Mock LoongFlow Adapter](mocks/loongflow-mock.ts)
- [Mock OpenEvolve Adapter](mocks/openevolve-mock.ts)
- [Test Runner Script](run_hybrid_e2e_tests.sh)
- [Federation Constitution](../CLAUDE.md)

---

**Author**: OpenEvolve Distinguished Engineer
**Version**: 2.0.0 (TypeScript/Jest)
**Last Updated**: 2025-02-22
