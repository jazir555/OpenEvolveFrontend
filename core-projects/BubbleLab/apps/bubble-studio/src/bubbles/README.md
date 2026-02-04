# Evolution Bubbles for BubbleLab

Integration of OpenEvolve evolutionary computation with BubbleLab workflows.

## Overview

These bubbles enable BubbleLab workflows to trigger, validate, and apply evolved code from OpenEvolve. They follow the Federation Constitution principles:

- **Air Gap**: No direct imports from core-projects, all integration via APIs
- **Runtime Truth**: Execute calls to verify functionality
- **Idempotency**: All operations can be safely retried
- **Configuration Explicitness**: All values via environment variables or explicit parameters
- **UTC Timestamps**: All time handling in UTC ISO-8601 format

## Bubbles

### 1. EvolutionTriggerBubble

Triggers OpenEvolve evolution workflows and monitors progress.

**Features:**
- Creates and executes OpenEvolve workflows
- Supports evolution, adversarial, and sovereign workflow types
- Monitors progress with adaptive polling
- Circuit breaker pattern for resilience
- Returns best solution and fitness metrics

**Usage:**
```typescript
import { EvolutionTriggerBubble } from '@/bubbles';

const bubble = new EvolutionTriggerBubble({
  problemStatement: 'Optimize the sorting algorithm',
  iterations: 100,
  populationSize: 50,
  workflowType: 'evolution',
});

const result = await bubble.action();
if (result.success) {
  console.log('Best solution:', result.bestSolution);
  console.log('Fitness:', result.fitness);
}
```

**Parameters:**
- `problemStatement`: Problem description for evolution
- `context`: Additional context (optional)
- `workflowType`: 'evolution' | 'adversarial' | 'sovereign'
- `iterations`: Number of iterations (1-10000, default: 100)
- `populationSize`: Population size (1-1000, default: 50)
- `teams`: OpenEvolve team IDs (optional)
- `gauntlets`: OpenEvolve gauntlet IDs (optional)

**Returns:**
- `success`: boolean
- `evolutionId`: Evolution execution ID
- `workflowId`: OpenEvolve workflow ID
- `bestSolution`: Best evolved solution
- `fitness`: Fitness score
- `iterations`: Iterations completed
- `progress`: Progress percentage (0-100)
- `timing`: Execution timing metrics

---

### 2. EvolutionApplicationBubble

Applies evolved code from OpenEvolve to target systems.

**Features:**
- Validates evolved code structure and syntax
- Supports multiple deployment methods (file, API, container, function)
- Runs tests before deployment (optional)
- Automatic rollback on failure (optional)
- Idempotent operations for safe retries
- Deployment verification after completion

**Usage:**
```typescript
import { EvolutionApplicationBubble } from '@/bubbles';

const bubble = new EvolutionApplicationBubble({
  evolvedCode: {
    code: 'function optimized() { ... }',
    language: 'typescript',
    evolutionId: 'evol-123',
  },
  targetConfig: {
    targetSystem: 'bubblelab',
    targetPath: '/src/optimized.ts',
    deploymentMethod: 'file',
    environment: 'development',
  },
  deploymentConfig: {
    autoDeploy: true,
    testBeforeDeploy: true,
    verifyAfterDeploy: true,
  },
});

const result = await bubble.action();
if (result.success) {
  console.log('Deployed to:', result.url);
}
```

**Parameters:**
- `evolvedCode`:
  - `code`: Evolved code content
  - `language`: Programming language
  - `version`: Version identifier (optional)
  - `metadata`: Additional metadata (optional)
  - `evolutionId`: Source evolution ID (optional)
  - `fitness`: Fitness score (optional)
- `targetConfig`:
  - `targetSystem`: 'bubblelab' | 'openevolve' | 'custom'
  - `targetPath`: Target file path or endpoint (optional)
  - `deploymentMethod`: 'file' | 'api' | 'container' | 'function'
  - `environment`: 'development' | 'staging' | 'production'
  - `rollbackEnabled`: Enable automatic rollback (default: true)
- `deploymentConfig` (optional):
  - `autoDeploy`: Deploy without manual approval (default: false)
  - `testBeforeDeploy`: Run tests before deployment (default: true)
  - `deployTimeout`: Deployment timeout in ms (default: 300000)
  - `verifyAfterDeploy`: Verify deployment after completion (default: true)

**Returns:**
- `success`: boolean
- `applicationId`: Application ID
- `deploymentId`: Deployment ID
- `status`: Current status
- `url`: Deployment URL
- `validation`: Validation results
- `rollbackAvailable`: Whether rollback is available
- `timing`: Execution timing metrics

---

### 3. EvolutionValidationBubble

Validates evolved results with formal methods.

**Features:**
- Z3 SMT solver for constraint verification
- LeanAide integration for formal proofs
- Comprehensive test suite execution
- Code coverage analysis
- Confidence scoring (0-1)
- Detailed validation reports with recommendations

**Usage:**
```typescript
import { EvolutionValidationBubble } from '@/bubbles';

const bubble = new EvolutionValidationBubble({
  evolvedCode: {
    code: 'function sorted(arr) { return arr.sort(); }',
    language: 'typescript',
  },
  validationLevel: 'full',
  runZ3Validation: true,
  runLeanAideProof: true,
  runTests: true,
  constraints: ['sorted output', 'same elements'],
  invariants: ['length preserved'],
});

const result = await bubble.action();
console.log('Valid:', result.valid);
console.log('Confidence:', result.confidence);
```

**Parameters:**
- `evolvedCode`: Evolved code to validate (same structure as EvolutionApplicationBubble)
- `validationLevel`: 'basic' | 'standard' | 'full' (default: 'standard')
- `runZ3Validation`: Run Z3 SMT solver validation (default: true)
- `runLeanAideProof`: Generate LeanAide formal proof (default: false)
- `runTests`: Run test suite (default: true)
- `constraints`: Constraints to verify (optional)
- `invariants`: Invariants to check (optional)

**Returns:**
- `success`: boolean
- `valid`: Overall validation result
- `confidence`: Confidence score (0-1)
- `z3`: Z3 validation results
- `leanaide`: LeanAide proof results
- `tests`: Test results
- `summary`: Human-readable validation summary
- `recommendations`: Recommendations for improvement
- `timing`: Execution timing metrics

---

## Workflow Compositions

Predefined workflow pipelines for common use cases.

### EvolutionPipeline

Complete evolution workflow: Trigger → Validate → Apply

```typescript
import { EvolutionPipeline } from '@/bubbles';

const result = await EvolutionPipeline.execute({
  problemStatement: 'Optimize sorting algorithm',
  iterations: 100,
  targetConfig: {
    targetSystem: 'bubblelab',
    targetPath: '/src/sorting.ts',
  },
});
```

**Flow:**
1. Triggers OpenEvolve evolution
2. Validates results with Z3, LeanAide, and tests
3. Applies validated code to target system

---

### ContinuousEvolution

Scheduled evolution for continuous optimization (daily runs)

```typescript
import { ContinuousEvolution } from '@/bubbles';

const result = await ContinuousEvolution.execute({
  problemStatement: 'Continuously optimize performance',
});
```

**Flow:**
1. Quick evolution (50 iterations)
2. Standard validation (Z3 + tests, no formal proofs)
3. Store metrics for tracking

**Schedule:** Daily at midnight (`0 0 * * *`)

---

### AdaptiveEvolution

Adaptive evolution with feedback loops and knowledge integration

```typescript
import { AdaptiveEvolution } from '@/bubbles';

const result = await AdaptiveEvolution.execute({
  problemStatement: 'Adaptively optimize system',
  learnFromHistory: true,
});
```

**Flow:**
1. Retrieve previous knowledge from evolution metrics
2. Trigger evolution with adaptive parameters
3. Full validation (Z3 + LeanAide + tests)
4. Capture learnings for next iteration

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BubbleLab Workflow                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              EvolutionTriggerBubble                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  openevolveApi.createWorkflow()                     │   │
│  │  openevolveApi.executeWorkflow()                    │   │
│  │  Monitor progress with adaptive polling             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Evolved Code]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             EvolutionValidationBubble                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Z3 SMT Solver → Constraint verification            │   │
│  │  LeanAide → Formal proof generation                 │   │
│  │  Test Suite → Quality validation                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Validated Code]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            EvolutionApplicationBubble                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Validate code structure                            │   │
│  │  Apply to target system (file/API/container)        │   │
│  │  Deploy with rollback support                       │   │
│  │  Monitor and verify deployment                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Deployed Solution]
```

---

## Federation Constitution Compliance

### 1. Air Gap (Source Code Isolation)
- No direct imports from `core-projects/OpenEvolve`
- All integration via `openevolveApi` client
- Clear separation between BubbleLab and OpenEvolve

### 2. Runtime Truth (Anti-Hallucination)
- Execute actual OpenEvolve API calls
- Probe scripts verify API availability
- No assumptions about API behavior

### 3. Idempotency (Replayability Pact)
- `EvolutionApplicationBubble` operations are idempotent
- Checksum-based verification prevents duplicate applications
- Safe to retry failed operations

### 4. Configuration Explicitness
- All parameters explicitly defined
- Environment variables for API endpoints
- No magic defaults ( timeouts, retries all configurable)

### 5. UTC Timestamp Handling
- All timestamps in UTC ISO-8601 format
- Consistent timezone handling across bubbles
- Clear conversion at system boundaries

---

## Error Handling

### Circuit Breaker Pattern

`EvolutionTriggerBubble` implements circuit breaker for resilience:

```typescript
// Circuit breaker opens after 3 consecutive failures
// Cooldown period: 1 minute
// Automatically resets after cooldown

if (EvolutionTriggerBubble.isCircuitBreakerOpen()) {
  throw new Error('Circuit breaker is open. Please try again later.');
}
```

### Retry Logic

All bubbles implement exponential backoff retry:

```typescript
// ApiClient configuration (in openevolveApi.ts)
{
  enableRetry: true,
  maxRetries: 3,
  retryDelay: 2000, // Base delay 2 seconds
}
```

### Validation Failures

`EvolutionValidationBubble` provides detailed error information:

```typescript
if (!validationResult.valid) {
  console.error('Validation failed:', validationResult.summary);
  console.error('Recommendations:', validationResult.recommendations);

  if (validationResult.z3?.errors) {
    console.error('Z3 errors:', validationResult.z3.errors);
  }

  if (validationResult.tests?.failed > 0) {
    console.error('Test failures:', validationResult.tests.failed);
  }
}
```

---

## Testing

### Unit Tests

```typescript
import { EvolutionTriggerBubble } from '@/bubbles';

describe('EvolutionTriggerBubble', () => {
  it('should trigger evolution successfully', async () => {
    const bubble = new EvolutionTriggerBubble({
      problemStatement: 'Test problem',
      iterations: 10,
    });

    const result = await bubble.action();
    expect(result.success).toBe(true);
    expect(result.bestSolution).toBeDefined();
  });
});
```

### Integration Tests

```typescript
import { EvolutionPipeline } from '@/bubbles';

describe('EvolutionPipeline', () => {
  it('should execute complete pipeline', async () => {
    const result = await EvolutionPipeline.execute({
      problemStatement: 'Optimize test function',
      targetConfig: {
        targetSystem: 'bubblelab',
        targetPath: '/test.ts',
      },
    });

    expect(result.evolution.success).toBe(true);
    expect(result.validation.valid).toBe(true);
    expect(result.application.success).toBe(true);
  });
});
```

---

## Monitoring and Observability

All bubbles use structured logging:

```typescript
logger.info({
  msg: 'Evolution completed successfully',
  component: 'EvolutionTriggerBubble',
  evolution_id: result.evolutionId,
  iterations: result.iterations,
  fitness: result.fitness,
  timing: result.timing,
});
```

**Log Format:** JSON Lines (`jsonl`)

**Required Context:**
- `correlation_id`: Request tracking
- `source_service`: BubbleLab
- `target_service`: OpenEvolve

---

## Future Enhancements

### Planned Features

1. **Knowledge Integration**
   - Retrieve previous evolutions from knowledge base
   - Learn from successful patterns
   - Avoid previously failed approaches

2. **Multi-Objective Evolution**
   - Support for multiple fitness functions
   - Pareto frontier optimization
   - Trade-off analysis

3. **Distributed Evolution**
   - Parallel evolution across multiple instances
   - Federated learning approach
   - Result aggregation

4. **Advanced Validation**
   - Property-based testing
   - Mutation testing
   - Security vulnerability scanning

---

## Contributing

When adding new evolution bubbles:

1. **Follow Federation Constitution**
   - Maintain air gap from core-projects
   - Use runtime truth over assumptions
   - Ensure idempotent operations

2. **Add Comprehensive Tests**
   - Unit tests for all methods
   - Integration tests for API calls
   - Contract tests for schema validation

3. **Document Thoroughly**
   - JSDoc comments for all public methods
   - Usage examples in README
   - Type definitions for all parameters

4. **Log Structured Events**
   - Use logger for all operations
   - Include correlation IDs
   - Track timing metrics

---

## License

Part of the BubbleLab project. See main project LICENSE file.

---

## Contact

For questions or issues:
- GitHub Issues: [BubbleLab Repository]
- Documentation: [BubbleLab Docs]
