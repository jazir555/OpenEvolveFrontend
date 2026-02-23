# PES Canonical Schemas Documentation

## Overview

This directory contains the canonical data models for Plan-Execute-Summarize (PES) pattern workflows, LoongFlow integration, and Hybrid PES-Evolution systems. These schemas serve as the **Anti-Corruption Layer** between different AI problem-solving paradigms.

## Architecture Principles

Following the **Federation Constitution**, these schemas enforce:

1. **Law of the "Air Gap"**: No direct imports from core-projects
2. **Law of "Runtime Truth"**: All schemas validated at runtime using Zod
3. **Law of Configuration Explicitness**: No magic defaults
4. **Law of UTC**: All timestamps in UTC ISO-8601 format
5. **Law of Idempotency**: All operations safe to replay

## Schema Files

### 1. `pes-canonical.ts` - Generic PES Pattern

Defines the fundamental Plan-Execute-Summarize pattern schemas:

- **Problem**: Input problem specification
- **ExecutionPlan**: Multi-step execution plan
- **ExecutionStep**: Individual plan step
- **ExecutionResult**: Complete execution outcome
- **Summary**: Evaluation and insights (PES "Summarize" phase)
- **PerformanceAssessment**: Multi-dimensional performance metrics

**Key Enums:**
- `ProblemType`: optimization, reasoning, generation, validation, etc.
- `ExecutionStepType`: plan, execute, summarize, validate, transform, aggregate
- `ExecutionState`: pending, running, completed, failed, cancelled, timeout, paused

**Usage Example:**
```typescript
import {
  Problem,
  validateProblem,
  createPESUTCTimestamp,
  createPESCorrelationId
} from './glue/schemas';

const problem = {
  id: createPESCorrelationId(),
  type: 'optimization',
  description: 'Optimize neural network hyperparameters',
  context: { dataset: 'MNIST' },
  constraints: ['max_layers <= 10'],
  success_criteria: ['accuracy > 0.95'],
  created_at: createPESUTCTimestamp(),
};

const validation = validateProblem(problem);
if (validation.success) {
  // Use validated problem
  executeProblem(validation.data);
}
```

### 2. `loongflow-canonical.ts` - LoongFlow Specific

Defines schemas specific to LoongFlow (PES + Evolutionary Optimization):

- **LoongFlowSolution**: Solution with evolutionary metadata (island_id, iteration, parent_id)
- **LLMConfig**: LLM provider configuration
- **WorkerConfig**: Planner, Executor, Summarizer configuration
- **EvolutionConfig**: MAP-Elites, island model, Boltzmann sampling config
- **LoongFlowConfig**: Complete LoongFlow task configuration
- **LoongFlowRequest/Response**: Request and response messages
- **LoongFlowCheckpoint**: Execution state checkpoint

**Key Features:**
- Direct mapping to LoongFlow's Python `Solution` dataclass
- Support for island-based parallel evolution
- MAP-Elites fitness map tracking
- Checkpoint/resume capability

**Usage Example:**
```typescript
import {
  LoongFlowConfig,
  validateLoongFlowRequest,
  transformLoongFlowResponseToExecutionResult
} from './glue/schemas';

const request = {
  problem_id: problemId,
  task_config: {
    task_name: 'Optimize architecture',
    llm_config: {
      provider: 'anthropic',
      model: 'claude-3-opus',
      temperature: 0.7,
      max_tokens: 2000,
    },
    workers: {
      planner: 'LLMPlanner',
      executor: 'CodeExecutor',
      summarizer: 'QualitySummarizer',
    },
    evolution: {
      iterations: 50,
      islands: 4,
      sample_size: 100,
      map_elites_enabled: true,
    },
  },
  initial_context: { dataset: 'ImageNet' },
};

// Execute and transform to canonical format
const result = await loongflowAdapter.execute(request);
const canonicalResult = transformLoongFlowResponseToExecutionResult(
  result,
  request.problem_id,
  planId
);
```

### 3. `hybrid-pes-evolution-canonical.ts` - Hybrid Integration

Defines schemas for combining PES and Evolutionary approaches:

- **HybridTask**: Task combining both paradigms
- **IntegrationStrategy**: sequential, parallel, interleaved, adaptive
- **AdaptiveTrigger**: Conditions for dynamic paradigm switching
- **EvolutionaryKnowledge**: Extracted knowledge for reuse
- **HybridExecutionResult**: Combined execution results
- **KnowledgeTransfer**: Knowledge transfer between paradigms

**Hybrid Task Types:**
- `pes_optimize`: PES planning → Evolutionary optimization
- `evolve_solve`: Evolutionary exploration → PES refinement
- `adaptive_execute`: Dynamic switching based on performance
- `parallel_hybrid`: Run both simultaneously, merge results
- `interleaved`: Alternate between paradigms

**Usage Example:**
```typescript
import {
  HybridTask,
  validateHybridTask,
  AdaptiveTriggerCondition,
  AdaptiveAction
} from './glue/schemas';

const hybridTask = {
  id: createPESCorrelationId(),
  type: 'adaptive_execute',
  problem: problem,
  pes_config: loongflowConfig,
  evolution_config: {
    generations: 100,
    population_size: 50,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
  },
  integration_strategy: 'adaptive',
  adaptive_triggers: [
    {
      condition: 'stagnation',
      threshold: 5,
      action: 'switch_to_evolution',
    },
    {
      condition: 'low_confidence',
      threshold: 0.6,
      action: 'increase_iterations',
    },
  ],
  created_at: createPESUTCTimestamp(),
};

const validation = validateHybridTask(hybridTask);
```

## Validation

All schemas use **Zod** for runtime validation:

```typescript
import { validateProblem } from './glue/schemas';

const validation = validateProblem(rawData);
if (!validation.success) {
  console.error('Validation failed:', validation.errors);
  // Handle validation errors
}

// Use validated data
const problem = validation.data;
```

## Transformation Functions

Each schema module provides transformation functions:

### To Canonical (from external format)
```typescript
import {
  transformProblemToCanonical,
  transformExecutionResultToCanonical,
  transformLoongFlowSolutionToCanonical
} from './glue/schemas';

const canonicalProblem = transformProblemToCanonical(rawProblem);
const canonicalResult = transformExecutionResultToCanonical(rawResult);
const canonicalSolution = transformLoongFlowSolutionToCanonical(rawSolution);
```

### From Canonical (to external format)
```typescript
import {
  transformCanonicalToProblem,
  transformCanonicalToLoongFlowSolution,
  transformCanonicalProblemToLoongFlowRequest
} from './glue/schemas';

const externalProblem = transformCanonicalToProblem(canonicalProblem);
const externalSolution = transformCanonicalToLoongFlowSolution(canonicalSolution);
const lfRequest = transformCanonicalProblemToLoongFlowRequest(problem, config);
```

## Type Guards

Runtime type checking utilities:

```typescript
import {
  isProblem,
  isExecutionResult,
  isLoongFlowSolution,
  isHybridTask
} from './glue/schemas';

if (isProblem(data)) {
  // TypeScript knows data is Problem here
  console.log(data.type, data.description);
}

if (isLoongFlowSolution(data)) {
  // TypeScript knows data is LoongFlowSolution
  console.log(data.score, data.island_id);
}
```

## Utility Functions

### Timestamp Creation (Law of UTC)
```typescript
import { createPESUTCTimestamp } from './glue/schemas';

const timestamp = createPESUTCTimestamp();
// Output: "2024-02-22T10:30:45.123Z"
```

### Correlation ID Creation
```typescript
import { createPESCorrelationId } from './glue/schemas';

const correlationId = createPESCorrelationId();
// Output: "550e8400-e29b-41d4-a716-446655440000"
```

## Constants

### Default Timeouts
```typescript
import { DEFAULT_TIMEOUTS } from './glue/schemas';

const timeout = DEFAULT_TIMEOUTS.NORMAL; // 15000ms
```

### Maximum Sizes
```typescript
import { MAX_SIZES } from './glue/schemas';

// PES limits
MAX_SIZES.EXECUTION_STEPS; // 1000
MAX_SIZES.PROBLEM_DESCRIPTION_LENGTH; // 100000

// LoongFlow limits
MAX_SIZES.LOONGFLOW_ISLANDS; // 100
MAX_SIZES.LOONGFLOW_ITERATIONS; // 10000

// Hybrid limits
MAX_SIZES.EVOLUTION_GENERATIONS; // 100000
MAX_SIZES.POPULATION_SIZE; // 100000
```

## Error Codes

```typescript
import { VALIDATION_ERRORS } from './glue/schemas';

if (error.code === VALIDATION_ERRORS.MISSING_FIELD) {
  // Handle missing required field
}
```

## Schema Registry

Central registry of all schemas:

```typescript
import { SchemaRegistry } from './glue/schemas';

console.log(SchemaRegistry.pes.name); // 'pes'
console.log(SchemaRegistry.pes.version); // '1.0.0'
console.log(SchemaRegistry.pes.schemas); // List of PES schemas
```

## Testing

Comprehensive test suites validate all schemas:

```bash
# Run all schema tests
npm test glue/schemas/__tests__

# Run specific schema test suite
npm test glue/schemas/__tests__/pes-schemas.test.ts
npm test glue/schemas/__tests__/loongflow-schemas.test.ts
npm test glue/schemas/__tests__/hybrid-schemas.test.ts
```

## Integration with Other Systems

### With OpenEvolve
```typescript
import { EvolutionaryKnowledge, transformLoongFlowSolutionToKnowledge } from './glue/schemas';

// Extract knowledge from LoongFlow solution
const knowledge = transformLoongFlowSolutionToKnowledge(
  loongflowSolution,
  problemId,
  'solution_pattern'
);

// Share with OpenEvolve
await openEvolveAdapter.ingestKnowledge(knowledge);
```

### With VectorDB (for retrieval)
```typescript
import { Problem } from './glue/schemas';

// Store problem embedding
await vectorDbAdapter.upsert({
  collection_name: 'problems',
  vectors: [embed(problem.description)],
  metadata: problem,
});
```

### With Graphiti (for knowledge graph)
```typescript
import { EvolutionaryKnowledge } from './glue/schemas';

// Add knowledge to knowledge graph
await graphitiAdapter.addTriplet({
  source: knowledge.source_id,
  relation: 'has_pattern',
  target: knowledge.content.pattern,
});
```

## Best Practices

1. **Always Validate**: Never trust external data, always validate with Zod schemas
2. **Use Transformation Functions**: Convert to/from canonical format at system boundaries
3. **Set Timeouts**: Always provide timeout_ms (Law of Configuration Explicitness)
4. **Track Correlation IDs**: Use correlation IDs for distributed tracing
5. **Follow UTC Law**: Always use createPESUTCTimestamp() for timestamps
6. **Handle Errors Gracefully**: Check validation.success before using data
7. **Use Type Guards**: Verify data types at runtime with is*() functions
8. **Document Metadata**: Use optional metadata fields for extensibility

## Anti-Corruption Layer Pattern

```typescript
// ❌ BAD: Direct pass-through
const result = await loongflowAPI.execute(rawRequest);
await openEvolveAPI.optimize(result); // Wrong format!

// ✅ GOOD: Anti-corruption layer
const canonicalRequest = transformToLoongFlowRequest(rawRequest);
const loongflowResult = await loongflowAPI.execute(canonicalRequest);
const canonicalResult = transformLoongFlowResponseToCanonical(
  loongflowResult,
  problemId,
  planId
);
const evolutionRequest = transformCanonicalToEvolutionRequest(canonicalResult);
await openEvolveAPI.optimize(evolutionRequest); // Correct format!
```

## Schema Versioning

All schemas are versioned (currently 1.0.0). When updating schemas:

1. Increment version in SchemaRegistry
2. Add transformation functions for backward compatibility
3. Update validation tests
4. Document breaking changes

## Troubleshooting

### Validation Errors
```typescript
const validation = validateProblem(data);
if (!validation.success) {
  console.error('Validation errors:');
  validation.errors.forEach(error => console.error(`  - ${error}`));
}
```

### Type Mismatches
```typescript
// Use type guards to check at runtime
if (isLoongFlowSolution(data)) {
  // Safe to access LoongFlowSolution properties
  console.log(data.score, data.island_id);
} else {
  console.error('Data is not a valid LoongFlowSolution');
}
```

### Timestamp Issues
```typescript
// Always use UTC timestamp utility
const timestamp = createPESUTCTimestamp(); // ✅ Correct

// Don't use local time
const badTimestamp = new Date().toISOString(); // ❌ May not be UTC
```

## Further Reading

- **CLAUDE.md**: Federation Constitution and operating principles
- **Task #1**: LoongFlow adapter implementation
- **Task #3**: Hybrid orchestration workflows
- **Zod Documentation**: https://zod.dev/

## Contact

For questions or issues with PES schemas, please refer to the project documentation or create an issue in the repository.
