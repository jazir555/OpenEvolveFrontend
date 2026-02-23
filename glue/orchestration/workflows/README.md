# Hybrid PES-Evolution Orchestration Workflows

This module provides sophisticated orchestration workflows that combine LoongFlow's Plan-Execute-Summarize (PES) with OpenEvolve's evolutionary optimization for advanced AI problem-solving.

## Overview

These workflows orchestrate LoongFlow (PES + Evolution) and OpenEvolve (pure evolution) in powerful hybrid patterns that enable advanced problem-solving beyond what either system can achieve alone.

### Available Workflows

#### 1. PESEvolutionWorkflow
**Purpose**: Sequential PES → Evolution → Summary

**Flow**:
1. Plan (LoongFlow) - Create execution plan
2. Execute (LoongFlow) - Execute PES to get initial solution
3. Optimize (OpenEvolve) - Evolve the solution
4. Summarize (LoongFlow) - Create summary and evaluation
5. Extract Knowledge - Extract and store evolutionary knowledge

**Use Cases**:
- Problems that benefit from both planning and evolutionary optimization
- When you need to track incremental improvements
- When knowledge extraction for future use is important

**Example**:
```typescript
const workflow = createPESEvolutionWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  checkpoints_enabled: true,
  default_timeout_ms: 300000, // 5 minutes
});

const result = await workflow.execute({
  problem: {
    id: 'prob-123',
    type: 'optimization',
    description: 'Optimize neural network hyperparameters',
    context: { dataset: 'MNIST' },
    constraints: ['max_layers <= 10'],
    success_criteria: ['accuracy > 0.95'],
    created_at: new Date().toISOString(),
  },
  pes_config: {
    max_iterations: 10,
    target_score: 0.9,
  },
  evolution_config: {
    generations: 10,
    population_size: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
  },
});
```

#### 2. KnowledgeExtractionWorkflow
**Purpose**: Extract knowledge from solutions and formulate new problems

**Flow**:
1. Retrieve Solutions - Get best solutions from LoongFlow database
2. Analyze Patterns - Identify patterns in solutions
3. Extract Knowledge - Create knowledge artifacts
4. Store in Graphiti - Optional knowledge graph storage
5. Vectorize - Optional embedding for semantic search
6. Formulate Problems - Create new problems based on knowledge gaps

**Use Cases**:
- Learning from past solutions
- Identifying knowledge gaps
- Building knowledge bases
- Semantic search over solutions

**Example**:
```typescript
const workflow = createKnowledgeExtractionWorkflow({
  loongflowAdapter,
  eventBus,
  graphitiAdapter, // Optional
  vectorDBAdapter, // Optional
  enable_graph_storage: true,
  enable_vectorization: true,
  enable_problem_formulation: true,
});

const result = await workflow.execute({
  island_id: 0,
  top_k: 10,
  min_score: 0.7,
  knowledge_types: ['solution_pattern', 'planning_strategy'],
});

console.log(`Extracted ${result.knowledge.length} knowledge items`);
console.log(`Formulated ${result.problems.length} new problems`);
```

#### 3. AdaptiveExecutionWorkflow
**Purpose**: Dynamically switch paradigms based on performance triggers

**Flow**:
1. Start with PES execution
2. Monitor confidence and progress
3. If trigger conditions met, switch paradigms
4. Continue with new paradigm
5. Select best result

**Use Cases**:
- When problem difficulty is unknown
- When runtime constraints exist
- When you need adaptive resource allocation

**Example**:
```typescript
const workflow = createAdaptiveExecutionWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  eventBus,
  max_paradigm_switches: 3,
  max_iterations: 10,
});

const result = await workflow.executeAdaptive({
  problem: {
    id: 'prob-456',
    type: 'reasoning',
    description: 'Prove mathematical theorem',
    context: { domain: 'mathematics' },
    constraints: [],
    success_criteria: ['proof_valid'],
    created_at: new Date().toISOString(),
  },
  initial_paradigm: 'PES',
  triggers: [
    {
      id: uuidv4(),
      condition: 'low_confidence',
      threshold: 0.7,
      action: 'switch_to_evolution',
    },
    {
      id: uuidv4(),
      condition: 'stagnation',
      threshold: 5,
      action: 'switch_to_evolution',
    },
  ],
});
```

#### 4. MultiStageReasoningWorkflow
**Purpose**: Complex multi-system reasoning with validation

**Flow**:
1. Plan (LoongFlow) - Create execution plan
2. Optimize (OpenEvolve) - Evolve solution
3. Validate (Z3/LeanAide) - Formal validation
4. Refine (OpenEvolve) - Fix validation errors
5. Summarize (LoongFlow) - Create summary
6. Extract Knowledge - Store learnings

**Use Cases**:
- Formal verification problems
- Mathematical proofs
- High-stakes reasoning requiring validation
- Safety-critical systems

**Example**:
```typescript
const workflow = createMultiStageReasoningWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  z3Adapter, // Optional
  leanAideAdapter, // Optional
  eventBus,
  enable_validation: true,
  enable_refinement: true,
  max_refinement_loops: 3,
});

const result = await workflow.executeReasoning({
  problem: {
    id: 'prob-789',
    type: 'validation',
    description: 'Prove program correctness',
    context: { language: 'C' },
    constraints: ['memory_safe'],
    success_criteria: ['no_buffer_overflows'],
    created_at: new Date().toISOString(),
  },
  validation_system: 'both',
  refinement_threshold: 0.8,
});
```

## Event Types

All workflows publish events to the event bus for monitoring and orchestration:

### Core Events
- **WorkflowStarted** - Workflow execution begun
- **WorkflowCompleted** - Workflow finished successfully
- **WorkflowFailed** - Workflow failed with error
- **StageCompleted** - Individual stage completed
- **StageFailed** - Individual stage failed

### Workflow-Specific Events

**PESEvolutionWorkflow**:
- `ProblemPlanned` - Plan created
- `SolutionExecuted` - PES execution finished
- `SolutionOptimized` - Evolution optimization finished
- `ResultSummarized` - Summary created
- `KnowledgeExtracted` - Knowledge extracted

**KnowledgeExtractionWorkflow**:
- `SolutionsRetrieved` - Solutions retrieved from database
- `PatternsAnalyzed` - Pattern analysis completed
- `KnowledgeExtracted` - Knowledge items extracted
- `GraphUpdated` - Knowledge graph updated (if enabled)
- `VectorIndexed` - Vectors indexed (if enabled)

**AdaptiveExecutionWorkflow**:
- `ParadigmSwitched` - Switched from PES to Evolution or vice versa
- `WorkflowCompleted` - Final result ready

**MultiStageReasoningWorkflow**:
- `StageCompleted` - Each of 6 stages
- `ProofVerified` - Validation completed (if Z3/LeanAide enabled)
- `ResultSummarized` - Final summary created
- `KnowledgeExtracted` - Knowledge extracted

## Error Handling

All workflows follow the Federation Constitution's failure management:

### Circuit Breakers
- Each adapter call goes through a circuit breaker
- Thresholds and timeouts configurable via environment variables
- Automatic state transitions (CLOSED → OPEN → HALF_OPEN)

### Retry Logic
- Automatic retry with exponential backoff
- Configurable max retries
- Jitter to prevent thundering herd

### Dead Letter Queue
- Failed events routed to DLQ
- Can replay failed events
- Preserves error context

### Example Error Handling
```typescript
try {
  const result = await workflow.execute(input);
} catch (error) {
  // Check if circuit breaker is open
  if (workflow.circuitBreaker.getState() === CircuitState.OPEN) {
    console.error('Service is down, using fallback');
    // Use cached result or fallback strategy
  } else {
    throw error; // Re-throw for retry
  }
}
```

## Performance Optimization

### Checkpoints
Enable checkpoints to resume long-running workflows:

```typescript
const workflow = createPESEvolutionWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  checkpoints_enabled: true,
  checkpoint_path: '/tmp/checkpoints',
});

// Workflow will save checkpoints at each stage
// Can resume from last checkpoint if workflow fails
```

### Parallel Execution
Some workflows support parallel execution:

```typescript
const result = await workflow.execute(input, {
  enable_parallel_stages: true, // For MultiStageReasoningWorkflow
});
```

### Resource Limits
Set resource limits via environment variables:

```bash
# Timeouts
PES_EVOLUTION_TIMEOUT_MS=300000
ADAPTIVE_TIMEOUT_MS=600000
MULTI_STAGE_TIMEOUT_MS=1200000

# Iteration limits
ADAPTIVE_MAX_ITERATIONS=10
MULTI_STAGE_MAX_REFINEMENT_LOOPS=3

# Circuit breaker thresholds
PES_EVOLUTION_CIRCUIT_THRESHOLD=5
ADAPTIVE_CIRCUIT_THRESHOLD=3
```

## Configuration

All workflows follow the **Law of Configuration Explicitness**:

### Required Environment Variables
```bash
# LoongFlow Adapter
LOONGFLOW_API_URL=http://loongflow-sidecar:8000
LOONGFLOW_TIMEOUT_MS=30000

# OpenEvolve Adapter
OPENEVOLVE_API_URL=http://openevolve:5000
TIMEOUT_MS=30000
```

### Optional Environment Variables
```bash
# Workflow-specific
PES_EVOLUTION_CHECKPOINTS_ENABLED=true
PES_EVOLUTION_MAX_RETRIES=3
ADAPTIVE_MAX_SWITCHES=3
ADAPTIVE_CONFIDENCE_THRESHOLD=0.7
KNOWLEDGE_ENABLE_GRAPH_STORAGE=true
KNOWLEDGE_ENABLE_VECTORIZATION=true
MULTI_STAGE_ENABLE_VALIDATION=true
MULTI_STAGE_ENABLE_REFINEMENT=true
```

## Usage Examples

### Complete End-to-End Example

```typescript
import {
  createWorkflow,
  WORKFLOWS,
  createPESEvolutionWorkflow,
} from './workflows';
import { LoongFlowAdapter } from '../adapters/loongflow-adapter';
import { OpenEvolveAdapter } from '../adapters/openevolve-adapter';

// Initialize adapters
const loongflowAdapter = new LoongFlowAdapter({
  api_url: process.env.LOONGFLOW_API_URL!,
  timeout_ms: 30000,
});

const openevolveAdapter = new OpenEvolveAdapter({
  api_url: process.env.OPENEVOLVE_API_URL!,
  timeout_ms: 30000,
});

// Create workflow
const workflow = createPESEvolutionWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  checkpoints_enabled: true,
  default_timeout_ms: 300000,
  max_retries: 3,
});

// Execute
const problem = {
  id: uuidv4(),
  type: 'optimization',
  description: 'Optimize deep learning model architecture',
  context: {
    dataset: 'ImageNet',
    base_model: 'ResNet50',
  },
  constraints: [
    'parameters <= 25M',
    'inference_time < 10ms',
  ],
  success_criteria: [
    'accuracy > 0.85',
    'parameters < 25M',
  ],
  created_at: new Date().toISOString(),
  priority: 9,
  tags: ['ml', 'computer_vision', 'optimization'],
};

try {
  const result = await workflow.execute({
    problem,
    pes_config: {
      max_iterations: 15,
      target_score: 0.85,
      concurrency: 4,
    },
    evolution_config: {
      generations: 20,
      population_size: 50,
      mutation_rate: 0.15,
      crossover_rate: 0.75,
      elitism_count: 2,
    },
    enable_optimization: true,
    enable_knowledge_extraction: true,
  });

  console.log('Workflow completed successfully');
  console.log('Final score:', result.integration_metrics.synergy_score);
  console.log('PES iterations:', result.integration_metrics.pes_iterations);
  console.log('Evolution generations:', result.integration_metrics.evolution_generations);
  console.log('Knowledge extracted:', result.knowledge_extracted.length);
} catch (error) {
  console.error('Workflow failed:', error);

  // Check circuit breaker state
  const cbState = loongflowAdapter.getCircuitBreakerState();
  console.log('LoongFlow circuit breaker:', cbState.state);
}
```

## Monitoring

All workflows provide comprehensive observability:

### Structured Logging
```json
{"level":"info","msg":"Stage 1 completed: Plan created","timestamp":"2025-02-22T10:30:00.000Z","correlation_id":"a1b2c3d4-...","stage":"plan","plan_id":"plan-123","agent_id":"agent-456"}
```

### Metrics
Each workflow returns integration metrics:
- `pes_iterations` - Number of PES iterations performed
- `evolution_generations` - Number of evolutionary generations
- `total_duration_ms` - Total execution time
- `synergy_score` - Combined effectiveness (0-1)
- `pes_time_ms` - Time spent in PES phases
- `evolution_time_ms` - Time spent in evolution phases
- `paradigm_switches` - Number of paradigm switches (adaptive workflow)

### Health Checks
```typescript
// Check adapter health
const loongflowHealth = await loongflowAdapter.healthCheck();
const openevolveHealth = await openevolveAdapter.healthCheck();

console.log('LoongFlow:', loongflowHealth.status);
console.log('OpenEvolve:', openevolveHealth.status);
```

## Best Practices

### 1. Use Appropriate Workflows
- **PESEvolutionWorkflow** - Most problems with clear objectives
- **KnowledgeExtractionWorkflow** - Learning from past executions
- **AdaptiveExecutionWorkflow** - Unknown problem difficulty
- **MultiStageReasoningWorkflow** - Formal verification required

### 2. Set Realistic Timeouts
```typescript
const workflow = createPESEvolutionWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  default_timeout_ms: 10 * 60 * 1000, // 10 minutes
});
```

### 3. Enable Checkpoints for Long Workflows
```typescript
const workflow = createMultiStageReasoningWorkflow({
  loongflowAdapter,
  openevolveAdapter,
  checkpoints_enabled: true,
  checkpoint_path: '/data/workflow-checkpoints',
});
```

### 4. Monitor Events
```typescript
eventBus.subscribe('WorkflowFailed', async (event) => {
  if (event.type === 'WorkflowFailed') {
    console.error('Workflow failed:', event.data.failure_reason);
    // Send alert, log to monitoring system, etc.
  }
});
```

### 5. Handle Failures Gracefully
```typescript
try {
  const result = await workflow.execute(input);
} catch (error) {
  // Use fallback, retry, or cached result
  const fallback = await getFallbackResult(input.problem.id);
  return fallback;
}
```

## Testing

Comprehensive test suite provided:

```bash
# Run all workflow tests
npm test -- glue/orchestration/workflows/__tests__/workflows.test.ts

# Run specific workflow tests
npm test -- --testNamePattern="PESEvolutionWorkflow"
npm test -- --testNamePattern="KnowledgeExtractionWorkflow"
npm test -- --testNamePattern="AdaptiveExecutionWorkflow"
npm test -- --testNamePattern="MultiStageReasoningWorkflow"
```

## Troubleshooting

### Workflow Stuck Running
1. Check adapter health: `await adapter.healthCheck()`
2. Check circuit breaker state: `adapter.getCircuitBreakerState()`
3. Check event logs for failures
4. Increase timeout if needed

### Low Synergy Score
1. Increase PES iterations
2. Increase evolution generations
3. Try adaptive workflow for automatic tuning
4. Check if problem is suitable for hybrid approach

### Memory Issues
1. Reduce `population_size` in evolution config
2. Reduce `concurrency` in PES config
3. Enable checkpoints to reduce in-memory state
4. Process problems in smaller batches

## Contributing

When adding new workflows:

1. Follow existing patterns in `/workflows/`
2. Implement all 6 stages where applicable
3. Publish events for major operations
4. Add comprehensive tests
5. Update this README
6. Follow Federation Constitution laws

## License

See project root LICENSE file.
