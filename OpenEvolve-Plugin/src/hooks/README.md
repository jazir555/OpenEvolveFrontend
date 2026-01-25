# OpenEvolve Custom React Hooks

This directory contains custom React hooks for the OpenEvolve plugin, providing interfaces to various backend services and state management.

## Available Hooks

### 1. useEvolution
**Location**: `useEvolution.ts`
**Purpose**: Manages genetic algorithm evolution workflows

#### Usage
```typescript
import { useEvolution } from '@/hooks';

function EvolutionComponent() {
  const {
    data,
    loading,
    error,
    progress,
    execute,
    getStatus,
    getResults,
    cancel,
    pause,
    resume,
    reset
  } = useEvolution();

  const handleStart = async () => {
    await execute({
      content: 'Optimize this code',
      mode: 'standard',
      parameters: {
        max_iterations: 100,
        population_size: 50,
        temperature: 0.7,
        top_p: 0.9
      },
      models: [{ provider: 'openai', model: 'gpt-4', api_key: 'xxx' }]
    });
  };

  return (
    <div>
      <button onClick={handleStart} disabled={loading}>
        Start Evolution
      </button>
      <div>Progress: {progress}%</div>
      {error && <div>Error: {error.message}</div>}
    </div>
  );
}
```

#### Features
- Real-time WebSocket updates
- Progress tracking (0-100%)
- Pause/resume/cancel support
- Automatic retry on connection loss
- Zustand store integration

---

### 2. useAdversarial
**Location**: `useAdversarial.ts`
**Purpose**: Manages red team vs blue team adversarial testing

#### Usage
```typescript
import { useAdversarial } from '@/hooks';

function AdversarialComponent() {
  const {
    data,
    loading,
    error,
    progress,
    currentRound,
    totalRounds,
    execute,
    getStatus,
    getResults,
    cancel,
    approvePatch,
    reset
  } = useAdversarial();

  const handleStart = async () => {
    await execute({
      content: 'Test this code',
      attack_modes: ['prompt_injection', 'jailbreak'],
      parameters: {
        num_rounds: 10,
        red_team_models: [{ provider: 'openai', model: 'gpt-4' }],
        blue_team_models: [{ provider: 'openai', model: 'gpt-4' }]
      }
    });
  };

  const handleApprovePatch = async (round: number) => {
    await approvePatch(round, true, 'Good fix');
  };

  return (
    <div>
      <button onClick={handleStart} disabled={loading}>
        Start Adversarial Test
      </button>
      <div>Round: {currentRound}/{totalRounds}</div>
      <div>Progress: {progress}%</div>
    </div>
  );
}
```

#### Features
- Multi-round testing
- Real-time attack/patch updates
- Patch approval workflow
- Vulnerability tracking
- WebSocket real-time updates

---

### 3. useDecomposition
**Location**: `useDecomposition.ts`
**Purpose**: Breaks down complex problems into manageable subtasks

#### Usage
```typescript
import { useDecomposition } from '@/hooks';

function DecompositionComponent() {
  const {
    data,
    loading,
    error,
    progress,
    execute,
    getStatus,
    getResults,
    cancel,
    updateTaskStatus,
    getExecutionPlan,
    getTaskById,
    getTasksByStatus,
    reset
  } = useDecomposition();

  const handleDecompose = async () => {
    const result = await execute({
      problem_statement: 'Build a web application',
      decomposition_strategy: 'hierarchical',
      max_depth: 3,
      include_dependencies: true,
      include_subtasks: true
    });

    console.log('Tasks:', result?.tasks);
    console.log('Execution Plan:', result?.execution_plan);
  };

  const handleUpdateTask = async (taskId: string) => {
    await updateTaskStatus(taskId, 'completed');
  };

  return (
    <div>
      <button onClick={handleDecompose} disabled={loading}>
        Decompose Problem
      </button>
      <div>Progress: {progress}%</div>
    </div>
  );
}
```

#### Features
- Multiple decomposition strategies
- Task dependency management
- Execution plan generation
- Task status tracking
- Hierarchical tree structure

---

### 4. useKnowledgeEngine
**Location**: `useKnowledgeEngine.ts`
**Purpose**: Manages knowledge graph and artifact operations

#### Usage
```typescript
import { useKnowledgeEngine } from '@/hooks';

function KnowledgeComponent() {
  const {
    artifacts,
    graphData,
    loading,
    error,
    progress,
    query,
    ingest,
    getGraph,
    getArtifacts,
    getArtifact,
    updateArtifact,
    deleteArtifact,
    getRelationships,
    semanticSearch,
    cancel,
    reset
  } = useKnowledgeEngine();

  const handleIngest = async () => {
    await ingest({
      content: 'Some knowledge',
      title: 'Document Title',
      language: 'python',
      tags: ['ml', 'ai'],
      metadata: { author: 'User' }
    });
  };

  const handleQuery = async () => {
    const results = await query({
      query: 'machine learning',
      context: 'neural networks',
      limit: 10,
      threshold: 0.7
    });

    console.log('Query Results:', results);
  };

  const handleSearch = async () => {
    const results = await semanticSearch('neural networks', 10);
    console.log('Search Results:', results);
  };

  return (
    <div>
      <button onClick={handleIngest} disabled={loading}>
        Ingest Knowledge
      </button>
      <button onClick={handleQuery} disabled={loading}>
        Query Knowledge Base
      </button>
      <div>Artifacts: {artifacts.length}</div>
      <div>Progress: {progress}%</div>
    </div>
  );
}
```

#### Features
- Semantic search
- Knowledge graph visualization
- Artifact CRUD operations
- Relationship mapping
- Progress tracking for ingestion

---

### 5. useLeanAIDE
**Location**: `useLeanAIDE.ts`
**Purpose**: Manages Lean 4 formal verification and theorem proving

#### Usage
```typescript
import { useLeanAIDE } from '@/hooks';

function LeanAIDEComponent() {
  const {
    theorem,
    proofAttempt,
    modelConfig,
    data,
    loading,
    error,
    progress,
    status,
    execute,
    verify,
    getModels,
    getStatus,
    getResults,
    cancel,
    reset,
    updateModelConfig,
    runBenchmark,
    getBenchmarkResults
  } = useLeanAIDE();

  const handleGenerate = async () => {
    const result = await execute({
      theorem: 'forall n : Nat, n + 0 = n',
      proof_attempt: '',
      model: 'gpt-4',
      temperature: 0.7
    });

    console.log('Generated Proof:', result?.code);
  };

  const handleVerify = async () => {
    const result = await verify({
      code: 'theorem add_zero : forall n : Nat, n + 0 = n := by simp',
      timeout: 30000
    });

    console.log('Verification Result:', result);
  };

  const handleBenchmark = async () => {
    const benchmarkId = await runBenchmark(
      [{ theorem: 'test theorem' }],
      'gpt-4',
      'lean4'
    );

    if (benchmarkId) {
      const results = await getBenchmarkResults(benchmarkId);
      console.log('Benchmark Results:', results);
    }
  };

  return (
    <div>
      <button onClick={handleGenerate} disabled={loading}>
        Generate Proof
      </button>
      <button onClick={handleVerify} disabled={loading}>
        Verify Proof
      </button>
      <div>Status: {status}</div>
      <div>Progress: {progress}%</div>
    </div>
  );
}
```

#### Features
- Lean 4 proof generation
- Formal verification
- Benchmarking
- Model configuration
- Proof status tracking
- Error handling with detailed messages

---

### 6. useHephaestus
**Location**: `useHephaestus.ts`
**Purpose**: Manages code generation, review, and optimization

#### Usage
```typescript
import { useHephaestus } from '@/hooks';

function HephaestusComponent() {
  const {
    data,
    loading,
    error,
    progress,
    currentOperation,
    execute,
    review,
    optimize,
    getStatus,
    getResults,
    cancel,
    reset,
    getSupportedLanguages,
    getTemplates,
    applyFix,
    getCodeMetrics
  } = useHephaestus();

  const handleGenerate = async () => {
    const result = await execute({
      requirement: 'Create a REST API server',
      language: 'python',
      framework: 'fastapi',
      include_tests: true,
      include_comments: true,
      style_guide: 'pep8'
    });

    console.log('Generated Code:', result?.code);
  };

  const handleReview = async () => {
    const result = await review(
      'def add(a, b): return a + b',
      'python'
    );

    console.log('Review Result:', result);
    console.log('Issues:', result?.issues);
    console.log('Overall Score:', result?.overall_score);
  };

  const handleOptimize = async () => {
    const result = await optimize(
      'def slow_function(): ...',
      'python',
      ['performance', 'readability']
    );

    console.log('Optimized Code:', result?.optimized_code);
    console.log('Performance Gain:', result?.performance_gain);
  };

  const handleGetMetrics = async () => {
    const metrics = await getCodeMetrics('some code', 'python');
    console.log('Code Metrics:', metrics);
  };

  return (
    <div>
      <button onClick={handleGenerate} disabled={loading}>
        Generate Code
      </button>
      <button onClick={handleReview} disabled={loading}>
        Review Code
      </button>
      <button onClick={handleOptimize} disabled={loading}>
        Optimize Code
      </button>
      <div>Operation: {currentOperation}</div>
      <div>Progress: {progress}%</div>
    </div>
  );
}
```

#### Features
- Multi-language code generation
- Automated code review
- Code optimization
- Metrics analysis
- Template-based generation
- Fix application

---

## Common Patterns

### Error Handling
All hooks follow a consistent error handling pattern:

```typescript
const { error, loading, execute } = useHook();

if (error) {
  return <div>Error: {error.message}</div>;
}

if (loading) {
  return <div>Loading...</div>;
}
```

### Progress Tracking
All hooks with long-running operations include progress tracking:

```typescript
const { progress, execute } = useHook();

return (
  <div>
    <ProgressBar value={progress} max={100} />
  </div>
);
```

### Cancellation
All operations can be cancelled:

```typescript
const { cancel, execute } = useHook();

const handleStart = async () => {
  await execute(params);
};

const handleCancel = () => {
  cancel();
};
```

### State Reset
All hooks support resetting their state:

```typescript
const { reset } = useHook();

const handleReset = () => {
  reset();
};
```

## API Endpoints

Each hook communicates with specific backend endpoints:

| Hook | Base Endpoint |
|------|---------------|
| useEvolution | `/api/v1/evolution` |
| useAdversarial | `/api/v1/adversarial` |
| useDecomposition | `/api/v1/decomposition` |
| useKnowledgeEngine | `/api/v1/knowledge` |
| useLeanAIDE | `/api/v1/leanaide` |
| useHephaestus | `/api/v1/hephaestus` |

## TypeScript Support

All hooks are fully typed with TypeScript. Exported types include:

- `EvolutionParams`, `EvolutionState`
- `AdversarialParams`, `AdversarialState`
- `DecompositionParams`, `DecompositionResult`
- `KnowledgeQueryParams`, `KnowledgeIngestParams`
- `LeanProofParams`, `VerificationParams`
- `CodeGenerationParams`, `CodeReviewResult`

## WebSocket Support

Hooks that support real-time updates use WebSocket connections:

- `useEvolution` - Evolution progress updates
- `useAdversarial` - Attack/patch generation updates

WebSocket connections are automatically managed:
- Auto-reconnect on disconnect
- Heartbeat monitoring
- Cleanup on unmount

## Best Practices

1. **Always check loading state** before allowing user actions
2. **Handle errors gracefully** with user-friendly messages
3. **Use cancel functionality** to clean up long-running operations
4. **Reset state** when components unmount or when switching contexts
5. **Leverage progress tracking** for better UX
6. **Store important results** in parent component state if needed
7. **Use appropriate hooks** for each use case (don't use `useHephaestus` for knowledge queries)

## Testing

Mock these hooks in your tests:

```typescript
import { renderHook } from '@testing-library/react';
import { useEvolution } from '@/hooks';

vi.mock('@/hooks', () => ({
  useEvolution: () => ({
    data: mockData,
    loading: false,
    error: null,
    progress: 100,
    execute: vi.fn(),
    cancel: vi.fn(),
    reset: vi.fn()
  })
}));
```

## Contributing

When adding new hooks:
1. Follow the established patterns
2. Include TypeScript types
3. Add error handling
4. Support cancellation
5. Include progress tracking
6. Add to this documentation
7. Export from `index.ts`

## License

Part of the OpenEvolve plugin ecosystem.
