# OpenEvolve Integration Library

A **generic, reusable library** for integrating OpenEvolve components into any frontend application.

## Purpose

This library provides a **unified API** for all OpenEvolve functionality:
- LeanAide (formal verification, MCTS, math queries)
- Evolution (evolutionary and adversarial algorithms)
- Knowledge Engine (knowledge graphs, extraction)
- Maker Engine (tool creation, execution)
- crewai (delegation, orchestration)
- Decomposition (problem breakdown)
- Verification (solution verification)
- Assembly (solution integration)
- Solution (generation and refinement)

## Installation

```bash
npm install @openevolve/integration-library
```

The library also provides optional React and Zustand integrations. To use them, ensure you have the peer dependencies installed:

```bash
npm install react zustand
```

## Architecture

```
@openevolve/integration-library
├── api/
│   ├── client.ts      ← Unified API client
│   ├── backend.ts     ← Backend communication
│   ├── errors.ts      ← Custom error classes
│   └── types.ts       ← Core API types
├── integrations/
│   ├── all-integrations.ts ← All integration adapters
│   └── base.ts             ← Base integration logic
├── react/
│   └── index.ts       ← React hooks and provider
├── store/
│   └── index.ts       ← Zustand store factory
├── testing/
│   └── index.ts       ← Mocking utilities for tests
└── utils/
    └── helpers.ts     ← Utility functions
```

## Quick Start

```typescript
import { OpenEvolveClient, IntegrationName } from '@openevolve/integration-library';

// Initialize the client
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  apiKey: process.env.OPENEVOLVE_API_KEY
});

// Use any integration
const decomposition = await client.integrations.decomposition.execute({
  operation: 'decompose',
  input: {
    problem: "Solve X",
    strategy: "hierarchical"
  }
});

const solution = await client.integrations.solution.execute({
  operation: 'generate',
  input: {
    problem: decomposition.sub_problems[0],
    strategy: "maker"
  }
});

const verification = await client.integrations.verification.execute({
  operation: 'verify',
  input: {
    solution: solution,
    requirements: ["quality", "correctness"]
  }
});
```

## State Management (Zustand)

The library provides a Zustand store factory for easy state management:

```typescript
import { createOpenEvolveStore } from '@openevolve/integration-library/store';

const useStore = createOpenEvolveStore();

function MyComponent() {
  const initialize = useStore(state => state.initialize);
  const execute = useStore(state => state.execute);
  const results = useStore(state => state.results);
  const loading = useStore(state => state.loading);

  // Initialize once
  useEffect(() => {
    initialize(client);
  }, []);

  return (
    <button onClick={() => execute('decomposition', { operation: 'decompose', ... })}>
      {loading['decomposition'] ? 'Running...' : 'Run Decomposition'}
    </button>
  );
}
```

## React Hooks

For React applications, use the provided hooks and provider:

```tsx
import { OpenEvolveProvider, useDecomposition } from '@openevolve/integration-library/react';

function App() {
  return (
    <OpenEvolveProvider client={client}>
      <MyComponent />
    </OpenEvolveProvider>
  );
}

function MyComponent() {
  const { data, error, loading, execute } = useDecomposition();

  return (
    <>
      <button onClick={() => execute({ operation: 'decompose', input: { ... } })}>
        Decompose Problem
      </button>
      {loading && <div>Loading...</div>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </>
  );
}
```

## Streaming Support

Integrations support streaming for real-time updates via WebSockets:

```typescript
const result = await client.executeStream(
  IntegrationName.DECOMPOSITION,
  { 
    operation: 'decompose', 
    input: { problem: "Solve X" } 
  },
  (update) => {
    console.log(`Progress: ${update.progress}%`);
    console.log(`Message: ${update.message}`);
  }
);
```

## Error Handling

The library provides structured, robust error handling across all layers. All errors are guaranteed to be returned as a subclass of `IntegrationError`.

```typescript
import { 
  IntegrationError, 
  ValidationError, 
  TimeoutError,
  isRetryableError,
  isCriticalError
} from '@openevolve/integration-library';

try {
  const result = await client.execute(...);
} catch (error) {
  // error is always an IntegrationError
  console.log(`Error in ${error.integration} (code: ${error.code})`);
  
  if (error instanceof ValidationError) {
    console.error('Input validation failed:', error.errors);
  } else if (error instanceof TimeoutError) {
    console.error('Request timed out after', error.details.timeout, 'ms');
  }

  // Use utility functions to decide on next steps
  if (isRetryableError(error)) {
    // maybe try again?
  }
}
```

Key robustness features:
- **Consistent Typing**: All public methods and hooks return typed errors.
- **Auto-mapping**: Backend and network failures are automatically mapped to descriptive error classes.
- **Initialization Safety**: The client handles failure to connect to backends or WebSockets gracefully.
- **Middleware Safety**: Errors in middleware are captured and wrapped to maintain consistency.
- **React/Store Safety**: Integrated hooks and state management handle race conditions and unmounts automatically.


## Testing

Mock the OpenEvolve client in your tests:

```typescript
import { createMockClient } from '@openevolve/integration-library/testing';

const mockClient = createMockClient({
  decomposition: { success: true, plan: "mock-plan" }
});

const result = await mockClient.integrations.decomposition.execute({ ... });
expect(result.plan).toBe("mock-plan");
```

## Middleware

You can intercept and modify execution using middleware:

```typescript
const loggingMiddleware: Middleware = async (context, next) => {
  console.log(`Starting ${context.integration}...`);
  const result = await next();
  console.log(`Finished ${context.integration}.`);
  return result;
};

const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  middleware: [loggingMiddleware]
});
```

## License

MIT License - see LICENSE file for details