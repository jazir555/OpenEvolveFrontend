# OpenEvolve Integration Adapters - Complete Guide

## Overview

The OpenEvolve Integration Library provides a high-level, type-safe SDK for interacting with OpenEvolve backend components. This guide covers the architectural principles and usage of every available adapter.

## Core Concepts

### Unified Execution
Every integration implements a standard `execute` method that takes an `operation` name and an `input` payload. This consistency allows for generic batching and middleware processing.

### Middleware Pipeline
The execution flow can be intercepted. Middlewares can log data, cache results, or modify inputs before they reach the backend.

---

## Integration Adapters

### 1. LeanAide Integration
**Target**: Formal mathematics and verification.

```typescript
const result = await client.integrations.leanaide.execute({
  operation: 'prove',
  input: {
    theorem: 'forall n : Nat, n + 0 = n',
    strategy: 'mcts'
  }
});
```

### 2. Evolution Integration
**Target**: Optimization and adversarial evolution.

```typescript
const result = await client.integrations.evolution.execute({
  operation: 'evolution',
  config: {
    initial_population: [...],
    fitness_function: 'max_profit'
  }
});
```

### 3. Knowledge Integration
**Target**: Knowledge graph extraction and querying.

```typescript
const result = await client.integrations.knowledge.execute({
  operation: 'extract',
  input: { document: '...' }
});
```

### 4. Maker Integration
**Target**: Dynamic tool creation and runtime execution.

```typescript
const result = await client.integrations.maker.execute({
  operation: 'create',
  input: { name: 'calculator', logic: '...' }
});
```

### 5. Decomposition Integration
**Target**: Problem breakdown and dependency analysis.

```typescript
const result = await client.integrations.decomposition.execute({
  operation: 'decompose',
  input: { problem: 'Build a rocket' }
});
```

---

## Advanced Features

### WebSocket Streaming
Long-running tasks like proof generation or evolutionary loops support progress updates:

```typescript
await client.executeStream(
  'leanaide',
  inputs,
  (update) => console.log(`Progress: ${update.progress}%`)
);
```

### Robust Error Handling
The library distinguishes between validation errors (client-side) and execution errors (server-side).

```typescript
try {
  await client.execute(...);
} catch (error) {
  if (error instanceof ValidationError) {
    // Check error.errors for field-level details
  }
}
```

### React Hooks
Seamlessly integrate into React components using specialized hooks:

```tsx
function MyComponent() {
  const { data, loading, execute } = useDecomposition();
  return <button onClick={() => execute({...})}>Run</button>;
}
```