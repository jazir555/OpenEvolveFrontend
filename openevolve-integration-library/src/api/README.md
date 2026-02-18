# OpenEvolve Unified API Client

The **core API client** that provides unified access to all OpenEvolve integrations. This is the heart of the integration library.

## Features

- **Unified Interface**: One client for all integrations
- **Type Safety**: Full TypeScript support with generic methods
- **Error Handling**: Custom error classes for all scenarios
- **Progress Tracking**: Stream updates during execution
- **Batch Support**: Execute multiple requests concurrently
- **Health Monitoring**: Check backend and integration status
- **Connection Management**: Automatic WebSocket connection handling
- **Retry Logic**: Automatic retry with exponential backoff
- **Request Metrics**: Track execution time and performance

## Installation

```bash
npm install @openevolve/integration-library
```

## Quick Start

```typescript
import { OpenEvolveClient, IntegrationName } from '@openevolve/integration-library';

// Create client
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  timeout: 30000,
  retryAttempts: 3,
  enableWebSocket: true,
  debug: true
});

// Execute an integration
const result = await client.execute(
  IntegrationName.LEANAIDE,
  {
    type: 'theorem_proving',
    statement: 'theorem my_theorem : ∀ x, x + 0 = x'
  }
);

console.log(result);
```

## Basic Usage

### 1. Client Initialization

```typescript
import { OpenEvolveClient } from '@openevolve/integration-library';

const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',        // Backend URL
  timeout: 30000,                            // Request timeout (ms)
  retryAttempts: 3,                          // Number of retry attempts
  enableWebSocket: true,                     // Enable WebSocket
  debug: false,                              // Enable debug logging
  apiKey: 'your-api-key',                    // Optional API key
  headers: {                                 // Optional custom headers
    'X-Custom-Header': 'value'
  }
});
```

### 2. Simple Execution

```typescript
import { IntegrationName } from '@openevolve/integration-library';

// Execute LeanAide integration
const proof = await client.execute(
  IntegrationName.LEANAIDE,
  {
    problem: 'Prove that sqrt(2) is irrational'
  }
);

// Execute Evolution integration
const evolved = await client.execute(
  IntegrationName.EVOLUTION,
  {
    prompt: 'Evolve a sorting algorithm',
    iterations: 100
  }
);

// Execute Maker integration
const invention = await client.execute(
  IntegrationName.MAKER,
  {
    prompt: 'Create a new type of renewable energy device',
    domain: 'energy'
  }
);
```

### 3. Execution with Options

```typescript
const result = await client.execute(
  IntegrationName.KNOWLEDGE,
  {
    query: 'What are the relationships between AI and knowledge graphs?'
  },
  {
    timeout: 60000,                           // Custom timeout
    retries: 5,                               // Custom retry count
    onProgress: (update) => {                 // Progress callback
      console.log(`Progress: ${update.progress}%`);
      console.log(`Stage: ${update.stage}`);
      console.log(`Message: ${update.message}`);
    },
    onComplete: (result) => {                 // Completion callback
      console.log('Execution complete!', result);
    },
    onError: (error) => {                     // Error callback
      console.error('Execution failed:', error);
    }
  }
);
```

### 4. Streaming Execution

```typescript
const result = await client.executeStream(
  IntegrationName.EVOLUTION,
  {
    prompt: 'Evolve a solution for the traveling salesman problem'
  },
  (update) => {
    console.log(`[${update.timestamp}] ${update.progress}%: ${update.message}`);
    // Access additional data
    if (update.data) {
      console.log('Current generation:', update.data.generation);
      console.log('Best fitness:', update.data.fitness);
    }
  }
);
```

### 5. Batch Execution

```typescript
const results = await client.executeBatch([
  {
    integration: IntegrationName.LEANAIDE,
    id: 'req1',
    inputs: { problem: 'Problem 1' }
  },
  {
    integration: IntegrationName.MAKER,
    id: 'req2',
    inputs: { prompt: 'Create something new' }
  },
  {
    integration: IntegrationName.EVOLUTION,
    id: 'req3',
    inputs: { prompt: 'Evolve this' }
  }
]);

// Process results
results.forEach(result => {
  if (result.success) {
    console.log(`${result.id} succeeded:`, result.result);
  } else {
    console.error(`${result.id} failed:`, result.error);
  }
});
```

### 6. Health Check

```typescript
const health = await client.healthCheck();

console.log('Overall status:', health.status);
console.log('Backend online:', health.backend.online);
console.log('Backend version:', health.backend.version);
console.log('Active connections:', health.backend.activeConnections);

// Check integration status
Object.entries(health.integrations).forEach(([name, status]) => {
  console.log(`${name}: ${status.status} (${status.responseTime}ms)`);
});
```

### 7. Connection Management

```typescript
// Manually connect
await client.connect();

// Check connection status
if (client.isConnected()) {
  console.log('Connected to backend');
}

// Get connection state
const state = client.getConnectionState();
console.log('Connection state:', state);
// Possible values: 'disconnected', 'connecting', 'connected', 'reconnecting', 'disconnecting'

// Disconnect
await client.disconnect();
```

## Advanced Usage

### Integration-Specific Access

```typescript
// Access specific integration
const leanaide = client.integrations.leanaide;
const evolution = client.integrations.evolution;
const knowledge = client.integrations.knowledge;
const maker = client.integrations.maker;
const crewai = client.integrations.crewai;
const decomposition = client.integrations.decomposition;
const verification = client.integrations.verification;
const assembly = client.integrations.assembly;

// Use integration directly
const proof = await leanaide.execute({
  type: 'theorem_proving',
  statement: '...'
});

const health = await leanaide.healthCheck();
const validation = await leanaide.validate(inputs);
```

### Request Metrics

```typescript
// Execute request
const result = await client.execute(
  IntegrationName.EVOLUTION,
  { prompt: '...' }
);

// Get metrics (if you have the execution ID)
const metrics = client.getMetrics(executionId);
console.log('Execution time:', metrics.duration, 'ms');
console.log('Success:', metrics.success);
console.log('Retries:', metrics.retries);

// Get all metrics
const allMetrics = client.getAllMetrics();

// Clear metrics
client.clearMetrics();
```

### Retry Configuration

```typescript
// Update retry configuration
client.updateRetryConfig({
  maxAttempts: 5,           // Maximum retry attempts
  initialDelay: 2000,       // Initial delay (ms)
  maxDelay: 30000,          // Maximum delay (ms)
  backoffMultiplier: 2,     // Exponential backoff multiplier
  retryOn4xx: false,        // Retry on 4xx errors
  retryOn5xx: true,         // Retry on 5xx errors
  retryableStatusCodes: [408, 429, 500, 502, 503, 504]
});
```

### Request Cancellation

```typescript
// Create abort controller
const controller = new AbortController();

// Execute with signal
const promise = client.execute(
  IntegrationName.EVOLUTION,
  { prompt: 'Long running task...' },
  { signal: controller.signal }
);

// Cancel after 5 seconds
setTimeout(() => {
  controller.abort();
}, 5000);

try {
  const result = await promise;
} catch (error) {
  if (error.code === 'CANCELLATION_ERROR') {
    console.log('Request was cancelled');
  }
}
```

## Error Handling

### Error Types

```typescript
import {
  IntegrationError,
  ConnectionError,
  AuthenticationError,
  AuthorizationError,
  ValidationError,
  ExecutionError,
  TimeoutError,
  RateLimitError,
  NotFoundError,
  ConfigurationError,
  NetworkError,
  CancellationError,
  ParseError,
  RetryError
} from '@openevolve/integration-library';

try {
  const result = await client.execute(...);
} catch (error) {
  if (error instanceof ValidationError) {
    // Handle validation errors
    console.error('Validation failed:', error.getErrorMessages());
    console.error('Field errors:', error.errors);
  } else if (error instanceof AuthenticationError) {
    // Handle authentication errors
    console.error('Authentication failed');
  } else if (error instanceof TimeoutError) {
    // Handle timeout errors
    console.error('Request timed out');
  } else if (error instanceof RateLimitError) {
    // Handle rate limiting
    const retryAfter = error.getRetryAfterMs();
    console.log(`Retry after ${retryAfter}ms`);
  } else if (error instanceof IntegrationError) {
    // Handle all integration errors
    console.error(`[${error.code}] ${error.message}`);
    console.error('Integration:', error.integration);
    console.error('Details:', error.details);
  }
}
```

### Error Utilities

```typescript
import {
  isRetryableError,
  isCriticalError,
  createIntegrationError
} from '@openevolve/integration-library';

// Check if error is retryable
if (isRetryableError(error)) {
  console.log('This error can be retried');
}

// Check if error is critical (should not retry)
if (isCriticalError(error)) {
  console.log('This is a critical error');
}
```

## Type Safety

### Generic Type Parameters

```typescript
// Define your input and output types
interface LeanAideInput {
  type: 'theorem_proving' | 'tactic_generation';
  statement: string;
  options?: {
    timeout?: number;
    maxDepth?: number;
  };
}

interface LeanAideOutput {
  proof: string;
  tactics: string[];
  metadata: {
    executionTime: number;
    proofLength: number;
  };
}

// Execute with full type safety
const result = await client.execute<
  typeof IntegrationName.LEANAIDE,
  LeanAideInput,
  LeanAideOutput
>(
  IntegrationName.LEANAIDE,
  {
    type: 'theorem_proving',
    statement: '...'
  }
);

// TypeScript knows the types
console.log(result.proof);           // string
console.log(result.tactics);         // string[]
console.log(result.metadata);        // { executionTime: number, proofLength: number }
```

## Examples

### Example 1: Mathematical Proof Generation

```typescript
import { OpenEvolveClient, IntegrationName } from '@openevolve/integration-library';

const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  debug: true
});

const proof = await client.execute(
  IntegrationName.LEANAIDE,
  {
    type: 'theorem_proving',
    statement: 'theorem pythagorean : ∀ a b c: ℝ, a² + b² = c² → is_right_triangle a b c'
  },
  {
    onProgress: (update) => {
      console.log(`Proof generation: ${update.progress}%`);
      console.log(`Current tactic: ${update.data?.currentTactic}`);
    }
  }
);

console.log('Generated proof:', proof.proof);
console.log('Tactics used:', proof.tactics);
```

### Example 2: Evolutionary Algorithm

```typescript
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000'
});

const evolved = await client.executeStream(
  IntegrationName.EVOLUTION,
  {
    prompt: 'Evolve an optimal solution for the knapsack problem',
    parameters: {
      populationSize: 100,
      generations: 50,
      mutationRate: 0.1
    }
  },
  (update) => {
    console.log(`Generation ${update.data.generation}:`);
    console.log(`  Best fitness: ${update.data.bestFitness}`);
    console.log(`  Average fitness: ${update.data.avgFitness}`);
    console.log(`  Progress: ${update.progress}%`);
  }
);

console.log('Final solution:', evolved.solution);
console.log('Fitness:', evolved.fitness);
```

### Example 3: Knowledge Graph Query

```typescript
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000'
});

const knowledge = await client.execute(
  IntegrationName.KNOWLEDGE,
  {
    query: 'Find all relationships between AI and machine learning',
    options: {
      depth: 3,
      includeProperties: true
    }
  }
);

console.log('Entities found:', knowledge.entities);
console.log('Relationships:', knowledge.relationships);
console.log('Paths:', knowledge.paths);
```

### Example 4: Invention Generation

```typescript
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000'
});

const invention = await client.execute(
  IntegrationName.MAKER,
  {
    prompt: 'Create a novel solution for carbon capture',
    domain: 'environmental technology',
    constraints: {
      cost: 'low',
      scalability: 'high',
      efficiency: 'high'
    }
  },
  {
    onProgress: (update) => {
      console.log(`Invention generation: ${update.progress}%`);
      console.log(`Stage: ${update.stage}`);
    }
  }
);

console.log('Generated invention:', invention.title);
console.log('Description:', invention.description);
console.log('Feasibility:', invention.feasibility);
```

### Example 5: Batch Processing

```typescript
const problems = [
  'Prove theorem 1',
  'Prove theorem 2',
  'Prove theorem 3'
];

const results = await client.executeBatch(
  problems.map((problem, index) => ({
    integration: IntegrationName.LEANAIDE,
    id: `problem-${index}`,
    inputs: {
      type: 'theorem_proving',
      statement: problem
    }
  }))
);

// Process results
results.forEach(result => {
  if (result.success) {
    console.log(`${result.id}: SUCCESS (${result.executionTime}ms)`);
    console.log('Proof:', result.result.proof);
  } else {
    console.error(`${result.id}: FAILED`);
    console.error('Error:', result.error.message);
  }
});

// Summary
const successCount = results.filter(r => r.success).length;
const avgTime = results.reduce((sum, r) => sum + r.executionTime, 0) / results.length;

console.log(`Success rate: ${successCount}/${results.length}`);
console.log(`Average time: ${avgTime}ms`);
```

## API Reference

### OpenEvolveClient

#### Constructor

```typescript
constructor(config: ClientConfig)
```

#### Methods

- `execute<TIntegration, TInputs, TResult>(integration, inputs, options?): Promise<TResult>`
- `executeStream<TInputs, TResult>(integration, inputs, onProgress, options?): Promise<TResult>`
- `executeBatch<TInputs, TResult>(requests: BatchRequest<TInputs>[]): Promise<BatchResult<TResult>[]>`
- `healthCheck(): Promise<HealthStatus>`
- `connect(): Promise<void>`
- `disconnect(): Promise<void>`
- `isConnected(): boolean`
- `getConnectionState(): ConnectionState`
- `getMetrics(executionId: string): RequestMetrics | null`
- `getAllMetrics(): Map<string, RequestMetrics>`
- `clearMetrics(): void`
- `updateRetryConfig(config: Partial<RetryConfig>): void`

#### Properties

- `integrations: IntegrationRegistry` - Access to all integration-specific methods

### ClientConfig

```typescript
interface ClientConfig {
  baseUrl: string;              // Backend URL
  timeout?: number;             // Request timeout (default: 30000)
  retryAttempts?: number;       // Retry attempts (default: 3)
  enableWebSocket?: boolean;    // Enable WebSocket (default: true)
  debug?: boolean;              // Debug logging (default: false)
  apiKey?: string;              // API key for authentication
  headers?: Record<string, string>;  // Custom headers
}
```

### ExecutionOptions

```typescript
interface ExecutionOptions {
  timeout?: number;                                    // Override timeout
  retries?: number;                                    // Override retries
  stream?: boolean;                                    // Enable streaming
  onProgress?: (update: ProgressUpdate) => void;       // Progress callback
  onComplete?: (result: any) => void;                  // Completion callback
  onError?: (error: Error) => void;                    // Error callback
  signal?: AbortSignal;                                // Cancellation signal
  metadata?: Record<string, any>;                      // Custom metadata
}
```

## Best Practices

1. **Always handle errors** - Use try-catch blocks and check error types
2. **Use progress callbacks** - Provide feedback for long-running operations
3. **Set appropriate timeouts** - Adjust based on expected execution time
4. **Monitor connection state** - Check connection status before critical operations
5. **Use batch operations** - Execute multiple requests concurrently when possible
6. **Track metrics** - Use metrics to monitor performance and identify issues
7. **Implement cancellation** - Use AbortSignal for long-running operations
8. **Configure retries appropriately** - Adjust retry settings based on your use case

## Troubleshooting

### Connection Issues

```typescript
// Check backend status
const health = await client.healthCheck();
if (!health.backend.online) {
  console.error('Backend is offline');
}

// Reconnect
await client.disconnect();
await client.connect();
```

### Timeout Issues

```typescript
// Increase timeout for long operations
const result = await client.execute(
  IntegrationName.EVOLUTION,
  { prompt: '...' },
  { timeout: 120000 }  // 2 minutes
);
```

### Rate Limiting

```typescript
// Handle rate limit errors
try {
  const result = await client.execute(...);
} catch (error) {
  if (error instanceof RateLimitError) {
    const retryAfter = error.getRetryAfterMs();
    await new Promise(resolve => setTimeout(resolve, retryAfter));
    // Retry the request
  }
}
```

## Support

For issues, questions, or contributions, please visit:
- GitHub: https://github.com/openevolve/integration-library
- Documentation: https://docs.openevolve.org
