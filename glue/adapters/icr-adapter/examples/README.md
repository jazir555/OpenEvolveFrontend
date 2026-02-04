# ICR Memory Integration Examples

This directory contains example code demonstrating the ICR Contextual Mode memory integration with Graphiti.

## Examples

### memory-usage.ts

Comprehensive example showing:
- Setting up the memory agent
- Making memory-enhanced contextual requests
- Retrieving and storing memories
- Learning from session outcomes
- Pattern analysis

## Running the Examples

### Prerequisites

1. Ensure ICR Core is running:
   ```bash
   # Start ICR service
   cd core-projects/icr
   python main.py
   ```

2. Ensure Graphiti is running:
   ```bash
   # Start Graphiti service
   cd glue/adapters/graphiti-adapter
   npm start
   ```

3. Set environment variables:
   ```bash
   export OPENEVOLVE_ICR_API_URL=http://localhost:8000
   export GRAPHITI_API_URL=http://localhost:8080
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password
   export TIMEOUT_MS=30000
   ```

### Run the Example

```bash
# Navigate to ICR adapter directory
cd glue/adapters/icr-adapter

# Build the TypeScript code
npm run build

# Run the example
node dist/examples/memory-usage.js
```

## Example Output

```
========================================
ICR Memory Integration Examples
========================================
Setting up ICR adapter with Graphiti memory integration...
✓ ICR adapter with memory initialized

=== Example 1: Memory-Enhanced Contextual Request ===

Prompt: Refine this React component to improve performance...

Executing with memory enhancement...

✓ Request completed

--- Result ---
Success: true
Iterations: 3
Execution Time: 2450 ms

--- Enriched Context ---
Historical Knowledge Items: 5
Related Patterns: 3
Suggested Approaches: 7
Common Pitfalls: 2
Confidence Score: 0.82

Top Suggested Approaches:
  1. Use useMemo for expensive computations
  2. Implement React.memo for component memoization
  3. Add useCallback for event handlers

--- Learning Result ---
Patterns Learned: 2
Patterns Updated: 1
New Relationships: 4
Insights Extracted: 6

========================================
Examples completed successfully!
========================================
```

## Configuration Examples

### Basic Configuration

```typescript
const memoryAgentConfig: MemoryAgentConfig = {
  graphiti: {
    graphitiAdapter: graphitiAdapter
  }
};
```

### Advanced Configuration

```typescript
const memoryAgentConfig: MemoryAgentConfig = {
  graphiti: {
    graphitiAdapter: graphitiAdapter,
    default_context_window: 10,
    max_historical_results: 50,
    enable_pattern_learning: true,
    enable_cross_session_learning: true
  },
  enable_historical_retrieval: true,
  enable_pattern_learning: true,
  default_context_window: 10,
  min_relevance_score: 0.5,
  max_historical_results: 50,
  learning_threshold: 0.8
};
```

## Common Use Cases

### 1. Quick Memory-Enhanced Request

```typescript
const response = await icrAdapter.createContextualRequestWithMemory(
  "Improve this code",
  { context_window: 5 }
);
```

### 2. Retrieve Historical Patterns

```typescript
const context = await memoryAgent.retrieveHistoricalKnowledge(
  "React optimization",
  10
);

console.log('Patterns:', context.related_patterns);
```

### 3. Store Custom Insights

```typescript
await memoryAgent.storeRefinementInsights(
  {
    session_id: 'session-123',
    mode: 'contextual',
    iterations: [...],
    total_iterations: 3,
    successful_iterations: 2,
    failed_iterations: 1,
    ...
  },
  'session-123'
);
```

### 4. Learn from Outcomes

```typescript
const result = await memoryAgent.learnFromSession(
  contextualSession,
  [sessionOutcome1, sessionOutcome2]
);
```

## Troubleshooting

### "Memory agent not configured"

Ensure you pass `memoryAgentConfig` when creating the ICR adapter:

```typescript
const icrAdapter = new ICRAdapter({
  memoryAgentConfig: memoryAgentConfig
});
```

### "No historical knowledge returned"

- Verify Graphiti has data: Check that previous sessions have been stored
- Increase context window: Try `context_window: 20`
- Check correlation logs: Review `enriched_context` in response

### "Graphiti connection failed"

- Verify Graphiti is running: Check `GRAPHITI_API_URL`
- Check Neo4j connection: Verify `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Run probe script: `bash probes/check_graphiti_memory.sh`

## Additional Resources

- [Memory Integration Documentation](../MEMORY_INTEGRATION.md)
- [ICR Adapter README](../README.md)
- [Graphiti Adapter Documentation](../../graphiti-adapter/README.md)
- [Federation Constitution](../../../../../CLAUDE.md)
