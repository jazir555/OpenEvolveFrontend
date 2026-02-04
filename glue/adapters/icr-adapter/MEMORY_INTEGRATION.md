# ICR Contextual Mode Memory Integration

## Overview

This document describes the integration between ICR's Contextual Mode and Graphiti for persistent long-term memory storage and retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ICR Adapter                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         EnhancedICRMemoryAgent                              │ │
│  │  - Historical Knowledge Retrieval                           │ │
│  │  - Pattern Learning                                        │ │
│  │  - Context Enrichment                                      │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         GraphitiMemoryManager                              │ │
│  │  - Episode Storage                                         │ │
│  │  - Knowledge Retrieval                                     │ │
│  │  - Graph Building                                          │ │
│  └────────────────────────┬───────────────────────────────────┘ │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Graphiti Adapter                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Canonical Schema Layer                              │ │
│  └────────────────────────┬───────────────────────────────────┘ │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Neo4j (Graph Database)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Flow

### 1. Memory-Enhanced Contextual Request

```
User Prompt
    ↓
Retrieve Historical Knowledge from Graphiti
    ↓
Enrich Request with Past Patterns
    ↓
Execute Refinement with Enhanced Context
    ↓
Store New Insights in Graphiti
    ↓
Update Pattern Relationships
    ↓
Future Sessions Benefit from Learning
```

### 2. Memory Storage

- **Refinement Insights**: Captures insights from each refinement iteration
- **Contextual Sessions**: Stores full agent collaboration context
- **Pattern Relationships**: Tracks successful patterns across sessions
- **Temporal Context**: Enables time-based filtering and learning

### 3. Memory Retrieval

- **Historical Knowledge**: Similar past refinements with outcomes
- **Related Patterns**: Successful patterns applicable to current context
- **Suggested Approaches**: Proven strategies from historical data
- **Common Pitfalls**: Errors to avoid based on past failures

## Canonical Schemas

### RefinementMemory

```typescript
{
  session_id: string;
  iteration_number: number;
  refinement_type: PatternType;
  prompt: string;
  content: string;
  outcome: RefinementOutcome;
  insights: string[];
  suggested_features?: string;
  bug_fixes?: string;
  quality_metrics?: {
    novelty_score?: number;
    quality_score?: number;
    improvement_percentage?: number;
  };
  execution_time_ms: number;
  timestamp_utc: string; // ISO-8601
}
```

### ContextualSession

```typescript
{
  session_id: string;
  mode: 'contextual';
  prompt: string;
  agents_involved: AgentType[];
  interactions: AgentInteraction[];
  context_window?: number;
  memory_compression_events?: MemoryCompressionEvent[];
  successes: number;
  failures: number;
  duration_ms: number;
  start_time_utc: string;
  end_time_utc: string;
  final_output?: string;
  quality_score?: number;
}
```

### PatternRelationship

```typescript
{
  pattern_id: string;
  pattern_type: PatternType;
  pattern_name: string;
  description: string;
  related_sessions: string[];
  success_rate: number; // 0-1
  avg_improvement?: number;
  avg_execution_time_ms: number;
  frequency: number;
  last_seen_utc: string;
  first_seen_utc: string;
}
```

### EnrichedContext

```typescript
{
  query: string;
  historical_knowledge: HistoricalKnowledge[];
  related_patterns: PatternRelationship[];
  suggested_approaches: string[];
  common_pitfalls: string[];
  success_probability?: number;
  confidence_score: number; // 0-1
  processing_time_ms: number;
  correlation_id: string;
  timestamp_utc: string;
}
```

## Usage Examples

### 1. Basic Memory-Enhanced Request

```typescript
import { ICRAdapter } from '@openevolve/icr-adapter';
import { EnhancedICRMemoryAgent, MemoryAgentConfig } from '@openevolve/icr-adapter';
import { GraphitiAdapter } from '@openevolve/graphiti-adapter';

// Create Graphiti adapter
const graphitiAdapter = new GraphitiAdapter({
  graphiti_api_url: process.env.GRAPHITI_API_URL!,
  neo4j_uri: process.env.NEO4J_URI!,
  neo4j_user: process.env.NEO4J_USER!,
  neo4j_password: process.env.NEO4J_PASSWORD!
});

// Create memory agent
const memoryAgentConfig: MemoryAgentConfig = {
  graphiti: {
    graphitiAdapter: graphitiAdapter,
    default_context_window: 5,
    max_historical_results: 20
  },
  enable_historical_retrieval: true,
  enable_pattern_learning: true,
  default_context_window: 5
};

// Create ICR adapter with memory
const icrAdapter = new ICRAdapter({
  memoryAgentConfig: memoryAgentConfig
});

// Make memory-enhanced contextual request
const response = await icrAdapter.createContextualRequestWithMemory(
  "Refine this React component to improve performance",
  {
    context_window: 5,
    enable_learning: true
  }
);

console.log('Result:', response.result.content);
console.log('Enriched Context:', response.enriched_context);
console.log('Learning Result:', response.learning_result);
```

### 2. Manual Memory Operations

```typescript
// Retrieve historical knowledge
const enrichedContext = await memoryAgent.retrieveHistoricalKnowledge(
  "React performance optimization",
  10, // context window
  correlationId
);

console.log('Historical Knowledge:', enrichedContext.historical_knowledge);
console.log('Related Patterns:', enrichedContext.related_patterns);
console.log('Suggested Approaches:', enrichedContext.suggested_approaches);

// Store refinement insights
await memoryAgent.storeRefinementInsights(
  refinementInsights,
  sessionId,
  correlationId
);

// Learn from session outcomes
const learningResult = await memoryAgent.learnFromSession(
  contextualSession,
  sessionOutcomes,
  correlationId
);

console.log('Patterns Learned:', learningResult.patterns_learned);
console.log('Confidence Score:', learningResult.confidence_score);
```

### 3. Pattern Analysis

```typescript
// Analyze patterns across multiple sessions
const patterns = await memoryAgent.analyzePatterns(
  contextualSessions,
  correlationId
);

for (const pattern of patterns) {
  console.log(`Pattern: ${pattern.pattern_name}`);
  console.log(`Success Rate: ${pattern.success_rate * 100}%`);
  console.log(`Frequency: ${pattern.frequency}`);
  console.log(`Related Sessions: ${pattern.related_sessions.length}`);
}
```

## Configuration

### Environment Variables

```bash
# ICR API Configuration
OPENEVOLVE_ICR_API_URL=http://icr-core:8000
TIMEOUT_MS=30000
MAX_RETRIES=3

# Graphiti Configuration
GRAPHITI_API_URL=http://graphiti-adapter:8080
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Memory Configuration
MEMORY_CONTEXT_WINDOW=5
MEMORY_MAX_RESULTS=20
MEMORY_LEARNING_THRESHOLD=0.7
```

### Memory Agent Configuration

```typescript
interface MemoryAgentConfig {
  // Graphiti memory configuration
  graphiti: GraphitiMemoryConfig;

  // Agent behavior configuration
  enable_historical_retrieval?: boolean;    // default: true
  enable_pattern_learning?: boolean;         // default: true
  enable_cross_session_learning?: boolean;   // default: true

  // Memory retrieval configuration
  default_context_window?: number;           // default: 5
  min_relevance_score?: number;              // default: 0.3
  max_historical_results?: number;           // default: 20

  // Learning configuration
  learning_threshold?: number;               // default: 0.7
  pattern_extraction_min_frequency?: number; // default: 2
}
```

## Pattern Learning

The memory system automatically learns from sessions:

1. **Successful Patterns**: Extracted from successful refinements
2. **Pattern Relationships**: Built across sessions with similar patterns
3. **Success Rates**: Calculated from historical outcomes
4. **Frequency Tracking**: Monitors how often patterns appear

### Pattern Types

- `iterative_refinement`: Traditional iterative improvements
- `agent_collaboration`: Multi-agent cooperation patterns
- `memory_compression`: Context compression strategies
- `context_switching`: Handling context changes
- `tool_usage`: Effective tool utilization
- `error_recovery`: Error handling and recovery
- `quality_improvement`: Quality enhancement patterns
- `novelty_generation`: Creative generation patterns
- `custom`: User-defined patterns

## Temporal Context

The memory system supports temporal filtering:

```typescript
// Query with time range
const historicalKnowledge = await memoryAgent.retrieveHistoricalKnowledge(
  {
    query: "React optimization",
    time_range: {
      start_utc: "2024-01-01T00:00:00Z",
      end_utc: "2024-12-31T23:59:59Z"
    }
  },
  10,
  correlationId
);
```

## Federation Constitution Compliance

### Law of Configuration Explicitness
- All configuration via environment variables or constructor options
- No magic defaults - required configuration validated at startup
- Crashes immediately if required variables are missing

### Law of UTC
- All timestamps in UTC ISO-8601 format
- No timezone conversions in memory storage
- Consistent temporal context across all sessions

### Law of Idempotency
- All memory operations safe to run multiple times
- Duplicate storage handled gracefully
- Pattern learning is idempotent

### Law of Runtime Truth
- Memory schemas reflect actual Graphiti API behavior
- Probe scripts verify integration before use
- Real API calls validate functionality

## Troubleshooting

### Memory Agent Not Working

1. Check Graphiti adapter is initialized:
   ```bash
   bash probes/check_graphiti_memory.sh
   ```

2. Verify environment variables are set:
   ```bash
   echo $GRAPHITI_API_URL
   echo $NEO4J_URI
   ```

3. Check memory agent configuration:
   ```typescript
   console.log('Has Memory Agent:', icrAdapter.hasMemoryAgent());
   ```

### No Historical Knowledge Returned

1. Verify Graphiti has data:
   ```typescript
   const stats = await graphitiAdapter.getStatistics();
   console.log('Episodes Count:', stats.episodes_count);
   ```

2. Check context window size:
   ```typescript
   // Increase context window
   const response = await icrAdapter.createContextualRequestWithMemory(
     prompt,
     { context_window: 20 } // Larger window
   );
   ```

3. Review correlation logs:
   ```typescript
   console.log('Enriched Context:', response.enriched_context);
   ```

## Performance Considerations

### Memory Retrieval Latency

- Typical retrieval: 100-500ms
- Depends on context window and graph size
- Cached patterns reduce latency (5-min TTL)

### Storage Overhead

- Episode storage: 50-200ms per episode
- Pattern learning: 200-1000ms per session
- Graph building: 500-2000ms for 10+ sessions

### Optimization Tips

1. Use appropriate context windows (5-10 is typical)
2. Enable pattern caching for frequent queries
3. Batch storage operations when possible
4. Use temporal filters to reduce query scope

## Future Enhancements

- [ ] Distributed memory across multiple Graphiti instances
- [ ] Advanced pattern mining algorithms
- [ ] Cross-session knowledge transfer
- [ ] Real-time pattern recommendation
- [ ] Memory compression for long-running sessions
- [ ] Federated learning across multiple ICR instances
