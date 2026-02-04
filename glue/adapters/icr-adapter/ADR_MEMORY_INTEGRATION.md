# ADR: ICR Contextual Mode Memory Integration

## Status

**Accepted** - 2026-02-03

## Context

ICR's Contextual Mode uses multiple specialized agents (main generator, iterative agent, memory agent) to collaboratively refine code through iterations. While effective for individual sessions, the system lacks:

1. **Persistent Memory**: Session insights are lost after completion
2. **Cross-Session Learning**: Each session starts without historical context
3. **Pattern Recognition**: Successful refinement patterns aren't identified or reused
4. **Temporal Context**: No ability to filter or learn from time-based patterns

The Federation Constitution requires:
- **Law of the "Air Gap"**: No direct imports from core-projects
- **Law of "Runtime Truth"**: Verify actual API behavior, not documentation
- **Law of Idempotency**: All operations safe to replay
- **Law of Configuration Explicitness**: All config via environment variables

## Decision

Integrate ICR's Contextual Mode with Graphiti temporal knowledge graph for persistent memory storage and retrieval.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ICR Adapter                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         EnhancedICRMemoryAgent (High-Level API)        │ │
│  │  - retrieveHistoricalKnowledge()                       │ │
│  │  - storeRefinementInsights()                           │ │
│  │  - learnFromSession()                                  │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │         GraphitiMemoryManager (Graphiti Integration)   │ │
│  │  - Episode formatting                                 │ │
│  │  - Knowledge transformation                           │ │
│  │  - Pattern extraction                                 │ │
│  └────────────────────────┬───────────────────────────────┘ │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Graphiti Adapter                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Canonical Schema Layer (ACL)                   │ │
│  └────────────────────────┬───────────────────────────────┘ │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Neo4j Graph DB                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Memory Canonical Schemas** (`memory/canonical.ts`)
   - `RefinementMemory`: Per-iteration insights
   - `ContextualSession`: Full session context
   - `PatternRelationship`: Cross-session pattern tracking
   - `EnrichedContext`: Historical knowledge + suggestions

2. **GraphitiMemoryManager** (`memory/graphiti-memory.ts`)
   - Converts ICR insights to Graphiti episodes
   - Retrieves and transforms knowledge from Graphiti
   - Builds contextual memory graphs
   - Implements pattern extraction and learning

3. **EnhancedICRMemoryAgent** (`memory/memory-agent.ts`)
   - High-level API for memory operations
   - Pattern caching for performance
   - Confidence scoring and relevance calculation
   - Suggests approaches and pitfalls

4. **ICRAdapter Integration** (`adapter.ts`)
   - `createContextualRequestWithMemory()`: Memory-enhanced requests
   - Automatic insight storage after refinement
   - Session outcome learning

## Rationale

### Why Graphiti?

1. **Temporal Knowledge Graph**: Built-in support for time-based queries
2. **Entity Extraction**: Automatic extraction of entities and relationships
3. **Community Detection**: Identifies clusters of related patterns
4. **Canonical Schema Compliance**: Fits Federation Constitution ACL pattern

### Why This Architecture?

**Air Gap Compliance**
- Memory integration lives in `glue/adapters/icr-adapter/src/memory/`
- No imports from `core-projects/icr`
- Graphiti access via canonical schema interface

**Runtime Truth**
- Probe script (`check_graphiti_memory.sh`) verifies integration
- Real API calls validate functionality
- Schema validation at runtime

**Idempotency**
- Storage operations use Graphiti's episode deduplication
- Pattern learning is additive and replay-safe
- Cache invalidation prevents stale data

**Configuration Explicitness**
- All configuration via constructor (no magic defaults)
- Environment variables for all external URLs
- Crashes immediately if required config missing

### Alternatives Considered

#### Alternative 1: In-Memory Cache Only
**Rejected**: Lost data on restart, no cross-session learning

#### Alternative 2: Direct Database Storage
**Rejected**: Bypasses Graphiti's entity extraction, violates ACL pattern

#### Alternative 3: Vector Database (RAGBits only)
**Rejected**: Lacks temporal context, no pattern relationships

## Consequences

### Positive

1. **Persistent Memory**: Session insights survive restarts
2. **Cross-Session Learning**: Patterns improve over time
3. **Intelligent Context**: Historical data enriches new requests
4. **Pattern Reuse**: Successful patterns automatically identified
5. **Temporal Filtering**: Time-based learning and retrieval

### Negative

1. **Complexity**: Additional components and integration points
2. **Latency**: Memory retrieval adds 100-500ms per request
3. **Storage Overhead**: Episode storage: 50-200ms per session
4. **Dependency**: Requires Graphiti and Neo4j availability

### Mitigation

1. **Performance**: Pattern caching (5-min TTL) reduces retrieval latency
2. **Reliability**: Circuit breakers and retry logic handle failures
3. **Fallback**: Graceful degradation to standard contextual mode
4. **Monitoring**: Structured logging with correlation IDs

## Implementation Details

### Memory Storage Flow

```typescript
// 1. Complete contextual refinement
const response = await icrAdapter.createContextualRequest(prompt);

// 2. Extract refinement insights
const insights = extractRefinementInsights(response);

// 3. Store in Graphiti as episode
await memoryAgent.storeRefinementInsights(insights, sessionId);

// 4. Build contextual session
const session = buildContextualSession(response);

// 5. Learn from outcomes
await memoryAgent.learnFromSession(session, outcomes);
```

### Memory Retrieval Flow

```typescript
// 1. Retrieve historical knowledge
const context = await memoryAgent.retrieveHistoricalKnowledge(
  query,
  contextWindow
);

// 2. Enrich prompt with suggestions
const enrichedPrompt = enrichPromptWithHistory(prompt, context);

// 3. Execute with enhanced context
const response = await icrAdapter.createContextualRequest(enrichedPrompt);
```

### Pattern Learning

```typescript
// Extract patterns from sessions
const patterns = await memoryAgent.analyzeSessions(sessions);

// Build pattern relationships
const relationships = buildPatternRelationships(patterns);

// Calculate success rates
patterns.forEach(p => {
  p.success_rate = calculateSuccessRate(p.related_sessions);
});
```

## Testing Strategy

### Unit Tests
- Schema validation for all memory types
- Episode formatting correctness
- Pattern extraction logic
- Confidence scoring accuracy

### Integration Tests
- Graphiti connectivity and storage
- Memory retrieval and filtering
- Cross-session learning
- Temporal queries

### Contract Tests
- Graphiti API compliance
- ICR API integration
- Schema validation at startup

### Probe Tests
- `check_graphiti_memory.sh`: 10 comprehensive probes
- API availability
- Schema definitions
- Method presence
- Flow verification

## Migration Path

### Phase 1: Schema and Infrastructure (Complete)
- ✅ Memory canonical schemas
- ✅ GraphitiMemoryManager
- ✅ EnhancedICRMemoryAgent
- ✅ ICR adapter integration

### Phase 2: Testing and Validation (Current)
- 🔄 Unit tests for memory components
- 🔄 Integration tests for Graphiti
- 🔄 Contract tests for API compliance
- 🔄 Probe script validation

### Phase 3: Rollout and Monitoring (Pending)
- ⏳ Gradual rollout with feature flags
- ⏳ Performance monitoring
- ⏳ Pattern learning effectiveness
- ⏳ User feedback collection

## References

- [Federation Constitution](../../../../../CLAUDE.md)
- [Graphiti Canonical Schema](../../schemas/graphiti-canonical.ts)
- [Memory Integration Documentation](./MEMORY_INTEGRATION.md)
- [Graphiti Adapter Documentation](../graphiti-adapter/README.md)

## Authors

- OpenEvolve Federation - Distinguished Engineer

## Revision History

- 2026-02-03: Initial ADR created
