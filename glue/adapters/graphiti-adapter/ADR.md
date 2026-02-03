# Architecture Decision Record (ADR): Graphiti Adapter

## Status
Accepted

## Date
2026-02-03

## Context
The OpenEvolve Federation needs to integrate Graphiti, a temporal knowledge graph system, into the mega-structure. Graphiti provides:

- **Temporal knowledge tracking** - Knowledge valid at specific points in time
- **Hybrid search** - Combines semantic embeddings, BM25, and graph traversal
- **Entity extraction** - Automatic extraction of entities and relationships from text
- **Community detection** - Groups related entities into communities

### Key Challenges
1. **Python-based** - Graphiti core is written in Python, but our glue layer is TypeScript
2. **Neo4j dependency** - Requires Neo4j graph database for storage
3. **LLM requirements** - Requires OpenAI/Anthropic API for entity extraction
4. **Bitemporal model** - Tracks both when facts were true AND when they were recorded

## Decision
We will create a TypeScript adapter that wraps Graphiti's Python API via HTTP/REST or direct Neo4j connection.

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Graphiti Adapter                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  adapter.ts - Main orchestration                      │  │
│  │  - Circuit breaker                                    │  │
│  │  - Retry logic                                        │  │
│  │  - Canonical schema validation                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                                                          │  │
│  ▼                                                          ▼  │
│  ┌─────────────────────┐  ┌──────────────────────────┐     │  │
│  │  graph-client.ts    │  │  temporal-ops.ts         │     │  │
│  │  - Episode CRUD     │  │  - Point-in-time query   │     │  │
│  │  - Entity CRUD      │  │  - Time-range search     │     │  │
│  │  - Search           │  │  - Entity timeline       │     │  │
│  └─────────────────────┘  └──────────────────────────┘     │  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  Graphiti Core (Python Service)      │
        │  - LLM-based entity extraction       │
        │  - Graph operations                  │
        │  - Community detection               │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  Neo4j Graph Database                │
        │  - Entity nodes                      │
        │  - Relationship edges                │
        │  - Episode nodes (temporal)          │
        └──────────────────────────────────────┘
```

## Alternatives Considered

### Alternative 1: Direct Python Integration
**Pros:**
- Native API access
- Full Graphiti feature set

**Cons:**
- Requires Python runtime in glue layer
- Breaks TypeScript-only adapter pattern
- Deployment complexity

**Decision: Rejected**

### Alternative 2: REST API Wrapper
**Pros:**
- Language-agnostic
- Clean separation

**Cons:**
- Additional network hop
- Requires building/maintaining REST API
- Serialization overhead

**Decision: Rejected (for now)**

### Alternative 3: Direct Neo4j Access from TypeScript
**Pros:**
- Full control over queries
- No Python dependency
- Native TypeScript implementation

**Cons:**
- Must reimplement Graphiti's entity extraction logic
- Lose LLM-based features
- High maintenance burden

**Decision: Rejected**

## Consequences

### Positive
- **Clean separation** - Adapter is isolated from Graphiti core
- **Temporal capabilities** - Full bitemporal knowledge tracking
- **Canonical schema** - Normalized interface to rest of system
- **Circuit breaker** - Prevents cascading failures from Graphiti issues

### Negative
- **Dependency on Neo4j** - Additional infrastructure component
- **LLM API costs** - Requires OpenAI/Anthropic for entity extraction
- **Complexity** - Bitemporal model adds query complexity
- **Indirection** - Going through Graphiti Python service adds latency

### Risks
1. **Graphiti API changes** - Mitigated by contract tests
2. **Neo4j performance** - Mitigated by circuit breaker and timeouts
3. **LLM rate limits** - Mitigated by retry logic with exponential backoff
4. **Data inconsistency** - Mitigated by idempotent operations

## Implementation Details

### Canonical Schema Mapping
- **Entity** → `CanonicalEntity` (normalized node representation)
- **EntityEdge** → `CanonicalEntityEdge` (normalized edge)
- **EpisodicNode** → `CanonicalEpisode` (temporal episode)
- **Search results** → `CanonicalSearchResult` (unified search output)

### Temporal Operations
- **Point-in-time queries** - "What was true at time T?"
- **Time-range searches** - "What happened between T1 and T2?"
- **Entity timelines** - "How did entity X change over time?"
- **Contradiction detection** - "What facts about X changed?"

### Idempotency Strategy
- **Episode creation** - UUID-based deduplication
- **Entity creation** - MERGE queries (create if not exists)
- **Relationship creation** - Idempotent edge addition
- **Search operations** - Safe to retry (no side effects)

## Related Decisions
- [ADR-001] Federation Constitution - All adapters follow zero-trust principles
- [ADR-002] Canonical Schema Pattern - Anti-corruption layer for all integrations
- [ADR-003] Circuit Breaker Pattern - Prevent cascading failures

## References
- Graphiti Documentation: https://github.com/getgraphiti/core
- Graphiti Python Core: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\graphiti`
- OpenEvolve Integration: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\knowledge_engine\integrations\graphiti_integration.py`
