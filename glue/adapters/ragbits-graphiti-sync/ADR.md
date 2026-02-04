# Architecture Decision Record (ADR)
# RAGBits-Graphiti Bidirectional Synchronization

## Status
Accepted

## Date
2025-02-03

## Context

We need to synchronize data between two critical knowledge systems:

1. **RAGBits**: Document-based RAG (Retrieval-Augmented Generation) system
   - Stores document chunks with embeddings
   - Provides semantic search and retrieval
   - Optimized for text-based queries

2. **Graphiti**: Temporal Knowledge Graph
   - Stores entities and relationships with temporal context
   - Provides graph-based reasoning and inference
   - Optimized for knowledge-based queries

### Problem Statement

- **Data Silos**: Information in one system is not available in the other
- **Inconsistent Context**: Users get different answers depending on which system is queried
- **Lost Knowledge**: Document updates don't reflect in knowledge graph and vice versa
- **Temporal Drift**: Systems diverge over time without synchronization

### Constraints

- Must follow **Federation Constitution** (6 Commandments)
- Must maintain **Air Gap** - no direct imports from core-projects
- Must be **Idempotent** - safe to run multiple times
- Must use **UTC timestamps** exclusively
- Must have **explicit configuration** - no magic defaults
- Must handle **failures gracefully** with circuit breakers

## Decision

Implement a bidirectional synchronization adapter using the **Anti-Corruption Layer** pattern.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Anti-Corruption Layer (ACL)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Canonical Schema (TypeScript/Zod)        │  │
│  │  - SyncOperation, SyncResult, Conflict            │  │
│  │  - All data normalized to canonical format        │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            Sync Manager (Orchestrator)            │  │
│  │  - Coordinates sync operations                    │  │
│  │  - Manages conflict detection & resolution        │  │
│  │  - Handles circuit breakers                       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────┐  ┌───────────────────────────┐  │
│  │ RAGBits →        │  │ Graphiti →                │  │
│  │ Graphiti Sync    │  │ RAGBits Sync              │  │
│  │ - Chunk → Episode│  │ - Entity → Keywords       │  │
│  │ - Entity Extract │  │ - Boost Factors           │  │
│  │ - Temporal Meta  │  │ - Query Enhancement       │  │
│  └───────────────────┘  └───────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Conflict Detector (Validator)             │  │
│  │  - Entity mismatches                              │  │
│  │  - Temporal inconsistencies                       │  │
│  │  - Semantic conflicts                             │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                                    │
         │ Canonical Format                   │ Canonical Format
         ↓                                    ↓
┌─────────────────┐                  ┌─────────────────┐
│   RAGBits API   │                  │  Graphiti API   │
└─────────────────┘                  └─────────────────┘
```

## Rationale

### Why Anti-Corruption Layer?

1. **Data Normalization**: RAGBits uses `snake_case`, Graphiti uses `camelCase`
2. **Schema Independence**: Changes in one system don't break the other
3. **Validation**: Canonical schema validates all data before sync
4. **Transformation**: Converts between different data models transparently

### Why Bidirectional Sync?

1. **Knowledge Enhancement**: Graph entities enhance RAG retrieval with boost factors
2. **Contextualization**: Document chunks provide context for knowledge graph
3. **Temporal Awareness**: Sync maintains temporal consistency across systems
4. **Query Optimization**: Users get best of both systems in one query

### Why Circuit Breakers?

1. **Prevent Cascading Failures**: If one system is down, the other continues
2. **Automatic Recovery**: Retry when service comes back online
3. **Graceful Degradation**: Serve cached/stale data during outages
4. **Observability**: Track service health and recovery

## Consequences

### Positive

- **Unified Knowledge**: Both systems have access to all information
- **Enhanced Retrieval**: RAG queries benefit from knowledge graph entities
- **Temporal Consistency**: Knowledge graph captures when information was valid
- **Conflict Resolution**: Automatic detection and resolution of data conflicts
- **Resilience**: System continues operating even if one component fails
- **Observability**: Comprehensive logging and metrics for monitoring

### Negative

- **Complexity**: Additional layer adds complexity to the system
- **Latency**: Sync operations add overhead to document ingestion
- **Storage**: Storing sync metadata increases storage requirements
- **Maintenance**: Need to monitor and maintain sync state

### Mitigations

- **Idempotent Operations**: Safe to retry without side effects
- **Batch Processing**: Reduce overhead by batching sync operations
- **Configurable Intervals**: Control sync frequency based on needs
- **Circuit Breakers**: Prevent cascading failures and excessive retries
- **Monitoring**: Comprehensive metrics and alerting

## Implementation Details

### Sync Flow: RAGBits → Graphiti

```
1. Document Ingestion Trigger
   ↓
2. Chunk Document
   ↓
3. Convert Chunk → Episode
   - Extract entities (NER/LLM)
   - Extract relationships
   - Add temporal metadata
   ↓
4. Add to Graphiti
   - Create episode with entities
   - Create relationships
   - Update communities
   ↓
5. Detect Conflicts
   - Entity mismatches
   - Temporal inconsistencies
   - Semantic conflicts
   ↓
6. Resolve Conflicts
   - Auto-resolve (if low severity)
   - Flag for manual resolution (if high severity)
   ↓
7. Update Sync State
   - Mark operation completed
   - Record metrics
```

### Sync Flow: Graphiti → RAGBits

```
1. Entity Update Trigger
   ↓
2. Extract Keywords
   - Entity name
   - Labels
   - Summary terms
   ↓
3. Create Boost Factor
   - Calculate confidence score
   - Determine boost value
   - Assign relevance weights
   ↓
4. Update RAGBits Metadata
   - Add keywords to chunks
   - Apply boost factors
   - Update retrieval indices
   ↓
5. Enhanced Query (On Retrieval)
   - Identify relevant entities
   - Build enhanced query with entity context
   - Apply boost factors to retrieval
   ↓
6. Return Enhanced Results
```

### Conflict Detection Strategies

#### Entity Mismatch
- **Detection**: Same entity name with different labels/attributes
- **Severity**: Low-Medium
- **Resolution**: Merge labels and attributes

#### Temporal Inconsistency
- **Detection**: Same data with different timestamps (drift > threshold)
- **Severity**: Medium-High
- **Resolution**: Use newest timestamp (newest_wins)

#### Semantic Conflict
- **Detection**: Similar content with low semantic similarity (< threshold)
- **Severity**: Medium
- **Resolution**: Manual review or merge with conflict marker

### Configuration

All configuration via environment variables:

```bash
# API Endpoints (Required)
RAGBITS_API_URL=http://ragbits:8000
GRAPHITI_API_URL=http://graphiti:8000

# Sync Configuration
SYNC_ENABLED=true
SYNC_BIDIRECTIONAL=true
SYNC_INTERVAL_MS=300000
SYNC_TIMEOUT_MS=30000
SYNC_MAX_RETRIES=3
SYNC_BATCH_SIZE=10

# Conflict Resolution
SYNC_CONFLICT_RESOLUTION=newest_wins
SYNC_AUTO_RESOLVE_CONFLICTS=true

# Detection Thresholds
SYNC_SEMANTIC_SIMILARITY_THRESHOLD=0.8
SYNC_TEMPORAL_DRIFT_THRESHOLD_MS=60000
```

## Testing Strategy

### 1. Contract Tests (Phase 2: The Contract)
- Run on container startup
- Verify API contracts
- Refuse to start if contracts violated
- Location: `tests/contract.test.ts`

### 2. Probe Scripts (Phase 1: The Probe)
- Verify API functionality
- Test sync operations end-to-end
- Test conflict detection
- Location: `probes/*.sh`

### 3. Integration Tests
- Test full sync pipeline
- Test conflict resolution
- Test error scenarios
- Test circuit breaker functionality

## Monitoring & Observability

### Metrics
- `sync_success_rate`: Percentage of successful syncs
- `avg_sync_latency_ms`: Average sync duration
- `conflict_count`: Number of conflicts detected
- `conflict_resolution_rate`: Percentage of auto-resolved conflicts
- `circuit_breaker_state`: Current state (closed/open/half_open)
- `active_operations`: Number of operations in progress

### Logging
- Structured JSON Lines format
- Correlation IDs for distributed tracing
- Include: source_service, target_service, operation_id

### Alerts
- High conflict rate (> 10% of operations)
- Circuit breaker open
- Sync latency > threshold
- Failed sync operations

## Alternatives Considered

### Alternative 1: Unidirectional Sync (RAGBits → Graphiti only)
**Rejected**: Graphiti knowledge would not enhance RAG retrieval

### Alternative 2: Direct Database Replication
**Rejected**: Violates "Untouchable DB" law, bypasses application logic

### Alternative 3: Event-Driven Sync (Message Queue)
**Rejected**: Adds complexity, not necessary for current scale
**Note**: Could be adopted in future if scale requires it

### Alternative 4: No Sync (Separate Systems)
**Rejected**: Defeats purpose of unified knowledge platform

## Future Improvements

1. **Event-Driven Architecture**: Use message queue for async sync
2. **Machine Learning**: Improve entity extraction with ML models
3. **Semantic Analysis**: Use embeddings for semantic conflict detection
4. **Graph Neural Networks**: Enhance knowledge graph with GNNs
5. **Distributed Locking**: Prevent concurrent sync conflicts
6. **Version Control**: Track data versions across systems

## References

- [Federation Constitution](../../CLAUDE.md)
- [RAGBits Canonical Schema](../../schemas/ragbits-canonical.ts)
- [Graphiti Canonical Schema](../../schemas/graphiti-canonical.ts)
- [Circuit Breaker Pattern](../../lib/circuit-breaker.ts)
- [Structured Logger](../../lib/logger.ts)

## Authors

OpenEvolve Federation Architecture Team

## Revision History

- 2025-02-03: Initial ADR created
