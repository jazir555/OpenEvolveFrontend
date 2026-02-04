# Proof Knowledge Base

Centralized system for storing, searching, and reusing formal proofs from Z3, LeanAide, and other formal verification systems.

## Overview

The Proof Knowledge Base (PKB) provides a unified repository for formal mathematical proofs with:

- **Semantic Search**: Find similar proofs using vector embeddings
- **Lineage Tracking**: Track proof dependencies and relationships using graph indexing
- **Validation**: Automatically validate proofs using Z3 and LeanAide
- **Idempotency**: All operations are safe to retry (Law of Idempotency)
- **Distributed Tracing**: Full correlation ID support for observability

## Architecture

```
Proof Generated (Z3 or LeanAide)
        ↓
Extract Metadata
        ↓
[Generate Embedding] + [Create Graph Episode]
        ↓                    ↓
   Vector Index          Graph Index
   (Semantic Search)     (Lineage Tracking)
        ↓                    ↓
        Unified Proof Knowledge Base
```

## Installation

```bash
npm install @openevolve/proof-knowledge-base
```

## Quick Start

```typescript
import { ProofKnowledgeBase, FormalProof, Theorem } from '@openevolve/proof-knowledge-base';

// Create knowledge base
const kb = new ProofKnowledgeBase({
  vectorIndexEnabled: true,
  graphIndexEnabled: true,
  validationEnabled: true,
  autoValidateOnStore: true,
  z3ApiUrl: 'http://z3-core:8000',
  leanaideApiUrl: 'http://leanaide-core:8000',
});

// Store a theorem
const theorem: Theorem = {
  id: 'theorem-1',
  statement: 'For all natural numbers n, n + 0 = n',
  type: 'theorem',
  constraints: ['n ∈ Nat'],
  dependencies: [],
  created_at: new Date().toISOString(),
};
await kb.storeTheorem(theorem);

// Store a proof
const proof: FormalProof = {
  id: 'proof-1',
  theorem_id: 'theorem-1',
  theorem: 'For all natural numbers n, n + 0 = n',
  proof: `theorem add_zero (n : Nat) : n + 0 = n := by
    induction n with
    | zero => rfl
    | succ n ih => rw [add_succ, ih]`,
  system: 'leanaide',
  status: 'valid',
  confidence: 1.0,
  tactics: ['induction', 'rfl', 'rw'],
  dependencies: [],
  timestamp_utc: new Date().toISOString(),
};
await kb.storeProof(proof);

// Search for similar proofs
const similar = await kb.searchSimilar(theorem, 10);
console.log(`Found ${similar.length} similar proofs`);

// Get proof lineage
const lineage = await kb.getProofLineage('proof-1', 3);
console.log(`Ancestors: ${lineage.ancestors.length}`);
console.log(`Descendants: ${lineage.descendants.length}`);

// Get metrics
const metrics = await kb.getMetrics();
console.log(`Total proofs: ${metrics.total_proofs}`);
```

## Canonical Schema

### Theorem

```typescript
interface Theorem {
  id: string;                    // UUID
  statement: string;             // Formal statement
  type: TheoremType;             // lemma, theorem, corollary, etc.
  constraints?: string[];        // Preconditions
  dependencies?: string[];       // IDs of dependencies
  metadata?: Record<string, any>;
  created_at: string;            // UTC ISO-8601
  updated_at?: string;           // UTC ISO-8601
}
```

### FormalProof

```typescript
interface FormalProof {
  id: string;                    // UUID
  theorem_id: string;            // Reference to theorem
  theorem: string;               // Theorem statement (denormalized)
  proof: string;                 // Proof content
  system: ProofSystem;           // z3, leanaide, coq, etc.
  status: ProofStatus;           // valid, invalid, partial, etc.
  confidence?: number;           // 0-1 (for AI-generated proofs)
  tactics?: string[];            // Tactics used
  dependencies?: string[];       // Proof dependencies
  metadata?: {
    proof_length?: number;
    verification_time_ms?: number;
    memory_used_mb?: number;
    solver_version?: string;
    tactics_count?: number;
  };
  timestamp_utc: string;         // UTC ISO-8601
  correlation_id?: string;       // For distributed tracing
}
```

### ProofLineage

```typescript
interface ProofLineage {
  proof_id: string;
  ancestors: Array<{
    proof_id: string;
    depth: number;
    relationship: string;
  }>;
  descendants: Array<{
    proof_id: string;
    depth: number;
    relationship: string;
  }>;
  full_tree?: Record<string, any>;
  computed_at: string;           // UTC ISO-8601
}
```

## API Reference

### ProofKnowledgeBase

Main repository interface.

#### `storeProof(proof, correlationId?)`

Store a proof in the knowledge base. Idempotent operation.

```typescript
const result = await kb.storeProof(proof, 'correlation-123');
// { success: true, proof_id: '...', timestamp: '...' }
```

#### `storeTheorem(theorem, correlationId?)`

Store a theorem in the knowledge base.

```typescript
const result = await kb.storeTheorem(theorem, 'correlation-123');
```

#### `searchSimilar(theorem, maxResults, correlationId?)`

Search for similar proofs using semantic search.

```typescript
const similarProofs = await kb.searchSimilar(theorem, 10);
// Returns: SimilarProof[]
```

#### `searchByContent(query, maxResults, correlationId?)`

Search proofs by natural language query.

```typescript
const results = await kb.searchByContent('commutative property', 10);
```

#### `validateDependencies(proofId, correlationId?)`

Validate that all proof dependencies are valid.

```typescript
const valid = await kb.validateDependencies('proof-1');
```

#### `getProofLineage(proofId, depth, correlationId?)`

Get the lineage of a proof (ancestors and descendants).

```typescript
const lineage = await kb.getProofLineage('proof-1', 3);
```

#### `updateProof(proofId, newProof, correlationId?)`

Update a proof (creates new version).

```typescript
const result = await kb.updateProof('proof-1', newProof);
// Returns previous and new version IDs
```

#### `getProof(proofId, correlationId?)`

Retrieve a proof by ID.

```typescript
const proof = await kb.getProof('proof-1');
```

#### `getMetrics(correlationId?)`

Get knowledge base metrics.

```typescript
const metrics = await kb.getMetrics();
// { total_proofs: 100, proofs_by_system: {...}, ... }
```

### ProofVectorIndex

Vector index for semantic search.

#### `indexProof(proof, correlationId?)`

Index a proof for vector search.

```typescript
const result = await vectorIndex.indexProof(proof);
```

#### `searchSimilarTheorems(theorem, k, correlationId?)`

Search for proofs with similar theorems.

```typescript
const similar = await vectorIndex.searchSimilarTheorems(theorem, 10);
```

#### `searchByContent(content, k, correlationId?)`

Search by natural language content.

```typescript
const results = await vectorIndex.searchByContent('prove addition', 10);
```

### ProofGraphIndex

Graph index for lineage tracking.

#### `storeProof(proof, correlationId?)`

Store a proof in the graph.

```typescript
const result = await graphIndex.storeProof(proof);
```

#### `getProofLineage(proofId, depth, correlationId?)`

Get proof lineage.

```typescript
const lineage = await graphIndex.getProofLineage('proof-1', 3);
```

#### `getProofDependencies(proofId, correlationId?)`

Get proof dependencies.

```typescript
const deps = await graphIndex.getProofDependencies('proof-1');
```

### ProofValidator

Proof validation and checking.

#### `validateProof(proofId, proof, revalidateDependencies, correlationId?)`

Validate a proof.

```typescript
const validation = await validator.validateProof('proof-1', proof, true);
```

#### `batchValidate(proofs, correlationId?)`

Batch validate multiple proofs.

```typescript
const validations = await validator.batchValidate([
  { proofId: 'p1', proof: proof1 },
  { proofId: 'p2', proof: proof2 },
]);
```

## Federation Constitution Compliance

This library follows the OpenEvolve Federation Constitution:

### Law of the "Air Gap" (Source Code Isolation)

No imports from `core-projects/`. All proof data uses canonical schemas.

### Law of "Runtime Truth" (Anti-Hallucination)

All features are tested via probe scripts before use:
- `probes/check_storage.sh` - Test storage operations
- `probes/check_search.sh` - Test semantic search
- `probes/check_validation.sh` - Test proof validation

### Law of the "Untouchable DB" (Read-Only State)

READ-ONLY access to proof databases. Writes only for initialization/restoration.

### Law of Idempotency (The Replayability Pact)

All operations are safe to run multiple times:
- `storeProof()` checks before inserting
- `storeTheorem()` checks before inserting
- `validateProof()` can be called repeatedly

### Law of Configuration Explicitness

All configuration via environment variables:
- `Z3_API_URL` - Z3 service URL (optional)
- `LEANAIDE_API_URL` - LeanAide service URL (optional)

Service crashes immediately if required config is missing.

### Law of UTC

All timestamps in UTC ISO-8601 format:
- `created_at: "2025-02-03T12:34:56.789Z"`
- `timestamp_utc: "2025-02-03T12:34:56.789Z"`

## Probes (Runtime Testing)

Before using the knowledge base, run the probe scripts to verify functionality:

```bash
# Test storage operations
cd glue/lib/proof-knowledge-base/probes
./check_storage.sh

# Test semantic search
./check_search.sh

# Test validation
./check_validation.sh
```

All probes must pass before the adapter can start (Law of Runtime Truth).

## Contract Tests

Validate canonical schema contracts:

```bash
npm test
```

The contract tests verify:
- Schema validation for all canonical types
- Rejection of invalid data
- Timestamp format (UTC ISO-8601)
- Idempotency of operations
- Data integrity and metadata preservation

## Integration with Verification Systems

### Z3 Integration

```typescript
import { transformZ3ResponseToFormalProof } from '@openevolve/proof-knowledge-base';

const z3Response = await z3.solve(problem);
const proof = transformZ3ResponseToFormalProof(
  z3Response,
  theoremId,
  correlationId
);
await kb.storeProof(proof);
```

### LeanAide Integration

```typescript
import { transformLeanAideResponseToFormalProof } from '@openevolve/proof-knowledge-base';

const leanResponse = await leanaide.verify(proofCode);
const proof = transformLeanAideResponseToFormalProof(
  leanResponse,
  theoremId,
  correlationId
);
await kb.storeProof(proof);
```

## Proof Reuse Workflow

1. **Generate Proof**: Z3 or LeanAide generates a proof
2. **Extract Metadata**: Extract theorem, tactics, dependencies
3. **Store**: Store in knowledge base with indexing
4. **Search**: When proving similar theorems, search for existing proofs
5. **Reuse**: Adapt similar proofs for new theorems
6. **Validate**: Validate the new proof

## Error Handling

All operations return structured results:

```typescript
const result = await kb.storeProof(proof);

if (result.success) {
  console.log(`Stored proof: ${result.proof_id}`);
} else {
  console.error(`Failed to store: ${result.error}`);
}
```

## Observability

All operations include structured logging with correlation IDs:

```typescript
{
  "level": "info",
  "msg": "Proof stored successfully",
  "timestamp": "2025-02-03T12:34:56.789Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_service": "proof-knowledge-base",
  "proof_id": "proof-1",
  "vector_indexed": true,
  "graph_indexed": true
}
```

## License

MIT

## Contributing

Contributions must follow the Federation Constitution. All code must:
1. Use canonical schemas (no raw core project data)
2. Include probe scripts for runtime testing
3. Be idempotent (safe to retry)
4. Use UTC timestamps
5. Include correlation IDs for tracing

## Support

For issues and questions:
- GitHub: https://github.com/openevolve/proof-knowledge-base/issues
- Documentation: https://docs.openevolve.org/proof-knowledge-base
