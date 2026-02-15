# ADR-004: Canonical Schema Design

## Status
**Accepted**

## Context
The OpenEvolve Frontend integrates 30+ external services, each with different data formats:
- OpenEvolve API: camelCase JSON
- RAGBits: snake_case with custom metadata
- Datapizza: Mixed camelCase/snake_case
- Z3: SMT-LIB format
- LeanAide: Lean theorem format

**The Problem**: Need to normalize these different formats for internal use.

## Decision
Implement **Canonical Schemas** using Zod/TypeScript interfaces for all major data types.

### Architecture
Following the **Anti-Corruption Layer (ACL)** pattern:

```
[Source A] → [Adapter A (Normalize to Canonical)] → [Event Bus] → [Adapter B (Map to Target)] → [Target B]
```

### Implementation

#### Canonical Schema Types
Define once, use everywhere:
- `FormalProof` - Universal proof representation
- `ProofValidation` - Standardized validation result
- `EvolutionRun` - Evolution execution result
- `DocumentChunk` - Document fragment for RAG
- `ProcessingResult` - Data processing outcome

#### Schema Location
- `glue/schemas/` - Canonical type definitions
- Individual adapters: `src/types/plugin-types.ts` (plugin-specific)

#### Example
```typescript
// Canonical: Universal proof format
export interface FormalProof {
  id: string;
  system: 'lean' | 'z3' | 'coq';
  theorem: string;
  proof: string;
  status: 'proven' | 'disproven' | 'unknown';
  metadata: ProofMetadata;
  correlation_id: string;
  created_at: string; // UTC ISO-8601
}

// Adapter: Convert from Z3 format
function fromZ3Proof(z3Proof: Z3Proof): FormalProof {
  return {
    id: z3Proof.proof_id,
    system: 'z3',
    theorem: z3Problem.smt_formula,
    proof: z3Proof.proof_trace,
    status: z3Proof.status === 'sat' ? 'proven' : 'disproven',
    metadata: { ... },
    correlation_id: ctx.id,
    created_at: new Date().toISOString()
  };
}
```

### Benefits
1. **Type Safety**: TypeScript catches schema mismatches at compile time
2. **Validation**: Zod validates data at runtime
3. **Isolation**: Changes to source APIs don't break internal code
4. **Documentation**: Schema serves as contract
5. **Testing**: Easy to mock/test with standard types

## Consequences

### Positive
- ✅ **Type safety**: Compile-time guarantees
- ✅ **Validation**: Runtime checking with Zod
- ✅ **Isolation**: Source API changes don't propagate
- ✅ **Testing**: Easy to create test data

### Negative
- ⚠️ **Boilerplate**: Need conversion functions for each adapter
- ⚠️ **Overhead**: Mapping adds small performance cost
- ⚠️ **Maintenance**: Schemas need updates when requirements change

### Mitigations
- Keep conversion functions simple and pure
- Use code generation for repetitive mappings
- Document canonical schemas clearly
- Version schemas when breaking changes occur

## Alternatives Considered

### Alternative 1: Pass Through Raw Data
**Description**: Use source API formats directly

**Pros**: No conversion overhead

**Cons**: Tight coupling, type mismatches, harder to test

**Rejected**: Violates isolation requirements

### Alternative 2: Universal Schema (One Size Fits All)
**Description**: Single schema that handles all cases

**Pros**: Consistent everywhere

**Cons**: Bloated, many optional fields, loses specificity

**Rejected**: Too complex, hard to validate

### Alternative 3: No Schema Validation
**Description**: Use `any` type everywhere

**Pros**: Maximum flexibility

**Cons**: No type safety, runtime errors, hard to debug

**Rejected**: Violates Federation Constitution reliability requirements

## Related Decisions
- [ADR-005: Anti-Corruption Layer Implementation](./005-acl.md)
- [ADR-007: Unified Verification System](./007-unified-verification.md)

## Implementation Date
2026-02-15

## Author
OpenEvolve Federation Team
