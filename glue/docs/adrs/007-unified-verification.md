# ADR-007: Unified Verification System

## Status
**Proposed**

## Context
The OpenEvolve Frontend integrates multiple proof systems:
- **Z3**: SMT solver for first-order logic
- **Lean**: Interactive theorem prover
- **LeanAide**: AI-assisted Lean proving
- **Coq**: Proof assistant (future)

Each system:
- Has different input format (SMT-LIB vs Lean vs Coq)
- Has different output format
- Requires different execution environment
- Returns different proof objects

**The Problem**: How to unify these diverse proof systems into a coherent verification workflow?

## Decision
Implement **Unified Verification System** with canonical proof format.

### Architecture

```
┌────────────────────────────────────────────────────────┐
│                Unified Verification Interface          │
│                  (Canonical Format)                   │
└─────────────┬──────────────────────┬─────────────────┘
              │                      │
      ┌───────┴───────┐      ┌──────┴────────┐
      │  Z3 Adapter  │      │ Lean Adapter  │
      └───────┬───────┘      └──────┬────────┘
              │                      │
      ┌───────┴───────┐      ┌──────┴────────┐
      │  Z3 Engine    │      │ Lean Engine    │
      └───────────────┘      └────────────────┘
```

### Implementation

#### Canonical Proof Format
```typescript
interface CanonicalProof {
  id: string;
  system: 'z3' | 'lean' | 'coq' | 'leanaide';

  // Input theorem (normalized)
  theorem: {
    name: string;
    statement: string;
    hypotheses: string[];
    conclusion: string;
  };

  // Proof object (system-specific, preserved)
  proof: {
    format: string;
    content: string;
    metadata: Record<string, unknown>;
  };

  // Verification result
  verification: {
    is_valid: boolean;
    confidence: number;
    execution_time_ms: number;
    errors: string[];
    warnings: string[];
  };

  // Metadata
  created_at: string;  // UTC ISO-8601
  created_by: string;
  correlation_id: string;
}
```

#### Verification Pipeline
1. **Ingest**: Receive proof in any format
2. **Normalize**: Convert to canonical format
3. **Route**: Send to appropriate verifier
4. **Verify**: Execute proof in native system
5. **Aggregate**: Collect results in canonical format
6. **Emit**: Publish canonical result

#### Unified API
```typescript
interface VerificationService {
  // Verify a proof
  verifyProof(
    proof: FormalProof,
    revalidate?: boolean
  ): Promise<ProofValidation>;

  // Batch verify
  batchVerify(
    proofs: FormalProof[]
  ): Promise<ProofValidation[]>;

  // Revalidate on dependency change
  revalidateOnDependencyChange(
    changedProofId: string,
    dependentIds: string[]
  ): Promise<RevalidationResult>;
}
```

### Per-System Verifiers

#### Z3 Verifier
```typescript
class Z3Verifier {
  async verify(proof: CanonicalProof): Promise<ProofValidation> {
    // Convert to SMT-LIB format
    const smt = this.convertToSMT2(proof);

    // Execute Z3
    const result = await this.executeZ3(smt);

    // Normalize result
    return this.normalizeResult(result);
  }
}
```

#### Lean Verifier
```typescript
class LeanVerifier {
  async verify(proof: CanonicalProof): Promise<ProofValidation> {
    // Convert to Lean format
    const lean = this.convertToLean(proof);

    // Execute Lean
    const result = await this.executeLean(lean);

    // Normalize result
    return this.normalizeResult(result);
  }
}
```

#### LeanAide Verifier
```typescript
class LeanAideVerifier {
  async verify(proof: CanonicalProof): Promise<ProofValidation> {
    // Convert to Lean format
    const lean = this.convertToLean(proof);

    // Call LeanAide API
    const result = await this.callLeanAide(lean);

    // Normalize result
    return this.normalizeResult(result);
  }
}
```

### Cross-Proof Validation
When proof B depends on proof A:
1. Verify proof A first
2. Use A's result to verify B
3. If A changes, revalidate B

```typescript
async function verifyWithDependencies(
  proof: CanonicalProof,
  dependencyChain: CanonicalProof[]
): Promise<ProofValidation> {
  // Verify dependencies first
  for (const dep of dependencyChain) {
    const depResult = await this.verify(dep);
    if (!depResult.is_valid) {
      return {
        is_valid: false,
        error: `Dependency ${dep.id} is invalid`
      };
    }
  }

  // All dependencies valid, verify this proof
  return await this.verify(proof);
}
```

## Consequences

### Positive
- ✅ **Unified interface**: One API for all proof systems
- ✅ **Type safety**: Canonical format enforced by TypeScript
- ✅ **Extensibility**: Easy to add new proof systems
- ✅ **Dependency tracking**: Knows which proofs depend on others

### Negative
- ⚠️ **Complexity**: Need conversion for each system
- ⚠️ **Performance**: Additional layer adds overhead
- ⚠️ **Maintenance**: Must keep converters updated

### Mitigations
- Keep converters simple and focused
- Cache conversion results
- Use circuit breakers for each verifier
- Monitor converter performance

## Alternatives Considered

### Alternative 1: Separate APIs per System
**Description**: Each proof system has its own API

**Pros**: Simpler, direct access

**Cons**: Fragmented interface, harder to use

**Rejected**: Violates goal of unified system

### Alternative 2: Lowest Common Denominator
**Description**: Use only features available in all systems

**Pros**: Simpler, consistent

**Cons**: Loses powerful features of each system

**Rejected**: Unacceptable functionality loss

### Alternative 3: Choose One System
**Description**: Standardize on Z3 or Lean only

**Pros**: Simplest, no conversion

**Cons**: Loses benefits of other systems

**Rejected**: Each system has unique strengths

## Related Decisions
- [ADR-004: Canonical Schema Design](./004-canonical-schema.md)
- [ADR-005: Anti-Corruption Layer](./005-acl.md)

## Implementation Date
2026-02-15 (Proposed)
2026-03-15 (Target Implementation)

## Author
OpenEvolve Federation Team
