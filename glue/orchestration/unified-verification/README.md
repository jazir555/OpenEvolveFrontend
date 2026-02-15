# Unified Verification System

Centralized proof verification for OpenEvolve, integrating multiple proof systems (Z3, LeanAide, etc.).

## Overview

This module implements the Unified Verification System (ADR-007) with:

1. **Canonical Proof Format** - Universal representation for all proof systems
2. **Verification Service** - Orchestrates proof verification across systems
3. **System Verifiers** - Pluggable adapters for Z3, LeanAide, etc.
4. **Cross-Validation** - Compare results from multiple systems
5. **Dependency Tracking** - Revalidate proofs when dependencies change

## Components

### UnifiedVerificationService
Main service for orchestrating proof verification.

**Features:**
- Single API for all proof systems
- Multiple verification strategies (z3_only, leanaide_only, parallel, sequential, hybrid)
- Cross-validation between systems
- Batch verification
- Dependency revalidation
- Circuit breaker protection per system
- Retry with exponential backoff

**Usage:**
```typescript
import { UnifiedVerificationService, Z3Verifier, LeanAideVerifier } from '@openevolve/unified-verification';

const service = new UnifiedVerificationService();

// Register verifiers
service.registerVerifier(new Z3Verifier('http://z3-api:8080'));
service.registerVerifier(new LeanAideVerifier('http://leanaide-api:8081'));

// Verify a proof
const request = {
  requestId: uuidv4(),
  problem: {
    id: uuidv4(),
    type: 'SMT_CONSTRAINTS',
    statement: '(declare-const x Int) (assert (> x 5)) (check-sat)',
  },
  constraints: {
    timeout: 30000,
    requiredConfidence: 0.95,
  },
  timestamp: new Date().toISOString(),
};

const result = await service.verifyProof(request, {
  strategy: 'parallel',
  crossValidate: true,
});
```

### Z3Verifier
Verifies SMT-LIB formatted proofs using Z3 SMT solver.

**Features:**
- Solves SMT constraints
- Extracts models
- Proof extraction
- Timeout handling
- Circuit breaker protection

### LeanAideVerifier
Verifies Lean proofs using AI-assisted theorem proving.

**Features:**
- Verifies Lean theorems
- Suggests tactics
- Error reporting
- Proof object generation
- Circuit breaker protection

## Canonical Proof Format

All proofs are normalized to a canonical format:

```typescript
interface CanonicalProof {
  id: string;
  system: 'z3' | 'lean' | 'coq' | 'leanaide';
  theorem: {
    name: string;
    statement: string;
    hypotheses: string[];
    conclusion: string;
  };
  proof: {
    format: string;
    content: string;
    metadata?: Record<string, unknown>;
  };
  verification: {
    status: 'pending' | 'verifying' | 'proven' | 'disproven';
    confidence: number; // 0.0 to 1.0
    errors: Array<{ message: string; line?: number }>;
    warnings: string[];
  };
  dependencies: string[];
  metadata: {
    created_at: string; // UTC ISO-8601
    created_by: string;
    tags: string[];
  };
}
```

## Verification Strategies

### z3_only
Use only Z3 SMT solver. Fastest for SMT problems.

### leanaide_only
Use only LeanAide. Best for theorem proving.

### parallel
Run all systems simultaneously. Fastest overall, but most resource-intensive.

### sequential
Run systems one after another. Stop early if first system succeeds.

### hybrid
Smart strategy: Z3 first (fast), then LeanAide if Z3 is inconclusive.

## Configuration

### Environment Variables

```bash
# Z3 Configuration
Z3_API_URL=http://z3-core:8000
Z3_TIMEOUT_MS=30000

# LeanAide Configuration
LEANAIDE_API_URL=http://leanaide-core:8001
LEANAIDE_TIMEOUT_MS=60000

# Verification Options
VERIFICATION_DEFAULT_STRATEGY=parallel
VERIFICATION_CROSS_VALIDATE=true
VERIFICATION_CONFIDENCE_THRESHOLD=0.95
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Unified Verification Service                 │
│                   (Canonical Format)                   │
└─────────────┬───────────────────┬─────────────────────┘
              │                   │
      ┌───────┴────────┐  ┌──────┴────────┐
      │  Z3 Adapter  │  │ Lean Adapter  │
      └───────┬────────┘  └──────┬────────┘
              │                   │
      ┌───────┴────────┐  ┌──────┴────────┐
      │  Z3 Engine    │  │ Lean Engine    │
      └───────────────┘  └───────────────┘
```

## Usage Examples

### Simple Verification
```typescript
const result = await service.verifyProof(request, {
  strategy: 'z3_only',
});
```

### Cross-Validation
```typescript
const result = await service.verifyProof(request, {
  strategy: 'parallel',
  crossValidate: true,
});

console.log(result.agreement); // true if all systems agree
console.log(result.confidence); // combined confidence 0.0-1.0
console.log(result.resolution); // 'verified' | 'not_verified' | 'inconclusive'
```

### Dependency Revalidation
```typescript
// When a dependency changes, revalidate dependent proofs
const results = await service.revalidateOnDependencyChange(
  changedProofId,
  dependentProofs
);
```

### Batch Verification
```typescript
const results = await service.batchVerify(
  [request1, request2, request3],
  { parallel: true }
);
```

## Testing

```bash
# Run tests
npm test

# Run specific test suite
npm test -- z3-verifier
npm test -- verification-service
```

## References
- [ADR-007: Unified Verification System](../../docs/adrs/007-unified-verification.md)
- [ADR-004: Canonical Schema Design](../../docs/adrs/004-canonical-schema.md)
- [Federation Constitution](../../README.md)
