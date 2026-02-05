# Lean 4 Bridge Architecture

## Overview

The Lean 4 Bridge provides formal verification capabilities for the RESE (Recursive Epistemic Solvability Engine) specification using Lean 4, a modern interactive theorem prover.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RESE Pipeline                            │
│              (glue/orchestration)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Canonical Schema
                       │ (Anti-Corruption Layer)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Lean 4 Bridge                             │
│                  (glue/lib/lean4_bridge)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Python Interface Layer                     │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │   Lean4Interface                              │   │    │
│  │  │   - formalize_constraint()                    │   │    │
│  │  │   - prove_theorem()                           │   │    │
│  │  │   - verify_proof()                            │   │    │
│  │  │   - elaborate_fdg()                           │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                                                        │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │   Circuit Breaker                             │   │    │
│  │  │   - Stop hammering if Lean 4 is down         │   │    │
│  │  │   - Automatic recovery                        │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                                                        │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │   Structured Logging                          │   │    │
│  │  │   - JSON output                               │   │    │
│  │  │   - Correlation IDs                           │   │    │
│  │  │   - UTC timestamps                            │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Translation Layer (ACL)                       │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │   ConstraintTranslator                         │   │    │
│  │  │   - RESE → Lean 4 syntax                      │   │    │
│  │  │   - Natural language → formal logic           │   │    │
│  │  │   - FDG → Lean 4 structures                   │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                               │
                               │ Subprocess / Lean Server Protocol
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      Lean 4 Process                           │
│                    (infra/lean4-docker)                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Lean 4 Core                             │    │
│  │              (v4.11.0)                               │    │
│  │                                                        │    │
│  │  ┌────────────────────────────────────────────┐      │    │
│  │  │   Mathlib                                   │      │    │
│  │  │   - Mathematical library                    │      │    │
│  │  │   - Data structures                         │      │    │
│  │  │   - Theorems & proofs                       │      │    │
│  │  └────────────────────────────────────────────┘      │    │
│  │                                                        │    │
│  │  ┌────────────────────────────────────────────┐      │    │
│  │  │   RESE Library                              │      │    │
│  │  │   - RESE.lean                                │      │    │
│  │  │   - Constraints.lean                         │      │    │
│  │  │   - FDG.lean                                 │      │    │
│  │  └────────────────────────────────────────────┘      │    │
│  │                                                        │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. Python Interface Layer (`lean4_interface.py`)

**Purpose**: Provide Python API to Lean 4 formal verification.

**Key Features**:
- **Circuit Breaker**: Prevents cascading failures when Lean 4 is down
- **Timeout Enforcement**: All operations bounded (Law of Configuration Explicitness)
- **Structured Logging**: JSON logs with correlation IDs
- **Idempotency**: All operations are repeatable

**Methods**:
```python
interface.formalize_constraint(constraint, constraint_type)
interface.prove_theorem(theorem_name, tactics)
interface.verify_proof(proof_code)
interface.elaborate_fdg(fdg)
```

**Error Handling**:
- `Lean4TimeoutError`: Operation exceeded timeout
- `Lean4VerificationError`: Proof verification failed
- `Lean4CircuitBreakerOpenError`: Too many failures, circuit open
- `Lean4SyntaxError`: Invalid Lean 4 syntax

### 2. Translation Layer (`constraint_translator.py`)

**Purpose**: Anti-Corruption Layer (ACL) for RESE → Lean 4 translation.

**Key Features**:
- **Natural Language → Lean 4**: Translate constraints to formal logic
- **FDG Formalization**: Convert functional dependency graphs to Lean 4
- **Syntax Validation**: Verify Lean 4 syntax before execution

**Translation Mappings**:
```
RESE Operator → Lean 4 Symbol
----------------------------
"and"         → "∧"
"or"          → "∨"
"not"         → "¬"
"implies"     → "→"
"forall"      → "∀"
"exists"      → "∃"
```

### 3. Lean 4 Library (`lean4/`)

**Purpose**: Formal definitions for RESE concepts in Lean 4.

**Modules**:

#### `RESE.lean`
- Main library structure
- Verification process orchestration
- Example theorems

#### `Constraints.lean`
- Category A: Hard parameter inequalities (physical laws)
- Category B: Soft statistical constraints (heuristics)
- Category C: Tacit assumptions (unstated beliefs)
- Category D: Inverted constraints (solution requirements)

#### `FDG.lean`
- Functional Dependency Graph structures
- Node and edge definitions
- Mechanistic isomorphism (ℑ_mech) calculation
- Acyclicity and well-foundedness properties

## Design Principles

### Following CLAUDE.md

1. **Law of the "Air Gap" (Source Code Isolation)**
   - No imports from `core-projects/` into Lean 4 bridge
   - Clean separation between RESE and Lean 4
   - All data translation via ACL

2. **Law of "Runtime Truth" (Anti-Hallucination)**
   - Verify Lean 4 installation at startup
   - Probe Lean 4 before use
   - Execute all translations before deployment

3. **Law of Configuration Explicitness**
   - All timeouts via environment variables
   - Circuit breaker thresholds configurable
   - No magic defaults

4. **Law of Idempotency**
   - Formalization is repeatable
   - Verification is deterministic
   - Circuit breaker state is recoverable

5. **Structured Logging**
   - All logs in JSON format
   - Correlation IDs for distributed tracing
   - UTC timestamps (Law of UTC)

6. **Timeout Enforcement**
   - All operations bounded
   - No infinite loops
   - Circuit breaker prevents cascading failures

## Circuit Breaker Pattern

### States

```
CLOSED → Normal operation
  ├─ Success → Stay CLOSED
  └─ Failure → Increment count
      └─ Count ≥ threshold → OPEN

OPEN → Reject all requests
  └─ Timeout elapsed → HALF_OPEN

HALF_OPEN → Allow limited requests
  ├─ Success → CLOSED
  └─ Failure → OPEN
```

### Configuration

```bash
LEAN4_CIRCUIT_BREAKER_THRESHOLD=5          # Failures before opening
LEAN4_CIRCUIT_BREAKER_TIMEOUT_MS=60000      # Time to stay open
LEAN4_CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS=3  # Attempts in half-open
```

## Verification Flow

### 1. Constraint Formalization

```
RESE Constraint
    ↓
Canonical Schema Validation
    ↓
ConstraintTranslator.translate_to_lean4()
    ↓
Lean 4 Syntax Check
    ↓
Lean4Interface.formalize_constraint()
    ↓
Formalized Lean 4 Code
```

### 2. Theorem Proving

```
Theorem Statement
    ↓
Generate Lean 4 Theorem
    ↓
Apply Tactics (List of Lean 4 tactics)
    ↓
Lean4Interface.prove_theorem()
    ↓
Proof Status (proved/partial/failed)
```

### 3. FDG Elaboration

```
RESE FDG (from Phase II)
    ↓
Extract nodes and edges
    ↓
ConstraintTranslator.translate_fdg_to_lean4()
    ↓
Generate Lean 4 structures
    ↓
Lean4Interface.elaborate_fdg()
    ↓
Formalized FDG in Lean 4
```

## Performance Considerations

### Lean 4 Startup Time
- **Problem**: Lean 4 has high startup overhead (~1-2 seconds)
- **Solution**: Keep Lean 4 process running (docker container)
- **Trade-off**: Memory usage vs. startup time

### Mathlib Caching
- **Problem**: Mathlib is large (~4GB)
- **Solution**: Pre-cache at build time
- **Trade-off**: Larger Docker image vs. faster builds

### Proof Search
- **Problem**: Some proofs take long time
- **Solution**: Configurable timeouts per operation
- **Trade-off**: Completeness vs. responsiveness

## Failure Scenarios

| Scenario | Strategy |
|----------|----------|
| Lean 4 process crashes | Circuit breaker opens, stop requests |
| Proof timeout | Return partial result, log to DLQ |
| Syntax error | Return error, don't crash |
| Mathlib not found | Log error, return graceful degradation |
| Out of memory | Kill process, restart container |

## Monitoring

### Key Metrics

- **Circuit Breaker State**: CLOSED/OPEN/HALF_OPEN
- **Verification Success Rate**: % of successful verifications
- **Average Verification Time**: ms per verification
- **Lean 4 Memory Usage**: MB of Lean 4 process
- **Proof Success Rate**: % of successful proofs

### Logging Example

```json
{
  "timestamp": "2026-02-04T12:34:56.789Z",
  "level": "info",
  "component": "lean4_interface",
  "source_service": "lean4_bridge",
  "target_service": "lean4_formal_verification",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "msg": "Constraint formalized successfully",
  "theorem_name": "theorem_lattice_defects_uniform_abc123",
  "verification_status": "verified",
  "execution_time_ms": 1234
}
```

## Future Enhancements

1. **Interactive Proving**: Add support for interactive proof sessions
2. **Proof Automation**: Integrate with auto-tactics (aesop, simp, etc.)
3. **Parallel Verification**: Verify multiple theorems in parallel
4. **Proof Export**: Export proofs to other formats (TPTP, etc.)
5. **Machine Learning**: Use ML to suggest proof tactics

## References

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [RESE Technical Manual](../README.md)
- [CLAUDE.md](../../CLAUDE.md)
