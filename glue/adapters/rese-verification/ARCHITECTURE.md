# Tiered Verification System Architecture

This document describes the architecture of the RESE Tiered Verification System, a unified 3-tier verification system integrating Z3, LeanAide, and Lean 4.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [System Architecture](#system-architecture)
4. [Tier Architecture](#tier-architecture)
5. [Data Flow](#data-flow)
6. [Component Design](#component-design)
7. [Integration Patterns](#integration-patterns)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [Security Considerations](#security-considerations)

## Overview

The Tiered Verification System provides a unified API for formal verification across three tiers of increasing rigor:

```
┌─────────────────────────────────────────────────────────┐
│              Tiered Verification System                 │
│                                                         │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Tier 1   │→ │   Tier 2     │→ │   Tier 3     │   │
│  │  (Z3)      │  │  (LeanAide)  │  │  (Lean 4)    │   │
│  │  Fast SAT  │  │  AI-Guided   │  │  Formal      │   │
│  └────────────┘  └──────────────┘  └──────────────┘   │
│       ↓                 ↓                  ↓            │
│  70% confidence   85% confidence   100% confidence      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Architecture Principles

Following CLAUDE.md laws:

1. **Law of Configuration Explicitness**
   - All configuration via environment variables
   - No magic defaults
   - Fail fast on missing config

2. **Law of Runtime Truth**
   - Verify solvers via probes before use
   - Trust execution, not documentation
   - Runtime contract validation

3. **Law of Idempotency**
   - All operations safe to run 100x
   - Check before create
   - UPSERT logic

4. **Circuit Breaker Pattern**
   - Detect solver failures
   - Stop hammering unhealthy services
   - Automatic recovery

5. **Structured Logging**
   - JSON format with correlation_id
   - Component labels
   - UTC timestamps

6. **Law of UTC**
   - All timestamps in UTC ISO-8601
   - No timezone ambiguity

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Application Layer                      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              TieredVerifier (Orchestrator)           │    │
│  │                                                        │    │
│  │  - verify()         - Main verification entry         │    │
│  │  - verify_with_tier() - Verify with specific tier     │    │
│  │  - escalate_tier()   - Escalate to next tier          │    │
│  │  - combine_results() - Combine tier results           │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                        Selection Layer                        │
│                                                               │
│  ┌─────────────────────┐  ┌──────────────────────────────┐   │
│  │ ProblemClassifier   │  │   SolverSelector             │   │
│  │                      │  │                              │   │
│  │ - classify()        │  │ - select_solver()            │   │
│  │ - should_escalate() │  │ - record_performance()       │   │
│  └─────────────────────┘  └──────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                         Result Layer                          │
│                                                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Z3       │  │  LeanAide    │  │   Lean 4     │         │
│  │   Result   │  │  Result      │  │   Result     │         │
│  └────────────┘  └──────────────┘  └──────────────┘         │
│       ↓                 ↓                  ↓                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         UnifiedVerificationResult                     │    │
│  │                                                       │    │
│  │  - Combines all tier results                         │    │
│  │  - Provides confidence score                         │    │
│  │  - Tracks escalation path                            │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                          Solver Layer                         │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Z3 Bridge   │  │ LeanAide     │  │  Lean 4      │       │
│  │              │  │ Bridge       │  │  Interface   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

## Tier Architecture

### Tier 1: Z3 Fast Verification

**Purpose**: Fast constraint satisfaction and contradiction detection

**Characteristics**:
- Execution time: <1 second
- Constraint count: 0-100
- Confidence: 70%
- Use case: Quick satisfiability checks

**Components**:
```
Z3VerificationResult
  - status: VerificationStatus
  - z3_result: sat/unsat/unknown
  - model: Dict[str, Any]
  - execution_time_ms: float
  - constraints_checked: int
```

**Decision Criteria**:
- Simple boolean constraints
- No quantifiers
- Linear arithmetic
- Fast response required

### Tier 2: LeanAide AI-Assisted Proving

**Purpose**: AI-guided theorem proving with autoformalization

**Characteristics**:
- Execution time: <1 minute
- Constraint count: 100-1000
- Confidence: 85%
- Use case: Medium-complexity theorems

**Components**:
```
LeanAideVerificationResult
  - status: VerificationStatus
  - proof_status: proved/failed/partial
  - proof_script: str
  - tactics_used: List[str]
  - autoformalization_confidence: float
```

**Decision Criteria**:
- Quantifiers present
- Nonlinear operations
- Medium complexity
- AI assistance beneficial

### Tier 3: Lean 4 Formal Verification

**Purpose**: Machine-checkable formal proofs

**Characteristics**:
- Execution time: No strict limit
- Constraint count: 1000+
- Confidence: 100%
- Use case: High-assurance verification

**Components**:
```
Lean4VerificationResult
  - status: VerificationStatus
  - verification_status: verified/errors
  - lean4_code: str
  - theorem_name: str
  - proof_object: str
```

**Decision Criteria**:
- Deep quantifier nesting
- Complex mathematical reasoning
- Machine-checkable proof required
- Maximum rigor needed

## Data Flow

```
User Request
    ↓
┌─────────────────────────────────────┐
│  1. Problem Classification          │
│     - Analyze problem statement     │
│     - Extract constraints           │
│     - Estimate complexity           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Solver Selection                │
│     - Check circuit breakers        │
│     - Select initial tier           │
│     - Plan escalation path          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Tier Execution                  │
│     - Execute verification          │
│     - Record performance            │
│     - Check results                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Escalation Decision             │
│     - Should escalate?              │
│     - Yes → Next tier               │
│     - No → Return results           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. Result Combination              │
│     - Combine all tier results      │
│     - Calculate confidence          │
│     - Generate summary              │
└─────────────────────────────────────┘
    ↓
Unified Verification Result
```

## Component Design

### ProblemClassifier

**Responsibility**: Classify verification problems

**Methods**:
- `classify()` - Main classification method
- `_classify_class()` - Classify problem class
- `_classify_domain()` - Classify problem domain
- `_compute_complexity()` - Compute complexity metrics

**Pattern**: Strategy pattern for different classification types

### SolverSelector

**Responsibility**: Select appropriate solver tier

**Methods**:
- `select_solver()` - Main selection method
- `_select_fast_first()` - Fast-first strategy
- `_select_accurate_first()` - Accurate-first strategy
- `_select_parallel()` - Parallel strategy
- `_select_adaptive()` - Adaptive strategy

**Pattern**: Strategy pattern with circuit breaker

### TieredVerifier

**Responsibility**: Orchestrate tiered verification

**Methods**:
- `verify()` - Main verification entry point
- `verify_with_tier()` - Verify with specific tier
- `escalate_tier()` - Escalate to next tier
- `combine_results()` - Combine results

**Pattern**: Orchestrator pattern with lazy initialization

## Integration Patterns

### Anti-Corruption Layer

All tier results are transformed to canonical format:

```
Z3 Format → Z3VerificationResult → UnifiedVerificationResult
LeanAide Format → LeanAideVerificationResult → UnifiedVerificationResult
Lean 4 Format → Lean4VerificationResult → UnifiedVerificationResult
```

### Circuit Breaker

Each solver has circuit breaker protection:

```
┌────────────────┐
│ Circuit Closed │ → Normal operation
└────────────────┘
         ↓ (failures ≥ threshold)
┌────────────────┐
│ Circuit Open   │ → Block requests
└────────────────┘
         ↓ (timeout elapsed)
┌────────────────┐
│ Half-Open      │ → Test with limited requests
└────────────────┘
```

### Idempotency

All operations are idempotent:

```python
# Safe to run multiple times
result = verifier.verify(problem)
result = verifier.verify(problem)  # Same result
```

## Error Handling

### Error Categories

1. **Transient Errors** (retry with exponential backoff)
   - Network timeouts
   - Temporary unavailability

2. **Logic Errors** (log and continue)
   - Invalid constraints
   - Unsatifiable problems

3. **System Errors** (circuit breaker)
   - Solver crashes
   - Persistent failures

### Error Recovery

```
Error Detected
    ↓
Error Classification
    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Transient?      │  │ Logic?          │  │ System?         │
│ Retry           │  │ Log & Continue  │  │ Circuit Breaker │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Performance Optimization

### Caching

- Problem classification results cached
- Solver performance tracked
- Successful results cached (configurable TTL)

### Parallel Execution

- Optional parallel tier execution
- Configurable max parallel solvers
- Result combination on completion

### Monitoring

- Execution time tracking
- Success rate monitoring
- Circuit breaker state tracking

## Security Considerations

### Input Validation

- All problem statements validated
- Constraint expressions sanitized
- Tier selection limited by max_tier

### Resource Limits

- Timeouts enforced per tier
- Maximum tier limit (max_tier)
- Memory limits via containerization

### Audit Trail

- All operations logged with correlation_id
- Escalation path tracked
- Performance metrics recorded

## Deployment

### Container Architecture

```
┌─────────────────────────────────────┐
│         Docker Container            │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Application  │  │  Solvers    │ │
│  │   (Python)   │  │  (Z3, Lean) │ │
│  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────┘
```

### Health Checks

- Probe script verifies all tiers
- Circuit breaker states exposed
- Performance metrics available

### Configuration

All configuration via environment variables:
- Timeouts per tier
- Constraint thresholds
- Escalation settings
- Solver preferences
