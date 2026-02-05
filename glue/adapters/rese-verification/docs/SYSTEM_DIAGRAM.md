# Tiered Verification System - Complete System Diagram

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         RESE Platform                                 │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                   Tiered Verification System                  │   │
│  │                                                                │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │            TieredVerifier (Orchestrator)               │   │   │
│  │  │                                                          │   │   │
│  │  │  verify() ─────────────────────────────────────────┐    │   │   │
│  │  │  verify_with_tier() ────────────────────────────────┤    │   │   │
│  │  │  escalate_tier() ────────────────────────────────────┤    │   │   │
│  │  │  combine_results() ─────────────────────────────────┤    │   │   │
│  │  │  get_verification_status() ──────────────────────────┤    │   │   │
│  │  └───────────────────────────────────────────────────────────┘    │   │
│  │                              │                                  │   │   │
│  └──────────────────────────────┼──────────────────────────────┘   │   │
│                                 │                                      │
│  ┌──────────────────────────────┴──────────────────────────────┐     │   │
│  │                       Selection Layer                       │     │   │
│  │                                                              │     │   │
│  │  ┌──────────────────────┐  ┌─────────────────────────────┐  │     │   │
│  │  │  ProblemClassifier   │  │    SolverSelector           │  │     │   │
│  │  │                      │  │                             │  │     │   │
│  │  │ - classify()         │  │ - select_solver()           │  │     │   │
│  │  │ - should_escalate()  │  │ - record_performance()      │  │     │   │
│  │  │ - estimate_tier()    │  │ - get_performance_stats()   │  │     │   │
│  │  └──────────────────────┘  └─────────────────────────────┘  │     │   │
│  └──────────────────────────────────────────────────────────────┘     │   │
│                                 │                                      │
│  ┌──────────────────────────────┴──────────────────────────────┐     │   │
│  │                        Result Layer                         │     │   │
│  │                                                              │     │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │     │   │
│  │  │Z3Verification│ │LeanAide     │  │Lean4         │        │     │   │
│  │  │Result       │  │Verification │  │Verification  │        │     │   │
│  │  │             │  │Result       │  │Result        │        │     │   │
│  │  │Tier 1       │  │Tier 2       │  │Tier 3        │        │     │   │
│  │  │70% conf.    │  │85% conf.    │  │100% conf.    │        │     │   │
│  │  └────────────┘  └──────────────┘  └──────────────┘        │     │   │
│  │         │                 │                  │              │     │   │
│  │         └─────────────────┴──────────────────┘              │     │   │
│  │                            │                                │     │   │
│  │  ┌─────────────────────────────────────────────────────┐   │     │   │
│  │  │        UnifiedVerificationResult                     │   │     │   │
│  │  │                                                      │   │     │   │
│  │  │  - Combines all tier results                         │   │     │   │
│  │  │  - Provides confidence score                          │   │     │   │
│  │  │  - Tracks escalation path                             │   │     │   │
│  │  │  - Human-readable summary                             │   │     │   │
│  │  └─────────────────────────────────────────────────────┘   │     │   │
│  └──────────────────────────────────────────────────────────────┘     │   │
│                                 │                                      │
│  ┌──────────────────────────────┴──────────────────────────────┐     │   │
│  │                        Solver Layer                         │     │   │
│  │                                                              │     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │     │   │
│  │  │ Z3 Bridge    │  │LeanAide      │  │Lean 4        │      │     │   │
│  │  │              │  │Bridge        │  │Interface     │      │     │   │
│  │  │rese-z3-bridge│  │z3_leanaide_  │  │lean4_bridge  │      │     │   │
│  │  │              │  │bridge        │  │              │      │     │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │     │   │
│  └──────────────────────────────────────────────────────────────┘     │   │
└───────────────────────────────────────────────────────────────────────┘
```

## Verification Flow

```
User Request
    │
    ├─ Problem Statement
    ├─ Constraints (optional)
    ├─ Variables (optional)
    └─ Metadata (optional)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Problem Classification                                │
│                                                         │
│  ProblemClassifier.classify()                           │
│    ├─ Analyze problem statement                         │
│    ├─ Extract constraints                               │
│    ├─ Identify domain (algebra, logic, etc.)            │
│    ├─ Compute complexity                                │
│    └─ Estimate starting tier                            │
│                                                         │
│  Output: ProblemClass, ProblemDomain, Complexity        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Solver Selection                                      │
│                                                         │
│  SolverSelector.select_solver()                         │
│    ├─ Check circuit breaker states                      │
│    ├─ Check solver availability                          │
│    ├─ Apply selection strategy                          │
│    ├─ Consider performance history                      │
│    └─ Plan escalation path                              │
│                                                         │
│  Output: SelectionResult (tier + alternatives)           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Tier Execution                                        │
│                                                         │
│  Tier 1 (Z3)                                            │
│    ├─ Fast SMT solving                                  │
│    ├─ <1 second target                                  │
│    ├─ 0-100 constraints                                 │
│    └─ 70% confidence                                    │
│         │                                               │
│         ├─ Success? → Return Result                     │
│         └─ Should escalate? → Tier 2                    │
│                                                         │
│  Tier 2 (LeanAide)                                      │
│    ├─ AI-assisted proving                               │
│    ├─ <1 minute target                                  │
│    ├─ 100-1000 constraints                              │
│    └─ 85% confidence                                    │
│         │                                               │
│         ├─ Success? → Return Result                     │
│         └─ Should escalate? → Tier 3                    │
│                                                         │
│  Tier 3 (Lean 4)                                        │
│    ├─ Formal verification                               │
│    ├─ No time limit                                     │
│    ├─ 1000+ constraints                                 │
│    └─ 100% confidence                                   │
│         │                                               │
│         └─ Return Result (final tier)                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Result Combination                                    │
│                                                         │
│  UnifiedVerificationResult                               │
│    ├─ Combine all tier results                          │
│    ├─ Calculate final confidence                        │
│    ├─ Identify successful tier                          │
│    ├─ Track escalation path                             │
│    └─ Generate human-readable summary                   │
│                                                         │
│  Output: UnifiedVerificationResult                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
User Response
    ├─ Success/Failure status
    ├─ Confidence score
    ├─ Successful tier
    ├─ Execution time
    ├─ Proof/Model (if available)
    └─ Human-readable summary
```

## Decision Tree

```
                    Start Verification
                            │
                            ▼
                    ┌───────────────┐
                    │ Problem has   │
                    │ quantifiers?  │
                    └───────────────┘
                      │           │
                     Yes          No
                      │           │
                      ▼           ▼
            ┌─────────────┐  ┌──────────────┐
            │Quantifier   │  │Has nonlinear?│
            │depth > 2?   │  └──────────────┘
            └─────────────┘    │           │
              │           │    Yes          No
              Yes          No    │            │
              │            │    ▼            ▼
              ▼            ▼  Tier 2      Tier 1
            Tier 3       Tier 2 (LeanAide)  (Z3)
            (Lean 4)
              │            │
              │            │
              └────────────┴───────────┐
                                      │
                                      ▼
                            ┌─────────────────┐
                            │Execute Tier     │
                            │with Timeout     │
                            └─────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            ┌───────────────┐                   ┌──────────────┐
            │ Status ==     │                   │Auto-escalate │
            │ VERIFIED?     │                   │enabled?      │
            └───────────────┘                   └──────────────┘
              │           │                         │           │
             Yes          No                       Yes          No
              │           │                         │           │
              ▼           ▼                         │           │
          SUCCESS    ┌─────────────┐                │           │
                     │Current tier │                │           │
                     │== Tier 3?   │                │           │
                     └─────────────┘                │           │
                       │           │                │           │
                      Yes          No               │           │
                       │           │                │           │
                       ▼           ▼                │           │
                    FAILURE   Escalate             │           │
                       │        to next            │           │
                       │           │                │           │
                       └───────────┴────────────────┴───────────┘
```

## Escalation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Escalation Decision Matrix                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Current Tier → Next Tier Conditions:                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 1 (Z3) → Tier 2 (LeanAide)                         │   │
│  │                                                          │   │
│  │ Escalate if ANY:                                        │   │
│  │  • Timeout > 1 second                                   │   │
│  │  • Status == UNKNOWN or TIMEOUT                         │   │
│  │  • Constraints > 100                                    │   │
│  │  • Quantifier depth > 2                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 2 (LeanAide) → Tier 3 (Lean 4)                     │   │
│  │                                                          │   │
│  │ Escalate if ANY:                                        │   │
│  │  • Timeout > 1 minute                                   │   │
│  │  • Status == FAILED or PARTIAL                          │   │
│  │  • Constraints > 1000                                   │   │
│  │  • Quantifier depth > 5                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 3 (Lean 4)                                          │   │
│  │                                                          │   │
│  │ • No escalation (final tier)                             │   │
│  │ • Returns result regardless of status                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Monitoring

```
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Tracking                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Per-Tier Metrics:                                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 1 (Z3)                                              │   │
│  │  • Total attempts: N                                     │   │
│  │  • Successful: M (M/N success rate)                      │   │
│  │  • Average time: X ms                                    │   │
│  │  • Circuit breaker: OPEN/CLOSED                         │   │
│  │  • Failure count: K (threshold: 5)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 2 (LeanAide)                                         │   │
│  │  • Total attempts: N                                     │   │
│  │  • Successful: M (M/N success rate)                      │   │
│  │  • Average time: X ms                                    │   │
│  │  • Circuit breaker: OPEN/CLOSED                         │   │
│  │  • Failure count: K (threshold: 5)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 3 (Lean 4)                                          │   │
│  │  • Total attempts: N                                     │   │
│  │  • Successful: M (M/N success rate)                      │   │
│  │  • Average time: X ms                                    │   │
│  │  • Circuit breaker: OPEN/CLOSED                         │   │
│  │  • Failure count: K (threshold: 3)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Circuit Breaker States:                                        │
│                                                                 │
│  ┌────────┐     Failures ≥     ┌────────┐     Timeout      ┌──────────┐
│  │ CLOSED │ ────────────────→ │  OPEN   │ ──────────────→ │ HALF-OPEN│
│  └────────┘                   └────────┘                   └──────────┘
│       ↑                           │                           │
│       │                           │        Success           │
│       └───────────────────────────┴───────────────────────────┘
│                         Reset on success
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

```
UnifiedVerificationResult
├── correlation_id: str
├── problem_class: ProblemClass
│   ├── CONSTRAINT_SAT
│   ├── THEOREM_PROVING
│   ├── OPTIMIZATION
│   ├── CONTRADICTION_DETECTION
│   └── MODEL_VALIDATION
├── problem_domain: ProblemDomain
│   ├── ALGEBRA
│   ├── ANALYSIS
│   ├── TOPOLOGY
│   ├── LOGIC
│   ├── PHYSICS
│   ├── ARITHMETIC
│   ├── GEOMETRY
│   └── GENERAL
├── tier1_result: Optional[Z3VerificationResult]
│   ├── status: VerificationStatus
│   ├── z3_result: str (sat/unsat/unknown)
│   ├── model: Optional[Dict[str, Any]]
│   ├── execution_time_ms: float
│   └── constraints_checked: int
├── tier2_result: Optional[LeanAideVerificationResult]
│   ├── status: VerificationStatus
│   ├── proof_status: str (proved/failed/partial)
│   ├── proof_script: Optional[str]
│   ├── tactics_used: List[str]
│   ├── autoformalization_confidence: float
│   └── execution_time_ms: float
├── tier3_result: Optional[Lean4VerificationResult]
│   ├── status: VerificationStatus
│   ├── verification_status: str (verified/errors)
│   ├── lean4_code: Optional[str]
│   ├── theorem_name: Optional[str]
│   ├── proof_object: Optional[str]
│   └── execution_time_ms: float
├── final_status: VerificationStatus
├── successful_tier: Optional[VerificationTier]
├── confidence: float (0.0 to 1.0)
├── escalation_path: List[VerificationTier]
├── escalation_reasons: List[str]
├── total_execution_time_ms: float
└── total_constraints_checked: int
```

## API Call Flow

```
User Application
    │
    │ verifier.verify("forall x, P(x) -> Q(x)")
    │
    ▼
TieredVerifier.verify()
    │
    ├─→ ProblemClassifier.classify()
    │   └─→ Returns: THEOREM_PROVING, LOGIC, complexity
    │
    ├─→ SolverSelector.select_solver()
    │   ├─→ Checks circuit breakers
    │   ├─→ Checks performance history
    │   ├─→ Applies selection strategy
    │   └─→ Returns: Tier 2 (LeanAide) recommended
    │
    ├─→ verify_with_tier(tier=Tier 2)
    │   │
    │   ├─→ LeanAide bridge.prove()
    │   │   ├─→ Autoformalization
    │   │   ├─→ Tactic suggestion
    │   │   ├─→ Proof execution
    │   │   └─→ Returns: LeanAideVerificationResult
    │   │
    │   ├─→ Check: is_successful()?
    │   │   ├─→ Yes: Return result
    │   │   └─→ No: Check should_escalate()?
    │   │       ├─→ Yes: escalate_tier()
    │   │       │   └─→ Try Tier 3
    │   │       └─→ No: Return failure
    │   │
    │   └─→ Returns: LeanAideVerificationResult
    │
    ├─→ unified.add_tier_result(result)
    │   ├─→ Updates final_status
    │   ├─→ Sets successful_tier
    │   ├─→ Calculates confidence
    │   └─→ Tracks escalation_path
    │
    └─→ Returns: UnifiedVerificationResult
        │
        ▼
User Application
    │
    ├─ result.is_successful() → True/False
    ├─ result.successful_tier → VerificationTier.TIER2_LEANAIDE
    ├─ result.confidence → 0.85
    ├─ result.tier2_result.proof_script → "theorem ..."
    └─ result.get_summary() → "Verified via Tier 2 (LeanAide)..."
```
