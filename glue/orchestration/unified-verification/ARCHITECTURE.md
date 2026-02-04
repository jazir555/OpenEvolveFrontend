# Architecture: Unified Verification Orchestrator

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATION                                   │
│  - Problem specification                                                     │
│  - Verification constraints                                                  │
│  - Options (strategy, confidence threshold, etc.)                           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UNIFIED VERIFICATION ORCHESTRATOR                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  API Layer                                                             │  │
│  │  - verify(problem, constraints, options)                              │  │
│  │  - verifyWithCrossValidation(problem, options)                        │  │
│  │  - verifyBatch(problems, constraints, options)                        │  │
│  │  - getStatistics()                                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Strategy Selector                                                     │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Analyze Problem Type                                        │  │  │
│  │  │    ├─ SMT_CONSTRAINTS       → Z3 (95% success)                 │  │  │
│  │  │    ├─ THEOREM_PROVING       → LeanAide (92% success)           │  │  │
│  │  │    ├─ FORMAL_VERIFICATION   → Parallel (cross-validation)      │  │  │
│  │  │    ├─ CODE_CORRECTNESS      → Hybrid (Z3 → LeanAide)          │  │  │
│  │  │    ├─ MODEL_CHECKING        → Z3 (90% success)                 │  │  │
│  │  │    └─ SAT_SOLVING           → Z3 (98% success)                 │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2. Select Execution Strategy                                   │  │  │
│  │  │    ├─ z3_only: Z3 only                                        │  │  │
│  │  │    ├─ leanaide_only: LeanAide only                            │  │  │
│  │  │    ├─ parallel: Both simultaneously                            │  │  │
│  │  │    ├─ sequential: Z3 first, then LeanAide                      │  │  │
│  │  │    └─ hybrid: Adaptive approach                               │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Cross Validator                                                       │  │
│  │                                                                        │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                  │  │
│  │  │                      │  │                      │                  │  │
│  │  │  Z3 Execution Path   │  │ LeanAide Exec Path   │                  │  │
│  │  │                      │  │                      │                  │  │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │                  │  │
│  │  │ │ Circuit Breaker  │ │  │ │ Circuit Breaker  │ │                  │  │
│  │  │ └────────┬─────────┘ │  │ └────────┬─────────┘ │                  │  │
│  │  │          │            │  │          │            │                  │  │
│  │  │ ▼        │            │  │ ▼        │            │                  │  │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │                  │  │
│  │  │ │   HTTP Request   │ │  │ │   HTTP Request   │ │                  │  │
│  │  │ │   + Timeout      │ │  │ │   + Timeout      │ │                  │  │
│  │  │ └────────┬─────────┘ │  │ └────────┬─────────┘ │                  │  │
│  │  │          │            │  │          │            │                  │  │
│  │  │ ▼        │            │  │ ▼        │            │                  │  │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │                  │  │
│  │  │ │  Retry Logic     │ │  │ │  Retry Logic     │ │                  │  │
│  │  │ │  (Backoff)       │ │  │ │  (Backoff)       │ │                  │  │
│  │  │ └────────┬─────────┘ │  │ └────────┬─────────┘ │                  │  │
│  │  └──────────┼───────────┘  └──────────┼───────────┘                  │  │
│  │             │                         │                               │  │
│  └─────────────┼─────────────────────────┼───────────────────────────────┘  │
│                │                         │                                  │
│                ▼                         ▼                                  │
│         ┌─────────────┐           ┌─────────────┐                            │
│         │  Z3 Result  │           │LeanAide Rslt│                            │
│         │             │           │             │                            │
│         │ • verified  │           │ • verified  │                            │
│         │ • conf.     │           │ • conf.     │                            │
│         │ • output    │           │ • output    │                            │
│         │ • proof     │           │ • proof     │                            │
│         │ • execTime  │           │ • execTime  │                            │
│         └─────────────┘           └─────────────┘                            │
│                │                         │                                  │
│                └────────────┬────────────┘                                  │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Comparison & Conflict Detection                                     │  │
│  │                                                                       │  │
│  │  • Compare verification outcomes                                      │  │
│  │  • Calculate confidence alignment                                     │  │
│  │  • Detect disagreements (4 types)                                     │  │
│  │  • Determine resolution (5 outcomes)                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Confidence Aggregator                                                │  │
│  │                                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Normalize Scores                                             │  │  │
│  │  │    • Historical accuracy                                        │  │  │
│  │  │    • Problem type match                                         │  │  │
│  │  │    • Execution quality                                          │  │  │
│  │  │    • Confidence consistency                                     │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2. Calculate Dynamic Weights                                     │  │  │
│  │  │    • Base strategy weights                                      │  │  │
│  │  │    • Success/failure adjustment                                 │  │  │
│  │  │    • Confidence level adjustment                                │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3. Combine Scores (Weighted Average)                            │  │  │
│  │  │    • Combined confidence: 0.95                                  │  │  │
│  │  │    • Confidence level: very_high                                │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 4. Generate Evidence Trail                                      │  │  │
│  │  │    • Per-system contributions                                   │  │  │
│  │  │    • Normalization factors                                      │  │  │
│  │  │    • Cross-validation agreement                                 │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Learning Feedback Loop                                              │  │
│  │                                                                       │  │
│  │  • Update strategy effectiveness (exponential moving average)        │  │
│  │  • Update confidence aggregator accuracy                             │  │
│  │  • Store results for analysis                                        │  │
│  │  • Adapt future strategy selection                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
         ┌──────────┐       ┌──────────┐       ┌──────────┐
         │   Z3     │       │LeanAide  │       │ Learning │
         │   API    │       │   API    │       │  Storage │
         └──────────┘       └──────────┘       └──────────┘
         (Core Project)    (Core Project)    (Vector DB +
                                                          Graphiti)
```

## Data Flow

### 1. Simple Verification Flow

```
User Request
    ↓
[Orchestrator.verify()]
    ↓
Strategy Selection (auto)
    ↓
Cross Validator (single system)
    ↓
Z3 API Request
    ↓
Normalization (to canonical format)
    ↓
Confidence Aggregation (single system)
    ↓
Result Storage
    ↓
Learning Update
    ↓
Return Result
```

### 2. Cross-Validation Flow

```
User Request
    ↓
[Orchestrator.verifyWithCrossValidation()]
    ↓
Strategy Selection (auto or manual)
    ↓
Cross Validator (both systems)
    ├─→ Z3 API Request
 │     ├─ Circuit Breaker Check
 │     ├─ HTTP Request (with timeout)
 │     └─ Retry Logic (if needed)
 │
 └─→ LeanAide API Request
       ├─ Circuit Breaker Check
       ├─ HTTP Request (with timeout)
       └─ Retry Logic (if needed)
    ↓
Result Comparison
    ├─ Agreement Detection
    └─ Conflict Detection
    ↓
Confidence Aggregation
    ├─ Normalize Scores
    ├─ Calculate Weights
    ├─ Combine Scores
    └─ Generate Evidence
    ↓
Resolution Determination
    ↓
Result Storage
    ↓
Learning Update (both systems)
    ↓
Return CrossValidationResult
```

### 3. Batch Verification Flow

```
Batch Request (N problems)
    ↓
[Orchestrator.verifyBatch()]
    ↓
Concurrency Control (5 parallel)
    ↓
For each batch:
    ├─ Problem 1: verify() ─┐
    ├─ Problem 2: verify() ─┤
    ├─ Problem 3: verify() ─┤ Parallel
    ├─ Problem 4: verify() ─┤ (max 5)
    └─ Problem 5: verify() ─┘
    ↓
Aggregate Results
    ↓
Return Map<ProblemID, Result>
```

## Component Interactions

### Strategy Selector → Cross Validator

```
Strategy Selector analyzes problem
    ↓
Returns StrategySelection {
        strategy: 'parallel' | 'sequential' | 'hybrid' | ...
        systems: ['z3', 'leanaide']
        expectedConfidence: 0.95
        reasoning: "..."
    }
    ↓
Cross Validator uses strategy to execute
```

### Cross Validator → Confidence Aggregator

```
Cross Validator gets results from systems
    ↓
Returns SystemResult[] {
        [0]: { system: 'z3', verified: true, confidence: 0.95, ... }
        [1]: { system: 'leanaide', verified: true, confidence: 0.90, ... }
    }
    ↓
Confidence Aggregator normalizes and combines
    ↓
Returns ConfidenceScore {
        combined: 0.93
        individual: { z3: 0.95, leanaide: 0.90 }
        weights: { z3: 0.52, leanaide: 0.48 }
        evidence: [...]
    }
```

### Orchestrator → Learning System

```
Orchestrator receives VerificationResult
    ↓
Extracts metrics:
    - system: 'z3'
    - strategy: 'parallel'
    - problemType: 'SMT_CONSTRAINTS'
    - verified: true
    - confidence: 0.95
    - executionTime: 1500
    ↓
Updates Strategy Selector effectiveness
    ↓
Updates Confidence Aggregator accuracy
    ↓
Stores result (for future learning)
```

## Error Handling Flow

```
HTTP Request
    ↓
Error Detected
    ↓
Error Type Classification
    ├─ Transient (network blip)
    │    └─→ Retry with exponential backoff
    │        └─→ Max 3 retries
    │
    ├─ Logic (bad data)
    │    └─→ Log to Dead Letter Queue
    │        └─→ Continue if possible
    │
    └─ System (target down)
         └─→ Circuit Breaker
             ├─→ Stop requests for 30s
             └─→ Try other system if available
    ↓
Return partial or error result
    ↓
Include error details in response
```

## Confidence Calculation Example

```
Z3 Result:
  - verified: true
  - confidence: 0.95
  - executionTime: 1500ms
  - no errors

LeanAide Result:
  - verified: true
  - confidence: 0.88
  - executionTime: 2000ms
  - no errors

Step 1: Normalize
  Z3: 0.95 × (0.90 × 0.30 + 0.95 × 0.30 + 1.00 × 0.20 + 1.00 × 0.20) = 0.93
  LeanAide: 0.88 × (0.88 × 0.30 + 0.92 × 0.30 + 1.00 × 0.20 + 0.80 × 0.20) = 0.82

Step 2: Calculate Weights
  Z3: 0.6 × 1.2 × (0.5 + 0.95) = 1.04
  LeanAide: 0.4 × 1.2 × (0.5 + 0.88) = 0.66

  Normalize: Z3 = 0.61, LeanAide = 0.39

Step 3: Combine
  Combined = 0.93 × 0.61 + 0.82 × 0.39 = 0.89

Step 4: Evidence
  [
    { source: 'z3_verification', weight: 0.61, ... },
    { source: 'leanaide_verification', weight: 0.39, ... },
    { source: 'cross_validation', weight: 0.20, ... }
  ]

Final Confidence: 0.89 (high)
```

## Strategy Selection Decision Tree

```
Start
  ↓
What is the problem type?
  ├─ SMT_CONSTRAINTS
  │   └─→ Z3 only (95% success rate)
  │
  ├─ THEOREM_PROVING
  │   └─→ LeanAide only (92% success rate)
  │
  ├─ FORMAL_VERIFICATION
  │   └─→ What confidence is required?
  │       ├─ ≥0.95 → Parallel (cross-validation)
  │       └─ <0.95 → Sequential (Z3 first)
  │
  ├─ CODE_CORRECTNESS
  │   └─→ Hybrid (Z3 → LeanAide if needed)
  │
  ├─ MODEL_CHECKING
  │   └─→ Z3 only (90% success rate)
  │
  └─ SAT_SOLVING
      └─→ Z3 only (98% success rate)
```

## Agreement Type Decision Matrix

```
                    Z3: Verified        Z3: Not Verified
                        │                     │
LeanAide: Verified  ────┼─────────────────────┼────────────
                        │                     │
                        │ Full Agreement      │ Disagreement
                        │ (both verified)     │ (conflict)
                        │                     │
LeanAide: Not Verified ──┼─────────────────────┼────────────
                        │                     │
                        │ Disagreement        │ Full Agreement
                        │ (conflict)          │ (both failed)
                        │                     │

Plus confidence alignment check:
  - Alignment >0.9: Full/Partial Agreement
  - Alignment 0.7-0.9: Partial Agreement
  - Alignment <0.7: Disagreement/Inconclusive
```

## Learning Feedback Loop

```
Verification Outcome
    ↓
Extract Metrics
    ├─ System used
    ├─ Problem type
    ├─ Success/Failure
    ├─ Confidence level
    ├─ Execution time
    └─ Strategy used
    ↓
Update Historical Data
    ├─ Strategy effectiveness
    │   └─ Exponential moving average (α=0.1)
    │
    ├─ System accuracy
    │   └─ Exponential moving average (α=0.1)
    │
    └─ Execution statistics
        └─ Rolling average
    ↓
Store Results (Vector DB + Graphiti)
    ├─ Semantic indexing
    ├─ Lineage tracking
    └─ Performance metrics
    ↓
Future Selections Improved
    ├─ Better strategy selection
    ├─ More accurate confidence estimates
    └─ Optimized execution plans
```

## Environment Configuration

```bash
# Required
Z3_URL="http://localhost:8080"           # Z3 endpoint
LEANAIDE_URL="http://localhost:8081"     # LeanAide endpoint

# Optional
Z3_TIMEOUT="30000"                      # Z3 timeout (ms)
LEANAIDE_TIMEOUT="45000"                # LeanAide timeout (ms)
Z3_HEALTH_CHECK="/health"               # Z3 health path
LEANAIDE_HEALTH_CHECK="/health"         # LeanAide health path
Z3_VERIFY_PATH="/verify"                # Z3 verify path
LEANAIDE_VERIFY_PATH="/verify"          # LeanAide verify path
DEBUG="true"                            # Enable debug logging
```

## Federation Constitution Compliance Matrix

| Law | Implementation | Verification |
|-----|----------------|--------------|
| Air Gap | All code in glue/; no imports from core-projects | Code review |
| Runtime Truth | 3 probe scripts verify APIs before use | `npm run probes` |
| Untouchable DB | Read-only access when integrated | Contract tests |
| Idempotency | Safe retries; check-before-create | Unit tests |
| Configuration Explicitness | All URLs/ports via env vars | Startup validation |
| UTC | All timestamps in UTC ISO-8601 | Log inspection |
