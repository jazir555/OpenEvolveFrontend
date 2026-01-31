# 3-Round Gauntlet System - Architecture Diagrams

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THREE-ROUND GAUNTLET SYSTEM                      │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   LoongFlow  │    │   Red Team   │    │  Gold Team   │          │
│  │   Round 1    │───▶│   Round 2    │───▶│   Round 3    │          │
│  │              │    │              │    │              │          │
│  │  - AI Eval   │    │ - Adversarial│    │ - Consensus  │          │
│  │  - 20% Wgt   │    │ - 30% Wgt    │    │ - 50% Wgt    │          │
│  │  - Fast (30s)│    │ - Med (2min) │    │ - Slow (5min)│          │
│  └───────┬──────┘    └──────┬───────┘    └──────┬───────┘          │
│          │                  │                    │                  │
│          ▼                  ▼                    ▼                  │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │           ORCHESTRATION & FILTERING LAYER                 │    │
│  │                                                           │    │
│  │  Progressive Filtering:                                   │    │
│  │  - Round 1: Score ≥ 0.5?                                  │    │
│  │  - Round 2: Score ≥ 0.6?                                  │    │
│  │  - Round 3: Score ≥ 0.7?                                  │    │
│  │  - Early termination on failure                           │    │
│  │                                                           │    │
│  │  Score Aggregation:                                       │    │
│  │  Final = (R1×0.2 + R2×0.3 + R3×0.5)                      │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              CONFIGURATION & DOMAIN TUNING                │    │
│  │                                                           │    │
│  │  Domain Presets:                                          │    │
│  │  - Finance:  Strict (0.7-0.9)                            │    │
│  │  - Science:  Moderate (0.5-0.7)                          │    │
│  │  - Web:      Lenient (0.3-0.6)                           │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Evaluation Flow Diagram

```
SOLUTION INPUT
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ ROUND 1: LOONGFLOW AI EVALUATION                           │
│ ─────────────────────────────────────────────────────────  │
│ Evaluator: LoongFlow GeneralEvaluator                        │
│ Metrics:                                                    │
│   - Correctness: AI reasoning                               │
│   - Quality: Code/Content assessment                        │
│   - Completeness: Requirement coverage                      │
│                                                              │
│ Returns: Score (0-1), Confidence, Feedback                  │
│                                                              │
│ Time: 10-30 seconds                                         │
│ Weight: 20%                                                 │
│ Threshold: 0.5 (configurable)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Score ≥ Threshold?
                       │
           ┌───────────┴───────────┐
           │ NO                   │ YES
           ▼                       ▼
    ┌─────────────┐      ┌─────────────────────┐
    │ TERMINATE   │      │ PROCEED TO ROUND 2  │
    │ Early exit  │      │ Continue evaluation │
    └─────────────┘      └─────────┬───────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│ ROUND 2: RED TEAM ADVERSARIAL EVALUATION                   │
│ ─────────────────────────────────────────────────────────  │
│ Evaluator: Red Team Attack System                           │
│ Metrics:                                                    │
│   - Robustness: Attack survival rate                        │
│   - Edge Cases: Corner case handling                       │
│   - Error Handling: Failure recovery                        │
│                                                              │
│ Attack Vectors (by domain):                                 │
│   Finance: Market crash, liquidity crisis, volatility       │
│   Science: Outliers, noise, parameter variations            │
│   Web: SQL injection, XSS, accessibility                    │
│                                                              │
│ Returns: Score (0-1), Attack success rate, Robustness       │
│                                                              │
│ Time: 1-2 minutes                                           │
│ Weight: 30%                                                 │
│ Threshold: 0.6 (configurable)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Score ≥ Threshold?
                       │
           ┌───────────┴───────────┐
           │ NO                   │ YES
           ▼                       ▼
    ┌─────────────┐      ┌─────────────────────┐
    │ TERMINATE   │      │ PROCEED TO ROUND 3  │
    │ Early exit  │      │ Final verification  │
    └─────────────┘      └─────────┬───────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│ ROUND 3: GOLD TEAM CONSENSUS VERIFICATION                  │
│ ─────────────────────────────────────────────────────────  │
│ Evaluator: Multi-Model Consensus System                     │
│ Metrics:                                                    │
│   - Consensus: Inter-evaluator agreement                    │
│   - Quality: Multi-dimensional assessment                  │
│   - Formal: Lean 4 verification (if applicable)            │
│                                                              │
│ Evaluators (by domain):                                    │
│   Finance: Analyst, Risk manager, Quant, Compliance         │
│   Science: Expert, Methodologist, Statistician              │
│   Web: UX designer, Frontend/backend, Accessibility         │
│                                                              │
│ Returns: Score (0-1), Consensus level, Verification status  │
│                                                              │
│ Time: 3-5 minutes                                           │
│ Weight: 50%                                                 │
│ Threshold: 0.7 (configurable)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ FINAL SCORE    │
              │ Calculation:   │
              │ Weighted avg   │
              │ of all rounds  │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   REPORT       │
              │ Generation     │
              └────────────────┘
```

## Configuration System Architecture

```
ThreeRoundConfig
│
├─ Round 1 Configuration
│  ├─ LLM Config (model, api_key, temperature)
│  ├─ Timeout (seconds)
│  ├─ Weight (0.0-1.0)
│  ├─ Threshold (0.0-1.0)
│  └─ Enabled (bool)
│
├─ Round 2 Configuration
│  ├─ Attack Vectors (list)
│  ├─ Attack Intensity (low/moderate/high/extreme)
│  ├─ Timeout (seconds)
│  ├─ Weight (0.0-1.0)
│  ├─ Threshold (0.0-1.0)
│  └─ Enabled (bool)
│
├─ Round 3 Configuration
│  ├─ Evaluators (list of specialized evaluators)
│  ├─ Consensus Threshold (0.0-1.0)
│  ├─ Formal Verification (bool)
│  ├─ Timeout (seconds)
│  ├─ Weight (0.0-1.0)
│  ├─ Threshold (0.0-1.0)
│  └─ Enabled (bool)
│
└─ Global Settings
   ├─ Early Termination (bool)
   ├─ Parallel Execution (bool)
   ├─ Aggregate Artifacts (bool)
   └─ Generate Detailed Report (bool)
```

## Data Flow Diagram

```
┌─────────────┐
│  Solution   │
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│          ThreeRoundGauntletOrchestrator            │
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │  Configuration Setup                         │  │
│  │  - Domain-specific config                   │  │
│  │  - Threshold validation                     │  │
│  │  - Evaluator initialization                 │  │
│  └─────────────────┬───────────────────────────┘  │
│                    │                              │
│                    ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  Round 1 Execution                          │  │
│  │  - Call LoongFlow evaluator                │  │
│  │  - Extract score, confidence, feedback      │  │
│  │  - Collect artifacts                        │  │
│  │  - Measure time                             │  │
│  └─────────────────┬───────────────────────────┘  │
│                    │                              │
│                    ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  Progressive Filtering Check                │  │
│  │  - Score ≥ Threshold?                       │  │
│  │  - Early termination if enabled             │  │
│  └─────────────────┬───────────────────────────┘  │
│                    │                              │
│           No ───────┴─────── Yes                 │
│           │                      │               │
│           ▼                      │               │
│      ┌────────┐                  │               │
│      │ Return │                  │               │
│      │Result  │                  │               │
│      └────────┘                  │               │
│                                  │               │
│                                  ▼               │
│                    ┌─────────────────────────┐  │
│                    │  Round 2 Execution      │  │
│                    │  - Red Team attacks     │  │
│                    │  - Robustness scoring    │  │
│                    └───────────┬─────────────┘  │
│                                │                 │
│                                ▼                 │
│                    ┌─────────────────────────┐  │
│                    │  Filtering Check        │  │
│                    └───────────┬─────────────┘  │
│                                │                 │
│                       No ───────┴─────── Yes     │
│                       │                      │    │
│                       ▼                      │    │
│                  ┌────────┐                 │    │
│                  │ Return │                 │    │
│                  │Result  │                 │    │
│                  └────────┘                 │    │
│                                            │    │
│                                            ▼    │
│                              ┌─────────────────┐ │
│                              │  Round 3        │ │
│                              │  Execution      │ │
│                              │  - Consensus    │ │
│                              │  - Verification │ │
│                              └───────┬─────────┘ │
│                                      │           │
│                                      ▼           │
│                              ┌─────────────────┐ │
│                              │  Calculate      │ │
│                              │  Final Score    │ │
│                              └───────┬─────────┘ │
│                                      │           │
│                                      ▼           │
│                              ┌─────────────────┐ │
│                              │  Generate       │ │
│                              │  Report         │ │
│                              └───────┬─────────┘ │
│                                      │           │
└──────────────────────────────────────┼───────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  FullGauntlet   │
                              │  Result         │
                              │                 │
                              │  - passed       │
                              │  - final_score  │
                              │  - rounds       │
                              │  - artifacts    │
                              │  - report       │
                              └─────────────────┘
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION ARCHITECTURE                    │
│                                                                 │
│  ┌──────────────┐                                              │
│  │  OpenEvolve  │                                              │
│  │  Evolution   │                                              │
│  │  Engine      │                                              │
│  └──────┬───────┘                                              │
│         │                                                      │
│         │ Uses for population filtering                        │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐     │
│  │     Three-Round Gauntlet Orchestrator                │     │
│  │                                                       │     │
│  │  ┌─────────────────────────────────────────────┐    │     │
│  │  │  Round 1: LoongFlow Evaluator               │    │     │
│  │  │  - Uses: evaluators/loongflow_adapter.py   │    │     │
│  │  └─────────────────────────────────────────────┘    │     │
│  │                                                       │     │
│  │  ┌─────────────────────────────────────────────┐    │     │
│  │  │  Round 2: Red Team Evaluator                │    │     │
│  │  │  - Placeholder for future implementation   │    │     │
│  │  └─────────────────────────────────────────────┘    │     │
│  │                                                       │     │
│  │  ┌─────────────────────────────────────────────┐    │     │
│  │  │  Round 3: Gold Team Evaluator               │    │     │
│  │  │  - Placeholder for future implementation   │    │     │
│  │  └─────────────────────────────────────────────┘    │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          │ Artifacts & Feedback              │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       Knowledge Engine                               │  │
│  │                                                       │  │
│  │  - Extract patterns from evaluations                 │  │
│  │  - Track gauntlet effectiveness                      │  │
│  │  - Recommend strategy improvements                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       BubbleLab Integration                           │  │
│  │                                                       │  │
│  │  - Gauntlet service integration                       │  │
│  │  - Bubble orchestration                              │  │
│  │  - Workflow management                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Domain-Specific Configurations

```
CONFIGURATION PRESETS
│
├─ FINANCE (Strict)
│  ├─ Round 1: Threshold 0.7, Weight 20%
│  ├─ Round 2: Threshold 0.8, Weight 30%
│  └─ Round 3: Threshold 0.9, Weight 50%
│     Use: Trading, risk management, portfolio optimization
│
├─ SCIENCE (Moderate)
│  ├─ Round 1: Threshold 0.5, Weight 20%
│  ├─ Round 2: Threshold 0.6, Weight 30%
│  └─ Round 3: Threshold 0.7, Weight 50%
│     Use: Experimental design, data analysis, research
│
└─ WEB (Lenient)
   ├─ Round 1: Threshold 0.3, Weight 20%
   ├─ Round 2: Threshold 0.5, Weight 30%
   └─ Round 3: Threshold 0.6, Weight 50%
      Use: Frontend, backend, full-stack development
```

## Report Structure

```
COMPREHENSIVE GAUNTLET REPORT
│
├─ Executive Summary
│  ├─ Status (Complete/Terminated)
│  ├─ Final Score
│  ├─ Rounds Completed
│  └─ Termination Reason (if applicable)
│
├─ Round Results
│  ├─ Round 1 (LoongFlow)
│  │  ├─ Passed/Failed
│  │  ├─ Score
│  │  ├─ Confidence
│  │  ├─ Evaluation Time
│  │  └─ Feedback
│  │
│  ├─ Round 2 (Red Team)
│  │  ├─ Passed/Failed
│  │  ├─ Score
│  │  ├─ Robustness Score
│  │  ├─ Attacks Attempted/Successful
│  │  ├─ Evaluation Time
│  │  └─ Feedback
│  │
│  └─ Round 3 (Gold Team)
│     ├─ Passed/Failed
│     ├─ Score
│     ├─ Consensus Score
│     ├─ Formal Verification Status
│     ├─ Evaluation Time
│     └─ Feedback
│
├─ Performance Metrics
│  ├─ Total Time
│  ├─ Per-Round Times
│  └─ Resource Usage
│
└─ Recommendations
   ├─ Strengths
   ├─ Weaknesses
   └─ Improvement Suggestions
```

---

**All diagrams illustrate the complete 3-Round Gauntlet Orchestrator system**
