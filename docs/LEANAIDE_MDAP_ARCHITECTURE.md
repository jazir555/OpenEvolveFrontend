# LeanAide MDAP/MAKER Architecture

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide MDAP/MAKER Integration

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Architecture](#2-component-architecture)
3. [Multi-Agent Flow Diagrams](#3-multi-agent-flow-diagrams)
4. [Voting Flow Diagrams](#4-voting-flow-diagrams)
5. [Integration Flows](#5-integration-flows)
6. [Performance Flows](#6-performance-flows)
7. [Data Flow Diagrams](#7-data-flow-diagrams)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEANAIDE MDAP/MAKER                             │
│                    MULTI-AGENT PROOF GENERATION                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│     MDAP      │         │    MAKER      │         │    ROMA       │
│               │         │               │         │               │
│ Multi-Agent   │         │ Error         │         │ Recursive     │
│ Execution     │         │ Correction    │         │ Decomposition │
│               │         │               │         │               │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌───────────────┐             ┌───────────────┐
            │   LEANAIDE    │             │  LEAN 4       │
            │   INTEGRATION │             │  VERIFIER     │
            └───────────────┘             └───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                            ┌───────────────┐
                            │ VERIFIED PROOF│
                            └───────────────┘
```

### 1.2 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  (workflow_engine.py, demo_mdap_maker.py, etc.)              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    INTEGRATION LAYER                         │
│  (maker_workflow_integration.py, roma_mdap_maker_engine.py)  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      CORE LAYER                              │
│  (mdap_engine.py, RedFlagger, VoteAggregator)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                      │
│  (LLM clients, Lean 4 server, Cache, Database)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 MDAP Components

```
┌─────────────────────────────────────────────────────────────┐
│                      MDAPOrchestrator                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Agent       │  │ Vote        │  │ Red         │         │
│  │ Selector    │  │ Aggregator  │  │ Flagger     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                  │
│                  ┌─────────────┐                             │
│                  │   MDAP      │                             │
│                  │    Cache    │                             │
│                  └─────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

**Component Responsibilities:**

- **AgentSelector**: Selects appropriate agents for theorem type
- **VoteAggregator**: Aggregates agent outputs using various strategies
- **RedFlagger**: Filters invalid or low-quality responses
- **MDAPCache**: Caches responses for performance

### 2.2 MAKER Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MAKER Workflow Integrator                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Sequential  │  │  Parallel   │  │  Recursive  │         │
│  │   MAKER     │  │   MAKER     │  │   MAKER     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                  │
│                  ┌─────────────┐                             │
│                  │   Hybrid    │                             │
│                  │   MAKER     │                             │
│                  └─────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 ROMA-MDAP-MAKER Components

```
┌─────────────────────────────────────────────────────────────┐
│                  ROMAMDAPMakerEngine                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    ROMA     │  │    MDAP     │  │   MAKER     │         │
│  │Decomposition│  │Multi-Agent  │  │Error        │         │
│  │             │  │Execution    │  │Correction   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                  │
│                  ┌─────────────┐                             │
│                  │Hierarchical │                             │
│                  │   Voting    │                             │
│                  └─────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Agent Flow Diagrams

### 3.1 MDAP Execution Flow

```
                    Input Theorem
                          │
                          ▼
                ┌─────────────────┐
                │  Agent Selector │
                └────────┬────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │    Select k Agents (2-8)        │
        │  ├─ constructive               │
        │  ├─ inductive                  │
        │  ├─ algebraic                  │
        │  └─ ... (more agents)          │
        └───────────────┬────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Parallel Agent Execution     │
        ├────────┬────────┬────────┬─────┤
        │Agent1  │Agent2  │Agent3  │...  │
        └───┬────┴───┬────┴───┬────┴───┬┘
            │        │        │        │
            ▼        ▼        ▼        ▼
        Proof1   Proof2   Proof3   ProofK
            │        │        │        │
            └────────┴────────┴────────┘
                     │
                     ▼
          ┌──────────────────┐
          │ Vote Aggregator  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Red Flagger     │
          └────────┬─────────┘
                   │
                   ▼
            ┌──────────────┐
            │ Best Proof   │
            │(or retry)    │
            └──────────────┘
```

### 3.2 Agent Type Selection

```
                    Input Theorem
                          │
                          ▼
                ┌─────────────────┐
                │ Theorem Analyzer│
                └────────┬────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Algebra │  │ Logic   │  │Topology│
      │ Detected│  │Detected │  │Detected│
      └────┬────┘  └────┬────┘  └────┬────┘
           │            │            │
           ▼            ▼            ▼
    ┌───────────┐ ┌───────────┐ ┌───────────┐
    │algebraic  │ │indirect   │ │structural │
    │computational│ │constructive│ │constructive│
    └───────────┘ └───────────┘ └───────────┘
           │            │            │
           └────────────┼────────────┘
                        │
                        ▼
                Selected Agents
```

---

## 4. Voting Flow Diagrams

### 4.1 Majority Voting

```
        Agent Outputs
              │
              ▼
    ┌─────────────────┐
    │ Collect Proofs  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Canonicalize    │  (Remove duplicates)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Count Votes     │
    │ proof_a: 3      │
    │ proof_b: 2      │
    │ proof_c: 1      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Find Maximum    │
    │ proof_a wins    │
    └────────┬────────┘
             │
             ▼
        Winner
```

### 4.2 First-K-Ahead Voting (MAKER)

```
        Agent Outputs (Sequential)
              │
              ▼
    ┌─────────────────┐
    │ k_ahead = 3     │
    └────────┬────────┘
             │
             ▼
      ┌────────────┐
      │ Vote 1:    │ proof_a (1)
      └────────────┘
             │
             ▼
      ┌────────────┐
      │ Vote 2:    │ proof_a (2)
      └────────────┘
             │
             ▼
      ┌────────────┐
      │ Vote 3:    │ proof_b (1)
      └────────────┘
             │
             ▼
      ┌────────────┐
      │ Vote 4:    │ proof_a (3) ✓
      └────────────┘
             │
             ▼
    proof_a reached k_ahead threshold
             │
             ▼
          STOP EARLY
             │
             ▼
        Winner: proof_a
```

### 4.3 Confidence-Weighted Voting

```
        Agent Outputs (with confidence)
              │
              ▼
    ┌─────────────────┐
    │ Extract         │
    │ Confidences     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Weight Votes    │
    │ proof_a: 0.9+0.8 = 1.7
    │ proof_b: 0.7 = 0.7
    │ proof_c: 0.5 = 0.5
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Find Maximum    │
    │ proof_a wins    │
    └────────┬────────┘
             │
             ▼
        Winner: proof_a
```

---

## 5. Integration Flows

### 5.1 ROMA-MDAP-MAKER Flow

```
                Input Theorem
                      │
                      ▼
        ┌───────────────────────┐
        │  ROMA Decomposition   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Decomposition DAG    │
        │  ┌─────┐  ┌─────┐    │
        │  │Goal1│  │Goal2│    │
        │  └──┬──┘  └──┬──┘    │
        │     │        │       │
        │     └───┬────┘       │
        │         ▼            │
        │     ┌─────┐          │
        │     │Main │          │
        │     └─────┘          │
        └───────────┬──────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │Goal1  │  │Goal2  │  │Main   │
    │MDAP   │  │MDAP   │  │MDAP   │
    └───┬───┘  └───┬───┘  └───┬───┘
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
        ┌───────────────────────┐
        │  MAKER Error          │
        │  Correction           │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Hierarchical Voting  │
        │  (across levels)      │
        └───────────┬───────────┘
                    │
                    ▼
              Final Proof
```

### 5.2 Workflow Integration Flow

```
        Decomposition Workflow
              (Stage 1-2)
                │
                ▼
    ┌───────────────────────┐
    │  Sub-Problems Created │
    │  ├─ SubProblem 1      │
    │  ├─ SubProblem 2      │
    │  └─ SubProblem 3      │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Stage 3A: MDAP       │
    │  Initial Proof Gen    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Quality Check        │
    │  └─ Red-Flagging      │
    └───────────┬───────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
    [Pass]          [Fail]
        │               │
        │               ▼
        │      ┌─────────────────┐
        │      │ Stage 3B: MDAP  │
        │      │ Refinement      │
        │      └────────┬────────┘
        │               │
        └───────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │  Stage 4: Reassembly  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Stage 5: Verification│
    │  (Lean 4)             │
    └───────────┬───────────┘
                │
                ▼
          Final Proof
```

---

## 6. Performance Flows

### 6.1 Caching Flow

```
        Request
          │
          ▼
    ┌─────────────┐
    │ Cache Lookup│
    └──────┬──────┘
           │
      ┌────┴────┐
      │         │
     Hit       Miss
      │         │
      ▼         ▼
 ┌─────────┐  ┌──────────┐
 │ Return  │  │Execute   │
 │ Cached  │  │Task      │
 │ Result  │  │          │
 └─────────┘  └─────┬────┘
                   │
                   ▼
            ┌──────────┐
            │Cache     │
            │Result    │
            └─────┬────┘
                  │
                  ▼
            Return Result
```

### 6.2 Parallel Execution Flow

```
        Task Request
              │
              ▼
    ┌───────────────────┐
    │ Create k Agents   │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ Async Task Group  │
    └─────────┬─────────┘
              │
              ▼
    ┌──────────────────────────────┐
    │  agent1  agent2  agent3  ...  │
    │    │      │      │           │
    │    ▼      ▼      ▼           │
    │  task1  task2  task3  ...    │
    │    │      │      │           │
    └────┼──────┼──────┼───────────┘
         │      │      │
         └──────┴──────┘
                │
                ▼
         Gather Results
                │
                ▼
         Aggregate Votes
```

---

## 7. Data Flow Diagrams

### 7.1 Complete Data Flow

```
    Theorem Statement
          │
          ▼
    ┌─────────────┐
    │ Validation  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Parsing     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Context      │
    │Enrichment   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Agent        │
    │Selection    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Prompt       │
    │Construction│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │LLM Request  │
    │(Parallel)   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Response     │
    │Collection   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Parsing      │
    │& Validation │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Red-Flagging │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Voting       │
    │Aggregation  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Winner       │
    │Selection   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Post-Processing│
    │(formatting) │
    └──────┬──────┘
           │
           ▼
      Final Proof
```

### 7.2 Error Handling Flow

```
        Execution
           │
           ▼
    ┌──────────────┐
    │ Error Check  │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     │           │
   Error       Success
     │           │
     ▼           ▼
┌─────────┐  ┌─────────┐
│Log      │  │ Return  │
│Error    │  │ Result  │
└────┬────┘  └─────────┘
     │
     ▼
┌─────────┐
│Retry?   │
└────┬────┘
     │
  ┌──┴──┐
  │Yes  │No
  ▼     ▼
┌─────┐ ┌─────────┐
│Retry│ │Fallback │
│     │ │Strategy │
└──┬──┘ └────┬────┘
   │         │
   └────┬────┘
        │
        ▼
   Return Result
```

---

## Appendix: Key Metrics

### Performance Metrics

- **Cache Hit Rate**: Target > 60%
- **Parallel Speedup**: 3-5x for k=5 agents
- **Voting Duration**: < 1 second for k=8
- **Red-Flagging Duration**: < 100ms per response
- **End-to-End Latency**: 10-30 seconds (typical)

### Quality Metrics

- **Red-Flag Detection Rate**: > 95%
- **Voting Agreement**: 60-80% (typical)
- **Proof Verification Rate**: > 90% (with red-flagging)
- **Confidence Calibration**: ±0.1 (target)

---

**Document End**

For more information, see:
- `LEANAIDE_MDAP_MAKER_GUIDE.md` - Complete usage guide
- `LEANAIDE_MDAP_MAKER_API.md` - API reference
- `LEANAIDE_MDAP_MAKER_EXAMPLES.md` - Real-world examples
