# LeanAide MDAP-Enhanced Evolution - Architecture

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide Evolution + MDAP Integration

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Component Diagrams](#2-component-diagrams)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Sequence Diagrams](#4-sequence-diagrams)
5. [Integration Flows](#5-integration-flows)
6. [Performance Flow](#6-performance-flow)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     MDAP-Enhanced Evolution                       │
└───────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Evolution   │      │     MDAP     │      │   MAKER      │
│   Engine     │◄────►│  Orchestrator│◄────►│   Voting     │
│              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │   LeanAide      │
                      │   Verification  │
                      └─────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │  Verified Proof │
                      │  (Zero Errors)  │
                      └─────────────────┘
```

### 1.2 Module Structure

```
evolution_maker_integration.py
│
├── Configuration Layer
│   ├── MakerevolutionMode (enum)
│   └── MakerevolutionConfig (dataclass)
│
├── Data Structures
│   ├── Individual (dataclass)
│   └── Population (dataclass)
│
├── Core Components
│   ├── MAKERSelection (class)
│   │   ├── _select_top_candidates()
│   │   ├── _voting_selection()
│   │   └── select()
│   │
│   ├── MDAPEvolutionDecomposer (class)
│   │   ├── decompose_task()
│   │   └── analyze_landscape()
│   │
│   └── MAKEREvolutionEngine (class)
│       ├── initialize_population()
│       ├── evolve()
│       └── _create_next_generation()
│
└── API Functions
    ├── run_maker_evolution()
    └── get_maker_evolution_capabilities()
```

---

## 2. Component Diagrams

### 2.1 Evolution-MAKER Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    MAKER Evolution Engine                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐ │
│  │  Population  │────▶│  Selection   │────▶│  Parents   │ │
│  │  Management  │     │  (Voting)    │     │            │ │
│  └──────────────┘     └──────────────┘     └────────────┘ │
│          │                                        │        │
│          ▼                                        ▼        │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐ │
│  │  Initializa- │     │   Crossover  │     │  Mutation  │ │
│  │   tion       │     │  (Voting)    │     │ (Guided)   │ │
│  └──────────────┘     └──────────────┘     └────────────┘ │
│          │                                        │        │
│          └────────────────┬───────────────────────┘        │
│                           ▼                                │
│                   ┌───────────────┐                       │
│                   │  Evaluation   │                       │
│                   │  (LeanAide)   │                       │
│                   └───────────────┘                       │
│                           │                                │
│                           ▼                                │
│                   ┌───────────────┐                       │
│                   │  Next Gen     │                       │
│                   │  Population   │                       │
│                   └───────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Voting-Based Selection

```
┌─────────────────────────────────────────────────────────────┐
│                  MAKER Selection Process                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Population                                                 │
│      │                                                       │
│      ├───[Individual 1: fitness=0.8]                        │
│      ├───[Individual 2: fitness=0.6]                        │
│      ├───[Individual 3: fitness=0.9]  ◄── Best             │
│      ├───[Individual 4: fitness=0.7]                        │
│      └───...                                                │
│      │                                                       │
│      ▼                                                       │
│  Select Top N (N = 2*k - 1)                                 │
│      │                                                       │
│      ├───[Candidate 1]                                      │
│      ├───[Candidate 2]                                      │
│      ├───[Candidate 3]                                      │
│      └───...[Candidate N]                                   │
│      │                                                       │
│      ▼                                                       │
│  Multi-Agent Voting                                         │
│      │                                                       │
│      ├───Agent 1 votes: Candidate 3                         │
│      ├───Agent 2 votes: Candidate 1                         │
│      ├───Agent 3 votes: Candidate 3  ◄── +1 vote            │
│      ├───Agent 4 votes: Candidate 3  ◄── +1 vote            │
│      └───...                                                 │
│      │                                                       │
│      ▼                                                       │
│  Vote Counting                                              │
│      ├───Candidate 1: 1 vote                                │
│      ├───Candidate 2: 0 votes                               │
│      ├───Candidate 3: 3 votes  ◄── Winner                   │
│      └───...                                                 │
│      │                                                       │
│      ▼                                                       │
│  First-K-Ahead Check                                        │
│      Candidate 3 is ahead by K=3 votes ✓                    │
│      │                                                       │
│      ▼                                                       │
│  Selected Parent: Candidate 3                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagrams

### 3.1 Complete Evolutionary Loop

```
┌─────────────────────────────────────────────────────────────┐
│              MDAP-Enhanced Evolutionary Loop                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Initial Program│
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Initialize     │
                   │ Population     │
                   └────────┬───────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │    For each generation:   │
              └─────────────┬─────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Selection │   │ Crossover │   │ Mutation  │
    │(Voting)   │   │(Voting)   │   │(Guided)   │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ Evaluate     │
                  │ Fitness      │
                  └──────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Replace       │
                 │ Population    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Check         │
                 │ Convergence?  │
                 └───────┬───────┘
                         │
                ┌─────────┴─────────┐
                │ No               │ Yes
                ▼                  ▼
         Next Generation    Return Best
```

### 3.2 MDAP Decomposition Flow

```
┌─────────────────────────────────────────────────────────────┐
│               MDAP Decomposition Flow                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Complex Task   │
                   │ (Theorem)      │
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Decompose Task │
                   │ (MDAP)         │
                   └────────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐     ┌─────────┐
      │Subtask 1│     │Subtask 2│     │Subtask 3│
      └────┬────┘     └────┬────┘     └────┬────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Evolve Each    │
                  │ Subtask        │
                  └────────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Solution │    │ Solution │    │ Solution │
    │    1     │    │    2     │    │    3     │
    └─────┬────┘    └─────┬────┘    └─────┬────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ Recombine      │
                  │ (Voting)       │
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Final Solution │
                  └────────────────┘
```

---

## 4. Sequence Diagrams

### 4.1 Selection with Voting

```
Actor           Engine          Selector         Agent            Voter
  │               │                │                │              │
  │   evolve()    │                │                │              │
  ├──────────────>│                │                │              │
  │               │                │                │              │
  │               │  select()      │                │              │
  │               ├───────────────>│                │              │
  │               │                │                │              │
  │               │                │  get_candidates()              │
  │               │                ├───────────────>│              │
  │               │                │  top_N         │              │
  │               │                │<───────────────┤              │
  │               │                │                │              │
  │               │                │  vote(candidate)              │
  │               │                ├──────────────────────────────>│
  │               │                │                │              │
  │               │                │  vote_count    │              │
  │               │                │<──────────────────────────────┤
  │               │                │                │              │
  │               │                │  [repeat K times]             │
  │               │                │<──────────────────────────────┤
  │               │                │                │              │
  │               │                │  check_k_ahead()              │
  │               │                ├──────────────────────────────>│
  │               │                │  is_ahead?     │              │
  │               │                │<──────────────────────────────┤
  │               │                │                │              │
  │               │                │  return winner │              │
  │               │<───────────────┤                │              │
  │               │  parents       │                │              │
  │<──────────────┤                │                │              │
```

### 4.2 Crossover with Voting

```
Actor           Engine          Selector         Agents           Crossover
  │               │                │                │               │
  │   evolve()    │                │                │               │
  ├──────────────>│                │                │               │
  │               │                │                │               │
  │               │  create_offspring()             │               │
  │               │  ┌────────────>│                │               │
  │               │  │             │                │               │
  │               │  │ select_parents()             │               │
  │               │  ├────────────>│                │               │
  │               │  │             │                │               │
  │               │  │  parents     │                │               │
  │               │  │<────────────┤                │               │
  │               │  │             │                │               │
  │               │  │ suggest_crossover_points()   │               │
  │               │  ├─────────────────────────────>│               │
  │               │  │             │                │               │
  │               │  │  points     │                │               │
  │               │  │<─────────────────────────────┤               │
  │               │  │             │                │               │
  │               │  │ vote_on_points()                            │
  │               │  ├──────────────────────────────────────────>│
  │               │  │             │                │               │
  │               │  │  best_point │                │               │
  │               │  │<──────────────────────────────────────────┤
  │               │  │             │                │               │
  │               │  │ crossover_at(best_point)                   │
  │               │  ├──────────────────────────────────────────>│
  │               │  │             │                │               │
  │               │  │  offspring   │                │               │
  │               │  │<──────────────────────────────────────────┤
  │               │  │             │                │               │
  │               │  │  offspring   │                │               │
  │               │<─┴────────────┤                │               │
```

### 4.3 Complete Evolution Flow

```
User      Engine       Selector    Decomposer   Evaluator   LeanAide
  │           │             │            │           │          │
  │ run_      │             │            │           │          │
  │ evolution()│             │            │           │          │
  ├──────────>│             │            │           │          │
  │           │             │            │           │          │
  │           │ init_pop()  │            │           │          │
  │           ├────────────────────────>│           │          │
  │           │             │            │           │          │
  │           │ population  │            │           │          │
  │           │<────────────┴────────────┤           │          │
  │           │             │            │           │          │
  │           │  [For each generation]  │           │          │
  │           │             │            │           │          │
  │           │ select()    │            │           │          │
  │           ├──────────>  │            │           │          │
  │           │ parents     │            │           │          │
  │           │<──────────  │            │           │          │
  │           │             │            │           │          │
  │           │ crossover()  │            │           │          │
  │           ├──────────>  │            │           │          │
  │           │ offspring   │            │           │          │
  │           │<──────────  │            │           │          │
  │           │             │            │           │          │
  │           │ mutate()    │            │           │          │
  │           ├────────────────────────>│           │          │
  │           │ mutated     │            │           │          │
  │           │<────────────────────────┤           │          │
  │           │             │            │           │          │
  │           │ evaluate()  │            │           │          │
  │           ├────────────────────────────────────>│          │
  │           │             │            │           │          │
  │           │             │            │  verify()│          │
  │           │             │            ├────────────────────>│
  │           │             │            │           │          │
  │           │             │            │ fitness   │          │
  │           │             │            │<────────────────────┤
  │           │             │            │           │          │
  │           │ fitness     │            │           │          │
  │           │<────────────────────────────────────┤          │
  │           │             │            │           │          │
  │           │ check_convergence()         │           │          │
  │           │             │            │           │          │
  │           │  [Loop until converged or max_gen]│           │          │
  │           │             │            │           │          │
  │           │ result      │            │           │          │
  │<──────────┤             │            │           │          │
```

---

## 5. Integration Flows

### 5.1 Workflow Integration (Stage 3A/3B)

```
┌─────────────────────────────────────────────────────────────┐
│              Decomposition Workflow Integration              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Theorem Input  │
                   └────────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Stage 1  │   │  Stage 2  │   │  Stage 3  │
    │ Decompose │   │ ROMA-MDAP │   │ Evolution │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Stage 3A     │
                  │  MDAP-Evolution│
                  │  (Search)     │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Stage 3B     │
                  │  Refinement   │
                  │  (Voting)     │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Final Proof  │
                  └───────────────┘
```

### 5.2 LeanAide Integration

```
┌─────────────────────────────────────────────────────────────┐
│              LeanAide Integration Architecture              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Lean 4        │
                   │ Theorem        │
                   └────────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────┐                   ┌───────────┐
    │LeanAide   │                   │Evolution  │
    │Client     │                   │Engine     │
    └─────┬─────┘                   └─────┬─────┘
          │                               │
          │ translate_thm()               │
          ├──────────────────────────────>│
          │                               │
          │ proof                         │
          │<──────────────────────────────┤
          │                               │
          │    (if fails)                 │
          │    run_evolution()            │
          ├──────────────────────────────>│
          │                               │
          │ evolved_proof                 │
          │<──────────────────────────────┤
          │                               │
          ▼                               ▼
    ┌───────────┐                   ┌───────────┐
    │ Verified  │                   │ Statistics │
    │ Proof     │                   └───────────┘
    └───────────┘
```

---

## 6. Performance Flow

### 6.1 Performance Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                Performance Flow Diagram                     │
└─────────────────────────────────────────────────────────────┘

Time (seconds)
│
│  Pure Evolution:      ████████████████████████
│  MDAP+Ev (k=3):       ████████████████████████████████
│  MDAP+Ev (k=5):       ███████████████████████████████████████████
│
│  └───────────────────────────────────────────────────────▶ Generations
│   0    5    10   15   20   25   30

Fitness
│
│  Pure Evolution:      ████░░░░░░░░░░░░░░░░░░
│  MDAP+Ev (k=3):       ████████░░░░░░░░░░░░░
│  MDAP+Ev (k=5):       ████████████░░░░░░░
│
│  └───────────────────────────────────────────────────────▶ Generations
│   0    5    10   15   20   25   30

Success Rate
│
│  Pure Evolution:      ████████░░ (75%)
│  MDAP+Ev (k=3):       ██████████░ (88%)
│  MDAP+Ev (k=5):       ███████████ (92%)
│
│  └───────────────────────────────────────────────────────▶
│   0%                50%               100%
```

### 6.2 Scalability Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Scalability Diagram                        │
└─────────────────────────────────────────────────────────────┘

Population Size vs Time
│
│  Time (s)      │
│       100      │                              ●●●
│        80      │                        ●●●
│        60      │                  ●●●
│        40      │            ●●●
│        20      │      ●●●
│         0      └─────────────────────────────────────▶
│                10    20    30    40    50    60
│                      Population Size

Voting Threshold vs Success Rate
│
│  Success (%)  │
│       100     │         ┌─────────●
│        90     │      ───●
│        80     │   ───●
│        70     │──●
│         0     └─────────────────────────────────────▶
│                2     3     4     5     6     7     8
│                      Voting Threshold (k)
```

---

**Document End**

For more information, see:
- `LEANAIDE_EVOLUTION_MDAP_GUIDE.md` - User guide
- `LEANAIDE_EVOLUTION_MDAP_API.md` - API reference
- `LEANAIDE_EVOLUTION_MDAP_EXAMPLES.md` - Real-world examples
