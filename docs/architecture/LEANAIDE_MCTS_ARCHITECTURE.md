# LeanAide MCTS Architecture

> **STATUS: implemented** (see `integrations/leanaide/leanaide_mcts.py` — `MCTS`, `MCTSTree`, `MCTSNode`, and the four phase classes `MCTSSelection`/`MCTSExpansion`/`MCTSSimulation`/`MCTSBackpropagation`; plus `integrations/leanaide/leanaide_mcts_strategies.py`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Diagrams](#component-diagrams)
3. [Data Flow](#data-flow)
4. [Sequence Diagrams](#sequence-diagrams)
5. [Integration Flows](#integration-flows)
6. [Performance Flows](#performance-flows)
7. [Class Hierarchy](#class-hierarchy)
8. [Design Patterns](#design-patterns)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LeanAide System                         │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   Lean 4     │      │  LeanAide    │      │    MCTS      │ │
│  │   Server     │◄────►│    Core      │◄────►│   Engine     │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│                                │                     │          │
│                                │                     │          │
│  ┌──────────────┐      ┌──────▼──────┐      ┌─────▼──────┐  │
│  │  Evolution   │      │ Decomposition│      │  Workflow  │  │
│  │   Engine     │◄────►│   Engine    │◄────►│ Integrator │  │
│  └──────────────┘      └─────────────┘      └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### MCTS Engine Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LeanProofMCTS                           │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  Selection   │   │  Expansion   │   │ Simulation   │       │
│  │    Phase     │──►│    Phase     │──►│    Phase     │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│                                                    │            │
│                                                    ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Backpropagation Phase                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  MCTS Tree                               │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │  │
│  │  │ Root   │─►│Child 1 │─►│Child 2 │─►│Child 3 │...     │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Diagrams

### MCTS Core Components

```
┌──────────────────────────────────────────────────────────────────┐
│                        MCTS Core                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        MCTS                               │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Attributes:                                       │  │   │
│  │  │  - exploration_constant: float                      │  │   │
│  │  │  - rollout_depth: int                              │  │   │
│  │  │  - rollout_episodes: int                           │  │   │
│  │  │  - total_simulations: int                          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                                                           │   │
│  │  Methods:                                                 │   │
│  │  + select(node): Node                                     │   │
│  │  + expand(node, actions): Node                            │   │
│  │  + simulate(node, action_gen, evaluator): float          │   │
│  │  + backpropagate(node, value): void                      │   │
│  │  + run_simulation(root): Node                            │   │
│  │  + get_best_child(node): Node                            │   │
│  │  + get_tree_statistics(root): Dict                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  MCTSNode     │  │ ProofContext  │  │  TacticAction │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### LeanProofMCTS Components

```
┌──────────────────────────────────────────────────────────────────┐
│                     LeanProofMCTS                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Configuration                         │   │
│  │  - exploration_constant: float = 1.414                   │   │
│  │  - simulations: int = 1000                               │   │
│  │  - rollout_depth: int = 5                                │   │
│  │  - temperature: float = 1.0                              │   │
│  │  - dirichlet_alpha: float = 0.3                          │   │
│  │  - dirichlet_epsilon: float = 0.25                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Components                             │   │
│  │                                                           │   │
│  │  ┌────────────────┐  ┌────────────────┐                  │   │
│  │  │  MCTS Engine   │  │ Tactic Library │                  │   │
│  │  │  (Core MCTS)   │  │                │                  │   │
│  │  └────────────────┘  └────────────────┘                  │   │
│  │                                                           │   │
│  │  ┌────────────────┐  ┌────────────────┐                  │   │
│  │  │ Action Generator│ │State Evaluator │                  │   │
│  │  └────────────────┘  └────────────────┘                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Methods:                                                        │
│  + search(context, client): (Sequence, Node)                    │
│  + _generate_actions(context): Actions                           │
│  + _evaluate_state(context, client): float                      │
│  + _add_dirichlet_noise(node): void                             │
│  + _extract_best_sequence(root): Sequence                       │
│  + get_action_probabilities(root, temp): Dict                   │
│  + get_statistics(): Dict                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Overall Data Flow

```
┌──────────────┐
│  User Input  │
│  (Theorem)   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Create ProofContext │
│  - goal              │
│  - hypotheses       │
│  - lemmas           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Initialize MCTS     │
│  - Create root node  │
│  - Add noise         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              MCTS Search Loop                    │
│  ┌────────────────────────────────────────────┐  │
│  │  1. SELECT: Navigate to leaf              │  │
│  │     (UCB policy)                          │  │
│  └─────────────┬──────────────────────────────┘  │
│                ▼                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  2. EXPAND: Add child nodes               │  │
│  │     (applicable tactics)                  │  │
│  └─────────────┬──────────────────────────────┘  │
│                ▼                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  3. SIMULATE: Run rollout                 │  │
│  │     (random/heuristic)                    │  │
│  └─────────────┬──────────────────────────────┘  │
│                ▼                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  4. BACKPROPAGATE: Update statistics      │  │
│  │     (path to root)                        │  │
│  └─────────────┬──────────────────────────────┘  │
│                │                                  │
│                └───── Until budget exhausted ─────┘
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────┐
│  Extract Best Path   │
│  - Most visited      │
│  - Highest value     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Return Proof        │
│  - Tactic sequence   │
│  - Statistics        │
└──────────────────────┘
```

### MCTS Node Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     MCTS Node                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ SELECT  │───►│ EXPAND  │───►│SIMULATE │───►│BACKPROP │  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘  │
│       │              │              │              │         │
│       ▼              ▼              ▼              ▼         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ get_ucb │    │ add_child│   │ rollout │    │ update  │  │
│  │ _score  │    │ _action  │   │ policy  │    │ N, W, Q │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                              │
│  State Updates:                                              │
│  - visit_count++ (N)                                         │
│  - total_value += reward (W)                                 │
│  - average_value = W/N (Q)                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Sequence Diagrams

### Complete MCTS Search Sequence

```
User          LeanProofMCTS       MCTS           MCTSNode        ProofContext
 │                 │               │                │                  │
 │  search(theorem)│               │                │                  │
 │────────────────►│               │                │                  │
 │                 │               │                │                  │
 │                 │  create root  │                │                  │
 │                 │──────────────►│                │                  │
 │                 │               │                │                  │
 │                 │               │  init(context)│                  │
 │                 │               │───────────────►│                  │
 │                 │               │                │                  │
 │                 │  for N simulations:           │                  │
 │                 │               │                │                  │
 │                 │  select(root) │                │                  │
 │                 │──────────────►│                │                  │
 │                 │               │                │                  │
 │                 │               │  get_ucb()     │                  │
 │                 │               │───────────────►│                  │
 │                 │               │                │                  │
 │                 │               │  select_child()│                  │
 │                 │               │◄───────────────│                  │
 │                 │◄──────────────│                │                  │
 │                 │               │                │                  │
 │                 │  expand(node) │                │                  │
 │                 │──────────────►│                │                  │
 │                 │               │                │                  │
 │                 │               │ add_child()    │                  │
 │                 │               │──────────────────────────────────►│
 │                 │               │◄──────────────────────────────────│
 │                 │               │                │                  │
 │                 │  simulate()   │                │                  │
 │                 │──────────────►│                │                  │
 │                 │               │                │                  │
 │                 │               │  rollout()     │                  │
 │                 │               │──────────────────────────────────►│
 │                 │               │◄──────────────────────────────────│
 │                 │◄──────────────│                │                  │
 │                 │               │                │                  │
 │                 │  backpropagate()                │                  │
 │                 │──────────────►│                │                  │
 │                 │               │                │                  │
 │                 │               │  update(path)  │                  │
 │                 │               │──────────────────────────────────►│
 │                 │               │                │                  │
 │                 │  extract_best()                │                  │
 │                 │──────────────►│                │                  │
 │                 │◄──────────────│                │                  │
 │                 │               │                │                  │
 │  return(sequence)               │                │                  │
 │◄────────────────│               │                │                  │
 │                 │               │                │                  │
```

### Integration with LeanAide Workflow

```
Workflow       MCTS Engine         Lean 4 Server    Tactic Library
  │                 │                    │                 │
  │ decompose()     │                    │                 │
  │────────────────►│                    │                 │
  │                 │                    │                 │
  │                 │ get_tactics()      │                 │
  │                 │────────────────────────────────────►│
  │                 │◄─────────────────────────────────────│
  │                 │                    │                 │
  │                 │ apply_tactic()     │                 │
  │                 │───────────────────►│                 │
  │                 │◄───────────────────│                 │
  │                 │                    │                 │
  │◄───────────────│                    │                 │
  │                 │                    │                 │
  │ solve_subgoal() │                    │                 │
  │────────────────►│                    │                 │
  │                 │                    │                 │
  │                 │ run_mcts_search()  │                 │
  │                 │                    │                 │
  │                 │──────────────────────────────────────►│
  │                 │◄───────────────────────────────────────│
  │                 │                    │                 │
  │◄───────────────│                    │                 │
  │                 │                    │                 │
  │ synthesize()    │                    │                 │
  │────────────────►│                    │                 │
  │                 │                    │                 │
  │◄───────────────│                    │                 │
```

---

## Integration Flows

### Stage 3A: Problem Decomposition

```
┌──────────────────────────────────────────────────────────────┐
│                    Decomposition Stage                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Theorem                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ MCTS Strategy    │                                       │
│  │ Search           │                                       │
│  │ - High-level     │                                       │
│  │ - Broad search   │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Extract          │                                       │
│  │ Sub-goals        │                                       │
│  │ - Identify       │                                       │
│  │   key lemmas     │                                       │
│  │ - Break down     │                                       │
│  │   proof steps    │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  Output Sub-problems                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Stage 3B: Sub-Problem Solving

```
┌──────────────────────────────────────────────────────────────┐
│                  Sub-Problem Solving                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sub-problem 1          Sub-problem 2          Sub-problem 3  │
│       │                       │                      │        │
│       ▼                       ▼                      ▼        │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐     │
│  │  MCTS   │           │  MCTS   │           │  MCTS   │     │
│  │  Search │           │  Search │           │  Search │     │
│  └────┬────┘           └────┬────┘           └────┬────┘     │
│       │                     │                     │          │
│       ▼                     ▼                     ▼          │
│  Proof 1               Proof 2               Proof 3         │
│       │                     │                     │          │
│       └─────────────────────┴─────────────────────┘          │
│                              │                                │
│                              ▼                                │
│                    Collection of Sub-proofs                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Stage 3C: Proof Synthesis

```
┌──────────────────────────────────────────────────────────────┐
│                    Proof Synthesis                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sub-proofs Collection                                       │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ MCTS Synthesize   │                                       │
│  │ - Combine proofs  │                                       │
│  │ - Fill gaps       │                                       │
│  │ - Optimize order  │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ MCTS Refine      │                                       │
│  │ - Polish proof    │                                       │
│  │ - Shorten steps   │                                       │
│  │ - Improve style   │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│     Final Proof                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Stage 5: Refinement

```
┌──────────────────────────────────────────────────────────────┐
│                      Refinement                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Initial Proof                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ Analyze Proof    │                                       │
│  │ - Identify issues│                                       │
│  │ - Find patterns  │                                       │
│  │ - Locate bloat   │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ MCTS Local       │                                       │
│  │ Refinement       │                                       │
│  │ - Focus on       │                                       │
│  │   problem areas  │                                       │
│  │ - Try alternatives│                                      │
│  │ - Optimize       │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│     Refined Proof                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance Flows

### Parallel MCTS Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Parallel MCTS                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Initial Proof Context                                       │
│       │                                                      │
│       ▼                                                      │
│  ┌────────────────────────────────────────────────┐          │
│  │            Parallel Executor                   │          │
│  │                                                │          │
│  │  ┌────────┐  ┌────────┐  ┌────────┐          │          │
│  │  │Worker 1│  │Worker 2│  │Worker 3│ ...      │          │
│  │  │        │  │        │  │        │          │          │
│  │  │ MCTS   │  │ MCTS   │  │ MCTS   │          │          │
│  │  │(N/3)   │  │(N/3)   │  │(N/3)   │          │          │
│  │  └───┬────┘  └───┬────┘  └───┬────┘          │          │
│  └──────┼───────────┼───────────┼───────────────┘          │
│         │           │           │                            │
│         └───────────┴───────────┴────────┐                   │
│                     │                   │                    │
│                     ▼                   │                    │
│            ┌─────────────────┐          │                    │
│            │ Merge Results   │          │                    │
│            │ - Combine trees │          │                    │
│            │ - Best path     │          │                    │
│            └────────┬────────┘          │                    │
│                     │                   │                    │
│                     ▼                   │                    │
│              Best Proof Sequence        │                    │
│                                          │                    │
└──────────────────────────────────────────┴────────────────────┘
```

### Transposition Table Flow

```
┌──────────────────────────────────────────────────────────────┐
│              Transposition Table Optimization                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  State S                                                    │
│    │                                                        │
│    ▼                                                        │
│  ┌─────────────┐                                           │
│  │ Compute     │                                           │
│  │ Hash(S)     │                                           │
│  └──────┬──────┘                                           │
│         │                                                  │
│         ▼                                                  │
│  ┌─────────────────┐                                      │
│  │ Lookup in Table │                                      │
│  └─────┬───────────┘                                      │
│        │                                                  │
│        ├─► Found? ──Yes──► Return Cached Value            │
│        │                                                  │
│        └─No──► Run Simulation                             │
│                  │                                        │
│                  ▼                                        │
│            ┌─────────────┐                               │
│            │ Store in     │                               │
│            │ Table        │                               │
│            └─────────────┘                               │
│                  │                                        │
│                  ▼                                        │
│            Return Value                                   │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                    Class Hierarchy                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  object                                                      │
│    │                                                         │
│    ├─── MCTS                                                │
│    │     - Core MCTS algorithm                              │
│    │                                                         │
│    ├─── MCTSNode                                            │
│    │     - Tree node                                        │
│    │                                                         │
│    └─── LeanProofMCTS                                       │
│          - Lean 4 specific MCTS                             │
│          - Inherits from MCTS concepts                      │
│                                                             │
│  dataclass                                                   │
│    │                                                         │
│    ├─── ProofContext                                        │
│    │     - Proof state representation                        │
│    │                                                         │
│    ├─── Tactic                                              │
│    │     - Tactic metadata                                  │
│    │                                                         │
│    ├─── TacticAction                                        │
│    │     - Action in search space                           │
│    │                                                         │
│    └─── MCTSResult                                          │
│          - Search result                                    │
│                                                             │
│  Enum                                                       │
│    │                                                         │
│    ├─── TacticStatus                                        │
│    └─── ProofState                                          │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

### Strategy Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                   Strategy Pattern                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    RolloutPolicy                       │  │
│  │  <<interface>>                                        │  │
│  │  + simulate(state): float                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ▲                                     │
│                       │                                     │
│        ┌──────────────┼──────────────┐                     │
│        │              │              │                      │
│  ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐                │
│  │  Random   │ │Heuristic  │ │  Policy   │                │
│  │  Rollout  │ │  Rollout  │ │  Rollout  │                │
│  └───────────┘ └───────────┘ └───────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Template Method Pattern

```
┌──────────────────────────────────────────────────────────────┐
│              Template Method Pattern                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                        MCTS                            │  │
│  │                                                         │  │
│  │  run_simulation(root):                                 │  │
│  │    node = select(root)           # Can be overridden   │  │
│  │    child = expand(node, actions)  # Can be overridden  │  │
│  │    value = simulate(child)         # Can be overridden│  │
│  │    backpropagate(child, value)    # Can be overridden │  │
│  │                                                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                       ▲                                     │
│                       │                                     │
│              ┌────────┴────────┐                            │
│              │                 │                            │
│        ┌─────┴─────┐    ┌─────┴─────┐                       │
│        │ Standard  │    │   Custom  │                       │
        │    MCTS   │    │    MCTS   │                       │
│        └───────────┘    └───────────┘                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Builder Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                 Builder Pattern                             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  MCTSConfigBuilder                     │  │
│  │                                                         │  │
│  │  + with_exploration(c): Builder                        │  │
│  │  + with_simulations(n): Builder                        │  │
│  │  + with_rollout_depth(d): Builder                      │  │
│  │  + with_temperature(t): Builder                        │  │
│  │  + build(): MCTS                                       │  │
│  │                                                         │  │
│  │  Usage:                                                │  │
│  │    mcts = MCTSBuilder()                                │  │
│  │           .with_exploration(1.414)                     │  │
│  │           .with_simulations(1000)                      │  │
│  │           .with_rollout_depth(10)                      │  │
│  │           .build()                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Observer Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                Observer Pattern                             │
│                                                              │
│  ┌──────────────────┐         ┌─────────────────────┐       │
│  │   Subject        │         │    Observer         │       │
│  │   (MCTS)         │         │  (ProgressMonitor)  │       │
│  │                  │         │                     │       │
│  │  - observers[]   │◄────────│+ update(event)      │       │
│  │                  │         │                     │       │
│  │  + attach(obs)   │         └─────────────────────┘       │
│  │  + detach(obs)   │                                     │
│  │  + notify(event) │                                     │
│  └──────────────────┘                                     │
│         │                                                   │
│         │ notifies                                         │
│         ▼                                                   │
│  Events:                                                    │
│  - on_simulation_start                                     │
│  - on_simulation_complete                                  │
│  - on_node_created                                         │
│  - on_value_updated                                        │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary

The LeanAide MCTS architecture is designed with:

1. **Modularity**: Clear separation between MCTS core and Lean 4 integration
2. **Extensibility**: Easy to add new strategies, policies, and evaluators
3. **Performance**: Transposition tables, parallelization, caching
4. **Integration**: Seamless integration with LeanAide workflow stages
5. **Maintainability**: Clean class hierarchy, design patterns

---

*Last Updated: 2025-12-30*
*Version: 1.0.0*
