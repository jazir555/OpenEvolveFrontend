# LeanAide MCTS-MDAP Architecture

> **STATUS: implemented** (see `integrations/leanaide/leanaide_mcts_mdap.py` — `MDAPMCTS`, `MDAPMCTSConfig`, `LeanAIDEMCTSMdap`; `integrations/leanaide/leanaide_mcts_mdap_complete.py`; MDAP voting/orchestration in `integrations/leanaide/leanaide_mdap.py` and `engines/other/mdap_engine.py`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Interaction](#component-interaction)
4. [Data Flow](#data-flow)
5. [Sequence Diagrams](#sequence-diagrams)
6. [Integration Patterns](#integration-patterns)
7. [Performance Flows](#performance-flows)

---

## System Overview

The MDAP-MCTS system integrates Monte Carlo Tree Search with Multi-Agent Decomposition to create a robust theorem proving system for Lean 4.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  LeanAide Client  │  Workflow Manager  │  Task Orchestrator     │
└────────┬────────────┬──────────────────┬────────────┬───────────┘
         │            │                  │            │
┌────────▼────────────▼──────────────────▼────────────▼───────────┐
│                     MDAP-MCTS Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   MCTS       │  │   MDAP       │  │   MAKER      │          │
│  │              │  │              │  │              │          │
│  │ - Selection  │  │ - Voting     │  │ - Decompose  │          │
│  │ - Expansion  │  │ - Red-flag   │  │ - Recursion  │          │
│  │ - Simulation │  │ - Consensus  │  │ - Compose    │          │
│  │ - Backprop   │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                  │                   │
│         └─────────────────┴──────────────────┘                   │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     Integration Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Lean 4 Server  │  Agent Pool  │  Cache Manager  │  Monitor     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagrams

### 1. Component Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        MDAPMCTS                                   │
│  (Main Orchestrator)                                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │ MCTSSelection  │  │MCTSExpansion   │  │MCTSSimulation  │      │
│  │                │  │ + MDAP Voting  │  │ + MAKER Voting │      │
│  │                │  │                │  │                │      │
│  │ - UCT Select   │  │ - Agent Vote   │  │ - MAKER Sim    │      │
│  │ - Tree Traverse│  │ - Red-flag     │  │ - Rollout      │      │
│  └────────────────┘  └────────────────┘  └────────────────┘      │
│           │                   │                    │               │
│           └───────────────────┴────────────────────┘               │
│                               │                                    │
│                      ┌────────▼────────┐                          │
│                      │MCTSBackprop     │                          │
│                      │                 │                          │
│                      │ - Update Stats  │                          │
│                      │ - Propagate Q   │                          │
│                      └─────────────────┘                          │
└────────────────────────────────────────────────────────────────────┘
```

### 2. MCTS-MDAP Integration

```
┌──────────────────────────────────────────────────────────────────┐
│                    MCTS with MDAP Integration                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Phase 1: Selection (Standard MCTS)                         │  │
│  │                                                             │  │
│  │   Root ──UCB──▶ Node1 ──UCB──▶ Node3 ──UCB──▶ Leaf        │  │
│  │         │          │                                          │  │
│  │         └─UCB──▶ Node2                                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Phase 2: Expansion (MDAP-Enhanced)                         │  │
│  │                                                             │  │
│  │   Leaf State                                                │  │
│  │       │                                                     │  │
│  │       ▼                                                     │  │
│  │   Available Actions: [intros, apply, rw, simp]             │  │
│  │       │                                                     │  │
│  │       ▼                                                     │  │
│  │   ┌─────────────────────────────────────────────┐          │  │
│  │   │ MDAP Voting                                  │          │  │
│  │   │                                              │          │  │
│  │   │   Agent1: apply (5 votes)                    │          │  │
│  │   │   Agent2: intros (3 votes)                   │          │  │
│  │   │   Agent3: apply (5 votes)                    │          │  │
│  │   │                                              │          │  │
│  │   │   Winner: apply (first-to-ahead-by-k=3)      │          │  │
│  │   └─────────────────────────────────────────────┘          │  │
│  │       │                                                     │  │
│  │       ▼                                                     │  │
│  │   New Child Node (action=apply)                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Phase 3: Simulation (MAKER-Enhanced)                       │  │
│  │                                                             │  │
│  │   New Child State                                          │  │
│  │       │                                                     │  │
│  │       ▼                                                     │  │
│  │   ┌─────────────────────────────────────────────┐          │  │
│  │   │ Loop: Until terminal or max depth            │          │  │
│  │   │                                              │          │  │
│  │   │   Current State                              │          │  │
│  │   │       │                                      │          │  │
│  │   │       ▼                                      │          │  │
│  │   │   ┌─────────────────────────────────┐       │          │  │
│  │   │   │ MAKER Voting Engine             │       │          │  │
│  │   │   │                                 │       │          │  │
│  │   │   │   Agents vote on tactic         │       │          │  │
│  │   │   │   Winner selected by k-ahead    │       │          │  │
│  │   │   │   Red-flagged actions filtered  │       │          │  │
│  │   │   └─────────────────────────────────┘       │          │  │
│  │   │       │                                      │          │  │
│  │   │       ▼                                      │          │  │
│  │   │   Apply Tactic → Next State                  │          │  │
│  │   └────────────────────────────────────┘        │          │  │
│  │                                              │          │  │
│  │   Return: Reward (0-1)                        │          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Phase 4: Backpropagation (Standard MCTS)                   │  │
│  │                                                             │  │
│  │   Update nodes from Leaf to Root:                          │  │
│  │     N += 1                                                  │  │
│  │     W += reward                                             │  │
│  │     Q = W / N                                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3. Data Structures

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Structure Hierarchy                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ProofState                                                       │
│  ├── goals: List[str]                                            │
│  ├── context: List[str]                                          │
│  ├── tactics_sequence: List[Tactic]                              │
│  ├── depth: int                                                  │
│  ├── is_complete: bool                                           │
│  └── hash: str                                                   │
│                                                                   │
│  Tactic                                                           │
│  ├── name: str                                                   │
│  ├── params: List[str]                                           │
│  └── metadata: Dict                                              │
│                                                                   │
│  MDAPMCTSNode (extends MCTSNode)                                 │
│  ├── MCTSNode attributes:                                        │
│  │   ├── state: ProofState                                       │
│  │   ├── parent: Node                                            │
│  │   ├── children: Dict[str, Node]                               │
│  │   ├── N: int (visits)                                         │
│  │   ├── W: float (total reward)                                 │
│  │   └── Q: float (avg reward)                                   │
│  └── MDAP enhancements:                                          │
│      ├── agent_votes: Dict[str, int]                             │
│      ├── red_flags: Dict[str, List[str]]                         │
│      ├── vote_confidence: float                                   │
│      └── maker_score: float                                      │
│                                                                   │
│  MCTSResult                                                      │
│  ├── best_proof: Optional[LeanProof]                             │
│  ├── success: bool                                               │
│  ├── search_iterations: int                                      │
│  ├── time_elapsed: float                                         │
│  ├── win_rate: float                                             │
│  └── confidence: float                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction

### 1. MCTS ↔ MDAP Interaction

```
MCTS Component              MDAP Component              Data Flow
─────────────               └─────────────               ──────────

    MCTS                        MDAP
     │                           │
     │  1. Request expansion     │
     ├──────────────────────────▶│
     │   - Current state         │
     │   - Available actions     │
     │                           │
     │                           │  2. Sample agents
     │                           │  - AgentSelector.pick()
     │                           │  - Get action suggestion
     │                           │
     │  3. Vote result           │
     │◀──────────────────────────┤
     │   - Selected action       │
     │   - Vote counts           │
     │   - Confidence            │
     │   - Red flags             │
     │                           │
     │  4. Create child node     │
     │   - Store voting metadata │
     │                           │
```

### 2. MCTS ↔ MAKER Interaction

```
MCTS Component              MAKER Component             Data Flow
─────────────               └──────────────             ──────────

    MCTS                        MAKER
     │                           │
     │  1. Request simulation    │
     ├──────────────────────────▶│
     │   - Current state         │
     │   - Max depth             │
     │                           │
     │                           │  2. MAKER voting loop
     │                           │  - VotingEngine.do_voting()
     │                           │  - First-to-ahead-by-k
     │                           │  - Red-flagging
     │                           │
     │  3. Selected tactic       │
     │◀──────────────────────────┤
     │   - Tactic                │
     │   - Next state            │
     │   - Confidence            │
     │                           │
     │  4. Continue rollout      │
     │   until terminal/depth    │
     │                           │
```

### 3. Agent Team Coordination

```
┌────────────────────────────────────────────────────────────────┐
│                      Agent Team Manager                         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Team                                                           │
│  ├── members: List[ModelConfig]                                │
│  ├── team_id: str                                              │
│  └── name: str                                                 │
│                                                                  │
│  ModelConfig (Agent)                                            │
│  ├── model_id: str                                             │
│  ├── api_key: str                                              │
│  ├── api_base: str                                             │
│  ├── temperature: float                                        │
│  ├── max_tokens: int                                           │
│  ├── problem_type_specialization: List[str]                    │
│  └── performance_metrics: Dict                                  │
│                                                                  │
│  Coordination Patterns:                                         │
│                                                                  │
│  1. Round-Robin Selection                                       │
│     Agent1 → Agent2 → Agent3 → Agent1 → ...                     │
│                                                                  │
│  2. Weighted Selection                                          │
│     Agent weights based on:                                     │
│     - Task type specialization                                  │
│     - Historical success rate                                   │
│     - Average proof length                                      │
│                                                                  │
│  3. Specialized Selection                                       │
│     Decomposition tasks → Decomposition specialists             │
│     Tactic selection → Tactic specialists                       │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Complete Search Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   Complete MDAP-MCTS Search Flow                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. INITIALIZATION                                                │
│     ├── Create team of agents                                    │
│     ├── Configure MCTS (c_param, rollout_depth, etc.)            │
│     ├── Configure MDAP (k_min, k_max, max_votes)                 │
│     └── Create root node from initial proof state                │
│                                                                   │
│  2. MAIN SEARCH LOOP                                             │
│     │                                                             │
│     ├── While not time_expired and iterations < max:             │
│     │                                                             │
│     │   2.1 SELECTION                                            │
│     │       └── Traverse tree using UCT to find leaf            │
│     │                                                             │
│     │   2.2 EXPANSION (with MDAP)                               │
│     │       ├── Get applicable tactics for leaf state            │
│     │       ├── MDAP voting:                                     │
│     │       │   ├── For each available tactic:                  │
│     │       │   │   ├── Agents vote on tactic                   │
│     │       │   │   ├── Red-flag unreliable responses           │
│     │       │   │   └── Check for first-to-ahead-by-k winner    │
│     │       │   └── Return winning tactic                       │
│     │       └── Create child node with winning tactic           │
│     │                                                             │
│     │   2.3 SIMULATION (with MAKER)                             │
│     │       ├── From new child state:                            │
│     │       │   ├── Loop until terminal or max depth:           │
│     │       │   │   ├── MAKER voting on tactic                  │
│     │       │   │   ├── Apply winning tactic                    │
│     │       │   │   └── Update state                           │
│     │       │   └── Return reward (0-1)                         │
│     │       └── Return simulation reward                        │
│     │                                                             │
│     │   2.4 BACKPROPAGATION                                     │
│     │       └── Update stats from child to root:                │
│     │           N += 1, W += reward, Q = W / N                   │
│     │                                                             │
│     └── End while                                                │
│                                                                   │
│  3. RESULT EXTRACTION                                             │
│     ├── Select child with highest visit count                    │
│     ├── Extract proof path from root to selected child           │
│     └── Return MCTSResult with proof and metadata                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Voting Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    MDAP Voting Data Flow                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│  ├── Current proof state                                       │
│  ├── Available tactics: [t1, t2, t3, ...]                      │
│  └── Agent team: [Agent1, Agent2, Agent3]                       │
│                                                                 │
│  Process:                                                       │
│  │                                                              │
│  │  Initialize: votes = {t1: 0, t2: 0, t3: 0, ...}           │
│  │                                                              │
│  │  While not has_winner(votes, k):                            │
│  │      │                                                        │
│  │      ├── Select agent (round-robin or weighted)             │
│  │      │                                                        │
│  │      ├── Agent.suggest_tactic(state) → tactic               │
│  │      │                                                        │
│  │      ├── Check red-flags:                                   │
│  │      │   ├── Response too long? → Skip                      │
│  │      │   ├── Invalid format? → Skip                         │
│  │      │   ├── Low confidence? → Skip                         │
│  │      │   └── Schema mismatch? → Skip                        │
│  │      │                                                        │
│  │      ├── If not red-flagged:                                │
│  │      │   votes[tactic] += 1                                 │
│  │      │                                                        │
│  │      └── Check: votes[tactic] >= k + max(other votes)?      │
│  │          If yes: return tactic as winner                    │
│  │                                                              │
│  Output:                                                        │
│  ├── Winner tactic                                             │
│  ├── Vote counts: {t1: 5, t2: 2, t3: 1, ...}                  │
│  ├── Confidence: winner_votes / total_votes                    │
│  └── Red flags: {t4: ["too_long"], t5: ["invalid_format"]}     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Sequence Diagrams

### 1. Expansion with Voting

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  MCTS    │   │ Expansion│   │  MDAP    │   │  Agent1  │   │  Agent2  │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │              │
     │ expand(node, │              │              │              │
     │   actions)   │              │              │              │
     │─────────────▶│              │              │              │
     │              │              │              │              │
     │              │ vote(state,  │              │              │
     │              │   actions, k)│              │              │
     │              │─────────────▶│              │              │
     │              │              │              │              │
     │              │              │ select_agent()│              │
     │              │              ├─────────────▶│              │
     │              │              │              │              │
     │              │              │              │ suggest()    │
     │              │              │              ├─────────────▶│
     │              │              │              │              │
     │              │              │              │  tactic=t1   │
     │              │              │              │◀─────────────┤
     │              │              │              │              │
     │              │              │ check_red_flags(t1)          │
     │              │              ├─────────────▶│              │
     │              │              │◀────┘       │              │
     │              │              │ not flagged  │              │
     │              │              │ votes[t1]++ │              │
     │              │              │              │              │
     │              │              │ select_agent()│              │
     │              │              ├─────────────────────────────▶│
     │              │              │              │              │
     │              │              │              │ suggest()    │
     │              │              │◀────────────────────────────┤
     │              │              │              │              │
     │              │              │ check_red_flags(t2)         │
     │              │              ├─────────────────────────────▶│
     │              │              │◀────┘       │              │
     │              │              │ not flagged  │              │
     │              │              │ votes[t2]++ │              │
     │              │              │              │              │
     │              │              │ ...continue until k-ahead...│
     │              │              │              │              │
     │              │◀─────────────┤ winner=t1    │              │
     │              │  votes={t1:5,│ conf=0.83    │              │
     │              │  t2:2, t3:1} │              │              │
     │              │              │              │              │
     │ create_child │              │              │              │
     │ (t1, votes)  │              │              │              │
     │◀─────────────┤              │              │              │
     │              │              │              │              │
     │ child_node   │              │              │              │
     │◀─────────────┤              │              │              │
```

### 2. Simulation with MAKER

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  MCTS    │   │Simulation│   │  MAKER   │   │VotingEng │   │  Agents   │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │              │
     │ simulate()   │              │              │              │
     │─────────────▶│              │              │              │
     │              │              │              │              │
     │              │ while not terminal:          │              │
     │              │              │              │              │
     │              │ do_voting(state, agents, k)│              │
     │              │─────────────▶│              │              │
     │              │              │              │              │
     │              │              │ get_vote(agent)              │
     │              │              │─────────────▶│              │
     │              │              │              │              │
     │              │              │              │ collect votes│
     │              │              │              ├─────────────▶│
     │              │              │              │              │
     │              │              │              │ check winner │
     │              │              │              │◀─────────────┤
     │              │              │              │              │
     │              │              │◀─────────────┤ tactic, votes│
     │              │              │  winner=t1  │              │
     │              │◀─────────────┤  conf=0.91  │              │
     │              │  tactic=t1   │              │              │
     │              │              │              │              │
     │              │ apply(t1) → new_state      │              │
     │              │              │              │              │
     │              │ ...repeat for max_depth... │              │
     │              │              │              │              │
     │              │ return reward=0.85         │              │
     │◀─────────────┤              │              │              │
     │  reward=0.85 │              │              │              │
```

---

## Integration Patterns

### 1. Workflow Integration

```
┌────────────────────────────────────────────────────────────────┐
│            Decomposition Workflow Integration                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Problem Analysis                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Input: Theorem statement                                   │ │
│  │ Output: Problem type, complexity, domain                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Stage 2: Strategy Selection                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ If simple (1-5 tactics):         → Pure MCTS              │ │
│  │ If medium (5-20 tactics):        → MCTS + MDAP            │ │
│  │ If complex (20+ tactics):        → MAKER + MCTS           │ │
│  │ If novel domain:                → MAKER Decomposition     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         │                  │                  │               │
│         ▼                  ▼                  ▼               │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │ Pure MCTS│      │MCTS+MDAP │      │MAKER+MCTS│           │
│  └─────┬────┘      └─────┬────┘      └─────┬────┘           │
│        │                 │                 │                 │
│        └─────────────────┴─────────────────┘                 │
│                            │                                   │
│                            ▼                                   │
│  Stage 3A: MDAP-MCTS Search (if MCTS+MDAP or MAKER+MCTS)      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Use MDAP voting during expansion                         │ │
│  │ - Use MAKER simulation for rollout                        │ │
│  │ - Adaptive strategy selection based on progress            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Stage 3B: Refinement                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Evaluate proof quality                                   │ │
│  │ - If low quality: Refine weak steps with MCTS             │ │
│  │ - Optimize tactic sequence                                │ │
│  │ - Simplify proof                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Stage 4: Verification                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Send proof to Lean 4 server                              │ │
│  │ - Verify correctness                                       │ │
│  │ - If verification fails: Return to Stage 3B                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Output: Verified proof + metadata                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2. Error Handling Flow

```
┌────────────────────────────────────────────────────────────────┐
│                     Error Handling Flow                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Try MDAP-MCTS Search                                          │
│  │                                                              │
│  ├─► Success? ──Yes──▶ Return proof                            │
│  │                                                              │
│  └─► No ──► Analyze failure                                    │
│           │                                                      │
│           ├─► All agents red-flagged?                           │
│           │     └─► Relax red-flagging rules                    │
│           │     └─► Retry with lower confidence threshold       │
│           │                                                      │
│           ├─► Voting never converged?                           │
│           │     └─► Reduce k-value (k_min, k_max)              │
│           │     └─► Increase max_votes_per_step                 │
│           │                                                      │
│           ├─► Timeout during search?                            │
│           │     └─► Return best partial result                  │
│           │     └──► If time permits, try pure MCTS fallback    │
│           │                                                      │
│           ├─► All actions red-flagged?                          │
│           │     └─► Disable red-flagging                        │
│           │     └─► Use best-effort fallback                    │
│           │                                                      │
│           └─► Other error?                                     │
│                 └─► Log error details                          │
│                 └─► Fall back to pure MCTS                     │
│                 └─► If still fails, return error               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Performance Flows

### 1. Parallelization Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                  Parallelization Architecture                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Main Thread (Orchestrator)                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Coordinate search                                         │ │
│  │ - Manage tree state                                        │ │
│  │ - Aggregate results                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│         │                                                        │
│         ├──▶ Parallel Simulations                             │
│         │    ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│         │    │ Sim 1    │  │ Sim 2    │  │ Sim 3    │        │
│         │    └──────────┘  └──────────┘  └──────────┘        │
│         │         │             │             │               │
│         │         └─────────────┴─────────────┘               │
│         │                       │                             │
│         │                       ▼                             │
│         │              Aggregate rewards                       │
│         │                                                        │
│         └──▶ Parallel Agent Queries                           │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│              │ Agent 1  │  │ Agent 2  │  │ Agent 3  │        │
│              └──────────┘  └──────────┘  └──────────┘        │
│                    │             │             │               │
│                    └─────────────┴─────────────┘               │
│                                  │                             │
│                                  ▼                             │
│                         Aggregate votes                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2. Caching Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                     Multi-Level Caching                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: Transposition Table (MCTS)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Key: State hash                                           │ │
│  │ Value: Node reference                                     │ │
│  │ Purpose: Reuse states across tree                          │ │
│  │ Size: Configurable (default 500MB)                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Level 2: MDAP Response Cache                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Key: (task_id, step_id, prompt_hash)                      │ │
│  │ Value: MDAPVoteResult                                     │ │
│  │ Purpose: Cache agent voting results                       │ │
│  │ Size: Configurable (default 5000 entries)                  │ │
│  │ TTL: Configurable (default: infinite)                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  Level 3: Lean 4 Verification Cache                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Key: (tactic_sequence, state_hash)                        │ │
│  │ Value: VerificationResult                                 │ │
│  │ Purpose: Avoid redundant verification                      │ │
│  │ Size: Configurable (default 10000 states)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Cache Invalidation:                                           │
│  - State hash changes → Invalidate Level 1                    │
│  - Prompt changes → Invalidate Level 2                        │
│  - New lemma introduced → Verify affected states              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Summary

The MDAP-MCTS architecture represents a carefully designed integration of:

1. **MCTS** - Provides efficient tree search with UCT selection
2. **MDAP** - Adds collective intelligence through multi-agent voting
3. **MAKER** - Enables recursive decomposition with error correction

The key architectural innovations are:

- **Voting-enhanced expansion**: Agents vote on best tactics during tree expansion
- **MAKER-enhanced simulation**: Uses robust voting for rollouts
- **Multi-level caching**: Reduces redundant computation
- **Parallel execution**: Accelerates simulations and agent queries
- **Error resilience**: Red-flagging and fallback mechanisms

This architecture achieves high theorem proving success rates while maintaining reasonable search times, making it practical for Lean 4 proof automation.

For more details:
- `LEANAIDE_MCTS_MDAP_GUIDE.md` - User guide
- `LEANAIDE_MCTS_MDAP_API.md` - API reference
- `LEANAIDE_MCTS_MDAP_EXAMPLES.md` - Usage examples
