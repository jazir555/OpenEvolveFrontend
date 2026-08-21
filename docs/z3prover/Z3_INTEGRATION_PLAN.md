# OpenEvolve: Z3 Theorem Prover Integration Plan

## 1. Executive Summary
This document outlines the strategic integration of the **Z3 Theorem Prover** (Microsoft Research) into the OpenEvolve Mega-Structure. While **Lean 4** provides higher-order formal verification (the "Proof"), **Z3** provides first-order constraint satisfaction and satisfiability checking (the "Search").

By adding Z3, OpenEvolve moves from a system that can *prove* a solution to one that can efficiently *filter* the search space of possible solutions before proof generation begins.

---

## 2. Theoretical Role: The "Feasibility Filter"
In the OpenEvolve architecture, Z3 acts as the bridge between the **Decomposition Engine** and the **Lean 4 Verification Engine**.

| Engine | Logic Level | Role in Invention |
| :--- | :--- | :--- |
| **Z3 Prover** | First-Order / SMT | **Satisfiability:** "Is this plan mathematically possible under these constraints?" |
| **NeuroMANCER** | Physics-Informed | **Physicality:** "Does this plan violate the laws of physics?" |
| **Lean 4** | Higher-Order | **Provability:** "Can we prove this result is 100% correct?" |

---

## 3. Integration Architecture

### A. The Vendored Location
Following the "Fortress" pattern of OpenEvolve, Z3 will be physically vendored into the root directory:
`C:\Users\mmeadow\Documents\OpenEvolve\Frontend\z3prover\`

### B. The Integration Registry
A new adapter will be created in the `integrations/` silo:
`integrations/z3/adapter.py`

### C. The Orchestration Logic
Z3 will be wired into the **Formal Gauntlet System** as a pre-verification step.

---

## 4. Primary Use Cases

### I. Constraint-Aware Decomposition
The **Comprehensive Decomposition Engine** currently uses 15+ strategies. Z3 will add a "Logical Consistency" check.
*   **Workflow:** When a problem is decomposed, Z3 checks if the resource estimates (time, effort, complexity) are logically consistent with the parent constraints.
*   **Result:** Binary rejection of impossible plans before they reach the Agentic loop.

### II. Skillbook Conflict Resolution (ACE)
As the **Agentic Context Engine (ACE)** grows, skills in the "Skillbook" may contradict each other.
*   **Workflow:** Z3 performs a "Conflict Scan" across the Skillbook, identifying sets of learned skills that create logical paradoxes.
*   **Result:** Automated pruning of the ACE memory to maintain a logically sound internal model.

### III. Algorithmic Invention (RESE)
In the discovery of new technologies, Z3 will handle **Parameter Optimization**.
*   **Workflow:** If an invention requires specific variable ranges (e.g., "Temperature must be > 500K but < Melting Point"), Z3 finds the valid "Satisfiable" region for the experiment.
*   **Result:** Precise, bounded inputs for the **NeuroMANCER** simulations.

---

## 5. Implementation Roadmap

### Step 1: Kernel Setup
1.  Initialize `Frontend/z3prover/` folder.
2.  Vendor the `z3-solver` Python source or binaries locally.
3.  Update `sys.path` in `openevolve_imports.py` to include the Z3 path.

### Step 2: The Z3 Adapter (`integrations/z3/adapter.py`)
Implement the `ConstraintInterface` with the following methods:
*   `check_satisfiability(constraints: List[str]) -> bool`
*   `find_model(constraints: List[str]) -> Dict[str, Any]` (to get valid parameters)
*   `minimize_objective(objective: str, constraints: List[str])`

### Step 3: Wiring into Gauntlets
Add a `SatisfiabilityGauntlet` to `formal_gauntlet_system.py`.
*   This gauntlet will convert natural language constraints (via **OneKE**) into SMT-LIB format for Z3 to verify.

---

## 6. Failure Mode Mitigation
Z3 prevents the "Combinatorial Explosion" problem. By failing "impossible" plans at the SMT level (milliseconds), we save the "Proof" engine (Lean 4) from attempting expensive proof searches on logically doomed candidates.

## 7. Final Impression
Integrating Z3 turns OpenEvolve into a **Complete Logic Stack**.
1.  **Search (MCTS)**
2.  **Satisfy (Z3)**
3.  **Simulate (NeuroMANCER)**
4.  **Succeed (Lean 4 Proof)**

This integration completes the journey from **Probabilistic AI** to **Universal Deterministic Intelligence**.

---
## STATUS (Reconciliation Note)
**Last reconciled: 2026-08-20**

- TYPE: Plan to vendor Z3 as z3prover/ and integrations/z3/adapter.py.
- VERIFICATION: integrations/z3/adapter.py does NOT exist (grep/Test-Path = missing). Actual Z3 integration ships as a BubbleLab ServiceBubble: core-projects/BubbleLab/packages/bubble-core/src/bubbles/service-bubble/openevolve-z3prover-bubble.ts and core-projects/BubbleLab/integrations/openevolve/service-bubbles/z3prover-bubble.ts (SMT operations: solve_smt, optimize, simplify, apply_tactic, fixedpoint_query).
- STATUS: PARTIALLY IMPLEMENTED — Z3 available as BubbleLab z3prover ServiceBubble; documented integrations/z3/adapter.py path NOT present.

