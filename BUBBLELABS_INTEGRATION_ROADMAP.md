# Roadmap: Full OpenEvolve Integration via BubbleLabs

## **Objective**
To transform the **OpenEvolve Mega-Structure** from a collection of high-density functional silos into a singular, steerable **"Invention OS"**. The goal is to enable a human operator to manage the entire lifecycle of complex invention—from vague goal to formal proof—via the **BubbleLabs** command-and-control interface.

---

## **Phase 1: Kernel Consolidation (Structural Integrity)**
*Goal: Eliminate import friction and unify the triplicated data models.*

### 1.1. Schema Unification (Single Source of Truth)
*   **Action:** Merge `sovereign_data_models.py`, `openevolve_structures.py`, and `workflow_structures.py` into a single, high-performance module: `openevolve.kernel.schema`.
*   **Outcome:** Resolve the `ImportError` issues where systems (e.g., Gauntlets) expect classes that have moved or been renamed.
*   **Verification:** Run `test_mega_structure_integration.py` ensuring zero import failures.

### 1.2. Namespace Sanitization
*   **Action:** Move the ~250 backup/patch files (`_FIXED`, `_backup`, `_v1`) into a protected `/archive` directory.
*   **Action:** Reorganize the remaining 500+ files into a standard package structure:
    *   `/engines` (Decomposition, MDAP, MCTS)
    *   `/verification` (LeanAide, Gauntlets)
    *   `/learning` (ACE, Skillbooks)
    *   `/adapters` (OneKE, Graphiti, etc.)

### 1.3. Integration Registry Activation
*   **Action:** Update `integrations/registry.py` to perform active heartbeats on the physical subfolders (`/OneKE`, `/neuromancer`).
*   **Outcome:** BubbleLabs can now visually display a "Service Grid" showing which of the 30+ engines are ready for execution.

---

## **Phase 2: The Unified Control Plane (API Layer)**
*Goal: Move from script-triggering to a state-aware orchestrator.*

### 2.1. Global State Machine Enforcement
*   **Action:** Implement a centralized state tracker in `openevolve_bubblelabs_api.py`.
*   **Workflow:** Track the "Invention Plan" status: `IDLE` → `DECOMPOSING` → `GAUNTLET_RUNNING` → `RESE_SOLVING` → `PROVING` → `COMPLETE`.
*   **Rigor:** Ensure no engine can be triggered unless its predecessor state is `VALIDATED`.

### 2.2. Recursive Feedback Wiring
*   **Action:** Wire the **ACE (Agentic Context Engine)** into the API so that any failure in a **Formal Gauntlet** automatically triggers a "Reflection Event."
*   **Outcome:** The system self-corrects the plan based on failure data before presenting it back to the BubbleLabs user.

---

## **Phase 3: Visual Command & Control (BubbleLabs Expansion)**
*Goal: Enable human-steered AGI orchestration.*

### 3.1. Decomposition Architect View
*   **Feature:** Transform the `ComprehensiveDecompositionEngine` output into an interactive graph in BubbleLabs.
*   **Action:** Allow users to drag-and-drop dependencies, edit `SubProblem` metadata, and override strategy selection (`HIERARCHICAL` vs `RISK_BASED`).

### 3.2. Live Adversarial Monitoring
*   **Feature:** Create a "War Room" widget for live monitoring of `adversarial_mdap_mcts.py`.
*   **Visualization:** Show the "Team Red" attack vectors (Edges, Assumptions, Boundaries) vs "Team Blue" defenses in real-time as they co-evolve toward a proof.

### 3.3. Formal Truth Monitor (Lean 4)
*   **Feature:** A specialized Streamlit component to monitor `leanaide_evolution.py`.
*   **Visualization:** Display the Genetic Algorithm's progress: Population fitness, crossover success, and the final Lean 4 proof string once found.

---

## **Phase 4: Autonomous Research-Quest (The Finish Line)**
*Goal: 100% autonomous discovery cycles.*

### 4.1. SOP-to-Execution Automation
*   **Action:** Link `sop_generator_research_quest.py` to the **RESE (Recursive Execution)** engine.
*   **Outcome:** Once a research stage is "Approved" in the UI, the system automatically generates the SOP and hands it to the relevant Agentic loop for execution.

### 4.2. Binary Trust Artifacts
*   **Feature:** Implement a "Certification" module.
*   **Outcome:** The UI generates a "Truth Package" containing:
    *   **Evidence Chain:** (via OneKE & Graphiti)
    *   **Physical Feasibility:** (via NeuroMANCER)
    *   **Logical Soundness:** (via Lean 4 Proof)
    *   **Adversarial Robustness:** (via Red Team Gauntlet Score)

---

## **Execution Priority Table**

| Priority | Task | Target Date |
| :--- | :--- | :--- |
| **P0** | **Kernel Consolidation:** Unify Schemas & Fix Imports | Week 1 |
| **P1** | **API Bridge:** Implement Global State Machine | Week 2 |
| **P2** | **UI Architecture:** Build the Decomposition Architect View | Week 3 |
| **P3** | **Closed Loop:** Integrate ACE Reflections into Discovery | Week 4 |

---

## **The End-State Vision**
OpenEvolve will function as a **Sovereign Research Command Center**. A user defines a new technology goal in **BubbleLabs**, the **Decomposition Engine** architectures the path, and the **30+ Expert Engines** execute and verify it until a **Binary Proof of Success** is delivered.
