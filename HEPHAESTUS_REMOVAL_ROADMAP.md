# Roadmap: Complete Hephaestus Removal

## **Objective**
To systematically remove all remaining string references, documentation notes, and backward-compatibility aliases for the deprecated **Hephaestus** system. Hephaestus has been fully replaced by the **MIT-licensed CrewAI** orchestration layer.

---

## **Phase 1: Codebase Cleanup (String References & Aliases)**
*Goal: Remove all non-functional mentions of Hephaestus in the source code.*

### 1.1. Alias Removal
*   **Target:** `ace_crewai_bridge.py`
    *   Remove `ACEHephaestusWorkflowBridge = ACECrewAIWorkflowBridge`.
    *   Remove `"ACEHephaestusWorkflowBridge"` from `__all__`.
*   **Target:** `bubblelabs_crewai_bridge.py`
    *   Remove any legacy aliasing for `BubbleLabsHephaestusBridge`.

### 1.2. Comment & Docstring Updates
*   **Action:** Conduct a global search-and-replace for the following files to remove historical migration notes:
    *   `ace_crewai_bridge.py`
    *   `bubblelabs_crewai_bridge.py`
    *   `claudiomiro_crewai_bridge.py`
    *   `crewai_unified_bridge.py`
    *   `datapizza_crewai_bridge.py`
    *   `decomposition_crewai_bridge.py`
    *   `leanaide_crewai_bridge.py`
    *   `openevolve_crewai_bridge.py`
    *   `roma_crewai_bridge.py`
    *   `steer_crewai_bridge.py`
    *   `sovereign_decomposition_crewai_integration.py`
*   **Details:** Remove notes like *"It replaces the AGPL-licensed Hephaestus integration"* and replace with purely functional descriptions.

### 1.3. Logic Pruning
*   **Target:** `crewai_unified_bridge.py`
    *   Remove mentions of *"backward compatible with Hephaestus"*.
*   **Target:** `crewai_state_management.py`
    *   Remove descriptions comparing local JSON state to the old *"Hephaestus ticket-based system"*.

---

## **Phase 2: Documentation & Analysis Sanitization**
*Goal: Update high-level reports to reflect a pure CrewAI architecture.*

### 2.1. Audit Report Cleanup
*   **Target:** `COMPREHENSIVE_CODEBASE_SECURITY_ANALYSIS.md`
    *   Remove sections referencing vulnerabilities in `Hephaestus/src/validation/check_executors.py` (as these files no longer exist).
*   **Target:** `COMPREHENSIVE_SECURITY_AND_QUALITY_ANALYSIS.md`
    *   Remove references to the tight coupling between CrewAI and Hephaestus.

### 2.2. Strategy & Roadmap Updates
*   **Target:** `SKILL.md`
    *   Remove legacy aliasing notes in sections `734` and `860-862`.
*   **Target:** `DETAILED_COMPLETION_REPORT.md` (Self-Audit)
    *   Ensure the "Orchestration Migration" section is updated to reflect that the transition is finalized and all legacy code has been purged.

---

## **Phase 3: Tooling Retirement**
*Goal: Decommission scripts that exist solely to monitor the migration.*

### 3.1. Verification Script Update
*   **Target:** `verify_core_crewai_files.py`
    *   This script currently scans for Hephaestus imports. 
    *   **Action:** After Phase 1 is complete, retire this script or repurpose it into a general "Sovereign Compliance" auditor.

---

## **Phase 4: Cache Purge**
*Goal: Remove all compiled remnants of the deprecated system.*

*   **Action:** Manually clear all `__pycache__` directories containing `.pyc` files with the name `hephaestus`.
*   **Files identified:**
    *   `integrations/bug_fixes/__pycache__/hephaestus_config_fix...`
    *   `__pycache__/bubblelabs_hephaestus_bridge...`
    *   `__pycache__/hephaestus_integration...`

---

## **Phase 5: Hardware Backend Cleanup (High Priority)**
*Goal: Resolve legacy hardware acceleration hooks originally tied to the Hephaestus ecosystem.*

### 5.1. Identification of Acceleration Stubs
*   **Task:** Investigate files related to solver execution (e.g., `blue_team_solver_engine.py`, `physics_validator.py`) for low-level hardware acceleration hooks (GPU/TPU/MPS).
*   **Focus:** Identify any "Simulation Mode" implementations that default to CPU execution when hardware backends are unavailable.

### 5.2. Logic Re-wiring
*   **Action:** Systematically remove hardware-related placeholder logic that was waiting for Hephaestus integration.
*   **Action:** Direct all hardware acceleration requests directly to the native solvers (e.g., **NeuroMANCER** on PyTorch or **Z3** solvers) instead of routing through deprecated bridge layers.
*   **Verification:** Ensure high-performance execution is verified via physical grounding tests rather than heuristic simulations.

---

## **Final State Definition**
Once this roadmap is complete, a search for the case-insensitive string `"hephaestus"` across the entire project directory should yield **zero results**. The system will be 100% described and implemented as a **CrewAI-native** architecture with direct, high-performance hardware backend utilization.
