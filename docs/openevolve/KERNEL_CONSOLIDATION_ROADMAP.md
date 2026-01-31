# Roadmap: Kernel Consolidation (Data Model Unification)

## **Objective**
To eliminate "Schema Drift" and import friction by merging the three fragmented data model files into a single, high-performance "Kernel" package.

---

## **Phase 1: The New Kernel (`openevolve.kernel`)**
*Goal: Create the single source of truth for all data structures.*

### 1.1. Create the Kernel Package
*   **Action:** Create `openevolve/kernel/__init__.py` and `openevolve/kernel/schema.py`.
*   **Action:** Define the "Master" version of core classes:
    *   `WorkflowState` (Merge versions from `openevolve_structures` and `workflow_structures`).
    *   `SubProblem` (Include all metadata fields needed by both Decomposition and Solving).
    *   `GauntletRoundRule` (Ensure unified logic for all 8 gauntlet types).

### 1.2. The Migration Bridge
*   **Action:** Update `openevolve_imports.py` to point all legacy aliases to the new `openevolve.kernel.schema`.
*   **Outcome:** Code that imports from `openevolve_structures` will automatically receive the new consolidated objects.

---

## **Phase 2: Global Import Enforcement**
*Goal: Force all 800+ root-level files to recognize the unified kernel.*

### 2.1. Standardize Imports
*   **Action:** Use a regex-based script to replace direct imports of fragmented models with a single import from `openevolve_imports`.
    *   *From:* `from openevolve_structures import WorkflowState`
    *   *To:* `from openevolve_imports import WorkflowState`

### 2.2. Validation of Class Signatures
*   **Action:** Run `test_mega_structure_integration.py` to ensure that merging models didn't break specialized logic (e.g., `GauntletExecution` attributes).

---

## **Phase 3: Cleanup**
*Goal: Remove the technical debt of triplication.*

*   **Action:** Once all imports are verified, move the three legacy files into `/archive/legacy_structures/`:
    1.  `sovereign_data_models.py`
    2.  `openevolve_structures.py`
    3.  `workflow_structures.py`

---

## **Final State**
A single, consistent data language used by all 30+ engines, allowing the Gauntlet System to seamlessly pass data to the Decomposition Engine without `AttributeError` or `ImportError`.
