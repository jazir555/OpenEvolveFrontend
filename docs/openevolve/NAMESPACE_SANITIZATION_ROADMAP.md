# Roadmap: Namespace Sanitization & Directory Restructuring

## **Objective**
To transform the root directory from an unmanaged list of 800+ scripts into a professional, navigable package structure while preserving 100% of functional logic.

---

## **Phase 1: The Archive Operation**
*Goal: Remove "Noise" from the active development path.*

### 1.1. Identify Patch/Backup Patterns
*   **Target:** All files containing `_FIXED`, `_backup`, `_v1`, `_v2`, `_simple`, `_copy`.
*   **Action:** Move these ~250 files into a timestamped `/archive/patches/` directory.

### 1.2. Retire Standalone Tests
*   **Action:** Move the hundreds of `test_*.py` and `demo_*.py` root files into a centralized `/tests` and `/demos` directory structure.

---

## **Phase 2: Functional Categorization**
*Goal: Group the 500+ logic files into domain-specific packages.*

### 2.1. Establish Core Packages
Move root files into the following new directories:
*   `/engines/decomposition`: (e.g., `comprehensive_decomposition_engine.py`)
*   `/engines/search`: (e.g., `hybrid_mcts_framework.py`, `adversarial_mdap_mcts.py`)
*   `/engines/verification`: (e.g., `leanaide_evolution.py`, `formal_gauntlet_system.py`)
*   `/ui/bubblelabs`: (e.g., `bubblelabs_ui_component.py`, `openevolve_bubblelabs_api.py`)

### 2.2. Update Path Logic
*   **Action:** Update the `PYTHONPATH` settings in `start_bubblelabs_integration.py` to ensure all new package locations are discoverable.

---

## **Phase 3: The "Active Root" Enforcement**
*Goal: Prevent future root-level clutter.*

*   **Rule:** The root directory should only contain entry points (`main.py`, `app.py`), global configs (`.env`, `config.yaml`), and roadmap documentation.
*   **Action:** Repurpose `verify_core_crewai_files.py` to flag any new non-authorized files created in the root.

---

## **Final State**
A codebase where "Decomposition" logic lives in one folder, "Search" in another, and the root is clean and manageable.
