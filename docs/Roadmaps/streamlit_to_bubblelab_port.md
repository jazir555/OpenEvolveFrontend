# Streamlit → BubbleLab TypeScript Port Roadmap

Goal: complete 1:1 feature parity by replacing all Streamlit UI surfaces with BubbleLab TypeScript, then remove Streamlit dependencies entirely.

## Scope Inventory (initial)

Primary UI entry points (Streamlit):
- `main.py` (main app shell / navigation)
- `ui_components.py` + `ui_components_additional.py` (core UI panels)
- `mainlayout.py` (layout + third‑party streamlit widgets)
- `workflow_engine.py` (UI‑emitting workflow orchestration)
- `analytics_dashboard.py`, `analytics_monitoring_dashboard.py`, `monitoring_dashboard.py`
- `adversarial.py`, `adversarial_testing.py`, `evolution.py` (domain dashboards)
- `knowledge_base_ui.py`, `knowledge_base` visualizations
- `bubblelabs_*_ui.py` / `bubblelabs_*_integration*.py` (integration UIs)
- `dependency_visualizer.py`, `workflow_visualization.py`
- `collaboration.py`, `collaboration_manager.py`
- `configuration_system.py`, `evaluator_uploader.py`, `export_import_manager.py`
- `tasks.py`, `suggestions.py`, `state.py`, `log_streaming.py`

BubbleLab target:
- Use `bubblelab-converted/src/components/openevolve` as the primary UI root unless another BubbleLab app is specified.

## Phased Order (execution can overlap)

### Phase 0 — Inventory & Parity Matrix
- Catalog each Streamlit screen/panel, inputs, outputs, and data dependencies.
- Map each to a BubbleLab component and backend API surface.
- Define parity checklist per screen (inputs, actions, outputs, charts, logs, state persistence).

### Phase 1 — Core Shell & Navigation
- BubbleLab app shell: tabs/sections, routing, global state, and persistence.
- Replace `main.py`, `mainlayout.py`, core `ui_components.py` layout with TS components.
- Ensure stateful workflows (session state equivalents) exist in BubbleLab.

### Phase 2 — Workflow Orchestration UI
- Port Streamlit workflow controls and run status views from `workflow_engine.py` UI hooks.
- Implement UI hooks as explicit events/state rather than Streamlit calls.
- Ensure 1:1 behavior for gauntlet status, stages, and progress.

### Phase 3 — Domain Dashboards
- Port analytics, monitoring, adversarial, evolution, knowledge base dashboards.
- Port charts, tables, and configuration panels.
- Implement streaming/log views as BubbleLab components.

### Phase 4 — Integrations & Tooling
- Port BubbleLabs integration UIs, LeanAide UI, GitHub integrations, and evaluators.
- Replace any Streamlit‑specific embedding (e.g., Streamlit components) with BubbleLab equivalents.

### Phase 5 — Cleanup & Removal
- Remove Streamlit dependencies from code and requirements.
- Remove Streamlit‑only runtime paths, CLI entrypoints, and docs.
- Update tests and integration harnesses to use BubbleLab UI.

## Acceptance Criteria (global)

- Every Streamlit UI element has a BubbleLab counterpart with identical behavior.
- All workflows are operable from BubbleLab UI with parity in outputs and side effects.
- No Streamlit imports or runtime calls remain in the codebase.
- Tests (unit/integration) are updated for BubbleLab UI and pass.

## Current Start Point

Start with Phase 0 inventory by enumerating Streamlit screens and documenting required data/controls. Then implement Phase 1 shell/routing in BubbleLab, followed by phase 2+ components in priority order.
