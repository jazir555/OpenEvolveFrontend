# BubbleLab UI → BubbleLab UI Port Roadmap (Full Parity)

## Goal
Replace every BubbleLab UI UI surface with BubbleLab TypeScript components while preserving full business logic and user workflows. The BubbleLab UI implementation will be removed once parity is confirmed.

## Phase 1 — Inventory + Mapping (Week 1)
1. **Inventory all BubbleLab UI UI surfaces**
   - Scan `*.py` for `import BubbleLab UI` + UI renderers (tabs, dashboards, sidebar, expander flows).
   - Map each UI surface to BubbleLab tab/section(s).
2. **Define API contracts for missing UI data**
   - Identify backend data required for each UI module (analytics, monitoring, dependencies, prompts, templates, collaboration, etc.).
   - Add REST/WebSocket endpoints for any missing data paths.
3. **Parity checklist**
   - Document features per BubbleLab UI module with a 1:1 parity checklist.

## Phase 2 — Port + Wire (Week 1–2)
1. **Port core tabs**
   - Evolution, Adversarial, Team, Gauntlet, Orchestrator, Dashboard (already in BubbleLab) — verify parity.
2. **Port remaining BubbleLab UI dashboards**
   - Analytics Dashboard (workflow/team/gauntlet/solution quality/knowledge stats).
   - Monitoring Dashboard (system/resource/alerts/metrics) + real-time websocket feed.
   - SGD Monitoring dashboard.
   - Dependency Visualizer + Workflow Visualizer.
   - Collaboration UI (real-time presence, content updates, cursor updates, sharing).
   - Prompt Manager + Content Management tools (templates, validation, prompt storage).
3. **Backend wiring**
   - Add missing API endpoints and ensure JSON contracts align with TS types.
   - Add any missing business logic (no simulated data).

## Phase 3 — Finalize + Remove BubbleLab UI (After parity lock)
1. **Final scan (gating step)**
   - Re-scan for any remaining BubbleLab UI UI usage and port anything missing.
2. **Remove BubbleLab UI UI codepaths**
   - Delete BubbleLab UI entrypoints (`main.py`, `mainlayout.py`, `sidebar.py`, `ui_components.py`, `ui_components_additional.py`, etc.).
   - Remove BubbleLab UI dependencies from Python modules (replace `st.session_state` with proper storage).
3. **End-to-end verification**
   - Run BubbleLab UI + API integration tests.
   - Confirm zero BubbleLab UI imports remain.

## Notes
- Before starting Phase 3, a **full scan** is required to ensure nothing is left unported.
- Any feature parity gaps discovered during Phase 2 are treated as blockers.

