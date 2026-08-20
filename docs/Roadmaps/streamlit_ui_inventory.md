# BubbleLab UI UI Inventory → BubbleLab Mapping

Goal: enumerate all BubbleLab UI UI surfaces, the data they touch, and the BubbleLab targets / backend endpoints for 1:1 parity.

## Primary Entry Points

1) `main.py`
   - Purpose: BubbleLab UI app shell, backend service control, startup thread, global CSS tweaks, entry navigation.
   - Data: orchestrator status, backend health, sidebar selections, session state flags.
   - Backend: `openevolve_orchestrator` start/stop/restart, `/health` in `api_server.py`.
   - BubbleLab target: App shell + top‑level routing + service control panel.

2) `mainlayout.py`
   - Purpose: main UI composition, tabs, dashboards, workflow panels, reporting, analytics, integrations.
   - Data: prompts, workflow state, analytics, reports, tasks, notifications, version control.
   - Backend: `OpenEvolveAPI`, `openevolve_orchestrator`, analytics/reporting functions, GitHub integrations.
   - BubbleLab target: `OpenEvolveApp.tsx` tabs + sub‑tabs.

3) `ui_components.py`
   - Purpose: modular UI builders for team manager, gauntlet designer, workflow orchestrator, monitoring, analytics, KB explorer, templates.
   - Data: `TeamManager`, `GauntletManager`, workflow history, knowledge engine artifacts.
   - Backend: `/teams`, `/gauntlets`, `/workflows`, `/statistics`, knowledge engine APIs.
   - BubbleLab target: dedicated TS tabs (Admin, Monitoring, Analytics, Workflow, Knowledge).

## BubbleLab UI Screens & Components (Mapping)

### Admin / Setup

- Team Manager
  - File: `ui_components.py::render_team_manager`
  - Data: `Team`, `ModelConfig`, team prompts and metadata
  - Backend: `/teams` (GET/POST/PUT/DELETE) in `api_server.py`
  - BubbleLab: `TeamManagerTab.tsx` (implemented)

- Gauntlet Designer
  - File: `ui_components.py::render_gauntlet_designer`
  - Data: `GauntletDefinition`, `GauntletRoundRule`
  - Backend: `/gauntlets` (GET/POST/PUT/DELETE) in `api_server.py`
  - BubbleLab: `GauntletDesignerTab.tsx` (implemented)

### Workflow Orchestration

- Workflow Orchestrator / Sovereign Workflow
  - Files: `workflow_engine.py` UI hooks, `ui_components.py::render_workflow_orchestrator`
  - Data: `WorkflowState`, `DecompositionPlan`, sub‑problem status, gauntlet reports
  - Backend: `/workflows`, `/workflows/{id}`, `/workflows/{id}/pause|resume|results`
  - BubbleLab target: `OrchestratorTab` + workflow stage panels

- Manual Review Panel
  - File: `ui_components.py::render_manual_review_panel`
  - Data: per‑subproblem edits, approvals, targeted feedback
  - Backend: workflow state persistence
  - BubbleLab target: nested panel in Orchestrator

### Analytics & Monitoring

- Analytics Dashboard
  - File: `ui_components.py::render_analytics_dashboard` and helpers
  - Data: workflow history, quality metrics, team performance
  - Backend: `/statistics`, `/icr/analytics/*`, `/icr/dashboard`
  - BubbleLab target: `AnalyticsDashboardTab` + charts

- Analytics Settings
  - File: `analytics.py::render_analytics_settings`
  - Data: data collection toggles, report defaults, retention policy
  - Backend: local UI state
  - BubbleLab target: `SettingsTab` (implemented)

- Monitoring Dashboard
  - Files: `ui_components.py::render_monitoring_tab`, `monitoring_dashboard.py`
  - Data: resource usage, alerts, logs
  - Backend: `/adaptive-mdap/*`, `/statistics`, log streaming
  - BubbleLab target: `MonitoringTab` + log viewer panel

### Evolution / Adversarial / Knowledge

- Evolution Engine
  - Files: `evolution.py`, `ui_components.py::render_openevolve_config_panel`
  - Data: evolution configs, run history
  - Backend: `run_unified_evolution`, `/api/openevolve/*`, `/determinism/*`
  - BubbleLab target: `EvolutionTab` + config editor

- Adversarial Testing
  - Files: `adversarial.py`, `adversarial_testing.py`, `ui_components.py` helpers
  - Data: attack modes, red team outcomes, verification results
  - Backend: adversarial pipelines + gauntlet endpoints
  - BubbleLab target: `AdversarialTestingTab`

- Knowledge Base Explorer
  - Files: `ui_components.py::render_knowledge_base_interface` + artifact views
  - Data: knowledge artifacts, graph views, search
  - Backend: knowledge engine APIs
  - BubbleLab target: `OpenEvolveDashboardTab` or dedicated Knowledge tab

### Templates, Reports, Tasks, Integrations

- Report Templates / Generator
  - Files: `ui_components.py::render_custom_report_generator`, `integrated_reporting.py`
  - Data: workflow summaries, exported artifacts
  - Backend: reporting APIs / filesystem output
  - BubbleLab target: `ReportTemplatesTab`

- Export / Import Manager
  - File: `export_import_manager.py`
  - Data: project snapshots, templates, settings, workflow history
  - Backend: local UI state + filesystem download/upload
  - BubbleLab target: `ExportImportTab` (implemented)

- Tasks / Suggestions / Collaboration
  - Files: `tasks.py`, `suggestions.py`, `collaboration.py`
  - Data: task objects, collaboration state
  - Backend: internal storage + collaboration server
  - BubbleLab target: `TasksTab` + Collaboration panel

- GitHub Integrations
  - File: `integrations.py` (GitHub), `bubblelabs_*` integrations
  - Data: repo list, commit/branch actions
  - Backend: `integrations.py` endpoints
  - BubbleLab target: `GithubIntegrationTab`

## API Endpoint Mapping (FastAPI)

Key endpoints in `api_server.py` to wire into BubbleLab:
- `GET /health`
- `GET /workflows`, `GET /workflows/{id}`, `POST /workflows/{id}/pause`, `POST /workflows/{id}/resume`, `GET /workflows/{id}/results`
- `GET /teams`, `POST /teams`, `PUT /teams/{team_name}`, `DELETE /teams/{team_name}`
- `GET /gauntlets`, `POST /gauntlets`, `PUT /gauntlets/{gauntlet_name}`, `DELETE /gauntlets/{gauntlet_name}`
- `GET /statistics`
- `GET /icr/dashboard`, `/icr/analytics/*`, `/icr/config`
- `POST /determinism/generate`, `POST /determinism/check`
- `POST /api/openevolve/*` (visualization, dspy, fixes)
- `POST /adaptive-mdap/*`, `GET /adaptive-mdap/*`

## BubbleLab Target Structure

Root: `openevolve-sdk/src/components/openevolve/main`
- `OpenEvolveApp.tsx` (app shell + tabs)
- Tabs to implement fully: `AnalyticsDashboardTab`, `MonitoringTab`, `OrchestratorTab`, `OpenEvolveDashboardTab`, `ReportTemplatesTab`, `GithubIntegrationTab`, `TasksTab`, `ModelDashboardTab`.

New/updated tabs started:
- `TeamManagerTab.tsx`
- `GauntletDesignerTab.tsx`
- `AdminTab.tsx` (hosts Teams/Gauntlets)
- `AnalyticsMonitoringTab.tsx`
- `SystemMonitoringTab.tsx`
- `SgdMonitoringTab.tsx`
- `WorkflowVisualizationTab.tsx`
- `OpenEvolveVisualizationTab.tsx`
- `DependencyGraphTab.tsx` (analysis + matrix views)
- `ExportImportTab.tsx`

## Immediate Next Port Targets

1) Workflow Orchestrator (parity with `workflow_engine.py` + `render_workflow_orchestrator`)
2) Knowledge Base Explorer and reports
3) Integrations (GitHub, BubbleLabs adapters)

## Notes

- BubbleLab UI must replace BubbleLab UI `st.session_state` with explicit app state + persistence (localStorage or backend).
- All BubbleLab UI‑only dependencies should be removed after porting.

