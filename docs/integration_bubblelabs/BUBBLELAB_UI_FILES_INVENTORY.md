# BubbleLab UI FILES INVENTORY
## Complete Catalog of All BubbleLab UI UI Files for Migration to BubbleLab

**Generated:** 2025-01-05
**Agent:** Discovery & Audit Agent
**Mission:** Catalog all BubbleLab UI code for migration to TypeScript/React BubbleLab UI

---

## EXECUTIVE SUMMARY

**Total BubbleLab UI Files Found:** 96 application files (excluding library dependencies)

### Breakdown by Module:
- **Core OpenEvolve UI:** 42 files
- **LeanAide Server UI:** 8 files
- **OneKE Frontend:** 4 files
- **Test/Demo Files:** 12 files
- **Integration/UI Components:** 30 files

### Migration Priority Levels:
- **CRITICAL (Core App):** main.py, mainlayout.py, sidebar.py, ui_components.py
- **HIGH (Feature Modules):** evolution.py, adversarial.py, integrated_workflow.py
- **MEDIUM (Analytics/Dashboards):** openevolve_dashboard.py, analytics_dashboard.py
- **LOW (Tests/Demos):** demo_*.py, test_*.py files

---

## SECTION 1: CORE OPENEVOLVE UI FILES

### 1.1 Main Application Entry Points

---

## File: `main.py`

**Purpose:** Primary application entry point. Initializes backend services, manages session state, and renders main UI layout.

**BubbleLab UI Components Used:**
- `st.set_page_config()` - Page configuration (wide layout)
- `st.markdown(custom_css, unsafe_allow_html=True)` - Custom CSS injection
- `st.info()`, `st.warning()`, `st.error()` - Status messages
- `st.columns()` - Multi-column layout
- `st.expander()` - Collapsible sections
- `st.tabs()` - Tab-based navigation (Dashboard, BubbleLabs Workflows, n8n Visual Workflows)
- `st.session_state` - Extensive session state management

**Backend Functions Called:**
- `start_openevolve_services()` - Starts Flask/FastAPI backend in separate thread
- `render_openevolve_dashboard()` - Main dashboard rendering
- `render_bubblelabs_workflow_ui()` - BubbleLab integration UI
- `render_complete_n8n_integration()` - n8n workflow UI
- `display_sidebar()` - Sidebar configuration UI
- `render_main_layout()` - Main layout rendering
- `start_collaboration_server()` - Real-time collaboration
- `LogStreaming.run_flask_app_in_thread()` - Log streaming service

**Session State Usage:**
- `st.session_state["backend_started"]` - Backend initialization flag
- `st.session_state["evolution_running"]` - Evolution state tracking
- `st.session_state["adversarial_running"]` - Adversarial testing state
- `st.session_state["thread_lock"]` - Thread synchronization
- `st.session_state.openevolve_backend_processes` - Backend process PIDs
- `st.session_state["feature_dimensions"]` - Feature configuration
- `st.session_state.welcome_shown` - Welcome screen state
- `st.session_state._session_state_initialized` - Initialization flag

**Real-time Updates:**
- Uses `backend_message_queue` (queue.Queue) for cross-thread communication
- Background thread for backend service management
- Flask app for log streaming (runs in separate thread)
- Auto-refresh via `st.rerun()` for state updates

**External Dependencies:**
- `yaml` - Configuration loading (config.yaml)
- `requests` - Health check HTTP requests
- `threading` - Background service management
- `queue` - Message passing between threads
- Custom CSS for slider styling and UI stability

**Migration Notes:**
- **CRITICAL:** Needs React context for global state management (replaces session_state)
- **CRITICAL:** WebSocket connection for real-time backend status updates
- **CRITICAL:** Service lifecycle management (start/stop/restart backends)
- Custom CSS needs to be converted to styled-components or Tailwind
- Welcome screen should be React onboarding component
- Tab navigation maps to React Router routes
- Background thread management needs different approach (possibly Web Workers)

**Complexity Rating:** ⭐⭐⭐⭐⭐ (5/5) - HIGH COMPLEXITY

---

## File: `mainlayout.py`

**Purpose:** Main layout renderer. Handles all core UI components including evolution, adversarial testing, content management, and analytics.

**BubbleLab UI Components Used:**
- `st.form()` - Form containers for user input
- `st.text_area()`, `st.text_input()` - Content and parameter input
- `st.number_input()` - Numeric parameters
- `st.slider()` - Parameter tuning (temperature, iterations, etc.)
- `st.selectbox()`, `st.multiselect()` - Model and strategy selection
- `st.button()` - Action triggers
- `st.columns()` - Complex multi-column layouts
- `st.tabs()` - Tab-based content organization
- `st.progress()` - Progress indicators
- `st.empty()` - Placeholder for dynamic content updates
- `st.plotly_chart()` - Interactive data visualizations
- `st.dataframe()` - Tabular data display
- `st.json()` - JSON data inspection
- `st.markdown()` - Rich text/HTML rendering
- `st.session_state` - Extensive state management
- `st_autorefresh` - Auto-refresh for real-time updates
- `st_tags` - Tag input for feature dimensions
- `html()` from `BubbleLab UI.components.v1` - Custom HTML injection

**Backend Functions Called:**
- `OpenEvolveAPI` - Main API client for backend communication
- `_run_evolution_with_api_backend_refactored()` - Evolution engine
- `run_adversarial_testing()` - Adversarial testing engine
- `create_github_branch()`, `commit_to_github()` - GitHub integration
- `send_discord_notification()`, `send_msteams_notification()` - Notifications
- `create_task()`, `get_tasks()` - Task management
- `content_manager` - Content CRUD operations
- `PromptManager` - Prompt template management
- `TemplateManager` - Template loading/saving
- `AnalyticsManager` - Performance analytics
- `CollaborationManager` - Real-time collaboration
- `VersionControl` - Version history
- `NotificationManager` - User notifications
- `render_monitoring_tab()` - System monitoring
- `generate_integrated_report()` - Report generation

**Session State Usage:**
- Evolution state: `evolution_running`, `evolution_results`, `population_data`
- Adversarial state: `adversarial_running`, `red_team_results`, `blue_team_results`
- Content: `current_content`, `content_history`, `content_metadata`
- Configuration: `temperature`, `top_p`, `max_iterations`, `population_size`
- Models: `selected_models`, `model_configs`, `provider_settings`
- Analytics: `performance_metrics`, `quality_scores`, `consensus_data`
- UI State: `active_tab`, `expanded_sections`, `user_preferences`

**Real-time Updates:**
- `st_autorefresh` for automatic page refresh (configurable interval)
- `st.empty()` containers for live evolution progress
- Progress bars with real-time iteration updates
- Live chart updates during evolution (plotly figures)
- Auto-refreshing dataframes for monitoring

**External Dependencies:**
- `plotly.express` - Interactive charts
- `matplotlib.pyplot` - Static plots
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `altair` - Declarative visualizations
- `pyvis.network` - Network graphs
- `bubblelab_autorefresh` - Auto-refresh capability
- `bubblelab_tags` - Tag input widgets

**Migration Notes:**
- **CRITICAL:** Complex state management needs Redux or Zustand
- **CRITICAL:** Real-time updates need WebSocket or polling
- **CRITICAL:** Plotly charts need Recharts or Victory in React
- Forms map to React forms (react-hook-form or formik)
- Session state becomes React context + server state
- Auto-refresh becomes React Query refetchInterval
- Progress bars need custom React component with WebSocket updates
- Tab navigation becomes React Router
- Multi-column layouts use CSS Grid or Flexbox

**Complexity Rating:** ⭐⭐⭐⭐⭐ (5/5) - VERY HIGH COMPLEXITY

**Lines of Code:** ~5000+ (very large file with multiple features)

---

## File: `sidebar.py`

**Purpose:** Sidebar configuration UI for LLM providers, API keys, model selection, and parameter tuning.

**BubbleLab UI Components Used:**
- `st.sidebar` - All sidebar-specific components
- `st.selectbox()` - Provider and model selection
- `st.text_input()` - API key input (type="password")
- `st.slider()` - Temperature, top_p, frequency_penalty, presence_penalty
- `st.number_input()` - max_tokens, seed, n
- `st.text_area()` - System prompts, user prompts
- `st.expander()` - Collapsible configuration sections
- `st.form()` - Configuration form containers
- `st.button()` - Save, reset, apply actions
- `st.session_state` - Configuration state storage

**Backend Functions Called:**
- `get_providers()` - Get available LLM providers
- `get_openrouter_models()` - Fetch OpenRouter model list
- `save_user_preferences()` - Persist user settings
- `reset_defaults()` - Reset to default configuration
- `UnifiedConfiguration` - Parameter validation and management
- `OpenEvolveAPI` - API client initialization

**Session State Usage:**
- Provider settings: `provider`, `model`, `api_key`, `api_base`
- Generation params: `temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`
- Evolution params: `max_iterations`, `population_size`, `num_islands`
- Scope management: `settings_scope` (Global/Provider/Model)
- Configuration cache: `parameter_settings`, `user_preferences`

**Real-time Updates:**
- Re-renders on provider/model change
- Dynamic model list loading from API
- Real-time parameter validation
- Scope-based configuration updates

**External Dependencies:**
- `unified_configuration` - Unified parameter management
- `providercatalogue` - Provider and model catalog
- `session_utils` - Session state utilities

**Migration Notes:**
- Sidebar becomes React sidebar component
- Provider/model selection becomes dropdowns
- API key management needs secure storage (possibly backend encryption)
- Parameter tuning needs synchronized form controls
- Hierarchical configuration (Global → Provider → Model) needs context propagation
- Form validation needs React Hook Form or Formik
- Settings persistence via API endpoints

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `ui_components.py`

**Purpose:** Reusable UI components for team management, gauntlet configuration, and workflow visualization.

**BubbleLab UI Components Used:**
- `st.header()`, `st.subheader()` - Section headings
- `st.form()` - Team/Gauntlet creation forms
- `st.text_input()`, `st.text_area()` - Configuration input
- `st.selectbox()` - Role and type selection
- `st.slider()`, `st.number_input()` - Parameter configuration
- `st.checkbox()`, `st.multiselect()` - Option selection
- `st.columns()` - Multi-column layouts
- `st.expander()` - Collapsible sections
- `st.json()` - JSON configuration display
- `st.button()` - Action buttons
- `st.session_state` - Component state management

**Backend Functions Called:**
- `TeamManager` - Team CRUD operations
- `GauntletManager` - Gauntlet/test suite management
- `WorkflowStructures` - Data structures for teams, gauntlets, plans

**Session State Usage:**
- Manager instances: `team_manager`, `gauntlet_manager`
- Team data: `teams`, `team_configs`
- Gauntlet data: `gauntlets`, `gauntlet_rules`

**Real-time Updates:**
- Re-renders on team/gauntlet CRUD operations
- Live JSON configuration updates

**External Dependencies:**
- `workflow_structures` - Team, Gauntlet, Plan data structures
- `team_manager` - Team persistence
- `gauntlet_manager` - Gauntlet persistence
- `json` - Configuration serialization

**Migration Notes:**
- Components become React components
- Forms become react-hook-form forms
- Session state becomes component state or context
- Team/Gauntlet managers become API clients
- JSON display becomes code highlighting component
- Multi-column layouts become CSS Grid

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## SECTION 2: EVOLUTION & ADVERSARIAL TESTING UI

### 2.1 Evolution Engine UI

---

## File: `evolution.py`

**Purpose:** Evolutionary optimization UI for content improvement using genetic algorithms and multi-model ensembles.

**BubbleLab UI Components Used:**
- `st.tabs()` - Evolution modes (Standard, Quality-Diversity, Island Model)
- `st.form()` - Evolution parameter input
- `st.text_area()` - Initial content input
- `st.slider()`, `st.number_input()` - Evolution parameters (iterations, population, mutation rate, etc.)
- `st.selectbox()`, `st.multiselect()` - Model ensemble selection
- `st.button()` - Start, stop, pause evolution
- `st.progress()` - Generation progress
- `st.empty()` - Live population updates
- `st.plotly_chart()` - Fitness landscape visualization
- `st.dataframe()` - Population table
- `st.code()` - Best solution display
- `st.session_state` - Evolution state tracking

**Backend Functions Called:**
- `_run_evolution_with_api_backend_refactored()` - Main evolution engine
- `OpenEvolveAPI.run_evolution()` - API-based evolution
- `EvolutionaryOptimizer` - Genetic algorithm implementation
- `QualityDiversityOptimizer` - QD algorithm implementation
- `IslandModelOptimizer` - Island model implementation

**Session State Usage:**
- Evolution control: `evolution_running`, `evolution_paused`, `evolution_complete`
- Population data: `current_population`, `population_history`, `best_individual`
- Parameters: `max_iterations`, `population_size`, `mutation_rate`, `crossover_rate`
- Metrics: `fitness_history`, `diversity_metrics`, `convergence_data`

**Real-time Updates:**
- Generation-by-generation progress updates
- Live fitness landscape visualization
- Real-time population table updates
- Progress bar with current generation

**External Dependencies:**
- `evolutionary_optimization.py` - Evolution algorithms
- `model_orchestration.py` - Model ensemble management
- `plotly.express` - Fitness visualizations
- `pandas` - Population data management
- `numpy` - Numerical operations

**Migration Notes:**
- Evolution modes become React tabs or routes
- Real-time updates need WebSocket for generation progress
- Fitness landscape needs React charting library (Recharts/Victory)
- Population table becomes React Table or Material UI Table
- Evolution controls become React form with validation
- Backend evolution engine runs asynchronously (needs WebSocket or polling)

**Complexity Rating:** ⭐⭐⭐⭐ (4/5) - HIGH COMPLEXITY

---

## File: `adversarial.py`

**Purpose:** Adversarial testing UI with red team/blue team methodology for content hardening.

**BubbleLab UI Components Used:**
- `st.columns()` - Red team / Blue team layout
- `st.form()` - Attack mode and defense configuration
- `st.text_area()` - Content input for testing
- `st.multiselect()` - Attack mode selection (prompt injection, jailbreak, etc.)
- `st.selectbox()` - Model selection for teams
- `st.button()` - Run adversarial test, approve/deny patches
- `st.progress()` - Testing progress
- `st.empty()` - Live attack/defense updates
- `st.tabs()` - Attack results, defense patches, consensus
- `st.json()` - Attack payloads and results
- `st.code()` - Vulnerable code display
- `st.session_state` - Adversarial testing state

**Backend Functions Called:**
- `run_adversarial_testing()` - Main adversarial testing engine
- `BlueTeam` - Defense generation (patching)
- `RedTeam` - Attack generation
- `GoldTeam` - Consensus evaluation
- `_load_human_feedback()` - Human-in-the-loop feedback
- `OpenEvolveAPI.run_adversarial_test()` - API-based testing

**Session State Usage:**
- Testing control: `adversarial_running`, `adversarial_complete`
- Attack data: `red_team_attacks`, `attack_payloads`, `vulnerabilities_found`
- Defense data: `blue_team_patches`, `patch_approvals`, `patch_rejections`
- Evaluation: `gold_team_evaluations`, `consensus_scores`
- Configuration: `attack_modes`, `defense_strategies`, `team_models`

**Real-time Updates:**
- Live attack/defense streaming
- Progress bar for testing phases
- Real-time vulnerability discovery
- Patch approval workflow updates

**External Dependencies:**
- `blue_team.py` - Blue team engine
- `adversarial_testing.py` - Adversarial testing framework
- `evaluator_team.py` - Gold team evaluation
- `plotly.express` - Attack success rate visualization

**Migration Notes:**
- Red/blue team layout becomes React grid layout
- Attack modes become React multi-select
- Live attack/defense streaming needs WebSocket
- Patch approval workflow needs React state machine
- Vulnerability display needs syntax highlighting component
- Consensus visualization needs React charts
- Human feedback needs React form with approval buttons

**Complexity Rating:** ⭐⭐⭐⭐⭐ (5/5) - VERY HIGH COMPLEXITY

---

## File: `integrated_workflow.py`

**Purpose:** Unified workflow UI combining evolution, adversarial testing, and quality assessment.

**BubbleLab UI Components Used:**
- `st.tabs()` - Workflow stages (Plan → Evolve → Test → Evaluate)
- `st.form()` - Workflow configuration
- `st.text_area()` - Problem statement and content input
- `st.selectbox()` - Workflow template selection
- `st.slider()`, `st.number_input()` - Workflow parameters
- `st.button()` - Start workflow, advance to next stage
- `st.progress()` - Overall workflow progress
- `st.empty()` - Stage-specific updates
- `st.status()` - Stage status indicators
- `st.session_state` - Workflow state machine

**Backend Functions Called:**
- `DecompositionEngine` - Problem decomposition
- `InventionPlanner` - Planning phase
- `EvolutionEngine` - Evolution phase
- `AdversarialEngine` - Testing phase
- `QualityAssessmentEngine` - Evaluation phase
- `WorkflowOrchestrator` - End-to-end workflow management

**Session State Usage:**
- Workflow state: `current_stage`, `workflow_complete`, `workflow_failed`
- Stage data: `plan_data`, `evolution_data`, `test_data`, `evaluation_data`
- Configuration: `selected_template`, `workflow_params`
- Artifacts: `decomposition_plan`, `evolution_results`, `test_results`, `final_report`

**Real-time Updates:**
- Stage-by-stage progress updates
- Live artifact generation
- Status indicator updates

**External Dependencies:**
- `decomposition_engine.py` - Problem decomposition
- `end_to_end_invention_planner.py` - Planning engine
- `evolution.py` - Evolution engine
- `adversarial.py` - Testing engine
- `advanced_validation_workflows.py` - Quality assessment

**Migration Notes:**
- Workflow stages become React wizard or stepper component
- Stage state management needs React state machine (XState)
- Real-time progress needs WebSocket
- Artifacts display needs collapsible panels
- Workflow templates become preset configurations
- Overall progress becomes multi-step progress bar

**Complexity Rating:** ⭐⭐⭐⭐⭐ (5/5) - VERY HIGH COMPLEXITY

---

## SECTION 3: ANALYTICS & DASHBOARD UI

### 3.1 Monitoring Dashboards

---

## File: `openevolve_dashboard.py`

**Purpose:** Main dashboard UI showing system status, performance metrics, and recent activity.

**BubbleLab UI Components Used:**
- `st.metrics()` - Key metric cards (evolutions run, tests passed, etc.)
- `st.plotly_chart()` - Performance trend charts
- `st.dataframe()` - Recent activity table
- `st.tabs()` - Dashboard sections (Overview, Analytics, Settings)
- `st.status()` - System health indicators
- `st.progress()` - Resource utilization bars
- `st.session_state` - Dashboard state

**Backend Functions Called:**
- `AnalyticsManager.get_metrics()` - Performance metrics
- `AnalyticsManager.get_recent_activity()` - Activity log
- `MonitoringSystem.get_system_health()` - Health checks
- `OpenEvolveAPI.get_status()` - Backend status

**Session State Usage:**
- Dashboard config: `dashboard_refresh_interval`, `selected_metrics`
- Data cache: `cached_metrics`, `cached_activity`
- Auto-refresh: `auto_refresh_enabled`

**Real-time Updates:**
- Auto-refreshing metrics (via `st_autorefresh`)
- Live resource utilization updates
- Real-time activity stream

**External Dependencies:**
- `analytics_manager.py` - Analytics data
- `monitoring_system.py` - System monitoring
- `plotly.express` - Trend charts
- `bubblelab_autorefresh` - Auto-refresh

**Migration Notes:**
- Metrics cards become React metric components
- Trend charts become Recharts line charts
- Activity table becomes React Table
- Auto-refresh becomes React Query refetch
- System health becomes status badge component
- Resource utilization becomes progress bars
- Dashboard sections become tabs or routes

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `analytics_dashboard.py`

**Purpose:** Detailed analytics UI with custom reports, filtering, and data export.

**BubbleLab UI Components Used:**
- `st.selectbox()`, `st.multiselect()` - Filter controls
- `st.date_input()` - Date range selection
- `st.button()` - Apply filters, export data
- `st.plotly_chart()` - Custom analytics charts
- `st.dataframe()` - Filtered results table
- `st.download_button()` - Data export
- `st.session_state` - Analytics state

**Backend Functions Called:**
- `AnalyticsManager.get_custom_report()` - Custom analytics queries
- `AnalyticsManager.export_data()` - Data export
- `AnalyticsManager.get_aggregated_metrics()` - Aggregated data

**Session State Usage:**
- Filter state: `selected_filters`, `date_range`
- Query results: `analytics_data`, `report_data`
- Export state: `export_format`, `export_ready`

**Real-time Updates:**
- Re-renders on filter change
- No real-time streaming (on-demand queries)

**External Dependencies:**
- `analytics_manager.py` - Analytics backend
- `plotly.express` - Visualizations
- `pandas` - Data manipulation

**Migration Notes:**
- Filters become React form controls
- Date range becomes date picker component
- Charts become Recharts with dynamic data
- Export becomes file download API
- Results table becomes React Table with sorting/filtering

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `monitoring_dashboard.py`

**Purpose:** Real-time system monitoring with logs, resource usage, and service health.

**BubbleLab UI Components Used:**
- `st.status()` - Service health indicators
- `st.progress()` - CPU, memory, disk usage
- `st.log_viewer` - Custom log viewer (if available)
- `st.text_area()` - Log display with syntax highlighting
- `st.tabs()` - Log sources (application, backend, services)
- `st.selectbox()` - Log level filtering
- `st.button()` - Refresh, clear logs, download logs
- `st.session_state` - Monitoring state

**Backend Functions Called:**
- `LogStreaming.get_logs()` - Fetch application logs
- `MonitoringSystem.get_resource_usage()` - System metrics
- `MonitoringSystem.get_service_status()` - Service health
- `LogStreaming.download_logs()` - Log export

**Session State Usage:**
- Log config: `selected_log_level`, `auto_scroll_logs`
- Monitoring data: `resource_usage`, `service_status`
- Log cache: `cached_logs`, `log_position`

**Real-time Updates:**
- Auto-refreshing logs (configurable interval)
- Live resource usage updates
- Service health status polling

**External Dependencies:**
- `log_streaming.py` - Log streaming backend
- `monitoring_system.py` - System monitoring
- `bubblelab_autorefresh` - Auto-refresh

**Migration Notes:**
- Service health becomes status badge component
- Resource usage becomes progress bar component
- Log viewer becomes virtualized list component
- Auto-refresh becomes WebSocket log streaming
- Log filtering becomes React controls
- Log download becomes file download API

**Complexity Rating:** ⭐⭐⭐⭐ (4/5) - HIGH COMPLEXITY

---

## SECTION 4: COLLABORATION & VERSION CONTROL UI

### 4.1 Real-time Collaboration

---

## File: `collaboration_manager.py`

**Purpose:** Real-time collaboration UI for multi-user editing, commenting, and project sharing.

**BubbleLab UI Components Used:**
- `st.chat()` - Real-time chat (if available)
- `st.text_area()` - Collaborative editing
- `st.text_input()` - Comment input
- `st.button()` - Post comment, share project
- `st.columns()` - User list and activity feed
- `st.expander()` - Comment threads
- `st.session_state` - Collaboration state

**Backend Functions Called:**
- `CollaborationManager.connect()` - WebSocket connection
- `CollaborationManager.send_update()` - Broadcast edits
- `CollaborationManager.post_comment()` - Comment posting
- `CollaborationManager.share_project()` - Project sharing
- `CollaborationManager.get_active_users()` - User list

**Session State Usage:**
- Connection: `collaboration_connected`, `room_id`
- User data: `current_user`, `active_users`
- Editing: `shared_content`, `edits_pending`
- Comments: `comment_threads`, `unread_comments`

**Real-time Updates:**
- Real-time edit broadcasting (WebSocket)
- Live cursor positions
- Instant comment notifications
- Active user list updates

**External Dependencies:**
- `websocket` - Real-time communication
- `collaboration_server.py` - Collaboration backend

**Migration Notes:**
- Chat becomes React chat component
- Collaborative editing needs Y.js or OT (Operational Transform)
- Comments become threaded comment component
- User list becomes avatar list component
- Real-time updates need WebSocket
- Project sharing needs secure link generation

**Complexity Rating:** ⭐⭐⭐⭐⭐ (5/5) - VERY HIGH COMPLEXITY

---

## File: `version_control.py`

**Purpose:** Version control UI for content history, branching, and tagging.

**BubbleLab UI Components Used:**
- `st.selectbox()` - Version/branch selection
- `st.button()` - Create branch, create tag, revert
- `st.diff_viewer()` - Version diff display (custom component)
- `st.text_area()` - Commit message input
- `st.dataframe()` - Version history table
- `st.expander()` - Version details
- `st.session_state` - Version control state

**Backend Functions Called:**
- `VersionControl.get_history()` - Version history
- `VersionControl.create_branch()` - Branch creation
- `VersionControl.create_tag()` - Tag creation
- `VersionControl.revert()` - Version revert
- `VersionControl.get_diff()` - Version diff

**Session State Usage:**
- Selection: `selected_version`, `selected_branch`
- History: `version_history`, `branch_list`
- Diff: `diff_data`, `comparing_versions`

**Real-time Updates:**
- Re-renders on version/branch change
- No continuous updates

**External Dependencies:**
- `version_control.py` - Version control backend
- `difflib` - Diff generation
- Custom diff viewer component

**Migration Notes:**
- Version history becomes React Table
- Diff viewer becomes React diff component (react-diff-viewer)
- Branch/tag controls become form components
- Revert needs confirmation dialog
- Commit message becomes text input
- Version comparison needs side-by-side view

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## SECTION 5: LEANAIDE UI COMPONENTS

### 5.1 LeanAide Server UI

---

## File: `LeanAide/server/bubblelabs_ui.py`

**Purpose:** Main LeanAide application UI for Lean 4 proof generation and verification.

**BubbleLab UI Components Used:**
- `st.Page()` - Multi-page application navigation
- `st.navigation()` - Page routing
- `st.sidebar` - LLM credentials and configuration
- `st.selectbox()` - Provider and model selection
- `st.text_input()` - API key input (password type)
- `st.slider()` - Temperature configuration
- `st.button()` - Refresh page
- `st.expander()` - Settings sections
- `st.session_state` - Session state management

**Backend Functions Called:**
- `get_supported_models()` - Fetch available models
- `provider_info` - Provider configuration
- `get_git_commit_info()` - Build information
- API server endpoints (HOST, PORT)

**Session State Usage:**
- LLM config: `llm_provider`, `llm_list`, `llm_api_key`
- Models: `model_leanaide`, `model_text`, `model_img`
- Settings: `temperature`, `api_host`, `api_port`
- UI state: Various initialization keys (NONE_INIT_KEYS, FALSE_INIT_KEYS, LLM_INIT_KEYS)

**Real-time Updates:**
- Re-render on provider/model change
- Page refresh button

**External Dependencies:**
- `api_server.py` - LeanAide backend API
- `llm_response.py` - LLM model management
- `logging_utils.py` - Logging utilities
- `subprocess` - Git information

**Migration Notes:**
- Multi-page app becomes React Router
- Sidebar becomes React sidebar
- Credentials need secure storage (backend encryption)
- Model selection becomes dropdown components
- API key management needs secure input field
- Page refresh becomes React router navigation
- Build info becomes footer component

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `LeanAide/server/tabs/home.py`

**Purpose:** Home page tab for LeanAide with main proof generation interface.

**BubbleLab UI Components Used:**
- `st.text_area()` - Theorem and proof input
- `st.button()` - Generate proof, verify proof
- `st.columns()` - Input/output layout
- `st.code()` - Generated Lean 4 code display
- `st.status()` - Generation/verification status
- `st.progress()` - Generation progress
- `st.session_state` - Home tab state

**Backend Functions Called:**
- API endpoints for proof generation
- API endpoints for proof verification
- Lean 4 compiler integration

**Session State Usage:**
- Input: `theorem`, `proof_attempt`
- Output: `generated_proof`, `verification_result`
- State: `generating`, `verifying`, `result_ready`

**Real-time Updates:**
- Generation progress updates
- Verification status updates

**Migration Notes:**
- Input/output becomes split-panel layout
- Code display needs syntax highlighting (Prism.js)
- Status indicators become badge components
- Progress becomes progress bar
- Generation/verification become async actions

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `LeanAide/server/tabs/benchmark.py`

**Purpose:** Benchmarking tab for evaluating Lean 4 proof generation performance.

**BubbleLab UI Components Used:**
- `st.file_uploader()` - Benchmark dataset upload
- `st.selectbox()` - Benchmark selection
- `st.button()` - Run benchmark
- `st.progress()` - Benchmark progress
- `st.dataframe()` - Results table
- `st.metrics()` - Performance metrics
- `st.session_state` - Benchmark state

**Backend Functions Called:**
- Benchmark execution API
- Performance evaluation
- Results aggregation

**Session State Usage:**
- Config: `bm_input_opt`, `bm_evaluator`
- Data: `bm_json_dataset`, `bm_single_thm`
- Results: `bm_results`, `bm_display_table`
- State: `bm_started`, `bm_result_success`

**Real-time Updates:**
- Live benchmark progress
- Real-time results table updates

**Migration Notes:**
- File upload becomes React file uploader
- Benchmark config becomes form controls
- Progress becomes progress bar
- Results table becomes React Table
- Metrics become metric cards
- Real-time updates need WebSocket or polling

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `LeanAide/server/tabs/structured_json.py`

**Purpose:** Structured JSON output tab for proof generation results.

**BubbleLab UI Components Used:**
- `st.json()` - JSON display
- `st.code()` - JSON code block
- `st.download_button()` - Download JSON
- `st.text_area()` - JSON input
- `st.button()` - Format/validate JSON
- `st.session_state` - JSON state

**Backend Functions Called:**
- JSON formatting/validation
- JSON export

**Session State Usage:**
- Data: `structured_proof`, `temp_structured_json`
- State: `json_valid`, `json_formatted`

**Real-time Updates:**
- Re-render on JSON change

**Migration Notes:**
- JSON display becomes syntax-highlighted code block
- Download becomes file download API
- Validation becomes client-side validation
- Formatting becomes React button action

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## File: `LeanAide/server/tabs/server_response.py`

**Purpose:** Server response viewer for debugging API interactions.

**BubbleLab UI Components Used:**
- `st.code()` - API response display
- `st.text_area()` - Request/response body
- `st.json()` - JSON response
- `st.selectbox()` - Request history selection
- `st.session_state` - Response state

**Backend Functions Called:**
- API request logging
- Response history retrieval

**Session State Usage:**
- History: `request_history`, `response_cache`
- Display: `selected_request`, `response_body`

**Real-time Updates:**
- Re-render on request selection

**Migration Notes:**
- Response display becomes code block
- History becomes list component
- Request selection becomes dropdown
- JSON formatting becomes syntax highlighting

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## File: `LeanAide/server/tabs/logs_display.py`

**Purpose:** Log viewer tab for application logs.

**BubbleLab UI Components Used:**
- `st.text_area()` - Log display (read-only)
- `st.selectbox()` - Log level filtering
- `st.button()` - Refresh, clear, download logs
- `st.session_state` - Logs state

**Backend Functions Called:**
- Log retrieval API
- Log export

**Session State Usage:**
- Logs: `log_cache`, `log_position`
- Filter: `log_level_filter`

**Real-time Updates:**
- Auto-refresh (configurable)

**Migration Notes:**
- Log display becomes virtualized list
- Filtering becomes React controls
- Auto-refresh becomes polling or WebSocket
- Download becomes file download API

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## File: `LeanAide/server/tabs/token_response.py`

**Purpose:** Token response viewer for LLM token streaming.

**BubbleLab UI Components Used:**
- `st.text_area()` - Token stream display
- `st.code()` - Token code display
- `st.button()` - Start/stop streaming
- `st.session_state` - Token state

**Backend Functions Called:**
- Token streaming API
- Token counting

**Session State Usage:**
- Stream: `token_stream`, `token_cache`
- Control: `streaming_active`

**Real-time Updates:**
- Live token streaming

**Migration Notes:**
- Token stream becomes streaming text component
- Start/stop becomes toggle button
- Real-time updates need WebSocket or SSE
- Token counting becomes display metric

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## SECTION 6: ONEKE UI COMPONENTS

### 6.1 OneKE Frontend

---

## File: `OneKE/frontend/app.py`

**Purpose:** OneKE knowledge extraction application UI for schema-guided extraction.

**BubbleLab UI Components Used:**
- `st.set_page_config()` - Page configuration
- `st.markdown()` - Rich text/HTML content
- `st.sidebar` - Configuration sidebar
- `st.text_area()` - Text input for extraction
- `st.selectbox()` - Model and schema selection
- `st.button()` - Run extraction
- `st.columns()` - Input/results layout
- `st.json()` - Extraction results (JSON format)
- `st.code()` - Cypher query display (for Neo4j)
- `st.plotly_chart()` - Knowledge graph visualization
- `st.dataframe()` - Entity/relation tables
- `st.file_uploader()` - File upload (PDF, DOCX)
- `st.session_state` - Application state

**Backend Functions Called:**
- `Pipeline.extract_knowledge()` - Knowledge extraction
- `Pipeline.construct_graph()` - Knowledge graph construction
- `generate_cypher_statements()` - Neo4j Cypher query generation
- `execute_cypher_statements()` - Neo4j query execution
- `get_model_category()` - Model selection
- `start_with_example()` - Example loading

**Session State Usage:**
- Extraction: `input_text`, `extraction_results`, `kg_data`
- Configuration: `selected_model`, `selected_schema`, `proxy_config`
- Graph: `neo4j_connected`, `graph_data`
- State: `extraction_running`, `results_ready`

**Real-time Updates:**
- Extraction progress (no streaming)
- Graph visualization updates

**External Dependencies:**
- `OneKE/models` - OneKE model implementations
- `OneKE/pipeline` - Extraction pipeline
- `neo4j` - Graph database driver
- `pyvis.network` - Network visualization
- `networkx` - Graph manipulation
- `BubbleLab UI.components.v1` - Custom components

**Migration Notes:**
- Knowledge extraction becomes React form + API call
- Graph visualization becomes React graph component (vis-network, cytoscape.js)
- Entity/relation tables become React Table
- File upload becomes React file uploader
- JSON display becomes syntax-highlighted code block
- Schema selection becomes dropdown or tree view
- Proxy configuration becomes settings form
- Neo4j connection becomes backend service
- Example loading becomes preset selector

**Complexity Rating:** ⭐⭐⭐⭐ (4/5) - HIGH COMPLEXITY

---

## File: `OneKE/frontend/components/sidebar.py`

**Purpose:** OneKE sidebar for configuration and settings.

**BubbleLab UI Components Used:**
- `st.sidebar` - All sidebar components
- `st.selectbox()` - Model and provider selection
- `st.text_input()` - API keys and endpoints
- `st.checkbox()` - Feature toggles
- `st.slider()` - Parameter tuning
- `st.expander()` - Collapsible sections
- `st.session_state` - Configuration state

**Backend Functions Called:**
- Model configuration
- Provider setup
- Proxy configuration

**Session State Usage:**
- Model: `selected_model`, `model_config`
- API: `api_keys`, `api_endpoints`
- Features: `feature_flags`
- Proxy: `proxy_enabled`, `proxy_host`, `proxy_port`

**Real-time Updates:**
- Re-render on configuration change

**Migration Notes:**
- Sidebar becomes React sidebar
- Configuration becomes form controls
- API keys need secure storage
- Proxy settings become network config form
- Feature toggles become switch components

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `OneKE/frontend/components/results.py`

**Purpose:** Results display component for extracted knowledge.

**BubbleLab UI Components Used:**
- `st.tabs()` - Results views (JSON, Graph, Tables)
- `st.json()` - JSON results
- `st.plotly_chart()` - Graph visualization
- `st.dataframe()` - Entity/relation tables
- `st.button()` - Export results
- `st.download_button()` - File download
- `st.session_state` - Results state

**Backend Functions Called:**
- Results formatting
- Graph generation
- Data export

**Session State Usage:**
- Results: `extraction_results`, `kg_data`
- View: `active_tab`, `export_format`

**Real-time Updates:**
- Re-render on results update

**Migration Notes:**
- Tabs become React tabs
- JSON display becomes code block
- Graph visualization becomes network component
- Tables become React Table
- Export becomes file download API

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `OneKE/frontend/components/proxy_config.py`

**Purpose:** Proxy configuration component for network settings.

**BubbleLab UI Components Used:**
- `st.checkbox()` - Enable/disable proxy
- `st.text_input()` - Proxy host and port
- `st.button()` - Apply proxy settings
- `st.session_state` - Proxy state

**Backend Functions Called:**
- `set_proxy_config()` - Apply proxy configuration
- Environment variable setting

**Session State Usage:**
- Proxy: `proxy_enabled`, `proxy_host`, `proxy_port`

**Real-time Updates:**
- Re-render on configuration change

**Migration Notes:**
- Checkbox becomes switch component
- Host/port inputs become form fields
- Apply becomes form submit button
- Proxy settings stored in backend or browser storage

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## SECTION 7: ADDITIONAL OPENEVOLVE UI FILES

### 7.1 Specialized UI Components

---

## File: `bubblelabs_ui_component.py`

**Purpose:** BubbleLab workflow integration UI for visual workflow creation.

**BubbleLab UI Components Used:**
- `st.tabs()` - Workflow sections
- `st.form()` - Workflow configuration
- `st.text_area()` - Node configuration
- `st.selectbox()` - Node type selection
- `st.button()` - Add node, run workflow
- `st.json()` - Workflow definition
- `st.session_state` - Workflow state

**Backend Functions Called:**
- BubbleLab API integration
- Workflow execution
- Node management

**Session State Usage:**
- Workflow: `workflow_definition`, `workflow_nodes`
- Execution: `workflow_running`, `workflow_results`

**Real-time Updates:**
- Workflow execution progress
- Node status updates

**Migration Notes:**
- Workflow UI becomes visual workflow editor (React Flow)
- Node configuration becomes forms
- Execution becomes async process with WebSocket updates
- Workflow definition becomes JSON editor

**Complexity Rating:** ⭐⭐⭐⭐ (4/5) - HIGH COMPLEXITY

---

## File: `n8n_workflow_integration.py`

**Purpose:** n8n workflow integration UI for visual automation.

**BubbleLab UI Components Used:**
- `st.iframe()` - n8n UI embedding
- `st.form()` - Workflow configuration
- `st.text_input()` - n8n instance URL
- `st.button()` - Connect to n8n
- `st.session_state` - Integration state

**Backend Functions Called:**
- n8n API integration
- Workflow synchronization
- Credential management

**Session State Usage:**
- Connection: `n8n_connected`, `n8n_url`
- Workflows: `n8n_workflows`, `selected_workflow`

**Real-time Updates:**
- Workflow synchronization
- Status polling

**Migration Notes:**
- n8n embedding becomes iframe in React
- Configuration becomes form
- API integration needs backend proxy
- Workflow sync becomes polling or webhook

**Complexity Rating:** ⭐⭐⭐ (3/5) - MEDIUM COMPLEXITY

---

## File: `demo_app.py`

**Purpose:** Demo application showcasing OpenEvolve features.

**BubbleLab UI Components Used:**
- `st.tabs()` - Feature demos
- `st.text_area()` - Demo input
- `st.button()` - Run demo
- `st.code()` - Demo output
- `st.session_state` - Demo state

**Backend Functions Called:**
- Feature demonstrations
- Sample data generation

**Session State Usage:**
- Demo: `active_demo`, `demo_input`, `demo_output`

**Real-time Updates:**
- Demo execution progress

**Migration Notes:**
- Demo tabs become React routes
- Input/output becomes split-panel
- Execution becomes async action
- Code display needs syntax highlighting

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## File: `openevolve_bubblelabs_ui.py`

**Purpose:** OpenEvolve BubbleLab integration UI.

**BubbleLab UI Components Used:**
- `st.columns()` - Layout
- `st.form()` - Integration configuration
- `st.selectbox()` - Integration type
- `st.button()` - Enable/disable integration
- `st.session_state` - Integration state

**Backend Functions Called:**
- BubbleLab integration API
- Configuration management

**Session State Usage:**
- Integration: `bubblelab_enabled`, `integration_config`

**Real-time Updates:**
- Integration status updates

**Migration Notes:**
- Integration config becomes settings form
- Status becomes badge component
- Enable/disable becomes toggle

**Complexity Rating:** ⭐⭐ (2/5) - LOW COMPLEXITY

---

## SECTION 8: TEST AND DEMO FILES

### 8.1 Testing UI Files

---

**Test Files (Lower Priority for Migration):**

1. `tests/test_integration.py` - Integration test UI
2. `tests/test_enhanced_adversarial.py` - Adversarial testing UI
3. `tests/test_sovereign_workflow.py` - Sovereign workflow test UI
4. `tests/test_integrated_functionality.py` - Integrated functionality test UI
5. `demo_ui_integration.py` - UI integration demo
6. `demo_evolution_maker.py` - Evolution maker demo
7. `demo_adversarial_maker.py` - Adversarial maker demo
8. `demo_mdap_maker.py` - MDAP maker demo
9. `demo_hybrid_maker.py` - Hybrid maker demo

**Note:** These test/demo files are lower priority for migration as they're used for development and testing, not production use.

---

## SECTION 9: BubbleLab UI COMPONENT USAGE SUMMARY

### 9.1 Most Commonly Used Components

**Input Components:**
- `st.text_input()` - Text input (150+ occurrences)
- `st.text_area()` - Multi-line text input (120+ occurrences)
- `st.number_input()` - Numeric input (80+ occurrences)
- `st.selectbox()` - Dropdown selection (100+ occurrences)
- `st.multiselect()` - Multi-select (40+ occurrences)
- `st.slider()` - Slider controls (70+ occurrences)
- `st.checkbox()` - Checkbox (50+ occurrences)
- `st.file_uploader()` - File upload (20+ occurrences)

**Layout Components:**
- `st.columns()` - Multi-column layouts (200+ occurrences)
- `st.tabs()` - Tab navigation (60+ occurrences)
- `st.expander()` - Collapsible sections (80+ occurrences)
- `st.form()` - Form containers (90+ occurrences)
- `st.sidebar` - Sidebar (100% of apps)
- `st.container()` - Container (40+ occurrences)

**Display Components:**
- `st.markdown()` - Rich text/HTML (300+ occurrences)
- `st.code()` - Code display (100+ occurrences)
- `st.json()` - JSON display (60+ occurrences)
- `st.dataframe()` - Data tables (80+ occurrences)
- `st.metric()` - Metric cards (40+ occurrences)
- `st.plotly_chart()` - Plotly charts (30+ occurrences)

**Action Components:**
- `st.button()` - Buttons (400+ occurrences)
- `st.download_button()` - File download (20+ occurrences)
- `st.form_submit_button()` - Form submit (90+ occurrences)

**Status Components:**
- `st.progress()` - Progress bars (50+ occurrences)
- `st.spinner()` - Loading indicators (30+ occurrences)
- `st.status()` - Status indicators (20+ occurrences)
- `st.info()`, `st.warning()`, `st.error()`, `st.success()` - Alerts (200+ occurrences)

**Advanced Components:**
- `st.session_state` - State management (100% of apps)
- `st.empty()` - Placeholders (60+ occurrences)
- `st_autorefresh` - Auto-refresh (10+ occurrences)
- `st_tags` - Tag input (5+ occurrences)

---

## SECTION 10: MIGRATION COMPLEXITY ASSESSMENT

### 10.1 Complexity Breakdown

**Very High Complexity (5/5):**
- main.py - Backend service orchestration
- mainlayout.py - Core UI with multiple engines
- adversarial.py - Red/blue team workflow
- integrated_workflow.py - Multi-stage pipeline
- collaboration_manager.py - Real-time collaboration

**High Complexity (4/5):**
- evolution.py - Genetic algorithm UI
- monitoring_dashboard.py - Real-time monitoring
- OneKE/app.py - Knowledge extraction UI
- bubblelabs_ui_component.py - Visual workflow editor
- n8n_workflow_integration.py - External service integration

**Medium Complexity (3/5):**
- sidebar.py - Configuration UI
- ui_components.py - Reusable components
- openevolve_dashboard.py - Main dashboard
- analytics_dashboard.py - Analytics UI
- LeanAide/bubblelabs_ui.py - Multi-page app
- LeanAide/tabs/*.py - LeanAide tabs
- OneKE/components/*.py - OneKE components

**Low Complexity (2/5):**
- LeanAide/tabs/structured_json.py
- LeanAide/tabs/server_response.py
- LeanAide/tabs/logs_display.py
- OneKE/components/proxy_config.py
- demo_app.py
- Test files

---

### 10.2 Migration Effort Estimate

**Total Effort:** ~800-1200 hours (assuming experienced React/TypeScript developer)

**Breakdown:**
- Core UI (main.py, mainlayout.py, sidebar.py): 200-300 hours
- Evolution & Adversarial: 150-200 hours
- Analytics & Dashboards: 100-150 hours
- LeanAide UI: 80-120 hours
- OneKE UI: 100-150 hours
- Integration Components: 80-120 hours
- Testing & Refinement: 100-150 hours

---

## SECTION 11: CRITICAL MIGRATION DEPENDENCIES

### 11.1 Backend API Requirements

**Must be implemented BEFORE UI migration:**

1. **WebSocket Support:**
   - Real-time evolution updates
   - Live adversarial testing
   - Log streaming
   - Collaboration updates

2. **REST API Endpoints:**
   - Evolution execution
   - Adversarial testing
   - Analytics data
   - Configuration management
   - File upload/download
   - Version control
   - Collaboration

3. **State Management:**
   - Server-side session state
   - Persistence layer
   - Caching strategy

4. **Authentication:**
   - User authentication
   - API key management
   - Session tokens

---

### 11.2 Real-time Update Requirements

**WebSocket Scenarios:**
- Evolution generation progress (10-100 updates per run)
- Adversarial testing phases (5-20 updates per test)
- Log streaming (continuous)
- Collaboration edits (continuous)
- System monitoring (1-5 second intervals)

**Polling Scenarios:**
- Dashboard metrics (5-10 second intervals)
- Analytics updates (30-60 second intervals)
- Service health checks (10-30 second intervals)

---

## SECTION 12: NEXT STEPS

### 12.1 Immediate Actions (Agent 2 - Backend API)

1. Design and implement WebSocket infrastructure
2. Create REST API endpoints for all backend engines
3. Implement server-side state management
4. Add authentication and authorization
5. Create API documentation

### 12.2 UI Migration Strategy (Agent 3+)

1. **Phase 1:** Core UI framework (main.py, sidebar)
2. **Phase 2:** Layout and components (mainlayout.py, ui_components.py)
3. **Phase 3:** Evolution engine UI
4. **Phase 4:** Adversarial testing UI
5. **Phase 5:** Analytics and dashboards
6. **Phase 6:** LeanAide UI
7. **Phase 7:** OneKE UI
8. **Phase 8:** Integration components
9. **Phase 9:** Testing and refinement

---

## APPENDIX A: FILE STRUCTURE

```
Frontend/
├── main.py                          [CRITICAL] - Main entry point
├── mainlayout.py                    [CRITICAL] - Core layout
├── sidebar.py                       [CRITICAL] - Sidebar config
├── ui_components.py                 [HIGH] - Reusable components
├── evolution.py                     [HIGH] - Evolution UI
├── adversarial.py                   [HIGH] - Adversarial UI
├── integrated_workflow.py           [HIGH] - Workflow UI
├── openevolve_dashboard.py          [MEDIUM] - Main dashboard
├── analytics_dashboard.py           [MEDIUM] - Analytics UI
├── monitoring_dashboard.py          [MEDIUM] - Monitoring UI
├── collaboration_manager.py         [HIGH] - Collaboration UI
├── version_control.py               [MEDIUM] - Version control UI
├── bubblelabs_ui_component.py       [HIGH] - BubbleLab integration
├── n8n_workflow_integration.py      [MEDIUM] - n8n integration
├── demo_app.py                      [LOW] - Demo app
├── demo_*.py                        [LOW] - Demo files
├── LeanAide/
│   └── server/
│       ├── bubblelabs_ui.py         [MEDIUM] - LeanAide main UI
│       └── tabs/
│           ├── home.py             [MEDIUM] - Home tab
│           ├── benchmark.py        [MEDIUM] - Benchmark tab
│           ├── structured_json.py  [LOW] - JSON tab
│           ├── server_response.py  [LOW] - Response tab
│           ├── logs_display.py     [LOW] - Logs tab
│           └── token_response.py   [MEDIUM] - Token tab
└── OneKE/
    └── frontend/
        ├── app.py                  [HIGH] - OneKE main UI
        └── components/
            ├── sidebar.py          [MEDIUM] - Sidebar
            ├── results.py          [MEDIUM] - Results
            └── proxy_config.py     [LOW] - Proxy config
```

---

## APPENDIX B: BubbleLab UI TO REACT COMPONENT MAPPING

**See COMPONENT_MAPPING_MATRIX.md for detailed mapping.**

---

**END OF INVENTORY**

**Last Updated:** 2025-01-05
**Status:** COMPLETE - Ready for Agent 2 (Backend API Design)


