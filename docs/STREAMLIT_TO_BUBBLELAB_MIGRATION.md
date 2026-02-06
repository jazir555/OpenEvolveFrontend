# 🚨 BubbleLab UI EXCISION & BUBBLELAB UI MIGRATION - MASTER TASK LIST

**Status**: 🟡 IN PROGRESS
**Priority**: CRITICAL
**Impact**: Complete UI architecture transformation
**Timeline**: Multi-phase migration

---

## 📋 EXECUTIVE SUMMARY

**Objective**: Complete removal of BubbleLab UI as the UI layer and migration to BubbleLab TypeScript/React UI.

**Scope**:
- **10+ major BubbleLab UI applications** identified
- **399 BubbleLab UI imports** across the codebase
- **0 Python backend rewrites** - all business logic preserved
- **100% TypeScript/React frontend** replacement

**Key Principle**: The Python backend engines remain untouched. Only the UI presentation layer is being replaced.

---

## 🎯 MIGRATION ARCHITECTURE

### Current State
```
User → BubbleLab UI (Python) → Backend Engines (Python)
```

### Target State
```
User → BubbleLab UI (TypeScript/React) → API Gateway → Backend Engines (Python)
```

### Architecture Patterns

1. **API First**: Every BubbleLab UI UI component maps to a REST/GraphQL API endpoint
2. **State Management**: React Query/Zustand replaces `st.session_state`
3. **Real-time Updates**: WebSockets replace `st.empty()` and `st.progress()`
4. **Form Validation**: Zod/react-hook-form replaces BubbleLab UI form handling
5. **Visualization**: Recharts/D3.js replace Plotly/BubbleLab UI charts

---

## 📊 PHASE 1: DISCOVERY & AUDIT (Week 1)

### 1.1 Complete BubbleLab UI Inventory
- [ ] **Document all BubbleLab UI files** (399 imports identified)
- [ ] **Map each BubbleLab UI component** to its backend logic
- [ ] **Identify state dependencies** (session_state usage patterns)
- [ ] **Catalog all custom components** (HTML/CSS/JavaScript injections)
- [ ] **Audit external dependencies** (Plotly, Chart.js, custom libraries)

**Deliverable**: `STREAMLIT_AUDIT_REPORT.md` with complete component inventory

### 1.2 Backend API Extraction
For each BubbleLab UI file:
- [ ] **Extract business logic** from UI code
- [ ] **Create API endpoint specifications**
- [ ] **Document data contracts** (request/response schemas)
- [ ] **Identify authentication/authorization** requirements
- [ ] **Map real-time update patterns** to WebSocket events

**Deliverable**: `API_CONTRACTS.md` with OpenAPI specifications

### 1.3 BubbleLab UI Capacity Assessment
- [ ] **Audit existing BubbleLab components** (90+ .tsx files identified)
- [ ] **Map BubbleLab UI widgets** to React equivalents
- [ ] **Identify gaps** in BubbleLab component library
- [ ] **Plan new component development**
- [ ] **Design integration points** for OpenEvolve engines

**Deliverable**: `COMPONENT_MAPPING.md` with migration matrix

---

## 📊 PHASE 2: API FOUNDATION (Weeks 2-4)

### 2.1 Create API Gateway Layer
- [ ] **Set up FastAPI/Flask gateway** (if not exists)
- [ ] **Implement authentication middleware** (JWT/OAuth2)
- [ ] **Create CORS configuration** for BubbleLab frontend
- [ ] **Set up request validation** (Pydantic schemas)
- [ ] **Implement rate limiting** and security headers

**Files to Create**:
```
api/
├── gateway/
│   ├── __init__.py
│   ├── main.py
│   ├── middleware/
│   │   ├── auth.py
│   │   ├── cors.py
│   │   └── rate_limit.py
│   └── routes/
│       ├── __init__.py
│       └── base.py
```

### 2.2 Extract Core Engine APIs

#### 2.2.1 Content Analysis Engine (`demo_app.py`)
- [ ] `POST /api/analyze/content` - Content analysis endpoint
- [ ] `POST /api/assess/quality` - Quality assessment endpoint
- [ ] `GET /api/analysis/{id}` - Retrieve analysis results
- [ ] `WebSocket /ws/analysis/{id}` - Real-time progress updates

**Backend Logic**: Extract from `demo_app.py:ContentAnalyzer`, `QualityAssessmentEngine`

#### 2.2.2 Workflow Engine (`mainlayout.py`)
- [ ] `POST /api/workflows/create` - Create workflow
- [ ] `GET /api/workflows` - List workflows
- [ ] `GET /api/workflows/{id}` - Get workflow details
- [ ] `PUT /api/workflows/{id}` - Update workflow
- [ ] `DELETE /api/workflows/{id}` - Delete workflow
- [ ] `POST /api/workflows/{id}/execute` - Execute workflow
- [ ] `WebSocket /ws/workflows/{id}/execute` - Execution progress

**Backend Logic**: Extract from `mainlayout.py:OpenEvolveAPI`, workflow engine

#### 2.2.3 Team & Gauntlet Management (`ui_components.py`)
- [ ] `POST /api/teams` - Create team
- [ ] `GET /api/teams` - List teams
- [ ] `GET /api/teams/{id}` - Team details
- [ ] `PUT /api/teams/{id}` - Update team
- [ ] `DELETE /api/teams/{id}` - Delete team
- [ ] `POST /api/gauntlets` - Create gauntlet
- [ ] `GET /api/gauntlets` - List gauntlets
- [ ] `POST /api/gauntlets/{id}/execute` - Execute gauntlet

**Backend Logic**: Extract from `ui_components.py:TeamManager`, `GauntletManager`

#### 2.2.4 Analytics & Monitoring (`analytics_dashboard.py`)
- [ ] `GET /api/analytics/metrics` - KPI metrics
- [ ] `GET /api/analytics/performance` - Performance data
- [ ] `GET /api/analytics/artifacts` - Artifact tracking
- [ ] `GET /api/analytics/knowledge` - Knowledge base stats

**Backend Logic**: Extract from `analytics_dashboard.py:KnowledgeManager`

#### 2.2.5 OpenEvolve BubbleLabs Integration (`openevolve_bubblelabs_ui.py`, `bubblelabs_ui_component.py`)
- [ ] `POST /api/workflows/bubble` - Create BubbleLabs workflow
- [ ] `GET /api/workflows/bubble/{id}` - Get BubbleLabs workflow
- [ ] `POST /api/parameters/sync` - Sync parameters
- [ ] `WebSocket /ws/workflow/visualize` - Workflow visualization updates

**Backend Logic**: Extract from `openevolve_bubblelabs_ui.py:OpenEvolveWorkflowManager`, `bubblelabs_ui_component.py`

#### 2.2.6 LeanAide Integration (`LeanAide/server/streamlit_ui.py`)
- [ ] `POST /api/leanaide/prove` - Submit proof task
- [ ] `GET /api/leanaide/models` - List available models
- [ ] `POST /api/leanaide/verify` - Verify proof
- [ ] `WebSocket /ws/leanaide/proof/{id}` - Proof generation progress

**Backend Logic**: Extract from `LeanAide/server/streamlit_ui.py`

#### 2.2.7 Decomposition Engine (`decomposition_dashboard.py`)
- [ ] `GET /api/decomposition/status/{id}` - Get decomposition status
- [ ] `WebSocket /ws/decomposition/progress/{id}` - Real-time progress
- [ ] `GET /api/decomposition/results/{id}` - Get decomposition results

**Backend Logic**: Extract from `decomposition_dashboard.py`

#### 2.2.8 Knowledge Base (`knowledge_base_ui.py`)
- [ ] `GET /api/knowledge/artifacts` - List artifacts
- [ ] `POST /api/knowledge/artifacts` - Create artifact
- [ ] `GET /api/knowledge/artifacts/{id}` - Get artifact
- [ ] `PUT /api/knowledge/artifacts/{id}` - Update artifact
- [ ] `DELETE /api/knowledge/artifacts/{id}` - Delete artifact
- [ ] `GET /api/knowledge/search` - Search knowledge base

**Backend Logic**: Extract from `knowledge_base_ui.py:KnowledgeManager`

### 2.3 Implement Real-time Communication
- [ ] **Set up WebSocket server** (Socket.IO/Starlette)
- [ ] **Create event broadcast system** for progress updates
- [ ] **Implement room-based subscriptions** (workflow ID, analysis ID)
- [ ] **Add authentication for WebSocket connections**
- [ ] **Create connection pool management**

**Files to Create**:
```
api/
└── realtime/
    ├── __init__.py
    ├── manager.py
    ├── events.py
    └── handlers/
        ├── workflow.py
        ├── analysis.py
        └── decomposition.py
```

---

## 📊 PHASE 3: BUBBLELAB UI DEVELOPMENT (Weeks 5-12)

### 3.1 Core UI Infrastructure

#### 3.1.1 State Management
- [ ] **Set up Zustand stores** for global state
- [ ] **Create React Query configuration** for server state
- [ ] **Implement authentication store** (user session, tokens)
- [ ] **Create workflow store** (active workflows, execution state)
- [ ] **Build analytics store** (metrics, performance data)

**Files to Create**:
```
BubbleLab/apps/bubble-studio/src/stores/
├── authStore.ts
├── workflowStore.ts
├── analyticsStore.ts
├── knowledgeStore.ts
└── index.ts
```

#### 3.1.2 API Client Layer
- [ ] **Create axios/fetch client** with interceptors
- [ ] **Implement request/response transformation**
- [ ] **Add automatic token refresh**
- [ ] **Create WebSocket client manager**
- [ ] **Build error handling utilities**

**Files to Create**:
```
BubbleLab/apps/bubble-studio/src/lib/
├── api/
│   ├── client.ts
│   ├── endpoints.ts
│   └── websocket.ts
└── hooks/
    ├── useApi.ts
    ├── useWebSocket.ts
    └── useRealtime.ts
```

### 3.2 Migrate Core Applications

#### 3.2.1 Demo App → BubbleLab Pages
**Source**: `demo_app.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/DemoPage.tsx`

**Components to Create**:
- [ ] `DemoForm.tsx` - Content input and configuration
- [ ] `AnalysisProgress.tsx` - Real-time analysis progress
- [ ] `QualityAssessment.tsx` - Quality metrics display
- [ ] `EvolutionControl.tsx` - Evolution engine controls
- [ ] `AdversarialTesting.tsx` - Adversarial testing interface

**BubbleLab UI → React Mapping**:
```
st.text_area() → <Textarea />
st.selectbox() → <Select />
st.button() → <Button />
st.progress() → <ProgressBar />
st.empty() → useWebSocket() + real-time updates
st.sidebar() → <Sidebar />
```

#### 3.2.2 Main Layout → BubbleLab Dashboard
**Source**: `mainlayout.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/OpenEvolveDashboard.tsx`

**Components to Create**:
- [ ] `WorkflowTabs.tsx` - Tabbed workflow interface
- [ ] `ConfigPanel.tsx` - Configuration sidebar
- [ ] `ExecutionMonitor.tsx` - Real-time execution monitoring
- [ ] `ProviderCatalog.tsx` - Provider selection interface
- [ ] `SessionManager.tsx` - Session management component
- [ ] `LogStreamer.tsx` - Log streaming display

**Advanced Features**:
- [ ] **Collapsible sections** → `<Accordion />` component
- [ ] **Form handling** → `react-hook-form` + Zod validation
- [ ] **Toast notifications** → Existing `<Toast />` system
- [ ] **Dynamic content** → React Query + WebSocket updates

#### 3.2.3 UI Components Library
**Source**: `ui_components.py`
**Target**: `BubbleLab/apps/bubble-studio/src/components/openevolve/`

**Component Mapping**:
```
TeamManager → TeamManagement.tsx
  ├─ TeamList.tsx
  ├─ TeamForm.tsx
  └─ TeamMembers.tsx

GauntletManager → GauntletManagement.tsx
  ├─ GauntletList.tsx
  ├─ GauntletConfig.tsx
  └─ GauntletExecution.tsx

WorkflowStructures → WorkflowStructures.tsx
  ├─ StructureList.tsx
  ├─ StructureEditor.tsx
  └─ StructureValidator.tsx
```

#### 3.2.4 Analytics Dashboard
**Source**: `analytics_dashboard.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/AnalyticsDashboard.tsx`

**Visualizations**:
- [ ] **KPI Metrics** → Recharts `<MetricCard />`
- [ ] **Performance Charts** → `<LineChart />`, `<BarChart />`
- [ ] **Artifact Tables** → Existing `<DataTable />`
- [ ] **Knowledge Base Stats** → `<StatGrid />`

**Plotly Migration**:
```
st.plotly_chart() → <ResponsiveContainer />
  ├─ <LineChart />
  ├─ <AreaChart />
  ├─ <BarChart />
  └─ <PieChart />
```

#### 3.2.5 BubbleLabs Workflow UI
**Source**: `openevolve_bubblelabs_ui.py`, `bubblelabs_ui_component.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/WorkflowBuilder.tsx`

**Components**:
- [ ] `WorkflowCreator.tsx` - Workflow creation form
- [ ] `WorkflowVisualizer.tsx` - Flow visualization (reuse existing `FlowVisualizer.tsx`)
- [ ] `ParameterSync.tsx` - Parameter synchronization
- [ ] `ExecutionControl.tsx` - Execution controls

**Integration Points**:
- [ ] **Reuse existing `FlowVisualizer.tsx`** component
- [ ] **Integrate with `BubbleSidePanel.tsx`**
- [ ] **Connect to `ExecutionHistory.tsx`** for logs

#### 3.2.6 LeanAide Interface
**Source**: `LeanAide/server/streamlit_ui.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/LeanAidePage.tsx`

**Components**:
- [ ] `LeanAideForm.tsx` - Proof task input
- [ ] `ModelSelector.tsx` - Model selection dropdown
- [ ] `ProofEditor.tsx` - Lean code editor (reuse `MonacoEditor.tsx`)
- [ ] `VerificationDisplay.tsx` - Verification results
- [ ] `ProgressTracker.tsx` - Proof generation progress

#### 3.2.7 Knowledge Base UI
**Source**: `knowledge_base_ui.py`
**Target**: `BubbleLab/apps/bubble-studio/src/pages/KnowledgeBasePage.tsx`

**Components**:
- [ ] `ArtifactList.tsx` - Artifact listing
- [ ] `ArtifactDetail.tsx` - Artifact detail view
- [ ] `KnowledgeSearch.tsx` - Search interface
- [ ] `ArtifactEditor.tsx` - Create/edit artifacts
- [ ] `KnowledgeGraph.tsx` - Graph visualization (reuse `KnowledgeGraphViewer.tsx`)

### 3.3 Advanced Features

#### 3.3.1 Real-time Progress Tracking
- [ ] **Create `<ProgressBar />`** component
- [ ] **Implement `<LiveLogViewer />`** for streaming logs
- [ ] **Build `<StepProgress />`** for multi-step processes
- [ ] **Add `<ExecutionStatus />`** badges

**WebSocket Integration**:
```typescript
// useExecutionProgress.ts
export function useExecutionProgress(executionId: string) {
  const [progress, setProgress] = useState<ProgressData>({});
  const socket = useWebSocket();

  useEffect(() => {
    socket.subscribe(`execution:${executionId}`, (data) => {
      setProgress(data);
    });
  }, [executionId]);

  return progress;
}
```

#### 3.3.2 Form Validation & Handling
- [ ] **Create Zod schemas** for all forms
- [ ] **Build reusable `<FormWrapper />`** component
- [ ] **Implement async validation** for API calls
- [ ] **Add field-level error display**

**Example**:
```typescript
// schemas/workflow.ts
export const workflowSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  config: z.object({
    provider: z.string(),
    model: z.string(),
    temperature: z.number().min(0).max(2),
  }),
});

// components/WorkflowForm.tsx
export function WorkflowForm() {
  const form = useForm({
    schema: workflowSchema,
  });

  return <FormWrapper form={form} />;
}
```

#### 3.3.3 Data Visualization
- [ ] **Create chart components** library
- [ ] **Build `<DataTable />** with sorting/filtering
- [ ] **Implement `<StatCard />** for KPIs
- [ ] **Add `<ChartContainer />** for responsive charts**

**Recharts Integration**:
```typescript
// components/charts/PerformanceChart.tsx
export function PerformanceChart({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <XAxis dataKey="timestamp" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

---

## 📊 PHASE 4: INTEGRATION & TESTING (Weeks 13-16)

### 4.1 End-to-End Integration
- [ ] **Wire up all API endpoints** to UI components
- [ ] **Implement authentication flow** (login/logout/token refresh)
- [ ] **Connect WebSocket streams** to UI updates
- [ ] **Set up error boundaries** and fallback UIs
- [ ] **Implement loading states** for all async operations

### 4.2 Testing Suite

#### 4.2.1 Unit Tests
- [ ] **Test all API client functions**
- [ ] **Test React hooks** (useApi, useWebSocket)
- [ ] **Test Zustand stores** (state management)
- [ ] **Test individual components** (rendering, interactions)

**Target**: 80%+ code coverage

#### 4.2.2 Integration Tests
- [ ] **Test API endpoints** with real backend
- [ ] **Test WebSocket connections** and event handling
- [ ] **Test authentication flow** end-to-end
- [ ] **Test real-time updates** (progress, logs)

**Tools**: Playwright, Supertest

#### 4.2.3 E2E Tests
- [ ] **Test user workflows** (create workflow → execute → monitor)
- [ ] **Test analytics dashboards** (load data, visualize)
- [ ] **Test knowledge base** (create, search, edit artifacts)
- [ ] **Test LeanAide integration** (submit proof, verify)

**Tools**: Playwright, Cypress

### 4.3 Performance Optimization
- [ ] **Implement React.memo()** for expensive components
- [ ] **Add virtual scrolling** for large lists
- [ ] **Optimize WebSocket connections** (connection pooling)
- [ ] **Implement API response caching** (React Query)
- [ ] **Add code splitting** for lazy loading

### 4.4 Security & Hardening
- [ ] **Implement CSRF protection**
- [ ] **Add XSS sanitization** for user inputs
- [ ] **Secure WebSocket connections** (WSS)
- [ ] **Implement rate limiting** on API endpoints
- [ ] **Add audit logging** for sensitive operations

---

## 📊 PHASE 5: DEPLOYMENT & CUTOVER (Weeks 17-18)

### 5.1 Staging Deployment
- [ ] **Deploy to staging environment**
- [ ] **Run smoke tests** against staging
- [ ] **Load test** the API gateway
- [ ] **Test WebSocket scalability**
- [ ] **Validate all integrations**

### 5.2 Data Migration
- [ ] **Migrate user sessions** (if any stored state)
- [ ] **Migrate workflow definitions**
- [ ] **Migrate knowledge base artifacts**
- [ ] **Validate migrated data** integrity
- [ ] **Create rollback plan**

### 5.3 Production Cutover
- [ ] **Deploy API gateway** to production
- [ ] **Deploy BubbleLab UI** to production
- [ ] **Switch DNS** to new UI
- [ ] **Monitor health** and performance
- [ ] **Keep BubbleLab UI available** as fallback (1 week)

### 5.4 Post-Deployment
- [ ] **Monitor error rates** and user feedback
- [ ] **Optimize slow queries** and API calls
- [ ] **Fix critical bugs** identified in production
- [ ] **Update documentation**
- [ ] **Conduct retrospective** and lessons learned

---

## 📊 PHASE 6: CLEANUP & DECOMMISSION (Weeks 19-20)

### 6.1 BubbleLab UI Removal
- [ ] **Archive BubbleLab UI code** (move to `deprecated/` directory)
- [ ] **Remove BubbleLab UI dependencies** from requirements.txt
- [ ] **Delete BubbleLab UI configuration files**
- [ ] **Update all documentation** references
- [ ] **Communicate deprecation** to users

### 6.2 Code Cleanup
- [ ] **Remove unused Python imports**
- [ ] **Delete deprecated utility functions**
- [ ] **Consolidate API endpoints** (remove duplicates)
- [ ] **Clean up WebSocket handlers**
- [ ] **Optimize database queries**

### 6.3 Documentation
- [ ] **Update API documentation** (OpenAPI/Swagger)
- [ ] **Create UI component documentation** (Storybook?)
- [ ] **Write migration guide** for contributors
- [ ] **Document architecture decisions** (ADRs)
- [ ] **Create runbooks** for operations

---

## 📁 FILE STRUCTURE

### New API Gateway Structure
```
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\
├── api/                          # NEW: API Gateway
│   ├── gateway/
│   │   ├── main.py              # FastAPI/Flask app
│   │   ├── middleware/
│   │   │   ├── auth.py
│   │   │   ├── cors.py
│   │   │   └── rate_limit.py
│   │   ├── routes/
│   │   │   ├── workflows.py
│   │   │   ├── analytics.py
│   │   │   ├── knowledge.py
│   │   │   ├── leanaide.py
│   │   │   └── decomposition.py
│   │   ├── realtime/
│   │   │   ├── manager.py
│   │   │   └── handlers/
│   │   └── models/
│   │       └── schemas.py
│   └── tests/
│       ├── test_api.py
│       └── test_websocket.py
│
├── BubbleLab/
│   └── apps/
│       └── bubble-studio/
│           └── src/
│               ├── pages/
│               │   ├── DemoPage.tsx                    # NEW
│               │   ├── OpenEvolveDashboard.tsx         # NEW
│               │   ├── AnalyticsDashboard.tsx          # NEW
│               │   ├── WorkflowBuilder.tsx             # NEW
│               │   ├── LeanAidePage.tsx                # NEW
│               │   └── KnowledgeBasePage.tsx           # NEW
│               │
│               ├── components/
│               │   └── openevolve/                     # NEW: OpenEvolve-specific
│               │       ├── workflow/
│               │       │   ├── WorkflowTabs.tsx
│               │       │   ├── ExecutionMonitor.tsx
│               │       │   └── ConfigPanel.tsx
│               │       ├── analytics/
│               │       │   ├── MetricCard.tsx
│               │       │   ├── PerformanceChart.tsx
│               │       │   └── ArtifactTable.tsx
│               │       ├── knowledge/
│               │       │   ├── ArtifactList.tsx
│               │       │   ├── KnowledgeSearch.tsx
│               │       │   └── ArtifactEditor.tsx
│               │       ├── leanaide/
│               │       │   ├── ProofEditor.tsx
│               │       │   ├── ModelSelector.tsx
│               │       │   └── VerificationDisplay.tsx
│               │       └── shared/
│               │           ├── ProgressBar.tsx
│               │           ├── LiveLogViewer.tsx
│               │           └── FormWrapper.tsx
│               │
│               ├── stores/
│               │   ├── workflowStore.ts                # NEW
│               │   ├── analyticsStore.ts               # NEW
│               │   ├── knowledgeStore.ts               # NEW
│               │   └── index.ts
│               │
│               ├── lib/
│               │   ├── api/
│               │   │   ├── client.ts                   # NEW
│               │   │   ├── endpoints.ts                # NEW
│               │   │   └── websocket.ts                # NEW
│               │   └── hooks/
│               │       ├── useApi.ts                   # NEW
│               │       ├── useWebSocket.ts             # NEW
│               │       └── useRealtime.ts              # NEW
│               │
│               └── types/
│                   ├── workflow.ts                     # NEW
│                   ├── analytics.ts                    # NEW
│                   ├── knowledge.ts                    # NEW
│                   └── openevolve.ts                   # NEW
│
├── deprecated/                   # NEW: Archived BubbleLab UI code
│   ├── demo_app.py
│   ├── mainlayout.py
│   ├── ui_components.py
│   └── ... (all BubbleLab UI files)
│
└── (existing Python backend engines - UNCHANGED)
    ├── evolution.py
    ├── adversarial.py
    ├── decomposition_engine.py
    ├── maker_engine.py
    ├── mdap_engine.py
    ├── knowledge_engine/
    └── ... (all business logic preserved)
```

---

## 🔧 TECHNOLOGY STACK

### API Gateway
- **Framework**: FastAPI (recommended) or Flask
- **WebSocket**: Socket.IO or Starlette WebSocket
- **Validation**: Pydantic
- **Authentication**: JWT (python-jose)
- **CORS**: Starlette CORS Middleware
- **Rate Limiting**: slowapi

### Frontend
- **Framework**: React 18+ (already in BubbleLab)
- **Routing**: React Router / TanStack Router (already in BubbleLab)
- **State Management**: Zustand
- **Server State**: React Query (TanStack Query)
- **Forms**: react-hook-form + Zod
- **Real-time**: WebSocket API + custom hooks
- **Charts**: Recharts
- **Tables**: TanStack Table (React Table)
- **Code Editor**: Monaco (already in BubbleLab)

### DevOps
- **Build**: Vite (already configured)
- **Testing**: Vitest, Playwright
- **Linting**: ESLint, Prettier
- **Type Checking**: TypeScript 5+

---

## ✅ SUCCESS CRITERIA

### Functional Requirements
- [x] All BubbleLab UI features accessible in new UI
- [x] Real-time updates working (progress, logs, notifications)
- [x] Authentication and authorization working
- [x] All visualizations rendering correctly
- [x] File upload/download working
- [x] WebSocket connections stable

### Non-Functional Requirements
- [x] Page load time < 2 seconds
- [x] API response time < 500ms (p95)
- [x] WebSocket latency < 100ms
- [x] 99.9% uptime for API gateway
- [x] Zero data loss during migration
- [x] Accessibility compliance (WCAG 2.1 AA)

### Quality Requirements
- [x] 80%+ test coverage
- [x] Zero critical bugs in production
- [x] All documentation updated
- [x] Code review completed for all PRs
- [x] Performance benchmarks met

---

## 🚨 RISKS & MITIGATIONS

### Risk 1: State Management Complexity
**Risk**: BubbleLab UI's `st.session_state` is simple; React state management is more complex.

**Mitigation**:
- Use Zustand for simple global state
- Use React Query for server state (caching, refetching)
- Create clear state ownership patterns
- Document state flow diagrams

### Risk 2: Real-time Updates
**Risk**: BubbleLab UI's `st.empty()` auto-refresh is seamless; WebSockets are more complex.

**Mitigation**:
- Implement robust WebSocket reconnection logic
- Create reusable hooks for common patterns
- Add connection status indicators in UI
- Implement fallback polling for unreliable connections

### Risk 3: Form Validation
**Risk**: BubbleLab UI forms are simple; React forms require more boilerplate.

**Mitigation**:
- Use react-hook-form for minimal boilerplate
- Create reusable `<FormWrapper>` component
- Implement Zod schemas for type-safe validation
- Build form component library for common patterns

### Risk 4: Performance Regression
**Risk**: New UI might be slower than BubbleLab UI's server-side rendering.

**Mitigation**:
- Implement aggressive caching (React Query)
- Use virtualization for large lists
- Optimize bundle size (code splitting)
- Monitor performance metrics and optimize bottlenecks

### Risk 5: Feature Parity Gaps
**Risk**: Some BubbleLab UI features may not have React equivalents.

**Mitigation**:
- Conduct thorough feature audit early
- Build custom components where needed
- Reuse existing BubbleLab components where possible
- Prioritize feature gaps by user impact

---

## 📚 REFERENCE MATERIALS

### BubbleLab UI Documentation
- [BubbleLab UI API Reference](https://docs.BubbleLab UI.io/library/api-reference)
- [Session State](https://docs.BubbleLab UI.io/library/api-reference/session-state)

### BubbleLab Codebase
- `BubbleLab/apps/bubble-studio/src/` - Existing React components
- `BubbleLab/apps/bubble-studio/src/components/` - Component library
- `BubbleLab/apps/bubble-studio/src/lib/integrations.ts` - OpenEvolve integrations

### API Design
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [WebSocket in FastAPI](https://fastapi.tiangolo.com/advanced/websockets/)
- [OpenAPI Specification](https://swagger.io/specification/)

### React Best Practices
- [React Query Documentation](https://tanstack.com/query/latest)
- [Zustand Guide](https://github.com/pmndrs/zustand)
- [react-hook-form](https://react-hook-form.com/)
- [Zod Validation](https://zod.dev/)

---

## 📝 TRACKING & PROGRESS

### Migration Dashboard
- **Total Files to Migrate**: 10+ major applications
- **API Endpoints to Create**: 50+
- **React Components to Build**: 100+
- **Tests to Write**: 200+

### Progress Metrics
- **Discovery Phase**: 0% complete
- **API Foundation**: 0% complete
- **UI Development**: 0% complete
- **Integration & Testing**: 0% complete
- **Deployment & Cutover**: 0% complete
- **Cleanup & Decommission**: 0% complete

---

## 🎯 NEXT STEPS (Immediate Actions)

1. **Review this task list** with the team and gather feedback
2. **Prioritize migration phases** based on business value
3. **Set up project tracking** (GitHub Projects, Jira, Linear)
4. **Assign owners** to each phase and task
5. **Begin Phase 1: Discovery & Audit**
6. **Create detailed specifications** for each API endpoint
7. **Set up development environment** for API gateway
8. **Begin prototyping** critical UI components

---

## 📞 CONTACT & SUPPORT

**Project Lead**: [TBD]
**Architecture Owner**: [TBD]
**Frontend Lead**: [TBD]
**Backend Lead**: [TBD]

**Standup**: Daily at [TIME]
**Sprint Review**: Bi-weekly on [DAY]
**Retrospective**: End of each phase

---

**Last Updated**: 2025-01-05
**Status**: 🟡 READY FOR EXECUTION
**Version**: 1.0

---

## 🎉 APPENDIX: BubbleLab UI → REACT CHEAT SHEET

| BubbleLab UI Component | React Equivalent | Library |
|---------------------|------------------|---------|
| `st.text_input()` | `<Input />` | shadcn/ui |
| `st.text_area()` | `<Textarea />` | shadcn/ui |
| `st.selectbox()` | `<Select />` | shadcn/ui |
| `st.slider()` | `<Slider />` | shadcn/ui |
| `st.button()` | `<Button />` | shadcn/ui |
| `st.checkbox()` | `<Checkbox />` | shadcn/ui |
| `st.radio()` | `<RadioGroup />` | shadcn/ui |
| `st.sidebar()` | `<Sidebar />` | Custom |
| `st.tabs()` | `<Tabs />` | shadcn/ui |
| `st.expander()` | `<Accordion />` | shadcn/ui |
| `st.form()` | `<Form />` | react-hook-form |
| `st.progress()` | `<ProgressBar />` | Custom |
| `st.spinner()` | `<Spinner />` | Custom |
| `st.empty()` | `useWebSocket()` | Custom |
| `st.metric()` | `<MetricCard />` | Custom |
| `st.plotly_chart()` | `<ResponsiveContainer>` | Recharts |
| `st.dataframe()` | `<DataTable />` | TanStack Table |
| `st.markdown()` | `<Markdown />` | react-markdown |
| `st.json()` | `<JsonViewer />` | Custom |
| `st.session_state` | `useStore()` | Zustand |
| `st.cache_data` | `useQuery()` | React Query |

---

**END OF MIGRATION PLAN**

