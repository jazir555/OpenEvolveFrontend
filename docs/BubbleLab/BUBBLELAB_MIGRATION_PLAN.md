# BubbleLab Migration Plan: Streamlit to Native React UI

## Executive Summary

**Objective:** Migrate OpenEvolve Frontend from Streamlit (Python) to BubbleLab's native React/TypeScript UI while preserving 100% functionality.

**Timeline:** 6 weeks
**Approach:** Phased migration with parallel systems during transition
**Team:** 2 Frontend Developers (React/TS), 1 Backend Developer (Python), 1 QA Engineer

---

## Current Architecture

```
┌─────────────────────┐
│  Streamlit UI (Py)  │
│  - st.session_state │
│  - st.* components  │
│  - Linear pages     │
└──────────┬──────────┘
           │ Direct Python calls
           ▼
┌─────────────────────────────┐
│  Python Backend (FastAPI)   │
│  - api_server.py            │
│  - workflow_engine.py       │
│  - Team/Gauntlet Managers   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────┐
│  Workflow Engine    │
│  - LLM Integration  │
│  - Sovereign Decomp │
└─────────────────────┘
```

## Target Architecture

```
┌──────────────────────────────────────┐
│   BubbleLab React UI (TypeScript)    │
│   - Zustand Stores                   │
│   - Clerk Authentication             │
│   - TanStack Router                  │
│   - React Flow Visualization         │
└──────────────┬───────────────────────┘
               │ HTTP/WebSocket
               ▼
┌──────────────────────────────────────┐
│     API Bridge Layer (NEW)           │
│     - api_bridge.py                  │
│     - CORS Middleware                │
│     - SSE Streaming                  │
│     - WebSocket Support              │
└──────────────┬───────────────────────┘
               │ Python function calls
               ▼
┌──────────────────────────────────────┐
│     Python Backend (FastAPI)         │
│     - api_server.py                  │
│     - workflow_engine.py             │
│     - Team/Gauntlet Managers         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│     Workflow Engine                  │
│     - LLM Integration                │
│     - Sovereign Decomposition        │
└──────────────────────────────────────┘
```

---

## Migration Phases Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 0** | Week 1 | Infrastructure | API Bridge, Auth, Types |
| **Phase 1** | Week 2 | Navigation & Layout | Routes, Dashboard, Sidebar |
| **Phase 2** | Week 3 | Configuration UI | Forms, Teams, Gauntlets, Settings |
| **Phase 3** | Week 4 | Execution & Streaming | Real-time updates, Results |
| **Phase 4** | Week 5 | Advanced Features | Benchmarks, Analytics, Files |
| **Phase 5** | Week 6 | Testing & Polish | Tests, Docs, Optimization |

---

## Component Mapping

### Streamlit → React Component Equivalents

#### Layout Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.tabs()` | TanStack Router | File-based routing |
| `st.sidebar` | `<Sidebar />` | Custom component |
| `st.columns()` | CSS Grid | Tailwind grid classes |
| `st.container()` | `<div>` | HTML div |
| `st.expander()` | `<details>` / Accordion | Native or shadcn/ui |

#### Input Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.text_input()` | `<input type="text">` | HTML input |
| `st.text_area()` | `<textarea>` | HTML textarea |
| `st.number_input()` | `<input type="number">` | HTML input |
| `st.slider()` | `<Slider />` | Custom component |
| `st.selectbox()` | `<Select />` | Custom or shadcn/ui |
| `st.multiselect()` | `<MultiSelect />` | Custom component |
| `st.checkbox()` | `<input type="checkbox">` | HTML checkbox |
| `st.radio()` | `<RadioGroup />` | Custom component |
| `st.file_uploader()` | `<FileUpload />` | Custom component |

#### Display Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.title()` | `<h1>` | HTML heading |
| `st.header()` | `<h2>` | HTML heading |
| `st.subheader()` | `<h3>` | HTML heading |
| `st.markdown()` | `<ReactMarkdown />` | react-markdown library |
| `st.code()` | `<CodeBlock />` | Prism or Shiki |
| `st.metric()` | `<MetricCard />` | Custom component |
| `st.progress()` | `<ProgressBar />` | Custom component |
| `st.dataframe()` | `<DataTable />` | TanStack Table |

#### Action Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.button()` | `<Button />` | BubbleLab existing |
| `st.form_submit_button()` | Form submit | React Hook Form |
| `st.download_button()` | `<DownloadButton />` | Custom component |

#### Status Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.info()` | Info alert | Custom or shadcn/ui |
| `st.warning()` | Warning alert | Custom or shadcn/ui |
| `st.error()` | Error alert | Custom or shadcn/ui |
| `st.success()` | Success alert | Custom or shadcn/ui |

#### Chart Components
| Streamlit | BubbleLab React | Implementation |
|-----------|-----------------|----------------|
| `st.plotly_chart()` | Plotly chart | react-plotly.js |
| `st.altair_chart()` | Vega-Lite | vega-lite |
| `st.pyplot()` | Chart.js | react-chartjs-2 |

---

## Route Structure

```
/                                    → WorkflowDashboard (main landing)
/workflows                           → WorkflowList (all workflows)
/workflows/create                    → WorkflowCreate (new workflow wizard)
/workflows/:id                       → WorkflowDetails (workflow overview)
/workflows/:id/configure             → WorkflowConfig (configure teams/gauntlets)
/workflows/:id/execute               → WorkflowExecution (execute workflow)
/workflows/:id/results               → WorkflowResults (view results)
/teams                               → TeamList (manage teams)
/teams/create                        → TeamCreate (new team)
/teams/:id                           → TeamEdit (edit team)
/gauntlets                           → GauntletList (manage gauntlets)
/gauntlets/create                    → GauntletCreate (new gauntlet)
/gauntlets/:id                       → GauntletEdit (edit gauntlet)
/benchmarks                          → BenchmarkList (run benchmarks)
/benchmarks/:id                      → BenchmarkDetails (benchmark results)
/analytics                           → AnalyticsDashboard (statistics)
/settings                            → SettingsPanel (app configuration)
```

---

## State Management Strategy

### Zustand Stores

| Store | Purpose | Replaces |
|-------|---------|----------|
| `uiStore` | Global UI state (existing) | - |
| `executionStore` | Per-flow execution state (existing, adapt) | st.session_state execution flags |
| `generationStore` | AI generation state (existing) | - |
| `configStore` (NEW) | LLM configuration | st.session_state llm_provider, model_* |
| `workflowStore` (NEW) | Workflow management | st.session_state workflow data |
| `teamStore` (NEW) | Team management | TeamManager backend |
| `gauntletStore` (NEW) | Gauntlet management | GauntletManager backend |

### Session State Migration

**From Streamlit:**
```python
st.session_state.llm_provider = "openai"
st.session_state.model_leanaide = "gpt-4"
st.session_state.temperature = 0.7
st.session_state.workflow_running = False
st.session_state.selected_teams = ["Content Analyzer"]
```

**To Zustand (configStore):**
```typescript
interface ConfigState {
  llmProvider: string;
  modelLeanAide: string;
  temperature: number;
  setLLMProvider: (provider: string) => void;
  setModel: (model: string) => void;
  setTemperature: (temp: number) => void;
}
```

---

## API Integration Strategy

### API Bridge Design (api_bridge.py)

**Purpose:** Bridge React frontend to Python backend with CORS, SSE, and WebSocket support.

**Key Features:**
1. CORS middleware for React frontend
2. Clerk JWT validation
3. Tenant ID extraction
4. SSE streaming for execution events
5. WebSocket support for real-time updates
6. Request/response transformation

**Endpoints:**
```
GET  /api/health                    → Health check
GET  /api/workflows                 → List workflows
POST /api/workflows                 → Create workflow
GET  /api/workflows/:id             → Get workflow details
PUT  /api/workflows/:id             → Update workflow
DEL  /api/workflows/:id             → Delete workflow
POST /api/workflows/:id/start       → Start execution
POST /api/workflows/:id/pause       → Pause execution
POST /api/workflows/:id/resume      → Resume execution
POST /api/workflows/:id/stop        → Stop execution
GET  /api/workflows/:id/results     → Get results
GET  /api/stream/workflow/:id       → SSE execution stream

GET  /api/teams                    → List teams
POST /api/teams                    → Create team
GET  /api/teams/:id                → Get team details
PUT  /api/teams/:id                → Update team
DEL  /api/teams/:id                → Delete team

GET  /api/gauntlets                → List gauntlets
POST /api/gauntlets                → Create gauntlet
GET  /api/gauntlets/:id            → Get gauntlet details
PUT  /api/gauntlets/:id            → Update gauntlet
DEL  /api/gauntlets/:id            → Delete gauntlet

GET  /api/settings/llm             → Get LLM settings
PUT  /api/settings/llm             → Update LLM settings
```

### Authentication Flow

```
React UI
  ↓ (Clerk authentication)
Clerk JWT Token
  ↓ (Authorization header)
API Bridge (api_bridge.py)
  ↓ (Validate JWT, extract tenant_id)
Python Backend (api_server.py)
  ↓ (Tenant-scoped operations)
Workflow Engine
```

---

## Real-time Communication

### SSE (Server-Sent Events) Implementation

**Use Case:** Workflow execution streaming

**Python Side:**
```python
@api_bridge.get("/stream/workflow/{workflow_id}")
async def stream_workflow_execution(workflow_id: str):
    async def event_generator():
        async for event in workflow_engine.stream_execution(workflow_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**React Side:**
```typescript
export function useExecutionStream(workflowId: string) {
  const addEvent = useExecutionStore(state => state.addEvent);

  useEffect(() => {
    const eventSource = new EventSource(
      `${API_BASE}/stream/workflow/${workflowId}`
    );

    eventSource.onmessage = (e) => {
      const event = JSON.parse(e.data);
      addEvent(event);
    };

    return () => eventSource.close();
  }, [workflowId]);
}
```

---

## Risk Mitigation

### Identified Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Real-time streaming failures** | High | Medium | Fallback polling, reconnection logic |
| **Authentication complexity** | High | Low | Clerk pre-built components, API key fallback |
| **State synchronization issues** | High | Medium | Zustand persistence, optimistic updates |
| **Performance degradation** | Medium | Medium | Virtualization, caching, lazy loading |
| **Lost Streamlit features** | High | Medium | Feature parity checklist, user testing |

### Rollback Strategy

1. Run Streamlit and BubbleLab in parallel
2. Feature flags to control UI
3. Keep Streamlit fallback for 30 days
4. Quick rollback via DNS/proxy switch
5. Monitor error rates and user feedback

---

## Success Metrics

### Technical Metrics
- Page load time < 2s
- Time to Interactive < 3s
- Lighthouse score > 90
- Test coverage > 80%
- Error rate < 1%
- Uptime > 99.5%

### User Experience Metrics
- 100% feature parity with Streamlit
- User migration rate > 80% in 30 days
- Support tickets < 5 per week
- NPS score > 50

---

## Deployment Strategy

1. **Week 1-5:** Development on feature branches
2. **Week 6:** Testing and staging deployment
3. **Beta Launch:** 10% of users
4. **Monitor:** Error rates, performance, feedback
5. **Gradual Rollout:** Ramp to 100%
6. **Stabilization:** Keep Streamlit fallback for 30 days

---

## Post-Migration Support

### Monitoring
- Sentry for error tracking
- PostHog for analytics
- Lighthouse CI for performance
- Pingdom for uptime

### Continuous Improvement
- Weekly: Review error logs and feedback
- Monthly: Performance optimization
- Quarterly: User surveys and features

