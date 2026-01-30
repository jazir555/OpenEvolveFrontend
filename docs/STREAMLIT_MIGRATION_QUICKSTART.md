# 🚀 STREAMLIT EXCISION - AGENT TASK QUICKSTART

**For Agent Execution & Task Management**

---

## 📋 IMMEDIATE NEXT ACTIONS (DO THIS FIRST)

### Phase 0: Preparation (Day 1)
- [x] **Read**: `STREAMLIT_TO_BUBBLELAB_MIGRATION.md` (full migration plan)
- [x] **Read**: `.claude/CLAUDE.md` (project constitution)
- [x] **Read**: `BubbleLab/.cursor/rules/bubblelab.mdc` (BubbleLab patterns)
- [ ] **Grep**: Find all Streamlit imports: `grep -r "import streamlit\|import st\|from streamlit" --include="*.py"`
- [ ] **Create**: Migration tracking spreadsheet (Google Sheets/Notion)
- [ ] **Set up**: Git branch `feature/streamlit-excision`

---

## 🎯 AGENT TASK BREAKDOWN

### Agent 1: Discovery & Audit Agent
**Timeline**: Week 1
**Priority**: CRITICAL

**Tasks**:
1. **Catalog ALL Streamlit files** (399 imports identified)
   - Run: `find . -name "*.py" -exec grep -l "streamlit\|st\." {} \;`
   - Document each file's purpose in `STREAMLIT_FILES_INVENTORY.md`
   - Identify dependencies between files

2. **Map backend logic for each Streamlit file**
   - Extract class/function names called by UI
   - Document data flow from UI → Backend
   - Identify session state usage patterns

3. **Create component mapping matrix**
   - Streamlit widget → React component mapping
   - Identify gaps in BubbleLab component library
   - Prioritize component development

**Deliverables**:
- `STREAMLIT_FILES_INVENTORY.md`
- `BACKEND_API_REQUIREMENTS.md`
- `COMPONENT_MAPPING_MATRIX.md`

---

### Agent 2: API Gateway Architect
**Timeline**: Weeks 2-4
**Priority**: CRITICAL

**Tasks**:
1. **Design REST API contracts** (OpenAPI 3.0 spec)
   - Use FastAPI or Flask
   - Define endpoints for all backend engines
   - Create Pydantic schemas for request/response

2. **Implement authentication middleware**
   - JWT token validation
   - User session management
   - CORS configuration for BubbleLab

3. **Create WebSocket infrastructure**
   - Setup Socket.IO or Starlette WebSocket
   - Implement room-based subscriptions
   - Add connection pool management

4. **Extract business logic** from Streamlit files
   - Keep Python engines intact
   - Create service layer for API calls
   - Add error handling and logging

**Deliverables**:
- `api/gateway/` directory with FastAPI/Flask app
- `API_OPENAPI_SPEC.yaml` (OpenAPI specification)
- `WEBSOCKET_EVENTS_SPEC.md` (event types and schemas)

**Example Endpoint Structure**:
```python
# api/gateway/routes/workflows.py
from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

router = APIRouter(prefix="/api/workflows")

class WorkflowCreate(BaseModel):
    name: str
    config: dict

@router.post("/")
async def create_workflow(workflow: WorkflowCreate):
    # Call existing workflow engine
    result = workflow_engine.create(workflow.dict())
    return result

@router.websocket("/{workflow_id}/execute")
async def execute_workflow(websocket: WebSocket, workflow_id: str):
    await websocket.accept()
    # Stream execution progress
    async for progress in workflow_engine.execute(workflow_id):
        await websocket.send_json(progress)
```

---

### Agent 3: React Component Developer
**Timeline**: Weeks 5-10
**Priority**: HIGH

**Tasks**:
1. **Create core React components** for OpenEvolve
   - Workflow management interface
   - Analytics dashboard components
   - Knowledge base UI
   - LeanAide interface

2. **Implement state management** (Zustand + React Query)
   - Create stores for workflows, analytics, knowledge
   - Set up API client with axios
   - Implement caching and refetching

3. **Build real-time update system**
   - WebSocket client hooks
   - Progress bars and live logs
   - Execution monitoring

4. **Migrate visualizations** from Plotly to Recharts
   - Line charts, bar charts, scatter plots
   - Responsive containers
   - Interactive tooltips

**Deliverables**:
- `BubbleLab/apps/bubble-studio/src/components/openevolve/` (component library)
- `BubbleLab/apps/bubble-studio/src/stores/` (state management)
- `BubbleLab/apps/bubble-studio/src/lib/api/` (API client)

**Component Template**:
```tsx
// BubbleLab/apps/bubble-studio/src/components/openevolve/workflow/WorkflowList.tsx
import { useWorkflows } from '@/lib/api/hooks';

export function WorkflowList() {
  const { data: workflows, isLoading } = useWorkflows();

  if (isLoading) return <div>Loading...</div>;

  return (
    <div className="workflow-list">
      {workflows?.map(workflow => (
        <WorkflowCard key={workflow.id} workflow={workflow} />
      ))}
    </div>
  );
}
```

---

### Agent 4: Integration & Test Engineer
**Timeline**: Weeks 11-14
**Priority**: HIGH

**Tasks**:
1. **Write integration tests** for API endpoints
   - Test all CRUD operations
   - Test WebSocket connections
   - Test error handling

2. **Create E2E tests** with Playwright
   - Test user workflows end-to-end
   - Test authentication flow
   - Test real-time updates

3. **Performance testing**
   - Load test API gateway
   - Test WebSocket scalability
   - Optimize slow queries

4. **Security audit**
   - Test authentication/authorization
   - Test input validation
   - Test rate limiting

**Deliverables**:
- `api/tests/` (test suite)
- `BubbleLab/apps/bubble-studio/src/e2e/` (E2E tests)
- `PERFORMANCE_TEST_REPORT.md`

---

### Agent 5: Deployment & Operations Engineer
**Timeline**: Weeks 15-18
**Priority**: MEDIUM

**Tasks**:
1. **Set up staging environment**
   - Deploy API gateway
   - Deploy BubbleLab UI
   - Configure domain and SSL

2. **Create deployment pipeline**
   - CI/CD configuration (GitHub Actions)
   - Automated testing in pipeline
   - Blue-green deployment strategy

3. **Monitor and optimize**
   - Set up logging and metrics
   - Create alerts for errors
   - Optimize performance bottlenecks

4. **Execute production cutover**
   - Deploy to production
   - Monitor health metrics
   - Keep Streamlit as fallback

**Deliverables**:
- `DEPLOYMENT_GUIDE.md`
- `OPERATIONS_RUNBOOK.md`
- Staging and production environments

---

## 🗂️ FILE ORGANIZATION

### New Files to Create
```
api/gateway/                          # NEW - API Gateway
├── main.py                          # FastAPI/Flask app entry point
├── middleware/
│   ├── auth.py                     # JWT authentication
│   ├── cors.py                     # CORS configuration
│   └── rate_limit.py               # Rate limiting
├── routes/
│   ├── workflows.py                # Workflow endpoints
│   ├── analytics.py                # Analytics endpoints
│   ├── knowledge.py                # Knowledge base endpoints
│   ├── leanaide.py                 # LeanAide endpoints
│   └── decomposition.py            # Decomposition endpoints
├── realtime/
│   ├── manager.py                  # WebSocket manager
│   └── handlers/
│       ├── workflow.py             # Workflow event handlers
│       ├── analytics.py            # Analytics event handlers
│       └── execution.py            # Execution event handlers
├── models/
│   └── schemas.py                  # Pydantic schemas
└── tests/
    ├── test_workflows.py
    ├── test_analytics.py
    └── test_websocket.py

BubbleLab/apps/bubble-studio/src/
├── pages/                           # NEW - Page components
│   ├── OpenEvolveDashboard.tsx     # Main dashboard (replaces mainlayout.py)
│   ├── AnalyticsDashboard.tsx      # Analytics (replaces analytics_dashboard.py)
│   ├── WorkflowBuilder.tsx         # Workflow builder (replaces openevolve_bubblelabs_ui.py)
│   ├── LeanAidePage.tsx            # LeanAide interface
│   └── KnowledgeBasePage.tsx       # Knowledge base UI
│
├── components/openevolve/           # NEW - OpenEvolve component library
│   ├── workflow/
│   │   ├── WorkflowList.tsx
│   │   ├── WorkflowCard.tsx
│   │   ├── ExecutionMonitor.tsx
│   │   └── ConfigPanel.tsx
│   ├── analytics/
│   │   ├── MetricCard.tsx
│   │   ├── PerformanceChart.tsx
│   │   └── ArtifactTable.tsx
│   ├── knowledge/
│   │   ├── ArtifactList.tsx
│   │   ├── KnowledgeSearch.tsx
│   │   └── ArtifactEditor.tsx
│   └── shared/
│       ├── ProgressBar.tsx
│       ├── LiveLogViewer.tsx
│       └── FormWrapper.tsx
│
├── stores/                          # NEW - State management
│   ├── workflowStore.ts
│   ├── analyticsStore.ts
│   ├── knowledgeStore.ts
│   └── index.ts
│
├── lib/api/                         # NEW - API client
│   ├── client.ts                   # Axios/fetch client
│   ├── endpoints.ts                # API endpoint definitions
│   └── websocket.ts                # WebSocket client
│
├── lib/hooks/                       # NEW - Custom hooks
│   ├── useApi.ts                   # API call hook
│   ├── useWebSocket.ts             # WebSocket hook
│   └── useRealtime.ts              # Real-time updates
│
└── types/                           # NEW - TypeScript types
    ├── workflow.ts
    ├── analytics.ts
    ├── knowledge.ts
    └── openevolve.ts

deprecated/                          # NEW - Archive Streamlit files
├── demo_app.py
├── mainlayout.py
├── ui_components.py
├── analytics_dashboard.py
├── openevolve_bubblelabs_ui.py
├── bubblelabs_ui_component.py
├── LeanAide/server/streamlit_ui.py
├── decomposition_dashboard.py
├── collaboration_manager.py
└── knowledge_base_ui.py
```

### Files to Modify (Integrations)
```
BubbleLab/apps/bubble-studio/src/lib/integrations.ts  # ✅ Already updated with OpenEvolve integrations
BubbleLab/apps/bubble-studio/src/routes/index.tsx     # Add new routes
BubbleLab/apps/bubble-studio/src/components/Sidebar.tsx  # Add navigation links
```

---

## 🔄 WORKFLOW EXAMPLE

### Example 1: Migrating demo_app.py

**Step 1: Extract Backend Logic**
```python
# demo_app.py (original)
import streamlit as st
from content_analyzer import ContentAnalyzer

def main():
    st.title("Content Analyzer")
    content = st.text_area("Enter content")
    if st.button("Analyze"):
        analyzer = ContentAnalyzer()
        result = analyzer.analyze(content)
        st.write(result)
```

**Step 2: Create API Endpoint**
```python
# api/gateway/routes/analytics.py
from fastapi import APIRouter
from content_analyzer import ContentAnalyzer

router = APIRouter(prefix="/api/analytics")

@router.post("/analyze")
async def analyze_content(content: str):
    analyzer = ContentAnalyzer()
    result = analyzer.analyze(content)
    return result
```

**Step 3: Create React Component**
```tsx
// BubbleLab/apps/bubble-studio/src/pages/DemoPage.tsx
import { useState } from 'react';
import { useAnalyzeContent } from '@/lib/hooks/useApi';

export function DemoPage() {
  const [content, setContent] = useState('');
  const { mutate: analyze, data: result } = useAnalyzeContent();

  return (
    <div>
      <h1>Content Analyzer</h1>
      <textarea value={content} onChange={(e) => setContent(e.target.value)} />
      <button onClick={() => analyze({ content })}>Analyze</button>
      {result && <div>{result}</div>}
    </div>
  );
}
```

**Step 4: Create API Hook**
```typescript
// BubbleLab/apps/bubble-studio/src/lib/hooks/useApi.ts
import { useMutation } from '@tanstack/react-query';
import { apiClient } from './client';

export function useAnalyzeContent() {
  return useMutation({
    mutationFn: (content: string) =>
      apiClient.post('/api/analytics/analyze', { content }),
  });
}
```

---

## ✅ WEEKLY CHECKPOINTS

### Week 1 Checkpoint
- [ ] All Streamlit files catalogued
- [ ] Component mapping complete
- [ ] API requirements documented

### Week 4 Checkpoint
- [ ] API gateway implemented
- [ ] All endpoints tested
- [ ] WebSocket infrastructure ready

### Week 10 Checkpoint
- [ ] All React components built
- [ ] State management working
- [ ] Real-time updates functional

### Week 14 Checkpoint
- [ ] All tests passing
- [ ] Performance optimized
- [ ] Security audited

### Week 18 Checkpoint
- [ ] Production deployment complete
- [ ] Monitoring in place
- [ ] Streamlit decommissioned

---

## 🚨 CRITICAL PATH TASKS

These tasks MUST be completed in order. Do not skip.

1. ✅ **Discovery**: Catalog all Streamlit files
2. ✅ **API Design**: Create OpenAPI specification
3. ✅ **API Gateway**: Implement authentication and core endpoints
4. ✅ **WebSocket**: Build real-time infrastructure
5. ✅ **Core Components**: Build workflow and analytics UIs
6. ✅ **Integration**: Connect frontend to backend
7. ✅ **Testing**: Comprehensive test coverage
8. ✅ **Deployment**: Staging and production
9. ✅ **Cutover**: Switch from Streamlit to BubbleLab
10. ✅ **Cleanup**: Archive Streamlit code

---

## 📞 AGENT COORDINATION

### Agent Communication
- **Daily Standup**: Sync on progress and blockers
- **Weekly Review**: Demo completed work
- **Slack/Discord**: #streamlit-excision channel
- **GitHub Issues**: Track tasks and bugs

### Handoff Process
1. Agent completes phase
2. Creates pull request with documentation
3. Reviews with next agent
4. Deploy to staging for validation
5. Next agent begins next phase

---

## 🎯 SUCCESS METRICS

### Phase Completion Criteria
- **Discovery**: 100% of Streamlit files catalogued
- **API Gateway**: All endpoints returning 200 OK
- **React Components**: All components rendering without errors
- **Integration**: End-to-end workflows working
- **Testing**: 80%+ code coverage
- **Deployment**: Production stable for 1 week

### Quality Gates
- ✅ Code review approved
- ✅ All tests passing
- ✅ Documentation updated
- ✅ Performance benchmarks met
- ✅ Security scan clean

---

**START EXECUTION**: Begin with Agent 1 - Discovery & Audit

**FIRST TASK**: Run `find . -name "*.py" -exec grep -l "streamlit" {} \;` and catalog results

**GOOD LUCK! 🚀**
