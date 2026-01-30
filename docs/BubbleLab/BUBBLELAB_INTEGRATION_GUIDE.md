# 🚀 OpenEvolve Plugin - BubbleLab Integration Guide

**How to actually USE the OpenEvolve plugin in BubbleLab's UI**

---

## 📋 TABLE OF CONTENTS
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Accessing in BubbleLab UI](#accessing-in-bubblelab-ui)
4. [Using in Workflows](#using-in-workflows)
5. [Navigation](#navigation)
6. [Practical Examples](#practical-examples)

---

## 🚀 QUICK START

### The 3-Step Process

```bash
# 1. Install the plugin
cd BubbleLab/apps/bubble-studio
npm install file:../../../OpenEvolve-Plugin

# 2. Add to BubbleLab's plugin registry
# (automatically done by installation)

# 3. Start BubbleLab
npm run dev
```

Then access at: `http://localhost:3000` → Look for "OpenEvolve" in the sidebar!

---

## 📦 INSTALLATION

### Step 1: Install the Plugin

```bash
cd BubbleLab/apps/bubble-studio

# Option A: Install from local path
npm install file:../../../OpenEvolve-Plugin

# Option B: Link for development
npm link ../../../OpenEvolve-Plugin

# Option C: Install as dependency (if published)
npm install @openevolve/plugin
```

### Step 2: Verify Installation

```bash
# Check package.json
cat package.json | grep -A 5 "dependencies"
```

You should see:
```json
{
  "dependencies": {
    "@openevolve/plugin": "file:../../../OpenEvolve-Plugin",
    // ... other dependencies
  }
}
```

### Step 3: Import in BubbleLab

**File**: `BubbleLab/apps/bubble-studio/src/main.tsx`

```typescript
import { OpenEvolvePlugin } from '@openevolve/plugin';
// or
import { OpenEvolvePlugin } from '@openevolve/plugin/dist/openevolve-plugin.es.js';

// Register the plugin
const plugins = [
  OpenEvolvePlugin,
  // ... other plugins
];
```

---

## 🎨 ACCESSING IN BUBBLELAB UI

### Method 1: Sidebar Navigation

When you start BubbleLab, OpenEvolve appears in the sidebar:

```
┌─────────────────────────────┐
│  BubbleLab Studio           │
├─────────────────────────────┤
│  🏠 Home                    │
│  💬 Bubbles                 │
│  ⚡ Templates               │
│  📊 Credentials             │
│  ⚙️  Settings                │
│  ─────────────────────────  │
│  🧬 OpenEvolve  ← NEW!      │
│    ├─ Dashboard             │
│    ├─ Analytics             │
│    ├─ Knowledge Base        │
│    └─ LeanAide              │
└─────────────────────────────┘
```

**Adding to Sidebar**:

**File**: `BubbleLab/apps/bubble-studio/src/components/Sidebar.tsx`

```typescript
import { OpenEvolveDashboard, AnalyticsDashboard, KnowledgeBasePage, LeanAidePage } from '@openevolve/plugin';

export function Sidebar() {
  return (
    <nav className="sidebar">
      {/* Existing menu items */}
      <SidebarItem icon="🏠" label="Home" to="/" />
      <SidebarItem icon="💬" label="Bubbles" to="/bubbles" />

      {/* Divider */}
      <SidebarDivider />

      {/* OpenEvolve Section */}
      <SidebarSection title="OpenEvolve" icon="🧬">
        <SidebarItem
          icon="📊"
          label="Dashboard"
          to="/openevolve"
          subitems={[
            { label: "Workflows", to: "/openevolve?tab=workflows" },
            { label: "Evolution", to: "/openevolve?tab=evolution" },
            { label: "Adversarial", to: "/openevolve?tab=adversarial" },
          ]}
        />
        <SidebarItem icon="📈" label="Analytics" to="/openevolve/analytics" />
        <SidebarItem icon="📚" label="Knowledge" to="/openevolve/knowledge" />
        <SidebarItem icon="🤖" label="LeanAide" to="/openevolve/leanaide" />
      </SidebarSection>
    </nav>
  );
}
```

### Method 2: Direct URL Access

Navigate directly to:
- `http://localhost:3000/openevolve` - Main Dashboard
- `http://localhost:3000/openevolve/analytics` - Analytics
- `http://localhost:3000/openevolve/knowledge` - Knowledge Base
- `http://localhost:3000/openevolve/leanaide` - LeanAide

### Method 3: From Bubble Editor

When creating or editing a Bubble, OpenEvolve services appear in the service catalog:

```
┌─────────────────────────────────┐
│  Add Service to Bubble          │
├─────────────────────────────────┤
│  Search services...             │
│                                 │
│  OpenEvolve Services:           │
│    🧬 Evolution Engine         │
│    ⚔️ Adversarial Testing       │
│    🔨 Maker Engine             │
│    🎯 MDAP                     │
│    🧩 Decomposition            │
│    📚 Knowledge Engine         │
│    🤖 LeanAide                 │
│    🔧 Hephaestus               │
│    🏛️ ROMA                     │
│    💡 Invention Planner         │
└─────────────────────────────────┘
```

---

## 🔧 USING IN WORKFLOWS

### Step 1: Create a New Bubble with OpenEvolve Service

```typescript
// In BubbleLab's flow editor
import { EvolutionService } from '@openevolve/plugin';

// 1. Drag Evolution Service into workflow
const bubble = {
  id: 'bubble-1',
  name: 'Content Evolution',
  className: 'EvolutionService',  // OpenEvolve service
  config: {
    content: 'Optimize this content',
    iterations: 10,
    temperature: 0.7,
    provider: 'anthropic',
    model: 'claude-3-sonnet',
  }
};
```

### Step 2: Configure Parameters

When you add an OpenEvolve service to a Bubble, a configuration panel appears:

**File**: `BubbleLab/apps/bubble-studio/src/components/openevolve/ConfigPanel.tsx`

```typescript
import { ConfigPanel } from '@openevolve/plugin';

export function BubbleEditor({ bubble, onUpdate }) {
  return (
    <div className="bubble-editor">
      <h2>Configure: {bubble.name}</h2>

      {/* Use OpenEvolve's ConfigPanel */}
      <ConfigPanel
        service={bubble.className}
        bubbleId={bubble.id}
        onConfigChange={(config) => onUpdate(config)}
      />
    </div>
  );
}
```

### Step 3: Execute Workflow

When you execute the workflow, OpenEvolve services run:

```typescript
import { usePluginExecution } from '@openevolve/plugin';

export function WorkflowExecutor({ workflow }) {
  const { execute, isExecuting } = usePluginExecution();

  const handleExecute = async () => {
    for (const bubble of workflow.bubbles) {
      if (bubble.className.startsWith('Evolution')) {
        await execute('evolution', bubble.config);
      }
      if (bubble.className.startsWith('Adversarial')) {
        await execute('adversarial', bubble.config);
      }
    }
  };

  return (
    <button onClick={handleExecute} disabled={isExecuting}>
      {isExecuting ? 'Running...' : 'Execute Workflow'}
    </button>
  );
}
```

### Step 4: Monitor Execution

Real-time monitoring of OpenEvolve execution:

```typescript
import { ExecutionMonitor } from '@openevolve/plugin';
import { useRealtimeEvolution } from '@openevolve/plugin';

export function WorkflowMonitor({ executionId }) {
  const { data, isConnected } = useRealtimeEvolution(executionId);

  return (
    <ExecutionMonitor
      executionId={executionId}
      service="evolution"
    />
  );
}
```

---

## 🧭 NAVIGATION

### Adding Routes to BubbleLab

**File**: `BubbleLab/apps/bubble-studio/src/routes/__root.tsx`

```typescript
import { createRouter } from '@tanstack/react-router';
import { OpenEvolveDashboard } from '@openevolve/plugin';

// Add OpenEvolve routes
const router = createRouter({
  // Existing routes
  '/': () => import('./routes/home').then(m => m.Home),
  '/bubbles': () => import('./routes/bubbles').then(m => m.Bubbles),

  // OpenEvolve routes
  '/openevolve': () => import('./routes/openevolve').then(m => m.OpenEvolveDashboard),
  '/openevolve/analytics': () => import('./routes/openevolve.analytics').then(m => m.AnalyticsDashboard),
  '/openevolve/knowledge': () => import('./routes/openevolve.knowledge').then(m => m.KnowledgeBasePage),
  '/openevolve/leanaide': () => import('./routes/openevolve.leanaide').then(m => m.LeanAidePage),
});

export { router };
```

**Create Route Files**:

**File**: `BubbleLab/apps/bubble-studio/src/routes/openevolve.tsx`

```typescript
import { OpenEvolveDashboard } from '@openevolve/plugin';

export function Route() {
  return <OpenEvolveDashboard />;
}
```

**File**: `BubbleLab/apps/bubble-studio/src/routes/openevolve.analytics.tsx`

```typescript
import { AnalyticsDashboard } from '@openevolve/plugin';

export function Route() {
  return <AnalyticsDashboard />;
}
```

---

## 💡 PRACTICAL EXAMPLES

### Example 1: Evolution Workflow

**Step 1: Create a Bubble**

```typescript
const evolutionBubble = {
  id: 'evo-1',
  name: 'Content Evolution',
  className: 'EvolutionService',
  config: {
    content: 'Write better documentation',
    iterations: 10,
    temperature: 0.7,
    populationSize: 5,
    provider: 'anthropic',
    model: 'claude-3-sonnet',
  }
};
```

**Step 2: Configure in UI**

```
┌──────────────────────────────────────┐
│  Evolution Service Configuration     │
├──────────────────────────────────────┤
│                                      │
│  Content to Evolve                   │
│  ┌────────────────────────────────┐  │
│  │ Write better documentation     │  │
│  └────────────────────────────────┘  │
│                                      │
│  Iterations:    [10] ▁▅▃▅▁▅▃▅▁▅▃    │
│  Temperature:  [0.7] ▁▅▃▅▁▅▃▅▁▅▃    │
│  Population:   [5]  ▁▅▃▅▁▅▃▅▁▅▃     │
│                                      │
│  Provider:     [Anthropic ▾]        │
│  Model:        [Claude 3 Sonnet ▾]  │
│                                      │
│  [Cancel]  [Apply Configuration]    │
└──────────────────────────────────────┘
```

**Step 3: Execute and Monitor**

```
┌──────────────────────────────────────┐
│  Execution Monitor                   │
├──────────────────────────────────────┤
│  ● Connected                         │
│                                      │
│  Progress: 70% ▁▅▃▅▁▅▃▅▁▅▃▁▅▃▁▅    │
│                                      │
│  Live Logs:                          │
│  [10:30:01] Starting evolution...     │
│  [10:30:05] Generation 1/10           │
│  [10:30:15] Generation 2/10           │
│  [10:30:25] Generation 3/10           │
│  ...                                 │
│                                      │
│  [Stop Execution]                    │
└──────────────────────────────────────┘
```

### Example 2: Adversarial Testing

```typescript
const adversarialBubble = {
  id: 'adv-1',
  name: 'Red Team Testing',
  className: 'AdversarialService',
  config: {
    targetContent: 'Secret API endpoint',
    attackMode: 'prompt-injection',
    redTeamProvider: 'openai',
    blueTeamProvider: 'anthropic',
    rounds: 3,
  }
};
```

### Example 3: Knowledge Base Integration

```typescript
import { ArtifactList, KnowledgeSearch } from '@openevolve/plugin';

export function KnowledgeTab() {
  return (
    <div>
      <KnowledgeSearch />
      <ArtifactList />
    </div>
  );
}
```

---

## 🎯 COMPLETE INTEGRATION CHECKLIST

### Installation
- [ ] Install plugin: `npm install file:../../../OpenEvolve-Plugin`
- [ ] Verify in package.json
- [ ] Import in main.tsx

### UI Integration
- [ ] Add to Sidebar component
- [ ] Create routes in routes/
- [ ] Add navigation links
- [ ] Test navigation works

### Workflow Integration
- [ ] Services appear in Bubble catalog
- [ ] ConfigPanel works in Bubble editor
- [ ] ExecutionMonitor displays progress
- [ ] Real-time updates via WebSocket
- [ ] Results display correctly

### Testing
- [ ] Navigate to /openevolve
- [ ] Create a workflow with Evolution service
- [ ] Configure parameters
- [ ] Execute workflow
- [ ] Monitor progress
- [ ] View results

---

## 🔌 HOW IT ALL CONNECTS

```
┌─────────────────────────────────────────────────────────────┐
│                    BubbleLab Studio                        │
│                                                               │
│  ┌──────────────┐                                            │
│  │   Sidebar     │                                           │
│  │              │                                            │
│  │  OpenEvolve ─┼─► /openevolve (route)                    │
│  │    ├─Dashboard  │                                            │
│  │    ├─Analytics  │                                            │
│  │    ├─Knowledge │                                            │
│  │    └─LeanAide  │                                            │
│  └──────────────┘                                            │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │         Flow Editor (Bubble Builder)          │           │
│  │                                                 │           │
│  │  [Drag OpenEvolve Service Here]               │           │
│  │                                                 │           │
│  │  ┌────────────────────┐                       │           │
│  │  │ Evolution Service  │                       │           │
│  │  │ ▼                  │                       │           │
│  │  │ [Config Panel]     │ ← From @openevolve/plugin│
│  │  │                    │                       │           │
│  │  └────────────────────┘                       │           │
│  └─────────────────────────────────────────────────┘           │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │         Execution Monitor                       │           │
│  │                                                 │           │
│  │  Progress: 70%                                  │           │
│  │  Live Logs: [streaming]                        │           │
│  │  Status: ✅ Connected                           │           │
│  │                                                 │           │
│  └─────────────────────────────────────────────────┘           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                          ↕ API calls
┌───────────────────────────────────────────────────────────────┐
│              API Gateway (FastAPI)                           │
│  ┌────────────────────────────────────────────────┐           │
│  │  POST /api/evolution/start                      │           │
│  │  WebSocket /ws/evolution/{id}                   │           │
│  │  GET /api/analytics/metrics                     │           │
│  └────────────────────────────────────────────────┘           │
└───────────────────────────────────────────────────────────────┘
                          ↕
┌───────────────────────────────────────────────────────────────┐
│              Python Backend Engines                          │
│  evolution.py, adversarial.py, maker.py, mdap.py, etc.       │
└───────────────────────────────────────────────────────────────┘
```

---

## 📱 USING THE UI

### 1. Main Dashboard

**URL**: `http://localhost:3000/openevolve`

```typescript
import { OpenEvolveDashboard } from '@openevolve/plugin';

// The dashboard includes:
// - Workflow tabs (Evolution, Adversarial, Maker, MDAP)
// - Configuration panel
// - Execution monitor
// - Real-time progress
// - Live log streaming
```

**What you see**:
```
┌─────────────────────────────────────────────────────┐
│  OpenEvolve Dashboard                               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Workflows] [Evolution] [Adversarial] [Maker]     │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────┐      │
│  │   Config Panel  │  │   Execution Monitor  │      │
│  │                 │  │                      │      │
│  │  Provider:      │  │  Progress: 70%       │      │
│  │  [Anthropic ▾]  │  │  ● Connected        │      │
│  │                 │  │                      │      │
│  │  Model:         │  │  Live Logs:         │      │
│  │  [Claude 3 ▾]   │  │  [10:30] Starting... │      │
│  │                 │  │  [10:31] Gen 1/10    │      │
│  │  Iterations:    │  │  [10:32] Gen 2/10    │      │
│  │  [10]           │  │  ...                │      │
│  │                 │  │                      │      │
│  │  [Start] [Stop] │  │  [View Results]      │      │
│  └─────────────────┘  └──────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### 2. Analytics Dashboard

**URL**: `http://localhost:3000/openevolve/analytics`

```typescript
import { AnalyticsDashboard } from '@openevolve/plugin';

// Displays:
// - KPI metrics cards
// - Performance charts
// - Artifact tables
// - Date range filters
```

### 3. Knowledge Base

**URL**: `http://localhost:3000/openevolve/knowledge`

```typescript
import { KnowledgeBasePage } from '@openevolve/plugin';

// Features:
// - Artifact search
// - Create/edit artifacts
// - Knowledge graph visualization
// - Version history
```

### 4. LeanAide

**URL**: `http://localhost:3000/openevolve/leanaide`

```typescript
import { LeanAidePage } from '@openevolve/plugin';

// Features:
// - Lean 4 proof editor
// - Model selection
// - Verification display
// - Progress tracking
```

---

## 🎨 CUSTOMIZATION

### Custom Sidebar Entry

**File**: `BubbleLab/apps/bubble-studio/src/components/Sidebar.tsx`

```typescript
import { useState } from 'react';
import { OpenEvolveSidebar } from '@openevolve/plugin';

export function Sidebar() {
  const [openEvolveOpen, setOpenEvolveOpen] = useState(false);

  return (
    <nav>
      {/* ... existing menu items */}

      {/* OpenEvolve Section */}
      <div className="menu-section">
        <div
          className="menu-header"
          onClick={() => setOpenEvolveOpen(!openEvolveOpen)}
        >
          <span className="icon">🧬</span>
          <span>OpenEvolve</span>
          <span className="arrow">{openEvolveOpen ? '▼' : '▶'}</span>
        </div>

        {openEvolveOpen && (
          <div className="submenu">
            <NavLink to="/openevolve">Dashboard</NavLink>
            <NavLink to="/openevolve/analytics">Analytics</NavLink>
            <NavLink to="/openevolve/knowledge">Knowledge</NavLink>
            <NavLink to="/openevolve/leanaide">LeanAide</NavLink>
          </div>
        )}
      </div>
    </nav>
  );
}
```

### Custom Bubble Node

**File**: `BubbleLab/apps/bubble-studio/src/components/flow_visualizer/nodes/CustomBubbles.tsx`

```typescript
import { EvolutionService } from '@openevolve/plugin';

export function OpenEvolveBubbleNode({ bubble }) {
  return (
    <div className="bubble-node openevolve-bubble">
      <img
        src="/integrations/evolution.svg"
        alt={bubble.className}
        className="service-icon"
      />
      <span className="bubble-name">{bubble.name}</span>

      {/* OpenEvolve specific UI */}
      <OpenEvolveConfigPanel bubble={bubble} />
    </div>
  );
}
```

---

## ⚙️ CONFIGURATION

### Environment Variables

**File**: `BubbleLab/apps/bubble-studio/.env`

```bash
# OpenEvolve API Configuration
VITE_OPENEVOLVE_API_URL=http://localhost:8000
VITE_OPENEVOLVE_WS_URL=ws://localhost:8000

# Or use full URL in production
VITE_OPENEVOLVE_API_URL=https://openevolve.ai/api
VITE_OPENEVOLVE_WS_URL=wss://openevolve.ai/ws
```

### Plugin Configuration

**File**: `BubbleLab/apps/bubble-studio/src/config/plugins.ts`

```typescript
export const OPENEVOLVE_CONFIG = {
  enabled: true,
  apiBaseUrl: import.meta.env.VITE_OPENEVOLVE_API_URL,
  wsBaseUrl: import.meta.env.VITE_OPENEVOLVE_WS_URL,
  services: [
    'evolution',
    'adversarial',
    'maker',
    'mdap',
    'decomposition',
    'knowledge',
    'leanaide',
    'hephaestus',
    'roma',
    'invention',
  ],
};
```

---

## 🧪 TESTING INTEGRATION

### Test 1: Verify Plugin Loads

```bash
cd BubbleLab/apps/bubble-studio
npm run dev

# Check browser console for:
# "OpenEvolve plugin loaded"
# "Registered 10 OpenEvolve services"
```

### Test 2: Navigate to Dashboard

```bash
# Open browser:
open http://localhost:3000/openevolve

# Should see:
# - OpenEvolve Dashboard
# - Configuration panel
# - Execution monitor
```

### Test 3: Create Workflow

```typescript
// In browser DevTools console:
import { useWorkflows } from '@openevolve/plugin';

// Create a workflow
const { createWorkflow } = useWorkflows();
await createWorkflow({
  name: 'Test Evolution',
  service: 'evolution',
  config: {
    content: 'Test content',
    iterations: 5,
  },
});
```

---

## 📞 SUPPORT

### Integration Issues?

**Problem**: Plugin not showing in sidebar
**Solution**:
```bash
# 1. Verify installation
cd BubbleLab/apps/bubble-studio
cat package.json | grep openevolve

# 2. Reinstall
npm install file:../../../OpenEvolve-Plugin

# 3. Restart dev server
npm run dev
```

**Problem**: Routes not working
**Solution**:
```bash
# 1. Check routes exist
ls src/routes/openevolve*

# 2. Verify router configuration
cat src/routes/__root.tsx | grep openevolve

# 3. Clear cache
rm -rf node_modules/.vite
npm run dev
```

**Problem**: API calls failing
**Solution**:
```bash
# 1. Check API Gateway is running
curl http://localhost:8000/health

# 2. Check environment variables
cat .env | grep OPENEVOLVE

# 3. Verify CORS configuration
curl -H "Origin: http://localhost:3000" http://localhost:8000/health
```

---

## ✅ SUCCESS CRITERIA

You'll know the integration works when:

- ✅ OpenEvolve appears in BubbleLab sidebar
- ✅ Can navigate to `/openevolve`
- ✅ Dashboard displays correctly
- ✅ Can configure OpenEvolve services
- ✅ Can execute workflows
- ✅ Real-time updates work
- ✅ Analytics display data
- ✅ Knowledge base is accessible
- ✅ LeanAide interface works

---

## 🎉 SUMMARY

The OpenEvolve plugin integrates with BubbleLab through:

1. **npm install** - Add as dependency
2. **Sidebar** - Add navigation menu
3. **Routes** - Add URL routing
4. **Bubble Catalog** - Services appear in workflow editor
5. **ConfigPanel** - Parameter configuration UI
6. **ExecutionMonitor** - Real-time progress tracking
7. **WebSocket** - Live updates

**That's it! The plugin is now part of BubbleLab!** 🚀

---

**Need more help?** Check out:
- `OpenEvolve-Plugin/README.md` - Plugin documentation
- `deploy/README.md` - Deployment guide
- `PROJECT_COMPLETE_FINAL.md` - Project status
