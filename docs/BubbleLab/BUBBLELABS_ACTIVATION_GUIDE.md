# 🚀 OPENEVOLVE PLUGIN - BUBBLELABS ACTIVATION GUIDE

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Status**: ✅ Production Ready

---

## 📋 TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Backend Activation](#backend-activation)
3. [Frontend Activation](#frontend-activation)
4. [Plugin Registration](#plugin-registration)
5. [Verification](#verification)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## ⚡ QUICK START (5 Minutes)

### Prerequisites
- ✅ BubbleLab Studio installed
- ✅ OpenEvolve backend API running
- ✅ Node.js 18+ installed
- ✅ Plugin built (`npm run build` completed)

### Fastest Activation Path

```bash
# 1. Start OpenEvolve backend API
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python openevolve_api.py --port 8000

# 2. Build the plugin (if not already built)
cd OpenEvolve-Plugin
npm run build

# 3. Copy plugin to BubbleLab
mkdir -p ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve
cp -r dist/* ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve/

# 4. Register plugin in BubbleLab
# Edit: BubbleLab/apps/bubble-studio/src/plugins/index.ts
# Add: import { OpenEvolvePlugin } from './openevolve/plugin';
#       registerPlugin(OpenEvolvePlugin);

# 5. Start BubbleLab Studio
cd ../../BubbleLab/apps/bubble-studio
npm run dev
```

**That's it!** The OpenEvolve plugin should now be available in BubbleLab Studio.

---

## 🔧 BACKEND ACTIVATION

### Step 1: Start the OpenEvolve API Server

The OpenEvolve backend provides all the workflow engines (Evolution, Adversarial, ROMA, Invention, etc.).

```bash
# Navigate to frontend directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Start the API server
python openevolve_api.py --port 8000

# Or with custom configuration
python openevolve_api.py \
  --port 8000 \
  --host 0.0.0.0 \
  --log-level info \
  --enable-cors
```

**Expected Output**:
```
✓ OpenEvolve API Server started
✓ Listening on http://0.0.0.0:8000
✓ Available endpoints:
  - POST /api/v1/evolution/start
  - POST /api/v1/adversarial/start
  - POST /api/v1/roma/solve
  - POST /api/v1/invention/plan
  ... (42+ endpoints)
```

### Step 2: Verify Backend is Running

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}
```

### Step 3: Configure Environment Variables (Optional)

Create a `.env` file in the OpenEvolve directory:

```bash
# OpenEvolve API Configuration
OPENEVOLVE_API_URL=http://localhost:8000
OPENEVOLVE_API_KEY=your_api_key_here
OPENEVOLVE_WS_URL=ws://localhost:8000/ws

# Feature Flags
ENABLE_ROMA=true
ENABLE_INVENTION=true
ENABLE_LEANAIDE=true
ENABLE_HEPHAEUSTUS=true
```

---

## 🎨 FRONTEND ACTIVATION

### Step 1: Build the OpenEvolve Plugin

```bash
# Navigate to plugin directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\OpenEvolve-Plugin

# Install dependencies (if not already installed)
npm install

# Build the plugin
npm run build

# Expected output:
# ✓ 745 modules transformed
# ✓ built in 6.33s
# ✓ dist/openevolve-plugin.es.js  1,557.32 kB
# ✓ dist/openevolve-plugin.umd.js  1,030.02 kB
```

### Step 2: Copy Plugin to BubbleLab

There are **two methods** to integrate the plugin with BubbleLab:

#### Method A: Direct Integration (Recommended)

```bash
# Create plugin directory in BubbleLab
mkdir -p ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve

# Copy built files
cp -r dist/* ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve/

# Verify files were copied
ls ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve/
# Should see: index.js, index.d.ts, components/, nodes/, etc.
```

#### Method B: NPM Package Integration

```bash
# In BubbleLab directory
cd ../../BubbleLab/apps/bubble-studio

# Install as local dependency
npm install file:../OpenEvolve-Plugin/OpenEvolve-Plugin-1.0.0.tgz

# Or create tarball first
cd ../OpenEvolve-Plugin
npm pack
# Creates: openevolve-plugin-1.0.0.tgz

# Then install in BubbleLab
cd ../../BubbleLab/apps/bubble-studio
npm install file:../../OpenEvolve-Plugin/openevolve-plugin-1.0.0.tgz
```

### Step 3: Configure BubbleLab Environment

Create or update `.env.local` in BubbleLab:

```bash
# BubbleLab Studio Configuration
VITE_OPENEVOLVE_API_URL=http://localhost:8000
VITE_OPENEVOLVE_WS_URL=ws://localhost:8000/ws
VITE_OPENEVOLVE_ENABLED=true
```

---

## 📝 PLUGIN REGISTRATION

### Step 1: Register Plugin in BubbleLab

Edit `BubbleLab/apps/bubble-studio/src/plugins/index.ts`:

```typescript
/**
 * OpenEvolve Plugin Registration
 */
import { OpenEvolvePlugin } from './openevolve/plugin';
import { registerPlugin } from '../core/plugin-registry';

// Register OpenEvolve plugin
registerPlugin({
  ...OpenEvolvePlugin,

  // Plugin metadata
  id: 'openevolve',
  name: 'OpenEvolve',
  version: '1.0.0',
  description: 'AI evolution and optimization platform',

  // Enable all capabilities
  capabilities: {
    workflows: true,
    analytics: true,
    knowledgeBase: true,
    leanAide: true,
    evolution: true,
    adversarial: true,
    maker: true,
    mdap: true,
    decomposition: true,
    hephaestus: true,
    roma: true,
    invention: true,
  },

  // Auto-initialize
  initialize: async () => {
    console.log('[OpenEvolve] Plugin registered successfully');
    return true;
 },
});

// Export for use in components
export { OpenEvolvePlugin };
```

### Step 2: Add OpenEvolve Routes (Optional)

If you want dedicated OpenEvolve pages, add to `BubbleLab/apps/bubble-studio/src/App.tsx`:

```typescript
import { OpenEvolveDashboard } from './plugins/openevolve/components/pages/OpenEvolveDashboard';
import { AnalyticsDashboard } from './plugins/openevolve/components/pages/AnalyticsDashboard';

const routes = [
  // ... existing routes

  // OpenEvolve routes
  {
    path: '/openevolve',
    element: <OpenEvolveDashboard />,
  },
  {
    path: '/openevolve/analytics',
    element: <AnalyticsDashboard />,
  },
];
```

### Step 3: Add OpenEvolve to Navigation (Optional)

Edit `BubbleLab/apps/bubble-studio/src/components/Navigation.tsx`:

```typescript
import { BrainCircuit } from 'lucide-react';

const navItems = [
  // ... existing items

  // OpenEvolve section
  {
    section: 'OpenEvolve',
    items: [
      {
        title: 'Dashboard',
        path: '/openevolve',
        icon: <BrainCircuit className="w-4 h-4" />,
      },
      {
        title: 'Analytics',
        path: '/openevolve/analytics',
        icon: <BarChart3 className="w-4 h-4" />,
      },
      {
        title: 'Workflows',
        path: '/openevolve/workflows',
        icon: <Workflow className="w-4 h-4" />,
      },
    ],
  },
];
```

---

## ✅ VERIFICATION

### 1. Check Plugin Registration

Start BubbleLab Studio and check browser console:

```bash
cd ../../BubbleLab/apps/bubble-studio
npm run dev
```

**Expected Console Output**:
```
[OpenEvolve] Plugin registered successfully
[OpenEvolve] Plugin initialized
[OpenEvolve] All 12 nodes registered
[OpenEvolve] All 5 config panels loaded
```

### 2. Test Plugin in Browser

Navigate to: `http://localhost:3000/openevolve`

You should see:
- ✅ OpenEvolve Dashboard loaded
- ✅ Sidebar with OpenEvolve menu items
- ✅ Node palette with OpenEvolve nodes
- ✅ Configuration panels accessible

### 3. Test Node Creation

1. Open BubbleLab Studio
2. Create new workflow
3. Drag "Evolution" node from palette
4. Configure node parameters
5. Run workflow
6. Check for successful execution

### 4. Verify API Communication

```bash
# Check browser Network tab for:
# - POST /api/v1/evolution/start (200 OK)
# - WebSocket connection established
# - Real-time updates received
```

---

## 💡 USAGE EXAMPLES

### Example 1: Evolution Workflow

```typescript
import { EvolutionNode } from '@openevolve/plugin/nodes';

// Create evolution node
const evolutionNode = new EvolutionNode({
  content: 'Optimize this algorithm...',
  mode: 'quality_diversity',
  generations: 100,
  populationSize: 50,
  mutationRate: 0.1,
  crossoverRate: 0.8,
});

// Execute
const result = await evolutionNode.execute({}, {
  apiUrl: 'http://localhost:8000',
  apiKey: 'your-key',
});

console.log('Best solution:', result.data.bestSolution);
```

### Example 2: ROMA Reasoning

```typescript
import { ROMANode } from '@openevolve/plugin/nodes';
import { useROMA } from '@openevolve/plugin/hooks';

function ROMAComponent() {
  const { execute, status, result } = useROMA();

  const handleExecute = async () => {
    try {
      const romaResult = await execute({
        task: 'Solve this complex problem...',
        reasoningMode: 'collaborative',
        agentCount: 5,
        agentRoles: ['analyst', 'critic', 'synthesizer', 'validator', 'explorer'],
        rounds: 3,
        confidenceThreshold: 0.7,
        includeReasoningTrace: true,
        enableVoting: true,
      });

      console.log('Solution:', romaResult.solution);
      console.log('Confidence:', romaResult.confidence);
    } catch (error) {
      console.error('ROMA execution failed:', error);
    }
  };

  return (
    <div>
      <button onClick={handleExecute} disabled={status.isRunning}>
        {status.isRunning ? 'Reasoning...' : 'Execute ROMA'}
      </button>
      {result && <div>{result.solution}</div>}
    </div>
  );
}
```

### Example 3: Invention Planning

```typescript
import { InventionNode } from '@openevolve/plugin/nodes';
import { useInvention } from '@openevolve/plugin/hooks';

function InventionPlanner() {
  const { createPlan, status, result } = useInvention();

  const handleCreate = async () => {
    try {
      const invention = await createPlan({
        goal: 'Create a revolutionary AI system...',
        domain: 'technology',
        innovativeness: 0.8,
        planningStages: ['research', 'ideation', 'prototyping', 'testing'],
        includePriorArt: true,
        includeFeasibility: true,
        includeRoadmap: true,
        detailLevel: 'comprehensive',
      });

      console.log('Invention Plan:', invention.plan);
      console.log('Prior Art:', invention.priorArt);
    } catch (error) {
      console.error('Invention planning failed:', error);
    }
  };

  return (
    <div>
      <button onClick={handleCreate} disabled={status.isRunning}>
        {status.isRunning ? 'Planning...' : 'Create Invention Plan'}
      </button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
```

---

## 🐛 TROUBLESHOOTING

### Issue 1: Plugin Not Appearing in BubbleLab

**Symptoms**: OpenEvolve menu not visible, nodes not in palette

**Solutions**:
1. Check plugin registration in `plugins/index.ts`
2. Verify files were copied to correct location
3. Check browser console for errors
4. Clear browser cache and restart

```bash
# Verify plugin files exist
ls ../../BubbleLab/apps/bubble-studio/src/plugins/openevolve/

# Should see:
# index.js
# index.d.ts
# components/
# nodes/
# hooks/
# etc.
```

### Issue 2: API Connection Refused

**Symptoms**: Network errors, 502 Bad Gateway

**Solutions**:
1. Verify OpenEvolve API is running:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Check API URL configuration:
   ```bash
   # In BubbleLab .env.local
   echo $VITE_OPENEVOLVE_API_URL
   # Should output: http://localhost:8000
   ```

3. Check CORS settings:
   ```bash
   # API should have CORS enabled
   curl -H "Origin: http://localhost:3000" \
        http://localhost:8000/api/health
   ```

### Issue 3: Nodes Not Executing

**Symptoms**: Nodes fail to execute, timeout errors

**Solutions**:
1. Check API key configuration
2. Verify API endpoints are accessible:
   ```bash
   curl -X POST http://localhost:8000/api/v1/evolution/start \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer your-key" \
        -d '{"content":"test","mode":"standard"}'
   ```

3. Check browser Network tab for error responses

### Issue 4: Build Errors

**Symptoms**: TypeScript errors, build fails

**Solutions**:
1. Clear node_modules and reinstall:
   ```bash
   cd OpenEvolve-Plugin
   rm -rf node_modules
   npm install
   npm run build
   ```

2. Check TypeScript version:
   ```bash
   npm list typescript
   # Should be: typescript@5.8.0
   ```

### Issue 5: WebSocket Connection Failing

**Symptoms**: No real-time updates, connection errors

**Solutions**:
1. Verify WebSocket URL:
   ```bash
   # In BubbleLab .env.local
   VITE_OPENEVOLVE_WS_URL=ws://localhost:8000/ws
   ```

2. Test WebSocket connection:
   ```bash
   # Using websocat or wscat
   wscat --connect ws://localhost:8000/ws
   ```

---

## 📊 FEATURE CHECKLIST

### After Activation, Verify All Features Work

- [ ] **Dashboard Loads**: Navigate to `/openevolve`
- [ ] **Analytics Working**: Navigate to `/openevolve/analytics`
- [ ] **All 12 Nodes Available**: In node palette
  - [ ] Evolution
  - [ ] Adversarial
  - [ ] Decomposition
  - [ ] Solution
  - [ ] Verification
  - [ ] Maker
  - [ ] MDAP
  - [ ] Knowledge Query
  - [ ] LeanAIDE
  - [ ] Hephaestus
  - [ ] ROMA
  - [ ] Invention
- [ ] **Config Panels Open**: Can open configuration for each node
- [ ] **Workflows Execute**: Can run workflows successfully
- [ ] **Real-time Updates**: See progress updates
- [ ] **Results Display**: Results shown correctly
- [ ] **API Calls Successful**: No console errors

---

## 🎯 NEXT STEPS

### 1. Explore the Plugin

```bash
# Open BubbleLab Studio
cd ../../BubbleLab/apps/bubble-studio
npm run dev

# Navigate to:
# http://localhost:3000/openevolve
```

### 2. Try Example Workflows

The plugin comes with example workflows for each engine. Check the documentation:
- Evolution workflow examples
- Adversarial testing examples
- ROMA reasoning examples
- Invention planning examples

### 3. Customize Configuration

Each node has configurable parameters. Access via:
- UI config panels
- API directly
- Configuration files

---

## 📚 ADDITIONAL RESOURCES

### Documentation
- [Plugin Architecture](../PLUGIN_ARCHITECTURE.md)
- [API Reference](./API_REFERENCE.md)
- [Node Documentation](./NODES.md)
- [Configuration Guide](./CONFIGURATION.md)

### Examples
- [Example Workflows](./examples/)
- [Integration Examples](./examples/integrations/)
- [API Usage Examples](./examples/api/)

### Support
- GitHub Issues: [openevolve/plugin/issues](https://github.com/openevolve/plugin/issues)
- Documentation: [docs.openevolve.ai](https://docs.openevolve.ai)
- Community: [discord.gg/openevolve](https://discord.gg/openevolve)

---

## ✅ ACTIVATION COMPLETE

Once you've completed these steps:

1. ✅ Backend API running on port 8000
2. ✅ Plugin built and copied to BubbleLab
3. ✅ Plugin registered in BubbleLab
4. ✅ Environment variables configured
5. ✅ BubbleLab Studio restarted
6. ✅ Dashboard accessible at `/openevolve`
7. ✅ All 12 nodes available in palette
8. ✅ API communication verified

**The OpenEvolve plugin is now fully activated and ready to use in BubbleLab!** 🎉

---

**End of Activation Guide**

**For questions or issues, please refer to the troubleshooting section or create a GitHub issue.**
