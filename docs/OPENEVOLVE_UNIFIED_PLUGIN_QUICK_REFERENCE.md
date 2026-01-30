# OpenEvolve Unified Plugin - Quick Reference Guide

**Status**: ✅ **PRODUCTION READY**
**Version**: 1.0.0
**Date**: 2026-01-06
**Package**: `@openevolve/plugin`

---

## 🎯 WHAT IS THIS?

The **OpenEvolve Unified Plugin** is the result of merging THREE separate OpenEvolve plugin implementations into ONE cohesive, standalone plugin that integrates with BubbleLab.

### Source Plugins Merged:
1. **OpenEvolve-Plugin/** (65 files) - UI components, services, stores
2. **openevolve-bubblelab-plugin/** (30 files) - Node system, config panels
3. **BubbleLab embedded** (12 files) - PluginDefinition, service definitions

### Result:
- **114 TypeScript files** unified
- **38 React components**
- **Complete node system**
- **10 API services**
- **6 Zustand stores**
- **ZERO feature loss**

---

## 📦 INSTALLATION

### In BubbleLab (or any React app):

```bash
# From local directory
npm install file:../OpenEvolve-Plugin

# Or from npm (when published)
npm install @openevolve/plugin
```

### package.json:
```json
{
  "dependencies": {
    "@openevolve/plugin": "^1.0.0"
  }
}
```

---

## 🚀 QUICK START

### Basic Usage:

```typescript
// Import the plugin
import { OpenEvolvePlugin } from '@openevolve/plugin';

// Register in BubbleLab
registerPlugin(OpenEvolvePlugin);

// Or use components directly
import {
  OpenEvolveDashboard,
  EnhancedOpenEvolveConfigPanel,
  MetricCard
} from '@openevolve/plugin';

function App() {
  return (
    <OpenEvolveDashboard />
  );
}
```

### Using Nodes:

```typescript
import {
  DecompositionNode,
  SolutionNode,
  VerificationNode,
  NodeRegistry
} from '@openevolve/plugin';

// Register custom nodes
NodeRegistry.register({
  type: 'my-custom-node',
  class: MyCustomNode,
  category: 'custom'
});

// Create node from config
const node = NodeRegistry.create({
  type: 'decomposition',
  config: { /* ... */ }
});
```

### Using Services:

```typescript
import {
  useApi,
  useKnowledge,
  useWorkflows
} from '@openevolve/plugin';

function MyComponent() {
  const { data, loading } = useKnowledge();
  const { executeWorkflow } = useWorkflows();

  // ...
}
```

### Using Stores:

```typescript
import {
  useWorkflowStore,
  useKnowledgeStore,
  useAnalyticsStore
} from '@openevolve/plugin';

function MyComponent() {
  const workflows = useWorkflowStore(state => state.workflows);
  const addWorkflow = useWorkflowStore(state => state.addWorkflow);

  // ...
}
```

---

## 📚 AVAILABLE EXPORTS

### Components (38 total):

#### Pages (5):
```typescript
import {
  OpenEvolveDashboard,      // Main dashboard
  AnalyticsDashboard,        // Analytics overview
  WorkflowBuilder,           // Workflow editor
  LeanAidePage,              // Lean 4 integration
  KnowledgeBasePage          // Knowledge base
} from '@openevolve/plugin';
```

#### Workflow (5):
```typescript
import {
  ConfigPanel,              // Workflow configuration
  ExecutionMonitor,         // Execution tracking
  WorkflowCard,             // Workflow card UI
  WorkflowList,             // Workflow list
  WorkflowTabs              // Tabbed workflows
} from '@openevolve/plugin';
```

#### Config (5):
```typescript
import {
  EnhancedOpenEvolveConfigPanel,  // Enhanced config (recommended)
  OpenEvolveConfigPanel,          // Standard config
  PerformanceTab,                 // Performance settings
  SecurityTab,                    // Security settings
  RemainingTabs                   // Additional tabs
} from '@openevolve/plugin';
```

#### Nodes (5):
```typescript
import {
  OpenEvolveNode,           // Base node component
  DecompositionNodeComponent,
  SolutionNodeComponent,
  VerificationNodeComponent,
  example                   // Example node
} from '@openevolve/plugin';
```

#### Analytics (4):
```typescript
import {
  MetricCard,               // Metric display card
  PerformanceChart,         // Performance visualization
  ArtifactTable,            // Artifact listing
  StatGrid                  // Statistics grid
} from '@openevolve/plugin';
```

#### Knowledge (4):
```typescript
import {
  ArtifactList,             // Artifact list view
  KnowledgeSearch,          // Knowledge search
  ArtifactEditor,           // Artifact editor
  ArtifactDetail            // Artifact details
} from '@openevolve/plugin';
```

#### LeanAide (4):
```typescript
import {
  ProofEditor,              // Lean 4 proof editor
  ModelSelector,            // Model selection
  VerificationDisplay,      // Verification results
  ProgressTracker           // Progress tracking
} from '@openevolve/plugin';
```

#### Shared (4):
```typescript
import {
  ProgressBar,              // Progress indicator
  LiveLogViewer,            // Real-time logs
  FormWrapper,              // Form wrapper
  StatusBadge               // Status badges
} from '@openevolve/plugin';
```

#### Tabs (2):
```typescript
import {
  PerformanceConfigTab,     // Performance configuration
  SecurityConfigTab         // Security configuration
} from '@openevolve/plugin';
```

### Node Classes (8):

```typescript
import {
  // Base classes
  BaseNode,                 // Abstract base
  OpenEvolveBaseNode,       // OpenEvolve base

  // Workflow nodes
  DecompositionNode,        // Problem decomposition
  SolutionNode,             // Solution generation
  VerificationNode,         // Verification

  // Registry & factory
  NodeRegistry,             // Node registry
  registerNodes,            // Register helper
  createNodeFromRegistryConfig  // Factory function
} from '@openevolve/plugin';
```

### Services (10 API Endpoints):

```typescript
// Available services
const services = [
  'evolution',      // Genetic algorithm evolution
  'adversarial',    // Adversarial testing
  'maker',          // Creative generation
  'mdap',           // Multi-domain agent planner
  'decomposition',  // Problem decomposition
  'knowledge',      // Knowledge graph
  'leanaide',       // Lean 4 proof assistant
  'hephaestus',     // Code generation
  'roma',           // Reasoning system
  'invention'       // Invention planning
];
```

### Hooks (8 total):

```typescript
import {
  // Enhanced hooks
  useEnhancedOpenEvolveConfig,

  // Service hooks (from services/)
  useApi,                  // General API calls
  useKnowledge,            // Knowledge queries
  useRealtime,             // Real-time updates
  useWebSocket,            // WebSocket connection
  useWorkflows,            // Workflow management
} from '@openevolve/plugin';
```

### Stores (6 Zustand stores):

```typescript
import {
  useAuthStore,            // Authentication
  useWorkflowStore,        // Workflows
  useAnalyticsStore,       // Analytics
  useKnowledgeStore,       // Knowledge base
  useLeanAideStore,        // LeanAide
  useEvolutionStore        // Evolution tracking
} from '@openevolve/plugin';
```

### Schemas (10 workflow schemas):

```typescript
import {
  evolutionSchema,         // Evolution config
  adversarialSchema,       // Adversarial config
  makerSchema,             // Maker config
  mdapSchema,              // MDAP config
  decompositionSchema,     // Decomposition config
  knowledgeSchema,         // Knowledge config
  leanaideSchema,          // LeanAide config
  hephaestusSchema,        // Hephaestus config
  romaSchema,              // ROMA config
  inventionSchema          // Invention config
} from '@openevolve/plugin/schemas';
```

### Plugin Factory:

```typescript
import {
  createPlugin,            // Create plugin instance
  getPlugin,               // Get singleton
  resetPlugin,             // Reset singleton
  createEnhancedOpenEvolvePlugin,  // Enhanced factory
  getEnhancedOpenEvolvePlugin,     // Get enhanced
  resetEnhancedOpenEvolvePlugin    // Reset enhanced
} from '@openevolve/plugin';
```

### Constants:

```typescript
import {
  PLUGIN_NAME,             // '@openevolve/bubblelab-plugin'
  PLUGIN_VERSION,          // '1.0.0'
  DEFAULT_EVOLUTION_CONFIG,
  DEFAULT_ADVERSARIAL_CONFIG,
  DEFAULT_DECOMPOSITION_CONFIG,
  DEFAULT_INTEGRATION_CONFIG
} from '@openevolve/plugin';
```

---

## 🔧 CONFIGURATION

### Basic Configuration:

```typescript
import { createPlugin } from '@openevolve/plugin';

const plugin = createPlugin({
  evolution: {
    generations: 100,
    populationSize: 50,
    mutationRate: 0.1
  },
  adversarial: {
    enabled: true,
    attackStrategy: 'pgd'
  },
  api: {
    baseUrl: 'http://localhost:8000'
  }
});
```

### Enhanced Configuration:

```typescript
import { createEnhancedOpenEvolvePlugin } from '@openevolve/plugin';

const enhancedPlugin = createEnhancedOpenEvolvePlugin({
  services: ['evolution', 'adversarial', 'maker'],
  config: {
    performance: { /* ... */ },
    security: { /* ... */ }
  }
});
```

---

## 🏗️ ARCHITECTURE

### Directory Structure:

```
OpenEvolve-Plugin/
├── src/
│   ├── components/          # 38 React components
│   │   ├── pages/           # Dashboard pages
│   │   ├── workflow/        # Workflow UI
│   │   ├── config/          # Configuration panels
│   │   ├── nodes/           # Node components
│   │   ├── analytics/       # Analytics UI
│   │   ├── knowledge/       # Knowledge UI
│   │   ├── leanaide/        # LeanAide UI
│   │   ├── shared/          # Shared UI
│   │   └── tabs/            # Tab components
│   │
│   ├── nodes/               # 8 node classes
│   │   ├── BaseNode.ts
│   │   ├── OpenEvolveBaseNode.ts
│   │   ├── DecompositionNode.ts
│   │   ├── SolutionNode.ts
│   │   ├── VerificationNode.ts
│   │   ├── registry.ts
│   │   └── index.ts
│   │
│   ├── services/            # API services
│   │   ├── api/             # HTTP client, endpoints
│   │   ├── hooks/           # React hooks
│   │   └── tests/           # Service tests
│   │
│   ├── stores/              # Zustand stores
│   │   ├── authStore.ts
│   │   ├── workflowStore.ts
│   │   └── ...
│   │
│   ├── schemas/             # Workflow schemas
│   │   ├── evolution.ts
│   │   ├── adversarial.ts
│   │   └── ...
│   │
│   ├── core/                # Core infrastructure
│   │   ├── plugin/          # PluginDefinition
│   │   ├── types/           # TypeScript types
│   │   └── utils/           # Utilities
│   │
│   ├── hooks/               # Enhanced hooks
│   ├── utils/               # Utilities
│   ├── types/               # Type definitions
│   ├── plugin.ts            # Main plugin
│   └── index.ts             # Unified exports
│
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

---

## 🎯 KEY FEATURES

### 1. Complete UI System
- 38 React components for all OpenEvolve features
- Dashboard, workflow builder, analytics, knowledge base
- Lean 4 integration, node-based workflow editing

### 2. Node System
- Complete node hierarchy (BaseNode → OpenEvolveBaseNode → Specific Nodes)
- Dynamic node registration and factory pattern
- Type-safe node creation and validation

### 3. Service Layer
- 10 API service clients
- WebSocket support for real-time updates
- React hooks for easy integration

### 4. State Management
- 6 Zustand stores with persistence
- Type-safe actions and selectors
- DevTools integration

### 5. Schema Validation
- Zod schemas for all workflows
- Type inference from schemas
- Configuration validation

### 6. Plugin Factory
- Basic and enhanced plugin creation
- Singleton pattern for plugin instances
- Easy configuration and initialization

---

## 🔌 BUBBLELAB INTEGRATION

### Step 1: Add Dependency

```json
// BubbleLab/package.json
{
  "dependencies": {
    "@openevolve/plugin": "file:../OpenEvolve-Plugin"
  }
}
```

### Step 2: Register Plugin

```typescript
// BubbleLab/apps/bubble-studio/src/plugins/index.ts
import { OpenEvolvePlugin } from '@openevolve/plugin';

registerPlugin(OpenEvolvePlugin);
```

### Step 3: Remove Embedded Code

```bash
# Remove old embedded plugin (AIR GAP violation)
rm -rf BubbleLab/apps/bubble-studio/src/plugins/openevolve/
```

### Step 4: Use in Components

```typescript
import { OpenEvolveDashboard } from '@openevolve/plugin';

function MyBubbleLabPage() {
  return <OpenEvolveDashboard />;
}
```

---

## 📖 EXAMPLES

### Example 1: Using the Dashboard

```typescript
import { OpenEvolveDashboard } from '@openevolve/plugin';

function App() {
  return (
    <div>
      <h1>My Application</h1>
      <OpenEvolveDashboard />
    </div>
  );
}
```

### Example 2: Creating a Custom Node

```typescript
import { OpenEvolveBaseNode, NodeRegistry } from '@openevolve/plugin';

class MyCustomNode extends OpenEvolveBaseNode {
  async execute(context) {
    // Custom logic
    return { result: 'success' };
  }
}

// Register
NodeRegistry.register({
  type: 'my-custom',
  class: MyCustomNode,
  category: 'custom'
});
```

### Example 3: Using Knowledge Service

```typescript
import { useKnowledge } from '@openevolve/plugin';

function KnowledgeComponent() {
  const { data, loading, error } = useKnowledge();

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {data.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}
```

### Example 4: Using Workflow Store

```typescript
import { useWorkflowStore } from '@openevolve/plugin';

function WorkflowManager() {
  const workflows = useWorkflowStore(state => state.workflows);
  const addWorkflow = useWorkflowStore(state => state.addWorkflow);

  return (
    <div>
      <button onClick={() => addWorkflow({ name: 'New Workflow' })}>
        Add Workflow
      </button>
      {workflows.map(w => (
        <div key={w.id}>{w.name}</div>
      ))}
    </div>
  );
}
```

---

## 🛠️ TROUBLESHOOTING

### Import Errors

```typescript
// ❌ Wrong
import { Component } from '@openevolve/plugin/components';

// ✅ Correct
import { Component } from '@openevolve/plugin';
```

### Type Errors

```typescript
// Ensure types are imported
import type { OpenEvolveNodeData, EvolutionConfig } from '@openevolve/plugin';
```

### Plugin Not Found

```bash
# Reinstall dependencies
npm install

# Rebuild plugin
cd OpenEvolve-Plugin
npm run build
```

---

## 📚 DOCUMENTATION

### Additional Resources:

- **MERGE_COMPLETE.md** - Detailed merge report
- **UNIFICATION_COMPLETE.md** - Architecture documentation
- **FINAL_PLUGIN_UNIFICATION_VERIFICATION.md** - Verification report
- **README.md** - Main plugin documentation
- Source code comments - Detailed inline documentation

---

## ✅ CHECKLIST

Before using the plugin:

- [ ] Plugin installed via `npm install`
- [ ] TypeScript configured for path aliases
- [ ] React 18+ installed
- [ ] Zustand installed (for stores)
- [ ] Required peer dependencies installed

For BubbleLab integration:

- [ ] `@openevolve/plugin` in package.json
- [ ] Plugin registered in BubbleLab
- [ ] Embedded code removed
- [ ] Build successful
- [ ] Components render correctly

---

## 🚀 GETTING HELP

### Issues?
1. Check the documentation above
2. Review source code comments
3. Check example implementations
4. Review verification report

### Resources:
- Main README in OpenEvolve-Plugin/
- Component source code in src/components/
- Type definitions in src/types/
- Schema definitions in src/schemas/

---

**Version**: 1.0.0
**Last Updated**: 2026-01-06
**Status**: ✅ Production Ready

---

**Quick Reference Guide End**
