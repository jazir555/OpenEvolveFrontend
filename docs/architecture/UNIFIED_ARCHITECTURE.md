# UNIFIED ARCHITECTURE DESIGN

> **STATUS: implemented** (the merged plugin exists at `OpenEvolve-Plugin/src/` with `core/plugin/`, `core/types/`, `core/utils/`, `nodes/`, `components/` (69 `.tsx` files), `services/api/`, `services/hooks/`, `hooks/`, `schemas/`, `stores/`, `pages/`, `lib/`, `types/`, `utils/`, and `plugin.ts` / `components.ts` / `index.ts` entry points. Node system present: `nodes/BaseNode.ts`, `OpenEvolveBaseNode.ts`, `DecompositionNode.ts`, `SolutionNode.ts`, `VerificationNode.ts`, `registry.ts`, plus `MDAPNode.ts`, `MAKERNode.ts`, `LeanAIDENode.ts`, `ROMANode.ts`, `CrewAINode.ts`, `PyGraphistryNode.ts`).
>
> *File-name drift vs. this spec:* `core/plugin/` ships `PluginDefinition.ts` + `BubbleLabPluginDefinition.ts` (no separate `services.ts` / `endpoints.ts` / `lifecycle.ts`); `core/types/` ships `plugin-types.ts`, `enhanced-plugin-types.ts`, `extended-plugin-types.ts`, `nodes.ts`, `plugin.ts`; `core/utils/` ships `validators.ts`, `helpers.ts`, `advancedUtilities.ts`, `nodeFactory.ts`, `constants.ts`, `createOpenEvolvePlugin.ts`, `createEnhancedOpenEvolvePlugin.ts`, `enhancedErrorHandling.ts`. The naming-convention examples `services/api/evolution.ts` and `services/hooks/useEvolution.ts` do not exist — the actual files are `services/api/client.ts`, `services/api/endpoints.ts`, `services/api/OpenEvolveAPI.ts`, `services/api/websocket.ts` and `services/hooks/useApi.ts`, `useWorkflows.ts`, `useKnowledge.ts`, `useRealtime.ts`, `useWebSocket.ts`.
>
> **Integration backend:** the plugin's API client targets `services/openevolve-api` (FastAPI, port 8000), which mounts all `/api/*` route groups; the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts` forwards `/api/*` verbatim (mounted at `/` in `apps/bubblelab-api/src/index.ts`).
>
> **Last reconciled: 2026-08-20**

**Phase**: 2 - Architecture Design
**Status**: COMPLETE
**Based On**: Complete Feature Inventory (107 files from 3 plugins)

---

## Executive Summary

This document defines the unified architecture for the merged OpenEvolve plugin that will incorporate ALL features from three separate plugin implementations while maintaining ZERO feature loss.

---

## Design Principles

### 1. ZERO FEATURE LOSS
Every feature from all three plugins must be present and functional

### 2. CLEAN SEPARATION
- Core infrastructure in `/src/core`
- Features organized by domain
- Clear module boundaries

### 3. BACKWARD COMPATIBILITY
- All original exports maintained
- Adapter layer for breaking changes
- Deprecation warnings

### 4. AIR GAP COMPLIANCE
- No code embedded in BubbleLab core
- Plugin is completely standalone
- BubbleLab imports as external dependency

---

## Unified Directory Structure

```
OpenEvolve-Plugin/ (MERGED - Single Source of Truth)
├── src/
│   ├── core/                          # NEW: Core infrastructure
│   │   ├── plugin/                    # Plugin definition
│   │   │   ├── PluginDefinition.ts    # [From P3] Main plugin interface
│   │   │   ├── services.ts            # [From P3] Service definitions
│   │   │   ├── endpoints.ts           # [From P3] API endpoints
│   │   │   └── lifecycle.ts           # [From P3] Lifecycle hooks
│   │   ├── types/                     # Merged type system
│   │   │   ├── index.ts
│   │   │   ├── plugin.ts              # [From P1] Base types
│   │   │   ├── enhanced.ts            # [From P2] Enhanced types
│   │   │   ├── extended.ts            # [From P2] Extended types
│   │   │   ├── nodes.ts               # [From P2] Node types
│   │   │   └── workflows.ts           # [From P1] Workflow types
│   │   ├── constants/                 # Constants
│   │   │   └── index.ts
│   │   └── utils/                     # Core utilities
│   │       ├── factories.ts           # [From P2] Plugin factories
│   │       ├── validation.ts          # [From P1] Validators
│   │       ├── helpers.ts             # [From P1] Helpers
│   │       └── advanced.ts            # [From P2] Advanced utils
│   │
│   ├── nodes/                         # [From P2] Complete node system
│   │   ├── base/
│   │   │   ├── BaseNode.ts            # [From P2] Abstract base
│   │   │   └── OpenEvolveBaseNode.ts  # [From P2] OpenEvolve base
│   │   ├── workflow/                  # Workflow-specific nodes
│   │   │   ├── DecompositionNode.ts   # [From P2]
│   │   │   ├── SolutionNode.ts        # [From P2]
│   │   │   └── VerificationNode.ts    # [From P2]
│   │   ├── registry/
│   │   │   ├── NodeRegistry.ts        # [From P2] Dynamic registry
│   │   │   └── NodeFactory.ts         # [From P2] Factory
│   │   └── index.ts
│   │
│   ├── components/                    # ALL 37 components
│   │   ├── pages/                     # [From P1] Dashboard pages (5)
│   │   │   ├── OpenEvolveDashboard.tsx
│   │   │   ├── AnalyticsDashboard.tsx
│   │   │   ├── WorkflowBuilder.tsx
│   │   │   ├── LeanAidePage.tsx
│   │   │   └── KnowledgeBasePage.tsx
│   │   ├── workflow/                  # [From P1] Workflow UI (5)
│   │   │   ├── ConfigPanel.tsx
│   │   │   ├── ExecutionMonitor.tsx
│   │   │   ├── WorkflowCard.tsx
│   │   │   ├── WorkflowList.tsx
│   │   │   └── WorkflowTabs.tsx
│   │   ├── config/                    # [From P2] Enhanced config (5)
│   │   │   ├── EnhancedConfigPanel.tsx
│   │   │   ├── OpenEvolveConfigPanel.tsx
│   │   │   ├── PerformanceTab.tsx
│   │   │   ├── SecurityTab.tsx
│   │   │   └── WorkflowTabs.tsx       # [From P2] Remaining tabs
│   │   ├── nodes/                     # [From P2] Node UI (5)
│   │   │   ├── OpenEvolveNode.tsx
│   │   │   ├── DecompositionNode.tsx
│   │   │   ├── SolutionNode.tsx
│   │   │   └── VerificationNode.tsx
│   │   ├── analytics/                 # [From P1] Analytics (4)
│   │   │   ├── MetricCard.tsx
│   │   │   ├── PerformanceChart.tsx
│   │   │   ├── ArtifactTable.tsx
│   │   │   └── StatGrid.tsx
│   │   ├── knowledge/                 # [From P1] Knowledge (4)
│   │   │   ├── ArtifactList.tsx
│   │   │   ├── KnowledgeSearch.tsx
│   │   │   ├── ArtifactEditor.tsx
│   │   │   └── ArtifactDetail.tsx
│   │   ├── leanaide/                  # [From P1] LeanAide (4)
│   │   │   ├── ProofEditor.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── VerificationDisplay.tsx
│   │   │   └── ProgressTracker.tsx
│   │   └── shared/                    # [From P1] Shared (4)
│   │       ├── ProgressBar.tsx
│   │       ├── LiveLogViewer.tsx
│   │       ├── FormWrapper.tsx
│   │       └── StatusBadge.tsx
│   │
│   ├── services/                      # [From P1] Service layer
│   │   ├── api/                       # API clients (10)
│   │   │   ├── client.ts              # Base client
│   │   │   ├── endpoints.ts           # [From P3] All endpoints
│   │   │   ├── websocket.ts           # Real-time
│   │   │   ├── evolution.ts
│   │   │   ├── adversarial.ts
│   │   │   ├── maker.ts
│   │   │   ├── mdap.ts
│   │   │   ├── decomposition.ts
│   │   │   ├── knowledge.ts
│   │   │   ├── leanaide.ts
│   │   │   ├── crewai.ts
│   │   │   ├── roma.ts
│   │   │   └── invention.ts
│   │   ├── hooks/                     # React hooks (8)
│   │   │   ├── useApi.ts              # [From P1]
│   │   │   ├── useWebSocket.ts        # [From P1]
│   │   │   ├── useKnowledge.ts        # [From P1]
│   │   │   ├── useRealtime.ts         # [From P1]
│   │   │   ├── useWorkflows.ts        # [From P1]
│   │   │   ├── useEvolution.ts        # [From P1]
│   │   │   ├── useEnhancedConfig.ts   # [From P2]
│   │   │   └── index.ts
│   │   └── index.ts
│   │
│   ├── stores/                        # [From P1] State management
│   │   ├── useAuthStore.ts            # Authentication
│   │   ├── useWorkflowStore.ts        # Workflow state
│   │   ├── useAnalyticsStore.ts       # Analytics state
│   │   ├── useKnowledgeStore.ts       # Knowledge state
│   │   ├── useLeanAideStore.ts        # LeanAide state
│   │   ├── useEvolutionStore.ts       # Evolution state
│   │   └── index.ts
│   │
│   ├── schemas/                       # [Merged P1+P3] All 10 schemas
│   │   ├── evolution.ts               # Merge best from both
│   │   ├── adversarial.ts
│   │   ├── maker.ts
│   │   ├── mdap.ts
│   │   ├── decomposition.ts
│   │   ├── knowledge.ts
│   │   ├── leanaide.ts
│   │   ├── crewai.ts
│   │   ├── roma.ts
│   │   ├── invention.ts
│   │   └── index.ts
│   │
│   ├── hooks/                         # [Merged P1+P2] Additional hooks
│   │   ├── useEnhancedOpenEvolveConfig.ts  # [From P2]
│   │   └── index.ts
│   │
│   ├── assets/                        # [From P1] Icons and images
│   │   ├── icons/
│   │   └── images/
│   │
│   ├── plugin.ts                      # [NEW/From P3] Main plugin export
│   └── index.ts                       # [NEW] Unified exports
│
├── examples/                          # [From P2] Examples
├── docs/                              # [Merged P1+P2] Documentation
├── tests/                             # [From P1] Tests
├── package.json                       # Unified
├── tsconfig.json                      # Unified
├── vite.config.ts                     # Unified
└── README.md                          # Unified

```

---

## Namespace Strategy

### Conflict Resolution

| Category | Conflict | Resolution |
|----------|----------|------------|
| **Components** | None | Keep all 37, organized by domain |
| **Types** | plugin-types.ts | Merge into `/src/core/types/` with different files |
| **Config Panels** | Multiple ConfigPanel | Keep both with different names: Base vs Enhanced |
| **Hooks** | useApi vs useKnowledge | All different, keep all |
| **Schemas** | Duplicate in P1 and P3 | Merge best from each |
| **Plugin Factory** | createPlugin vs createEnhancedPlugin | Keep both, export from same place |

### Naming Conventions

1. **Components**: PascalCase, organized by domain
   - `components/pages/*` - Page-level components
   - `components/workflow/*` - Workflow-specific
   - `components/config/*` - Configuration panels
   - `components/nodes/*` - Node UI components

2. **Types**: camelCase files, descriptive names
   - `core/types/plugin.ts` - Base plugin types
   - `core/types/enhanced.ts` - Enhanced types from P2
   - `core/types/nodes.ts` - Node-specific types

3. **Services**: camelCase, function-based
   - `services/api/evolution.ts` - Evolution service
   - `services/hooks/useEvolution.ts` - Evolution hook

---

## Type System Unification

### Type Files Structure

```typescript
// core/types/index.ts - Main type export
export * from './plugin';      // Base types from P1
export * from './enhanced';    // Enhanced types from P2
export * from './extended';    // Extended types from P2
export * from './nodes';       // Node types from P2
export * from './workflows';   // Workflow types from P1

// Backward compatibility re-exports
export type {
  // From original P1
  PluginConfig,
  WorkflowConfig,
  // From original P2
  OpenEvolveNodeData,
  EnhancedPluginConfig,
} from './legacy';
```

### Type Merging Strategy

1. **Base Types** (P1): Keep as-is
2. **Enhanced Types** (P2): Keep as-is, import separately
3. **Extended Types** (P2): Keep as-is, import separately
4. **Node Types** (P2): Keep as-is, import separately
5. **Legacy Types**: Create compatibility layer

---

## Integration Architecture

### How Components Work Together

```
┌─────────────────────────────────────────────────────────┐
│                    Plugin Definition                    │
│                    (core/plugin/)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────┐                  ┌───────▼────────┐
│  Node System   │                  │  Component     │
│  (nodes/)      │◄─────────────────►│  Layer         │
│               │                  │  (components/)  │
└───────┬────────┘                  └────────────────┘
        │                                   │
        │            ┌──────────────────────┤
        │            │                      │
┌───────▼────┐  ┌───▼──────┐  ┌──────────▼──────┐
│ Services   │  │ Stores   │  │    Schemas       │
│(services/) │  │(stores/) │  │  (schemas/)      │
└────────────┘  └──────────┘  └─────────────────┘
```

### Data Flow

1. **Plugin Definition** → Registers all services
2. **Services** → Provide API and hooks
3. **Hooks** → Connect to Stores
4. **Stores** → Manage state
5. **Components** → Use hooks and render UI
6. **Nodes** → Use components for workflow

---

## Export Strategy

### Main Export Structure

```typescript
// index.ts - Main entry point

// Core exports
export { PluginDefinition } from './core/plugin';
export * from './core/types';
export * from './core/utils';

// Node system
export * from './nodes';
export { NodeRegistry, NodeFactory } from './nodes/registry';

// All components (37)
export * from './components';

// All services
export * from './services';

// All stores
export * from './stores';

// All schemas
export * from './schemas';

// All hooks
export * from './hooks';

// Backward compatibility
export {
  // Original P1 exports
  createPlugin,
  // Original P2 exports
  createEnhancedPlugin,
  // Original P3 exports
  OpenEvolvePlugin,
} from './core/plugin/factories';
```

### Backward Compatibility Layer

```typescript
// For users of original plugins
export const LegacyOpenEvolvePlugin = {
  // P1 exports
  ...P1Exports,
  // P2 exports
  ...P2Exports,
  // P3 exports
  ...P3Exports,
};
```

---

## Build Configuration

### Package.json Structure

```json
{
  "name": "@openevolve/plugin",
  "version": "2.0.0",
  "main": "./dist/index.js",
  "module": "./dist/index.esm.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.esm.js",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./components": "./dist/components/index.js",
    "./nodes": "./dist/nodes/index.js",
    "./services": "./dist/services/index.js"
  }
}
```

---

## Phase Handoff

### Ready for Phase 3: Core Infrastructure Merge

Phase 3 will use this architecture to:

1. Create the `/src/core/` directory structure
2. Merge type systems (P1 + P2)
3. Merge utilities (P1 + P2)
4. Integrate PluginDefinition (from P3)
5. Set up module boundaries

### Dependencies

- Phase 1 inventory complete ✅
- Architecture designed ✅
- Ready for file migration ⏳

---

## Validation

- [x] All 95+ files have a home in the structure
- [x] No naming conflicts unresolved
- [x] Backward compatibility planned
- [x] AIR GAP compliant (standalone plugin)
- [x] Zero feature loss maintained
- [x] Clear integration paths defined

---

**END OF PHASE 2 - ARCHITECTURE DESIGN COMPLETE**

**Next Phase**: Phase 3 - Core Infrastructure Merge
