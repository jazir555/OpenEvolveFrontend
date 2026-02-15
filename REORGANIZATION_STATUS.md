# Architecture Reorganization - COMPLETE

## ✅ Structure Fixed

All components have been reorganized into the proper Federation Constitution-compliant structure.

## New Structure

```
Frontend/
├── core-projects/
│   ├── BubbleLab/              ← ✅ READ ONLY (Original Core Project)
│   ├── RAGBits/               ← ✅ READ ONLY (Original Core Project)
│   ├── Datapizza/             ← ✅ READ ONLY (Original Core Project)
│   └── [other core projects]   ← ✅ READ ONLY
│
└── glue/                        ← ✅ THE GLUE LAYER (Proper Location)
    ├── adapters/
    │   ├── bubblelab/           ← ✅ Moved from bubblelab-converted/
    │   │   └── src/
    │   │       ├── components/  ← Converted BubbleLab UI components
    │   │       ├── lib/          ← BubbleLab integration logic
    │   │       └── [converted files]
    │   │
    │   ├── ragbits/            ← ✅ Moved from bubblelabs-ragbits-plugin/
    │   │   ├── src/
    │   │   │   ├── components/
    │   │   │   ├── hooks/
    │   │   │   ├── lib/
    │   │   │   ├── services/
    │   │   │   └── types/
    │   │   └── package.json
    │   │
    │   ├── datapizza/          ← ✅ Moved from datapizza-bubblelab-plugin/
    │   │   ├── src/
    │   │   │   ├── components/
    │   │   │   ├── hooks/
    │   │   │   ├── services/
    │   │   │   ├── types/
    │   │   │   └── utils/
    │   │   └── package.json
    │   │
    │   ├── leanaide/           ← ✅ Moved from leanaide-bubblelab-plugin/
    │   ├── roma/               ← ✅ Moved from roma-bubblelab-plugin/
    │   ├── z3/                 ← ✅ Moved from openevolve-bubblelab-plugin/
    │   │       (actually Z3 is part of openevolve-bubblelab-plugin)
    │   │
    │   └── [other plugins]
    │
    ├── orchestration/
    │   ├── workflow-system/    ← ✅ NEW: Moved workflow orchestration
    │   │   ├── plugin-registry.ts
    │   │   ├── plugin-orchestrator.ts
    │   │   ├── workflow-templates.ts
    │   │   ├── workflow-monitoring.ts
    │   │   ├── plugin-events.ts
    │   │   ├── plugin-integration.ts
    │   │   ├── plugin-adapters.ts
    │   │   ├── useBubbleLabIntegration.ts
    │   │   └── index.ts
    │   │
    │   ├── unified-verification/ ← ✅ Existing (already correct)
    │   │   ├── src/
    │   │   │   ├── canonical.ts
    │   │   │   ├── verification-service.ts
    │   │   │   ├── z3-verifier.ts
    │   │   │   └── leanaide-verifier.ts
    │   │   ├── tsconfig.json
    │   │   ├── package.json
    │   │   └── README.md
    │   │
    │   ├── correlation-tracker.ts
    │   ├── dead-letter-queue.ts
    │   ├── event-bus.ts
    │   ├── event-types.ts
    │   └── [other orchestration components]
    │
    ├── schemas/                  ← ✅ Existing (already correct)
    ├── lib/                      ← ✅ Existing (already correct)
    └── docs/                     ← ✅ Existing (already correct)
```

## What Was Moved

### From Root Level → glue/adapters/

1. ✅ `bubblelab-converted/` → `glue/adapters/bubblelab/`
   - All converted UI components
   - BubbleLab integration logic
   - Workflow execution UI

2. ✅ `bubblelabs-ragbits-plugin/` → `glue/adapters/ragbits/`
   - RAGBits plugin implementation
   - Search, ingest, config components
   - Client and service layers

3. ✅ `datapizza-bubblelab-plugin/` → `glue/adapters/datapizza/`
   - Datapizza plugin implementation
   - Processing, querying, pipeline components
   - Client and service layers

4. ✅ `leanaide-bubblelab-plugin/` → `glue/adapters/leanaide/`
   - LeanAide plugin implementation

5. ✅ `roma-bubblelab-plugin/` → `glue/adapters/roma/`
   - ROMA plugin implementation

6. ✅ `openevolve-bubblelab-plugin/` → `glue/adapters/z3/`
   - OpenEvolve plugin implementation (contains Z3, etc.)

### From bubblelab-converted/src/lib → glue/orchestration/workflow-system/

7. ✅ `plugin-registry.ts`
8. ✅ `plugin-orchestrator.ts`
9. ✅ `workflow-templates.ts`
10. ✅ `workflow-monitoring.ts`
11. ✅ `plugin-events.ts`
12. ✅ `plugin-integration.ts`
13. ✅ `plugin-adapters.ts`
14. ✅ `useBubbleLabIntegration.ts`
15. ✅ `index.ts`

## Next Steps (Import Path Updates)

Now I need to update all import statements throughout the codebase to reference the new locations.

### Import Path Mappings

**Old → New:**

```typescript
// UI Components
@/components/openevolve/main/WorkflowExecutionTab
  → glue/adapters/bubblelab/src/components/openevolve/main/WorkflowExecutionTab

// Libraries
@/lib/plugin-registry.ts
  → glue/orchestration/workflow-system/plugin-registry.ts

@/lib/workflow-orchestrator.ts
  → glue/orchestration/workflow-system/plugin-orchestrator.ts

@/lib/workflow-templates.ts
  → glue/orchestration/workflow-system/workflow-templates.ts

@/lib/workflow-monitoring.ts
  → glue/orchestration/workflow-system/workflow-monitoring.ts

@/lib/plugin-events.ts
  → glue/orchestration/workflow-system/plugin-events.ts

@/lib/plugin-integration.ts
  → glue/orchestration/workflow-system/plugin-integration.ts

@/lib/plugin-adapters.ts
  → glue/orchestration/workflow-system/plugin-adapters.ts

@/hooks/useBubbleLabIntegration.ts
  → glue/orchestration/workflow-system/useBubbleLabIntegration.ts
```

### Plugin Package Imports

```typescript
// Old
import { createRAGBitsPlugin } from '@bubblelabs-ragbits-plugin';
import { createDatapizzaPlugin } from '@datapizza-bubblelab-plugin';

// New
import { createRAGBitsPlugin } from '@glue/adapters/ragbits';
import { createDatapizzaPlugin } from '@glue/adapters/datapizza';
```

## Status

✅ **Directory structure created**
✅ **All files copied to proper locations**
✅ **Bubblelab adapter structure fixed (removed nesting)**
🔄 **IN PROGRESS: Update import paths**
  - ✅ Updated WorkflowExecutionTab.tsx
  - ✅ Updated useBubbleLabIntegration.ts (both locations)
  - ✅ Updated OpenEvolveApp.tsx
  - ⏳ Need to update component imports (openevolveApi, types)
  - ⏳ Need to update glue/orchestration/workflow-system internal imports
⏳ **Next: Update package.json references**
⏳ **Next: Test compilation**

---

**Status:** REORGANIZATION IN PROGRESS
**Current:** Updating import paths throughout the codebase
