# 🚨 ARCHITECTURAL ISSUE IDENTIFIED

## Current Structure (INCORRECT)

```
Frontend/
├── core-projects/
│   └── BubbleLab/              ← READ ONLY (Core Project)
│       └── [original UI code]
│
├── bubblelab-integration-sdk/         ← ❌ VIOLATION: Should be in glue/
│   ├── src/components/
│   └── [converted UI code]
│
├── bubblelabs-ragbits-plugin/    ← ❌ VIOLATION: Should be in glue/adapters/
├── datapizza-bubblelab-plugin/    ← ❌ VIOLATION: Should be in glue/adapters/
│
└── glue/                        ← ✅ CORRECT: Glue Layer exists!
    ├── adapters/
    ├── orchestration/
    └── lib/
```

## Problem

According to the **Federation Constitution (CLAUDE.md)**:

1. **Law of Air Gap**: `./core-projects/` is effectively a third-party vendor library (READ ONLY)
2. **The Ban**: No imports targeting files inside `./core-projects/`
3. **The Glue Layer**: All adapters and integration code should be in `glue/`

**Current violations:**
- `bubblelab-integration-sdk/` exists at root level (should be in `glue/`)
- `bubblelabs-*-plugin/` folders at root (should be in `glue/adapters/`)

## Correct Architecture

```
Frontend/
├── core-projects/
│   ├── BubbleLab/              ← READ ONLY (Original)
│   ├── RAGBits/               ← READ ONLY (Original)
│   ├── Datapizza/             ← READ ONLY (Original)
│   └── [other core projects]
│
└── glue/                        ← THE GLUE LAYER
    ├── adapters/
    │   ├── bubblelab/
    │   │   ├── ui-adapter.ts          ← Adapts BubbleLab UI
    │   │   ├── api-adapter.ts         ← Adapts BubbleLab API
    │   │   └── BubbleLabIntegrationTab.tsx
    │   │
    │   ├── ragbits/
    │   │   ├── client-adapter.ts       ← RAGBits client adapter
    │   │   ├── search-adapter.ts       ← Search integration
    │   │   └── ragbitsClient.ts
    │   │
    │   └── datapizza/
    │       ├── client-adapter.ts       ← Datapizza client adapter
    │       └── datapizzaClient.ts
    │
    ├── orchestration/
    │   ├── unified-verification/   ← Existing: ✅ Correct location
    │   ├── correlation-tracker.ts
    │   ├── event-bus.ts
    │   └── ...
    │
    ├── schemas/
    │   └── [canonical data models]
    │
    └── lib/
        ├── logger.ts
        ├── retry.ts
        └── circuit-breaker.ts
```

## What Should Happen

### Option 1: Move to Glue (RECOMMENDED)

Move all integration code into the glue layer:

```bash
# Move bubblelab-integration-sdk to glue/adapters/bubblelab

  mv bubblelab-integration-sdk glue/adapters/bubblelab
# Move plugins to glue/adapters
mv bubblelabs-ragbits-plugin glue/adapters/ragbits
mv datapizza-bubblelab-plugin glue/adapters/datapizza
mv leanaide-bubblelab-plugin glue/adapters/leanaide
# etc.
```

### Option 2: Clarify Purpose

If `bubblelab-integration-sdk` is NOT the glue layer but something else, we need to:

1. **Rename it** to clarify its purpose
2. **Document why it exists** separate from core-projects
3. **Document the relationship** between core-projects/BubbleLab and bubblelab-integration-sdk

## Questions for User

1. **What IS bubblelab-integration-sdk?**
   - Is it a modified/custom version of BubbleLab?
   - Is it OpenEvolve-specific customizations on top of BubbleLab?
   - Why is it separate from core-projects/BubbleLab?

2. **What is the relationship?**
   - Does bubblelab-integration-sdk wrap/core-projects/BubbleLab?
   - Does it import from core-projects/BubbleLab?
   - Or is it a fork/modification?

3. **Should it be in glue/?**
   - If it's integration/adaptation code → YES, move to glue/adapters/
   - If it's a standalone app → Keep separate, but clarify purpose

## Immediate Action Needed

**STOP** and clarify before proceeding:

1. ✅ Keep `core-projects/BubbleLab` as READ ONLY
2. ❌ STOP using `bubblelab-integration-sdk` at root level
3. ✅ Move all integration code to `glue/`
4. ✅ Document the architecture clearly

## Recommendation

Following the Federation Constitution, the proper flow should be:

```
core-projects/BubbleLab (READ ONLY)
    ↓
glue/adapters/bubblelab/ (WRITES/ADAPTS)
    ↓
BubbleLab-converted/ui-components (PROCESSED UI)
    ↓
Frontend App (USES)
```

The conversion/adaptation logic should be IN THE GLUE LAYER, not at the root level.
