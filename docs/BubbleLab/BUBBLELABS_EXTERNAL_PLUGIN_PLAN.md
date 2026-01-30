# OpenEvolve BubbleLabs Plugin - External Package Plan

**Date:** 2025-12-30
**Status:** 📋 Planning Phase (External Plugin Approach)

---

## Overview

Create **OpenEvolve as a separate BubbleLabs plugin package** that extends BubbleLabs without modifying its core code.

### Key Principle: **No Core Modifications**

**❌ WRONG:** Modify `BubbleLab/packages/bubble-core/src/bubble-factory.ts`
- Changes lost when BubbleLabs updates
- Need to maintain fork
- Merge conflicts on updates

**✅ CORRECT:** Create separate plugin package
- `@openevolve/bubblelabs-bubbles` or `openevolve-bubblelabs-plugin`
- Extends BubbleLabs at runtime
- Survives BubbleLabs updates
- No merge conflicts

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Project Structure                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OpenEvolve-Frontend/                                           │
│  ├── openevolve-bubblelabs-plugin/  ← NEW PLUGIN PACKAGE       │
│  │   ├── package.json                                          │
│  │   ├── src/                                                  │
│  │   │   ├── bubbles/                                          │
│  │   │   │   ├── openevolve.schema.ts                         │
│  │   │   │   ├── openevolve.ts                                │
│  │   │   │   └── index.ts                                     │
│  │   │   ├── registration.ts  ← Dynamic registration          │
│  │   │   └── index.ts                                         │
│  │   ├── tsconfig.json                                         │
│  │   └── build/  ← Compiled JS                                │
│  │                                                               │
│  ├── BubbleLab/  ← UNMODIFIED (external dependency)           │
│  │   ├── packages/bubble-core/                                 │
│  │   └── apps/bubble-studio/                                   │
│  │                                                               │
│  └── ... (other OpenEvolve files)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
openevolve-bubblelabs-plugin/
├── package.json                    # Plugin manifest
├── tsconfig.json                   # TypeScript config
├── README.md                       # Plugin documentation
├── src/
│   ├── bubbles/
│   │   ├── openevolve.schema.ts    # 272 parameter Zod schemas
│   │   ├── openevolve.utils.ts     # Helper functions
│   │   ├── openevolve.ts           # Main bubble class
│   │   └── index.ts                # Bubble exports
│   ├── registration.ts              # Dynamic registration logic
│   ├── types.ts                    # Shared types
│   └── index.ts                    # Main export
├── build/                          # Compiled output
└── tests/
    └── openevolve.test.ts
```

---

## File Contents

### `package.json`

```json
{
  "name": "@openevolve/bubblelabs-bubbles",
  "version": "1.0.0",
  "description": "OpenEvolve bubbles for BubbleLabs visual workflow builder",
  "main": "build/index.js",
  "types": "build/index.d.ts",
  "scripts": {
    "build": "tsc",
    "watch": "tsc --watch",
    "test": "vitest",
    "prepublishOnly": "npm run build"
  },
  "keywords": [
    "bubblelabs",
    "bubble",
    "openevolve",
    "evolutionary-computing",
    "workflow",
    "plugin"
  ],
  "peerDependencies": {
    "@bubblelab/bubble-core": "^1.0.0",
    "zod": "^3.0.0"
  },
  "devDependencies": {
    "@bubblelab/bubble-core": "^1.0.0",
    "typescript": "^5.0.0",
    "zod": "^3.0.0",
    "vitest": "^1.0.0"
  },
  "files": [
    "build",
    "README.md",
    "package.json"
  ]
}
```

### `tsconfig.json`

```json
{
  "extends": "../tsconfig.json",
  "compilerOptions": {
    "outDir": "./build",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "composite": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "build", "tests"]
}
```

---

## Registration Strategy

### `src/registration.ts`

```typescript
import type { BubbleFactory } from '@bubblelab/bubble-core';
import type { BubbleClassWithMetadata } from '@bubblelab/bubble-core';
import { OpenEvolveBubble } from './bubbles/openevolve.js';

/**
 * Register OpenEvolve bubbles with BubbleFactory
 *
 * Call this during app initialization to load OpenEvolve plugin
 *
 * @example
 * ```typescript
 * import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';
 *
 * registerOpenEvolveBubbles(bubbleFactory);
 * ```
 */
export function registerOpenEvolveBubbles(factory: BubbleFactory): void {
  try {
    // Register the main OpenEvolve bubble
    factory.register('openevolve', OpenEvolveBubble as BubbleClassWithMetadata);

    console.log('✅ OpenEvolve bubbles registered successfully');
    console.log('   - openevolve (sovereign, evolution, adversarial)');
  } catch (error) {
    console.error('❌ Failed to register OpenEvolve bubbles:', error);
    throw error;
  }
}

/**
 * Get list of bubbles provided by this plugin
 */
export function getProvidedBubbles(): string[] {
  return ['openevolve'];
}

/**
 * Get plugin metadata
 */
export function getPluginMetadata() {
  return {
    name: '@openevolve/bubblelabs-bubbles',
    version: '1.0.0',
    description: 'OpenEvolve evolutionary computing workflows',
    bubbles: ['openevolve'],
    operations: {
      openevolve: ['sovereign', 'evolution', 'adversarial']
    },
    parameters: {
      total: 272,
      categories: 19
    }
  };
}
```

---

## Integration Methods

### Method 1: Direct Import (Simplest)

```typescript
// In BubbleLab app initialization
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';

// After BubbleFactory is initialized
registerOpenEvolveBubbles(bubbleFactory);
```

### Method 2: Plugin Array

```typescript
// Load multiple plugins
const plugins = [
  () => import('@openevolve/bubblelabs-bubbles'),
  () => import('@custom/other-bubbles'),
];

for (const loadPlugin of plugins) {
  const plugin = await loadPlugin();
  plugin.registerOpenEvolveBubbles(bubbleFactory);
}
```

### Method 3: Config-Based

```typescript
// bubble.config.json
{
  "plugins": ["@openevolve/bubblelabs-bubbles"]
}

// Load from config
import config from './bubble.config.json';

for (const pluginName of config.plugins) {
  const plugin = await import(pluginName);
  plugin.registerOpenEvolveBubbles(bubbleFactory);
}
```

---

## Python Backend (No BubbleLabs Changes)

Add API endpoints to existing OpenEvolve backend:

```python
# api_server.py - Add these endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow_engine import run_sovereign_workflow

class SovereignRequest(BaseModel):
    problem_statement: str
    team_config: dict = {}
    gauntlet_config: dict = {}
    openevolve_parameters: dict = {}

@app.post("/api/openevolve/sovereign")
async def run_sovereign_workflow_api(request: SovereignRequest):
    """Execute Sovereign Decomposition workflow"""
    try:
        from workflow_structures import WorkflowState

        workflow_state = WorkflowState(
            problem_statement=request.problem_statement,
            # Map parameters...
        )

        result = await run_sovereign_workflow(
            workflow_state=workflow_state,
            team_manager=team_manager,
            gauntlet_manager=gauntlet_manager,
        )

        return {
            "workflow_id": str(workflow_state.workflow_id),
            "status": "completed",
            "result": result,
            "performance_metrics": {
                "total_time_seconds": workflow_state.execution_time,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openevolve/evolution")
async def run_evolution_workflow_api(request: EvolutionRequest):
    """Execute Evolution workflow"""
    pass

@app.post("/api/openevolve/adversarial")
async def run_adversarial_workflow_api(request: AdversarialRequest):
    """Execute Adversarial Testing workflow"""
    pass
```

---

## Implementation Steps

### Phase 1: Plugin Package Setup
1. ✅ Create planning documentation
2. ⏳ Create `openevolve-bubblelabs-plugin/` directory
3. ⏳ Create `package.json`
4. ⏳ Create `tsconfig.json`
5. ⏳ Create folder structure

### Phase 2: Bubble Implementation
6. ⏳ Create `openevolve.schema.ts` (272 parameters, 19 categories)
7. ⏳ Create `openevolve.utils.ts` (helper functions)
8. ⏳ Create `openevolve.ts` (ServiceBubble class, 3 operations)
9. ⏳ Create `bubbles/index.ts` (bubble exports)

### Phase 3: Registration System
10. ⏳ Create `registration.ts` (registerOpenEvolveBubbles function)
11. ⏳ Create `types.ts` (shared types)
12. ⏳ Create `index.ts` (main export)

### Phase 4: Python Backend
13. ⏳ Add `/api/openevolve/sovereign` endpoint
14. ⏳ Add `/api/openevolve/evolution` endpoint
15. ⏳ Add `/api/openevolve/adversarial` endpoint
16. ⏳ Test with curl/Postman

### Phase 5: Integration
17. ⏳ Decide integration method (import/config/loader)
18. ⏳ Implement plugin loading in BubbleLab app
19. ⏳ Test bubble registration
20. ⏳ Test compilation
21. ⏳ Create test BubbleFlow

### Phase 6: Testing & Documentation
22. ⏳ Create unit tests
23. ⏳ Create integration tests
24. ⏳ Test end-to-end execution
25. ⏳ Write plugin README
26. ⏳ Create usage examples

---

## Updated Todo List

1. ⏳ Create plugin directory structure
2. ⏳ Create package.json with peerDependencies
3. ⏳ Create tsconfig.json
4. ⏳ Create openevolve.schema.ts (all 272 parameters)
5. ⏳ Create openevolve.utils.ts
6. ⏳ Create openevolve.ts bubble class
7. ⏳ Create bubbles/index.ts
8. ⏳ Create registration.ts
9. ⏳ Create types.ts
10. ⏳ Create main index.ts
11. ⏳ Add Python API endpoints
12. ⏳ Implement plugin loading
13. ⏳ Test registration
14. ⏳ Test compilation
15. ⏳ Test end-to-end
16. ⏳ Write README
17. ⏳ Create examples

---

## Benefits

✅ **No Core Modifications** - BubbleLabs untouched
✅ **Survives Updates** - Works with future versions
✅ **Easy Distribution** - Publish as npm package
✅ **Independent Versioning** - Plugin own version
✅ **Clean Separation** - Modular architecture
✅ **Easy Testing** - Test in isolation
✅ **No Merge Conflicts** - Separate codebase

---

## Comparison

| Aspect | Core Mod (❌) | Plugin (✅) |
|--------|--------------|-------------|
| Modifies BubbleLabs | Yes | No |
| Survives Updates | No | Yes |
| Merge Conflicts | Yes | No |
| Distribution | Fork | npm package |
| Maintenance | High | Low |
| Testing | Integrated | Isolated |
| Versioning | Coupled | Independent |

---

## Installation

```bash
# Install in BubbleLab
cd BubbleLab
npm install @openevolve/bubblelabs-bubbles

# Or link local for development
cd openevolve-bubblelabs-plugin
npm link

cd BubbleLab
npm link @openevolve/bubblelabs-bubbles
```

## Usage

```typescript
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';

// Initialize
registerOpenEvolveBubbles(bubbleFactory);

// Use in BubbleLabs UI
// 1. Drag "OpenEvolve" bubble to canvas
// 2. Select operation (sovereign/evolution/adversarial)
// 3. Configure parameters
// 4. Execute workflow
```

---

## Summary

**Key Changes:**
1. ✅ Separate plugin package - Not in BubbleLab core
2. ✅ Dynamic registration - Register at runtime
3. ✅ npm distribution - Can publish independently
4. ✅ No modifications - BubbleLabs stays clean

**What Stays Same:**
- ServiceBubble implementation
- Zod schemas (272 parameters)
- Three operations
- Python API endpoints

**Result:**
OpenEvolve extends BubbleLabs as a plugin that can be installed, updated, and maintained independently.

---

*End of External Plugin Plan*
