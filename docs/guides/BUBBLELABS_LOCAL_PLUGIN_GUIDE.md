# OpenEvolve BubbleLabs Plugin - Local Development Guide

**Date:** 2025-12-30
**Approach:** Local Plugin (No npm/publishing)

---

## Overview

Create **OpenEvolve as a local BubbleLabs plugin** in the OpenEvolve-Frontend directory, without npm installation or publishing.

### Key Principle: **Local Development Only**

**No npm install** - Plugin lives alongside BubbleLab and OpenEvolve code
**No publishing** - Not distributed via npm
**Local only** - For use within this project

---

## Architecture

```
OpenEvolve-Frontend/                    (root directory)
├── openevolve-bubblelabs-plugin/       ← Local plugin (NEW)
│   ├── src/
│   │   ├── bubbles/
│   │   │   ├── openevolve.schema.ts
│   │   │   ├── openevolve.utils.ts
│   │   │   ├── openevolve.ts
│   │   │   └── index.ts
│   │   ├── registration.ts
│   │   └── index.ts
│   ├── build/                          ← Compiled JS
│   ├── tsconfig.json
│   └── package.json                    ← Local package reference
│
├── BubbleLab/                          ← External dependency (UNMODIFIED)
│   ├── packages/bubble-core/
│   └── apps/bubble-studio/
│
├── api_server.py                       ← Add API endpoints here
├── workflow_engine.py
└── ... (other OpenEvolve files)
```

---

## Package Structure

```
openevolve-bubblelabs-plugin/
├── package.json                    # Local package manifest
├── tsconfig.json                   # TypeScript config
├── src/
│   ├── bubbles/
│   │   ├── openevolve.schema.ts    # 272 parameter Zod schemas
│   │   ├── openevolve.utils.ts     # Helper functions
│   │   ├── openevolve.ts           # Main bubble class
│   │   └── index.ts                # Bubble exports
│   ├── registration.ts              # Dynamic registration logic
│   └── index.ts                    # Main export
└── build/                          # Compiled output (generated)
```

---

## package.json (Local)

```json
{
  "name": "@openevolve/bubblelabs-bubbles",
  "version": "1.0.0",
  "description": "OpenEvolve bubbles for BubbleLabs (local plugin)",
  "main": "./build/index.js",
  "types": "./build/index.d.ts",
  "scripts": {
    "build": "tsc",
    "watch": "tsc --watch",
    "clean": "rm -rf build"
  },
  "keywords": ["bubblelabs", "openevolve", "plugin", "local"],
  "devDependencies": {
    "typescript": "^5.0.0"
  },
  "peerDependencies": {
    "@bubblelab/bubble-core": "*",
    "zod": "*"
  }
}
```

**Note:** No `files` array (local only, not publishing)

---

## tsconfig.json

```json
{
  "extends": "../tsconfig.json",
  "compilerOptions": {
    "outDir": "./build",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "composite": true,
    "baseUrl": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "build", "tests"],
  "references": [
    { "path": "../BubbleLab/packages/bubble-core" }
  ]
}
```

**Note:** Uses project references to link to BubbleLab core

---

## BubbleLab Integration

### Update BubbleLab tsconfig

Add reference to local plugin in BubbleLab:

```json
// BubbleLab/tsconfig.json (or apps/bubble-studio/tsconfig.json)
{
  "references": [
    { "path": "../../packages/bubble-core" },
    { "path": "../../openevolve-bubblelabs-plugin" }  // ← Add this
  ]
}
```

### Import in BubbleLab App

```typescript
// In BubbleLab/apps/bubble-studio/src/main.tsx (or app initialization)

// Import registration function from local plugin
import { registerOpenEvolveBubbles } from '../../openevolve-bubblelabs-plugin/build/index.js';

// After bubbleFactory is initialized
registerOpenEvolveBubbles(bubbleFactory);
```

---

## Registration Function

```typescript
// openevolve-bubblelabs-plugin/src/registration.ts
import type { BubbleFactory } from '@bubblelab/bubble-core';
import { OpenEvolveBubble } from './bubbles/openevolve.js';

/**
 * Register OpenEvolve bubbles with BubbleFactory
 * Call this during BubbleLab app initialization
 */
export function registerOpenEvolveBubbles(factory: BubbleFactory): void {
  try {
    factory.register('openevolve', OpenEvolveBubble);
    console.log('✅ OpenEvolve bubbles registered (local plugin)');
  } catch (error) {
    console.error('❌ Failed to register OpenEvolve bubbles:', error);
    throw error;
  }
}
```

---

## Build Process

```bash
# In openevolve-bubblelabs-plugin directory

# Build the plugin
npm run build

# Or directly with TypeScript
tsc --project tsconfig.json

# Watch mode for development
npm run watch
```

Output goes to `build/` directory:
```
build/
├── bubbles/
│   ├── openevolve.schema.d.ts
│   ├── openevolve.schema.js
│   ├── openevolve.utils.d.ts
│   ├── openevolve.utils.js
│   ├── openevolve.d.ts
│   ├── openevolve.js
│   └── index.d.ts
│       └── index.js
├── registration.d.ts
├── registration.js
├── index.d.ts
└── index.js
```

---

## Integration in BubbleLab

### Method 1: Direct Import (Simplest)

```typescript
// BubbleLab/apps/bubble-studio/src/main.tsx

import { registerOpenEvolveBubbles } from '../../openevolve-bubblelabs-plugin/build/index.js';

// After bubbleFactory initialization
registerOpenEvolveBubbles(bubbleFactory);
```

### Method 2: Import Statement

```typescript
// In BubbleLab app initialization file

import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';

// This works if you add to tsconfig paths:
{
  "compilerOptions": {
    "paths": {
      "@openevolve/bubblelabs-bubbles": ["../../openevolve-bubblelabs-plugin/build/index"]
    }
  }
}
```

### Method 3: Dynamic Import

```typescript
// In BubbleLab app initialization

async function loadLocalPlugins() {
  try {
    const plugin = await import('../../openevolve-bubblelabs-plugin/build/index.js');
    plugin.registerOpenEvolveBubbles(bubbleFactory);
    console.log('✅ Local plugin loaded');
  } catch (error) {
    console.error('❌ Failed to load local plugin:', error);
  }
}

loadLocalPlugins();
```

---

## Python Backend (Local)

Add endpoints to existing `api_server.py`:

```python
# api_server.py - Add these 3 endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

### Phase 1: Plugin Setup (3 tasks)
1. ⏳ Create `openevolve-bubblelabs-plugin/` directory
2. ⏳ Create `package.json` (local package)
3. ⏳ Create `tsconfig.json` (with project references)

### Phase 2: Bubble Implementation (4 tasks)
4. ⏳ Create `openevolve.schema.ts` (272 parameters, 19 categories)
5. ⏳ Create `openevolve.utils.ts` (helper functions)
6. ⏳ Create `openevolve.ts` (ServiceBubble class, 3 operations)
7. ⏳ Create `bubbles/index.ts` (exports)

### Phase 3: Registration (2 tasks)
8. ⏳ Create `registration.ts` (registerOpenEvolveBubbles function)
9. ⏳ Create `index.ts` (main export)

### Phase 4: Python Backend (3 tasks)
10. ⏳ Add `POST /api/openevolve/sovereign` endpoint
11. ⏳ Add `POST /api/openevolve/evolution` endpoint
12. ⏳ Add `POST /api/openevolve/adversarial` endpoint

### Phase 5: BubbleLab Integration (3 tasks)
13. ⏳ Update BubbleLab tsconfig.json (add plugin reference)
14. ⏳ Add plugin import to BubbleLab app
15. ⏳ Call `registerOpenEvolveBubbles(bubbleFactory)`

### Phase 6: Build & Test (5 tasks)
16. ⏳ Build plugin: `npm run build` in plugin directory
17. ⏳ Test compilation: Check for TypeScript errors
18. ⏳ Test registration: Check BubbleFactory.list() includes 'openevolve'
19. ⏳ Create test BubbleFlow
20. ⏳ Test end-to-end execution

---

## Build & Test Workflow

```bash
# Terminal 1: Build plugin
cd openevolve-bubblelabs-plugin
npm run build

# Terminal 2: Start Python backend
python api_server.py

# Terminal 3: Start BubbleLab
cd BubbleLab
npm run dev

# In BubbleLab UI:
# 1. Check that "OpenEvolve" appears in bubble list
# 2. Drag "OpenEvolve" to canvas
# 3. Configure parameters
# 4. Execute workflow
```

---

## File Paths (Absolute)

```
Plugin:
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\openevolve-bubblelabs-plugin\

BubbleLab:
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLab\

Python Backend:
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\api_server.py
```

---

## Import Paths

```typescript
// In BubbleLab app (relative import)
import { registerOpenEvolveBubbles } from '../../openevolve-bubblelabs-plugin/build/index.js';

// Or with path mapping (if configured)
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';
```

---

## Quick Start

```bash
# 1. Create plugin directory
mkdir openevolve-bubblelabs-plugin
cd openevolve-bubblelabs-plugin

# 2. Initialize
cat > package.json << 'EOF'
{
  "name": "@openevolve/bubblelabs-bubbles",
  "version": "1.0.0",
  "main": "./build/index.js",
  "scripts": {
    "build": "tsc"
  }
}
EOF

# 3. Create TypeScript config
cat > tsconfig.json << 'EOF'
{
  "extends": "../tsconfig.json",
  "compilerOptions": {
    "outDir": "./build",
    "rootDir": "./src"
  }
}
EOF

# 4. Create structure
mkdir -p src/bubbles

# 5. Build
npm run build
```

---

## Key Differences from npm Approach

| Aspect | npm Package | Local Plugin |
|--------|-------------|--------------|
| **Installation** | `npm install` | Create directory |
| **Location** | node_modules/ | openevolve-bubblelabs-plugin/ |
| **Publishing** | npm publish | None (local only) |
| **Imports** | Package name | Relative path or path mapping |
| **Updates** | npm update | Rebuild locally |
| **Distribution** | Public/private npm | File system only |

---

## Benefits

✅ **No npm required** - Pure local development
✅ **No publishing** - Stays in project
✅ **Fast iteration** - Edit, build, test immediately
✅ **Full control** - No external dependencies
✅ **Easy debugging** - Source code always available
✅ **No version conflicts** - Always in sync

---

## Files to Create

### Plugin Files (9 files)
1. `openevolve-bubblelabs-plugin/package.json`
2. `openevolve-bubblelabs-plugin/tsconfig.json`
3. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.schema.ts`
4. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.utils.ts`
5. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.ts`
6. `openevolve-bubblelabs-plugin/src/bubbles/index.ts`
7. `openevolve-bubblelabs-plugin/src/registration.ts`
8. `openevolve-bubblelabs-plugin/src/index.ts`
9. `openevolve-bubblelabs-plugin/README.md`

### Python Files (Modify existing)
10. `api_server.py` - Add 3 endpoints

### BubbleLab Files (Minimal changes)
11. `BubbleLab/tsconfig.json` or `apps/bubble-studio/tsconfig.json` - Add plugin reference
12. `BubbleLab/apps/bubble-studio/src/main.tsx` - Add plugin import

---

## Common Issues

### Issue: TypeScript can't find @bubblelab/bubble-core
**Solution:** Use project references in tsconfig.json

### Issue: Can't import plugin in BubbleLab
**Solution:** Use relative path: `../../openevolve-bubblelabs-plugin/build/index.js`

### Issue: Changes not reflected
**Solution:** Rebuild plugin with `npm run build`

### Issue: BubbleFactory not found
**Solution:** Import from BubbleLab core: `import { BubbleFactory } from '@bubblelab/bubble-core'`

---

## Success Criteria

✅ Plugin builds without errors
✅ Plugin registers with BubbleFactory
✅ "OpenEvolve" appears in BubbleLabs UI
✅ All 272 parameters configurable
✅ Python API responds correctly
✅ End-to-end execution works
✅ No BubbleLabs core modifications

---

## Summary

**This is a LOCAL plugin for local development only:**

- No npm install
- No publishing
- No distribution
- Pure local file system
- Edit → Build → Test workflow
- Stays in OpenEvolve-Frontend directory

**Result:** OpenEvolve extends BubbleLabs as a local plugin without modifying BubbleLab core.

---

*End of Local Development Guide*
