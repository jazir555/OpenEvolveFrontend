# OpenEvolve BubbleLabs Plugin - Quick Reference

**Date:** 2025-12-30
**Approach:** External Plugin (No BubbleLabs Core Modifications)

---

## What We're Building

**External npm package** that extends BubbleLabs at runtime

```
┌────────────────────────────────────────────┐
│         BubbleLabs UI                      │
│  ┌─────────┐    ┌──────────┐              │
│  │  HTTP   │───▶│OpenEvolve│───▶ Slack    │
│  │  Input  │    │  Plugin  │              │
│  └─────────┘    └──────────┘              │
└────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
    Python Backend         BubbleLabs Core
    (api_server.py)        (UNMODIFIED)
```

---

## Package Structure

```
openevolve-bubblelabs-plugin/
├── package.json              # Plugin manifest
├── tsconfig.json             # TypeScript config
├── README.md                 # Documentation
├── src/
│   ├── bubbles/
│   │   ├── openevolve.schema.ts    # 272 Zod params
│   │   ├── openevolve.utils.ts     # Helpers
│   │   ├── openevolve.ts           # Bubble class
│   │   └── index.ts                # Exports
│   ├── registration.ts              # registerOpenEvolveBubbles()
│   └── index.ts
└── build/                   # Compiled JS (generated)
```

---

## Key Files

### 1. `package.json`
```json
{
  "name": "@openevolve/bubblelabs-bubbles",
  "version": "1.0.0",
  "main": "build/index.js",
  "peerDependencies": {
    "@bubblelab/bubble-core": "^1.0.0",
    "zod": "^3.0.0"
  }
}
```

### 2. `src/registration.ts`
```typescript
export function registerOpenEvolveBubbles(factory: BubbleFactory): void {
  factory.register('openevolve', OpenEvolveBubble);
  console.log('✅ OpenEvolve bubbles registered');
}
```

### 3. `src/bubbles/openevolve.ts`
```typescript
export class OpenEvolveBubble extends ServiceBubble<T, R> {
  static readonly bubbleName = 'openevolve';
  static readonly schema = OpenEvolveParamsSchema;

  protected async performAction() {
    // Call Python API
    const response = await fetch('http://localhost:8000/api/openevolve/...');
    return await response.json();
  }
}
```

---

## Registration (How BubbleLabs Loads Plugin)

### Method 1: Direct Import
```typescript
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';

registerOpenEvolveBubbles(bubbleFactory);
```

### Method 2: Plugin Array
```typescript
const plugins = [
  () => import('@openevolve/bubblelabs-bubbles'),
  () => import('@custom/other-plugin'),
];

for (const loadPlugin of plugins) {
  const plugin = await loadPlugin();
  plugin.registerOpenEvolveBubbles(bubbleFactory);
}
```

---

## Python Backend (Add to OpenEvolve)

```python
# api_server.py
@app.post("/api/openevolve/sovereign")
async def run_sovereign_workflow_api(request: SovereignRequest):
    workflow_state = WorkflowState(
        problem_statement=request.problem_statement,
        # Map parameters...
    )
    result = await run_sovereign_workflow(workflow_state)
    return {"workflow_id": ..., "status": ..., "result": result}

@app.post("/api/openevolve/evolution")
async def run_evolution_workflow_api(request: EvolutionRequest):
    # Similar

@app.post("/api/openevolve/adversarial")
async def run_adversarial_workflow_api(request: AdversarialRequest):
    # Similar
```

---

## 3 Operations

### 1. Sovereign Decomposition
```json
{
  "operation": "sovereign",
  "problem_statement": "Solve this problem",
  "teams": {...},
  "gauntlets": {...},
  "openevolve_parameters": {...} // 272 params
}
```

### 2. Evolution
```json
{
  "operation": "evolution",
  "problem_statement": "Optimize this",
  "teams": {...},
  "evolution_settings": {...},
  "openevolve_parameters": {...}
}
```

### 3. Adversarial
```json
{
  "operation": "adversarial",
  "problem_statement": "Test this",
  "teams": {...},
  "adversarial_settings": {...},
  "openevolve_parameters": {...}
}
```

---

## 272 Parameters (19 Categories)

1. **core_evolution** (23) - evolution_mode, max_iterations, population_size
2. **model_config** (18) - model_configs, api_key, api_base
3. **quality_diversity** (19) - feature_dimensions, archive_size
4. **multi_objective** (15) - objectives, objective_weights
5. **adversarial** (20) - attack_model, defense_strategy
6. **island_model** (17) - num_islands, migration_interval
7. **selection** (18) - elite_ratio, selection_method
8. **evaluation** (25) - cascade_evaluation, ensemble_size
9. **prompt_engineering** (12) - prompt_template, system_prompt
10. **artifact_management** (10) - enable_artifacts, max_artifact_size
11. **resource_management** (11) - memory_limit_mb, cpu_limit
12. **database_storage** (10) - db_path, db_type
13. **evolution_tracing** (12) - trace_enabled, trace_level
14. **early_stopping** (9) - early_stopping, patience
15. **distributed_processing** (10) - distributed, num_workers
16. **advanced_research** (20) - novelty_search, meta_learning
17. **custom_requirements** (8) - custom_fitness, custom_operators
18. **ui_visualization** (8) - enable_visualization, plot_frequency
19. **experimental** (7) - experimental_features, beta_algorithms

---

## Installation (Local Only)

**No npm install - This is a local plugin**

```bash
# 1. Create plugin directory (in OpenEvolve-Frontend/)
mkdir openevolve-bubblelabs-plugin
cd openevolve-bubblelabs-plugin

# 2. Initialize package.json (see guide)
# 3. Create tsconfig.json (see guide)
# 4. Build plugin
npm run build

# Plugin is now available at:
# ./openevolve-bubblelabs-plugin/build/index.js
```

---

## Usage

### 1. Register Plugin
```typescript
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';

registerOpenEvolveBubbles(bubbleFactory);
```

### 2. Use in BubbleLabs UI
- Open BubbleLabs
- Drag "OpenEvolve" from sidebar
- Configure parameters
- Execute workflow

---

## Implementation Order

### Phase 1: Setup (3 tasks)
1. Create plugin directory
2. Create package.json
3. Create tsconfig.json

### Phase 2: Bubble (4 tasks)
4. Create openevolve.schema.ts (272 params)
5. Create openevolve.utils.ts
6. Create openevolve.ts (ServiceBubble)
7. Create bubbles/index.ts

### Phase 3: Registration (3 tasks)
8. Create registration.ts
9. Create types.ts
10. Create index.ts

### Phase 4: Python Backend (3 tasks)
11. Add /api/openevolve/sovereign
12. Add /api/openevolve/evolution
13. Add /api/openevolve/adversarial

### Phase 5: Integration (4 tasks)
14. Implement plugin loading
15. Test registration
16. Test compilation
17. Test end-to-end

### Phase 6: Documentation (1 task)
18. Write README

---

## Key Benefits

✅ **No BubbleLabs modifications** - External plugin
✅ **Survives updates** - Works with future versions
✅ **Easy distribution** - Publish to npm
✅ **Independent versioning** - Plugin own version
✅ **No merge conflicts** - Separate codebase
✅ **Easy testing** - Test in isolation

---

## Files to Create

**New Files (Plugin):**
- `openevolve-bubblelabs-plugin/package.json`
- `openevolve-bubblelabs-plugin/tsconfig.json`
- `openevolve-bubblelabs-plugin/src/bubbles/openevolve.schema.ts`
- `openevolve-bubblelabs-plugin/src/bubbles/openevolve.utils.ts`
- `openevolve-bubblelabs-plugin/src/bubbles/openevolve.ts`
- `openevolve-bubblelabs-plugin/src/bubbles/index.ts`
- `openevolve-bubblelabs-plugin/src/registration.ts`
- `openevolve-bubblelabs-plugin/src/types.ts`
- `openevolve-bubblelabs-plugin/src/index.ts`
- `openevolve-bubblelabs-plugin/README.md`

**Modify Files (Python Backend):**
- `api_server.py` - Add 3 endpoints

**Do NOT Modify:**
- ❌ BubbleLab/packages/bubble-core/
- ❌ BubbleLab/packages/shared-schemas/
- ❌ Any BubbleLab core files

---

## Comparison: Core Mod vs Plugin

| Aspect | Core Mod (❌) | Plugin (✅) |
|--------|--------------|-------------|
| Modifies BubbleLabs | Yes | No |
| Survives Updates | No | Yes |
| Merge Conflicts | Yes | No |
| Distribution | Fork | npm package |
| Maintenance | High | Low |
| Location | BubbleLab/core | openevolve-bubblelabs-plugin/ |

---

## Quick Start (Local)

```bash
# 1. Create plugin directory
mkdir openevolve-bubblelabs-plugin
cd openevolve-bubblelabs-plugin

# 2. Create package.json (local, no install)
# 3. Create tsconfig.json (with project references)
# 4. Create structure: mkdir -p src/bubbles
# 5. Create bubble files (openevolve.ts, etc.)
# 6. Build: npm run build
# 7. Import in BubbleLab: import from '../../openevolve-bubblelabs-plugin/build/index.js'
# 8. Register: registerOpenEvolveBubbles(bubbleFactory)
```

---

## Success Criteria

✅ Plugin compiles without errors
✅ Plugin registers with BubbleFactory
✅ "OpenEvolve" appears in BubbleLabs UI
✅ All 272 parameters configurable
✅ Python API responds correctly
✅ End-to-end execution works
✅ No BubbleLabs core modifications

---

*End of Quick Reference*
