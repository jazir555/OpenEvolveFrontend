# OpenEvolve BubbleLabs Integration - Plan Summary

**Date:** 2025-12-30
**Status:** ✅ Documentation Complete - Ready to Implement

---

## What Changed

### Original Plan (WRONG)
- Modify `BubbleLab/packages/bubble-core/src/bubble-factory.ts`
- Modify `BubbleLab/packages/shared-schemas/src/types.ts`
- Add to core BubbleLab exports
- **Problem:** Changes lost when BubbleLabs updates

### Corrected Plan (RIGHT)
- Create separate plugin package: `openevolve-bubblelabs-plugin/`
- Register bubbles at runtime via function call
- No modifications to BubbleLab core
- **Benefit:** Survives updates, no merge conflicts

---

## Created Documentation

### 1. **BUBBLELABS_EXTERNAL_PLUGIN_PLAN.md** (Full Plan)
Complete implementation plan with:
- Architecture diagrams
- Package structure
- File contents (package.json, tsconfig.json, registration.ts)
- Integration methods (3 options)
- Python backend endpoints
- 18 implementation steps
- Benefits comparison

### 2. **BUBBLELABS_PLUGIN_QUICK_REF.md** (Quick Reference)
Quick reference with:
- Package structure overview
- Key file examples
- Registration methods
- 3 operations explained
- 272 parameters (19 categories)
- Installation instructions
- Implementation order (18 tasks)
- Success criteria

### 3. **Todo List** (14 Tasks)
Organized implementation tasks:
- Phase 1: Setup (3 tasks)
- Phase 2: Bubble Implementation (4 tasks)
- Phase 3: Registration (3 tasks)
- Phase 4: Python Backend (3 tasks)
- Phase 5: Integration (4 tasks)
- Phase 6: Testing & Docs (2 tasks)

---

## Package Structure

```
OpenEvolve-Frontend/
├── openevolve-bubblelabs-plugin/     ← NEW PLUGIN PACKAGE
│   ├── package.json
│   ├── tsconfig.json
│   ├── README.md
│   ├── src/
│   │   ├── bubbles/
│   │   │   ├── openevolve.schema.ts
│   │   │   ├── openevolve.utils.ts
│   │   │   ├── openevolve.ts
│   │   │   └── index.ts
│   │   ├── registration.ts
│   │   └── index.ts
│   └── build/                        (generated)
│
├── BubbleLab/                        ← UNMODIFIED
│   ├── packages/bubble-core/         (don't touch!)
│   └── apps/bubble-studio/
│
├── api_server.py                     (add endpoints here)
└── workflow_engine.py                (existing)
```

---

## Key Concept: Dynamic Registration

### Traditional (Wrong)
```typescript
// In BubbleLab core - DON'T DO THIS
// bubble-factory.ts
this.register('openevolve', OpenEvolveBubble);
```

### Plugin Approach (Right)
```typescript
// In separate plugin package
// openevolve-bubblelabs-plugin/src/registration.ts
export function registerOpenEvolveBubbles(factory: BubbleFactory) {
  factory.register('openevolve', OpenEvolveBubble);
}

// In BubbleLab app (only place BubbleLabs is modified)
import { registerOpenEvolveBubbles } from '@openevolve/bubblelabs-bubbles';
registerOpenEvolveBubbles(bubbleFactory);
```

---

## Registration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Install Plugin                                         │
├─────────────────────────────────────────────────────────────────┤
│ npm install @openevolve/bubblelabs-bubbles                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Import Registration Function                            │
├─────────────────────────────────────────────────────────────────┤
│ import { registerOpenEvolveBubbles } from                      │
│   '@openevolve/bubblelabs-bubbles';                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Call Registration Function                              │
├─────────────────────────────────────────────────────────────────┤
│ registerOpenEvolveBubbles(bubbleFactory);                      │
│                                                                 │
│ Inside plugin:                                                  │
│   factory.register('openevolve', OpenEvolveBubble);            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Bubble Available in UI                                  │
├─────────────────────────────────────────────────────────────────┤
│ - "OpenEvolve" appears in BubbleLabs sidebar                    │
│ - All 272 parameters configurable                               │
│ - 3 operations: sovereign, evolution, adversarial              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

### Plugin Files (9 new files)
1. `openevolve-bubblelabs-plugin/package.json`
2. `openevolve-bubblelabs-plugin/tsconfig.json`
3. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.schema.ts`
4. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.utils.ts`
5. `openevolve-bubblelabs-plugin/src/bubbles/openevolve.ts`
6. `openevolve-bubblelabs-plugin/src/bubbles/index.ts`
7. `openevolve-bubblelabs-plugin/src/registration.ts`
8. `openevolve-bubblelabs-plugin/src/types.ts`
9. `openevolve-bubblelabs-plugin/src/index.ts`

### Python Files (Add to existing)
10. `api_server.py` - Add 3 endpoints:
    - `POST /api/openevolve/sovereign`
    - `POST /api/openevolve/evolution`
    - `POST /api/openevolve/adversarial`

### BubbleLab Files (DO NOT MODIFY)
- ❌ Don't touch `BubbleLab/packages/bubble-core/`
- ❌ Don't touch `BubbleLab/packages/shared-schemas/`
- ✅ Only add plugin loading to `BubbleLab/apps/bubble-studio/` (1-time setup)

---

## Implementation Tasks

### Phase 1: Setup (3 tasks)
- [ ] Create `openevolve-bubblelabs-plugin/` directory
- [ ] Create `package.json` with peerDependencies
- [ ] Create `tsconfig.json`

### Phase 2: Bubble Implementation (4 tasks)
- [ ] Create `openevolve.schema.ts` (272 params, 19 categories)
- [ ] Create `openevolve.utils.ts` (helper functions)
- [ ] Create `openevolve.ts` (ServiceBubble class)
- [ ] Create `bubbles/index.ts` (exports)

### Phase 3: Registration (3 tasks)
- [ ] Create `registration.ts` (registerOpenEvolveBubbles)
- [ ] Create `types.ts` (shared types)
- [ ] Create `index.ts` (main export)

### Phase 4: Python Backend (3 tasks)
- [ ] Add `POST /api/openevolve/sovereign` endpoint
- [ ] Add `POST /api/openevolve/evolution` endpoint
- [ ] Add `POST /api/openevolve/adversarial` endpoint

### Phase 5: Integration (4 tasks)
- [ ] Implement plugin loading in BubbleLab app
- [ ] Test plugin registration
- [ ] Test compilation (`tsc --build`)
- [ ] Create test BubbleFlow

### Phase 6: Testing & Documentation (2 tasks)
- [ ] Test end-to-end execution
- [ ] Write plugin README

---

## Benefits Summary

| Benefit | Explanation |
|---------|-------------|
| **No Core Mods** | BubbleLabs stays pristine |
| **Survives Updates** | Works with future versions |
| **Easy Distribution** | Publish as npm package |
| **Independent Versioning** | Plugin own version |
| **No Merge Conflicts** | Separate codebase |
| **Easy Testing** | Test in isolation |
| **Modular** | Clean separation |

---

## Quick Comparison

| Aspect | Core Mod (❌) | Plugin (✅) |
|--------|--------------|-------------|
| Modifies BubbleLabs | Yes | No |
| Survives Updates | No | Yes |
| Merge Conflicts | Yes | No |
| Distribution | Fork | npm |
| Maintenance | High | Low |
| Location | BubbleLab/core | openevolve-bubblelabs-plugin/ |

---

## Next Steps

### Ready to Implement?

Start with:

```bash
# 1. Create plugin directory
mkdir openevolve-bubblelabs-plugin
cd openevolve-bubblelabs-plugin

# 2. Initialize package
npm init -y

# 3. Create structure
mkdir -p src/bubbles

# 4. Start implementing
# - Create package.json (see BUBBLELABS_PLUGIN_QUICK_REF.md)
# - Create openevolve.schema.ts (all 272 parameters)
# - Create openevolve.ts (ServiceBubble class)
# - Create registration.ts (registerOpenEvolveBubbles)
```

### Need More Info?

See:
- `BUBBLELABS_EXTERNAL_PLUGIN_PLAN.md` - Full implementation plan
- `BUBBLELABS_PLUGIN_QUICK_REF.md` - Quick reference guide
- `BubbleLab/packages/bubble-core/CREATE_BUBBLE_README.md` - Bubble creation patterns
- `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/hello-world.ts` - Example bubble

---

## Status

✅ **Documentation Complete** - All plans written
✅ **Todo List Created** - 18 tasks organized
⏳ **Ready to Implement** - Waiting to start

**Approach:** External npm package plugin (no BubbleLabs core modifications)

---

*End of Summary*
