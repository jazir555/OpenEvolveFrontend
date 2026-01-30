# 🔧 OpenEvolve Plugin Extraction & Unification Task

**Task ID**: OP-EXTRACT-001
**Priority**: HIGH
**Status**: PENDING
**Created**: 2026-01-06

---

## 🎯 Mission Objective

Extract and unify the OpenEvolve plugin implementation so that:
1. ✅ **`OpenEvolve-Plugin/`** becomes the single authoritative standalone plugin
2. ✅ All OpenEvolve code is **removed from BubbleLab core** (`BubbleLab/apps/bubble-studio/src/plugins/openevolve/`)
3. ✅ BubbleLab can import OpenEvolve as an **external dependency**
4. ✅ BubbleLab remains updateable from upstream without merge conflicts

---

## 📋 Context

### The Problem

OpenEvolve plugin code currently exists in TWO places:
1. **`OpenEvolve-Plugin/`** - Standalone plugin directory ✅ (Correct)
2. **`BubbleLab/apps/bubble-studio/src/plugins/openevolve/`** - Embedded in BubbleLab core ❌ (Violates AIR GAP principle)

### The Solution

Following the **"AIR GAP"** law from CLAUDE.md:
- Core projects should be **READ-ONLY** and **IMMUTABLE**
- No OpenEvolve code should exist inside `BubbleLab/`
- BubbleLab should import OpenEvolve as an **external plugin**

---

## 🗂️ Current State Analysis

### Location 1: Standalone Plugin ✅
**Path**: `OpenEvolve-Plugin/`

**Contains**:
```
OpenEvolve-Plugin/
├── src/
│   ├── components/          # 26 React components
│   │   ├── analytics/       # MetricCard, PerformanceChart, etc.
│   │   ├── knowledge/       # ArtifactList, KnowledgeSearch, etc.
│   │   ├── leanaide/        # ProofEditor, VerificationDisplay, etc.
│   │   ├── pages/           # Dashboard pages
│   │   ├── shared/          # Shared UI components
│   │   └── workflow/        # Workflow-specific components
│   ├── services/            # API clients and hooks
│   │   ├── api/             # API client implementations
│   │   └── hooks/           # React hooks
│   ├── stores/              # Zustand state stores
│   ├── schemas/             # Zod validation schemas
│   ├── types/               # TypeScript definitions
│   ├── utils/               # Utility functions
│   └── assets/              # Icons and images
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

**Status**: ✅ This is the CORRECT location

### Location 2: Embedded in BubbleLab ❌
**Path**: `BubbleLab/apps/bubble-studio/src/plugins/openevolve/`

**Contains**:
```
BubbleLab/apps/bubble-studio/src/plugins/openevolve/
├── plugin.ts                # Main plugin definition
└── schemas/                 # Zod schemas
    ├── evolution.ts
    ├── adversarial.ts
    ├── maker.ts
    ├── mdap.ts
    ├── decomposition.ts
    ├── knowledge.ts
    ├── leanaide.ts
    ├── hephaestus.ts
    ├── roma.ts
    └── invention.ts
```

**Status**: ❌ This VIOLATES the AIR GAP principle - must be removed

---

## 🔄 Phase-by-Phase Extraction Plan

### Phase 1: Inventory & Comparison (Agent 1)

**Objective**: Complete audit of both locations

**Tasks**:
1. **Schema Comparison**
   - Compare schemas in `OpenEvolve-Plugin/src/schemas/` vs `BubbleLab/.../schemas/`
   - Identify differences, conflicts, and missing schemas
   - Document which schema is more complete/accurate

2. **Feature Audit**
   - Catalog all features in standalone plugin
   - Check if embedded plugin has any unique features not in standalone
   - Identify any BubbleLab-specific customizations

3. **Dependency Analysis**
   - Check imports and dependencies in embedded plugin
   - Identify any BubbleLab-specific dependencies
   - Document any hardcoded paths or references

4. **Integration Points**
   - Find how BubbleLab currently imports the embedded plugin
   - Document the plugin registration mechanism
   - Identify configuration requirements

**Deliverables**:
- `INVENTORY_REPORT.md` - Complete feature comparison
- `SCHEMA_DIFF.md` - Schema differences analysis
- `DEPENDENCY_ANALYSIS.md` - Dependency mapping
- `INTEGRATION_POINTS.md` - How BubbleLab uses the plugin

---

### Phase 2: Unify & Complete Standalone Plugin (Agent 2)

**Objective**: Ensure standalone plugin has ALL features

**Tasks**:
1. **Merge Schemas**
   - Take the best schema from each location
   - Ensure all 10 workflow types have complete schemas
   - Add any missing schemas to standalone plugin

2. **Feature Completeness**
   - Add any features from embedded plugin not in standalone
   - Ensure all API endpoints are covered
   - Add any missing hooks or services

3. **Plugin Definition**
   - Ensure standalone plugin exports proper `PluginDefinition`
   - Include all service definitions
   - Add lifecycle hooks if present in embedded version

4. **Export Structure**
   - Ensure clean export structure
   - Export plugin definition, schemas, components
   - Provide TypeScript types for all exports

**Deliverables**:
- Updated `OpenEvolve-Plugin/` with all features
- Complete schema set
- Proper `PluginDefinition` export
- Feature completeness verified

---

### Phase 3: Configure External Import (Agent 3)

**Objective**: Make standalone plugin importable by BubbleLab

**Tasks**:
1. **Package Configuration**
   - Update `package.json` with proper name/version
   - Configure exports (main, module, types)
   - Set up proper build output

2. **Build Configuration**
   - Ensure TypeScript compilation is correct
   - Configure Vite for library mode
   - Generate proper `.d.ts` type definition files

3. **Distribution Structure**
   - Set up `dist/` directory structure
   - Ensure all assets are included
   - Test npm pack/install locally

4. **Import Path Configuration**
   - Set up path aliases if needed
   - Configure for both ESM and CJS
   - Ensure compatibility with BubbleLab's build system

**Deliverables**:
- Buildable standalone plugin
- Installable npm package
- Proper TypeScript definitions
- Import/test in external project

---

### Phase 4: Update BubbleLab Integration (Agent 4)

**Objective**: Modify BubbleLab to use external plugin

**Tasks**:
1. **Install External Plugin**
   - Add `@openevolve/plugin` to BubbleLab's `package.json`
   - Configure import paths
   - Set up any required environment variables

2. **Update Imports**
   - Find all imports of embedded plugin
   - Replace with external plugin imports
   - Update component registrations

3. **Plugin Registration**
   - Update BubbleLab's plugin loader
   - Register OpenEvolve as external plugin
   - Test plugin discovery and loading

4. **Configuration Migration**
   - Move any BubbleLab-specific configs to standalone plugin
   - Set up environment-based configuration
   - Document configuration requirements

**Deliverables**:
- BubbleLab imports external plugin
- Plugin loads correctly
- All functionality working
- Configuration documented

---

### Phase 5: Remove Embedded Plugin (Agent 5)

**Objective**: Clean up OpenEvolve code from BubbleLab core

**Tasks**:
1. **Remove Plugin Directory**
   - Delete `BubbleLab/apps/bubble-studio/src/plugins/openevolve/`
   - Verify no files remain

2. **Update References**
   - Search for any remaining references to embedded plugin
   - Update all imports to use external plugin
   - Clean up any lingering configuration

3. **Verify Clean State**
   - Ensure no OpenEvolve code in BubbleLab core
   - Check git status shows only intended changes
   - Verify BubbleLab still builds successfully

4. **Documentation**
   - Update BubbleLab documentation
   - Note OpenEvolve as external dependency
   - Add installation instructions

**Deliverables**:
- BubbleLab core free of OpenEvolve code
- All imports updated
- BubbleLab builds and runs correctly
- Documentation updated

---

### Phase 6: Validation & Testing (Agent 6)

**Objective**: Complete end-to-end validation

**Tasks**:
1. **Build Verification**
   - Build standalone plugin: `cd OpenEvolve-Plugin && npm run build`
   - Build BubbleLab: `cd BubbleLab && npm run build`
   - Verify both build successfully

2. **Integration Testing**
   - Test all OpenEvolve services load correctly
   - Test all workflow schemas work
   - Test all UI components render
   - Test API integration

3. **Feature Validation**
   - Test evolution workflows
   - Test adversarial workflows
   - Test all 10 workflow types
   - Test LeanAide integration
   - Test knowledge base
   - Test analytics

4. **Upstream Compatibility**
   - Verify BubbleLab can update from upstream
   - Test no merge conflicts would occur
   - Verify OpenEvolve plugin doesn't block BubbleLab updates

**Deliverables**:
- All tests passing
- Build successful
- Full feature validation
- Upstream compatibility verified
- Final release report

---

## 📊 Success Criteria

### Must Have
- ✅ Zero OpenEvolve code in BubbleLab core
- ✅ Standalone plugin contains ALL features
- ✅ BubbleLab imports OpenEvolve externally
- ✅ All functionality working
- ✅ BubbleLab can update from upstream
- ✅ Build successful for both projects

### Should Have
- ✅ Clean separation of concerns
- ✅ Proper TypeScript types
- ✅ Good documentation
- ✅ Easy installation

### Nice to Have
- ✅ Monorepo support
- ✅ Version locking strategy
- ✅ Automated testing between projects

---

## 🚨 Critical Constraints

### Must NOT
- ❌ Leave any OpenEvolve code in BubbleLab core
- ❌ Break BubbleLab's ability to update from upstream
- ❌ Lose any features during migration
- ❌ Create circular dependencies

### Must DO
- ✅ Follow AIR GAP principle
- ✅ Maintain all functionality
- ✅ Keep BubbleLab core pristine
- ✅ Document all changes

---

## 📁 File Structure After Completion

### Standalone Plugin (Authoritative)
```
OpenEvolve-Plugin/                    # ✅ Keep and enhance
├── src/
│   ├── components/                   # All UI components
│   ├── services/                     # API and hooks
│   ├── stores/                       # State management
│   ├── schemas/                      # All 10 workflow schemas
│   ├── types/                        # TypeScript types
│   ├── utils/                        # Utilities
│   ├── assets/                       # Icons, images
│   └── index.ts                      # Main export
├── package.json                      # Properly configured
├── tsconfig.json
├── vite.config.ts
└── README.md                         # Complete documentation
```

### BubbleLab Core (Clean)
```
BubbleLab/                            # ✅ Upstream-compatible
├── apps/
│   └── bubble-studio/
│       ├── package.json              # ✅ Has @openevolve/plugin as dependency
│       ├── src/
│       │   ├── plugins/
│       │   │   └── index.ts          # ✅ Imports OpenEvolve externally
│       │   └── ...
│       └── ...
└── ...
```

### Removed
```
❌ BubbleLab/apps/bubble-studio/src/plugins/openevolve/   # DELETE THIS
```

---

## 🔄 Import Flow After Migration

### Before (Embedded - Wrong)
```typescript
// BubbleLab imports from internal directory
import { OpenEvolvePlugin } from '@/plugins/openevolve/plugin';
import { evolutionSchema } from '@/plugins/openevolve/schemas/evolution';
```

### After (External - Correct)
```typescript
// BubbleLab imports from external package
import { OpenEvolvePlugin, evolutionSchema } from '@openevolve/plugin';

// Or if using local monorepo
import { OpenEvolvePlugin, evolutionSchema } from 'OpenEvolve-Plugin';
```

---

## 📝 Agent Instructions

### For All Agents
1. Read this entire task document first
2. Understand the AIR GAP principle from CLAUDE.md
3. Document everything you do
4. Test your changes before marking complete
5. Communicate blockers immediately

### Handoff Protocol
- Complete your phase deliverables
- Verify against success criteria
- Document issues encountered
- Create clear handoff for next phase

---

## 🎯 Quick Start for Phase 1 Agent

```bash
# Navigate to project
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# 1. Compare schemas
diff -r OpenEvolve-Plugin/src/schemas/ BubbleLab/apps/bubble-studio/src/plugins/openevolve/schemas/

# 2. Find all imports of embedded plugin in BubbleLab
cd BubbleLab
grep -r "plugins/openevolve" apps/bubble-studio/src/

# 3. Check plugin registration
grep -r "OpenEvolve" apps/bubble-studio/src/

# 4. Create inventory report
# Document all findings
```

---

## 📊 Progress Tracking

| Phase | Agent | Status | Progress | Notes |
|-------|-------|--------|----------|-------|
| 1. Inventory | Agent 1 | ⏳ Pending | 0% | Not started |
| 2. Unify | Agent 2 | ⏳ Pending | 0% | Not started |
| 3. Configure | Agent 3 | ⏳ Pending | 0% | Not started |
| 4. Update BubbleLab | Agent 4 | ⏳ Pending | 0% | Not started |
| 5. Remove Embedded | Agent 5 | ⏳ Pending | 0% | Not started |
| 6. Validation | Agent 6 | ⏳ Pending | 0% | Not started |

---

## 🔗 Related Documentation

- [CLAUDE.md - AIR GAP Law](./CLAUDE.md)
- [OpenEvolve-Plugin README](./OpenEvolve-Plugin/README.md)
- [BubbleLab Integration](./BubbleLab/apps/bubble-studio/)

---

**END OF TASK DOCUMENTATION**

*Remember: The goal is CLEAN SEPARATION following the AIR GAP principle. BubbleLab core must remain pristine and updateable from upstream.*
