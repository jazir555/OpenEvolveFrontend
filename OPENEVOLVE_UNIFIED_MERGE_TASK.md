# 🚀 OpenEvolve Plugin - Grand Unification Task

**Task ID**: OP-UNIFY-001
**Priority**: CRITICAL
**Status**: PENDING
**Created**: 2026-01-06

---

## 🎯 Mission Objective

Merge **all three** OpenEvolve plugin implementations into ONE unified standalone plugin while **retaining ALL unique functionality** from each:

### Source Plugins (3 locations)
1. **`OpenEvolve-Plugin/`** - Full-featured plugin with UI, services, stores
2. **`openevolve-bubblelab-plugin/`** - Node-based plugin with registry system
3. **`BubbleLab/apps/bubble-studio/src/plugins/openevolve/`** - Embedded schemas ❌

### Target
**`OpenEvolve-Plugin/`** - The unified, standalone plugin with ALL features

### Requirements
- ✅ **ZERO feature loss** - Keep all unique functionality from all three
- ✅ **AIR GAP compliance** - Remove all OpenEvolve code from BubbleLab core
- ✅ **BubbleLab can update** - No embedded plugin code in core
- ✅ **Single source of truth** - One canonical plugin implementation

---

## 📊 Source Plugin Analysis

### Plugin 1: OpenEvolve-Plugin/

**Location**: `OpenEvolve-Plugin/`

**Strengths & Unique Features**:
```
✅ Complete UI Component Library (26 components)
   ├── Analytics: MetricCard, PerformanceChart, ArtifactTable, StatGrid
   ├── Knowledge: ArtifactList, KnowledgeSearch, ArtifactEditor, ArtifactDetail
   ├── LeanAide: ProofEditor, ModelSelector, VerificationDisplay, ProgressTracker
   ├── Pages: OpenEvolveDashboard, AnalyticsDashboard, WorkflowBuilder, etc.
   ├── Shared: ProgressBar, LiveLogViewer, FormWrapper, StatusBadge
   └── Workflow: ConfigPanel, ExecutionMonitor, etc.

✅ Services Layer
   ├── API clients for all workflows
   ├── React hooks (useEvolution, useLeanAide, etc.)
   └── WebSocket support for real-time updates

✅ State Management
   ├── Zustand stores (useAuthStore, useWorkflowStore, etc.)
   └── Centralized state management

✅ Schemas (10 workflow types)
   └── Complete Zod validation schemas

✅ TypeScript Types
   └── Comprehensive type definitions

✅ Assets
   └── Icons, images, branding
```

**Files**: ~100+ TypeScript/React files
**Lines of Code**: ~5,000+
**Package**: Proper npm package structure

---

### Plugin 2: openevolve-bubblelab-plugin/

**Location**: `openevolve-bubblelab-plugin/`

**Strengths & Unique Features**:
```
✅ Node System Architecture
   ├── BaseNode abstract class
   ├── OpenEvolveBaseNode (extends BaseNode)
   ├── DecompositionNode
   ├── SolutionNode
   └── VerificationNode

✅ Node Registry
   └── Dynamic node registration and discovery

✅ Enhanced Configuration System
   ├── EnhancedOpenEvolveConfigPanel
   ├── OpenEvolveConfigPanel
   ├── PerformanceConfigTab
   ├── SecurityConfigTab
   └── RemainingTabs (covers all other configs)

✅ Plugin Creation Utilities
   ├── createOpenEvolvePlugin()
   ├── createEnhancedOpenEvolvePlugin()
   ├── EnhancedOpenEvolvePlugin factory
   └── Plugin lifecycle management

✅ Advanced Type System
   ├── plugin-types.ts
   ├── enhanced-plugin-types.ts
   ├── extended-plugin-types.ts
   └── nodeTypes.ts (for React Flow)

✅ React Hooks
   └── useEnhancedOpenEvolveConfig

✅ Advanced Utilities
   └── advancedUtilities.ts
```

**Files**: ~30 TypeScript/React files
**Lines of Code**: ~2,000+
**Package**: Node-based workflow system

---

### Plugin 3: Embedded in BubbleLab

**Location**: `BubbleLab/apps/bubble-studio/src/plugins/openevolve/`

**Strengths & Unique Features**:
```
✅ PluginDefinition Export
   └── Official BubbleLab plugin interface

✅ Service Definitions (10 services)
   ├── evolution
   ├── adversarial
   ├── maker
   ├── mdap
   ├── decomposition
   ├── knowledge
   ├── leanaide
   ├── hephaestus
   ├── roma
   └── invention

✅ Schema References
   └── Points to schema files for each workflow

✅ API Integration
   └── Complete API endpoint configuration

✅ Lifecycle Hooks
   ├── onBeforeExecute
   ├── onAfterExecute
   └── onError

✅ Icon References
   └── Icon paths for BubbleLab UI
```

**Files**: 11 files (1 plugin.ts + 10 schemas)
**Lines of Code**: ~500
**Status**: ❌ VIOLATES AIR GAP - Must be extracted and removed

---

## 🎯 Merge Strategy: Keep ALL Unique Features

### Feature Matrix

| Feature Category | Plugin 1 | Plugin 2 | Plugin 3 | Unified Strategy |
|------------------|----------|----------|----------|------------------|
| **UI Components** | ✅ 26 components | ❌ | ❌ | **Keep from P1** |
| **Node System** | ❌ | ✅ Full system | ❌ | **Keep from P2** |
| **Node Registry** | ❌ | ✅ Yes | ❌ | **Keep from P2** |
| **Services Layer** | ✅ API + hooks | ❌ | ❌ | **Keep from P1** |
| **State Stores** | ✅ Zustand | ❌ | ❌ | **Keep from P1** |
| **Schemas** | ✅ Complete | ❌ | ✅ (maybe better) | **Merge both** |
| **Plugin Factory** | ❌ | ✅ Advanced | ❌ | **Keep from P2** |
| **PluginDefinition** | ❌ | ❌ | ✅ | **Keep from P3** |
| **Config Panels** | ❌ | ✅ Enhanced | ❌ | **Keep from P2** |
| **Types** | ✅ Good | ✅ Advanced | ❌ | **Merge both** |
| **API Integration** | ❌ | ❌ | ✅ Complete | **Keep from P3** |
| **Lifecycle Hooks** | ❌ | ❌ | ✅ Yes | **Keep from P3** |
| **Utilities** | ✅ Basic | ✅ Advanced | ❌ | **Merge both** |
| **Assets** | ✅ Complete | ❌ | ❌ | **Keep from P1** |
| **Documentation** | ✅ README | ✅ Multiple docs | ❌ | **Merge both** |

---

## 🏗️ Proposed Unified Architecture

```
OpenEvolve-Plugin/ (UNIFIED - Single Source of Truth)
├── src/
│   ├── core/                          # [NEW] Shared core infrastructure
│   │   ├── PluginDefinition.ts        # [From P3] Official plugin definition
│   │   ├── types/                     # [Merged P1+P2] All types
│   │   │   ├── index.ts
│   │   │   ├── plugin-types.ts        # [From P2] Enhanced types
│   │   │   ├── enhanced-plugin-types.ts
│   │   │   ├── extended-plugin-types.ts
│   │   │   ├── nodes.ts               # [From P2] Node types
│   │   │   └── workflows.ts           # [From P1] Workflow types
│   │   ├── constants/                 # [From P3] Service definitions
│   │   │   ├── services.ts            # [From P3] All 10 services
│   │   │   └── endpoints.ts           # [From P3] API endpoints
│   │   └── utils/                     # [Merged P1+P2] All utilities
│   │       ├── createPlugin.ts        # [From P2] Plugin factory
│   │       ├── createEnhancedPlugin.ts
│   │       ├── validation.ts
│   │       └── advancedUtilities.ts   # [From P2]
│   │
│   ├── nodes/                         # [From P2] Complete node system
│   │   ├── BaseNode.ts                # [From P2] Abstract base
│   │   ├── OpenEvolveBaseNode.ts      # [From P2]
│   │   ├── DecompositionNode.ts       # [From P2]
│   │   ├── SolutionNode.ts            # [From P2]
│   │   ├── VerificationNode.ts        # [From P2]
│   │   ├── registry.ts                # [From P2] Node registry
│   │   └── index.ts
│   │
│   ├── components/                    # [From P1] All UI components
│   │   ├── pages/                     # [From P1]
│   │   │   ├── OpenEvolveDashboard.tsx
│   │   │   ├── AnalyticsDashboard.tsx
│   │   │   ├── WorkflowBuilder.tsx
│   │   │   ├── LeanAidePage.tsx
│   │   │   └── KnowledgeBasePage.tsx
│   │   ├── workflow/                  # [From P1]
│   │   │   ├── ConfigPanel.tsx
│   │   │   └── ExecutionMonitor.tsx
│   │   ├── config/                    # [From P2] Enhanced configs
│   │   │   ├── EnhancedOpenEvolveConfigPanel.tsx
│   │   │   ├── OpenEvolveConfigPanel.tsx
│   │   │   ├── PerformanceConfigTab.tsx
│   │   │   ├── SecurityConfigTab.tsx
│   │   │   └── RemainingTabs.tsx
│   │   ├── analytics/                 # [From P1]
│   │   │   ├── MetricCard.tsx
│   │   │   ├── PerformanceChart.tsx
│   │   │   ├── ArtifactTable.tsx
│   │   │   └── StatGrid.tsx
│   │   ├── knowledge/                 # [From P1]
│   │   │   ├── ArtifactList.tsx
│   │   │   ├── KnowledgeSearch.tsx
│   │   │   ├── ArtifactEditor.tsx
│   │   │   └── ArtifactDetail.tsx
│   │   ├── leanaide/                  # [From P1]
│   │   │   ├── ProofEditor.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── VerificationDisplay.tsx
│   │   │   └── ProgressTracker.tsx
│   │   ├── nodes/                     # [From P2] Node UI components
│   │   │   ├── OpenEvolveNode.tsx
│   │   │   ├── DecompositionNodeComponent.tsx
│   │   │   ├── SolutionNodeComponent.tsx
│   │   │   └── VerificationNodeComponent.tsx
│   │   └── shared/                    # [From P1]
│   │       ├── ProgressBar.tsx
│   │       ├── LiveLogViewer.tsx
│   │       ├── FormWrapper.tsx
│   │       └── StatusBadge.tsx
│   │
│   ├── services/                      # [From P1] Complete services layer
│   │   ├── api/                       # [From P1]
│   │   │   ├── evolutionClient.ts
│   │   │   ├── adversarialClient.ts
│   │   │   ├── leanaideClient.ts
│   │   │   └── ... (all 10 services)
│   │   ├── hooks/                     # [From P1]
│   │   │   ├── useEvolution.ts
│   │   │   ├── useLeanAide.ts
│   │   │   ├── useWorkflow.ts
│   │   │   └── ...
│   │   └── api.ts                     # [From P3] API integration
│   │
│   ├── stores/                        # [From P1] State management
│   │   ├── useAuthStore.ts
│   │   ├── useWorkflowStore.ts
│   │   ├── useAnalyticsStore.ts
│   │   ├── useKnowledgeStore.ts
│   │   ├── useLeanAideStore.ts
│   │   └── useEvolutionStore.ts
│   │
│   ├── schemas/                       # [Merged P1+P3] All schemas
│   │   ├── evolution.ts               # [Merge best from both]
│   │   ├── adversarial.ts
│   │   ├── maker.ts
│   │   ├── mdap.ts
│   │   ├── decomposition.ts
│   │   ├── knowledge.ts
│   │   ├── leanaide.ts
│   │   ├── hephaestus.ts
│   │   ├── roma.ts
│   │   ├── invention.ts
│   │   └── index.ts
│   │
│   ├── hooks/                         # [Merged P1+P2] All hooks
│   │   ├── useEnhancedOpenEvolveConfig.ts  # [From P2]
│   │   └── ...                             # [From P1]
│   │
│   ├── utils/                         # [Merged P1+P2]
│   │   ├── createOpenEvolvePlugin.ts       # [From P2]
│   │   ├── createEnhancedOpenEvolvePlugin.ts
│   │   ├── advancedUtilities.ts            # [From P2]
│   │   └── ...                             # [From P1]
│   │
│   ├── assets/                        # [From P1]
│   │   ├── icons/
│   │   └── images/
│   │
│   ├── types/                         # [Merged P1+P2]
│   │   ├── plugin-types.ts             # [From P2]
│   │   ├── enhanced-plugin-types.ts
│   │   ├── extended-plugin-types.ts
│   │   ├── nodes.ts                    # [From P2]
│   │   ├── index.ts
│   │   └── ...
│   │
│   ├── plugin.ts                      # [NEW/From P3] Main plugin export
│   └── index.ts                       # [NEW] Unified exports
│
├── examples/                          # [From P2]
│   └── ...
├── docs/                              # [Merged P1+P2]
│   ├── ARCHITECTURE.md                # [From P1]
│   ├── API.md                         # [From P1]
│   ├── WORKFLOWS.md                   # [From P1]
│   ├── ADVANCED_FEATURES.md           # [From P2]
│   ├── NODE_REGISTRY_README.md        # [From P2]
│   └── IMPLEMENTATION_SUMMARY.md      # [From P2]
├── tests/                             # [From P1]
│   └── ...
├── package.json                       # [Unified]
├── tsconfig.json                      # [Unified]
├── vite.config.ts                     # [Unified]
└── README.md                          # [Unified - Merged]
```

---

## 📋 Phase-by-Phase Unification Plan

### Phase 1: Complete Feature Inventory (Agent 1)

**Objective**: Catalog EVERY feature from all three plugins

**Tasks**:
1. **Deep Scan Plugin 1** (`OpenEvolve-Plugin/`)
   - List all 26 components with descriptions
   - Catalog all services, hooks, stores
   - Document all schemas, types, utils
   - List all assets

2. **Deep Scan Plugin 2** (`openevolve-bubblelab-plugin/`)
   - Document entire node system
   - List all registry features
   - Catalog all config panels
   - Document plugin factory utilities

3. **Deep Scan Plugin 3** (Embedded)
   - Document PluginDefinition structure
   - List all 10 service definitions
   - Compare schemas with P1/P2
   - Document API integration

4. **Create Feature Matrix**
   - Map every feature to source plugin
   - Identify overlaps vs. unique features
   - Note differences in implementations
   - Flag potential conflicts

5. **Schema Comparison**
   - Compare all 10 schemas across plugins
   - Identify which is most complete
   - Note schema differences
   - Plan schema merge strategy

**Deliverables**:
- `COMPLETE_FEATURE_INVENTORY.md` - Every feature from all 3 plugins
- `SCHEMA_COMPARISON_MATRIX.md` - Detailed schema comparison
- `FEATURE_OVERLAP_ANALYSIS.md` - What's unique vs. shared
- `UNIFICATION_STRATEGY.md` - Detailed merge plan

---

### Phase 2: Architecture Design (Agent 2)

**Objective**: Design unified architecture that accommodates ALL features

**Tasks**:
1. **Directory Structure Design**
   - Plan merged directory layout
   - Ensure all features have proper homes
   - Design clear separation of concerns

2. **Namespace Strategy**
   - Resolve any naming conflicts
   - Design consistent naming conventions
   - Plan backward compatibility

3. **Import Path Strategy**
   - Design internal import structure
   - Plan external API surface
   - Ensure tree-shaking works

4. **Integration Architecture**
   - Design how P1's components work with P2's nodes
   - Plan how P3's PluginDefinition fits in
   - Design service layer integration

5. **Type System Unification**
   - Merge P1 and P2 type systems
   - Resolve type conflicts
   - Design type export structure

6. **Export Strategy**
   - Plan main exports
   - Design backward compatibility layer
   - Ensure all original exports work

**Deliverables**:
- `UNIFIED_ARCHITECTURE.md` - Complete architecture design
- `DIRECTORY_STRUCTURE.md` - Merged directory plan
- `NAMESPACE_STRATEGY.md` - Naming resolution
- `TYPE_UNIFICATION_PLAN.md` - Type system merge
- `EXPORT_COMPATIBILITY_PLAN.md` - Backward compatibility

---

### Phase 3: Core Infrastructure Merge (Agent 3)

**Objective**: Merge core infrastructure (types, utils, plugin definition)

**Tasks**:
1. **Merge Type Systems**
   - Combine P1 and P2 types
   - Resolve conflicts
   - Create unified type exports

2. **Merge Utilities**
   - Combine P1 basic utils + P2 advanced utils
   - Resolve any conflicts
   - Test all utility functions

3. **Integrate Plugin Definition**
   - Add P3's PluginDefinition
   - Create plugin factory (from P2)
   - Set up plugin lifecycle hooks

4. **Merge Constants**
   - Add service definitions (from P3)
   - Add API endpoints (from P3)
   - Create unified constants

5. **Set Up Core Structure**
   - Create new directory structure
   - Set up module boundaries
   - Configure build system

**Deliverables**:
- Merged type system
- Merged utilities
- Plugin definition integrated
- Core infrastructure working
- Build system configured

---

### Phase 4: Component Layer Merge (Agent 4)

**Objective**: Merge all UI components from P1 and P2

**Tasks**:
1. **Migrate P1 Components** (26 components)
   - All analytics components
   - All knowledge components
   - All LeanAide components
   - All page components
   - All shared components

2. **Migrate P2 Components**
   - All node UI components
   - All enhanced config panels
   - All config tabs

3. **Resolve Conflicts**
   - Check for component name conflicts
   - Resolve any styling conflicts
   - Merge similar components if needed

4. **Update Imports**
   - Fix all component imports
   - Update component references
   - Test component rendering

5. **Create Component Index**
   - Create clean component exports
   - Organize by category
   - Ensure backward compatibility

**Deliverables**:
- All 30+ components migrated
- All conflicts resolved
- All imports working
- Component index created
- Components render correctly

---

### Phase 5: Node System Integration (Agent 5)

**Objective**: Integrate P2's complete node system

**Tasks**:
1. **Migrate Node Classes**
   - BaseNode
   - OpenEvolveBaseNode
   - DecompositionNode
   - SolutionNode
   - VerificationNode

2. **Migrate Node Registry**
   - Complete registry system
   - Node factory functions
   - Node discovery

3. **Integrate with Components**
   - Connect node UI components
   - Wire up config panels
   - Test node workflows

4. **Node-Service Integration**
   - Connect nodes to P1's services
   - Wire up API calls
   - Test node execution

**Deliverables**:
- Complete node system working
- Registry functional
- Nodes integrated with services
- Node workflows tested

---

### Phase 6: Services & State Management (Agent 6)

**Objective**: Integrate services layer and state stores

**Tasks**:
1. **Migrate API Services**
   - All 10 service clients
   - WebSocket integration
   - Error handling

2. **Migrate React Hooks**
   - All custom hooks
   - Hook integration with components
   - Hook testing

3. **Migrate State Stores**
   - All Zustand stores
   - Store integration with services
   - Store testing

4. **Integrate API Configuration**
   - P3's API endpoints
   - P3's service definitions
   - Configuration management

**Deliverables**:
- All services working
- All hooks functional
- All stores integrated
- API configuration complete
- Services tested

---

### Phase 7: Schema Unification (Agent 7)

**Objective**: Merge all schemas from P1, P2, P3

**Tasks**:
1. **Compare All Schemas**
   - Evolution schemas (P1 vs P3)
   - All 10 workflow types
   - Identify best version of each

2. **Merge Schemas**
   - Take best features from each
   - Ensure complete coverage
   - Maintain backward compatibility

3. **Validate Schemas**
   - Test schema validation
   - Check Zod integration
   - Test schema exports

4. **Document Schemas**
   - Document all schema fields
   - Create schema examples
   - Update schema docs

**Deliverables**:
- Unified schemas (all 10 workflows)
- Schema documentation
- Schema validation working
- Schema examples

---

### Phase 8: Documentation & Examples (Agent 8)

**Objective**: Merge and update all documentation

**Tasks**:
1. **Merge READMEs**
   - Combine P1 and P2 READMEs
   - Add all features
   - Update installation

2. **Merge Documentation**
   - ARCHITECTURE.md (P1)
   - ADVANCED_FEATURES.md (P2)
   - NODE_REGISTRY_README.md (P2)
   - IMPLEMENTATION_SUMMARY.md (P2)
   - API.md (P1)
   - WORKFLOWS.md (P1)

3. **Update Examples**
   - Combine examples from P2
   - Add new examples for unified features
   - Test all examples

4. **Create Migration Guide**
   - Guide for users of each original plugin
   - Breaking changes documentation
   - Upgrade path

**Deliverables**:
- Complete unified README
- All documentation merged
- All examples working
- Migration guide created

---

### Phase 9: Testing & Validation (Agent 9)

**Objective**: Comprehensive testing of unified plugin

**Tasks**:
1. **Migrate Tests**
   - Migrate tests from P1
   - Create tests for P2 features
   - Test P3 integrations

2. **Integration Testing**
   - Test all services
   - Test all nodes
   - Test all components
   - Test all workflows

3. **Build Validation**
   - Build successful
   - No TypeScript errors
   - No ESLint errors
   - Bundle size reasonable

4. **Feature Validation**
   - Test all P1 features work
   - Test all P2 features work
   - Test all P3 features work
   - Zero feature loss confirmed

**Deliverables**:
- All tests passing
- Build successful
- All features validated
- Zero feature loss verified

---

### Phase 10: BubbleLab Integration & Cleanup (Agent 10)

**Objective**: Update BubbleLab to use external plugin, remove embedded code

**Tasks**:
1. **Update BubbleLab Dependencies**
   - Add unified plugin as dependency
   - Configure import paths
   - Update package.json

2. **Update BubbleLab Imports**
   - Replace embedded imports with external
   - Update plugin registration
   - Test plugin loading

3. **Remove Embedded Plugin**
   - Delete `BubbleLab/apps/bubble-studio/src/plugins/openevolve/`
   - Verify no references remain
   - Clean up imports

4. **Final Validation**
   - BubbleLab builds successfully
   - OpenEvolve plugin loads
   - All features work
   - BubbleLab can update from upstream

**Deliverables**:
- BubbleLab using external plugin
- Embedded plugin removed
- BubbleLab builds and runs
- All features working
- AIR GAP compliance achieved

---

## ✅ Success Criteria

### Must Have (Non-negotiable)
- ✅ **ZERO feature loss** - Every feature from all 3 plugins present and working
- ✅ **All components** - All 26+ components from P1 + all from P2
- ✅ **Complete node system** - Entire P2 node system functional
- ✅ **All services** - All 10 services from P1 working
- ✅ **All schemas** - All 10 workflow schemas present
- ✅ **AIR GAP compliance** - Zero OpenEvolve code in BubbleLab core
- ✅ **Build successful** - Both unified plugin and BubbleLab build
- ✅ **Tests passing** - All functionality tested

### Should Have
- ✅ Clean architecture
- ✅ Good code organization
- ✅ Comprehensive documentation
- ✅ Backward compatibility maintained

### Nice to Have
- ✅ Improved performance
- ✅ Better TypeScript types
- ✅ Enhanced developer experience

---

## 📊 Progress Tracking

| Phase | Agent | Focus | Status |
|-------|-------|-------|--------|
| 1 | Agent 1 | Inventory | ⏳ Pending |
| 2 | Agent 2 | Architecture | ⏳ Pending |
| 3 | Agent 3 | Core Infrastructure | ⏳ Pending |
| 4 | Agent 4 | Components | ⏳ Pending |
| 5 | Agent 5 | Node System | ⏳ Pending |
| 6 | Agent 6 | Services & State | ⏳ Pending |
| 7 | Agent 7 | Schemas | ⏳ Pending |
| 8 | Agent 8 | Documentation | ⏳ Pending |
| 9 | Agent 9 | Testing | ⏳ Pending |
| 10 | Agent 10 | BubbleLab Integration | ⏳ Pending |

---

## 🚨 Critical Reminders

### For All Agents
1. **ZERO FEATURE LOSS** - This is the #1 rule
2. **Document everything** - Every decision, every merge
3. **Test thoroughly** - Don't break existing features
4. **Communicate early** - Raise issues immediately
5. **Think about next phase** - Make handoffs easy

### For Phase Leads
1. **Read previous phase outputs** - Build on what came before
2. **Validate against inventory** - Check Phase 1's feature list
3. **Test your changes** - Ensure nothing breaks
4. **Create clean handoff** - Next phase depends on you

---

## 🎯 Final Result

After all 10 phases, you will have:

```
✅ OpenEvolve-Plugin/ - Single unified plugin with ALL features
   ├── All 30+ components from P1 + P2
   ├── Complete node system from P2
   ├── All services and stores from P1
   ├── Plugin definition from P3
   ├── All schemas (merged from all)
   ├── Complete documentation (merged)
   └── Zero feature loss

✅ BubbleLab/ - Clean core, using external plugin
   ├── No embedded OpenEvolve code
   ├── Imports from @openevolve/plugin
   ├── Can update from upstream
   └── AIR GAP compliant

✅ Documentation
   ├── Complete README
   ├── Migration guides
   ├── API docs
   └── Examples
```

---

**Let's UNIFY these plugins! 🚀**

*Remember: Keep EVERYTHING from ALL THREE plugins. No feature left behind.*
