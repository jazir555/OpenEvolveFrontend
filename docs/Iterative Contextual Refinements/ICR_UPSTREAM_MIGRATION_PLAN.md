# ICR Upstream Migration Plan

**Document Type:** Implementation/Porting Plan  
**Date:** 2026-02-17  
**Source:** `Iterative-Contextual-Refinements/` (Current/Local Version)  
**Target:** `Iterative-Contextual-Refinements-main/` (Upstream/Newest Version)  
**Priority:** High - Security and Feature Updates

---

## Executive Summary

This document provides a comprehensive plan to port all custom changes from the local ICR version to the upstream version. The upstream version contains critical updates, bug fixes, and new features that need to be integrated while preserving our custom integrations (MathSolver, ICR patterns, etc.).

### Key Statistics

| Metric | Local Version | Upstream Version |
|--------|--------------|------------------|
| **Total Files** | ~50 files | ~100 files |
| **Total Size** | ~2.5 MB | ~5.0 MB |
| **Last Updated** | 2026-01-31 | 2026-02-15 |
| **Major Features** | 7 modes + MathSolver | 7 modes (enhanced) |
| **State Management** | Basic | Advanced with handlers |
| **Documentation** | Extensive | Minimal |

---

## Directory Structure Comparison

### Local Version Structure
```
Iterative-Contextual-Refinements/
├── AdaptiveDeepthink/          ✅ Present
├── Agentic/                    ✅ Present
├── Components/                 ⚠️  Custom components
├── Contextual/                 ✅ Present
├── Core/                       ⚠️  Modified
├── Deepthink/                  ✅ Present
├── GenerativeUI/               ⚠️  Custom feature
├── Logos/                      ✅ Present
├── MathSolver/                 🔴  CUSTOM - Not in upstream
├── Parsing/                    ⚠️  Modified
├── React/                      ⚠️  Custom feature
├── Refine/                     ✅ Present
├── Routing/                    ⚠️  Modified
├── styles/                     ⚠️  Modified
├── UI/                         ⚠️  Modified
├── Utils/                      ⚠️  Modified
├── .github/                    ⚠️  Custom CI/CD
├── .npmrc                      ✅ Same
├── COMPREHENSIVE_GAP_ANALYSIS.md 🔴  Local only
├── FINAL_INTEGRATION_SUMMARY.md  🔴  Local only
├── FOURTH_GAP_ANALYSIS.md      🔴  Local only
├── GAP_ANALYSIS_REPORT.md      🔴  Local only
├── INTEGRATION_FIXES_SUMMARY.md 🔴  Local only
├── MATH_SOLVER_INTEGRATION_SUMMARY.md 🔴  Local only
├── metadata.json               🔴  Local only
├── THIRD_GAP_ANALYSIS.md       🔴  Local only
├── VERIFICATION_REPORT.md      🔴  Local only
└── vite.config.ts              ⚠️  Modified
```

### Upstream Version Structure
```
Iterative-Contextual-Refinements-main/
├── AdaptiveDeepthink/          ✅ Enhanced
├── Agentic/                    ✅ Enhanced
├── Contextual/                 ✅ Enhanced
├── Core/                       ✅ New state serializer
├── Deepthink/                  ✅ Enhanced
├── Logos/                      ✅ Same
├── Refine/                     ✅ Enhanced
├── Routing/                    ✅ Enhanced
├── Styles/                     ✅ New structure
├── UI/                         ✅ Enhanced
├── .gitignore                  ✅ Standard
├── index.css                   ✅ Updated
├── index.html                  ✅ Updated
├── index.tsx                   ✅ Updated
├── package.json                ✅ Updated deps
├── README.md                   ✅ Updated
├── tsconfig.json               ✅ Updated
└── vite.config.ts              ✅ Updated
```

---

## Critical Differences

### 1. Core State Management (HIGH PRIORITY)

**Local:** Basic state management  
**Upstream:** Advanced state serializer with mode-specific handlers

**New Files in Upstream:**
```
Core/StateSerializer/
├── handlers/
│   ├── AdaptiveDeepthinkStateHandler.ts
│   ├── AgenticStateHandler.ts
│   ├── ContextualStateHandler.ts
│   ├── DeepthinkStateHandler.ts
│   ├── WebsiteModeStateHandler.ts
│   └── index.ts
├── index.ts
├── ModeStateHandler.ts
├── SerializationEngine.ts
├── StateSanitizer.ts
├── StateVersion.ts
```

**Porting Action:** 
- ✅ Merge upstream state serializer into local Core/
- ⚠️  Add MathSolver state handler (custom)
- ⚠️  Preserve local state enhancements

---

### 2. MathSolver Integration (CUSTOM FEATURE)

**Local:** Full MathSolver integration (8 files)  
**Upstream:** Not present

**Files to Port:**
```
MathSolver/
├── MathSolverMode.ts
├── MathSolverUI.tsx
├── MathSolverCore.ts
├── MathSolverPrompts.ts
├── MathSolverAPI.ts
├── MathTools.ts
├── index.ts
└── __tests__/MathSolver.integration.test.ts
```

**Porting Action:**
- 🔴 Create MathSolver directory in upstream
- 🔴 Copy all MathSolver files
- ⚠️  Update Core/Types.ts to include 'mathsolver' mode
- ⚠️  Update Core/State.ts to include MathSolver state
- ⚠️  Update Core/App.ts to initialize MathSolver
- ⚠️  Add MathSolver to export/import functionality

---

### 3. GenerativeUI Mode (CUSTOM FEATURE)

**Local:** Full GenerativeUI with interaction capture  
**Upstream:** Not present (or minimal)

**Files to Port:**
```
GenerativeUI/
├── GenerativeUICore.ts
├── GenerativeUI.tsx
├── GenerativeUI.css
├── GenerativeUIPrompts.ts
└── InteractionTracker.ts (if exists)
```

**Porting Action:**
- 🔴 Preserve GenerativeUI directory
- ⚠️  Merge any upstream UI improvements
- ⚠️  Update mode selectors

---

### 4. React Mode (CUSTOM FEATURE)

**Local:** React application development mode  
**Upstream:** Not present

**Files to Port:**
```
React/
├── ReactMode.tsx
├── ReactBuildManager.tsx
├── ReactPrompts.ts
└── ReactWorkerAgents.ts
```

**Porting Action:**
- 🔴 Preserve React directory
- ⚠️  Merge any upstream improvements

---

### 5. Enhanced Routing (MEDIUM PRIORITY)

**Local:** Basic routing with model selection  
**Upstream:** Enhanced with API estimation, provider management

**New Files in Upstream:**
```
Routing/
├── ApiCallEstimator.ts         🔵 NEW
├── ApiConfig.ts                🔵 NEW
├── ApiKeyUI.ts                 🔵 NEW
├── DeepthinkConfigController.ts 🔵 NEW
├── ProviderManagementUI.ts     🔵 NEW
├── ProviderManager.ts          🔵 NEW
├── PromptRefiner.ts            🔵 NEW
├── PromptsModal/               🔵 NEW
│   ├── PromptsModalLayout.tsx
│   └── PromptsModalManager.tsx
```

**Porting Action:**
- ✅ Copy all new routing files
- ⚠️  Merge local auto-refine configuration
- ⚠️  Merge local ICR configuration
- ⚠️  Update ModelConfig.ts with local enhancements

---

### 6. Deepthink Enhancements (MEDIUM PRIORITY)

**Local:** Basic Deepthink with iterative corrections  
**Upstream:** Enhanced with solution pool, knowledge integration

**New Files in Upstream:**
```
Deepthink/
├── DeepthinkConfigPanel.tsx    🔵 NEW
├── DeepthinkConfigPanel.css    🔵 NEW
├── DeepthinkIterativeHistory.ts 🔵 NEW
├── DeepthinkKnowledge.css      🔵 NEW
├── DeepthinkModals.css         🔵 NEW
├── DeepthinkResults.css        🔵 NEW
├── SolutionPool.tsx            🔵 NEW
├── SolutionPool.css            🔵 NEW
└── Prompt Templates/           🔵 NEW
    ├── FinancialDocumentExtraction.ts
    └── Generalized.ts
```

**Porting Action:**
- ✅ Copy all new Deepthink files
- ⚠️  Merge local iterative corrections
- ⚠️  Preserve local prompt templates

---

### 7. Agentic Mode Enhancements (MEDIUM PRIORITY)

**Local:** Basic agentic mode  
**Upstream:** Enhanced with ArxivAPI, improved UI

**New Files in Upstream:**
```
Agentic/
├── ArxivAPI.ts                 🔵 NEW
├── AgenticImportExport.tsx     🔵 NEW
├── AgenticUI.css               🔵 UPDATED
└── AgenticUI.tsx               🔵 UPDATED
```

**Porting Action:**
- ✅ Copy ArxivAPI.ts
- ✅ Copy AgenticImportExport.tsx
- ⚠️  Merge UI improvements
- ⚠️  Preserve local tool integrations

---

### 8. Component Structure (LOW PRIORITY)

**Local:** Components/ directory with shared components  
**Upstream:** No Components/ directory (components in mode directories)

**Porting Action:**
- ⚠️  Review Components/ directory
- ⚠️  Move reusable components to appropriate locations
- ⚠️  Preserve custom components

---

### 9. Styles and CSS (LOW PRIORITY)

**Local:** styles/ directory  
**Upstream:** Styles/ directory (capitalized)

**Porting Action:**
- ⚠️  Merge style improvements from upstream
- ⚠️  Preserve local customizations
- ⚠️  Standardize naming (styles vs Styles)

---

### 10. Build Configuration (MEDIUM PRIORITY)

**Local:** Modified vite.config.ts, tsconfig.json  
**Upstream:** Updated configurations

**Files to Merge:**
```
├── vite.config.ts              ⚠️  Merge changes
├── tsconfig.json               ⚠️  Merge changes
├── package.json                ⚠️  Merge dependencies
└── package-lock.json           ⚠️  Regenerate after merge
```

**Porting Action:**
- ⚠️  Compare vite.config.ts line by line
- ⚠️  Merge TypeScript configurations
- ⚠️  Update package.json with new dependencies
- ⚠️  Regenerate package-lock.json

---

## Documentation Migration

### Local Documentation (Preserve)

These documents are **local-only** and should be preserved:

1. **COMPREHENSIVE_GAP_ANALYSIS.md** - Our gap analysis
2. **FINAL_INTEGRATION_SUMMARY.md** - MathSolver integration
3. **FOURTH_GAP_ANALYSIS.md** - Gap analysis
4. **GAP_ANALYSIS_REPORT.md** - Original gap report
5. **INTEGRATION_FIXES_SUMMARY.md** - Fixes applied
6. **MATH_SOLVER_INTEGRATION_SUMMARY.md** - MathSolver docs
7. **THIRD_GAP_ANALYSIS.md** - Historical analysis
8. **VERIFICATION_REPORT.md** - Verification results
9. **metadata.json** - Project metadata

**Action:** Create `docs/local/` directory in upstream and copy all local documentation

### Upstream Documentation (Merge)

1. **README.md** - Merge local enhancements
2. **Deepthink/DeepthinkDocs.md** - Merge with local docs

---

## Integration Points to Preserve

### 1. ICR Integration Layer

**Location:** `glue/adapters/icr-adapter/` (outside ICR directory)

**Status:** ✅ Already separate  
**Action:** No changes needed - integration layer is external

### 2. MathSolver Backend API

**Location:** `BubbleLab/services/openevolve-api/api/icr.py`

**Status:** ✅ Already separate  
**Action:** No changes needed - API is external

### 3. Pattern Storage

**Location:** Various Python files (icr_integration.py, etc.)

**Status:** ✅ Already separate  
**Action:** No changes needed - Python integration is external

---

## Porting Phases

### Phase 1: Core Infrastructure (Week 1)

**Priority:** CRITICAL  
**Estimated Effort:** 3-4 days

**Tasks:**
1. ✅ Copy Core/StateSerializer/ from upstream
2. ✅ Create MathSolverStateHandler.ts (custom)
3. ✅ Merge Core/Types.ts (add 'mathsolver' mode)
4. ✅ Merge Core/State.ts (add MathSolver state)
5. ✅ Merge Core/App.ts (initialize MathSolver)
6. ✅ Update Core/ConfigManager.ts
7. ✅ Test state serialization/deserialization
8. ✅ Test MathSolver mode switching

**Deliverables:**
- Working state management with MathSolver support
- All 7 modes + MathSolver functional
- Export/import working for all modes

---

### Phase 2: Mode Enhancements (Week 2)

**Priority:** HIGH  
**Estimated Effort:** 4-5 days

**Tasks:**
1. ✅ Copy Deepthink enhancements (SolutionPool, ConfigPanel, etc.)
2. ✅ Copy Agentic enhancements (ArxivAPI, ImportExport)
3. ✅ Copy Routing enhancements (ProviderManagement, ApiEstimator)
4. ✅ Merge local auto-refine configuration
5. ✅ Merge local ICR configuration
6. ✅ Update PromptsModal with local customizations
7. ✅ Test all modes with new features

**Deliverables:**
- All mode enhancements from upstream
- Local customizations preserved
- All modes tested and working

---

### Phase 3: Custom Features (Week 3)

**Priority:** HIGH  
**Estimated Effort:** 3-4 days

**Tasks:**
1. ✅ Port MathSolver directory to upstream
2. ✅ Port GenerativeUI directory to upstream
3. ✅ Port React directory to upstream
4. ✅ Update mode selectors to include custom modes
5. ✅ Update documentation for custom modes
6. ✅ Test custom modes in new structure

**Deliverables:**
- MathSolver fully integrated
- GenerativeUI fully integrated
- React mode fully integrated
- All custom modes working

---

### Phase 4: Build and Configuration (Week 4)

**Priority:** MEDIUM  
**Estimated Effort:** 2-3 days

**Tasks:**
1. ✅ Merge vite.config.ts
2. ✅ Merge tsconfig.json
3. ✅ Merge package.json dependencies
4. ✅ Regenerate package-lock.json
5. ✅ Update .gitignore
6. ✅ Test build process
7. ✅ Test development server
8. ✅ Test production build

**Deliverables:**
- Build configuration merged
- All dependencies up to date
- Build process working
- Development workflow functional

---

### Phase 5: Documentation and Testing (Week 5)

**Priority:** MEDIUM  
**Estimated Effort:** 2-3 days

**Tasks:**
1. ✅ Create docs/local/ directory
2. ✅ Copy all local documentation
3. ✅ Merge README.md
4. ✅ Update DeepthinkDocs.md
5. ✅ Create migration guide
6. ✅ Run all tests
7. ✅ Create test report
8. ✅ Update CHANGELOG

**Deliverables:**
- All documentation preserved
- Migration guide created
- All tests passing
- CHANGELOG updated

---

## File-by-File Porting Checklist

### Core Directory

| File | Action | Status | Notes |
|------|--------|--------|-------|
| Core/App.ts | Merge | ⏳ Pending | Add MathSolver initialization |
| Core/State.ts | Merge | ⏳ Pending | Add MathSolver state fields |
| Core/Types.ts | Merge | ⏳ Pending | Add 'mathsolver' to ApplicationMode |
| Core/ConfigManager.ts | Merge | ⏳ Pending | Add MathSolver config |
| Core/StateSerializer/ | Copy | ⏳ Pending | New from upstream |
| Core/StateSerializer/handlers/ | Merge | ⏳ Pending | Add MathSolverStateHandler |
| Core/Parsing.ts | Compare | ⏳ Pending | Check for improvements |
| Core/JsonParser.ts | Copy | ⏳ Pending | New from upstream |
| Core/OutputCleaner.ts | Copy | ⏳ Pending | New from upstream |
| Core/SuggestionParser.ts | Compare | ⏳ Pending | Check for improvements |

### Mode Directories

| Directory | Action | Status | Notes |
|-----------|--------|--------|-------|
| AdaptiveDeepthink/ | Merge | ⏳ Pending | Copy upstream, merge local |
| Agentic/ | Merge | ⏳ Pending | Copy ArxivAPI, merge UI |
| Contextual/ | Merge | ⏳ Pending | Copy upstream enhancements |
| Deepthink/ | Merge | ⏳ Pending | Copy SolutionPool, ConfigPanel |
| MathSolver/ | Port | ⏳ Pending | Create in upstream |
| GenerativeUI/ | Port | ⏳ Pending | Preserve in upstream |
| React/ | Port | ⏳ Pending | Preserve in upstream |
| Refine/ | Merge | ⏳ Pending | Copy upstream enhancements |

### Routing Directory

| File | Action | Status | Notes |
|------|--------|--------|-------|
| Routing/ApiCallEstimator.ts | Copy | ⏳ Pending | New from upstream |
| Routing/ApiConfig.ts | Copy | ⏳ Pending | New from upstream |
| Routing/ApiKeyUI.ts | Copy | ⏳ Pending | New from upstream |
| Routing/DeepthinkConfigController.ts | Copy | ⏳ Pending | New from upstream |
| Routing/ProviderManagementUI.ts | Copy | ⏳ Pending | New from upstream |
| Routing/ProviderManager.ts | Copy | ⏳ Pending | New from upstream |
| Routing/PromptRefiner.ts | Copy | ⏳ Pending | New from upstream |
| Routing/PromptsModal/ | Copy | ⏳ Pending | New from upstream |
| Routing/ModelConfig.ts | Merge | ⏳ Pending | Add auto-refine, ICR config |
| Routing/ModelSelectionUI.ts | Merge | ⏳ Pending | Add auto-refine toggle |
| Routing/RoutingManager.ts | Merge | ⏳ Pending | Merge local changes |
| Routing/PromptsManager.ts | Merge | ⏳ Pending | Merge local changes |

### Configuration Files

| File | Action | Status | Notes |
|------|--------|--------|-------|
| vite.config.ts | Merge | ⏳ Pending | Compare line by line |
| tsconfig.json | Merge | ⏳ Pending | Merge TypeScript settings |
| package.json | Merge | ⏳ Pending | Merge dependencies |
| package-lock.json | Regenerate | ⏳ Pending | After package.json merge |
| .gitignore | Merge | ⏳ Pending | Merge ignore patterns |
| index.html | Compare | ⏳ Pending | Check for improvements |
| index.tsx | Compare | ⏳ Pending | Check for improvements |
| index.css | Compare | ⏳ Pending | Check for improvements |

---

## Risk Assessment

### High Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MathSolver integration breaks | High | Medium | Extensive testing, backup current version |
| State serialization fails | High | Medium | Test all modes, preserve old handlers |
| Build process breaks | Medium | Low | Keep old configs, test incrementally |
| Custom modes don't work | High | Medium | Port carefully, test each mode |

### Medium Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| UI styling issues | Medium | High | Compare CSS, test visually |
| Prompt templates lost | Medium | Low | Backup all prompts |
| Dependencies conflict | Medium | Medium | Test in clean environment |

### Low Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Documentation outdated | Low | High | Update during migration |
| Minor bugs introduced | Low | Medium | Comprehensive testing |

---

## Testing Strategy

### Unit Tests

1. **State Serialization Tests**
   - Test save/load for all modes
   - Test MathSolver state specifically
   - Test cross-mode state compatibility

2. **Mode Functionality Tests**
   - Test each mode independently
   - Test mode switching
   - Test configuration loading

3. **Integration Tests**
   - Test ICR adapter integration
   - Test MathSolver API integration
   - Test export/import functionality

### Manual Testing

1. **Visual Testing**
   - Test UI rendering for all modes
   - Test responsive design
   - Test dark/light themes

2. **Functional Testing**
   - Test all 7 modes + MathSolver
   - Test configuration changes
   - Test export/import

3. **Performance Testing**
   - Test load times
   - Test memory usage
   - Test bundle size

---

## Rollback Plan

If migration fails:

1. **Immediate Rollback**
   - Preserve current `Iterative-Contextual-Refinements/` directory
   - Rename to `Iterative-Contextual-Refinements-backup/`
   - Continue using current version

2. **Partial Rollback**
   - Revert specific phases that failed
   - Keep successful phases
   - Fix failed phases incrementally

3. **Hybrid Approach**
   - Keep upstream as base
   - Cherry-pick critical features from local
   - Accept some feature loss if necessary

---

## Success Criteria

### Phase 1 Success
- [ ] State serializer working
- [ ] MathSolver state handler created
- [ ] All modes can switch
- [ ] Export/import functional

### Phase 2 Success
- [ ] All upstream enhancements copied
- [ ] Local customizations merged
- [ ] All modes tested
- [ ] No regressions

### Phase 3 Success
- [ ] MathSolver integrated
- [ ] GenerativeUI integrated
- [ ] React mode integrated
- [ ] All custom modes working

### Phase 4 Success
- [ ] Build configuration merged
- [ ] All dependencies updated
- [ ] Build process working
- [ ] Dev server functional

### Phase 5 Success
- [ ] All documentation preserved
- [ ] Migration guide created
- [ ] All tests passing
- [ ] CHANGELOG updated

---

## Timeline Summary

| Phase | Duration | Start Date | End Date | Status |
|-------|----------|------------|----------|--------|
| Phase 1: Core Infrastructure | 3-4 days | 2026-02-18 | 2026-02-21 | ⏳ Pending |
| Phase 2: Mode Enhancements | 4-5 days | 2026-02-22 | 2026-02-26 | ⏳ Pending |
| Phase 3: Custom Features | 3-4 days | 2026-02-27 | 2026-03-02 | ⏳ Pending |
| Phase 4: Build & Config | 2-3 days | 2026-03-03 | 2026-03-05 | ⏳ Pending |
| Phase 5: Documentation | 2-3 days | 2026-03-06 | 2026-03-08 | ⏳ Pending |

**Total Estimated Duration:** 14-19 days (3 weeks)

---

## Resource Requirements

### Personnel
- 1-2 TypeScript developers
- 1 tester (can be developer)

### Tools
- Git for version control
- VS Code or similar IDE
- Node.js 18+
- npm or yarn

### Environment
- Clean development environment
- Test browsers (Chrome, Firefox, Safari)
- Performance profiling tools

---

## Communication Plan

### Daily Standups
- **When:** Daily at 9:00 AM
- **Duration:** 15 minutes
- **Attendees:** All developers
- **Format:** What did yesterday, what today, blockers

### Weekly Reviews
- **When:** Fridays at 3:00 PM
- **Duration:** 1 hour
- **Attendees:** All stakeholders
- **Format:** Demo progress, review metrics, adjust plan

### Status Reports
- **Frequency:** Weekly
- **Format:** Email + document update
- **Recipients:** Project stakeholders

---

## Approval Required

Before starting migration:

- [ ] Review this plan
- [ ] Approve timeline
- [ ] Allocate resources
- [ ] Set up backup strategy
- [ ] Schedule kickoff meeting

**Approvals Needed From:**
- [ ] Technical Lead
- [ ] Project Manager
- [ ] Product Owner

---

**Document Version:** 1.0  
**Created:** 2026-02-17  
**Next Review:** After Phase 1 completion  
**Status:** ⏳ Pending Approval
