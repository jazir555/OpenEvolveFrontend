# ICR Upstream Migration - Complete Master Plan

**Date:** 2026-02-17  
**Status:** Phase 1 Complete, Comprehensive Planning  
**Total Upstream Files to Review:** 70+ files

---

## Executive Summary

After comprehensive analysis, there are **70+ upstream files** that need to be reviewed for merging. These are organized into **Priority Tiers** based on impact and effort.

### File Count by Category

| Category | Upstream Files | Local Equivalent | Action |
|----------|---------------|------------------|--------|
| **StateSerializer** | 15 | 0 | ✅ MERGED (Phase 1) |
| **Prompt Templates** | 2 | 0 | ✅ MERGED (Phase 1) |
| **Core Utilities** | 5 | 4 (Parsing/) | ⚠️ MERGE improvements |
| **Routing/Provider** | 10 | 8 | ⚠️ MERGE improvements |
| **UI Components** | 35 | 27 (Components/) | ⚠️ MERGE improvements |
| **Mode Enhancements** | 15 | 15 | ⚠️ MERGE improvements |
| **Build Config** | 2 | 2 | ⚠️ MERGE improvements |

---

## Priority Tier 1: CRITICAL (Merge Immediately)

### 1.1 Core Application Files (5 files)

**Files:**
- `Core/App.ts` - Upstream has improvements (+6.6 KB)
- `Core/ConfigManager.ts` - Upstream base + local enhancements
- `Core/State.ts` - Upstream improvements
- `Core/Types.ts` - Upstream base + local MathSolver types
- `Core/SuggestionParser.ts` - Upstream version

**Action Required:**
1. ⚠️  Compare line-by-line with local versions
2. ⚠️  Merge upstream improvements
3. ⚠️  Preserve local enhancements (MathSolver, ICR, auto-refine)
4. ⚠️  Integrate StateSerializer

**Estimated Effort:** 4-6 hours

---

### 1.2 Routing Enhancements (6 files)

**Files:**
- `Routing/ApiCallEstimator.ts` (NEW) - API cost estimation
- `Routing/ApiConfig.ts` (NEW) - API configuration
- `Routing/ApiKeyUI.ts` (NEW) - API key management UI
- `Routing/ProviderManager.ts` (NEW) - Provider management
- `Routing/ProviderManagementUI.ts` (NEW) - Provider UI
- `Routing/DeepthinkConfigController.ts` (NEW) - Deepthink config

**Action Required:**
1. ✅ Copy all NEW files to local
2. ⚠️  Merge with existing RoutingManager.ts
3. ⚠️  Integrate with local ICR configuration
4. ⚠️  Update ModelSelectionUI.ts

**Estimated Effort:** 3-4 hours

---

### 1.3 UI Components - Critical (10 files)

**Files from `Styles/Components/`:**
- `CodeMirrorFileEditor.tsx` (NEW) - File editor component
- `CodeMirrorFileEditor.css` (NEW) - File editor styles
- `FileUpload.tsx` (NEW) - File upload component
- `FileUpload.css` (NEW) - File upload styles
- `AppModeSelector.tsx` - Upstream version
- `ModelParameters.tsx` - Upstream version
- `Sidebar.tsx` - Upstream version
- `SidebarHeader.tsx` - Upstream version
- `SidebarFooter.tsx` - Upstream version
- `MainContent.tsx` - Upstream version

**Action Required:**
1. ✅ Copy NEW components (CodeMirrorFileEditor, FileUpload)
2. ⚠️  Compare existing components
3. ⚠️  Merge upstream improvements
4. ⚠️  Preserve local customizations

**Estimated Effort:** 4-6 hours

---

## Priority Tier 2: HIGH (Merge This Week)

### 2.1 Mode Enhancements (15 files)

**Agentic Mode:**
- `Agentic/ArxivAPI.ts` (NEW) - Arxiv API integration
- `Agentic/AgenticImportExport.tsx` (NEW) - Import/export UI
- `Agentic/AgenticUI.tsx` - Upstream improvements
- `Agentic/AgenticCore.ts` - Upstream improvements
- `Agentic/AgenticCoreLangchain.ts` - Upstream improvements

**Deepthink Mode:**
- `Deepthink/DeepthinkIterativeHistory.ts` (NEW) - Iterative history
- `Deepthink/DeepthinkConfigPanel.tsx` - Upstream improvements
- `Deepthink/SolutionPool.tsx` - Upstream improvements
- `Deepthink/Deepthink.tsx` - Upstream improvements
- `Deepthink/DeepthinkAgents.ts` - Upstream improvements

**Contextual Mode:**
- `Contextual/Contextual.tsx` - Upstream improvements
- `Contextual/ContextualCore.ts` - Upstream improvements
- `Contextual/ContextualUI.tsx` - Upstream improvements

**Adaptive Deepthink:**
- `AdaptiveDeepthink/AdaptiveDeepthinkMode.tsx` - Upstream improvements
- `AdaptiveDeepthink/AdaptiveDeepthinkCore.ts` - Upstream improvements

**Action Required:**
1. ✅ Copy NEW files
2. ⚠️  Merge improvements into local versions
3. ⚠️  Preserve local iterative corrections
4. ⚠️  Preserve local ICR integration

**Estimated Effort:** 6-8 hours

---

### 2.2 DiffModal Components (6 files)

**Files from `Styles/Components/DiffModal/`:**
- `DiffModalController.ts` - Upstream improvements
- `EvolutionViewer.ts` - Upstream improvements
- `SequentialViewer.ts` - Upstream improvements
- `types.ts` - Upstream version
- `utils.ts` - Upstream version

**Action Required:**
1. ⚠️  Compare with local Components/DiffModal/
2. ⚠️  Merge upstream improvements
3. ⚠️  Preserve local enhancements

**Estimated Effort:** 3-4 hours

---

### 2.3 RenderMathMarkdown Component

**Files:**
- `Styles/Components/RenderMathMarkdown.tsx` - Upstream improvements
- `Styles/Components/RenderMathMarkdown.css` - Upstream improvements

**Action Required:**
1. ⚠️  Compare with local Components/RenderMathMarkdown.tsx
2. ⚠️  Merge upstream improvements
3. ⚠️  Local version has significant enhancements - careful merge needed

**Estimated Effort:** 2-3 hours

---

## Priority Tier 3: MEDIUM (Merge Next Week)

### 3.1 Styles/CSS Reorganization

**Upstream Structure:**
```
Styles/
├── Global.css
├── Content.css
├── Buttons.css
├── Inputs.css
├── Layout.css
├── Shiki.css
├── Shiki.ts
└── Components/
    ├── ActionButton.tsx
    ├── ActionButton.css
    └── ...
```

**Local Structure:**
```
styles/
├── global.css
├── content.css
├── buttons.css
├── inputs.css
├── layout.css
├── sidebar.css
├── monaco-overrides.css
└── components/
    ├── buttons.css
    ├── inputs.css
    └── ...
```

**Action Required:**
1. ⚠️  Compare CSS files
2. ⚠️  Merge upstream improvements
3. ⚠️  Preserve local customizations
4. ⚠️  Consider standardizing naming (styles vs Styles)

**Estimated Effort:** 4-6 hours

---

### 3.2 UI Management Files

**Files from `UI/`:**
- `UI/CommonUI.ts` - Upstream utilities
- `UI/Controls.ts` - Upstream controls
- `UI/GlobalModals.ts` - Upstream modals
- `UI/LayoutController.ts` - Upstream layout
- `UI/Layout.tsx` - Upstream layout
- `UI/setupCodeExecutionToggle.ts` - Code execution toggle
- `UI/Sidebar.ts` - Upstream sidebar
- `UI/Tabs.ts` - Upstream tabs
- `UI/Theme.ts` - Upstream theme
- `UI/UIManager.ts` - Upstream UI manager

**Note:** Local has Components/ directory instead of UI/

**Action Required:**
1. ⚠️  Review if any utilities should be merged
2. ⚠️  setupCodeExecutionToggle.ts may be useful
3. ⚠️  Compare with local Components/ structure

**Estimated Effort:** 3-4 hours

---

### 3.3 Refine/Website Mode

**Files:**
- `Refine/WebsiteUI.ts` - Upstream improvements (+7.5 KB)
- `Refine/WebsiteLogic.ts` - Upstream version

**Action Required:**
1. ⚠️  Compare with local Refine/WebsiteUI.ts
2. ⚠️  Merge upstream improvements
3. ⚠️  Preserve local MathSolver integration

**Estimated Effort:** 2-3 hours

---

## Priority Tier 4: LOW (Review as Needed)

### 4.1 Parsing Utilities

**Upstream:**
- `Core/Parsing.ts` (611 bytes)
- `Core/JsonParser.ts` (3,607 bytes)
- `Core/OutputCleaner.ts` (4,665 bytes)
- `Core/SuggestionParser.ts` (3,232 bytes)

**Local:**
- `Parsing/JsonParser.ts` (8,361 bytes) - ENHANCED
- `Parsing/OutputCleaner.ts` (4,795 bytes) - ENHANCED
- `Parsing/SuggestionParser.ts` (3,320 bytes) - ENHANCED
- `Parsing/index.ts` (1,135 bytes)

**Recommendation:** Local versions are MORE comprehensive. Review upstream for any useful utilities but likely no merge needed.

---

### 4.2 Build Configuration

**Files:**
- `vite.config.ts` - Upstream improvements
- `tsconfig.json` - Upstream TypeScript config
- `package.json` - Upstream dependencies
- `package-lock.json` - Regenerate after merge

**Action Required:**
1. ⚠️  Compare vite.config.ts line-by-line
2. ⚠️  Merge upstream improvements
3. ⚠️  Update dependencies from upstream package.json
4. ⚠️  Regenerate package-lock.json

**Estimated Effort:** 2-4 hours

---

### 4.3 Documentation

**Upstream:**
- `README.md` - Upstream version

**Local:**
- `README.md` - Enhanced with local features
- `COMPREHENSIVE_GAP_ANALYSIS.md` - Local only
- `FINAL_INTEGRATION_SUMMARY.md` - Local only
- Multiple gap analysis reports - Local only

**Recommendation:** Preserve local documentation, merge any upstream README improvements.

---

## Migration Checklist

### Phase 1: CRITICAL ✅ COMPLETE

- [x] StateSerializer framework (15 files)
- [x] Custom mode handlers (MathSolver, GenerativeUI, React)
- [x] Prompt templates (2 files)
- [ ] Core/App.ts integration with StateSerializer
- [ ] Core/ConfigManager.ts integration with StateSerializer

### Phase 2: HIGH PRIORITY

- [ ] Routing enhancements (6 files)
- [ ] UI Components - Critical (10 files)
- [ ] Core utilities merge (5 files)

### Phase 3: MEDIUM PRIORITY

- [ ] Mode enhancements (15 files)
- [ ] DiffModal components (6 files)
- [ ] RenderMathMarkdown merge (2 files)

### Phase 4: LOW PRIORITY

- [ ] CSS reorganization review
- [ ] UI management files review
- [ ] Build configuration merge
- [ ] Documentation updates

---

## File Status Matrix

| File Path | Upstream | Local | Action | Priority |
|-----------|----------|-------|--------|----------|
| `Core/StateSerializer/*` | ✅ 15 files | ❌ None | ✅ MERGED | ✅ P0 |
| `Deepthink/Prompt Templates/*` | ✅ 2 files | ❌ None | ✅ MERGED | ✅ P0 |
| `Core/App.ts` | ✅ Enhanced | ✅ Enhanced | ⚠️ MERGE | ⚠️ P1 |
| `Core/ConfigManager.ts` | ✅ Base | ✅ Enhanced | ⚠️ MERGE | ⚠️ P1 |
| `Routing/ApiCallEstimator.ts` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P1 |
| `Routing/ProviderManager.ts` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P1 |
| `Components/CodeMirrorFileEditor.tsx` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P1 |
| `Components/FileUpload.tsx` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P1 |
| `Agentic/ArxivAPI.ts` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P2 |
| `Deepthink/DeepthinkIterativeHistory.ts` | ✅ NEW | ❌ None | ✅ COPY | ⚠️ P2 |
| `Parsing/*` | ✅ Base | ✅ Enhanced | ⚠️ REVIEW | ❌ P4 |

---

## Risk Assessment

### HIGH RISK
- **Core/App.ts merge** - Could break initialization
  - Mitigation: Test thoroughly, keep backup
- **StateSerializer integration** - Could break export/import
  - Mitigation: Test all modes, preserve local handlers

### MEDIUM RISK
- **CSS reorganization** - Could affect styling
  - Mitigation: Visual testing, screenshot comparison
- **Routing changes** - Could affect API calls
  - Mitigation: Test all providers, keep local ICR config

### LOW RISK
- **Component updates** - Mostly additive
  - Mitigation: Test each component
- **Documentation** - No functional impact
  - Mitigation: Review for accuracy

---

## Estimated Total Effort

| Phase | Files | Effort |
|-------|-------|--------|
| Phase 1 (Critical) | 17 | 6-8 hours |
| Phase 2 (High) | 16 | 7-10 hours |
| Phase 3 (Medium) | 21 | 9-13 hours |
| Phase 4 (Low) | 16 | 6-10 hours |
| **TOTAL** | **70+** | **28-41 hours (1-2 weeks)** |

---

## Next Actions

1. ✅ Complete Phase 1.6-1.7 (Core Integration)
2. ⏳ Copy Routing enhancements
3. ⏳ Copy UI Components (CodeMirrorFileEditor, FileUpload)
4. ⏳ Merge Core utilities
5. ⏳ Continue with Phase 2-4

---

**Document Version:** 3.0 (Complete Master Plan)  
**Created:** 2026-02-17  
**Status:** ⏳ Phase 1 Complete, Ready for Phase 2
