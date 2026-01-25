# Hephaestus Reference Scan Report

**Date**: 2026-01-24
**Scan Type**: Comprehensive codebase scan for "hephaestus" references (case-insensitive)
**Purpose**: Identify all Hephaestus references to be scrubbed after dependency removal

---

## Executive Summary

**Total Files with References**: **1,183 files**
- Python files: 130 files (~1,274 occurrences)
- Markdown documentation: 441 files
- TypeScript/TSX: 54 files
- Other: 558 files (JSON, YAML, TXT, etc.)

**Critical Finding**: Many references are **active integration code** that will break if Hephaestus is removed without proper replacement.

---

## Breakdown by File Type

### 1. Python Files (130 files, ~1,274 occurrences)

#### High-Priority Files with Active Integration Code

| File | Occurrences | Type | Priority |
|------|-------------|------|----------|
| `workflow_engine.py` | 31 | Sync functions | **CRITICAL** |
| `bubblelabs_maker_integration.py` | 65 | Class definitions + sync | **CRITICAL** |
| `openevolve_workflow_manager_integrated.py` | 22 | Integration | **HIGH** |
| `invention_planner_integrations.py` | 43 | Integration | **HIGH** |
| `scripts/update_phase6_crewai.py` | 67 | Migration script | **MEDIUM** |
| `ui_components.py` | 12 | UI references | **MEDIUM** |
| `sgd_orchestrator_agent.py` | 45 | Integration | **HIGH** |

#### Bridge Files (All references are migration comments)

| File | References | Status |
|------|------------|--------|
| `roma_crewai_bridge.py` | 2 | Migration comments |
| `roma_mdap_maker_crewai_bridge.py` | 2 | Migration comments |
| `crewai_unified_bridge.py` | 9 | Migration comments |
| `crewai_unified_flow.py` | 5 | Migration comments |
| `openevolve_crewai_bridge.py` | 5 | Migration comments |
| `decomposition_crewai_bridge.py` | 2 | Migration comments |
| `bubblelabs_crewai_bridge.py` | 6 | Migration comments |
| `ace_crewai_bridge.py` | 6 | Migration comments |
| `datapizza_crewai_bridge.py` | 4 | Migration comments |
| `claudiomiro_crewai_bridge.py` | 5 | Migration comments |
| `leanaide_crewai_bridge.py` | 5 | Migration comments |
| `steer_crewai_bridge.py` | 3 | Migration comments |

#### MCP Tools Files

| File | Occurrences | Type |
|------|-------------|------|
| `roma_mcp_tools.py` | 2 | Comments |
| `roma_mdap_maker_mcp_tools.py` | 1 | Comments |
| `openevolve_mcp_tools.py` | 7 | Comments |
| `bubblelabs_mcp_tools.py` | 3 | Comments |
| `datapizza_mcp_tools.py` | 2 | Comments |
| `claudiomiro_mcp_tools.py` | 2 | Comments |
| `leanaide_mcp_tools.py` | 2 | Comments |
| `steer_mcp_tools.py` | 7 | Comments |
| `ace_mcp_tools.py` | 1 | Comments |
| `guardrails_mcp_tools.py` | 1 | Comments |
| `c2c_mcp_tools.py` | 6 | Comments |
| `lmql_mcp_tools.py` | 1 | Comments |
| `decomposition_mcp_tools.py` | 9 | Comments |

### 2. Markdown Documentation Files (441 files)

#### High-Impact Documentation

| File | Occurrences | Content Type |
|------|-------------|--------------|
| `BUBBLELABS_MAKER_HEPHAEUSTUS_DOCUMENTATION.md` | 56 | Integration guide |
| `BUBBLELABS_MAKER_QUICK_REFERENCE.md` | 34 | Quick reference |
| `CODEBASE_ANALYSIS_COMPREHENSIVE_REPORT.md` | 29 | Analysis |
| `ARCHITECTURE.md` | 1 | Architecture |
| `BLUE_RED_GOLD_TEAM_INTEGRATION.md` | 4 | Integration docs |
| `CREWAI_MIGRATION_MASTER_TASKLIST.md` | 19 | Migration docs |
| `CREWAI_MIGRATION_FINAL_SUMMARY.md` | 7 | Migration docs |
| `CREWAI_MIGRATION_COMPLETE.md` | 8 | Migration docs |
| `README.md` | 1 | Main README |

### 3. TypeScript/TSX Files (54 files)

#### Core Plugin Files

| File | Type | Priority |
|------|------|----------|
| `OpenEvolve-Plugin/src/nodes/HephaestusNode.ts` | Node definition | **CRITICAL** |
| `OpenEvolve-Plugin/src/hooks/useHephaestus.ts` | Hook | **CRITICAL** |
| `OpenEvolve-Plugin/src/services/api/OpenEvolveAPI.ts` | API service | **HIGH** |
| `BubbleLab/integrations/openevolve/index.ts` | Integration export | **HIGH** |
| `openevolve-pygraphistry-plugin/src/components/HephaestusViz.tsx` | Visualization | **MEDIUM** |

---

## Categories of References

### Category 1: Active Integration Code (CRITICAL)

These references represent **functional code** that integrates with Hephaestus. Removing Hephaestus without addressing these will cause runtime errors.

#### Class Definitions

```python
# bubblelabs_maker_integration.py
class HephaestusDelegation:
    """Represents a delegation to/from Hephaestus"""

class HephaestusDelegationManager:
    """Manages delegations to/from Hephaestus"""
```

#### Sync Functions (workflow_engine.py)

```python
# Lines 2349-2361: Sync solution to Hephaestus
workflow_state.sync_solution_to_hephaestus_ticket(...)

# Lines 2428-2435: Sync critique to Hephaestus
workflow_state.sync_critique_to_hephaestus_ticket(...)

# Lines 2536-2543: Sync verification to Hephaestus
workflow_state.sync_verification_to_hephaestus_ticket(...)

# Lines 2595-2601: Sync subproblem status to Hephaestus
workflow_state.sync_subproblem_status_to_hephaestus(...)
```

#### Availability Flags

```python
# Multiple files check if Hephaestus is available
HEPHAEUSTUS_AVAILABLE = True/False
if HEPHAEUSTUS_AVAILABLE:
    # ... integration code
```

#### Sync Methods (bubblelabs_maker_integration.py)

```python
# Line 521-547
def sync_from_hephaestus(self) -> int:
    """Sync delegation statuses from Hephaestus"""
```

### Category 2: Migration Documentation (LOW PRIORITY)

These are comments and documentation explaining the migration from Hephaestus to CrewAI. These can be removed or updated once migration is complete.

#### File Headers

```python
# This file has been migrated from Hephaestus (AGPL) to CrewAI (MIT).
# All Hephaestus references have been replaced with CrewAI equivalents.
```

#### Bridge Documentation

```python
# CrewAI Unified Bridge - Complete Replacement for Hephaestus Unified Bridge
# This module provides the unified bridge interface that replaces hephaestus_unified_bridge.py
```

### Category 3: TypeScript/React Components (HIGH PRIORITY)

#### Node Definition

```typescript
// OpenEvolve-Plugin/src/nodes/HephaestusNode.ts
export class HephaestusNode {
    // ...
}
```

#### Hook

```typescript
// OpenEvolve-Plugin/src/hooks/useHephaestus.ts
export function useHephaestus() {
    // ...
}
```

### Category 4: Historical Documentation (MEDIUM PRIORITY)

#### Migration Reports

- `CREWAI_MIGRATION_MASTER_TASKLIST.md`
- `CREWAI_MIGRATION_FINAL_SUMMARY.md`
- `CREWAI_MIGRATION_COMPLETE.md`
- `PHASE6_MIGRATION_COMPLETE.md`

These documents describe the migration process and can be archived or removed once migration is verified complete.

---

## Files Requiring Immediate Action

### Critical (Must Fix Before Removal)

1. **workflow_engine.py** (31 occurrences)
   - Remove all `sync_*_to_hephaestus_ticket()` calls
   - Remove Hephaestus status checks
   - Update workflow state management

2. **bubblelabs_maker_integration.py** (65 occurrences)
   - Rename `HephaestusDelegation` → `CrewAIDelegation`
   - Rename `HephaestusDelegationManager` → `CrewAIDelegationManager`
   - Remove `sync_from_hephaestus()` method
   - Update all UI references

3. **OpenEvolve-Plugin/src/nodes/HephaestusNode.ts**
   - Rename to `CrewAINode.ts` or remove
   - Update all imports

4. **OpenEvolve-Plugin/src/hooks/useHephaestus.ts**
   - Rename to `useCrewAI.ts` or remove
   - Update all imports

### High Priority (Should Fix Soon)

5. **sgd_orchestrator_agent.py** (45 occurrences)
6. **invention_planner_integrations.py** (43 occurrences)
7. **openevolve_workflow_manager_integrated.py** (22 occurrences)
8. **BubbleLab/integrations/openevolve/index.ts** (exports HephaestusBubble)

### Medium Priority (Clean Up)

9. All bridge files (comments only)
10. All MCP tools files (comments only)
11. Documentation files (441 markdown files)

---

## Recommended Action Plan

### Phase 1: Critical Code (DO FIRST)

1. **Remove Hephaestus sync functions from workflow_engine.py**
   - Lines 2349-2361: `sync_solution_to_hephaestus_ticket()`
   - Lines 2428-2435: `sync_critique_to_hephaestus_ticket()`
   - Lines 2536-2543: `sync_verification_to_hephaestus_ticket()`
   - Lines 2595-2601: `sync_subproblem_status_to_hephaestus()`

2. **Refactor bubblelabs_maker_integration.py**
   ```python
   # Rename classes
   HephaestusDelegation → CrewAIDelegation
   HephaestusDelegationManager → CrewAIDelegationManager

   # Remove methods
   sync_from_hephaestus()
   _render_hephaestus_tracker()

   # Update UI
   "Sync from Hephaestus" → "Sync from CrewAI"
   ```

3. **Update TypeScript files**
   - Rename `HephaestusNode.ts` → `CrewAINode.ts`
   - Rename `useHephaestus.ts` → `useCrewAI.ts`
   - Update all imports across 54 files

### Phase 2: Integration Points

4. **Update remaining Python files**
   - Remove `HEPHAEUSTUS_AVAILABLE` checks
   - Update conditional logic that depends on Hephaestus
   - Remove unused imports

5. **Update BubbleLab integration**
   - Remove `HephaestusBubble` from `BubbleLab/integrations/openevolve/index.ts`
   - Update or remove hephaestus-bubble implementation

### Phase 3: Documentation Cleanup

6. **Archive migration documentation**
   - Move to `docs/archive/migration/`
   - Or remove if migration is verified complete

7. **Update active documentation**
   - Remove Hephaestus from README.md
   - Update ARCHITECTURE.md
   - Update integration guides

8. **Clean up comments**
   - Remove "migrated from Hephaestus" headers
   - Update bridge documentation
   - Remove legacy references

---

## Specific File-by-File Recommendations

### workflow_engine.py
```python
# REMOVE these lines:
- Lines 2349-2361: sync_solution_to_hephaestus_ticket()
- Lines 2428-2435: sync_critique_to_hephaestus_ticket()
- Lines 2536-2543: sync_verification_to_hephaestus_ticket()
- Lines 2595-2601: sync_subproblem_status_to_hephaestus()

# SEARCH AND REPLACE:
"Sync to Hephaestus" → "CrewAI workflow update"
"Could not sync to Hephaestus" → "Could not update CrewAI workflow"
```

### bubblelabs_maker_integration.py
```python
# RENAME CLASSES:
HephaestusDelegation → CrewAIDelegation
HephaestusDelegationManager → CrewAIDelegationManager

# REMOVE METHODS:
sync_from_hephaestus()
_render_hephaustus_tracker()

# UPDATE BUTTON TEXT:
"🔄 Sync from Hephaestus" → "🔄 Sync Delegations"

# REMOVE IMPORTS:
if HEPHAEUSTUS_AVAILABLE:  # Remove this check
```

### OpenEvolve-Plugin Files
```typescript
// RENAME FILES:
HephaestusNode.ts → CrewAINode.ts
useHephaestus.ts → useCrewAI.ts
HephaestusViz.tsx → CrewAIViz.tsx

// UPDATE EXPORTS:
export { HephaestusBubble } → export { CrewAIBubble }

// UPDATE ALL IMPORTS (54 files)
import { HephaestusNode } → import { CrewAINode }
import { useHephaestus } → import { useCrewAI }
```

### Bridge Files (All)
```python
# UPDATE HEADERS:
"Replaces Hephaestus" → "CrewAI orchestration bridge"
"Backward compatible with Hephaestus" → "CrewAI workflow bridge"

# These are just comments, so low priority but should be cleaned up
```

---

## Risk Assessment

### High Risk (Breaking Changes)

| Component | Risk | Impact |
|-----------|------|--------|
| `workflow_engine.py` sync functions | **CRITICAL** | Workflow state sync will fail |
| `HephaestusDelegation` classes | **CRITICAL** | BubbleLabs integration will break |
| TypeScript node definitions | **HIGH** | OpenEvolve plugin will break |
| BubbleLab exports | **HIGH** | Plugin ecosystem will break |

### Medium Risk (Confusion)

| Component | Risk | Impact |
|-----------|------|--------|
| Documentation references | **MEDIUM** | Outdated docs confuse users |
| Migration comments | **LOW** | Developer confusion |
| Available flags | **MEDIUM** | Conditional logic may fail |

### Low Risk (Cosmetic)

| Component | Risk | Impact |
|-----------|------|--------|
| Historical migration docs | **LOW** | Just clutter |
| Comment references | **LOW** | No functional impact |
| Test/verify scripts | **LOW** | May fail but not production |

---

## Summary Statistics

```
Total References Found:
├── Python Files:           130 files  (~1,274 occurrences)
├── Markdown Documentation: 441 files  (count varies)
├── TypeScript/TSX:          54 files
└── Other:                  558 files

Priority Breakdown:
├── CRITICAL (must fix):     4 files
├── HIGH (should fix):      30+ files
├── MEDIUM (clean up):     100+ files
└── LOW (documentation):  1000+ files

Estimated Effort:
├── Critical code fixes:     4-8 hours
├── Integration updates:     8-12 hours
├── TypeScript updates:      4-6 hours
├── Documentation cleanup:   2-4 hours
└── TOTAL:                  18-30 hours
```

---

## Next Steps

1. ✅ **Review this report** and confirm removal scope
2. ⬜ **Create feature branch** for hephaestus removal
3. ⬜ **Start with critical files** (workflow_engine.py, bubblelabs_maker_integration.py)
4. ⬜ **Update TypeScript files** (OpenEvolve-Plugin)
5. ⬜ **Run tests** to catch breaking changes
6. ⬜ **Update documentation**
7. ⬜ **Create PR** for review
8. ⬜ **Merge and verify**

---

**Generated**: 2026-01-24
**Scan Method**: `grep -ri "hephaestus" --include="*.py" --include="*.md" --include="*.ts" --include="*.tsx"`
**Author**: Claude Code (Scan Tool)
**Project**: OpenEvolve Frontend
