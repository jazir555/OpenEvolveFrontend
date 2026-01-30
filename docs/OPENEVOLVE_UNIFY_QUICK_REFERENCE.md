# 🚀 OpenEvolve Plugin Unification - Quick Reference

**Agent quick reference for the 3-way plugin merge.**

---

## 🎯 The Mission

Merge **THREE** OpenEvolve plugins into **ONE** unified standalone plugin.

### Source Plugins
1. **`OpenEvolve-Plugin/`** - 26 components, services, stores (~5,000 LOC)
2. **`openevolve-bubblelab-plugin/`** - Node system, registry (~2,000 LOC)
3. **`BubbleLab/apps/bubble-studio/src/plugins/openevolve/`** - Embedded ❌

### Target
**`OpenEvolve-Plugin/`** - Unified plugin with ALL features from all three

### Golden Rule
**ZERO FEATURE LOSS** - Keep every unique feature from all three plugins

---

## 📦 What Each Plugin Brings

### Plugin 1: OpenEvolve-Plugin/
✅ **UI Components** (26 total)
- Analytics, Knowledge, LeanAide, Pages, Shared

✅ **Services Layer**
- API clients, React hooks, WebSocket support

✅ **State Management**
- Zustand stores

✅ **Schemas**
- 10 workflow types

### Plugin 2: openevolve-bubblelab-plugin/
✅ **Node System**
- BaseNode, registry, factory

✅ **Enhanced Config**
- Advanced config panels, tabs

✅ **Plugin Factory**
- createPlugin utilities

✅ **Advanced Types**
- Enhanced plugin types

### Plugin 3: Embedded
✅ **PluginDefinition**
- Official BubbleLab interface

✅ **Service Definitions**
- 10 services with icons

✅ **API Integration**
- Complete endpoint config

✅ **Lifecycle Hooks**
- onBeforeExecute, onAfterExecute, onError

---

## 🗺️ 10-Phase Overview

```
Phase 1  → Complete Feature Inventory
Phase 2  → Architecture Design
Phase 3  → Core Infrastructure Merge
Phase 4  → Component Layer Merge
Phase 5  → Node System Integration
Phase 6  → Services & State Management
Phase 7  → Schema Unification
Phase 8  → Documentation & Examples
Phase 9  → Testing & Validation
Phase 10 → BubbleLab Integration & Cleanup
```

---

## ⚡ Quick Start Commands

### Phase 1: Inventory
```bash
# Scan all plugins
find OpenEvolve-Plugin/src -type f -name "*.tsx" -o -name "*.ts" | wc -l
find openevolve-bubblelab-plugin/src -type f -name "*.tsx" -o -name "*.ts" | wc -l
find BubbleLab/apps/bubble-studio/src/plugins/openevolve -type f

# List all components
ls OpenEvolve-Plugin/src/components/
ls openevolve-bubblelab-plugin/src/components/

# Compare schemas
diff OpenEvolve-Plugin/src/schemas/ BubbleLab/apps/bubble-studio/src/plugins/openevolve/schemas/
```

### Phase 3: Core Merge
```bash
# Create merged structure
mkdir -p OpenEvolve-Plugin/src/core/types
mkdir -p OpenEvolve-Plugin/src/core/utils
mkdir -p OpenEvolve-Plugin/src/core/constants

# Merge types
# Merge utilities
# Add plugin definition
```

### Phase 4: Components
```bash
# Count components
find OpenEvolve-Plugin/src/components -name "*.tsx" | wc -l  # Should be 26
find openevolve-bubblelab-plugin/src/components -name "*.tsx" | wc -l

# Migrate all components to unified structure
```

### Phase 9: Testing
```bash
cd OpenEvolve-Plugin
npm install
npm run build
npm test

cd ../BubbleLab
npm run build
```

---

## 📊 Feature Checklist

### Must End Up With (From All 3 Plugins)

**From Plugin 1 (OpenEvolve-Plugin/):**
- [ ] All 26 UI components
- [ ] All API service clients (10 services)
- [ ] All React hooks
- [ ] All Zustand stores
- [ ] All schemas
- [ ] All types
- [ ] All utilities
- [ ] All assets (icons, images)

**From Plugin 2 (openevolve-bubblelab-plugin/):**
- [ ] Complete node system (BaseNode, etc.)
- [ ] Node registry
- [ ] Enhanced config panels
- [ ] Plugin factory utilities
- [ ] Advanced types
- [ ] Node UI components
- [ ] Config tabs

**From Plugin 3 (Embedded):**
- [ ] PluginDefinition
- [ ] Service definitions (10)
- [ ] API endpoints config
- [ ] Lifecycle hooks
- [ ] Icon references
- [ ] Best schema versions

**Total Expected:**
- 30+ components
- 10+ services
- 10+ schemas
- Complete node system
- Full plugin infrastructure

---

## 🚨 Red Flags

### Critical Issues
- 🔴 **Missing feature** - Any feature from source plugins not present
- 🔴 **Build failure** - Unified plugin doesn't build
- 🔴 **Type errors** - TypeScript compilation fails
- 🔴 **Import errors** - Broken imports after merge
- 🔴 **Feature broken** - Previously working feature now broken

### Warning Signs
- 🟡 **Duplicate code** - Same logic in multiple places
- 🟡 **Naming inconsistency** - Confusing or inconsistent names
- 🟡 **Missing documentation** - Undocumented features
- 🟡 **Circular dependencies** - Import cycles

---

## ✅ Phase Completion Checklist

Use this before declaring your phase complete:

### All Phases
- [ ] All assigned deliverables complete
- [ ] Zero features lost (check against Phase 1 inventory)
- [ ] Code compiles without errors
- [ ] Changes documented
- [ ] Ready for handoff

### Phase-Specific

**Phase 1**
- [ ] Every feature from all 3 plugins cataloged
- [ ] Feature matrix complete
- [ ] Schema comparison done
- [ ] Overlaps identified
- [ ] Unique features marked

**Phase 3**
- [ ] Type system merged
- [ ] Utilities merged
- [ ] Plugin definition added
- [ ] Core infrastructure working
- [ ] Build configured

**Phase 4**
- [ ] All 30+ components present
- [ ] All components render
- [ ] No naming conflicts
- [ ] Imports resolved
- [ ] Components organized

**Phase 10**
- [ ] BubbleLab imports external plugin
- [ ] Embedded plugin deleted
- [ ] BubbleLab builds successfully
- [ ] All features working
- [ ] AIR GAP compliant

---

## 📝 Progress Report Template

```markdown
## Phase X Progress Report

**Agent**: [Your number]
**Date**: [Timestamp]

### Completed
- [ ] Task 1 - Description
- [ ] Task 2 - Description

### In Progress
- [ ] Task 3 - Description (50% complete)

### Blocked
- [ ] Task 4 - Description
  **Reason**: [Explain why]
  **Help needed**: [What you need]

### Features Added
- Features from Plugin X: [List]
- Features from Plugin Y: [List]
- Features from Plugin Z: [List]

### Issues Found
1. [Issue description]
   - Impact: [High/Medium/Low]
   - Resolution: [How you fixed it]

### Validation
- [ ] Zero feature loss verified
- [ ] Build successful
- [ ] No TypeScript errors
- [ ] Ready for next phase

### Next Phase
- Handoff notes for next agent
- Warnings or things to watch
- Recommendations
```

---

## 🆘 Getting Help

### If You're Stuck
1. **Check Phase 1's inventory** - See what features exist
2. **Review architecture** - Check Phase 2's design
3. **Test incrementally** - Don't wait until the end
4. **Document the issue** - Write it down clearly
5. **Ask early** - Don't stay stuck

### Issue Escalation
```markdown
## Phase Issue Escalation

**Phase**: [Your phase]
**Agent**: [Your number]
**Task**: [What you're doing]
**Issue**: [Clear description]
**Attempted**: [What you tried]
**Impact**: [How this blocks progress]
**Help**: [Specific question or request]
```

---

## 🎯 Success Metrics

### Track These Numbers
- **Components**: Should have 30+ (26 from P1 + more from P2)
- **Services**: Should have 10+
- **Schemas**: Should have 10
- **Node Classes**: Should have 5+
- **Type Files**: Should have 10+
- **Utilities**: Should have 15+
- **Documentation Files**: Should have 8+
- **Examples**: Should have all from P2

### Quality Gates
- ✅ Zero TypeScript errors
- ✅ Zero ESLint warnings
- ✅ Build time < 60 seconds
- ✅ Bundle size < 500KB
- ✅ All tests passing
- ✅ Zero feature loss

---

## 🔗 Quick Links

- [Full Task Document](./OPENEVOLVE_UNIFIED_MERGE_TASK.md)
- [Phase 1 Task](./OPENEVOLVE_UNIFIED_MERGE_TASK.md#phase-1-complete-feature-inventory-agent-1)
- [Architecture](./OPENEVOLVE_UNIFIED_MERGE_TASK.md#proposed-unified-architecture)
- [Feature Matrix](./OPENEVOLVE_UNIFIED_MERGE_TASK.md#feature-matrix)

---

## 💡 Tips

1. **Start with Phase 1's inventory** - Know what you're merging
2. **Follow Phase 2's architecture** - Don't deviate from the plan
3. **Test as you go** - Don't wait until the end
4. **Communicate early** - Raise issues immediately
5. **Document everything** - Future agents will thank you

---

## 🎉 Final Result

When all 10 phases are complete:

```
✅ OpenEvolve-Plugin/ - ONE unified plugin
   ✅ All features from Plugin 1
   ✅ All features from Plugin 2
   ✅ All features from Plugin 3
   ✅ Zero feature loss
   ✅ Clean architecture
   ✅ Complete documentation

✅ BubbleLab/ - Clean core
   ✅ No embedded OpenEvolve code
   ✅ Uses external plugin
   ✅ Can update from upstream
   ✅ AIR GAP compliant
```

---

**Remember: ZERO FEATURE LOSS! Keep everything from all three plugins!**

*Good luck, agents! 🚀*
