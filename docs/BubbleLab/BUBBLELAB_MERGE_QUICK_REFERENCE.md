# 🚀 BubbleLabs Plugin Merge - Agent Quick Reference

**Quick Reference Guide for Agents executing the BubbleLabs plugin merge task.**

---

## 📍 Location

- **Task Document**: `BUBBLELAB_PLUGIN_MERGE_TASK.md`
- **Plugin 1**: `openevolve-bubblelab-plugin/`
- **Plugin 2**: `leanaide-bubblelab-plugin/`
- **Target**: `openevolve-bubblelab-plugin-merged/` (new)

---

## 🎯 Your Mission

Merge two OpenEvolve BubbleLab plugins into **ONE unified plugin** with **ALL features retained**.

---

## ⚡ Quick Start Checklist

### Before You Begin
- [ ] Read the full task document
- [ ] Identify your assigned phase
- [ ] Review previous phase outputs
- [ ] Understand dependencies

### During Execution
- [ ] Follow your phase instructions
- [ ] Document all decisions
- [ ] Track progress in log
- [ ] Mark issues immediately

### Before Handoff
- [ ] Complete all deliverables
- [ ] Validate outputs
- [ ] Update documentation
- [ ] Notify next agent

---

## 🗺️ Phase Overview

```
Phase 1 (Agent 1) → Discovery & Analysis
     ↓
Phase 2 (Agent 2) → Architecture Design
     ↓
Phase 3 (Agent 3) → Code Migration
     ↓
Phase 4 (Agent 4) → Integration & Resolution
     ↓
Phase 5 (Agent 5) → Testing & Validation
     ↓
Phase 6 (Agent 6) → Documentation & Release
```

---

## 📦 Plugin Summary

### openevolve-bubblelab-plugin
- **Purpose**: General OpenEvolve workflows
- **Key Features**:
  - Evolution nodes (MCTS)
  - Adversarial training
  - Decomposition engine
  - Knowledge engine
  - crewai bridge
  - MDAP/MAKER integration
- **Exports**: 50+ nodes, components, hooks

### leanaide-bubblelab-plugin
- **Purpose**: LeanAIDE formal verification
- **Key Features**:
  - LeanAIDE client
  - Verification UI
  - RAGBits search
  - Autoformalization
  - Analytics dashboard
- **Exports**: 30+ services, components, hooks

---

## 🎨 Naming Conflicts to Resolve

### Known Conflicts
1. **Node Systems**: Both have node registries
   - Solution: Namespace by domain (Evolution vs Verification)

2. **Config Panels**: Both have config UI
   - Solution: Merge into tabbed interface

3. **Service Clients**: Both have service layers
   - Solution: Unified service interface

4. **Plugin Registries**: Both have plugin systems
   - Solution: Type-discriminated union

5. **Type Names**: Duplicate types (Config, Result, etc.)
   - Solution: Domain-specific prefixes

---

## 🔧 Common Commands

### Exploration
```bash
# List all files in a plugin
find openevolve-bubblelab-plugin/src -type f -name "*.ts*" | sort

# Find all exports
grep -r "export" openevolve-bubblelab-plugin/src --include="*.ts" | cut -d: -f1 | sort -u

# Find all imports
grep -r "import" leanaide-bubblelab-plugin/src --include="*.ts*" | head -50
```

### Analysis
```bash
# Count TypeScript files
find openevolve-bubblelab-plugin/src -name "*.ts*" | wc -l

# Find duplicate names
grep -r "export.*Node" openevolve-bubblelab-plugin/src leanaide-bubblelab-plugin/src

# Check dependencies
cat openevolve-bubblelab-plugin/package.json leanaide-bubblelab-plugin/package.json
```

### Migration
```bash
# Create backup
cp -r openevolve-bubblelab-plugin openevolve-bubblelab-plugin.backup

# Create merged directory
mkdir -p openevolve-bubblelab-plugin-merged/src/{core,nodes,components,services,hooks}

# Copy files
cp -r openevolve-bubblelab-plugin/src/* openevolve-bubblelab-plugin-merged/src/
```

### Validation
```bash
# Install dependencies
cd openevolve-bubblelab-plugin-merged && npm install

# Check types
npx tsc --noEmit

# Run tests
npm test

# Build
npm run build

# Check bundle size
ls -lh dist/
```

---

## 📝 Report Templates

### Phase 1 Report Template
```markdown
# Phase 1: Discovery & Analysis Report

## Feature Inventory
### Plugin 1 Features
- [ ] Evolution nodes (X files)
- [ ] Adversarial nodes (X files)
- [ ] Decomposition nodes (X files)
...

### Plugin 2 Features
- [ ] LeanAIDE client (X files)
- [ ] Verification UI (X files)
- [ ] RAGBits search (X files)
...

## Dependency Analysis
| Package | Plugin 1 Ver | Plugin 2 Ver | Conflict? |
|---------|--------------|--------------|-----------|
| react   | ^18.0.0      | ^18.0.0      | ❌        |
| ...     | ...          | ...          | ...       |

## Type Mapping
[Document all types and relationships]

## Export Compatibility
[Document all exports and compatibility]

## Issues Found
1. [Issue description]
2. [Issue description]
...
```

### Phase 2 Report Template
```markdown
# Phase 2: Architecture Design Report

## Proposed Directory Structure
```
[Merged structure]
```

## Namespace Strategy
[Explain naming resolution]

## Integration Architecture
[Explain how features integrate]

## Type System Design
[Explain type unification]

## Breaking Changes
[List all breaking changes]
```

---

## 🚨 Red Flags to Watch

### Critical Issues
- 🚨 **Feature Loss**: Any feature not present in merge
- 🚨 **Type Errors**: Unresolved TypeScript conflicts
- 🚨 **Import Failures**: Broken import paths
- 🚨 **Test Failures**: Tests not passing
- 🚨 **Build Failures**: Build not working

### Warning Signs
- ⚠️ **Duplicate Code**: Same logic in multiple places
- ⚠️ **Complex Dependencies**: Circular dependencies
- ⚠️ **Naming Inconsistency**: Confusing names
- ⚠️ **Missing Documentation**: Undocumented features
- ⚠️ **Performance Issues**: Slow operations

---

## ✅ Quality Gates

### Before Moving to Next Phase
- [ ] All deliverables complete
- [ ] All documentation written
- [ ] All issues resolved or documented
- [ ] Peer review completed
- [ ] Tests passing (if applicable)

### Phase-Specific Gates

**Phase 1**
- [ ] Feature inventory 100% complete
- [ ] All dependencies mapped
- [ ] All types cataloged
- [ ] All exports documented

**Phase 2**
- [ ] Architecture approved
- [ ] Naming strategy defined
- [ ] Type system designed
- [ ] Breaking changes identified

**Phase 3**
- [ ] All files migrated
- [ ] Build working
- [ ] No import errors
- [ ] Types compile

**Phase 4**
- [ ] All conflicts resolved
- [ ] All imports fixed
- [ ] All features integrated
- [ ] Config merged

**Phase 5**
- [ ] All tests passing
- [ ] Zero type errors
- [ ] Build successful
- [ ] All features validated

**Phase 6**
- [ ] All docs updated
- [ ] Migration guide written
- [ ] Examples working
- [ ] Release ready

---

## 🆘 Getting Help

### If You're Stuck
1. **Check the task document**: Full details in `BUBBLELAB_PLUGIN_MERGE_TASK.md`
2. **Review previous phases**: Look at what earlier agents did
3. **Document the issue**: Write it down with context
4. **Ask for help**: Tag in the main thread

### Issue Escalation Template
```markdown
## Issue Escalation

**Phase**: [Your phase]
**Agent**: [Agent number]
**Context**: [What you were doing]
**Issue**: [Description]
**Attempts**: [What you tried]
**Blocker**: [Yes/No - cannot proceed]
**Help Needed**: [Specific question or request]
```

---

## 📊 Progress Updates

### Daily Update Template
```markdown
## Progress Update - [Date]

**Agent**: [Your number/role]
**Phase**: [Your phase]

### Completed
- [ ] Task 1
- [ ] Task 2

### In Progress
- [ ] Task 3 (50% complete)

### Blocked
- [ ] Task 4 - [Reason]

### Plans
- Next: [What's next]

### Issues
- [Issue description]
```

---

## 🎯 Success Criteria

### Must Have
- ✅ Zero feature loss
- ✅ All tests passing
- ✅ Zero TypeScript errors
- ✅ Build successful
- ✅ Backward compatible exports

### Should Have
- ✅ Clean code organization
- ✅ Consistent naming
- ✅ Comprehensive docs
- ✅ Good performance

### Nice to Have
- ✅ Improved architecture
- ✅ Better DX (developer experience)
- ✅ Reduced bundle size
- ✅ Enhanced features

---

## 🔗 Quick Links

- [Full Task Document](./BUBBLELAB_PLUGIN_MERGE_TASK.md)
- [Plugin 1 README](./openevolve-bubblelab-plugin/README.md)
- [Plugin 2 README](./leanaide-bubblelab-plugin/README.md)
- [Project Architecture](./ARCHITECTURE.md)
- [Project Guidelines](./CLAUDE.md)

---

**Good luck! Remember: The goal is ZERO feature loss. Take your time, validate everything, document often.**
