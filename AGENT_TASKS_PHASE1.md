# Agent Task Specification

## Agent 1 - Phase 1: Complete Feature Inventory

### Objective
Catalog EVERY feature from THREE OpenEvolve plugin locations while ensuring ZERO feature loss.

### Plugin Locations to Scan

1. **OpenEvolve-Plugin/**
   - Path: `C:/Users/mmeadow/Documents/OpenEvolve/Frontend/OpenEvolve-Plugin/`
   - Size: ~5,000 LOC, 100+ files
   - Focus: Complete UI component library, services, stores

2. **openevolve-bubblelab-plugin/**
   - Path: `C:/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve-bubblelab-plugin/`
   - Size: ~2,000 LOC, 30 files
   - Focus: Node system, registry, factory

3. **BubbleLab/apps/bubble-studio/src/plugins/openevolve/**
   - Path: `C:/Users/mmeadow/Documents/OpenEvolve/Frontend/BubbleLab/apps/bubble-studio/src/plugins/openevolve/`
   - Size: ~500 LOC, 11 files
   - Focus: PluginDefinition, service definitions

### Tasks

1. Scan Plugin 1 completely
2. Scan Plugin 2 completely
3. Scan Plugin 3 completely
4. Create feature matrix
5. Compare all 10 schemas

### Deliverables

1. `COMPLETE_FEATURE_INVENTORY.md`
2. `SCHEMA_COMPARISON_MATRIX.md`
3. `FEATURE_OVERLAP_ANALYSIS.md`
4. `UNIFICATION_STRATEGY.md`

### Tools to Use

- **Glob**: Find all files in each plugin
- **Read**: Read key files to understand functionality
- **Write**: Create deliverable markdown files

### Start Commands

```bash
cd C:/Users/mmeadow/Documents/OpenEvolve/Frontend

# Find all TypeScript/React files
glob OpenEvolve-Plugin/src/**/*.ts
glob openevolve-bubblelab-plugin/src/**/*.ts
glob BubbleLab/apps/bubble-studio/src/plugins/openevolve/**/*.ts
```

### Critical Success Factors

- ZERO feature loss - document EVERYTHING
- Be thorough and systematic
- Include exact file paths
- Note all components, services, hooks, stores
- Compare schemas in detail
