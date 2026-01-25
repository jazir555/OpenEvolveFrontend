# OpenEvolve Documentation Index

Complete index of all OpenEvolve documentation with descriptions and links.

---

## Core Documentation

### [README.md](README.md)
**Main project README**

**Contents:**
- Project overview and features
- Quick start guide
- **Configuration System** (NEW)
- Architecture overview
- Usage examples

**Best for:**
- First-time users
- Understanding what OpenEvolve does
- Getting started quickly

**Key sections:**
- Configuration System (lines 175-315)
- Adapter Pattern
- Migration Guide summary
- Available Configuration Presets
- Centralized Import Management

---

### [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
**Complete guide for migrating from old to new configuration system**

**Contents:**
- Quick reference table (old vs new)
- Why migrate (benefits)
- Step-by-step migration instructions
- 10 common migration patterns with examples
- Validation and testing procedures
- Rollback strategies
- FAQ

**Best for:**
- Developers migrating existing code
- Understanding the differences between old and new systems
- Learning best practices

**Key sections:**
- Quick Reference (comparison table)
- Step-by-Step Migration (5 steps)
- Common Patterns (10 detailed examples)
- Validation and Testing (5 test procedures)
- Rollback Strategy (4 rollback options)

---

### [API_REFERENCE.md](API_REFERENCE.md)
**Complete API documentation for all classes and functions**

**Contents:**
- UnifiedConfiguration class (all methods and properties)
- Factory functions
- EvolutionAdapter class
- AdversarialAdapter class
- Import system (openevolve_imports)
- API classes (EvolutionAPI, AdversarialAPI, etc.)
- Result types (EvolutionResult, AdversarialResult)
- Exceptions

**Best for:**
- Developers needing detailed API information
- Understanding function signatures and parameters
- Looking up specific methods or properties

**Key sections:**
- UnifiedConfiguration (main class)
- Factory Functions (8 functions)
- EvolutionAdapter (adapter pattern)
- Import System (centralized imports)
- Result Types (structured results)

---

### [ARCHITECTURE.md](ARCHITECTURE.md)
**System architecture and design documentation**

**Contents:**
- Configuration system architecture
- Data flow diagrams
- Component details (UnifiedConfiguration, adapters, etc.)
- Design patterns (6 patterns used)
- Module relationships
- Extension points
- Performance considerations

**Best for:**
- Understanding system design
- Architects and senior developers
- Contributors to the codebase
- Performance optimization

**Key sections:**
- Configuration System (architecture diagram)
- Data Flow (flow chart)
- Component Details (5 main components)
- Design Patterns (6 patterns with examples)
- Extension Points (adding new features)

---

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
**Common issues and solutions**

**Contents:**
- Import issues (3 common issues)
- Configuration issues (4 common issues)
- Validation issues (2 common issues)
- Execution issues (3 common issues)
- Performance issues (3 common issues)
- Migration issues (2 common issues)
- Debugging tips (5 techniques)
- FAQ

**Best for:**
- Troubleshooting problems
- Debugging code
- Understanding error messages
- Performance tuning

**Key sections:**
- Import Issues (module availability, circular imports)
- Configuration Issues (parameters, defaults, merging)
- Execution Issues (adapter failures, timeouts)
- Debugging Tips (logging, tracing, profiling)

---

## Quick Navigation Guides

### By User Type

**New Users:**
1. Start with [README.md](README.md) - Overview
2. Read "Quick Start" section
3. Try examples in demo files

**Developers:**
1. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Understand changes
2. Reference [API_REFERENCE.md](API_REFERENCE.md) - Look up APIs
3. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Solve issues

**Architects:**
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. Review [README.md](README.md) "Configuration System" section
3. Study design patterns in [ARCHITECTURE.md](ARCHITECTURE.md)

**Contributors:**
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - "Extension Points"
2. Study [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Common patterns
3. Reference [API_REFERENCE.md](API_REFERENCE.md) - API details

---

### By Task

**I want to...**

**...get started quickly:**
→ [README.md](README.md) - Quick Start section

**...migrate my code:**
→ [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Step-by-Step Migration

**...look up an API:**
→ [API_REFERENCE.md](API_REFERENCE.md) - Find your class/function

**...understand the architecture:**
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Overview and diagrams

**...fix a problem:**
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Find your issue

**...add a new feature:**
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Extension Points

**...learn best practices:**
→ [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Common Patterns

**...configure parameters:**
→ [API_REFERENCE.md](API_REFERENCE.md) - UnifiedConfiguration

**...use adapters:**
→ [API_REFERENCE.md](API_REFERENCE.md) - EvolutionAdapter/AdversarialAdapter

**...handle imports:**
→ [API_REFERENCE.md](API_REFERENCE.md) - Import System
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Import Issues

---

### By Concept

**Configuration System:**
- [README.md](README.md) - Configuration System section
- [API_REFERENCE.md](API_REFERENCE.md) - UnifiedConfiguration class
- [ARCHITECTURE.md](ARCHITECTURE.md) - Configuration System architecture
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Configuration creation patterns

**Adapter Pattern:**
- [README.md](README.md) - Adapter Pattern section
- [API_REFERENCE.md](API_REFERENCE.md) - EvolutionAdapter, AdversarialAdapter
- [ARCHITECTURE.md](ARCHITECTURE.md) - Adapter Pattern design
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Adapter usage patterns

**Import Management:**
- [README.md](README.md) - Centralized Import Management section
- [API_REFERENCE.md](API_REFERENCE.md) - Import System
- [ARCHITECTURE.md](ARCHITECTURE.md) - openevolve_imports component
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Import Issues

**Validation:**
- [API_REFERENCE.md](API_REFERENCE.md) - UnifiedConfiguration.validate()
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Configuration Validation pattern
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Validation Issues

**Migration:**
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Complete guide
- [README.md](README.md) - Migration Guide summary
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Migration Issues

---

## Related Files

### Core System Files

1. **unified_configuration.py**
   - UnifiedConfiguration class
   - Factory functions
   - Validation logic

2. **openevolve_imports.py**
   - Centralized import management
   - Availability flags
   - API wrapper classes

3. **evolution_adapter.py**
   - EvolutionAdapter class
   - Adapter factory
   - EvolutionResult type

4. **adversarial_adapter.py**
   - AdversarialAdapter class
   - Adapter factory
   - AdversarialResult type

### Demo Files (with migration examples)

1. **demo_evolution_maker.py**
   - Shows new import pattern
   - Adapter usage examples
   - Migration comments

2. **demo_adversarial_maker.py**
   - Adversarial testing examples
   - New patterns

3. **Other demo files**
   - Various integration examples
   - See each file for specific patterns

---

## Documentation Standards

All OpenEvolve documentation follows these standards:

### Formatting
- Markdown format
- Clear section headings
- Code examples with syntax highlighting
- Tables for comparisons and references

### Code Examples
- Complete, runnable examples
- Before/after comparisons for migrations
- Comments explaining key points
- Error handling shown

### Structure
- Table of contents for navigation
- Logical section grouping
- Cross-references between documents
- Consistent terminology

### Audience
- Clear language
- Minimal jargon
- Explanations for complex concepts
- Practical focus

---

## Reading Paths

### Path 1: Quick Start (15 minutes)
1. [README.md](README.md) - Read "Configuration System" section
2. Try a demo file (e.g., `demo_evolution_maker.py`)
3. Reference [API_REFERENCE.md](API_REFERENCE.md) as needed

### Path 2: Full Understanding (2 hours)
1. [README.md](README.md) - Complete README
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
3. [API_REFERENCE.md](API_REFERENCE.md) - Study APIs
4. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Learn patterns

### Path 3: Migration (1 hour)
1. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Read completely
2. Try "Step-by-Step Migration" with your code
3. Reference [API_REFERENCE.md](API_REFERENCE.md) for new APIs
4. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if issues arise

### Path 4: Deep Dive (4 hours)
1. [README.md](README.md) - Complete README
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Complete architecture
3. [API_REFERENCE.md](API_REFERENCE.md) - All APIs
4. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - All patterns
5. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - All issues
6. Study demo files
7. Review source code

---

## Getting Help

If you can't find what you need:

1. **Check all documentation** - Use this index
2. **Read demo files** - They contain practical examples
3. **Enable debug logging** - See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
4. **Check import status** - `from openevolve_imports import print_import_status`
5. **Review source code** - Well-documented with docstrings

---

## Contributing to Documentation

When adding new features:

1. **Update API_REFERENCE.md** - Document new APIs
2. **Update ARCHITECTURE.md** - Explain architecture changes
3. **Add to MIGRATION_GUIDE.md** - If it's a breaking change
4. **Create demo file** - Show usage examples
5. **Update this index** - Add new documentation

---

**Last Updated:** 2025-01-03
**Version:** 1.0.0

For the latest documentation, check the OpenEvolve repository.
