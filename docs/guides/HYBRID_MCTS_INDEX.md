# Hybrid MCTS Framework - Complete Index

## Overview

This document provides a comprehensive index of all Hybrid MCTS Framework files and their purposes. The framework integrates three hybrid approaches combining Monte Carlo Tree Search (MCTS) with Evolutionary Algorithms for theorem proving in Lean 4.

## Core Implementation Files

### 1. hybrid_mcts_framework.py (66 KB, 1,878 lines)
**Main Framework Implementation**

Primary implementation containing all core components:

**Key Components:**
- `HybridMCTSConfig` - Unified configuration (40+ parameters)
- `HybridMCTSResult` - Unified result structure
- `HybridMCTSEngine` - Main search engine
- `HybridCache` - Caching system
- `HybridMCTSMonitor` - Monitoring system
- `AdaptiveHybridSelector` - Adaptive approach selection
- `HybridMCTSWithLeanAide` - LeanAide integration
- `HybridBenchmark` - Benchmarking system
- `CombinedHybridMCTS` - Multi-approach combination
- `HybridMCTSPresets` - Configuration presets
- `HybridMCTSWorkflowIntegrator` - Workflow integration

**Use this file for:**
- Importing the framework
- Creating custom configurations
- Extending functionality
- Understanding implementation details

### 2. test_hybrid_mcts_framework.py (34 KB, 990 lines)
**Comprehensive Test Suite**

Complete testing framework with 150+ test cases:

**Test Coverage:**
- Configuration validation
- Result serialization
- Cache operations
- Monitor functionality
- Selector logic
- LeanAide integration
- All three approaches
- Workflow integration
- Performance scenarios
- Integration tests

**Use this file for:**
- Running tests: `pytest test_hybrid_mcts_framework.py -v`
- Understanding expected behavior
- Writing additional tests
- Validation of changes

### 3. demo_hybrid_mcts.py (18 KB, 532 lines)
**Interactive Demonstrations**

Nine demonstration scripts showcasing framework capabilities:

**Demos:**
1. Basic Usage
2. Evolved Policies
3. Evolutionary Nodes
4. Coevolution
5. Adaptive Selection
6. Combined Search
7. Workflow Integration
8. Quick Utilities
9. Approach Comparison

**Use this file for:**
- Learning the framework
- Testing installations
- Demonstrating capabilities
- Interactive exploration

**Run:**
```bash
python demo_hybrid_mcts.py                    # All demos
python demo_hybrid_mcts.py --demo evolved     # Specific demo
python demo_hybrid_mcts.py --interactive      # Interactive mode
```

## Documentation Files

### 4. HYBRID_MCTS_FRAMEWORK_GUIDE.md (19 KB)
**Complete User Guide**

Comprehensive documentation for framework users:

**Contents:**
- Quick start guide
- Basic usage examples
- Core components reference
- Three hybrid approaches detailed guides
- Advanced features
- Configuration reference
- Preset descriptions
- Workflow integration
- Benchmarking
- Testing
- Performance considerations
- Troubleshooting
- Best practices

**Use this file for:**
- Learning the framework
- Reference during development
- Understanding features
- Troubleshooting issues

### 5. HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md (17 KB)
**Implementation Summary**

Technical summary of the framework implementation:

**Contents:**
- Files overview
- Key features implemented
- Architecture highlights
- Testing coverage
- Performance characteristics
- Usage examples
- Integration points
- Future enhancements
- Validation & QA

**Use this file for:**
- Understanding implementation
- Architecture decisions
- Performance analysis
- Integration planning

### 6. HYBRID_MCTS_QUICK_REFERENCE.md (8.9 KB)
**Quick Reference Card**

Condensed reference for common tasks:

**Contents:**
- Installation
- Basic usage
- Three approaches
- Advanced features
- Common parameters
- Workflow integration
- Testing commands
- Troubleshooting tips
- Import patterns
- Common patterns
- Tips

**Use this file for:**
- Quick lookup
- Common commands
- Fast reference
- Daily use

### 7. HYBRID_MCTS_ARCHITECTURE.md (52 KB)
**Architecture Documentation**

Detailed architecture and design documentation:

**Contents:**
- System architecture
- Component interaction
- Data flow diagrams
- Design patterns
- Extension points
- Module relationships
- Class hierarchies
- API contracts

**Use this file for:**
- Understanding architecture
- Extending the framework
- Design decisions
- Component relationships

### 8. HYBRID_MCTS_API.md (46 KB)
**API Reference**

Complete API documentation:

**Contents:**
- All classes documented
- All methods with signatures
- Parameters and return types
- Usage examples
- Type information
- Exceptions
- Dependencies

**Use this file for:**
- API lookup
- Method signatures
- Parameter reference
- Development

### 9. HYBRID_MCTS_EXAMPLES.md (29 KB)
**Usage Examples**

Comprehensive collection of usage examples:

**Contents:**
- Basic examples
- Advanced examples
- Integration examples
- Workflow examples
- Performance examples
- Customization examples

**Use this file for:**
- Learning by example
- Copy-paste code
- Understanding patterns
- Best practices

### 10. HYBRID_MCTS_INTEGRATION.md (17 KB)
**Integration Guide**

Guide for integrating with other systems:

**Contents:**
- OpenEvolve integration
- LeanAide integration
- MCTS integration
- Workflow integration
- API integration
- Custom integrations

**Use this file for:**
- Integrating framework
- Connecting components
- Custom workflows
- System architecture

### 11. HYBRID_MCTS_COMPARISON.md (24 KB)
**Approach Comparison**

Detailed comparison of the three approaches:

**Contents:**
- Approach descriptions
- Performance comparison
- Use case analysis
- Strengths and weaknesses
- Recommendation matrix
- Benchmark results

**Use this file for:**
- Selecting approaches
- Understanding differences
- Performance analysis
- Decision making

### 12. HYBRID_MCTS_GUIDE.md (29 KB)
**User Guide**

Comprehensive user guide:

**Contents:**
- Getting started
- Installation
- Configuration
- Usage patterns
- Best practices
- Troubleshooting
- FAQ

**Use this file for:**
- Onboarding
- Learning framework
- Common questions
- Best practices

### 13. HYBRID_MCTS_TEST_SUITE_README.md (12 KB)
**Test Suite Documentation**

Documentation for the test suite:

**Contents:**
- Test organization
- Test categories
- Running tests
- Writing tests
- Test coverage
- CI/CD integration

**Use this file for:**
- Understanding tests
- Writing new tests
- Running test suite
- Coverage analysis

### 14. HYBRID_MCTS_TEST_SUITE_COMPLETION_REPORT.md (8.8 KB)
**Test Completion Report**

Report on test suite completion:

**Contents:**
- Test statistics
- Coverage report
- Test results
- Validation status
- Quality metrics

**Use this file for:**
- Validation status
- Quality assessment
- Test coverage
- Compliance

### 15. HYBRID_MCTS_TEST_QUICK_START.md (2.7 KB)
**Testing Quick Start**

Quick guide for running tests:

**Contents:**
- Installation
- Running tests
- Common commands
- Troubleshooting

**Use this file for:**
- Quick testing
- Test commands
- Validation

### 16. test_hybrid_mcts.py (68 KB)
**Alternative Test File**

Additional test implementations and examples.

## File Relationships

```
Implementation
├── hybrid_mcts_framework.py          (Core framework)
├── test_hybrid_mcts_framework.py     (Test suite)
└── demo_hybrid_mcts.py               (Demonstrations)

Documentation (Start Here)
├── HYBRID_MCTS_QUICK_REFERENCE.md    ← Quick reference
├── HYBRID_MCTS_FRAMEWORK_GUIDE.md    ← User guide
└── HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md ← Technical summary

Detailed Documentation
├── HYBRID_MCTS_ARCHITECTURE.md       (Architecture)
├── HYBRID_MCTS_API.md                (API reference)
├── HYBRID_MCTS_EXAMPLES.md           (Examples)
├── HYBRID_MCTS_INTEGRATION.md        (Integration)
├── HYBRID_MCTS_COMPARISON.md         (Comparison)
└── HYBRID_MCTS_GUIDE.md              (User guide)

Testing Documentation
├── HYBRID_MCTS_TEST_SUITE_README.md  (Test documentation)
├── HYBRID_MCTS_TEST_SUITE_COMPLETION_REPORT.md (Report)
└── HYBRID_MCTS_TEST_QUICK_START.md   (Quick start)

Additional Files
└── test_hybrid_mcts.py              (Alternative tests)
```

## Quick Start Path

### For New Users
1. Start: `HYBRID_MCTS_QUICK_REFERENCE.md`
2. Learn: `HYBRID_MCTS_FRAMEWORK_GUIDE.md`
3. Examples: `demo_hybrid_mcts.py`
4. Practice: `test_hybrid_mcts_framework.py`

### For Developers
1. Architecture: `HYBRID_MCTS_ARCHITECTURE.md`
2. API: `HYBRID_MCTS_API.md`
3. Examples: `HYBRID_MCTS_EXAMPLES.md`
4. Code: `hybrid_mcts_framework.py`

### For Integrators
1. Overview: `HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md`
2. Integration: `HYBRID_MCTS_INTEGRATION.md`
3. API: `HYBRID_MCTS_API.md`
4. Examples: `HYBRID_MCTS_EXAMPLES.md`

### For Researchers
1. Comparison: `HYBRID_MCTS_COMPARISON.md`
2. Architecture: `HYBRID_MCTS_ARCHITECTURE.md`
3. Implementation: `HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md`
4. Code: `hybrid_mcts_framework.py`

## File Statistics

### Code Files
| File | Size | Lines | Purpose |
|------|------|-------|---------|
| hybrid_mcts_framework.py | 66 KB | 1,878 | Main implementation |
| test_hybrid_mcts_framework.py | 34 KB | 990 | Test suite |
| demo_hybrid_mcts.py | 18 KB | 532 | Demonstrations |
| **Total Code** | **118 KB** | **3,400** | **Complete framework** |

### Documentation Files
| File | Size | Type |
|------|------|------|
| HYBRID_MCTS_ARCHITECTURE.md | 52 KB | Architecture |
| HYBRID_MCTS_API.md | 46 KB | API reference |
| HYBRID_MCTS_EXAMPLES.md | 29 KB | Examples |
| HYBRID_MCTS_GUIDE.md | 29 KB | User guide |
| HYBRID_MCTS_COMPARISON.md | 24 KB | Comparison |
| HYBRID_MCTS_FRAMEWORK_GUIDE.md | 19 KB | Complete guide |
| HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md | 17 KB | Implementation |
| HYBRID_MCTS_INTEGRATION.md | 17 KB | Integration |
| HYBRID_MCTS_TEST_SUITE_README.md | 12 KB | Test docs |
| HYBRID_MCTS_QUICK_REFERENCE.md | 8.9 KB | Quick ref |
| HYBRID_MCTS_TEST_SUITE_COMPLETION_REPORT.md | 8.8 KB | Test report |
| HYBRID_MCTS_TEST_QUICK_START.md | 2.7 KB | Test quick start |
| **Total Docs** | **~290 KB** | **12 files** |

### All Files
- **Total Size**: ~408 KB
- **Total Files**: 16
- **Code Lines**: 3,400+
- **Documentation Pages**: 12

## Three Hybrid Approaches

### 1. Evolved Rollout Policies
**File**: `hybrid_mcts_framework.py` (lines 800-1000)

Uses genetic algorithms to evolve MCTS rollout policies:
- Policy genome representation
- Genetic operators (mutation, crossover)
- Fitness evaluation
- Online adaptation during search

**Best for**: Medium-complexity, structured problems

### 2. Evolutionary MCTS Nodes
**File**: `hybrid_mcts_framework.py` (lines 1000-1200)

Evolves populations of MCTS search trees:
- Node population management
- Convergence tracking
- Evolution at each node
- Parallel evaluation

**Best for**: High-complexity, deep search problems

### 3. Coevolving Decision Trees
**File**: `hybrid_mcts_framework.py` (lines 1200-1400)

Coevolves proof strategies with problem instances:
- Tree population evolution
- Pareto front optimization
- Complexity-fitness trade-off
- Coevolutionary cycles

**Best for**: Diverse strategies, open-ended problems

## Key Features

### Core Features
- ✅ Unified interface for all three approaches
- ✅ Adaptive approach selection
- ✅ Four combination strategies (best, voting, weighted, ensemble)
- ✅ Comprehensive caching system
- ✅ Real-time monitoring
- ✅ LeanAide formal verification
- ✅ Benchmarking and comparison
- ✅ Workflow integration
- ✅ 8 configuration presets
- ✅ Extensible architecture

### Advanced Features
- ✅ Parallel execution
- ✅ Distributed-ready design
- ✅ Checkpoint saving/loading
- ✅ Custom metrics
- ✅ Resource tracking
- ✅ Error handling
- ✅ Comprehensive logging
- ✅ Type hints throughout
- ✅ 150+ test cases
- ✅ Interactive demos

## Usage Summary

### Basic Import
```python
from hybrid_mcts_framework import (
    HybridMCTSEngine,
    HybridMCTSConfig,
    HybridMCTSApproach,
    create_framework_from_preset,
    quick_search,
)
```

### Basic Usage
```python
# Create engine
engine = create_framework_from_preset("balanced")

# Search
result = await engine.search("theorem test (a : nat) : a + 0 = a")

# Access results
print(f"Success: {result.success}")
print(f"Fitness: {result.best_fitness}")
print(f"Proof: {result.best_proof}")
```

### Running Tests
```bash
# All tests
pytest test_hybrid_mcts_framework.py -v

# Specific test
pytest test_hybrid_mcts_framework.py::TestHybridMCTSEngine -v

# Coverage
pytest test_hybrid_mcts_framework.py --cov=hybrid_mcts_framework
```

### Running Demos
```bash
# All demos
python demo_hybrid_mcts.py

# Specific demo
python demo_hybrid_mcts.py --demo evolved

# Interactive
python demo_hybrid_mcts.py --interactive
```

## Support Resources

### Learning Path
1. **Beginner**: Quick Reference → Framework Guide → Demos
2. **Intermediate**: Examples → API Reference → Tests
3. **Advanced**: Architecture → Implementation → Integration

### Documentation Flow
```
Quick Start
    ↓
Framework Guide
    ↓
Examples & API
    ↓
Architecture & Implementation
    ↓
Integration & Customization
```

### Troubleshooting
1. Check `HYBRID_MCTS_QUICK_REFERENCE.md` for common issues
2. See `HYBRID_MCTS_FRAMEWORK_GUIDE.md` for detailed solutions
3. Run tests: `pytest test_hybrid_mcts_framework.py`
4. Run demos: `python demo_hybrid_mcts.py`

## Version Information

- **Framework Version**: 1.0.0
- **Implementation Date**: December 30, 2025
- **Status**: Production Ready
- **Python Version**: 3.8+
- **Dependencies**: numpy, aiohttp, pytest, psutil (optional)

## Contact & Contributions

For questions, issues, or contributions:
- See main OpenEvolve documentation
- Check GitHub issues
- Review test cases for examples
- Refer to API documentation

---

**Last Updated**: 2025-12-30
**Total Framework Size**: 408 KB
**Implementation Status**: Complete ✅
