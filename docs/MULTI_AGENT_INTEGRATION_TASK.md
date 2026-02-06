# Multi-Agent Project Integration Task Specification

**Date**: 2026-01-02
**Mission**: Integrate 7 high-priority external projects into OpenEvolve
**Approach**: Decoupled adapter pattern with multiple specialist agents
**Timeline**: Parallel execution across 7 specialist agents

---

## Integration Architecture Strategy

### Core Principles

1. **Zero Modification Policy**: Never modify source files of external projects
2. **Adapter Pattern**: Create bridge adapters that wrap external functionality
3. **Interface Abstraction**: Define OpenEvolve interfaces that adapters implement
4. **Version Isolation**: Use dependency management (conda/virtualenv) to prevent conflicts
5. **MCP Server Pattern**: Where possible, use MCP (Model Context Protocol) for isolation
6. **Configuration-Driven**: All integration behavior configurable via YAML/JSON
7. **Graceful Degradation**: System functions even if external projects are unavailable

### Directory Structure

```
openevolve/
├── integrations/                    # NEW: Integration layer
│   ├── __init__.py
│   ├── base/                        # Abstract interfaces
│   │   ├── __init__.py
│   │   ├── knowledge_interface.py   # Base knowledge graph interface
│   │   ├── extraction_interface.py  # Base extraction interface
│   │   ├── experimentation_interface.py  # Base experiment interface
│   │   ├── optimization_interface.py    # Base optimization interface
│   │   └── uq_interface.py          # Base UQ interface
│   ├── graphiti/                    # Graphiti integration
│   │   ├── __init__.py
│   │   ├── adapter.py               # Adapter implementation
│   │   ├── config.yaml              # Integration config
│   │   └── bridge.py                # OpenEvolve bridge
│   ├── oneke/                       # OneKE integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── config.yaml
│   │   └── bridge.py
│   ├── curie/                       # Curie integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── config.yaml
│   │   └── bridge.py
│   ├── neuromancer/                 # NeuroMANCER integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── config.yaml
│   │   └── bridge.py
│   ├── pygraphistry/                # pygraphistry integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── config.yaml
│   │   └── bridge.py
│   ├── uqtestfuns/                  # uqtestfuns integration
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── config.yaml
│   │   └── bridge.py
│   └── global_chem/                 # global-chem integration
│       ├── __init__.py
│       ├── adapter.py
│       ├── config.yaml
│       └── bridge.py
├── docs/
│   └── integrations/                # NEW: Integration guides
│       ├── GRAPHITI_INTEGRATION_GUIDE.md
│       ├── ONEKE_INTEGRATION_GUIDE.md
│       ├── CURIE_INTEGRATION_GUIDE.md
│       ├── NEUROMANCER_INTEGRATION_GUIDE.md
│       ├── PYGRAPHISTRY_INTEGRATION_GUIDE.md
│       ├── UQTESTFUNS_INTEGRATION_GUIDE.md
│       ├── GLOBAL_CHEM_INTEGRATION_GUIDE.md
│       └── INTEGRATION_ARCHITECTURE.md
└── projects/                        # External projects (git submodules)
    ├── graphiti/
    ├── OneKE/
    ├── curie/
    ├── neuromancer/
    ├── pygraphistry/
    ├── uqtestfuns/
    └── global-chem/
```

---

## Agent Task Assignments

### Agent 1: Graphiti Integration Specialist

**Mission**: Integrate Graphiti temporally-aware knowledge graph
**Priority**: P0 (Critical)
**Gaps Filled**: GAP-14 (Temporal Knowledge), GAP-10 (Knowledge Extraction)

**Tasks**:
1. Create `integrations/base/knowledge_interface.py` with abstract knowledge graph interface
2. Create `integrations/graphiti/adapter.py` implementing the interface
3. Create `integrations/graphiti/bridge.py` connecting to OpenEvolve knowledge_engine
4. Create `integrations/graphiti/config.yaml` for Neo4j configuration
5. Write `docs/integrations/GRAPHITI_INTEGRATION_GUIDE.md`
6. Create tests in `tests/integrations/test_graphiti_integration.py`
7. Update `knowledge_engine/bedrock_kb.py` to use Graphiti adapter

**Requirements**:
- Support Neo4j and FalkorDB backends
- Implement temporal metadata tracking
- Hybrid search: semantic + BM25 + graph traversal
- MCP server optional for isolation
- Zero modifications to Graphiti source

---

### Agent 2: OneKE Integration Specialist

**Mission**: Integrate OneKE schema-guided knowledge extraction
**Priority**: P2 (High)
**Gaps Filled**: GAP-2 (Physics Domain Knowledge), GAP-10 (Knowledge Extraction)

**Tasks**:
1. Create `integrations/base/extraction_interface.py` with abstract extraction interface
2. Create `integrations/oneke/adapter.py` implementing the interface
3. Create `integrations/oneke/bridge.py` connecting to workflow_knowledge_extractor
4. Create `integrations/oneke/config.yaml` for model and schema configuration
5. Define physics/chemistry schemas in `integrations/oneke/schemas/`
6. Write `docs/integrations/ONEKE_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_oneke_integration.py`
8. Update `workflow_knowledge_extractor.py` to use OneKE adapter

**Requirements**:
- Support NER, RE, EE, Triple extraction
- Multi-agent extraction workflows
- Custom schema definitions for physics/chemistry
- Docker support (optional, can use Conda)
- Zero modifications to OneKE source

---

### Agent 3: Curie Integration Specialist

**Mission**: Integrate Curie automated scientific experimentation
**Priority**: P1.5 (High)
**Gaps Filled**: GAP-4 (Experimental Data), GAP-12 (Experiment Automation)

**Tasks**:
1. Create `integrations/base/experimentation_interface.py` with abstract experiment interface
2. Create `integrations/curie/adapter.py` implementing the interface
3. Create `integrations/curie/bridge.py` connecting to SOP Generator and validation systems
4. Create `integrations/curie/config.yaml` for experiment configuration
5. Define domain-specific experiment templates in `integrations/curie/templates/`
6. Write `docs/integrations/CURIE_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_curie_integration.py`
8. Update `sop_generator.py` to integrate Curie experiment protocols

**Requirements**:
- Hypothesis → experiment → result pipeline
- Integration with SOP Generator for protocols
- Statistical validation framework
- Reflection-based refinement
- Zero modifications to Curie source

---

### Agent 4: NeuroMANCER Integration Specialist

**Mission**: Integrate NeuroMANCER physics-informed optimization
**Priority**: P3 (Medium)
**Gaps Filled**: GAP-3 (Numerical Computation), GAP-1 (Continuous Math)

**Tasks**:
1. Create `integrations/base/optimization_interface.py` with abstract optimization interface
2. Create `integrations/neuromancer/adapter.py` implementing the interface
3. Create `integrations/neuromancer/bridge.py` connecting to LeanAide and solver systems
4. Create `integrations/neuromancer/config.yaml` for optimization configuration
5. Define physics-informed problem templates in `integrations/neuromancer/templates/`
6. Write `docs/integrations/NEUROMANCER_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_neuromancer_integration.py`
8. Update `leanaide_client.py` to use NeuroMANCER for numerical optimization

**Requirements**:
- Physics-informed system identification
- Constrained optimization solver
- PyTorch isolation (use separate environment)
- Hybrid solver: LeanAide (symbolic) + NeuroMANCER (numerical)
- Zero modifications to NeuroMANCER source

---

### Agent 5: pygraphistry Integration Specialist

**Mission**: Integrate pygraphistry interactive graph visualization and ML
**Priority**: P2 (High)
**Gaps Filled**: GAP-7 (Domain Tactics), GAP-10 (Pattern Mining), GAP-11 (Analytics)

**Tasks**:
1. Create `integrations/graphistry/adapter.py` for visualization adapter
2. Create `integrations/graphistry/bridge.py` connecting to knowledge visualization
3. Create `integrations/graphistry/config.yaml` for visualization configuration
4. Implement UMAP + DBSCAN clustering pipeline
5. Create GPU-accelerated graph analytics
6. Write `docs/integrations/PYGRAPHISTRY_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_pygraphistry_integration.py`
8. Update `knowledge_graph_visualizer.py` to use pygraphistry adapter

**Requirements**:
- Interactive web-based graph visualization
- GPU acceleration with cuML
- UMAP embeddings + DBSCAN clustering
- BubbleLab UI iframe embedding
- Zero modifications to pygraphistry source

---

### Agent 6: uqtestfuns Integration Specialist

**Mission**: Integrate uqtestfuns uncertainty quantification test functions
**Priority**: P3 (Medium)
**Gaps Filled**: GAP-15 (Uncertainty Quantification)

**Tasks**:
1. Create `integrations/base/uq_interface.py` with abstract UQ interface
2. Create `integrations/uqtestfuns/adapter.py` implementing the interface
3. Create `integrations/uqtestfuns/bridge.py` connecting to validation systems
4. Create `integrations/uqtestfuns/config.yaml` for UQ configuration
5. Implement UQ validation pipeline
6. Write `docs/integrations/UQTESTFUNS_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_uqtestfuns_integration.py`
8. Update verification systems to use UQ validation

**Requirements**:
- Test function library integration
- Probabilistic input specifications
- Lightweight dependency (NumPy, SciPy only)
- Validation pipeline for uncertainty propagation
- Zero modifications to uqtestfuns source

---

### Agent 7: global-chem Integration Specialist

**Mission**: Integrate global-chem chemical knowledge graph
**Priority**: P4 (Optional)
**Gaps Filled**: GAP-13 (Chemical/Biological Knowledge), GAP-2 (Domain Knowledge)

**Tasks**:
1. Create `integrations/global_chem/adapter.py` for chemical knowledge adapter
2. Create `integrations/global_chem/bridge.py` connecting to knowledge base
3. Create `integrations/global_chem/config.yaml` for chemical domain configuration
4. Implement SMILES/SMARTS parsing pipeline
5. Create chemical property prediction integration
6. Write `docs/integrations/GLOBAL_CHEM_INTEGRATION_GUIDE.md`
7. Create tests in `tests/integrations/test_global_chem_integration.py`
8. Update OneKE schemas to use global-chem knowledge

**Requirements**:
- Community-curated chemical lists
- SMILES/SMARTS support
- Domain-specific knowledge for chemistry/biology
- Integration with OneKE for entity recognition
- Zero modifications to global-chem source

---

## Integration Template (Same for All Agents)

### 1. Base Interface

```python
# integrations/base/[domain]_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class [Domain]Interface(ABC):
    """Abstract interface for [domain] functionality"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the [domain] system"""
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input using [domain] capabilities"""
        pass

    @abstractmethod
    async def validate(self) -> bool:
        """Validate the [domain] system is working"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the [domain] system"""
        pass
```

### 2. Adapter Implementation

```python
# integrations/[project]/adapter.py
from integrations.base.[domain]_interface import [Domain]Interface
import [project_library]

class [Project]Adapter([Domain]Interface):
    """Adapter for [Project] - wraps external functionality"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize [Project] client"""
        # Create client without modifying [Project] source
        self.client = [project_library].Client(**config)

    async def process(self, input_data: Any) -> Any:
        """Process using [Project]"""
        # Call [Project] API through adapter
        return await self.client.process(input_data)

    async def validate(self) -> bool:
        """Validate [Project] is working"""
        # Test connection/operation
        return await self.client.ping()

    async def shutdown(self) -> None:
        """Shutdown [Project] client"""
        if self.client:
            await self.client.close()
```

### 3. Bridge to OpenEvolve

```python
# integrations/[project]/bridge.py
from integrations.[project].adapter import [Project]Adapter
from openevolve.[existing_module] import [ExistingClass]

class [Project]Bridge:
    """Bridge between [Project] and OpenEvolve"""

    def __init__(self, config_path: str):
        self.adapter = [Project]Adapter(self._load_config(config_path))
        self.openevolve = [ExistingClass]()

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML"""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    async def integrate(self, data: Any) -> Any:
        """Integrate [Project] functionality into OpenEvolve workflow"""
        # 1. Process through [Project]
        result = await self.adapter.process(data)

        # 2. Transform to OpenEvolve format
        openevolve_format = self._to_openevolve_format(result)

        # 3. Pass to OpenEvolve system
        return await self.openevolve.process(openevolve_format)

    def _to_openevolve_format(self, data: Any) -> Any:
        """Transform [Project] output to OpenEvolve format"""
        # Conversion logic here
        pass
```

### 4. Configuration File

```yaml
# integrations/[project]/config.yaml
project:
  name: [Project Name]
  version: [version]
  enabled: true

connection:
  # Project-specific connection settings
  url: localhost
  port: 7687
  api_key: ${API_KEY}  # Environment variable

features:
  # Feature flags
  feature_1: true
  feature_2: false

integration:
  # OpenEvolve integration settings
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

performance:
  # Performance settings
  max_workers: 4
  timeout: 30
  batch_size: 100
```

### 5. Integration Guide Template

```markdown
# [PROJECT] Integration Guide

## Overview
[Brief description of project and purpose]

## Purpose
[Why this project is integrated into OpenEvolve]

## Technical Implementation
[Detailed technical implementation]

### Architecture
[Architecture diagram and description]

### Integration Points
[Where and how it connects to OpenEvolve]

### Configuration
[Configuration options and usage]

## Usage Examples
[Code examples showing usage]

## API Reference
[API documentation for adapter/bridge]

## Testing
[How to test the integration]

## Troubleshooting
[Common issues and solutions]

## Future Enhancements
[Planned improvements]
```

---

## Master Orchestrator Task

### Agent 8: Integration Orchestrator

**Mission**: Coordinate all integration agents and ensure consistency

**Tasks**:
1. Create `INTEGRATION_ARCHITECTURE.md` describing overall architecture
2. Create `integrations/__init__.py` with factory pattern for all adapters
3. Create `integrations/registry.py` for dynamic integration loading
4. Create `integrations/health_monitor.py` for checking integration status
5. Ensure all adapters follow same interface pattern
6. Create integration testing suite (`tests/integrations/test_all_integrations.py`)
7. Generate master integration status dashboard
8. Document cross-integration dependencies and workflows

---

## Success Criteria

### For Each Agent

1. ✅ Adapter implements base interface correctly
2. ✅ Zero modifications to external project source
3. ✅ Configuration file with all options documented
4. ✅ Integration guide complete (6+ sections)
5. ✅ Tests with >80% coverage
6. ✅ Graceful degradation when external project unavailable
7. ✅ Works with OpenEvolve existing systems

### Overall System

1. ✅ All 7 projects integrated via adapters
2. ✅ System works with any subset of integrations enabled
3. ✅ No dependency conflicts (use isolation where needed)
4. ✅ Performance impact <10% overhead
5. ✅ All tests passing
6. ✅ Documentation complete and consistent

---

## Execution Plan

### Phase 1: Foundation (Agent 8 - Orchestrator)
- Day 1: Create base interfaces
- Day 2: Create integration registry and health monitor
- Day 3: Create testing framework

### Phase 2: Parallel Integration (Agents 1-7)
- Week 1: All agents work in parallel on their projects
- Week 2: Continue implementation, daily sync meetings
- Week 3: Testing, documentation, refinement

### Phase 3: Integration & Validation (Agent 8)
- Week 4: End-to-end testing of all integrations
- Week 4: Performance optimization
- Week 4: Documentation review and finalization

---

## Communication Protocol

### Daily Standup Format

Each agent reports:
1. **Progress**: What was accomplished yesterday
2. **Plan**: What will be done today
3. **Blockers**: Any obstacles preventing progress
4. **Needs**: Help required from other agents

### Integration Points Handoff

When Agent A needs functionality from Agent B:
1. Create interface contract in `base/`
2. Agent A implements to interface (mock initially)
3. Agent B implements interface
4. Swap mock for real implementation
5. Integration testing

---

## Getting Started

Each agent should:
1. Read this task specification completely
2. Read their assigned project's documentation
3. Create the integration directory structure
4. Implement adapter following base interface
5. Create configuration file
6. Write integration guide
7. Create tests
8. Submit pull request when complete

**Agents should work in parallel and coordinate through daily standups.**

---

**END OF TASK SPECIFICATION**

