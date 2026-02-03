# OpenEvolve: AGENTS.md - AI Coding Agent Guide

> **Purpose**: This document provides essential context for AI coding agents working on the OpenEvolve project. It covers architecture, conventions, testing, and development workflows.

---

## 1. Project Overview

**OpenEvolve** is a unified evolutionary optimization platform that combines two cutting-edge systems:

- **OpenEvolve Core** - Quality Diversity (MAP-Elites), Multi-Objective (NSGA-II), and Adversarial Co-evolution
- **LoongFlow PES** - Plan-Execute-Summarize paradigm with reasoning-guided search

### Key Capabilities
- Automatic strategy selection (PES, QD, MO, Adversarial, or Standard mode)
- 3-Round Gauntlet System for quality evaluation (LoongFlow AI Eval → Red Team Attack → Gold Team Verification)
- Knowledge-guided evolution with temporal knowledge graph storage
- Domain-specific optimizers for Finance, Trading, Science, Engineering, Pharma, and Web Design
- 60% fewer evaluations through intelligent search strategies

---

## 2. Technology Stack

### Core Technologies
| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | >=3.10 |
| Web UI | Streamlit | Latest |
| API Framework | FastAPI | >=0.104.0 |
| Database | SQLite/PostgreSQL | - |
| Cache | Redis | 7.x |
| Formal Verification | Z3 Solver | >=4.12.0 |
| Knowledge Graph | Neo4j | Latest |
| Vector Store | Qdrant | Latest |

### LLM Integrations
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Local models via various adapters

### Key Python Dependencies
```
z3-solver>=4.12.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
streamlit
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0.1
pydantic>=2.5.0
```

---

## 3. Project Structure

```
c:\Users\mmeadow\Documents\OpenEvolve\Frontend/
├── Core Application Files
│   ├── main.py                    # Streamlit UI entry point
│   ├── app.py                     # Demo application
│   ├── api_server.py              # FastAPI server
│   ├── evolution.py               # Core evolution engine
│   ├── decomposition_engine.py    # Problem decomposition
│   └── workflow_engine.py         # Workflow orchestration
│
├── Team System (Red/Blue/Gold)
│   ├── red_team.py                # Adversarial testing
│   ├── blue_team.py               # Fix generation
│   ├── evaluator_team.py          # Consensus evaluation
│   └── team_manager.py            # Team coordination
│
├── Knowledge & Quality Systems
│   ├── knowledge_base.py          # Knowledge storage
│   ├── quality_assessment.py      # Quality evaluation
│   ├── quality_gate_engine.py     # Quality gates
│   └── gauntlet_manager.py        # 3-round gauntlet
│
├── Integration Systems
│   ├── openevolve_*.py            # OpenEvolve integrations
│   ├── leanaide_*.py              # LeanAide integration (theorem proving)
│   ├── bubblelabs_*.py            # BubbleLabs integration
│   ├── roma_*.py                  # ROMA integration
│   ├── z3_*.py                    # Z3 prover integration
│   └── crewai_*.py                # CrewAI integration
│
├── Configuration & Infrastructure
│   ├── config.py                  # Configuration management
│   ├── config_loader.py           # Config loading utilities
│   ├── parameter_manager.py       # 272+ parameter management
│   ├── deploy.py                  # Deployment automation
│   └── docker-compose.yml         # Docker orchestration
│
├── Tests
│   ├── tests/                     # Organized test suites
│   ├── test_*.py                  # Individual test files
│   ├── conftest.py                # Pytest configuration
│   └── pytest.ini                 # Pytest settings
│
├── Documentation
│   └── docs/knowledge_engine/     # Extensive documentation (100+ MD files)
│
└── Sub-projects/Directories
    ├── BubbleLab/                 # Enterprise integration system
    ├── LeanAide/                  # Lean 4 theorem prover integration
    ├── crewAI/                    # CrewAI agent system
    ├── z3prover/                  # Z3 formal verification
    ├── knowledge_engine/          # Knowledge graph system
    ├── adaptive_mdap/             # Adaptive MDAP system
    └── openevolve/                # OpenEvolve package
```

---

## 4. Build and Development Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_with_testing.txt  # Include test deps

# Or install as package
pip install -e .
pip install -e ".[dev]"  # With development dependencies
```

### Running the Application
```bash
# Start Streamlit UI
streamlit run main.py

# Start API server
python api_server.py
# Or
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Run demo
python app.py
```

### Docker Commands
```bash
# Build and start all services
docker-compose up -d

# Build specific image
docker build -t openevolve:latest .

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Makefile Commands
```bash
make help              # Show available commands
make install           # Install dependencies
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests
make docker-build      # Build Docker image
make docker-up         # Start Docker containers
make lint              # Run linting
make format            # Format code with Black
make health-check      # Run health checks
```

---

## 5. Testing Strategy

### Test Organization
Tests are organized in multiple ways:

1. **`tests/` directory** - Organized test suites by category:
   - `tests/integration/` - Integration tests
   - `tests/unit/` - Unit tests
   - `tests/benchmarks/` - Performance benchmarks
   - `tests/gauntlets/` - Gauntlet system tests
   - `tests/knowledge_engine/` - Knowledge system tests

2. **Root-level `test_*.py` files** - Individual test modules for specific components

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage (minimum 87% required)
pytest --cov=. --cov-report=term-missing --cov-fail-under=87

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m "not slow"    # Exclude slow tests

# Run specific test files
pytest test_sovereign_*.py -v
pytest test_decomposition_*.py -v

# Parallel execution
pytest -x  # Stop on first failure
pytest --tb=short  # Short traceback format
```

### Test Markers
| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast, isolated unit tests |
| `@pytest.mark.integration` | Tests requiring external services |
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.postgres` | Tests requiring PostgreSQL |
| `@pytest.mark.redis` | Tests requiring Redis |

### Test Fixtures (conftest.py)
- `temp_db_path` - Temporary database file
- `temp_dir` - Temporary directory
- `result` - TestResult tracking instance
- `cleanup_test_resources` - Automatic resource cleanup

---

## 6. Code Style Guidelines

### Formatting (Black)
```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311']
```

### Import Ordering (isort)
```toml
[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["openevolve", "crewai", "roma", "bubblelab"]
```

### Linting Rules
- **Flake8**: Max line length 100, compatible with Black
- **Pylint**: Disabled strict rules for WIP code (C0111, C0103, R0903, R0913)
- **Bandit**: Security scanning (skips B101 for assert statements in tests)

### Type Checking (mypy)
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Set to true in production
check_untyped_defs = true
```

### Code Quality Commands
```bash
# Format code
black .
isort .

# Lint
flake8 openevolve/ --max-line-length=100
pylint openevolve/

# Type check
mypy .

# Security scan
bandit -r . -f json -o bandit-report.json

# Run all quality checks
black --check .
flake8 .
mypy .
bandit -r .
```

---

## 7. Configuration System

### Configuration Hierarchy
1. **Environment Variables** - Highest priority
2. **`.env` file** - Local environment settings
3. **`config.yaml`** - YAML configuration
4. **Default values** - Code-level defaults

### Key Configuration Files
| File | Purpose |
|------|---------|
| `.env` | Local environment variables (gitignored) |
| `.env.example` | Template for required env vars |
| `config.yaml` | Default YAML configuration |
| `z3_config.yaml` | Z3 prover configuration |
| `decomposition_config_lean.yaml` | Decomposition settings |

### Required Environment Variables
```bash
# API Keys (Required for LLM operations)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false

# Security (CRITICAL for production)
SECRET_KEY=<generate-with-secrets.token_hex(32)>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Parameter Management
The system uses a comprehensive parameter system with 272+ parameters managed through:
- `parameter_manager.py` - Central parameter management
- `parameter_definitions.py` - Parameter schemas and validation
- `parameter_sync_manager.py` - Synchronization across components

---

## 8. Security Considerations

### Critical Security Rules
1. **NEVER commit `.env` files** - Contains API keys and secrets
2. **Use strong SECRET_KEY in production** - Generate with `python -c 'import secrets; print(secrets.token_hex(32))'`
3. **Validate all inputs** - Use `input_validation.py` for user input
4. **Encrypt stored API keys** - Use `KEY_ENCRYPTION_KEY` for storage

### Security Tools
```bash
# Run security scan
bandit -r . -f json -o security_report.json

# Check for secrets
git-secrets --scan

# Dependency vulnerability check
safety check
```

### Security Features
- JWT token authentication
- API key management with role-based access
- Input sanitization and validation
- Circuit breaker pattern for external calls
- Rate limiting
- SQL injection prevention (SQLAlchemy parameterized queries)

---

## 9. Development Conventions

### File Naming
- **Core modules**: `snake_case.py`
- **Test files**: `test_*.py`
- **Integration files**: `*_integration.py`
- **Configuration**: `*.yaml`, `*.json`

### Class Naming
- **Data models**: PascalCase (e.g., `EvolutionResult`)
- **Strategies**: Suffix with type (e.g., `SemanticDecomposition`, `QualityGateEngine`)
- **Managers**: Suffix with Manager (e.g., `TeamManager`, `KnowledgeManager`)

### Import Patterns
```python
# Standard library first
import logging
import json
from typing import List, Dict, Optional

# Third-party packages
import streamlit as st
import numpy as np
from fastapi import FastAPI

# First-party modules
from parameter_manager import ParameterManager
from quality_assessment import QualityAssessmentEngine
```

### Error Handling Pattern
```python
from error_handler import with_error_handling, ErrorSeverity

@with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda: None)
def critical_function():
    """Function with automatic error handling."""
    pass
```

### Logging Pattern
```python
import logging
logger = logging.getLogger(__name__)

# Good logging
logger.info("Processing problem: %s", problem_id)
logger.error("Failed to decompose: %s", error, exc_info=True)
```

---

## 10. Deployment Process

### Deployment Environments
| Environment | Config File | Purpose |
|-------------|-------------|---------|
| Development | `deploy_config_development.json` | Local development |
| Staging | `deploy_config_staging.json` | Pre-production testing |
| Production | `deploy_config_production.json` | Live deployment |

### Deployment Commands
```bash
# Deploy to specific environment
python deploy.py --environment development
python deploy.py --environment staging
python deploy.py --environment production

# Or use Makefile
make deploy-dev
make deploy-staging
make deploy-prod
```

### Docker Production Deployment
```bash
# Build production image
docker build -t openevolve:latest . --target production

# Run with docker-compose
docker-compose -f docker-compose.yml up -d
```

### Pre-deployment Checklist
- [ ] All tests passing (`pytest`)
- [ ] Code coverage >= 87%
- [ ] Security scan clean (`bandit`)
- [ ] Linting passed (`flake8`, `black --check`)
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Health checks passing

---

## 11. Key Architectural Patterns

### Strategy Pattern
Used extensively for decomposition and evolution strategies:
```python
from abc import ABC, abstractmethod

class DecompositionStrategyBase(ABC):
    @abstractmethod
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        raise NotImplementedError
```

### Circuit Breaker Pattern
For external service calls:
```python
from sovereign_reliability import with_circuit_breaker

@with_circuit_breaker(failure_threshold=5, timeout=60.0)
def external_api_call():
    pass
```

### Knowledge Extraction Pattern
```python
# After each evolution run
knowledge_engine.extract_patterns(result)
knowledge_engine.store_in_graph(solution)
```

---

## 12. Common Development Tasks

### Adding a New Decomposition Strategy
1. Create class inheriting from `DecompositionStrategyBase`
2. Implement `decompose()` and `get_strategy_name()` methods
3. Register in `decomposition_engine.py`
4. Add tests in `tests/test_decomposition_*.py`

### Adding a New Quality Gate
1. Create quality gate class in `quality_gate_engine.py`
2. Implement validation logic
3. Register in quality gate registry
4. Add corresponding tests

### Adding a New Integration
1. Create `*_integration.py` or `*_bridge.py` file
2. Implement adapter pattern for external system
3. Add configuration in `config.yaml`
4. Create tests following `test_*_integration.py` naming

### Adding API Endpoints
1. Add endpoint in `api_server.py`
2. Define request/response models with Pydantic
3. Add authentication if needed
4. Document in `docs/knowledge_engine/API_REFERENCE.md`

---

## 13. Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Database Lock Errors**
```bash
# Remove lock files
rm *.db-journal *.db-shm *.db-wal
```

**Test Timeouts**
```bash
# Increase timeout in pytest.ini
# Or run specific test without timeout
pytest test_specific.py -p no:timeout
```

**Memory Issues**
```bash
# Run with memory management enabled
MEMORY_MANAGEMENT_ENABLED=true python main.py
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
DEBUG=true python main.py
```

---

## 14. Documentation References

### Key Documentation Files
| Document | Purpose |
|----------|---------|
| `docs/knowledge_engine/UNIFIED_EVOLUTION_ENGINE_GUIDE.md` | Complete system guide (2000+ lines) |
| `docs/knowledge_engine/API_REFERENCE.md` | API documentation |
| `docs/knowledge_engine/MIGRATION_GUIDE.md` | Migration instructions |
| `docs/knowledge_engine/TROUBLESHOOTING.md` | Common issues |
| `docs/knowledge_engine/PERFORMANCE_TUNING.md` | Optimization guide |

### Domain Guides
Located in `docs/knowledge_engine/domains/`:
- `finance_guide.md`
- `trading_guide.md`
- `science_guide.md`
- `engineering_guide.md`
- `pharma_guide.md`
- `web_design_guide.md`

---

## 15. Contact & Support

- **Documentation**: `docs/knowledge_engine/`
- **Issues**: Use GitHub Issues (if configured)
- **Test Reports**: Check `test_reports.html` after running tests

---

**Last Updated**: February 2, 2026

**Project Status**: Production Ready

**License**: Apache-2.0 (pyproject.toml) / MIT (README.md)
