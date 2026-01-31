# Mathematical Knowledge Integration

Complete integration between Z3 SMT solver, LeanAIDE formal verification, OpenEvolve workflows, and BubbleLabs UI.

## Overview

This integration provides:

- **Z3 SMT Solver**: Constraint satisfaction and optimization
- **LeanAIDE**: Theorem proving and formal verification
- **Knowledge Extraction**: ML-powered pattern learning from solutions
- **Unified Bridge**: Intelligent solver selection and consensus
- **OpenEvolve Integration**: Workflow orchestration
- **BubbleLabs UI**: Visual problem solving interface
- **MCP Tools**: AI assistant integration (Claude, Cursor)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BubbleLabs UI                            │
│         (Visual problem solving interface)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  OpenEvolve Workflow                        │
│       (Orchestration, collaboration, version control)       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Unified Mathematical Bridge                    │
│    (Solver selection, consensus, cross-system learning)     │
└──────────┬─────────────────────┬────────────────────────────┘
           │                     │
┌──────────▼─────────┐  ┌────────▼──────────┐
│   Z3 Connector     │  │ LeanAIDE Connector │
│  (SMT solving)     │  │ (Theorem proving)  │
└──────────┬─────────┘  └────────┬───────────┘
           │                     │
┌──────────▼─────────────────────▼───────────┐
│         Knowledge Manager                   │
│  (Pattern extraction, strategy learning)    │
└──────────┬──────────────────────────────────┘
           │
┌──────────▼──────────┐  ┌──────────────────┐
│      Database       │  │  Redis Cache     │
│ (SQLAlchemy models) │  │  (Performance)   │
└─────────────────────┘  └──────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install z3-solver sqlalchemy redis fastapi uvicorn

# For LeanAIDE
pip install leanaide  # or build from source
```

### 2. Configuration

Create `config.yaml`:

```yaml
database:
  url: sqlite:///math_knowledge.db

z3:
  timeout_ms: 30000
  memory_limit_mb: 4096

leanaide:
  enabled: true
  host: localhost
  port: 7654

api:
  enabled: true
  host: 0.0.0.0
  port: 8765
```

Or use environment variables:

```bash
export MATH_KNOWLEDGE_DB_URL=sqlite:///math_knowledge.db
export MATH_KNOWLEDGE_Z3_TIMEOUT_MS=30000
export MATH_KNOWLEDGE_LEANAIDE_HOST=localhost
export MATH_KNOWLEDGE_API_PORT=8765
```

### 3. Run API Server

```bash
python z3_api.py
```

Or with Docker:

```bash
docker-compose -f docker-compose.math-knowledge.yml up -d
```

### 4. Test the Integration

```bash
curl http://localhost:8765/health

curl -X POST http://localhost:8765/solve/z3 \
  -H "Content-Type: application/json" \
  -d '{
    "content": "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
    "format": "smtlib"
  }'
```

## Usage Examples

### Basic Z3 Solving

```python
from z3_solver_connector import get_z3_connector, Z3SolverConfig

async def example():
    connector = get_z3_connector()
    
    smtlib = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (= (+ x y) 10))
    (assert (> x 0))
    (assert (> y 0))
    (check-sat)
    (get-model)
    """
    
    result = await connector.solve_smtlib(
        smtlib,
        Z3SolverConfig(timeout_ms=10000)
    )
    
    print(f"Status: {result.status}")
    print(f"Model: {result.model}")
```

### Lean Theorem Proving

```python
from leanaide_production_connector import get_leanaide_connector

async def example():
    connector = await get_leanaide_connector()
    
    theorem = """
    theorem example : ∀ n : ℕ, n + 0 = n := by
      intro n
      simp
    """
    
    result = await connector.prove_theorem(theorem)
    print(f"Success: {result['success']}")
    print(f"Proof: {result['proof']}")
```

### Unified Solving

```python
from unified_math_bridge_complete import get_unified_bridge_complete, SolverSystem

async def example():
    bridge = await get_unified_bridge_complete()
    
    result = await bridge.solve(
        problem="x + y = 10, x > 0, y > 0",
        preferred_solver=SolverSystem.HYBRID,
        timeout=60
    )
    
    print(f"Result: {result['result_status']}")
    print(f"Verified: {result['verified']}")
```

### Knowledge Extraction

```python
from z3_knowledge_complete import get_z3_knowledge_manager

async def example():
    manager = await get_z3_knowledge_manager()
    
    # Learn from solution
    await manager.learn_from_solution(
        problem_statement="Linear system",
        constraints=["x + y = 10", "x - y = 2"],
        result="success",
        proof="substitution"
    )
    
    # Find similar solutions
    similar = await manager.find_similar_solutions(
        problem_statement="System of equations",
        constraints=["2x + 3y = 15"],
        top_k=5
    )
    
    # Get strategy recommendation
    strategy = await manager.get_recommended_strategy(
        problem_statement="New problem",
        constraints=["x + y = 5"]
    )
```

### MCP Tools

```python
from math_mcp_tools import get_math_mcp_tools

async def example():
    tools = await get_math_mcp_tools()
    
    # Solve with Z3
    result = await tools.execute_tool("z3_solve", {
        "problem": "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
        "format": "smtlib"
    })
    
    # Search patterns
    patterns = await tools.execute_tool("math_pattern_search", {
        "query": "linear system",
        "top_k": 5
    })
```

## API Endpoints

### Health & Info
- `GET /health` - Health check
- `GET /` - API info
- `GET /stats` - System statistics

### Z3 Operations
- `POST /solve/z3` - Solve SMT-LIB
- `POST /solve/natural` - Natural language solving
- `GET /z3/strategies` - Get strategies
- `GET /z3/features/{record_id}` - Get features

### Lean Operations
- `POST /prove/lean` - Prove theorem
- `POST /prove/auto` - Auto-prove
- `POST /translate/lean` - Translate to Lean
- `GET /proofs/{proof_id}` - Get proof

### Unified Operations
- `POST /solve/unified` - Unified solving
- `POST /solve/consensus` - Consensus solving
- `POST /solve/hybrid` - Hybrid solving

### Knowledge Operations
- `POST /knowledge/learn` - Learn from solution
- `POST /knowledge/search` - Search patterns
- `POST /knowledge/similar` - Find similar
- `GET /knowledge/strategy` - Get strategy
- `GET /knowledge/stats` - Knowledge stats

### Analytics
- `GET /analytics/summary` - Summary report
- `POST /analytics/performance` - Performance report
- `POST /analytics/quality` - Quality report

## File Structure

```
knowledge_engine/integrations/
├── README.md                              # This file
├── IMPLEMENTATION_SUMMARY.md              # Implementation summary
├── GAP_ANALYSIS_AND_PRODUCTION_CONNECTORS.md  # Gap analysis
│
├── Core Implementation (64KB)
├── z3_knowledge_complete.py              # Z3 knowledge management
├── leanaide_integration_complete.py      # LeanAIDE integration
├── unified_math_bridge_complete.py       # Unified bridge
│
├── Production Connectors (31KB)
├── z3_solver_connector.py                # Real Z3 connector
├── leanaide_production_connector.py      # Real LeanAIDE connector
├── leanaide_real_connector.py            # Additional Lean connector
│
├── API & Services (37KB)
├── z3_api.py                             # FastAPI server
├── z3_server_complete.py                 # Complete server
├── z3_knowledge_api.py                   # Knowledge API
│
├── Database & Persistence (13KB)
├── math_knowledge_models.py              # SQLAlchemy models
├── math_knowledge_persistence.py         # Persistence layer
│
├── MCP & Configuration (36KB)
├── math_mcp_tools.py                     # MCP tools
├── math_knowledge_config.py              # Configuration
│
├── Deployment (7KB)
├── docker-compose.math-knowledge.yml     # Docker compose
├── Dockerfile.math-knowledge             # Docker image
│
├── Testing & Demo (27KB)
├── comprehensive_test.py                 # Full test suite
├── complete_integration_example.py       # Demo
│
└── Documentation (21KB)
    ├── IMPLEMENTATION_SUMMARY.md
    └── GAP_ANALYSIS_AND_PRODUCTION_CONNECTORS.md

Total: ~20 files, ~325KB, ~8,000 lines
```

## Testing

```bash
# Run comprehensive tests
python comprehensive_test.py

# Run specific component tests
python -m pytest test_z3_knowledge.py
python -m pytest test_leanaide.py
python -m pytest test_unified_bridge.py
```

## Docker Deployment

```bash
# Build and run
docker-compose -f docker-compose.math-knowledge.yml up -d

# View logs
docker-compose logs -f math-knowledge-api

# Scale workers
docker-compose up -d --scale math-knowledge-api=3
```

## Monitoring

- **Metrics**: http://localhost:9090/metrics
- **Grafana**: http://localhost:3000
- **Health**: http://localhost:8765/health
- **Stats**: http://localhost:8765/stats

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATH_KNOWLEDGE_DB_URL` | `sqlite:///math_knowledge.db` | Database URL |
| `MATH_KNOWLEDGE_REDIS_ENABLED` | `false` | Enable Redis cache |
| `MATH_KNOWLEDGE_Z3_TIMEOUT_MS` | `30000` | Z3 timeout |
| `MATH_KNOWLEDGE_LEANAIDE_HOST` | `localhost` | LeanAIDE host |
| `MATH_KNOWLEDGE_API_PORT` | `8765` | API port |
| `MATH_KNOWLEDGE_LOG_LEVEL` | `INFO` | Log level |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - See LICENSE file

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@openevolve.ai

## Acknowledgments

- Z3 Theorem Prover (Microsoft Research)
- Lean 4 (Lean FRO)
- LeanAIDE Project
- OpenEvolve Community
