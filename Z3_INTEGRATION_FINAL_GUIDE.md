# Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration
## Complete Implementation Guide v2.0

**A production-ready, enterprise-grade integration suite for constraint solving, theorem proving, and formal verification.**

---

## 🎯 Overview

This integration provides a comprehensive framework connecting:
- **Microsoft Z3** - SMT solver for constraints and optimization
- **LeanAIDE** - AI-powered formal verification in Lean 4
- **OpenEvolve** - Evolutionary workflow engine
- **BubbleLabs** - Visual workflow and UI platform

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔧 **Constraint Solving** | SAT/SMT solving with multiple backends |
| 📐 **Theorem Proving** | Formal proof generation and verification |
| ⚡ **Optimization** | Linear, non-linear, and multi-objective |
| 🔄 **Translation** | SMT-LIB ↔ Lean 4 bidirectional conversion |
| 🤖 **AI Agents** | CrewAI multi-agent workflows |
| 📊 **Visualization** | Real-time graphs, proof trees, landscapes |
| 💾 **Persistence** | Database storage with SQLite/PostgreSQL |
| 🔌 **API** | RESTful API with WebSocket support |
| 📈 **Monitoring** | Performance tracking and alerting |
| 🧠 **Knowledge** | Pattern learning and strategy extraction |

---

## 📦 Installation

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd z3-integration

# Install dependencies
pip install -r requirements.txt

# Run configuration wizard
python z3_cli.py config init

# Start the server
python z3_cli.py server
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f z3-integration

# Scale API servers
docker-compose up -d --scale z3-integration=3
```

---

## 🚀 Quick Start Guide

### 1. Solve a Constraint Problem

```python
import asyncio
from z3_leanaide_openevolve_integration import solve_with_z3_leanaide

async def main():
    result = await solve_with_z3_leanaide("""
        Find x and y where:
        - x > 0 and x < 10
        - y = x + 5
    """)
    
    print(f"Category: {result['classification']['category']}")
    print(f"Solution: {result['solution']['content']}")
    print(f"Verified: {result['solution']['verification_status']}")

asyncio.run(main())
```

### 2. Use the CLI

```bash
# Solve from command line
z3-cli solve "x > 0 and x < 10" --variables '[{"name":"x","type":"INTEGER"}]'

# Run optimization
z3-cli optimize "x + y" --variables '[...]' --constraints '[...]' --direction minimize

# Prove theorem
z3-cli prove theorem.smt2 --extract-proof

# Monitor performance
z3-cli monitor --watch
```

### 3. Call the REST API

```bash
# Health check
curl http://localhost:8765/health

# Solve constraints
curl -X POST http://localhost:8765/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)",
    "timeout": 30
  }'

# Get metrics
curl http://localhost:8765/metrics
```

---

## 📚 Module Reference

### Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| `z3prover_integration.py` | 983 | Base Z3 interface |
| `z3prover_advanced.py` | 1,199 | Advanced features (optimization, arrays, BV) |
| `z3_leanaide_bridge.py` | 1,005 | Z3 ↔ Lean translation |
| `z3_leanaide_openevolve_integration.py` | 1,048 | Workflow integration |
| `z3_leanaide_bubblelabs_ui.py` | 896 | Basic UI components |
| `z3_bubblelabs_advanced_ui.py` | 708 | Advanced visualizations |

### Integration Modules

| File | Lines | Purpose |
|------|-------|---------|
| `z3_mcp_tools.py` | 847 | MCP protocol tools |
| `z3_crewai_bridge.py` | 752 | Multi-agent workflows |
| `z3_result_cache.py` | 684 | Intelligent caching |
| `z3_performance_monitor.py` | 730 | Performance tracking |
| `z3_knowledge_extraction.py` | 662 | Knowledge management |

### Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `z3_config_manager.py` | 667 | Configuration management |
| `z3_database_models.py` | 579 | Database ORM models |
| `z3_api_server.py` | 614 | REST API server |
| `z3_cli.py` | 513 | Command line interface |
| `z3_config.yaml` | 270 | Configuration template |

---

## 🔧 Configuration

### Configuration File (`z3_config.yaml`)

```yaml
z3:
  enabled: true
  timeout: 60.0
  memory_limit_mb: 8192
  num_threads: 4
  proof_generation: true

leanaide:
  enabled: true
  host: "localhost"
  port: 7654
  timeout: 300.0

cache:
  enabled: true
  max_size: 10000
  default_ttl: 7200
  policy: "lru"
  persistent_storage: true

server:
  enabled: true
  host: "0.0.0.0"
  port: 8765
  
  security:
    api_key_required: false
    rate_limiting:
      enabled: true
      requests_per_minute: 60
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `Z3_CONFIG_PATH` | Config file path | `./z3_config.yaml` |
| `DB_PASSWORD` | Database password | - |
| `REDIS_URL` | Redis connection URL | - |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 🎨 Usage Patterns

### Pattern 1: Constraint Solving

```python
from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType, get_z3_solver_engine

solver = get_z3_solver_engine()

variables = [
    Z3Variable("x", Z3ConstraintType.INTEGER),
    Z3Variable("y", Z3ConstraintType.INTEGER)
]

constraints = [
    Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER),
    Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER),
    Z3Constraint("(= y (+ x 5))", Z3ConstraintType.INTEGER)
]

result = solver.solve_constraints(variables, constraints)

if result.is_sat():
    print(f"Solution: {result.model.assignments}")
```

### Pattern 2: Multi-Objective Optimization

```python
from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective

solver = Z3AdvancedSolver()

result = solver.optimize(
    variables=variables,
    constraints=constraints,
    objectives=[
        ("x", OptimizationObjective.MAXIMIZE),
        ("y", OptimizationObjective.MAXIMIZE)
    ],
    multi_objective_strategy="pareto"
)

print(f"Pareto front: {len(result.pareto_front)} solutions")
```

### Pattern 3: Cross-Verification

```python
from z3_leanaide_bridge import get_z3_leanaide_bridge_sync, VerificationStrategy

bridge = get_z3_leanaide_bridge_sync()

result = await bridge.verify_with_both(
    problem=theorem,
    strategy=VerificationStrategy.CONSENSUS
)

print(f"Agreement: {result.agreement}")
print(f"Confidence: {result.confidence_score}")
```

### Pattern 4: Agent Workflow

```python
from z3_crewai_bridge import get_z3_agent_coordinator, AgentRole

coordinator = get_z3_agent_coordinator()
coordinator.create_solver_agent("solver_1")
coordinator.create_prover_agent("prover_1")

session = await coordinator.execute_collaborative(
    session_id="session_001",
    problem=theorem,
    strategy="parallel"
)

print(f"Consensus: {session.consensus_reached}")
```

### Pattern 5: Monitored Execution

```python
from z3_performance_monitor import monitored

@monitored("constraint_solving")
def solve_with_monitoring(params):
    # ... solving logic
    return result

# Check metrics
from z3_performance_monitor import get_z3_performance_monitor
monitor = get_z3_performance_monitor()
bottlenecks = monitor.get_bottlenecks(5)
```

---

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/config` | Configuration |

### Solving Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/solve` | Constraint solving |
| POST | `/optimize` | Optimization |
| POST | `/prove` | Theorem proving |
| POST | `/solve-complete` | Full workflow |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Performance metrics |
| GET | `/knowledge/patterns` | Proof patterns |
| GET | `/knowledge/insights` | Mathematical insights |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws` | General WebSocket |
| `/ws/progress/{id}` | Operation progress |

---

## 📊 Monitoring & Observability

### Performance Dashboard

Access Grafana at `http://localhost:3000` (admin/admin)

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| solve_time | Average solve time | >30s warning |
| error_rate | Percentage of errors | >10% warning |
| queue_depth | Pending operations | >10 warning |
| memory_usage | Memory consumption | >1GB warning |

### Custom Alerts

```python
from z3_performance_monitor import get_z3_performance_monitor

monitor = get_z3_performance_monitor()
monitor.set_threshold("solve_time", 60.0, Severity.ERROR)

monitor.add_alert_handler(lambda alert: send_slack(alert))
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest test_z3_leanaide_integration.py -v
```

### Run Specific Test Category

```bash
# Unit tests
pytest test_z3_leanaide_integration.py::TestZ3Integration -v

# Integration tests
pytest test_z3_leanaide_integration.py -m integration -v

# Performance tests
pytest test_z3_leanaide_integration.py -m performance -v
```

### Load Testing

```bash
# Using locust
locust -f load_test.py --host=http://localhost:8765
```

---

## 🚀 Deployment

### Production Checklist

- [ ] Configure production database (PostgreSQL)
- [ ] Set up Redis for distributed caching
- [ ] Enable SSL/TLS
- [ ] Configure API keys and rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Set up backup strategy

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: z3-integration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: z3-integration
  template:
    metadata:
      labels:
        app: z3-integration
    spec:
      containers:
      - name: z3-integration
        image: openevolve/z3-integration:2.0.0
        ports:
        - containerPort: 8765
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
```

---

## 📈 Performance Tuning

### Optimization Tips

1. **Enable Caching**
   ```python
   from z3_result_cache import get_z3_result_cache
   cache = get_z3_result_cache(CacheConfig(max_size=10000))
   ```

2. **Use Portfolio Solving**
   ```python
   result = solver.solve_portfolio(smtlib, parallel=True)
   ```

3. **Parallel Processing**
   ```python
   config = Z3Config(num_threads=8)
   ```

4. **Incremental Solving**
   ```python
   state_id = solver.create_incremental_state(vars, constraints)
   ```

### Benchmarks

| Operation | Small | Medium | Large |
|-----------|-------|--------|-------|
| Constraint Solving | <10ms | <100ms | <1s |
| Theorem Proving | <100ms | <1s | <10s |
| Optimization | <100ms | <1s | <10s |
| Translation | <50ms | <200ms | <1s |

---

## 🔐 Security

### Best Practices

1. **API Key Authentication**
   ```yaml
   server:
     security:
       api_key_required: true
       api_keys:
         - "sk-xxx"
   ```

2. **Rate Limiting**
   ```yaml
   rate_limiting:
     enabled: true
     requests_per_minute: 60
     burst_size: 10
   ```

3. **CORS Configuration**
   ```yaml
   cors:
     enabled: true
     allowed_origins:
       - "https://yourdomain.com"
   ```

---

## 📝 Troubleshooting

### Common Issues

**Issue: Z3 not found**
```bash
# Solution: Install Z3
sudo apt-get install z3
pip install z3-solver
```

**Issue: Database connection failed**
```bash
# Solution: Check database configuration
python z3_cli.py config validate
```

**Issue: Out of memory**
```yaml
# Solution: Reduce memory limits
z3:
  memory_limit_mb: 4096
  timeout: 30
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python z3_cli.py --verbose solve ...
```

---

## 🤝 Contributing

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black z3_*.py

# Type checking
mypy z3_*.py
```

### Adding New Features

1. Create module: `z3_new_feature.py`
2. Add tests: `test_z3_new_feature.py`
3. Update CLI: Add command to `z3_cli.py`
4. Update API: Add endpoint to `z3_api_server.py`
5. Update docs: Add to this guide

---

## 📄 License

This integration is part of the OpenEvolve project and follows the same license terms.

---

## 📞 Support

- **Documentation**: See `/docs` endpoint
- **Issues**: Report on GitHub Issues
- **Discussions**: Join GitHub Discussions
- **Email**: support@openevolve.ai

---

## 🎉 Acknowledgments

- Microsoft Research for Z3
- Lean Prover team
- OpenEvolve contributors
- BubbleLabs team

---

**Total Implementation: ~15,000 lines of code**

*Built with ❤️ by OpenEvolve*
