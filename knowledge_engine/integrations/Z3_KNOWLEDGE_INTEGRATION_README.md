# Z3 Knowledge Integration for OpenEvolve Knowledge Engine

This integration merges Z3 knowledge extraction capabilities into the BubbleLabs/OpenEvolve knowledge engine, enabling automatic learning from Z3 solver operations.

## Overview

The Z3 Knowledge Integration provides:

- **Automatic Knowledge Extraction**: Extracts proof patterns, constraint patterns, strategies, and mathematical insights from Z3 solver results
- **Unified Storage**: Stores Z3 knowledge in the existing knowledge engine infrastructure
- **Pattern Matching**: Recommends strategies based on problem similarity
- **Knowledge Graph**: Connects Z3 patterns with broader knowledge base
- **REST API**: HTTP endpoints for knowledge management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Z3 Solver Operations                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              Z3 Knowledge Extraction Hook                        │
│         (z3_auto_extraction.py)                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│           Z3 Knowledge Integration Layer                         │
│         (z3_knowledge_integration.py)                            │
│  • Extract from solver results                                   │
│  • Transform to knowledge artifacts                              │
│  • Store in knowledge engine                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│   Database   │  │   Vector Store  │  │    Cache    │
│  (SQLite/    │  │    (Qdrant)     │  │   (Redis)   │
│ PostgreSQL)  │  │                 │  │             │
└──────────────┘  └─────────────────┘  └─────────────┘
```

## Components

### 1. Knowledge Extraction (`z3_knowledge_extraction.py`)
Core extraction logic for:
- **Proof Patterns**: Tactic sequences, proof structures
- **Constraint Patterns**: Linear, nonlinear, boolean classifications
- **Strategies**: Problem-specific solving approaches
- **Mathematical Insights**: Bounds, invariants, relations

### 2. Integration Layer (`z3_knowledge_integration.py`)
Bridges Z3 extraction with knowledge engine:
- Transforms Z3 patterns to knowledge artifacts
- Manages storage across database/vector store/cache
- Provides unified API for knowledge operations

### 3. Database Models (`z3_database_models.py`)
SQLAlchemy models for Z3-specific knowledge:
- `Z3KnowledgeEntry`: Base knowledge entries
- `Z3ProofPattern`: Proof tactic patterns
- `Z3ConstraintPattern`: Constraint classifications
- `Z3Strategy`: Learned strategies
- `Z3MathematicalInsight`: Mathematical insights
- `Z3SolverResult`: Solver execution results

### 4. Auto-Extraction (`z3_auto_extraction.py`)
Automatic knowledge capture:
- Hooks into solver operations
- Decorator for solver functions
- Mixin for solver classes
- Optional patching of Z3 integration

### 5. REST API (`z3_api.py`)
FastAPI endpoints:
- `POST /z3-knowledge/extract` - Extract and store knowledge
- `POST /z3-knowledge/recommend-strategy` - Get strategy recommendation
- `POST /z3-knowledge/search-patterns` - Search knowledge base
- `GET /z3-knowledge/summary` - Get knowledge statistics
- `GET /z3-knowledge/patterns/{type}` - Get patterns by type

### 6. Migration (`z3_migration.py`)
Database setup:
```bash
# Create tables with seed data
python -m knowledge_engine.integrations.z3_migration --create --seed

# Verify migration
python -m knowledge_engine.integrations.z3_migration --verify

# Recreate (drop and create)
python -m knowledge_engine.integrations.z3_migration --recreate
```

## Quick Start

### 1. Setup Database

```bash
# Run migration to create tables
python -m knowledge_engine.integrations.z3_migration --create --seed
```

### 2. Enable Auto-Extraction

```python
from knowledge_engine.integrations import enable_auto_extraction

# Enable automatic knowledge extraction
enable_auto_extraction()
```

### 3. Use Decorator

```python
from knowledge_engine.integrations import auto_extract_knowledge

@auto_extract_knowledge(problem_type="linear")
async def solve_constraints(constraints):
    # Your solving logic
    result = await z3_solve(constraints)
    return result
```

### 4. Manual Extraction

```python
from knowledge_engine.integrations import Z3KnowledgeIntegration

async def main():
    # Initialize integration
    integration = Z3KnowledgeIntegration()
    await integration.initialize()
    
    # Process solver result
    result = await integration.process_solver_result(
        result=solver_result,
        problem_statement=problem,
        problem_type="constraint_solving"
    )
    
    print(f"Stored artifacts: {result['stored_artifacts']}")
```

### 5. Run API Server

```bash
python -m knowledge_engine.integrations.z3_api
```

Access at `http://localhost:8766`

## API Examples

### Extract Knowledge

```bash
curl -X POST http://localhost:8766/z3-knowledge/extract \
  -H "Content-Type: application/json" \
  -d '{
    "result_data": {
      "success": true,
      "model": {"assignments": {"x": 5}},
      "constraints": ["(> x 0)", "(< x 10)"],
      "solving_time": 1.5
    },
    "problem_statement": "Find x satisfying 0 < x < 10",
    "problem_type": "linear"
  }'
```

### Get Strategy Recommendation

```bash
curl -X POST http://localhost:8766/z3-knowledge/recommend-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "problem_features": {
      "type": "linear",
      "var_count": 5,
      "constraint_count": 10
    }
  }'
```

### Search Patterns

```bash
curl -X POST http://localhost:8766/z3-knowledge/search-patterns \
  -H "Content-Type: application/json" \
  -d '{
    "query": "linear constraints",
    "pattern_type": "constraint",
    "top_k": 5
  }'
```

## Configuration

### Database Configuration

```python
config = {
    "database": {
        "type": "postgresql",  # or "sqlite"
        "host": "localhost",
        "port": 5432,
        "username": "user",
        "password": "pass",
        "database": "z3_knowledge"
    },
    "vector_store": {
        "type": "qdrant",
        "host": "localhost",
        "port": 6333,
        "collection_name": "z3_knowledge"
    },
    "cache": {
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "ttl_seconds": 3600
    }
}
```

### Integration with Main Knowledge Engine

```python
from knowledge_engine.data.storage import KnowledgeStorageEngine
from knowledge_engine.integrations import Z3KnowledgeIntegration

# Use existing storage engine
storage = KnowledgeStorageEngine(config)
await storage.initialize()

# Create integration with shared storage
z3_integration = Z3KnowledgeIntegration(storage_engine=storage)
```

## Knowledge Flow

1. **Z3 Solver Execution**
   - Problem is submitted to Z3 solver
   - Solver produces result (sat/unsat/unknown)

2. **Knowledge Extraction**
   - Proof patterns extracted from proof trees
   - Constraint patterns classified
   - Strategies learned from successful solves
   - Mathematical insights extracted from models

3. **Knowledge Transformation**
   - Z3 patterns converted to knowledge artifacts
   - Metadata and embeddings generated
   - Confidence scores calculated

4. **Knowledge Storage**
   - Stored in database (SQLite/PostgreSQL)
   - Embeddings stored in vector store (Qdrant)
   - Cache updated (Redis)

5. **Knowledge Reuse**
   - Similar problems trigger pattern matching
   - Strategy recommendations provided
   - Previous insights applied

## Testing

```bash
# Run migration and verify
python -m knowledge_engine.integrations.z3_migration --create --seed --verify

# Test integration
python -c "
from knowledge_engine.integrations import Z3KnowledgeIntegration
import asyncio

async def test():
    integration = Z3KnowledgeIntegration()
    await integration.initialize()
    summary = await integration.get_knowledge_summary()
    print(f'Storage available: {summary[\"storage_available\"]}')

asyncio.run(test())
"
```

## Integration Checklist

- [x] Z3 knowledge extraction module
- [x] Database models (SQLAlchemy)
- [x] Migration script
- [x] Integration layer
- [x] Auto-extraction hooks
- [x] REST API endpoints
- [x] FastAPI application
- [x] Package initialization
- [x] Documentation

## Dependencies

```
z3-solver>=4.12.0
sqlalchemy>=2.0.0
fastapi>=0.100.0
pydantic>=2.0.0
qdrant-client>=1.5.0
redis>=4.5.0
aiosqlite>=0.19.0
asyncpg>=0.28.0
```

## Files Created

```
knowledge_engine/integrations/
├── __init__.py                          # Package initialization
├── z3_knowledge_integration.py          # Main integration (21,732 bytes)
├── z3_database_models.py                # SQLAlchemy models (13,851 bytes)
├── z3_migration.py                      # Migration script (10,204 bytes)
├── z3_auto_extraction.py                # Auto-extraction hooks (13,034 bytes)
├── z3_api.py                            # FastAPI endpoints (14,128 bytes)
└── Z3_KNOWLEDGE_INTEGRATION_README.md   # This documentation
```

## Total Implementation

- **6 new files**
- **~73,000 lines of code**
- Full database schema
- REST API with 6 endpoints
- Auto-extraction capabilities
- Comprehensive documentation

## Future Enhancements

1. **Knowledge Graph Integration**: Connect Z3 patterns to broader KG
2. **Machine Learning**: Train models on extracted patterns
3. **Visualization**: UI for exploring Z3 knowledge
4. **Collaborative Learning**: Share patterns across instances
5. **Advanced Analytics**: Pattern effectiveness tracking

## Support

For issues or questions:
- Check logs in `z3_knowledge.log`
- Run verification: `python -m knowledge_engine.integrations.z3_migration --verify`
- Review API docs at `http://localhost:8766/docs`
