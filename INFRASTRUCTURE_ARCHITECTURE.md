# OpenEvolve Infrastructure Architecture

## Visual Overview

```
                         OpenEvolve Development Environment

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                          Docker Host (localhost)                            │
  │                                                                              │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │                 openevolve-network (Bridge Network)                   │  │
  │  │                                                                      │  │
  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │  │
  │  │  │   PostgreSQL     │  │     Qdrant       │  │      Redis       │   │  │
  │  │  │  (port 5432)     │  │ (ports 6333/34)  │  │   (port 6379)    │   │  │
  │  │  │                  │  │                  │  │                  │   │  │
  │  │  │  • Users         │  │  • Collections   │  │  • Cache         │   │  │
  │  │  │  • Projects      │  │  • Vectors       │  │  • Sessions      │   │  │
  │  │  │  • Tasks         │  │  • Embeddings    │  │  • Messages      │   │  │
  │  │  │  • Schema        │  │  • Search        │  │  • Queue         │   │  │
  │  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │  │
  │  │           │ Volumes             │ Volumes             │ Volumes      │  │
  │  │           ▼                     ▼                     ▼              │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │                                                                              │
  └─────────────────────────────────────────────────────────────────────────────┘
```

## Service Dependencies

```
Hephaestus MCP:
├── PostgreSQL (task storage)
├── Qdrant (vector memory)
└── Redis (caching)

Vibe-Kanban:
├── PostgreSQL (app data)
└── Redis (sessions/caching)

OpenEvolve API:
├── PostgreSQL (data persistence)
├── Qdrant (knowledge engine)
└── Redis (LLM caching)
```

## Port Mapping Summary

```
Internal → External Mapping:

PostgreSQL:
└── 5432 → 5432 (Database)

Qdrant:
├── 6333 → 6333 (HTTP API)
└── 6334 → 6334 (gRPC API)

Redis:
└── 6379 → 6379 (Redis Protocol)

pgAdmin [optional]:
└── 80 → 5050 (Web UI)

Redis Commander [optional]:
└── 8081 → 8081 (Web UI)
```

## Connection Examples

### PostgreSQL
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="openevolve",
    user="openevolve",
    password="your-password"
)
```

### Qdrant
```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://localhost",
    port=6333
)
```

### Redis
```python
import redis

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)
```

## Health Check Endpoints

```
PostgreSQL:
└── tcp://localhost:5432
    └── Verify with: pg_isready -h localhost -p 5432

Qdrant:
└── http://localhost:6333/health
    └── Returns: { "status": "ok" }

Redis:
└── tcp://localhost:6379
    └── Verify with: redis-cli ping
    └── Returns: PONG
```

---

*This diagram provides a visual reference for the OpenEvolve infrastructure architecture.*
