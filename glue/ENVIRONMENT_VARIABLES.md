# OpenEvolve Environment Variables Registry

**Single Source of Truth for All Configuration**

Following the Federation Constitution - Law of Configuration Explicitness:
- **NO MAGIC DEFAULTS** - Every configurable value must be explicitly set
- Application **CRASHES IMMEDIATELY** if required vars are missing
- All values are type-validated at startup

---

## Table of Contents

- [Core OpenEvolve Configuration](#core-openevolve-configuration)
- [API Gateway Configuration](#api-gateway-configuration)
- [Infrastructure Configuration](#infrastructure-configuration)
- [Adapter Configuration](#adapter-configuration)
  - [BubbleLab Adapter](#bubblelab-adapter)
  - [Graphiti Adapter](#graphiti-adapter)
  - [VectorDB Adapter](#vectordb-adapter)
  - [OpenEvolve Adapter](#openevolve-adapter)
  - [ICR Adapter](#icr-adapter)
  - [LeanAide Adapter](#leanaide-adapter)
  - [Z3 Adapter](#z3-adapter)
  - [RESE Adapters](#rese-adapters)
- [Knowledge Engine Configuration](#knowledge-engine-configuration)
- [Plugin Configuration](#plugin-configuration)
- [Event Bus Configuration](#event-bus-configuration)
- [Observability Configuration](#observability-configuration)
- [PES (Prompt Evolution Strategy) Configuration](#pes-prompt-evolution-strategy-configuration)

---

## Core OpenEvolve Configuration

### Logging & Telemetry

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_LOG_LEVEL` | string | No | INFO | Log level: DEBUG, INFO, WARNING, ERROR | `INFO` | Must be one of: DEBUG, INFO, WARNING, ERROR |
| `LOG_LEVEL` | string | No | INFO | Global log level (fallback) | `INFO` | Must be one of: DEBUG, INFO, WARNING, ERROR |
| `LOG_FORMAT` | string | No | json | Log format: json or text | `json` | Must be one of: json, text |

### Service Ports

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_ORCHESTRATOR_PORT` | port | No | 8080 | Orchestrator service port | `8080` | 1-65535 |
| `ORCHESTRATOR_PORT` | port | No | 8080 | Orchestrator port (fallback) | `8080` | 1-65535 |

### Service Toggles

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_SERVICES__REST_API` | boolean | No | true | Enable REST API service | `true` | true/false |
| `OPENEVOLVE_SERVICES__GRAPHQL_API` | boolean | No | true | Enable GraphQL API service | `true` | true/false |
| `OPENEVOLVE_SERVICES__EVENT_BUS` | boolean | No | true | Enable Event Bus service | `true` | true/false |
| `OPENEVOLVE_SERVICES__MCP_SERVER` | boolean | No | true | Enable MCP Server service | `true` | true/false |
| `OPENEVOLVE_SERVICES__TELEMETRY` | boolean | No | true | Enable Telemetry service | `true` | true/false |

### REST API Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_REST_API__HOST` | string | No | 0.0.0.0 | REST API bind address | `0.0.0.0` | Valid IP or hostname |
| `OPENEVOLVE_REST_API__PORT` | port | No | 8000 | REST API port | `8000` | 1-65535 |
| `OPENEVOLVE_REST_API__CORS_ORIGINS` | string | No | * | CORS origins (comma-separated) | `http://localhost:3000,https://example.com` | Valid URLs or * |
| `MAX_REQUEST_SIZE` | number | No | 100 | Max request size in MB | `100` | Positive integer |
| `TIMEOUT` | number | No | 300 | Request timeout in seconds | `300` | Positive integer |

### GraphQL Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_GRAPHQL__HOST` | string | No | 0.0.0.0 | GraphQL API bind address | `0.0.0.0` | Valid IP or hostname |
| `OPENEVOLVE_GRAPHQL__PORT` | port | No | 8001 | GraphQL API port | `8001` | 1-65535 |
| `OPENEVOLVE_GRAPHQL__ENABLE_PLAYGROUND` | boolean | No | true | Enable GraphQL Playground (dev only) | `false` | true/false |

### Rate Limiting

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | number | No | 100 | Requests per minute per IP | `100` | Positive integer |
| `RATE_LIMIT_BURST_SIZE` | number | No | 10 | Burst capacity | `10` | Positive integer |
| `RATE_LIMIT_ENABLED` | boolean | No | true | Enable rate limiting | `true` | true/false |
| `RATE_LIMIT_PER_MINUTE` | number | No | 100 | Rate limit per minute | `100` | Positive integer |
| `RATE_LIMIT_BURST` | number | No | 10 | Burst size | `10` | Positive integer |

### Security

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `SECRET_KEY` | string | **YES** | NONE | Application secret key (generate with: `python -c "import secrets; print(secrets.token_hex(32))"`) | `a1b2c3d4e5f6...` | Min 32 chars, recommended 64 hex chars |
| `JWT_ALGORITHM` | string | No | HS256 | JWT signing algorithm | `HS256` | Valid JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | number | No | 30 | Access token lifetime in minutes | `30` | Positive integer |
| `REFRESH_TOKEN_EXPIRE_DAYS` | number | No | 7 | Refresh token lifetime in days | `7` | Positive integer |
| `API_KEY` | string | No | NONE | External API key (optional) | `sk-...` | Valid API key format |

### Worker Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `WORKERS` | number | No | 0 | Number of worker processes (0 = auto) | `4` | Non-negative integer |

---

## API Gateway Configuration

### Server Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `API_HOST` | string | No | 0.0.0.0 | Gateway bind address | `0.0.0.0` | Valid IP or hostname |
| `API_PORT` | port | No | 8000 | Gateway port | `8000` | 1-65535 |
| `API_RELOAD` | boolean | No | False | Auto-reload on code changes | `True` | true/false |

### Clerk JWT Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `CLERK_ISSUER` | url | No | NONE | Clerk JWT issuer URL | `https://your-instance.clerk.accounts.dev` | Valid HTTPS URL |
| `CLERK_JWKS_URL` | url | No | NONE | Override JWKS URL | `https://your-instance.clerk.accounts.dev/.well-known/jwks.json` | Valid HTTPS URL |
| `CLERK_AUDIENCE` | string | No | NONE | JWT audience claim | `my-app-id` | Non-empty string |
| `CLERK_JWKS_CACHE_TTL_SECONDS` | number | No | 3600 | JWKS cache TTL | `3600` | Positive integer |

### CORS Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `CORS_ORIGINS` | string | No | http://localhost:3000,http://localhost:8000 | Allowed CORS origins (JSON array) | `["http://localhost:3000"]` | Valid JSON array of URLs |
| `CORS_ALLOW_CREDENTIALS` | boolean | No | True | Allow credentials | `True` | true/false |
| `CORS_ALLOW_METHODS` | string | No | * | Allowed methods (JSON array) | `["*"]` | Valid JSON array |
| `CORS_ALLOW_HEADERS` | string | No | * | Allowed headers (JSON array) | `["*"]` | Valid JSON array |

### File Upload Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `MAX_FILE_SIZE` | number | No | 10485760 | Max file size in bytes (10MB default) | `10485760` | Positive integer |
| `UPLOAD_DIR` | string | No | ./uploads | Upload directory path | `./uploads` | Valid directory path |

### WebSocket Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `WS_HEARTBEAT_INTERVAL` | number | No | 30 | WebSocket heartbeat interval in seconds | `30` | Positive integer |
| `WS_MAX_CONNECTIONS` | number | No | 100 | Max concurrent WebSocket connections | `100` | Positive integer |

### Evolution Orchestrator

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `EVOLUTION_ORCHESTRATOR_URL` | url | No | http://localhost:8003/evolve | Evolution orchestrator endpoint | `http://localhost:8003/evolve` | Valid HTTP URL |

---

## Infrastructure Configuration

### Valkey/Redis Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `VALKEY_HOST` | string | No | localhost | Valkey/Redis host | `localhost` or `valkey` | Valid hostname |
| `VALKEY_PORT` | port | No | 6379 | Valkey/Redis port | `6379` | 1-65535 |
| `VALKEY_PASSWORD` | string | No | NONE | Valkey/Redis password | `your-password` | String |
| `REDIS_HOST` | string | No | localhost | Redis host (fallback) | `localhost` | Valid hostname |
| `REDIS_PORT` | port | No | 6379 | Redis port (fallback) | `6379` | 1-65535 |
| `REDIS_DB` | number | No | 0 | Redis database number | `0` | 0-15 |
| `REDIS_PASSWORD` | string | No | NONE | Redis password | `your-password` | String |

### Database Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `DATABASE_URL` | url | No | NONE | Database connection URL | `postgresql://user:pass@localhost/db` or `sqlite:///./openevolve.db` | Valid database URL |
| `DB_HOST` | string | No | localhost | Database host | `localhost` | Valid hostname |
| `DB_PORT` | port | No | 5432 | Database port | `5432` | 1-65535 |
| `DB_USERNAME` | string | No | openevolve | Database username | `openevolve` | Non-empty string |
| `DB_PASSWORD` | string | **YES** | NONE | Database password | `your-password-here` | Non-empty string |
| `DB_NAME` | string | No | openevolve_kg | Database name | `openevolve_kg` | Non-empty string |

### Backup Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `BACKUP_DIR` | string | No | ./backups | Backup directory | `./backups` | Valid directory path |
| `BACKUP_RETENTION_DAYS` | number | No | 30 | Backup retention period in days | `30` | Positive integer |

### Development Settings

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `DEBUG` | boolean | No | false | Enable debug mode | `false` | true/false |
| `RELOAD` | boolean | No | false | Enable auto-reload (dev only) | `false` | true/false |
| `NODE_ENV` | string | No | production | Node environment | `development` or `production` | development or production |

---

## Adapter Configuration

### BubbleLab Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `BUBBLELAB_PORT` | port | No | 3001 | BubbleLab adapter port | `3001` | 1-65535 |
| `BUBBLELAB_API_URL` | url | **YES** | NONE | BubbleLab core API URL | `http://bubblelab-core:8000` | Valid HTTP URL |
| `BUBBLELAB_API_KEY` | string | **YES** | NONE | BubbleLab API key | `your-bubblelab-api-key-here` | Non-empty string |
| `BUBBLELAB_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `BUBBLELAB_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |

### Graphiti Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `GRAPHITI_PORT` | port | No | 3000 | Graphiti adapter port | `3000` | 1-65535 |
| `NEO4J_URI` | url | No | bolt://localhost:7687 | Neo4j connection URI | `bolt://neo4j:7687` | Valid bolt:// URL |
| `NEO4J_USER` | string | No | neo4j | Neo4j username | `neo4j` | Non-empty string |
| `NEO4J_PASSWORD` | string | **YES** | NONE | Neo4j password | `your-neo4j-password-here` | Non-empty string |
| `GRAPHITI_API_URL` | url | No | http://localhost:8000 | Graphiti API URL | `http://graphiti:8000` | Valid HTTP URL |
| `GRAPHITI_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `OPENAI_API_KEY` | string | No | NONE | OpenAI API key for entity extraction | `sk-your-openai-api-key` | Valid OpenAI key format |
| `ANTHROPIC_API_KEY` | string | No | NONE | Anthropic API key (alternative) | `sk-ant-your-key` | Valid Anthropic key format |
| `UPDATE_COMMUNITIES` | boolean | No | false | Enable community detection | `false` | true/false |
| `STORE_RAW_EPISODES` | boolean | No | true | Store raw episode content | `true` | true/false |

### VectorDB Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `VECTORDB_PORT` | port | No | 3004 | VectorDB adapter port | `3004` | 1-65535 |
| `VECTORDB_TYPE` | string | No | pinecone | Vector database type | `pinecone`, `qdrant`, `weaviate` | Must be: pinecone, qdrant, or weaviate |
| `VECTORDB_API_URL` | url | **YES** | NONE | VectorDB API URL | `https://your-pinecone-instance.pinecone.io` or `http://qdrant:6333` | Valid HTTP/HTTPS URL |
| `VECTORDB_API_KEY` | string | **YES** | NONE | VectorDB API key | `your-pinecone-api-key-here` | Non-empty string |
| `VECTORDB_CONNECTION_STRING` | string | No | NONE | VectorDB connection string (alternative) | `mongodb://user:pass@host:port` | Valid connection string |
| `VECTORDB_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `VECTORDB_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |
| `PINECONE_ENVIRONMENT` | string | No | NONE | Pinecone environment (if using Pinecone) | `us-west1-gcp` | Non-empty string |
| `TIMEOUT_MS` | number | No | 5000 | General timeout in milliseconds | `5000` | Positive integer |
| `MAX_RETRIES` | number | No | 3 | General max retries | `3` | Non-negative integer |

### OpenEvolve Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_PORT` | port | No | 3003 | OpenEvolve adapter port | `3003` | 1-65535 |
| `OPENEVOLVE_API_URL` | url | **YES** | NONE | OpenEvolve core API URL | `http://openevolve-core:8000` | Valid HTTP URL |
| `OPENEVOLVE_API_KEY` | string | **YES** | NONE | OpenEvolve API key | `your-openevolve-api-key-here` | Non-empty string |
| `OPENEVOLVE_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `OPENEVOLVE_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |
| `DEFAULT_REQUEST_TIMEOUT` | number | No | 30000 | Default request timeout in milliseconds | `30000` | Positive integer |

### ICR Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `ICR_PORT` | port | No | 3002 | ICR adapter port | `3002` | 1-65535 |
| `ICR_API_URL` | url | **YES** | NONE | ICR core API URL | `http://icr-core:8000` | Valid HTTP URL |
| `ICR_API_KEY` | string | **YES** | NONE | ICR API key | `your-icr-api-key-here` | Non-empty string |
| `ICR_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `ICR_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |
| `ICR_RETRY_DELAY_MS` | number | No | 1000 | Retry delay in milliseconds | `1000` | Positive integer |

### LeanAide Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `LEANAIDE_PORT` | port | No | 3006 | LeanAide adapter port | `3006` | 1-65535 |
| `LEANAIDE_API_URL` | url | **YES** | NONE | LeanAide core API URL | `http://leanaide-core:8000` | Valid HTTP URL |
| `LEANAIDE_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `LEANAIDE_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |

### Z3 Adapter

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `Z3_PORT` | port | No | 3005 | Z3 adapter port | `3005` | 1-65535 |
| `Z3_SOLVER_PATH` | string | No | /usr/bin/z3 | Path to Z3 solver binary | `/usr/bin/z3` | Valid file path |
| `Z3_TIMEOUT_MS` | number | No | 30000 | Z3 solver timeout in milliseconds | `30000` | Positive integer |
| `Z3_MAX_MEMORY_MB` | number | No | 4096 | Max memory for Z3 in MB | `4096` | Positive integer |
| `Z3_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |
| `Z3_API_URL` | url | No | NONE | Z3 API URL (if using API instead of binary) | `http://z3-service:8080` | Valid HTTP URL |
| `Z3_HEALTH_CHECK` | string | No | /health | Z3 health check path | `/health` | Non-empty string |
| `Z3_VERIFY_PATH` | string | No | /verify | Z3 verify endpoint path | `/verify` | Non-empty string |

### RESE Adapters

#### RESE DEE (Deep Exploration Engine)

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RESE_DEE_PORT` | port | No | 8001 | RESE DEE adapter port | `8001` | 1-65535 |
| `RESE_DEE_EXPLORATION_DEPTH` | number | No | 10 | Maximum exploration depth | `10` | Positive integer |
| `RESE_DEE_MCTS_ITERATIONS` | number | No | 1000 | MCTS iterations | `1000` | Positive integer |
| `RESE_DEE_MCTS_EXPLORATION_CONSTANT` | number | No | 1.414 | MCTS exploration constant (UCB1) | `1.414` | Positive float |
| `RESE_DEE_CONVERGENCE_THRESHOLD` | number | No | 0.001 | Convergence threshold | `0.001` | Positive float |
| `RESE_DEE_EXPLORATION_TIMEOUT_MS` | number | No | 10000 | Exploration timeout in milliseconds | `10000` | Positive integer |
| `RESE_DEE_MAX_HYPOTHESES` | number | No | 100 | Maximum hypotheses to track | `100` | Positive integer |
| `RESE_DEE_PATTERN_RECOGNITION_THRESHOLD` | number | No | 0.7 | Pattern recognition threshold | `0.7` | 0.0-1.0 |

#### RESE LLTL (Lean Latent Transformation Layer)

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RESE_LLTDL_PORT` | port | No | 8002 | RESE LLTL adapter port | `8002` | 1-65535 |
| `RESE_LLTDL_ENCODING_DIM` | number | No | 128 | Encoding dimension | `128` | Positive integer |
| `RESE_LLTDL_USE_POSITIONAL` | boolean | No | true | Use positional encoding | `true` | true/false |
| `RESE_LLTDL_USE_TYPE_EMBEDDING` | boolean | No | true | Use type embedding | `true` | true/false |
| `RESE_LLTDL_USE_CATEGORY_EMBEDDING` | boolean | No | true | Use category embedding | `true` | true/false |
| `RESE_LLTDL_MAX_SEQUENCE_LENGTH` | number | No | 512 | Max sequence length | `512` | Positive integer |
| `RESE_LLTDL_CACHE_SIZE` | number | No | 1000 | Cache size | `1000` | Positive integer |
| `RESE_LLTDL_DEFAULT_LOSS_TYPE` | string | No | mse | Default loss type | `mse` | mse, crossentropy, etc. |
| `RESE_LLTDL_COMBINATION_STRATEGY` | string | No | weighted_sum | Embedding combination strategy | `weighted_sum` | weighted_sum, concatenation, etc. |
| `RESE_LLTDL_NORMALIZE_WEIGHTS` | boolean | No | true | Normalize combination weights | `true` | true/false |
| `RESE_LLTDL_LEARNING_RATE` | number | No | 0.001 | Learning rate | `0.001` | Positive float |
| `RESE_LLTDL_TIMEOUT_MS` | number | No | 3000 | Request timeout in milliseconds | `3000` | Positive integer |
| `RESE_LLTDL_ENABLE_RTREE` | boolean | No | false | Enable R-tree indexing | `false` | true/false |
| `RESE_LLTDL_ENABLE_LSH` | boolean | No | false | Enable LSH indexing | `false` | true/false |
| `RESE_LLTDL_ENABLE_HAG` | boolean | No | false | Enable HAG indexing | `false` | true/false |
| `RESE_LLTDL_CONTRADICTION_THRESHOLD` | number | No | 0.8 | Contradiction detection threshold | `0.8` | 0.0-1.0 |
| `RESE_LLTDL_MAX_CONTRADICTIONS` | number | No | 1000 | Max contradictions to track | `1000` | Positive integer |

#### RESE SCE (Symbolic Constraint Engine)

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RESE_SCE_PORT` | port | No | 8003 | RESE SCE adapter port | `8003` | 1-65535 |
| `SCE_TIMEOUT_MS` | number | No | 5000 | Default timeout in milliseconds | `5000` | Positive integer |
| `SCE_CONSTRAINT_TIMEOUT_MS` | number | No | 3000 | Constraint solving timeout | `3000` | Positive integer |
| `SCE_CONTRADICTION_DETECTION_TIMEOUT_MS` | number | No | 10000 | Contradiction detection timeout | `10000` | Positive integer |
| `SCE_MAX_ITERATIONS` | number | No | 1000 | Max solving iterations | `1000` | Positive integer |
| `SCE_MAX_CONSTRAINTS` | number | No | 10000 | Max constraints | `10000` | Positive integer |
| `SCE_MAX_CONTRADICTION_SET_SIZE` | number | No | 100 | Max contradiction set size | `100` | Positive integer |
| `SCE_CIRCUIT_BREAKER_THRESHOLD` | number | No | 5 | Circuit breaker failure threshold | `5` | Positive integer |
| `SCE_CIRCUIT_BREAKER_TIMEOUT_MS` | number | No | 60000 | Circuit breaker timeout in milliseconds | `60000` | Positive integer |
| `SCE_ENABLE_LEAN4_INTEGRATION` | boolean | No | false | Enable Lean 4 integration | `false` | true/false |
| `SCE_ENABLE_TACIT_MINING` | boolean | No | true | Enable tacit assumption mining | `true` | true/false |
| `SCE_CONTRADICTION_DETECTION` | boolean | No | true | Enable contradiction detection | `true` | true/false |
| `SCE_FORMAL_VERIFICATION` | boolean | No | true | Enable formal verification | `true` | true/false |

#### RESE Phase 2, 4, Verification, Workflow, Z3 Bridge

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RESE_PHASE2_PORT` | port | No | 8004 | RESE Phase 2 adapter port | `8004` | 1-65535 |
| `RESE_PHASE4_PORT` | port | No | 8006 | RESE Phase 4 adapter port | `8006` | 1-65535 |
| `RESE_VERIFICATION_PORT` | port | No | 8007 | RESE Verification adapter port | `8007` | 1-65535 |
| `RESE_Z3_BRIDGE_PORT` | port | No | 8008 | RESE Z3 Bridge adapter port | `8008` | 1-65535 |
| `RESE_LEANAIDE_WORKFLOW_PORT` | port | No | 8009 | RESE LeanAide Workflow adapter port | `8009` | 1-65535 |
| `RESE_LEANAIDE_API_URL` | url | No | http://leanaide-core:8000 | LeanAide API URL | `http://leanaide-core:8000` | Valid HTTP URL |
| `RESE_LEANAIDE_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `RESE_LEANAIDE_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |

### Other Adapters

#### Curie-GlobalChem Integration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `CURIE_GLOBALCHEM_PORT` | port | No | 5000 | Curie adapter port | `5000` | 1-65535 |
| `CURIE_GLOBALCHEM_API_URL` | url | **YES** | NONE | Curie API URL | `http://curie-core:8000` | Valid HTTP URL |
| `CURIE_GLOBALCHEM_API_KEY` | string | **YES** | NONE | Curie API key | `your-curie-api-key-here` | Non-empty string |
| `CURIE_GLOBALCHEM_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `CURIE_GLOBALCHEM_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |

#### LMQL-DSPY Integration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `LMQL_DSPY_PORT` | port | No | 5001 | LMQL adapter port | `5001` | 1-65535 |
| `LMQL_DSPY_API_URL` | url | **YES** | NONE | LMQL API URL | `http://lmql-core:8000` | Valid HTTP URL |
| `LMQL_DSPY_API_KEY` | string | **YES** | NONE | LMQL API key | `your-lmql-api-key-here` | Non-empty string |
| `LMQL_DSPY_TIMEOUT_MS` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |
| `LMQL_DSPY_MAX_RETRIES` | number | No | 3 | Max retry attempts | `3` | Non-negative integer |

#### RagBits-Graphiti Sync

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RAGBITS_GRAPHITI_SYNC_PORT` | port | No | 3010 | RagBits-Graphiti sync port | `3010` | 1-65535 |
| `RAGBITS_API_URL` | url | **YES** | NONE | RagBits API URL | `http://ragbits-core:8000` | Valid HTTP URL |
| `GRAPHITI_API_URL` | url | **YES** | NONE | Graphiti API URL | `http://graphiti:3000` | Valid HTTP URL |
| `RAGBITS_GRAPHITI_SYNC_TIMEOUT_MS` | number | No | 30000 | Sync timeout in milliseconds | `30000` | Positive integer |
| `RAGBITS_GRAPHITI_SYNC_INTERVAL_MS` | number | No | 60000 | Sync interval in milliseconds | `60000` | Positive integer |

---

## Knowledge Engine Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `VECTOR_STORE_HOST` | string | No | localhost | Vector store host | `localhost` | Valid hostname |
| `VECTOR_STORE_PORT` | port | No | 6333 | Vector store port | `6333` | 1-65535 |
| `CACHE_HOST` | string | No | localhost | Cache host (Redis) | `localhost` | Valid hostname |
| `CACHE_PORT` | port | No | 6379 | Cache port | `6379` | 1-65535 |
| `CACHE_DB` | number | No | 0 | Cache database number | `0` | 0-15 |
| `SERVER_HOST` | string | No | 0.0.0.0 | Server bind address | `0.0.0.0` | Valid IP |
| `SERVER_PORT` | port | No | 8000 | Server port | `8000` | 1-65535 |
| `SERVER_WORKERS` | number | No | 4 | Number of worker processes | `4` | Positive integer |
| `LLM_PROVIDER` | string | No | openai | LLM provider | `openai`, `anthropic`, `google` | Valid provider name |
| `LLM_MODEL` | string | No | gpt-4o | LLM model name | `gpt-4o` | Valid model name |
| `LLM_API_KEY` | string | **YES** | NONE | LLM API key | `your_api_key_here` | Non-empty string |
| `LLM_BASE_URL` | url | No | NONE | LLM base URL (for custom endpoints) | `https://api.example.com` | Valid HTTP URL |
| `JWT_SECRET` | string | **YES** | NONE | JWT secret for auth | `your_jwt_secret_here` | Min 32 chars |
| `KE_VALIDATE_CONFIG` | string | No | warn | Config validation mode | `warn`, `error`, `ignore` | One of: warn, error, ignore |

---

## Plugin Configuration

### BubbleLab Plugin

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `VITE_OPENEVOLVE_API_URL` | url | No | http://localhost:8000 | OpenEvolve API base URL for plugin | `http://localhost:8000` | Valid HTTP URL |

### Datapizza Plugin

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `DATAPIZZA_API_URL` | url | No | /api/datapizza | Datapizza API URL | `https://api.datapizza.com` | Valid HTTP URL |
| `DATAPIZZA_API_KEY` | string | No | NONE | Datapizza API key | `your-key-here` | Non-empty string |
| `DATAPIZZA_TIMEOUT` | number | No | 30000 | Request timeout in milliseconds | `30000` | Positive integer |

### OpenAI / LLM Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENAI_API_KEY` | string | No | NONE | OpenAI API key | `sk-...` | Valid OpenAI key format |
| `ANTHROPIC_API_KEY` | string | No | NONE | Anthropic (Claude) API key | `sk-ant-...` | Valid Anthropic key format |
| `GOOGLE_API_KEY` | string | No | NONE | Google (Gemini) API key | `AIza...` | Valid Google key format |
| `GEMINI_API_KEY` | string | No | NONE | Gemini API key (alias for GOOGLE_API_KEY) | `AIza...` | Valid Google key format |
| `AI_API_KEY` | string | No | NONE | Generic AI API key (fallback) | `sk-...` | Non-empty string |
| `API_KEY` | string | No | NONE | Generic API key (fallback) | `sk-...` | Non-empty string |
| `OPENROUTER_API_KEY` | string | No | NONE | OpenRouter API key | `sk-or-...` | Valid OpenRouter key |

---

## Event Bus Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `EVENT_BUS_TYPE` | string | No | memory | Event bus implementation | `memory`, `redis`, `valkey` | One of: memory, redis, valkey |
| `EVENT_BUS_URL` | url | No | NONE | Event bus connection URL | `redis://event-bus:6379` | Valid redis:// URL |
| `OPENEVOLVE_EVENT_BUS__ENABLED` | boolean | No | true | Enable event bus | `true` | true/false |
| `OPENEVOLVE_EVENT_BUS__BACKEND` | string | No | valkey | Event bus backend type | `valkey` | One of: memory, redis, valkey |
| `OPENEVOLVE_EVENT_BUS__HOST` | string | No | localhost | Event bus host | `localhost` | Valid hostname |
| `OPENEVOLVE_EVENT_BUS__PORT` | port | No | 6379 | Event bus port | `6379` | 1-65535 |
| `OPENEVOLVE_EVENT_BUS__PASSWORD` | string | No | NONE | Event bus password | `your-password` | String |
| `EVENT_BUS_MAX_EVENTS` | number | No | 10000 | Max events in memory | `10000` | Positive integer |
| `EVENT_BUS_PERSIST_EVENTS` | boolean | No | true | Persist events to disk | `true` | true/false |

### Unified Knowledge Query

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `RAGBITS_URL` | url | No | NONE | RagBits service URL | `http://ragbits:8000` | Valid HTTP URL |
| `GRAPHITI_URL` | url | No | NONE | Graphiti service URL | `http://graphiti:3000` | Valid HTTP URL |
| `VECTORDB_URL` | url | No | NONE | VectorDB service URL | `http://vectordb:3004` | Valid HTTP URL |

---

## Observability Configuration

### OpenTelemetry

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `OPENEVOLVE_TELEMETRY__ENABLED` | boolean | No | true | Enable telemetry | `true` | true/false |
| `OPENEVOLVE_TELEMETRY__SERVICE_NAME` | string | No | openevolve | Service name for telemetry | `openevolve` | Non-empty string |
| `OPENEVOLVE_TELEMETRY__OTLP_ENDPOINT` | url | No | http://localhost:4317 | OTLP endpoint | `http://localhost:4317` | Valid HTTP URL |
| `OPENEVOLVE_TELEMETRY__METRICS_ENABLED` | boolean | No | true | Enable metrics collection | `true` | true/false |
| `OPENEVOLVE_TELEMETRY__TRACING_ENABLED` | boolean | No | true | Enable distributed tracing | `true` | true/false |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | url | No | http://localhost:4317 | OTLP endpoint (fallback) | `http://localhost:4317` | Valid HTTP URL |
| `SERVICE_NAME` | string | No | unknown-service | Service name (fallback) | `my-service` | Non-empty string |
| `ENABLE_TRACING` | boolean | No | false | Enable tracing (adapter-level) | `false` | true/false |

### Metrics

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `PROMETHEUS_PORT` | port | No | 9090 | Prometheus metrics port | `9090` | 1-65535 |
| `METRICS_PREFIX` | string | No | openevolve_ | Metrics prefix | `openevolve_` | Non-empty string |

---

## PES (Prompt Evolution Strategy) Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `PES_COST_OPTIMIZATION` | boolean | No | false | Enable cost optimization | `false` | true/false |
| `PES_MAX_COST_USD` | number | No | 10.0 | Maximum budget in USD | `10.0` | Positive float |
| `PES_COST_WARNING` | number | No | 0.7 | Budget warning threshold (0.0-1.0) | `0.7` | 0.0-1.0 |
| `PES_COST_CRITICAL` | number | No | 0.9 | Budget critical threshold (0.0-1.0) | `0.9` | 0.0-1.0 |
| `PES_EARLY_STOPPING` | boolean | No | true | Enable early stopping | `true` | true/false |
| `PES_STOPPING_PATIENCE` | number | No | 5 | Early stopping patience | `5` | Positive integer |
| `PES_MIN_IMPROVEMENT` | number | No | 0.001 | Minimum improvement threshold | `0.001` | Positive float |
| `PES_PLANNING` | boolean | No | true | Enable PES planning phase | `true` | true/false |
| `PES_SUMMARIZATION` | boolean | No | true | Enable PES summarization phase | `true` | true/false |
| `PES_AUTO_SELECT` | boolean | No | true | Auto-select PES strategy | `true` | true/false |
| `PES_USE_CHEAP_MODELS` | boolean | No | true | Use cheaper models for execution | `true` | true/false |
| `PES_CHEAP_MODEL` | string | No | gpt-3.5-turbo | Cheap model name | `gpt-3.5-turbo` | Valid model name |
| `PES_EXPENSIVE_MODEL` | string | No | gpt-4o | Expensive model name | `gpt-4o` | Valid model name |
| `PES_PROMPT_TOKEN_PRICE` | number | No | 0.00001 | Price per prompt token | `0.00001` | Non-negative float |
| `PES_COMPLETION_TOKEN_PRICE` | number | No | 0.00003 | Price per completion token | `0.00003` | Non-negative float |

---

## Orchestration Configuration

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `PIPELINE_TIMEOUT_MS` | number | No | 300000 | Global pipeline timeout in milliseconds | `300000` | Positive integer |
| `MAX_RETRIES` | number | No | 3 | Global max retry attempts | `3` | Non-negative integer |
| `CIRCUIT_BREAKER_THRESHOLD` | number | No | 5 | Circuit breaker failure threshold | `5` | Positive integer |
| `CIRCUIT_BREAKER_TIMEOUT_MS` | number | No | 60000 | Circuit breaker timeout in milliseconds | `60000` | Positive integer |
| `CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS` | number | No | 3 | Half-open state attempts | `3` | Positive integer |
| `DLQ_MAX_SIZE` | number | No | 1000 | Dead Letter Queue max size | `1000` | Positive integer |

---

## Testing & Development

### Contract Testing

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `SKIP_INTEGRATION_TESTS` | boolean | No | false | Skip integration tests | `false` | true/false |
| `TIMEOUT_MS` | number | No | 30000 | Test timeout in milliseconds | `30000` | Positive integer |

### Jest Setup

| Variable | Type | Required | Default | Description | Example | Validation |
|----------|------|----------|---------|-------------|---------|------------|
| `VECTORDB_TYPE` | string | No | qdrant | Test VectorDB type | `qdrant` | Valid type name |
| `VECTORDB_URL` | string | No | http://localhost:6333 | Test VectorDB URL | `http://localhost:6333` | Valid HTTP URL |

---

## Summary

**Required Variables (Crash if Missing):**
- `SECRET_KEY` (or `JWT_SECRET`)
- `DB_PASSWORD` (if using database)
- `NEO4J_PASSWORD` (if using Graphiti)
- `LLM_API_KEY` (if using Knowledge Engine)
- `OPENAI_API_KEY` (if using entity extraction)
- All `*_API_KEY` variables for adapters you're using
- All `*_API_URL` variables for adapters you're using

**Validation Rules:**
- Ports: 1-65535
- URLs: Must parse as valid URL
- Booleans: `true`/`false` or `1`/`0`
- Numbers: Must be valid numeric values

**Default Behavior:**
- Application **crashes immediately** with clear error message if required var is missing
- Optional vars use documented defaults
- No silent failures - explicit configuration only
