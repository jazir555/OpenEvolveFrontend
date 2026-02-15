# Glue Layer Directory

## Purpose

The Glue Layer is the integration federation that orchestrates communication between 30+ immutable Open Source systems.

## Architecture

```
glue/
├── adapters/         # Per-project "Sidecars" (Anti-Corruption Layer)
├── orchestration/    # Global Event Bus / Workflow Engine
├── schemas/          # Canonical Data Models (Zod/Pydantic)
└── lib/              # Shared Utilities (Logger, Retry logic, Circuit Breakers)
```

## Core Principles

### 1. The Anti-Corruption Layer (ACL)

**Problem:** Project A uses `snake_case`, Project B uses `camelCase`, Project C uses `XML`.

**Solution:** Never pass data directly between projects. Always normalize to Canonical Schemas.

**Flow:**
```
[Source A] → [Adapter A (Normalize to Canonical)] → [Event Bus] → [Adapter B (Map to Target)] → [Target B]
```

### 2. The 6 Immutable Laws

1. **Air Gap**: No imports from `core-projects/`
2. **Runtime Truth**: Probe APIs before implementing (don't trust docs)
3. **Untouchable DB**: SELECT privileges only (unless restoring backup)
4. **Idempotency**: All actions safe to run 100 times
5. **Configuration Explicitness**: No magic defaults, validate all env vars
6. **UTC**: All timestamps in UTC ISO-8601

### 3. Failure Management

- **Transient Failure** (network blip) → Exponential Backoff Retry with Jitter
- **Logic Failure** (bad data) → Dead Letter Queue (DLQ), don't block pipeline
- **System Failure** (target down) → Circuit Breaker, stop hammering

## Directory Details

### `/adapters/{project}-adapter/`

Per-project sidecars that:
- Normalize data to/from Canonical Schemas
- Implement circuit breakers and retries
- Contain probe scripts to verify APIs
- Include contract tests to prevent breaking changes

### `/orchestration/`

Global coordination layer:
- Event Bus for async communication
- Workflow Engine for multi-step processes
- Correlation ID tracking
- Dead Letter Queue management

### `/schemas/`

Canonical Data Models:
- Define the "truth" for data structures
- Language-agnostic (Zod/TS, Pydantic/Python)
- Version controlled
- Single source of truth for contracts

### `/lib/`

Shared utilities:
- Structured Logger (JSON Lines format)
- Retry Logic (exponential backoff with jitter)
- Circuit Breaker implementation
- Timeout enforcement (MANDATORY on all HTTP requests)
- UTC timestamp conversion

## Development Protocol

Before writing any glue code:

1. **SCAN**: Read Core Project source to find APIs
2. **PROBE**: Write `probes/check_api.sh` to verify it works
3. **MODEL**: Define Canonical Schema
4. **IMPLEMENT**: Write Adapter with Circuit Breakers
5. **ISOLATE**: Dockerize (shares network, no file sharing)
6. **DOCUMENT**: Write ADR.md explaining "Why" and "Gotchas"

## Observability Requirements

All logs must be JSON Lines with:
- `correlation_id`: Request trace identifier
- `source_service`: Where the event originated
- `target_service`: Where the event is going
- `msg`: Human-readable message
- `error`: Error details (if applicable)

**Bad:** `console.log("Error happened")`
**Good:** `logger.error({ msg: "User Sync Failed", error: err.message, correlation_id: ctx.id, retry_count: 2 })`

## Environment Configuration

All environment variables are documented in the following files:

- **[ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md)** - Complete registry of all environment variables
- **[../.env.schema](../.env.schema)** - Schema template with all variables and defaults
- **[lib/env-schema.ts](./lib/env-schema.ts)** - TypeScript schema for validation
- **[lib/env-validator.ts](./lib/env-validator.ts)** - Validation library

### Quick Start

1. Copy the schema:
   ```bash
   cp .env.schema .env
   ```

2. Fill in required values (application crashes if missing):
   - `SECRET_KEY` or `JWT_SECRET` - Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`
   - `NEO4J_PASSWORD` - If using Graphiti
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - If using LLM features
   - All `*_API_URL` and `*_API_KEY` variables for adapters you're using

3. Validate configuration (automatic on startup):
   ```typescript
   import { validateEnvWithTypes } from './lib/env-validator';
   import { getSchemaForComponent } from './lib/env-schema';

   const config = validateEnvWithTypes(getSchemaForComponent('graphiti'));
   ```

### Required Variables (Application Crashes if Missing)

See the [complete documentation](./ENVIRONMENT_VARIABLES.md) for full details, but at minimum:
- `SECRET_KEY` or `JWT_SECRET` - Application security key
- Database password (if using database)
- All `*_API_KEY` variables for services you're connecting to
- All `*_API_URL` variables for services you're connecting to

### Configuration Validation

Following the **Law of Configuration Explicitness**:
- NO magic defaults
- All values must be explicitly set via environment variables
- Application crashes immediately with clear error if required vars are missing
- Type validation at startup (ports, URLs, booleans, numbers)

See [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) for complete documentation.
