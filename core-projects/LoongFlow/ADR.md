# ADR: LoongFlow HTTP API Wrapper

## Status

Accepted (2026-02-22)

## Context

LoongFlow is a sophisticated Plan-Execute-Summarize (PES) evolution framework, but it was designed as a CLI tool. To integrate it into the OpenEvolve federation, we need HTTP endpoints that other services can call.

### The Problem

1. **CLI-only interface**: LoongFlow's `GeneralPESAgent` extends `BasePESRunner` which expects to be run from command line
2. **Blocking execution**: Evolutions run to completion before returning; no way to track progress
3. **File-based config**: Requires YAML config files passed as CLI arguments
4. **No state API**: Running evolutions don't expose their state externally

### Constraints

From the Federation Constitution (CLAUDE.md):

- **Law of the Air Gap**: Cannot import from `core-projects/` in glue code
- **Law of Runtime Truth**: Must verify API actually works, not just assume
- **Law of Configuration Explicitness**: All config via environment variables
- **Law of Idempotency**: Operations must be safe to retry
- **Law of UTC**: All timestamps in UTC

## Decision

Create a **FastAPI wrapper** (`api_server.py`) that:

1. Provides REST endpoints for starting, monitoring, and retrieving evolutions
2. Uses background tasks for async evolution execution
3. Manages evolution state in memory (upgradeable to Redis)
4. Validates required environment variables at startup
5. Follows Federation Constitution principles

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  LoongFlow API Server                   │
│  ┌───────────────────────────────────────────────────┐ │
│  │         FastAPI Endpoints                         │ │
│  │  • POST   /api/v1/evolve                          │ │
│  │  • GET    /api/v1/status/{id}                     │ │
│  │  • GET    /api/v1/solutions/{id}                  │ │
│  │  • GET    /health                                 │ │
│  └───────────────┬───────────────────────────────────┘ │
│                  │                                       │
│  ┌───────────────▼───────────────────────────────────┐ │
│  │         Evolution State Manager                   │ │
│  │  (In-Memory Dictionary, upgradeable to Redis)     │ │
│  └───────────────┬───────────────────────────────────┘ │
│                  │                                       │
│  ┌───────────────▼───────────────────────────────────┐ │
│  │         Background Task Runner                    │ │
│  │  • Runs evolutions asynchronously                 │ │
│  │  • Updates state periodically                     │ │
│  │  • Handles errors gracefully                      │ │
│  └───────────────┬───────────────────────────────────┘ │
│                  │                                       │
└──────────────────┼───────────────────────────────────────┘
                   │
                   │ TODO: Phase 2 Integration
                   │
                   ▼
         ┌─────────────────────┐
         │  LoongFlow PES Core │
         │  (Not yet wrapped)  │
         └─────────────────────┘
```

## Implementation Details

### Phase 1: API Structure (Current)

✅ **Completed**:

1. FastAPI server with proper request/response models
2. Background task execution using FastAPI's `BackgroundTasks`
3. In-memory state management for evolutions
4. Health check endpoint
5. CRUD operations for evolutions
6. Environment variable validation at startup
7. Structured logging with correlation IDs
8. Dockerfile for containerization
9. Updated docker-compose configuration
10. Test script for validation

⚠️ **Simplified**:

1. Evolution logic is **simulated**, not actual LoongFlow PES
2. Solutions are placeholders
3. No integration with `GeneralPESAgent`

### Phase 2: Full Integration (Future)

To fully integrate with LoongFlow, we need to:

1. **Refactor `BasePESRunner`**:
   ```python
   class BasePESRunner(ABC):
       def start(self) -> None:  # CLI mode (existing)
           ...

       async def start_async(self, callbacks: ProgressCallbacks) -> EvolutionResult:
           """New async mode for API integration"""
           ...
   ```

2. **Add Progress Callbacks**:
   ```python
   @dataclass
   class ProgressCallbacks:
       on_generation: Callable[[int, float], None]  # gen, fitness
       on_complete: Callable[[Solution], None]
       on_error: Callable[[Exception], None]
   ```

3. **Extract State Management**:
   ```python
   class EvolutionStateManager:
       def get_status(self, evolution_id: str) -> EvolutionStatus
       def update_progress(self, evolution_id: str, generation: int, fitness: float)
       def store_result(self, evolution_id: str, result: EvolutionResult)
   ```

4. **Support Multiple Agent Types**:
   - General Agent (done)
   - Math Agent (needs integration)
   - ML Agent (needs integration)

## Alternatives Considered

### Alternative 1: Modify LoongFlow Core

**Pros**:
- Tightest integration
- No wrapper overhead

**Cons**:
- Violates **Law of the Air Gap** (core projects should be immutable)
- Makes upstream updates harder
- Mixing concerns (CLI + API in same codebase)

**Rejected**: Federation Constitution prohibits modifying core projects.

### Alternative 2: Separate Microservice

**Pros**:
- Complete isolation
- Can be written in any language

**Cons**:
- More complex deployment
- Network overhead
- Harder to share code

**Rejected**: Python wrapper is simpler and allows direct imports.

### Alternative 3: Use Existing LoongFlow CLI via subprocess

**Pros**:
- No refactoring needed
- Uses tested code path

**Cons**:
- Hard to track progress
- No real-time status updates
- Process management complexity

**Rejected**: Doesn't provide the user experience we want.

## Consequences

### Positive

1. ✅ Clean HTTP interface for other services
2. ✅ Follows Federation Constitution principles
3. ✅ Easy to test and debug
4. ✅ Can be extended to support streaming/WebSocket
5. ✅ Environment-based configuration (12-factor app)

### Negative

1. ⚠️ Currently provides **simulated** results, not real evolution
2. ⚠️ Phase 2 requires refactoring LoongFlow core
3. ⚠️ In-memory state is lost on restart (Redis will fix this)
4. ⚠️ No authentication/authorization yet

### Risks

1. **Integration Complexity**: Phase 2 requires deep LoongFlow knowledge
2. **Performance**: Python async may not handle high concurrency well
3. **Memory**: In-memory state doesn't scale to thousands of evolutions

## Migration Path

### Phase 1 (Current)
- Deploy API wrapper with simulated evolution
- Verify endpoints work correctly
- Get feedback from adapter developers

### Phase 2 (Future)
- Refactor `BasePESRunner` for async support
- Add progress callbacks
- Integrate real LoongFlow PES logic
- Add Redis for distributed state

### Phase 3 (Production)
- Add authentication (OIDC)
- Add rate limiting
- Add monitoring and alerting
- Multi-worker deployment

## Testing

```bash
# Manual testing
docker-compose -f docker-compose.loongflow-core.yml up -d
./docker/test_api.sh

# Automated testing (to be implemented)
pytest tests/api/test_endpoints.py
```

## References

- [CLAUDE.md](../CLAUDE.md) - Federation Constitution
- [API.md](./API.md) - API Documentation
- [docker-compose.loongflow-core.yml](../docker-compose.loongflow-core.yml) - Deployment config

## Authors

- Claude Code (Distinguished Engineer & Guardian of Stability)

## Related ADRs

- None yet (this is the first LoongFlow ADR)
