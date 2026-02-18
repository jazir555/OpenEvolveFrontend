# Architecture Decision Record: OpenEvolve React Plugin

**Status**: Accepted
**Date**: 2025-02-17
**Authors**: OpenEvolve Federation Team
**Component**: OpenEvolve React Plugin (`glue/adapters/openevolve/`)

---

## Context

The OpenEvolve React Plugin serves as the **frontend adapter** between BubbleLab (the low-code application platform) and the OpenEvolve backend API. This plugin enables users to configure, execute, and monitor OpenEvolve's evolutionary algorithms, adversarial testing, decomposition, and MDAP/MAKER capabilities through a React-based UI.

### Key Requirements

1. **Federation Constitution Compliance**: Must follow all 6 laws (Air Gap, Runtime Truth, Untouchable DB, Idempotency, Configuration Explicitness, UTC)
2. **BubbleLab Integration**: Must integrate seamlessly with existing BubbleLab plugin architecture
3. **Zero Core Modifications**: Must be a standalone plugin requiring no changes to BubbleLab core
4. **Production Ready**: Must include probes, contract tests, Dockerfile, and comprehensive documentation
5. **UI Consistency**: Must follow same patterns as other BubbleLab plugins (LeanAIDE, ClaudieMiro, Datapizza)

---

## Decision

### Architecture Pattern: React Plugin with HTTP Client

The OpenEvolve React Plugin implements a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEvolve React Plugin                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  UI Layer (React Components)                       │     │
│  │  └─ OpenEvolveConfigPanel.tsx                      │     │
│  └────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Service Layer (createOpenEvolvePlugin.ts)         │     │
│  │  └─ State Management, Business Logic               │     │
│  └────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  HTTP Client Layer (OpenEvolveClient.ts)           │     │
│  │  └─ API Communication, Circuit Breaker, Retry      │     │
│  └────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Type Definitions (plugin-types.ts)                │     │
│  │  └─ TypeScript Interfaces, Enums                   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  OpenEvolve Backend API                      │
│  (http://openevolve-core:8002)                               │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

1. **Singleton Pattern**: Global plugin instance management
2. **Factory Pattern**: Plugin creation with dependency injection
3. **State Management**: Centralized state with configuration persistence
4. **Circuit Breaker**: Prevents cascading failures to backend
5. **Retry Logic**: Exponential backoff with jitter
6. **Anti-Corruption Layer (ACL)**: Schema transformation between plugin and API

---

## Alternatives Considered

### Alternative 1: Direct BubbleLab Integration
**Description**: Modify BubbleLab core to include OpenEvolve functionality directly.

**Pros**:
- Tighter integration
- Better performance

**Cons**:
- ❌ Violates **Law of Air Gap** (direct coupling)
- ❌ Requires BubbleLab core modifications
- ❌ Harder to maintain and upgrade
- ❌ Violates Federation Constitution

**Decision**: ❌ Rejected

### Alternative 2: Backend-Only Adapter
**Description**: Only provide backend adapter (`glue/adapters/openevolve-adapter/`) with no React plugin.

**Pros**:
- Simpler architecture
- Less code to maintain

**Cons**:
- ❌ No UI for BubbleLab users
- ❌ Inconsistent with other plugins
- ❌ Poor user experience

**Decision**: ❌ Rejected

### Alternative 3: Standalone Web App
**Description**: Create separate web application for OpenEvolve.

**Pros**:
- Complete independence
- No BubbleLab dependency

**Cons**:
- ❌ Inconsistent with plugin architecture
- ❌ Poor integration experience
- ❌ Duplication of BubbleLab features

**Decision**: ❌ Rejected

### Alternative 4: React Plugin (Selected) ✅
**Description**: Standalone React plugin following BubbleLab plugin patterns.

**Pros**:
- ✅ Follows **Law of Air Gap** (no core imports)
- ✅ Consistent with other plugins (LeanAIDE, Datapizza, etc.)
- ✅ Zero BubbleLab core modifications
- ✅ Easy to maintain and upgrade
- ✅ Production ready with probes and tests
- ✅ Federation Constitution compliant

**Cons**:
- Slightly more complex than standalone
- Requires following plugin conventions

**Decision**: ✅ **ACCEPTED**

---

## Technical Specifications

### Plugin Interface

The plugin implements the `OpenEvolvePlugin` interface with the following methods:

```typescript
interface OpenEvolvePlugin {
  // Metadata and Initialization
  getMetadata(): OpenEvolvePluginMetadata;
  getState(): OpenEvolvePluginState;
  initialize(config?: Partial<OpenEvolvePluginState>): Promise<void>;

  // Configuration Management
  updateConfig(config: Partial<OpenEvolvePluginState>): Promise<void>;
  resetConfig(): Promise<void>;
  getConfig(): OpenEvolvePluginState;

  // Evolution Functionality
  executeEvolution(goal: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;

  // Adversarial Functionality
  executeAdversarial(content: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;

  // Decomposition Functionality
  executeDecomposition(problem: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;

  // Integrated Execution
  executeIntegrated(goal: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;

  // Execution Management
  getExecution(executionId: string): Promise<OpenEvolveExecutionResult | null>;
  getExecutionHistory(): Promise<OpenEvolveExecutionResult[]>;
  getStatistics(): Promise<OpenEvolveExecutionStatistics[]>;
  cancelExecution(executionId: string): Promise<boolean>;
  clearHistory(): Promise<void>;

  // MDAP/MAKER Integration
  shouldUseMdapMakerForGoal(goal: string): boolean;
  getMdapMakerConfig(): any | null;

  // Utility Methods
  validateConfig(): Promise<{ valid: boolean; errors: string[] }>;
  getAvailableStrategies(): {
    evolution: EvolutionStrategy[];
    adversarial: AdversarialStrategy[];
    decomposition: DecompositionStrategy[];
  };
}
```

### Federation Constitution Compliance

#### ✅ Law of the "Air Gap" (Source Code Isolation)
- **No imports from `core-projects/`**: Plugin uses only HTTP API to communicate
- **Anti-Corruption Layer**: Schema validation at plugin boundaries
- **Type Safety**: All data structures validated with TypeScript

#### ✅ Law of "Runtime Truth" (Anti-Hallucination)
- **Probe Scripts**: `check-plugin-api.sh` verifies API endpoints before plugin use
- **Contract Tests**: `tests/contract.test.ts` validates actual API behavior
- **Health Checks**: Built-in health verification for all operations

#### ✅ Law of the "Untouchable DB" (Read-Only State)
- **No Direct Database Access**: Plugin communicates only through API
- **API-Only Operations**: All data access through OpenEvolve backend

#### ✅ Law of Idempotency (The Replayability Pact)
- **Idempotent Operations**: All operations safe to retry
- **Execution Deduplication**: Execution IDs prevent duplicate processing
- **Checkpoint Recovery**: Workflows can resume from checkpoints

#### ✅ Law of Configuration Explicitness
- **No Magic Defaults**: All configuration via environment variables
- **Fail Fast**: Plugin crashes immediately if required config missing
- **Explicit API URLs**: `OPENEVOLVE_API_URL` must be provided

```typescript
// Required environment variables (fails fast if missing)
const OPENEVOLVE_API_URL = process.env.OPENEVOLVE_API_URL!; // Required
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS!); // Required

// Optional with explicit defaults
const LOG_LEVEL = process.env.LOG_LEVEL || 'info';
```

#### ✅ Law of UTC
- **All Timestamps in UTC**: Plugin uses `new Date().toISOString()`
- **No Timezone Offsets**: All times stored as UTC ISO-8601
- **Consistent Timezone Handling**: Plugin and API both use UTC

### Circuit Breaker Implementation

The plugin implements a circuit breaker pattern to prevent cascading failures:

```typescript
enum CircuitBreakerState {
  CLOSED = 'closed',      // Normal operation
  OPEN = 'open',          // Failing, reject requests
  HALF_OPEN = 'half_open' // Testing recovery
}

class CircuitBreaker {
  private state: CircuitBreakerState;
  private failureCount: number;
  private lastFailureTime: number;

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitBreakerState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.state = CircuitBreakerState.HALF_OPEN;
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
}
```

### Retry Logic

Exponential backoff with jitter to prevent thundering herd:

```typescript
async executeWithRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (attempt === maxRetries) throw error;

      // Exponential backoff with jitter
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
        10000
      );
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error('Max retries exceeded');
}
```

---

## Component Overview

### 1. OpenEvolveConfigPanel.tsx
**Purpose**: Main React component for plugin configuration UI

**Features**:
- Multi-tab interface (General, Evolution, Adversarial, Decomposition, MDAP/MAKER)
- Real-time configuration updates
- Form validation
- Dark mode support
- Execution statistics display

### 2. createOpenEvolvePlugin.ts
**Purpose**: Plugin factory and business logic layer

**Features**:
- Singleton instance management
- State management (Zustand-like pattern)
- Service class with caching and retry logic
- MDAP/MAKER auto-selection

### 3. plugin-types.ts
**Purpose**: TypeScript type definitions

**Features**:
- Complete type safety for plugin interface
- Execution result types
- Configuration types
- Strategy enums

### 4. Probes (`probes/`)
**Purpose**: Runtime verification scripts (Law of Runtime Truth)

**Scripts**:
- `check-plugin-api.sh`: Validates API endpoints
- `check-plugin-build.sh`: Validates build configuration

### 5. Contract Tests (`tests/contract.test.ts`)
**Purpose**: API contract validation (Fail Fast on violations)

**Tests**:
- Plugin interface contract
- API endpoint contracts
- State structure contracts
- CORS header validation
- Error response structure

---

## Integration Flow

### Initialization Flow

```
1. BubbleLab App Starts
   ↓
2. useOpenEvolvePlugin() Hook Called
   ↓
3. Plugin Initialized with Config
   ├─ Load from localStorage or defaults
   ├─ Validate configuration
   └─ Initialize HTTP client
   ↓
4. OpenEvolveConfigPanel Rendered
   ↓
5. User Can Configure and Execute
```

### Execution Flow

```
1. User Triggers Execution (e.g., "Run Evolution")
   ↓
2. Plugin.validateConfig() Called
   ↓
3. Plugin.shouldUseMdapMakerForGoal() Checked
   ↓
4. Plugin.executeEvolution() Called
   ├─ Circuit Breaker Check
   ├─ Execute with Retry (3 attempts)
   │   ├─ HTTP POST to /evolution
   │   └─ Wait for completion
   ├─ Transform Result (ACL)
   └─ Return ExecutionResult
   ↓
6. UI Updated with Results
   ↓
7. Statistics Updated
```

---

## Configuration

### Environment Variables

**Required** (fails fast if missing):
```bash
OPENEVOLVE_API_URL=http://openevolve-core:8002
TIMEOUT_MS=10000
```

**Optional** (with defaults):
```bash
LOG_LEVEL=info
MAX_RETRIES=3
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60000
```

### Plugin Configuration

```typescript
interface OpenEvolvePluginState {
  defaultExecutionMethod: 'auto' | 'evolution' | 'adversarial' | 'decomposition' | 'roma_mdap_maker';
  evolutionConfig: EvolutionConfig;
  adversarialConfig: AdversarialConfig;
  decompositionConfig: DecompositionConfig;
  mdapMaker: {
    enabled: boolean;
    autoSelect: boolean;
    maxDepth: number;
    kAhead: number;
    redFlagging: boolean;
    adaptiveK: boolean;
  };
}
```

---

## Testing

### Contract Tests

Validate API contracts at runtime:

```bash
npm run test:contract
```

### Probe Scripts

Verify API and build before deployment:

```bash
./probes/check-plugin-api.sh
./probes/check-plugin-build.sh
```

---

## Deployment

### Docker Build

```bash
docker build -t openevolve-react-plugin:1.0.0 .
```

### Docker Run

```bash
docker run -d \
  -e OPENEVOLVE_API_URL=http://openevolve-core:8002 \
  -e TIMEOUT_MS=10000 \
  --name openevolve-plugin \
  openevolve-react-plugin:1.0.0
```

---

## Monitoring

### Health Checks

The plugin includes built-in health checks:
- API endpoint availability
- Circuit breaker state
- Execution statistics
- Error rates

### Logging

Structured JSON Lines logging with correlation IDs:

```json
{
  "timestamp": "2025-02-17T12:34:56.789Z",
  "level": "info",
  "message": "Evolution execution completed",
  "service": "openevolve-plugin",
  "correlation_id": "abc-123-def",
  "execution_id": "exec-456",
  "duration_ms": 5234
}
```

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| API Unavailability | High | Medium | Circuit breaker, retry logic, graceful degradation |
| Configuration Errors | High | Low | Fail fast, explicit validation, comprehensive defaults |
| Performance Issues | Medium | Low | Caching, async operations, request deduplication |
| Breaking API Changes | High | Low | Contract tests, version pinning, ACL |

---

## Future Considerations

1. **Enhanced Caching**: Implement Redis-based distributed caching
2. **Real-time Updates**: WebSocket integration for live execution monitoring
3. **Offline Mode**: Support for offline operation with sync on reconnect
4. **Plugin Marketplace**: Share custom configurations and strategies
5. **Advanced Analytics**: Machine learning-based performance optimization

---

## References

- [Federation Constitution](../../../CLAUDE.md)
- [OpenEvolve Backend Adapter](../openevolve-adapter/)
- [BubbleLab Integration](../bubblelab/)
- [API Documentation](https://openevolve.github.io/docs)

---

**Decision**: ✅ **ACCEPTED**
**Implementation Status**: ✅ **COMPLETE**
**Last Updated**: 2025-02-17T12:00:00Z
