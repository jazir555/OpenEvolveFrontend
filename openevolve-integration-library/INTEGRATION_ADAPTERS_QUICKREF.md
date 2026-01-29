# Integration Adapters - Quick Reference

## Project Structure (v1.1.0)

```
src/
├── api/
│   ├── backend.ts          # Backend communication (Axios/Socket.io)
│   ├── client.ts           # Unified OpenEvolveClient
│   ├── errors.ts           # Custom Error classes
│   ├── middleware.ts       # Built-in middlewares (Logging, Caching)
│   └── types.ts            # ALL core API and validation types
├── integrations/
│   ├── base.ts             # Base adapter class
│   ├── all-integrations.ts # All 9 integration implementations
│   └── index.ts            # Main integration entry point
├── react/
│   └── index.ts            # Provider and hooks (useDecomposition, etc.)
├── store/
│   └── index.ts            # Zustand store factory
├── testing/
│   └── index.ts            # createMockClient utility
└── index.ts                # Main library entry point
```

## Integration Adapters Summary

### 1. **BaseIntegrationAdapter** (`base.ts`)
**Purpose**: Foundation for all adapters

**Key Features**:
- ✅ Automatic request/response transformation
- ✅ Reusable validation helpers (`validateRequired`, `validateEnum`)
- ✅ WebSocket streaming routing
- ✅ Standardized error mapping

### 2. **Supported Integrations**
| Integration | Operations |
|-------------|------------|
| **LeanAide** | `translate`, `prove`, `verify`, `mcts`, `query` |
| **Evolution** | `evolution`, `adversarial`, `coevolution` |
| **Knowledge** | `query`, `extract`, `search`, `stats` |
| **Maker** | `create`, `execute`, `validate`, `list` |
| **Hephaestus** | `delegate`, `status`, `create`, `list` |
| **Decomposition** | `decompose`, `subproblems`, `dependencies` |
| **Verification** | `verify`, `checks`, `validate` |
| **Assembly** | `assemble`, `integrate`, `optimize` |
| **Solution** | `generate`, `optimize`, `refine` |

---

## Common Interface

```typescript
// Constructor
constructor(client: BackendClient)

// Generic execution
async execute<TInputs, TResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>

// Real-time updates
async executeStream<TInputs, TResult>(inputs, onProgress, options?): Promise<TResult>

// Schema retrieval
getSchema(): ParameterSchema

// Health monitoring
async healthCheck(): Promise<IntegrationHealth>
```

## Error Hierarchy

```typescript
IntegrationError (Base)
├── ConnectionError    # Network/Connectivity
├── ValidationError    # Input structure
├── ExecutionError     # Server-side failure
├── TimeoutError       # Request expired
└── AuthenticationError # API Key issues
```

## Quick Start Pattern

### 1. Client Setup
```typescript
import { OpenEvolveClient } from '@openevolve/integration-library';
const client = new OpenEvolveClient({ baseUrl: '...', apiKey: '...' });
```

### 2. Direct Execution
```typescript
const result = await client.integrations.leanaide.execute({
  operation: 'prove',
  input: { theorem: '...' }
});
```

### 3. Middleware Hook
```typescript
import { loggingMiddleware } from '@openevolve/integration-library';
const client = new OpenEvolveClient({
  baseUrl: '...',
  middleware: [loggingMiddleware]
});
```