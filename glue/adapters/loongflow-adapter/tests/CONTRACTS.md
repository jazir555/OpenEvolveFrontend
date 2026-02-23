# LoongFlow API Contracts

## Overview

This document defines the API contracts between the LoongFlow adapter and the LoongFlow core system. These contracts serve as a defense mechanism against breaking changes from API updates.

**Purpose:** Phase 2 - The Contract (Defense)

**Law of Runtime Truth:** These contracts are validated against the actual LoongFlow API, not documentation.

## Contract Testing Strategy

### Test Levels

1. **Fixture Tests (Offline):** Validate test fixtures against canonical schemas
2. **Contract Tests (Online):** Validate actual API responses against Zod schemas
3. **Integration Tests (End-to-End):** Validate full workflows

### When Tests Run

- **Adapter Startup:** All contract tests must pass before adapter initializes
- **Docker Health Check:** Periodic validation of API availability
- **CI/CD Pipeline:** Block deployments if contracts are violated
- **On-Demand:** Manual validation via `npm run test:contract`

### Contract Violation Handling

If any contract test fails:

1. **Log detailed error** with expected vs actual values
2. **Refuse to start adapter** (fail-fast principle)
3. **Alert operations team** via structured logging
4. **Block deployments** until resolved
5. **Document the breaking change** in an ADR

---

## Tested Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify service availability and basic functionality

**Expected Response:**
```typescript
{
  status: 'healthy' | 'unhealthy' | 'ok' | 'error',
  version?: string,
  timestamp?: string  // UTC ISO-8601, ends with 'Z'
}
```

**Contract:**
- `status` must be one of the allowed values
- `timestamp` must be valid UTC ISO-8601 format
- HTTP status code must be 200

**Why This Matters:**
- Validates the service is running
- Confirms timestamp format compliance (Law of UTC)
- Provides early warning of service issues

---

### 2. Problem Submission

**Endpoint:** `POST /pes/submit`

**Purpose:** Submit a problem for evolutionary optimization

**Request Body:**
```typescript
{
  task: string,                  // Required: Problem description
  max_iterations?: number,       // Optional: Max iterations (positive integer)
  target_score?: number,         // Optional: Target score (0-1)
  concurrency?: number,          // Optional: Parallel workers (positive integer)
  initial_code?: string,         // Optional: Starting solution
  initial_score?: number,        // Optional: Initial score (0-1)
  initial_evaluation?: string,   // Optional: Initial evaluation
  workspace_path?: string,       // Optional: Workspace directory
  metadata?: Record<string, any> // Optional: Additional metadata
}
```

**Expected Response:**
```typescript
{
  agent_id: string,    // UUID v4 format
  status: string,      // Status of the agent
  message: string,     // Human-readable message
  timestamp?: string   // UTC ISO-8601
}
```

**Contract:**
- `task` is required and cannot be empty
- `max_iterations` must be positive integer if provided
- `target_score` must be between 0 and 1 if provided
- `concurrency` must be positive integer if provided
- `agent_id` must be valid UUID v4
- HTTP status codes: 200 or 201 on success, 400/422 on validation error

**Why This Matters:**
- Core operation for submitting problems
- Validates input sanitization
- Ensures proper error handling

---

### 3. Agent State Query

**Endpoint:** `GET /pes/agents/{agent_id}/state`

**Purpose:** Query the current state of a running PES agent

**Expected Response:**
```typescript
{
  agent_id: string,              // UUID v4
  status: 'idle' | 'running' | 'interrupted' | 'completed' | 'failed',
  current_iteration: number,     // Non-negative integer
  max_iterations: number,        // Positive integer
  target_score: number,          // 0-1
  best_score: number,            // 0-1
  start_time: string,            // UTC ISO-8601
  end_time?: string,             // UTC ISO-8601 (if completed/failed)
  completion_count: number,      // Non-negative integer
  total_prompt_tokens: number,   // Non-negative integer
  total_completion_tokens: number, // Non-negative integer
  total_cost: number             // Non-negative
}
```

**Contract:**
- `status` must be one of the allowed values
- `current_iteration` must be between 0 and `max_iterations`
- All scores must be between 0 and 1
- All timestamps must be UTC ISO-8601 format
- Token counts and costs must be non-negative

**Why This Matters:**
- Tracks progress of evolutionary optimization
- Validates resource usage metrics
- Ensures proper state transitions

---

### 4. Agent Interruption

**Endpoint:** `POST /pes/agents/{agent_id}/interrupt`

**Purpose:** Interrupt a running PES agent

**Expected Response:**
```typescript
{
  message: string  // Human-readable message
}
```

**Contract:**
- Should be idempotent (interrupting stopped agent is no-op)
- HTTP status code 200 on success

**Why This Matters:**
- Allows graceful shutdown
- Validates idempotency requirements (Law of Idempotency)

---

### 5. Execution Result

**Endpoint:** `GET /pes/agents/{agent_id}/result`

**Purpose:** Retrieve final execution result of a PES agent

**Expected Response:**
```typescript
{
  agent_id: string,              // UUID v4
  status: string,                // Final status
  final_solution?: string,       // Best solution found
  final_score?: number,          // 0-1
  best_solutions?: Solution[],   // Array of best solutions
  total_iterations: number,      // Non-negative integer
  total_tokens: number,          // Non-negative integer
  total_cost: number,            // Non-negative
  was_interrupted: boolean,
  start_time: string,            // UTC ISO-8601
  end_time: string,              // UTC ISO-8601
  error?: string                 // Error message if failed
}
```

**Contract:**
- `final_score` must be between 0 and 1 if provided
- `best_solutions` must be array of valid Solution objects
- Timestamps must be UTC ISO-8601 format
- All numeric metrics must be non-negative

**Why This Matters:**
- Retrieves optimization results
- Validates solution quality metrics

---

### 6. Solution Data Structure

**Core Solution Object:**
```typescript
{
  solution: string,              // Generated solution (code, text, etc.)
  solution_id: string,           // UUID v4
  generate_plan: string,         // Planning rationale
  score: number,                 // 0-1
  evaluation: string,            // Evaluation result
  summary: string,               // Reflection summary
  parent_id: string,             // UUID v4 (empty for initial population)
  island_id: number,             // Non-negative integer
  iteration: number,             // Non-negative integer
  metadata?: Record<string, any>,
  created_at?: string,           // UTC ISO-8601
  fitness_map_key?: Record<string, any>  // MAP-Elites position
}
```

**Contract:**
- All required fields must be present
- `score` must be between 0 and 1
- `island_id` and `iteration` must be non-negative integers
- `solution_id` and `parent_id` must be valid UUIDs
- `parent_id` can be empty string for initial population
- All timestamps must be UTC ISO-8601 format

**Why This Matters:**
- Core data structure for evolutionary optimization
- Validates complete Solution dataclass fields
- Ensures proper type safety

---

### 7. Database Status

**Endpoint:** `GET /database/status`

**Purpose:** Query evolutionary database status

**Query Parameters:**
- `island_id?: number` - Filter by specific island

**Expected Response:**
```typescript
{
  global_status: {
    current_iteration: number,   // Non-negative integer
    best_score: number,          // 0-1
    total_solutions: number      // Non-negative integer
  },
  island_status?: {
    [island_id: number]: {
      best_score: number,        // 0-1
      total_solutions: number    // Non-negative integer
    }
  }
}
```

**Contract:**
- All scores must be between 0 and 1
- All counts must be non-negative integers
- `island_status` keys must be non-negative integers

**Why This Matters:**
- Tracks overall optimization progress
- Validates island model functionality

---

### 8. Best Solutions Query

**Endpoint:** `GET /database/best`

**Purpose:** Retrieve best solutions from database

**Query Parameters:**
- `island_id?: number` - Filter by specific island
- `top_k?: number` - Limit number of results

**Expected Response:**
```typescript
Solution[]  // Array of valid Solution objects
```

**Contract:**
- Returns array of valid Solution objects
- Results must be sorted by score (descending)
- If `top_k` specified, must return at most `top_k` results
- If `island_id` specified, must filter to that island only

**Why This Matters:**
- Retrieves top solutions
- Validates sorting and filtering logic

---

### 9. Checkpoint Operations

**Endpoint:** `POST /database/checkpoints`

**Purpose:** Save a checkpoint of evolutionary state

**Request Body:**
```typescript
{
  checkpoint_path: string,
  tag: string
}
```

**Expected Response:**
```typescript
{
  message: string,
  checkpoint: {
    checkpoint_path: string,
    tag: string,
    created_at: string,      // UTC ISO-8601
    iteration: number,       // Non-negative integer
    completion_count: number // Non-negative integer
  }
}
```

**Endpoint:** `GET /database/checkpoints`

**Purpose:** List available checkpoints

**Query Parameters:**
- `checkpoint_path: string` - Checkpoint directory path

**Expected Response:**
```typescript
Array<{
  checkpoint_path: string,
  tag: string,
  created_at: string,      // UTC ISO-8601
  iteration: number,
  completion_count: number
}>
```

**Contract:**
- All timestamps must be UTC ISO-8601 format
- `iteration` and `completion_count` must be non-negative
- Checkpoints should be in chronological order

**Why This Matters:**
- Supports state persistence
- Enables recovery from failures

---

### 10. Error Responses

**All error responses follow this format:**
```typescript
{
  detail: string,          // Human-readable error message
  error_code?: string,     // Machine-readable error code
  timestamp?: string       // UTC ISO-8601
}
```

**HTTP Status Codes:**
- `400` - Bad Request (invalid input)
- `404` - Not Found (agent/solution doesn't exist)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error
- `503` - Service Unavailable
- `504` - Gateway Timeout

**Contract:**
- `detail` must always be present
- `error_code` should be present for known errors
- `timestamp` should be UTC ISO-8601 if present

**Why This Matters:**
- Consistent error handling
- Enables proper error recovery

---

## Canonical Schema Compliance

All contracts must comply with the canonical schemas defined in:
`glue/schemas/loongflow-canonical.ts`

### Key Canonical Types

1. **LoongFlowSolution** - Complete solution object with all evolutionary metadata
2. **LoongFlowState** - Agent state enum (idle, planning, executing, evolving, etc.)
3. **LoongFlowConfig** - Complete configuration for PES execution
4. **LoongFlowRequest** - Request to execute LoongFlow
5. **LoongFlowResponse** - Response from LoongFlow execution

---

## Federation Constitution Compliance

### Law of Runtime Truth
- Tests execute against real LoongFlow API
- No mocking in critical contract tests
- Probe scripts verify API behavior before writing tests

### Law of Configuration Explicitness
- All required environment variables are validated
- No magic defaults - service crashes if `LOONGFLOW_API_URL` is missing
- Timeout variables must be positive integers

### Law of UTC
- All timestamps validated as UTC ISO-8601 format
- Timestamps must end with 'Z' indicator
- No timezone conversions in glue layer

### Law of Idempotency
- Document which operations are idempotent
- Validate structures support idempotent operations
- Empty `parent_id` allowed for initial population

### Law of Air Gap
- Adapter does not import from `core-projects/LoongFlow`
- All data transformed to/from canonical schemas
- Anti-corruption layer enforced

---

## Updating Contracts

When LoongFlow core API changes:

1. **Update the contract schema** in this document
2. **Update the Zod validation** in `contract.test.ts`
3. **Update test fixtures** in `fixtures/test-data.ts`
4. **Run contract tests** to verify new contracts
5. **Update adapter code** if breaking changes
6. **Create ADR** documenting the change
7. **Get approval** before deploying

### Example: Adding New Field

```typescript
// 1. Update contract schema
const PESAgentStateContract = z.object({
  // ... existing fields
  new_field: z.string().optional(),  // Add new field
});

// 2. Update fixture
export const RUNNING_AGENT_STATE = {
  // ... existing fields
  new_field: 'new_value',  // Add to fixture
};

// 3. Update documentation
// Add field to this CONTRACTS.md document
```

---

## Running Contract Tests

### Locally

```bash
# Set environment variables
export LOONGFLOW_API_URL=http://localhost:8000
export LOONGFLOW_TIMEOUT_MS=30000

# Run all contract tests
npm run test:contract

# Run with verbose output
VERBOSE=true npm run test:contract

# Run as standalone script
ts-node tests/contract-runner.ts

# Skip integration tests (fixture tests only)
SKIP_CONTRACT_TESTS=true npm run test:contract
```

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD npm run test:contract || exit 1
```

### CI/CD Pipeline

```yaml
# Example GitHub Actions
- name: Run Contract Tests
  env:
    LOONGFLOW_API_URL: http://loongflow-service:8000
    LOONGFLOW_TIMEOUT_MS: 30000
  run: |
    npm run test:contract
    if [ $? -ne 0 ]; then
      echo "Contract tests failed - blocking deployment"
      exit 1
    fi
```

---

## Contract Violation Response

If contract tests fail:

### 1. Immediate Actions
- **Stop the adapter** - Do not start with violated contracts
- **Log the violation** with full details
- **Alert the team** via monitoring system

### 2. Investigation
- **Check if API is down** - Run health check manually
- **Review API changelog** - Did LoongFlow core change?
- **Check environment** - Are env vars set correctly?

### 3. Resolution Options

**Option A: Fix Adapter (Preferred)**
- Update adapter to match new API
- Update contracts to reflect new reality
- Add integration tests for new behavior

**Option B: Pin Core Version**
- Lock to specific LoongFlow version
- Document version constraint
- Plan migration to new version

**Option C: Rollback Core**
- Revert LoongFlow to previous version
- Fix breaking change in core
- Re-release with backwards compatibility

### 4. Prevention
- **Subscribe to LoongFlow changelog**
- **Run contract tests in CI/CD**
- **Monitor production API calls**
- **Version lock core dependencies**

---

## Contract Test Metrics

### Success Criteria
- **All contracts pass** before adapter starts
- **Tests complete in under 30 seconds**
- **Zero false positives** (tests pass when they shouldn't)
- **Zero false negatives** (tests fail when they shouldn't)

### Monitoring
- **Contract test pass rate** - Should be 100%
- **Test execution time** - Alert if > 30 seconds
- **Contract violation frequency** - Should be zero
- **API availability** - Uptime percentage

---

## Additional Resources

- **Adapter Code:** `glue/adapters/loongflow-adapter/src/adapter.ts`
- **Canonical Schemas:** `glue/schemas/loongflow-canonical.ts`
- **Test Fixtures:** `glue/adapters/loongflow-adapter/tests/fixtures/test-data.ts`
- **Contract Tests:** `glue/adapters/loongflow-adapter/tests/contract.test.ts`
- **Test Runner:** `glue/adapters/loongflow-adapter/tests/contract-runner.ts`

---

**Last Updated:** 2026-02-22

**Maintained By:** OpenEvolve Federation

**Status:** Active
