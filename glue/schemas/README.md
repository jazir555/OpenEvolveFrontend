# Canonical Schemas - Anti-Corruption Layer

This directory contains the canonical data models for the OpenEvolve Federation's glue layer. All adapters must normalize their data to/from these schemas.

## Purpose

The **Anti-Corruption Layer (ACL)** prevents data format inconsistencies from propagating through the system. Each core project (Z3, LeanAide, etc.) uses different data formats, but all glue code must use these canonical schemas.

## Law of the Air Gap

**CRITICAL**: Do not import or reference files in `./core-projects/`. If you need a utility function from a core project, rewrite it in the glue layer. Dependency leakage is fatal.

## Installation

These schemas require **Zod** for runtime validation:

```bash
npm install zod
npm install -D ts-node  # For running validation examples
```

## Schema Files

### 1. `z3-canonical.ts`
Defines canonical schemas for Z3 SMT solver interactions:
- `SolverRequest` - Request to solve a constraint problem
- `SolverResponse` - Response from Z3 after solving
- `KnowledgeGraphResponse` - Knowledge graph extracted from mathematical content
- `Entity` and `Relation` - Building blocks for knowledge graphs

### 2. `leanaide-canonical.ts`
Defines canonical schemas for LeanAide (Lean 4) interactions:
- `ProofVerificationRequest` - Request to verify a Lean 4 proof
- `ProofVerificationResponse` - Response from proof verification
- `LeanCompilationRequest` - Request to compile Lean 4 code
- `LeanCompilationResponse` - Response from code compilation
- `LeanMessage` - Compiler messages (errors, warnings, hints)

### 3. `index.ts`
Central export point for all schemas. **Always import from here**, not from individual schema files.

### 4. `validation-examples.ts`
Demonstration and testing of schema validation. Run with:

```bash
ts-node glue/schemas/validation-examples.ts
```

## Usage Examples

### Validating a Z3 Request

```typescript
import { validateSolverRequest } from './glue/schemas';

const requestData = {
  problem: "(declare-const x Int) (assert (> x 10))",
  timeout_ms: 5000,
};

const validation = validateSolverRequest(requestData);
if (!validation.success) {
  console.error('Invalid request:', validation.errors);
  return;
}

// Use validation.data safely - TypeScript knows it's valid
const request = validation.data;
```

### Creating a LeanAide Request

```typescript
import {
  ProofVerificationRequest,
  createCorrelationId,
  createUTCTimestamp,
  DEFAULT_TIMEOUTS
} from './glue/schemas';

const request: ProofVerificationRequest = {
  proof_code: 'theorem example : Nat := by trivial',
  theorem: '∃ n: Nat, n = n',
  timeout_ms: DEFAULT_TIMEOUTS.NORMAL,
  correlation_id: createCorrelationId(),
};
```

### Transforming External Format to Canonical

```typescript
import { transformZ3ResponseToCanonical } from './glue/schemas';

const rawZ3Response = {
  result: 'SAT',
  time: 45,
  version: '4.12.1',
};

const canonical = transformZ3ResponseToCanonical(
  rawZ3Response,
  '550e8400-e29b-41d4-a716-446655440000'
);

// canonical now conforms to SolverResponse schema
console.log(canonical.result); // 'sat'
```

## Schema Structure

All canonical schemas follow this structure:

```typescript
{
  // Core data fields (required)
  field1: type,
  field2: type,

  // Optional metadata for observability
  metadata?: {
    execution_time_ms?: number,
    memory_used_mb?: number,
    version?: string,
  },

  // Correlation ID for distributed tracing
  correlation_id?: string (UUID),

  // UTC timestamp (Law of UTC)
  timestamp: string (ISO-8601),
}
```

## Validation Rules

### Required Fields
All required fields must be present and non-empty:
- Z3: `problem`, `timeout_ms`
- LeanAide: `proof_code`, `theorem`, `timeout_ms`

### Timeout Validation
- Must be a positive integer
- Maximum 5 minutes (300,000ms)
- Use `DEFAULT_TIMEOUTS` constants for recommended values

### Timestamp Format
All timestamps must be:
- UTC timezone
- ISO-8601 format
- Created using `createUTCTimestamp()` utility

### Correlation IDs
- Must be valid UUID v4
- Use `createCorrelationId()` utility to generate
- Pass through all requests/responses for distributed tracing

## Constants

### Default Timeouts
```typescript
DEFAULT_TIMEOUTS.QUICK      // 5 seconds
DEFAULT_TIMEOUTS.NORMAL    // 15 seconds
DEFAULT_TIMEOUTS.LONG      // 1 minute
DEFAULT_TIMEOUTS.EXTENDED  // 5 minutes (maximum)
```

### Maximum Sizes
```typescript
MAX_SIZES.PROBLEM_LENGTH      // 100KB
MAX_SIZES.PROOF_CODE_LENGTH   // 500KB
MAX_SIZES.IMPORTS_COUNT       // 100
MAX_SIZES.TACTICS_COUNT       // 1000
```

## Error Handling

All validation functions return:

```typescript
{
  success: boolean,
  data?: T,           // Present if success is true
  errors?: string[]   // Present if success is false
}
```

### Example Error Response

```typescript
{
  success: false,
  errors: [
    "timeout_ms: Number must be greater than 0",
    "problem: String must contain at least 1 character(s)"
  ]
}
```

## Testing

Run validation tests to verify schemas are working:

```bash
# Using ts-node
ts-node glue/schemas/validation-examples.ts

# Or add to package.json scripts
npm run validate-schemas
```

Expected output:
```
✓ Valid SolverRequest passed validation
✓ Valid SolverResponse passed validation
✓ Valid KnowledgeGraphResponse passed validation
✓ Invalid SolverRequest correctly rejected
✓ Invalid SolverResponse correctly rejected
```

## Design Principles

### 1. Zero Trust
Validate all data at the boundary. Never trust external formats.

### 2. Fail Fast
If required fields are missing, crash immediately with a clear error message.

### 3. Law of UTC
All timestamps are UTC. Convert on ingestion, process in UTC.

### 4. Configuration Explicitness
No magic defaults. All configurable values must be explicit.

### 5. Idempotency
All transformations must be safe to run multiple times.

## Adapter Integration

When creating a new adapter:

1. **Import schemas from index.ts**
   ```typescript
   import { SolverRequest, validateSolverRequest } from '../schemas';
   ```

2. **Validate incoming data**
   ```typescript
   const validation = validateSolverRequest(externalData);
   if (!validation.success) {
     throw new Error(`Invalid request: ${validation.errors.join(', ')}`);
   }
   ```

3. **Transform to canonical format**
   ```typescript
   const canonical = transformToCanonical(externalData);
   ```

4. **Validate outgoing data**
   ```typescript
   const validation = validateSolverResponse(canonical);
   if (!validation.success) {
     throw new Error(`Invalid response: ${validation.errors.join(', ')}`);
   }
   ```

5. **Pass canonical data to event bus**
   Never pass raw external data between services.

## Contract Testing

Each adapter should have contract tests that verify:
1. The external API returns the expected fields
2. Transformations produce valid canonical data
3. All validation rules are enforced

Example contract test:

```typescript
import { validateSolverResponse } from '../schemas';

test('Z3 adapter produces valid canonical response', async () => {
  const response = await z3Adapter.solve(problem);

  const validation = validateSolverResponse(response);
  expect(validation.success).toBe(true);
  expect(validation.data?.result).toMatch(/sat|unsat|unknown/);
});
```

## Versioning

Current schema version: **1.0.0**

When updating schemas:
1. Increment version in `SchemaRegistry`
2. Update all adapters to use new version
3. Maintain backward compatibility if possible
4. Document breaking changes in CHANGELOG

## Troubleshooting

### Issue: "Cannot find module 'zod'"
**Solution**: Install dependencies: `npm install zod`

### Issue: Validation fails for valid data
**Solution**: Check that all required fields are present and types match exactly. Use the examples in the schema files as reference.

### Issue: Timestamp validation fails
**Solution**: Use `createUTCTimestamp()` utility. Don't create timestamps manually.

### Issue: Correlation ID validation fails
**Solution**: Use `createCorrelationId()` utility. Don't create UUIDs manually.

## Support

For questions or issues with canonical schemas:
1. Check the validation examples in `validation-examples.ts`
2. Review the schema definitions in each `*-canonical.ts` file
3. Consult the main CLAUDE.md constitution

## References

- **CLAUDE.md**: The Federation Constitution
- **Architecture**: Anti-Corruption Layer pattern
- **Zod Documentation**: https://zod.dev/
