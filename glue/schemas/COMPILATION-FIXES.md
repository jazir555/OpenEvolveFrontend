# Schema Compilation Fixes - Task #42

## Date
2026-02-22

## Summary
Fixed all TypeScript compilation errors in the canonical schema files. All 12 schema files now compile cleanly and pass validation.

## Issues Fixed

### 1. hybrid-pes-evolution-canonical.ts (Line 591)
**Error:** Property 'score' does not exist on union type

**Issue:** The code tried to access `result.best_solution.score` but TypeScript couldn't guarantee that property exists on all union members.

**Fix:** Added proper type guard with nested condition:
```typescript
// Before
quality_score: result.best_solution
  ? ('fitness' in result.best_solution ? result.best_solution.fitness : result.best_solution.score)
  : 0

// After
quality_score: result.best_solution
  ? ('fitness' in result.best_solution ? result.best_solution.fitness : 
      ('score' in result.best_solution ? result.best_solution.score : 0))
  : 0
```

### 2. vectordb-canonical.ts (Line 214)
**Error:** Property 'float' does not exist on type 'ZodNumber'

**Issue:** The `.float()` method is deprecated in newer Zod versions.

**Fix:** Removed the deprecated `.float()` call:
```typescript
// Before
score: z.number()
  .float("Score must be a float")
  .describe("Similarity score (higher is more similar)")

// After
score: z.number()
  .describe("Similarity score (higher is more similar)")
```

### 3. z3-canonical.ts (Line 321)
**Error:** Type '"greater_than"' is not comparable to allowed relation types

**Issue:** Test data used `"greater_than"` as a relation type, but the schema only allows specific enum values: 'depends_on', 'implements', 'refines', 'contradicts', 'implies', 'equivalent_to', 'instance_of', 'uses', 'defines', 'proves', 'other'.

**Fix:** Changed the test data to use a valid enum value:
```typescript
// Before
type: "greater_than"

// After
type: "implies"
```

### 4-7. validate-all-schemas.ts (Lines 246, 248, 276, 278)
**Error:** Property 'errors' does not exist on SafeParseReturnType

**Issue:** Zod's `safeParse()` returns:
- `success: true` with data
- `success: false` with `error` property (singular, not plural)

The code incorrectly tried to access `.errors` (plural) which doesn't exist.

**Fix:** Changed to access `.error.issues` for proper error reporting:
```typescript
// Before
console.log('  Errors:', nodeEmbResult.errors);
testResults.push({ 
  name: 'KarateClub NodeEmbeddingRequest', 
  passed: false, 
  error: nodeEmbResult.errors?.join(', ') 
});

// After
console.log('  Errors:', nodeEmbResult.error);
testResults.push({ 
  name: 'KarateClub NodeEmbeddingRequest', 
  passed: false, 
  error: nodeEmbResult.error?.issues.map(i => i.message).join(', ') 
});
```

## Verification

### Compilation Test
```bash
cd glue/schemas
npx tsc --noEmit *.ts
# Exit code: 0 (Success)
```

### Validation Test
```bash
npx tsx validate-all-schemas.ts
# Result: 12/12 tests passed (100%)
```

### Affected Files
1. `hybrid-pes-evolution-canonical.ts` - Type safety fix
2. `vectordb-canonical.ts` - Zod API update
3. `z3-canonical.ts` - Enum value correction
4. `validate-all-schemas.ts` - Zod error handling fix

## Impact
- All canonical schemas now compile without errors
- All validation tests pass
- No breaking changes to schema definitions
- Improved type safety and error handling

## Remaining Work
None - all schema compilation errors have been resolved.
