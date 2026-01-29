# Idempotency and Race Condition Test Suite

## Overview

This test suite validates the idempotency and race condition fixes implemented for the Evolution Graph API. The tests ensure compliance with **CLAUDE.md Law 4: THE LAW OF IDEMPOTENCY**.

## Test Files

### 1. `idempotency.test.ts`
Tests for idempotency guarantees:
- Duplicate createEvolutionRun requests return same result
- Idempotency key prevents duplicate processing
- Cached response returned for duplicate idempotency keys
- Different idempotency keys create separate runs
- 48-hour TTL enforcement
- Sequential request safety

### 2. `race-conditions.test.ts`
Tests for race condition prevention:
- 100 concurrent requests with same (userId, evolutionId) result in 1 run
- 100 concurrent requests with same (runId, nodeId) result in 1 node
- 1000 concurrent requests without duplicates
- Unique constraint enforcement
- Last write wins behavior

### 3. `transactions.test.ts`
Tests for transaction atomicity:
- Atomic deletion of nodes and assets
- Rollback on transaction failure
- Partial failure handling
- File cleanup after successful transaction
- Concurrent transaction isolation

### 4. `load.test.ts`
Tests for behavior under high load:
- 1000 concurrent run creation requests
- 100 concurrent runs with different IDs
- 1000 concurrent node upserts
- Mixed load scenarios
- Performance metrics
- Lock contention detection

## Running the Tests

### Run All Tests
```bash
cd BubbleLab/apps/bubblelab-api
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/*.test.ts --timeout 60000
```

### Run Individual Test Files
```bash
# Idempotency tests
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/idempotency.test.ts --timeout 60000

# Race condition tests
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/race-conditions.test.ts --timeout 60000

# Transaction tests
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/transactions.test.ts --timeout 60000

# Load tests
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/load.test.ts --timeout 120000
```

### Run with Coverage
```bash
DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test --coverage src/test/idempotency/*.test.ts
```

## Test Database

The tests use a separate SQLite database to avoid affecting development data:
- Database file: `test-idempotency.db`
- Migrations: `./drizzle-sqlite`
- Auto-created on first run
- Deleted after tests complete

## Expected Results

### Idempotency Tests
- ✅ All duplicate requests return same ID
- ✅ Idempotency keys are cached correctly
- ✅ Expired keys are not returned
- ✅ 48-hour TTL is enforced
- ✅ 100 sequential requests create 1 record

### Race Condition Tests
- ✅ 100 concurrent requests create 1 record
- ✅ 1000 concurrent requests complete in < 30 seconds
- ✅ Unique constraints prevent duplicates
- ✅ No orphaned records
- ✅ Last write wins behavior confirmed

### Transaction Tests
- ✅ Nodes and assets deleted atomically
- ✅ Failed transactions rollback correctly
- ✅ Files deleted only after successful commit
- ✅ Concurrent transactions isolated
- ✅ No partial updates

### Load Tests
- ✅ 1000 concurrent requests complete in < 30 seconds
- ✅ 100 nodes created concurrently in < 20 seconds
- ✅ No performance degradation over batches
- ✅ Lock contention < 5 seconds for 100 requests
- ✅ Data integrity maintained

## Troubleshooting

### Tests Fail with "UNIQUE constraint failed"
**Cause**: Database schema not updated with new constraints
**Fix**: Run migrations
```bash
cd BubbleLab/apps/bubblelab-api
DATABASE_URL=file:./test-idempotency.db drizzle-kit migrate
```

### Tests Timeout
**Cause**: System under heavy load or insufficient resources
**Fix**: Increase timeout
```bash
bun test --timeout 120000
```

### "Database is locked" Errors
**Cause**: SQLite concurrency limits under extreme load
**Fix**: This is expected behavior - tests validate that the system handles locks correctly

## Performance Benchmarks

Expected performance on modern hardware (M1/M2 MacBook, 16GB RAM):

| Test | Concurrency | Expected Duration |
|------|-------------|-------------------|
| 100 concurrent run upserts | 100 | < 2 seconds |
| 1000 concurrent run upserts | 1000 | < 30 seconds |
| 100 concurrent node upserts | 100 | < 2 seconds |
| 1000 concurrent node upserts | 1000 | < 30 seconds |
| Mixed load (50 runs, 500 nodes) | 550 | < 30 seconds |

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Idempotency Tests
  run: |
    cd BubbleLab/apps/bubblelab-api
    DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test src/test/idempotency/*.test.ts --timeout 120000

- name: Check Coverage
  run: |
    cd BubbleLab/apps/bubblelab-api
    DATABASE_URL=file:./test-idempotency.db BUBBLE_ENV=test bun test --coverage src/test/idempotency/*.test.ts
```

## CLAUDE.md Compliance

All tests validate compliance with:

### Law 4: THE LAW OF IDEMPOTENCY
> Every "Glue Action" must be safe to run 100 times.

✅ **Verified by**:
- Idempotency tests ensure 100 sequential requests create 1 record
- Race condition tests ensure 1000 concurrent requests create 1 record
- Transaction tests ensure operations are replay-safe

### Law 2: THE LAW OF "RUNTIME TRUTH"
> You generally do not trust the documentation. You trust execution.

✅ **Verified by**:
- Database constraints enforce uniqueness (not just application logic)
- Transactions guarantee atomicity (database-level)
- Race condition tests validate behavior under real concurrency

## Next Steps

1. **Run the test suite**: Execute all tests and verify they pass
2. **Review performance metrics**: Compare against expected benchmarks
3. **Fix any failures**: Address test failures before deploying to production
4. **Add to CI**: Integrate tests into continuous integration pipeline
5. **Monitor in production**: Set up alerts for idempotency violations

## Support

For issues or questions:
- Review the test code for detailed implementation
- Check CLAUDE.md for architecture principles
- Review IDEMPOTENCY_RACE_CONDITION_FIXES_SUMMARY.md for fix details
