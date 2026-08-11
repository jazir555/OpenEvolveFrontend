# Idempotency & Race Condition Fixes - Quick Reference

**Quick Guide for Developers** - 2026-01-19

---

## 🎯 WHAT WAS FIXED

| Bug # | Issue | Fix | Impact |
|-------|-------|-----|--------|
| **#9** | Evolution run creation race condition | Database-level upsert | ❌ Race Condition → ✅ Thread-Safe |
| **#10** | Node upsert race condition | Atomic upsert with unique constraint | ❌ Duplicates → ✅ Unique Records |
| **#11** | Missing transaction wrapper | Added db.transaction() | ❌ Partial Failures → ✅ Atomic Operations |
| **#14** | No request deduplication | Idempotency keys table | ❌ Duplicate Mutations → ✅ Retry-Safe |
| **#13** | Frontend can't send idempotency keys | Updated API client | ❌ Unsafe Retries → ✅ Safe Retries |

---

## 🚀 QUICK START

### Using Idempotency Keys

```typescript
// Frontend: Generate UUID and send with request
import { v4 as uuidv4 } from 'uuid';

const response = await evolutionApi.startEvolution({
  evolutionId: 'my-evolution',
  config: { /* ... */ },
  idempotencyKey: uuidv4(), // ✅ Safe to retry!
});

// If network fails, retry with SAME key
// Backend will return cached response instead of creating duplicate
```

### Database Upserts

```typescript
// ❌ DON'T: Check-then-act (race condition!)
const existing = await db.findFirst({...});
if (existing) {
  await db.update({...});
} else {
  await db.insert({...});
}

// ✅ DO: Atomic upsert (thread-safe!)
const [result] = await db
  .insert(table)
  .values({...})
  .onConflictDoUpdate({
    target: [table.col1, table.col2], // Unique constraint columns
    set: {...}
  })
  .returning();
```

### Transactions

```typescript
// ❌ DON'T: Multi-step operations without transaction
await db.delete(nodes);
await db.delete(assets); // Might fail, leaving nodes orphaned!

// ✅ DO: Wrap in transaction
await db.transaction(async (tx) => {
  await tx.delete(nodes);
  await tx.delete(assets);
  await tx.update(runs);
});
// All succeed or all roll back together
```

---

## 🔧 FILES CHANGED

### Backend
- ✅ `BubbleLab/apps/bubblelab-api/src/db/schema-sqlite.ts` (unique constraint, idempotency table)
- ✅ `BubbleLab/apps/bubblelab-api/src/db/schema-postgres.ts` (unique constraint, idempotency table)
- ✅ `BubbleLab/apps/bubblelab-api/src/db/schema.ts` (export idempotencyKeys)
- ✅ `BubbleLab/apps/bubblelab-api/src/routes/evolution-graph.ts` (upserts, transactions, idempotency)
- ✅ `BubbleLab/apps/bubblelab-api/src/schemas/evolution-graph.ts` (idempotencyKey in schema)

### Frontend
- ✅ `BubbleLab/apps/bubble-studio/src/services/evolutionApi.ts` (idempotency key support)

---

## 🧪 TESTING

### Test Idempotency
```typescript
const key = uuidv4();

// First call
const r1 = await fetch('/api/evolution-graph/runs', {
  method: 'POST',
  body: JSON.stringify({ evolutionId: 'test', idempotencyKey: key }),
});

// Retry with same key
const r2 = await fetch('/api/evolution-graph/runs', {
  method: 'POST',
  body: JSON.stringify({ evolutionId: 'test', idempotencyKey: key }),
});

// Should return exact same response
assert.deepEqual(await r1.json(), await r2.json());
```

### Test Race Conditions
```typescript
// Fire 100 concurrent requests
const promises = Array(100).fill(null).map((_, i) =>
  fetch('/api/evolution-graph/runs', {
    method: 'POST',
    body: JSON.stringify({
      evolutionId: `race-test-${i}`,
      userId: 'user-123',
    }),
  })
);

await Promise.all(promises);

// Verify: Exactly 100 records, no duplicates
const runs = await db.query.evolutionRuns.findMany();
assert.equal(runs.length, 100);
```

---

## 📋 MIGRATION STEPS

### 1. Backup Database
```bash
# PostgreSQL
pg_dump dbname > backup.sql

# SQLite
cp dev.db dev.db.backup
```

### 2. Drop Old Constraint
```sql
-- SQLite
DROP INDEX IF EXISTS evolution_runs_evolutionIdUnique;

-- PostgreSQL
ALTER TABLE evolution_runs
DROP CONSTRAINT IF EXISTS evolution_runs_evolutionIdUnique;
```

### 3. Add New Constraint
```sql
-- SQLite
CREATE UNIQUE INDEX evolution_runs_userEvolutionUnique
ON evolution_runs (user_id, evolution_id);

-- PostgreSQL
ALTER TABLE evolution_runs
ADD CONSTRAINT evolution_runs_userEvolutionUnique
UNIQUE (user_id, evolution_id);
```

### 4. Create Idempotency Table
```sql
-- See full SQL in IDEMPOTENCY_RACE_CONDITION_FIXES_SUMMARY.md
CREATE TABLE idempotency_keys (...);
CREATE INDEX idempotency_keys_key_idx ...;
CREATE INDEX idempotency_keys_expires_at_idx ...;
```

### 5. Deploy
```bash
# Deploy backend
cd BubbleLab/apps/bubblelab-api
git pull
npm install
npm run build
npm start

# Deploy frontend
cd BubbleLab/apps/bubble-studio
git pull
npm install
npm run build
```

---

## ⚠️ COMMON MISTAKES

### ❌ Mistake 1: Check-Then-Act
```typescript
// WRONG: Race condition!
const exists = await db.query.users.findFirst({
  where: eq(users.email, email),
});

if (!exists) {
  await db.insert(users).values({ email });
  // Two concurrent requests can both pass the check!
}
```

### ❌ Mistake 2: No Transaction
```typescript
// WRONG: Partial failure possible!
await db.delete(orders);
await db.delete(orderItems); // If this fails, orders are deleted but items remain!
```

### ❌ Mistake 3: Ignoring Idempotency
```typescript
// WRONG: Network retry creates duplicates!
await fetch('/api/create', {
  method: 'POST',
  body: JSON.stringify(data),
}); // If network fails, retry creates duplicate!
```

---

## ✅ BEST PRACTICES

### 1. Always Use Database Constraints
```typescript
// Unique constraint at schema level
export const users = sqliteTable('users', {
  email: text('email').notNull().unique(), // ✅ Database enforces uniqueness
});
```

### 2. Always Use Transactions for Multi-Step
```typescript
await db.transaction(async (tx) => {
  // All or nothing
  await tx.insert(orders);
  await tx.insert(orderItems);
  await tx.update(inventory);
});
```

### 3. Always Send Idempotency Keys for Mutations
```typescript
await api.createResource({
  data: { /* ... */ },
  idempotencyKey: uuidv4(), // ✅ Safe to retry
});
```

### 4. Trust the Database, Not Your Code
```typescript
// ❌ Don't: Application-level checks
if (await checkExists()) {
  await create();
}

// ✅ Do: Database-level constraints
await db.insert(table).values(...).onConflictDoUpdate({...});
```

---

## 📚 RESOURCES

- **Full Documentation**: `IDEMPOTENCY_RACE_CONDITION_FIXES_SUMMARY.md`
- **Constitution**: `CLAUDE.md` - Section 1: Law of Idempotency
- **Drizzle ORM**: https://orm.drizzle.team/docs/insert#on-conflict
- **PostgreSQL**: https://www.postgresql.org/docs/current/ddl-constraints.html
- **Stripe Idempotency**: https://stripe.com/docs/api/idempotent_requests

---

## 🎯 CHEAT SHEET

```typescript
// Idempotency key
import { v4 as uuidv4 } from 'uuid';
const key = uuidv4();

// Upsert
await db.insert(table)
  .values({...})
  .onConflictDoUpdate({
    target: [table.col1, table.col2],
    set: {...}
  });

// Transaction
await db.transaction(async (tx) => {
  await tx.delete(table1);
  await tx.delete(table2);
});

// Unique constraint
sqliteTable('table', {
  col1: text('col1').notNull(),
  col2: text('col2').notNull(),
}, (table) => ({
  unique: unique().on(table.col1, table.col2),
}));
```

---

**Status**: 🟢 All bugs fixed, production ready
**Date**: 2026-01-19
