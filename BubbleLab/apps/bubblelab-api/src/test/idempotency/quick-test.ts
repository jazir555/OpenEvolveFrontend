/**
 * QUICK IDEMPOTENCY AND RACE CONDITION TESTS
 *
 * Simplified tests that work without idempotency_keys table migration.
 * Focuses on database-level idempotency (upserts) and race conditions.
 */

// @ts-expect-error - Bun test types
import { beforeEach, afterEach, describe, test, expect, beforeAll } from 'bun:test';
import { db } from '../../db/index.js';
import { evolutionRuns, evolutionNodes, users } from '../../db/schema.js';
import { eq, and, count } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/libsql/migrator';

const TEST_USER_ID = 'quick-test-user';

describe('Quick Idempotency & Race Condition Tests', () => {
  beforeAll(async () => {
    console.log('Setting up quick test database...');
    try {
      await migrate(db, { migrationsFolder: './drizzle-sqlite' });
      console.log('✅ Quick test database migrations completed');
    } catch (error) {
      console.error('❌ Quick test database migration failed:', error);
    }
  });

  beforeEach(async () => {
    // Clean up before each test
    await db.delete(evolutionNodes);
    await db.delete(evolutionRuns);
    await db.delete(users).where(eq(users.clerkId, TEST_USER_ID));

    // Create test user
    await db.insert(users).values({
      clerkId: TEST_USER_ID,
      firstName: 'Quick',
      lastName: 'Test',
      email: 'quick@example.com',
      appType: 'nodex',
    });
  });

  afterEach(async () => {
    // Clean up after each test
    await db.delete(evolutionNodes);
    await db.delete(evolutionRuns);
  });

  describe('Database-Level Idempotency (Bug #9)', () => {
    test('should return same result when creating run with same (userId, evolutionId)', async () => {
      const evolutionId = 'quick-test-idempotency';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Quick Test',
      };

      // First creation
      const [run1] = await db
        .insert(evolutionRuns)
        .values(payload)
        .onConflictDoUpdate({
          target: [evolutionRuns.userId, evolutionRuns.evolutionId],
          set: {
            status: payload.status,
            name: payload.name,
            updatedAt: new Date(),
          },
        })
        .returning();

      // Second creation with same (userId, evolutionId)
      const [run2] = await db
        .insert(evolutionRuns)
        .values(payload)
        .onConflictDoUpdate({
          target: [evolutionRuns.userId, evolutionRuns.evolutionId],
          set: {
            status: payload.status,
            name: payload.name,
            updatedAt: new Date(),
          },
        })
        .returning();

      // Should return the same ID
      expect(run1.id).toBe(run2.id);
      expect(run1.evolutionId).toBe(run2.evolutionId);
      expect(run1.userId).toBe(run2.userId);

      // Should only have one record in database
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(
          and(
            eq(evolutionRuns.userId, TEST_USER_ID),
            eq(evolutionRuns.evolutionId, evolutionId)
          )
        );

      expect(allRuns.length).toBe(1);
    });

    test('should handle 100 sequential identical requests safely', async () => {
      const evolutionId = 'quick-test-sequential';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Sequential Test',
      };

      let lastId: number | null = null;

      // Execute 100 sequential upserts
      for (let i = 0; i < 100; i++) {
        const [run] = await db
          .insert(evolutionRuns)
          .values(payload)
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: payload.status,
              name: payload.name,
              updatedAt: new Date(),
            },
          })
          .returning();

        if (lastId === null) {
          lastId = run.id;
        } else {
          // All subsequent requests should return the same ID
          expect(run.id).toBe(lastId);
        }
      }

      // Verify only one record exists
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);
    });
  });

  describe('Race Conditions (Bug #10)', () => {
    test('should handle 100 concurrent requests with same (userId, evolutionId)', async () => {
      const evolutionId = 'quick-test-race-100';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Race Test',
      };

      // Launch 100 concurrent upsert operations
      const promises = Array.from({ length: 100 }, () =>
        db
          .insert(evolutionRuns)
          .values(payload)
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: payload.status,
              name: payload.name,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      // Wait for all operations to complete
      const results = await Promise.all(promises);

      // All results should have the same ID
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(1);
      expect(ids.length).toBe(100);

      // Verify only one record exists in database
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);
    });

    test('should handle 100 concurrent node upserts', async () => {
      // First create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'quick-test-node-race',
          status: 'running',
        })
        .returning();

      const nodeId = 'quick-node-1';
      const payload = {
        runId: run.id,
        nodeId,
        parentNodeId: null,
        generation: 1,
        status: 'completed' as const,
        fitness: 0.95,
        score: 0.88,
        label: 'Quick Node',
        htmlAssetId: null,
        thumbnailAssetId: null,
        metadata: null,
      };

      // Launch 100 concurrent upsert operations
      const promises = Array.from({ length: 100 }, () =>
        db
          .insert(evolutionNodes)
          .values(payload)
          .onConflictDoUpdate({
            target: [evolutionNodes.runId, evolutionNodes.nodeId],
            set: {
              status: payload.status,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      // All results should have the same ID
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(1);

      // Verify only one record exists in database
      const allNodes = await db
        .select()
        .from(evolutionNodes)
        .where(
          and(eq(evolutionNodes.runId, run.id), eq(evolutionNodes.nodeId, nodeId))
        );

      expect(allNodes.length).toBe(1);
    });

    test('should verify no orphaned records after 100 concurrent requests', async () => {
      const numRequests = 100;
      const evolutionId = 'quick-test-orphan';

      // Execute concurrent requests
      const promises = Array.from({ length: numRequests }, () =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId,
            status: 'running',
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: 'running',
              updatedAt: new Date(),
            },
          })
      );

      await Promise.all(promises);

      // Count all records for this evolutionId
      const countResult = await db
        .select({ count: count() })
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(countResult[0].count).toBe(1);
    });
  });

  describe('Performance Under Load', () => {
    test('should handle 500 concurrent run upserts in reasonable time', async () => {
      const numRequests = 500;
      const evolutionId = 'quick-test-load-500';

      const startTime = Date.now();

      // Launch concurrent requests
      const promises = Array.from({ length: numRequests }, () =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId,
            status: 'running',
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: 'running',
              updatedAt: new Date(),
            },
          })
      );

      await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`⏱️  500 concurrent requests completed in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / numRequests).toFixed(2)}ms per request`);

      // Verify single record
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);

      // Performance check: should complete in reasonable time (< 15 seconds)
      expect(duration).toBeLessThan(15000);
    });
  });

  describe('Transaction Atomicity (Bug #11)', () => {
    test('should rollback all operations if transaction fails', async () => {
      // Create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'quick-test-transaction',
          status: 'running',
        })
        .returning();

      // Create nodes
      await db.insert(evolutionNodes).values([
        {
          runId: run.id,
          nodeId: 'node-1',
          parentNodeId: null,
          generation: 1,
          status: 'completed',
          fitness: 0.9,
          score: 0.8,
          label: 'Node 1',
          htmlAssetId: null,
          thumbnailAssetId: null,
          metadata: null,
        },
        {
          runId: run.id,
          nodeId: 'node-2',
          parentNodeId: 'node-1',
          generation: 2,
          status: 'completed',
          fitness: 0.85,
          score: 0.75,
          label: 'Node 2',
          htmlAssetId: null,
          thumbnailAssetId: null,
          metadata: null,
        },
      ]);

      // Verify initial state
      const initialNodes = await db
        .select()
        .from(evolutionNodes)
        .where(eq(evolutionNodes.runId, run.id));

      expect(initialNodes.length).toBe(2);

      // Execute transaction that fails
      try {
        await db.transaction(async (tx) => {
          // Delete nodes
          await tx.delete(evolutionNodes).where(eq(evolutionNodes.runId, run.id));

          // Simulate failure before completing
          throw new Error('Simulated transaction failure');
        });
      } catch (error) {
        expect((error as Error).message).toBe('Simulated transaction failure');
      }

      // Verify rollback: nodes should still exist
      const rolledBackNodes = await db
        .select()
        .from(evolutionNodes)
        .where(eq(evolutionNodes.runId, run.id));

      expect(rolledBackNodes.length).toBe(2);
    });
  });
});
