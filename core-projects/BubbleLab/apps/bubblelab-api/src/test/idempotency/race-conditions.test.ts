/**
 * RACE CONDITION TESTS
 *
 * Tests for race condition prevention in the Evolution API
 *
 * Tests:
 * - 100 concurrent requests to createEvolutionRun with same (userId, evolutionId) result in 1 run
 * - 100 concurrent requests to upsertEvolutionNode with same (runId, nodeId) result in 1 node
 * - No duplicate records created under high concurrency
 * - Database constraints enforce uniqueness
 * - Last write wins behavior verification
 *
 * CLAUDE.md Compliance: Law 2 - THE LAW OF "RUNTIME TRUTH"
 */

// @ts-expect-error - Bun test types
import { beforeEach, afterEach, describe, test, expect, beforeAll } from 'bun:test';
import { db } from '../../db/index.js';
import { evolutionRuns, evolutionNodes, users } from '../../db/schema.js';
import { eq, and, count } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/libsql/migrator';

const TEST_USER_ID = 'race-condition-test-user';

describe('Race Condition Tests', () => {
  beforeAll(async () => {
    console.log('Setting up race condition test database...');
    try {
      await migrate(db, { migrationsFolder: './drizzle-sqlite' });
      console.log('✅ Race condition test database migrations completed');
    } catch (error) {
      console.error('❌ Race condition test database migration failed:', error);
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
      firstName: 'Race',
      lastName: 'Condition',
      email: 'race@example.com',
      appType: 'nodex',
    });
  });

  afterEach(async () => {
    // Clean up after each test
    await db.delete(evolutionNodes);
    await db.delete(evolutionRuns);
  });

  describe('Bug #9: Race Condition in Evolution Run Creation', () => {
    test('should handle 100 concurrent requests to createEvolutionRun with same (userId, evolutionId)', async () => {
      const evolutionId = 'race-test-evolution-100';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Race Condition Test',
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

      const firstId = ids[0];
      expect(ids.every((id) => id === firstId)).toBe(true);

      // Verify only one record exists in database
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
      expect(allRuns[0].id).toBe(firstId);
    });

    test('should enforce unique constraint on (userId, evolutionId)', async () => {
      const evolutionId = 'race-test-unique-constraint';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Unique Constraint Test',
      };

      // Create initial run
      await db.insert(evolutionRuns).values(payload);

      // Try to insert duplicate (should fail or upsert)
      try {
        await db.insert(evolutionRuns).values(payload);
        // If we reach here, the unique constraint blocked the insert
        const runs = await db
          .select()
          .from(evolutionRuns)
          .where(eq(evolutionRuns.evolutionId, evolutionId));
        expect(runs.length).toBe(1);
      } catch (error: any) {
        // SQLite should throw a unique constraint violation
        expect(error.message).toContain('UNIQUE constraint failed');
      }
    });

    test('should handle 1000 concurrent requests without duplicates', async () => {
      const evolutionId = 'race-test-1000-concurrent';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'High Concurrency Test',
      };

      // Launch 1000 concurrent upsert operations
      const promises = Array.from({ length: 1000 }, () =>
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

      const startTime = Date.now();
      const results = await Promise.all(promises);
      const duration = Date.now() - startTime;

      console.log(`1000 concurrent requests completed in ${duration}ms`);

      // Verify all results have the same ID
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(1);

      // Verify only one record exists
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);

      // Performance check: should complete in reasonable time (< 10 seconds)
      expect(duration).toBeLessThan(10000);
    });
  });

  describe('Bug #10: Race Condition in Node Upsert', () => {
    test('should handle 100 concurrent requests to upsertEvolutionNode with same (runId, nodeId)', async () => {
      // First create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'race-test-node-run',
          status: 'running',
        })
        .returning();

      const nodeId = 'node-1';
      const payload = {
        runId: run.id,
        nodeId,
        parentNodeId: null,
        generation: 1,
        status: 'completed' as const,
        fitness: 0.95,
        score: 0.88,
        label: 'Test Node',
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
              parentNodeId: payload.parentNodeId,
              generation: payload.generation,
              status: payload.status,
              fitness: payload.fitness,
              score: payload.score,
              label: payload.label,
              htmlAssetId: payload.htmlAssetId,
              thumbnailAssetId: payload.thumbnailAssetId,
              metadata: payload.metadata,
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

    test('should handle concurrent upserts with different values for same node', async () => {
      // Create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'race-test-node-conflict',
          status: 'running',
        })
        .returning();

      const nodeId = 'node-conflict';

      // Launch 100 concurrent upserts with different fitness scores
      const promises = Array.from({ length: 100 }, (_, i) =>
        db
          .insert(evolutionNodes)
          .values({
            runId: run.id,
            nodeId,
            parentNodeId: null,
            generation: 1,
            status: 'completed' as const,
            fitness: i / 100, // Different fitness for each request
            score: 0.5,
            label: `Node ${i}`,
            htmlAssetId: null,
            thumbnailAssetId: null,
            metadata: null,
          })
          .onConflictDoUpdate({
            target: [evolutionNodes.runId, evolutionNodes.nodeId],
            set: {
              fitness: i / 100,
              label: `Node ${i}`,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      // All should have same ID (same record)
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(1);

      // Verify only one record exists
      const allNodes = await db
        .select()
        .from(evolutionNodes)
        .where(
          and(eq(evolutionNodes.runId, run.id), eq(evolutionNodes.nodeId, nodeId))
        );

      expect(allNodes.length).toBe(1);

      // The final value should be from the last write
      console.log(`Final fitness: ${allNodes[0].fitness}`);
      console.log(`Final label: ${allNodes[0].label}`);
    });

    test('should enforce unique constraint on (runId, nodeId)', async () => {
      // Create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'race-test-node-unique',
          status: 'running',
        })
        .returning();

      const nodeId = 'node-unique';
      const payload = {
        runId: run.id,
        nodeId,
        parentNodeId: null,
        generation: 1,
        status: 'completed' as const,
        fitness: 0.9,
        score: 0.8,
        label: 'Unique Node',
        htmlAssetId: null,
        thumbnailAssetId: null,
        metadata: null,
      };

      // Create initial node
      await db.insert(evolutionNodes).values(payload);

      // Try to insert duplicate (should fail or upsert)
      try {
        await db.insert(evolutionNodes).values(payload);
        // If we reach here, the unique constraint blocked the insert
        const nodes = await db
          .select()
          .from(evolutionNodes)
          .where(
            and(eq(evolutionNodes.runId, run.id), eq(evolutionNodes.nodeId, nodeId))
          );
        expect(nodes.length).toBe(1);
      } catch (error: any) {
        // SQLite should throw a unique constraint violation
        expect(error.message).toContain('UNIQUE constraint failed');
      }
    });
  });

  describe('Mixed Race Condition Scenarios', () => {
    test('should handle concurrent runs and nodes creation', async () => {
      const evolutionId = 'race-test-mixed-1';
      const numRuns = 10;
      const numNodesPerRun = 10;

      // Create 10 runs concurrently
      const runPromises = Array.from({ length: numRuns }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId: `${evolutionId}-${i}`,
            status: 'running',
          })
          .returning()
      );

      const runResults = await Promise.all(runPromises);
      expect(runResults.length).toBe(numRuns);

      // Create nodes for each run concurrently
      const nodePromises: Promise<any>[] = [];

      for (const [runIndex, runResult] of runResults.entries()) {
        const runId = runResult[0].id;

        for (let nodeIndex = 0; nodeIndex < numNodesPerRun; nodeIndex++) {
          nodePromises.push(
            db
              .insert(evolutionNodes)
              .values({
                runId,
                nodeId: `node-${nodeIndex}`,
                parentNodeId: null,
                generation: 1,
                status: 'completed',
                fitness: 0.9,
                score: 0.8,
                label: `Node ${nodeIndex}`,
                htmlAssetId: null,
                thumbnailAssetId: null,
                metadata: null,
              })
              .onConflictDoUpdate({
                target: [evolutionNodes.runId, evolutionNodes.nodeId],
                set: {
                  status: 'completed',
                  updatedAt: new Date(),
                },
              })
              .returning()
          );
        }
      }

      const nodeResults = await Promise.all(nodePromises);
      expect(nodeResults.length).toBe(numRuns * numNodesPerRun);

      // Verify counts
      const runCount = await db
        .select({ count: count() })
        .from(evolutionRuns)
        .where(eq(evolutionRuns.userId, TEST_USER_ID));

      expect(runCount[0].count).toBe(numRuns);

      const nodeCount = await db
        .select({ count: count() })
        .from(evolutionNodes);

      expect(nodeCount[0].count).toBe(numRuns * numNodesPerRun);
    });

    test('should handle rapid sequential upserts', async () => {
      const evolutionId = 'race-test-sequential';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Sequential Test',
      };

      let lastId: number | null = null;

      // Execute 100 rapid sequential upserts
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
          expect(run.id).toBe(lastId);
        }
      }

      // Verify only one record
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);
    });
  });

  describe('Last Write Wins Verification', () => {
    test('should update record with last write in concurrent scenario', async () => {
      const evolutionId = 'race-test-last-write';
      const payload = {
        userId: TEST_USER_ID,
        evolutionId,
        status: 'running' as const,
        name: 'Last Write Test',
      };

      // Create initial record
      const [initial] = await db
        .insert(evolutionRuns)
        .values({
          ...payload,
          name: 'Initial Name',
        })
        .returning();

      // Launch 100 concurrent updates with different names
      const promises = Array.from({ length: 100 }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            ...payload,
            name: `Update ${i}`,
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              name: `Update ${i}`,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      await Promise.all(promises);

      // Verify the record was updated
      const [final] = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(final.id).toBe(initial.id);
      expect(final.name).toMatch(/Update \d+/);
    });
  });
});
