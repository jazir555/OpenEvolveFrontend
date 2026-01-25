/**
 * CONCURRENT LOAD TESTS
 *
 * Tests for behavior under high concurrent load
 *
 * Tests:
 * - 1000 concurrent requests to create runs
 * - Verify final state is consistent
 * - Check for orphaned records
 * - Measure performance under load
 * - Check for deadlocks and lock contention
 *
 * CLAUDE.md Compliance: Law 2 - THE LAW OF "RUNTIME TRUTH"
 */

// @ts-expect-error - Bun test types
import { beforeEach, afterEach, describe, test, expect, beforeAll } from 'bun:test';
import { db } from '../../db/index.js';
import { evolutionRuns, evolutionNodes, evolutionAssets, users } from '../../db/schema.js';
import { eq, and, count, sql } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/libsql/migrator';

const TEST_USER_ID = 'load-test-user';

describe('Concurrent Load Tests', () => {
  beforeAll(async () => {
    console.log('Setting up load test database...');
    try {
      await migrate(db, { migrationsFolder: './drizzle-sqlite' });
      console.log('✅ Load test database migrations completed');
    } catch (error) {
      console.error('❌ Load test database migration failed:', error);
    }
  });

  beforeEach(async () => {
    // Clean up before each test
    await db.delete(evolutionAssets);
    await db.delete(evolutionNodes);
    await db.delete(evolutionRuns);
    await db.delete(users).where(eq(users.clerkId, TEST_USER_ID));

    // Create test user
    await db.insert(users).values({
      clerkId: TEST_USER_ID,
      firstName: 'Load',
      lastName: 'Test',
      email: 'load@example.com',
      appType: 'nodex',
    });
  });

  afterEach(async () => {
    // Clean up after each test
    await db.delete(evolutionAssets);
    await db.delete(evolutionNodes);
    await db.delete(evolutionRuns);
  });

  describe('High Concurrency Evolution Run Creation', () => {
    test('should handle 1000 concurrent run creation requests', async () => {
      const numRequests = 1000;
      const evolutionId = 'load-test-1000-concurrent';

      const startTime = Date.now();

      // Launch 1000 concurrent requests
      const promises = Array.from({ length: numRequests }, () =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId,
            status: 'running',
            name: 'Load Test Run',
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: 'running',
              name: 'Load Test Run',
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`✅ 1000 concurrent requests completed in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / numRequests).toFixed(2)}ms per request`);

      // Verify all results have the same ID
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(1);
      expect(ids.length).toBe(numRequests);

      // Verify only one record exists
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);

      // Performance assertions
      expect(duration).toBeLessThan(30000); // Should complete in < 30 seconds
    });

    test('should handle 100 concurrent runs with different evolution IDs', async () => {
      const numRuns = 100;

      const startTime = Date.now();

      // Create 100 different runs concurrently
      const promises = Array.from({ length: numRuns }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId: `load-test-run-${i}`,
            status: 'running',
          })
          .returning()
      );

      const results = await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`✅ Created ${numRuns} runs concurrently in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / numRuns).toFixed(2)}ms per run`);

      expect(results.length).toBe(numRuns);

      // Verify all unique IDs
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(numRuns);

      // Verify count in database
      const countResult = await db
        .select({ count: count() })
        .from(evolutionRuns)
        .where(eq(evolutionRuns.userId, TEST_USER_ID));

      expect(countResult[0].count).toBe(numRuns);

      // Performance check
      expect(duration).toBeLessThan(20000);
    });

    test('should verify no orphaned records after 500 concurrent requests', async () => {
      const numRequests = 500;
      const evolutionId = 'load-test-orphan-check';

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

      console.log(`📊 Total records for ${evolutionId}: ${countResult[0].count}`);
      console.log(`📊 Expected: 1, Actual: ${countResult[0].count}`);

      expect(countResult[0].count).toBe(1);

      // Check for any other orphaned records for this user
      const userRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.userId, TEST_USER_ID));

      console.log(`📊 Total runs for user: ${userRuns.length}`);
      console.log(`📊 Evolution IDs: ${[...new Set(userRuns.map((r) => r.evolutionId))].join(', ')}`);

      expect(userRuns.length).toBe(1);
    });
  });

  describe('High Concurrency Node Upsert', () => {
    test('should handle 1000 concurrent node upserts', async () => {
      // First create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'load-test-node-run',
          status: 'running',
        })
        .returning();

      const nodeId = 'load-test-node-1000';
      const numRequests = 1000;

      const startTime = Date.now();

      // Launch 1000 concurrent node upserts
      const promises = Array.from({ length: numRequests }, (_, i) =>
        db
          .insert(evolutionNodes)
          .values({
            runId: run.id,
            nodeId,
            parentNodeId: null,
            generation: 1,
            status: 'completed',
            fitness: i / 1000, // Different fitness values
            score: 0.8,
            label: `Node ${i}`,
            htmlAssetId: null,
            thumbnailAssetId: null,
            metadata: null,
          })
          .onConflictDoUpdate({
            target: [evolutionNodes.runId, evolutionNodes.nodeId],
            set: {
              fitness: i / 1000,
              label: `Node ${i}`,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`✅ 1000 concurrent node upserts completed in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / numRequests).toFixed(2)}ms per upsert`);

      // Verify single record
      const allNodes = await db
        .select()
        .from(evolutionNodes)
        .where(
          and(eq(evolutionNodes.runId, run.id), eq(evolutionNodes.nodeId, nodeId))
        );

      expect(allNodes.length).toBe(1);

      // All results should have same ID
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(1);

      // Performance check
      expect(duration).toBeLessThan(30000);
    });

    test('should handle 100 concurrent nodes with different IDs', async () => {
      // Create a run
      const [run] = await db
        .insert(evolutionRuns)
        .values({
          userId: TEST_USER_ID,
          evolutionId: 'load-test-nodes-diff',
          status: 'running',
        })
        .returning();

      const numNodes = 100;

      const startTime = Date.now();

      // Create 100 different nodes concurrently
      const promises = Array.from({ length: numNodes }, (_, i) =>
        db
          .insert(evolutionNodes)
          .values({
            runId: run.id,
            nodeId: `node-${i}`,
            parentNodeId: null,
            generation: 1,
            status: 'completed',
            fitness: 0.9,
            score: 0.8,
            label: `Node ${i}`,
            htmlAssetId: null,
            thumbnailAssetId: null,
            metadata: null,
          })
          .returning()
      );

      const results = await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`✅ Created ${numNodes} nodes concurrently in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / numNodes).toFixed(2)}ms per node`);

      expect(results.length).toBe(numNodes);

      // Verify all unique IDs
      const ids = results.map((r) => r[0].id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(numNodes);

      // Verify count
      const countResult = await db
        .select({ count: count() })
        .from(evolutionNodes)
        .where(eq(evolutionNodes.runId, run.id));

      expect(countResult[0].count).toBe(numNodes);

      // Performance check
      expect(duration).toBeLessThan(20000);
    });
  });

  describe('Mixed Load Scenarios', () => {
    test('should handle concurrent runs, nodes, and assets', async () => {
      const numRuns = 50;
      const numNodesPerRun = 10;

      const startTime = Date.now();

      // Create runs concurrently
      const runPromises = Array.from({ length: numRuns }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId: `mixed-load-run-${i}`,
            status: 'running',
          })
          .returning()
      );

      const runResults = await Promise.all(runPromises);
      console.log(`✅ Created ${numRuns} runs`);

      // Create nodes for all runs concurrently
      const nodePromises: Promise<any>[] = [];

      for (const [runIndex, runResult] of runResults.entries()) {
        const runId = runResult[0].id;

        for (let nodeIndex = 0; nodeIndex < numNodesPerRun; nodeIndex++) {
          nodePromises.push(
            db
              .insert(evolutionNodes)
              .values({
                runId,
                nodeId: `node-${runIndex}-${nodeIndex}`,
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
              .returning()
          );
        }
      }

      await Promise.all(nodePromises);

      const duration = Date.now() - startTime;

      console.log(`✅ Created ${numRuns * numNodesPerRun} nodes in ${duration}ms`);
      console.log(`⏱️  Average: ${(duration / (numRuns * numNodesPerRun)).toFixed(2)}ms per node`);

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

      // Performance check
      expect(duration).toBeLessThan(30000);
    });
  });

  describe('Performance Metrics', () => {
    test('should measure performance under sustained load', async () => {
      const numBatches = 10;
      const requestsPerBatch = 100;
      const evolutionId = 'load-test-sustained';

      const totalStartTime = Date.now();
      const batchTimes: number[] = [];

      for (let batch = 0; batch < numBatches; batch++) {
        const batchStartTime = Date.now();

        // Execute batch of concurrent requests
        const promises = Array.from({ length: requestsPerBatch }, () =>
          db
            .insert(evolutionRuns)
            .values({
              userId: TEST_USER_ID,
              evolutionId: `${evolutionId}-batch-${batch}`,
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

        const batchDuration = Date.now() - batchStartTime;
        batchTimes.push(batchDuration);

        console.log(
          `Batch ${batch + 1}/${numBatches}: ${batchDuration}ms (${(batchDuration / requestsPerBatch).toFixed(2)}ms per request)`
        );
      }

      const totalDuration = Date.now() - totalStartTime;
      const avgBatchTime = batchTimes.reduce((a, b) => a + b, 0) / batchTimes.length;
      const minBatchTime = Math.min(...batchTimes);
      const maxBatchTime = Math.max(...batchTimes);

      console.log(`\n📊 Performance Summary:`);
      console.log(`   Total Duration: ${totalDuration}ms`);
      console.log(`   Avg Batch Time: ${avgBatchTime.toFixed(2)}ms`);
      console.log(`   Min Batch Time: ${minBatchTime}ms`);
      console.log(`   Max Batch Time: ${maxBatchTime}ms`);
      console.log(`   Throughput: ${(numBatches * requestsPerBatch / (totalDuration / 1000)).toFixed(2)} requests/sec`);

      // Verify no performance degradation (max should not be more than 3x min)
      expect(maxBatchTime / minBatchTime).toBeLessThan(3);
    });

    test('should detect lock contention under high concurrency', async () => {
      const numRequests = 100;
      const evolutionId = 'load-test-lock-contention';

      const startTime = Date.now();

      // Launch concurrent requests
      const promises = Array.from({ length: numRequests }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId,
            status: 'running',
            name: `Lock Test ${i}`,
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              name: `Lock Test ${i}`,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      const duration = Date.now() - startTime;

      console.log(`⏱️  Lock contention test: ${duration}ms for ${numRequests} concurrent requests`);
      console.log(`⏱️  Average: ${(duration / numRequests).toFixed(2)}ms per request`);

      // If there's severe lock contention, this will take much longer
      // A healthy system should complete 100 requests in < 5 seconds
      expect(duration).toBeLessThan(5000);

      // Verify single record
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);
    });
  });

  describe('Data Consistency Under Load', () => {
    test('should maintain data integrity with 500 concurrent upserts', async () => {
      const numRequests = 500;
      const evolutionId = 'load-test-integrity';

      // Execute concurrent upserts
      const promises = Array.from({ length: numRequests }, (_, i) =>
        db
          .insert(evolutionRuns)
          .values({
            userId: TEST_USER_ID,
            evolutionId,
            status: i % 2 === 0 ? 'running' : 'paused',
            name: `Integrity Test ${i}`,
          })
          .onConflictDoUpdate({
            target: [evolutionRuns.userId, evolutionRuns.evolutionId],
            set: {
              status: i % 2 === 0 ? 'running' : 'paused',
              name: `Integrity Test ${i}`,
              updatedAt: new Date(),
            },
          })
          .returning()
      );

      const results = await Promise.all(promises);

      // Verify data integrity
      const allRuns = await db
        .select()
        .from(evolutionRuns)
        .where(eq(evolutionRuns.evolutionId, evolutionId));

      expect(allRuns.length).toBe(1);

      const run = allRuns[0];
      expect(run.userId).toBe(TEST_USER_ID);
      expect(run.evolutionId).toBe(evolutionId);
      expect(['running', 'paused']).toContain(run.status);

      console.log(`✅ Final status: ${run.status}`);
      console.log(`✅ Final name: ${run.name}`);
    });
  });
});
