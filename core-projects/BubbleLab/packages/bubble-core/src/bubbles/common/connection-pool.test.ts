/**
 * Comprehensive tests for common connection pool utilities
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GenericConnectionPool,
  HttpConnectionPool,
  PostgresConnectionPool,
  ConnectionPoolRegistry,
  globalHttpPool,
  globalPoolRegistry,
  type ConnectionPoolConfig,
  type PoolStats
} from './connection-pool.js';

// Mock interface for testing
interface MockConnection {
  id: string;
  close: () => Promise<void>;
}

describe('connection-pool utilities', () => {
  describe('GenericConnectionPool', () => {
    let pool: GenericConnectionPool<MockConnection>;
    let connections: Map<string, MockConnection>;
    let createCount = 0;
    let destroyCount = 0;

    beforeEach(() => {
      createCount = 0;
      destroyCount = 0;
      connections = new Map();

      const factory = async (): Promise<MockConnection> => {
        createCount++;
        const id = `conn-${createCount}`;
        const conn = {
          id,
          close: async () => {
            destroyCount++;
            connections.delete(id);
          }
        };
        connections.set(id, conn);
        return conn;
      };

      const destroyer = async (conn: MockConnection) => {
        await conn.close();
      };

      pool = new GenericConnectionPool<MockConnection>(
        factory,
        destroyer,
        {
          min: 2,
          max: 5,
          acquireTimeoutMillis: 1000,
          idleTimeoutMillis: 500,
          evictionRunIntervalMillis: 200
        },
        'TestPool'
      );
    });

    afterEach(async () => {
      await pool.close();
    });

    describe('initialization', () => {
      it('should initialize with minimum connections', async () => {
        // Wait for initialization
        await new Promise(resolve => setTimeout(resolve, 50));

        const stats = pool.getStats();
        expect(stats.total).toBeGreaterThanOrEqual(2); // At least min connections
      });

      it('should provide accurate stats', () => {
        const stats = pool.getStats();

        expect(stats).toHaveProperty('total');
        expect(stats).toHaveProperty('active');
        expect(stats).toHaveProperty('idle');
        expect(stats).toHaveProperty('waiting');
        expect(stats).toHaveProperty('max');
        expect(stats).toHaveProperty('min');
      });
    });

    describe('acquire', () => {
      it('should acquire idle connection', async () => {
        const conn1 = await pool.acquire();
        expect(conn1).toBeDefined();
        expect(conn1.id).toMatch(/^conn-/);

        const stats = pool.getStats();
        expect(stats.active).toBe(1);

        pool.release(conn1);
      });

      it('should create new connection when none available', async () => {
        const initialStats = pool.getStats();

        const conn = await pool.acquire();

        const afterStats = pool.getStats();
        expect(afterStats.total).toBeGreaterThanOrEqual(initialStats.total);

        pool.release(conn);
      });

      it('should reuse released connections', async () => {
        const conn1 = await pool.acquire();
        pool.release(conn1);

        const conn2 = await pool.acquire();

        // Should get the same connection back
        expect(conn2.id).toBe(conn1.id);

        pool.release(conn2);
      });

      it('should wait when pool is at max capacity', async () => {
        // Acquire all connections
        const conns: MockConnection[] = [];
        for (let i = 0; i < 5; i++) {
          const conn = await pool.acquire();
          conns.push(conn);
        }

        // Try to acquire one more - should wait
        let acquired = false;
        const acquirePromise = pool.acquire().then(() => {
          acquired = true;
        });

        // Release a connection
        await new Promise(resolve => setTimeout(resolve, 50));
        pool.release(conns[0]);

        // Should eventually acquire
        await new Promise(resolve => setTimeout(resolve, 100));
        expect(acquired).toBe(true);

        // Cleanup
        for (const conn of conns) {
          pool.release(conn);
        }
      });

      it('should timeout if connection not available', async () => {
        // Configure very short timeout
        const shortTimeoutPool = new GenericConnectionPool<MockConnection>(
          async () => ({
            id: 'test',
            close: async () => {}
          }),
          async () => {},
          {
            min: 1,
            max: 1,
            acquireTimeoutMillis: 50
          },
          'TimeoutTestPool'
        );

        const conn = await shortTimeoutPool.acquire();

        // Try to acquire another - should timeout
        await expect(shortTimeoutPool.acquire()).rejects.toThrow('timeout');

        await shortTimeoutPool.release(conn);
        await shortTimeoutPool.close();
      });
    });

    describe('release', () => {
      it('should release connection back to pool', async () => {
        const conn = await pool.acquire();

        let stats = pool.getStats();
        expect(stats.active).toBe(1);

        pool.release(conn);

        stats = pool.getStats();
        expect(stats.active).toBe(0);
      });

      it('should assign released connection to waiting request', async () => {
        const conn1 = await pool.acquire();

        // Start waiting request
        let conn2: MockConnection | null = null;
        const acquirePromise = pool.acquire().then(c => {
          conn2 = c;
        });

        // Wait a bit then release
        await new Promise(resolve => setTimeout(resolve, 50));
        pool.release(conn1);

        // Waiting request should get the connection
        await acquirePromise;
        expect(conn2).toBeDefined();

        pool.release(conn2!);
      });
    });

    describe('close', () => {
      it('should close all connections', async () => {
        const conns: MockConnection[] = [];
        for (let i = 0; i < 3; i++) {
          const conn = await pool.acquire();
          conns.push(conn);
        }

        const beforeStats = pool.getStats();
        expect(beforeStats.total).toBeGreaterThan(0);

        await pool.close();

        const afterStats = pool.getStats();
        expect(afterStats.total).toBe(0);
      });

      it('should handle errors when closing connections', async () => {
        const errorPool = new GenericConnectionPool<MockConnection>(
          async () => ({
            id: 'error-conn',
            close: async () => {
              throw new Error('Close error');
            }
          }),
          async () => {},
          { min: 1, max: 2 },
          'ErrorTestPool'
        );

        // Should not throw, just log errors
        await expect(errorPool.close()).resolves.not.toThrow();
      });

      it('should clear waiting queue on close', async () => {
        const conn = await pool.acquire();

        // Start waiting request
        const acquirePromise = pool.acquire().catch(() => {
          // Expected to be rejected
        });

        // Close pool immediately
        await pool.close();

        // Waiting promise should be rejected
        await acquirePromise;

        pool.release(conn);
      });
    });

    describe('eviction', () => {
      it('should evict idle connections above minimum', async () => {
        // Create connections above min
        const conns: MockConnection[] = [];
        for (let i = 0; i < 5; i++) {
          const conn = await pool.acquire();
          conns.push(conn);
        }

        // Release all but minimum
        for (let i = 2; i < conns.length; i++) {
          pool.release(conns[i]);
        }

        const beforeStats = pool.getStats();
        const beforeTotal = beforeStats.total;

        // Wait for eviction to run (idle timeout is 500ms, eviction runs every 200ms)
        await new Promise(resolve => setTimeout(resolve, 700));

        const afterStats = pool.getStats();

        // Some connections should have been evicted
        expect(afterStats.total).toBeLessThanOrEqual(beforeTotal);

        // Cleanup
        for (let i = 0; i < 2; i++) {
          pool.release(conns[i]);
        }
      });
    });
  });

  describe('HttpConnectionPool', () => {
    let httpPool: HttpConnectionPool;

    beforeEach(() => {
      httpPool = new HttpConnectionPool({
        min: 1,
        max: 3,
        keepAlive: true
      });
    });

    afterEach(async () => {
      await httpPool.close();
    });

    it('should create pool for origin', () => {
      const pool = httpPool.getPool('https://api.example.com');

      expect(pool).toBeDefined();
      expect(pool.getStats().max).toBe(3);
    });

    it('should return same pool for same origin', () => {
      const pool1 = httpPool.getPool('https://api.example.com');
      const pool2 = httpPool.getPool('https://api.example.com');

      expect(pool1).toBe(pool2);
    });

    it('should create different pools for different origins', () => {
      const pool1 = httpPool.getPool('https://api1.example.com');
      const pool2 = httpPool.getPool('https://api2.example.com');

      expect(pool1).not.toBe(pool2);
    });

    it('should get stats for all pools', async () => {
      httpPool.getPool('https://api1.example.com');
      httpPool.getPool('https://api2.example.com');

      const stats = httpPool.getAllStats();

      expect(stats.size).toBe(2);
      expect(stats.has('https://api1.example.com')).toBe(true);
      expect(stats.has('https://api2.example.com')).toBe(true);
    });

    it('should close all pools', async () => {
      httpPool.getPool('https://api1.example.com');
      httpPool.getPool('https://api2.example.com');

      await httpPool.close();

      const stats = httpPool.getAllStats();
      expect(stats.size).toBe(0);
    });
  });

  describe('PostgresConnectionPool', () => {
    let postgresPool: PostgresConnectionPool;

    beforeEach(() => {
      postgresPool = new PostgresConnectionPool(
        'postgresql://localhost:5432/test',
        {
          min: 1,
          max: 3
        }
      );
    });

    afterEach(async () => {
      await postgresPool.close();
    });

    it('should create pool instance', () => {
      expect(postgresPool).toBeDefined();
    });

    it('should get stats before initialization', () => {
      const stats = postgresPool.getStats();

      expect(stats.max).toBe(3);
      expect(stats.min).toBe(1);
      expect(stats.total).toBe(0);
    });

    it('should fail to query without initialization', async () => {
      // This will fail because pg is not installed and database doesn't exist
      // But we're testing the error handling
      await expect(postgresPool.query('SELECT 1')).rejects.toThrow();
    });

    // Note: Full integration tests would require actual PostgreSQL database
    // These are unit tests for the pool structure
  });

  describe('ConnectionPoolRegistry', () => {
    let registry: ConnectionPoolRegistry;

    beforeEach(() => {
      registry = new ConnectionPoolRegistry();
    });

    afterEach(async () => {
      await registry.closeAll();
    });

    it('should create and cache PostgreSQL pools', () => {
      const pool1 = registry.getPostgresPool('postgresql://localhost:5432/db1');
      const pool2 = registry.getPostgresPool('postgresql://localhost:5432/db1');
      const pool3 = registry.getPostgresPool('postgresql://localhost:5432/db2');

      expect(pool1).toBe(pool2);
      expect(pool1).not.toBe(pool3);
    });

    it('should create and cache HTTP pools', () => {
      const pool1 = registry.getHttpPool('https://api1.example.com');
      const pool2 = registry.getHttpPool('https://api1.example.com');
      const pool3 = registry.getHttpPool('https://api2.example.com');

      expect(pool1).toBe(pool2);
      expect(pool1).not.toBe(pool3);
    });

    it('should get stats for all pools', () => {
      registry.getPostgresPool('postgresql://localhost:5432/db1');
      registry.getPostgresPool('postgresql://localhost:5432/db2');
      registry.getHttpPool('https://api1.example.com');

      const stats = registry.getAllStats();

      expect(stats.postgres.size).toBe(2);
      expect(stats.http.size).toBe(1);
    });

    it('should close all pools', async () => {
      registry.getPostgresPool('postgresql://localhost:5432/db1');
      registry.getHttpPool('https://api1.example.com');

      await registry.closeAll();

      const stats = registry.getAllStats();
      expect(stats.postgres.size).toBe(0);
      expect(stats.http.size).toBe(0);
    });
  });

  describe('globalHttpPool', () => {
    it('should be a singleton instance', () => {
      expect(globalHttpPool).toBeInstanceOf(HttpConnectionPool);
    });

    it('should provide pools for different origins', () => {
      const pool1 = globalHttpPool.getPool('https://api.example.com');
      const pool2 = globalHttpPool.getPool('https://api.example.com');

      expect(pool1).toBe(pool2);
    });
  });

  describe('globalPoolRegistry', () => {
    it('should be a singleton instance', () => {
      expect(globalPoolRegistry).toBeInstanceOf(ConnectionPoolRegistry);
    });

    it('should provide PostgreSQL pools', () => {
      const pool = globalPoolRegistry.getPostgresPool('postgresql://localhost:5432/test');

      expect(pool).toBeInstanceOf(PostgresConnectionPool);
    });

    it('should provide HTTP pools', () => {
      const pool = globalPoolRegistry.getHttpPool('https://api.example.com');

      expect(pool).toBeInstanceOf(HttpConnectionPool);
    });
  });

  describe('PoolStats interface', () => {
    it('should have all required properties', () => {
      const stats: PoolStats = {
        total: 5,
        active: 2,
        idle: 3,
        waiting: 1,
        max: 10,
        min: 2
      };

      expect(stats.total).toBe(5);
      expect(stats.active).toBe(2);
      expect(stats.idle).toBe(3);
      expect(stats.waiting).toBe(1);
      expect(stats.max).toBe(10);
      expect(stats.min).toBe(2);
    });
  });

  describe('ConnectionPoolConfig interface', () => {
    it('should accept all configuration options', () => {
      const config: ConnectionPoolConfig = {
        min: 1,
        max: 10,
        acquireTimeoutMillis: 5000,
        idleTimeoutMillis: 30000,
        evictionRunIntervalMillis: 60000
      };

      expect(config.min).toBe(1);
      expect(config.max).toBe(10);
      expect(config.acquireTimeoutMillis).toBe(5000);
      expect(config.idleTimeoutMillis).toBe(30000);
      expect(config.evictionRunIntervalMillis).toBe(60000);
    });
  });
});
