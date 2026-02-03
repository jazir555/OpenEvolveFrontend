/**
 * Connection pool manager for external services
 * Provides reusable connection pools for HTTP, PostgreSQL, and Redis
 */

import { CONNECTION_POOL, TIMEOUT } from './constants.js';
import { generateCorrelationId } from '../../utils/error-handler.js';

/**
 * Generic connection pool interface
 */
export interface ConnectionPool<T> {
  acquire(): Promise<T>;
  release(connection: T): void;
  close(): Promise<void>;
  getStats(): PoolStats;
}

/**
 * Pool statistics
 */
export interface PoolStats {
  total: number;
  active: number;
  idle: number;
  waiting: number;
  max: number;
  min: number;
}

/**
 * Connection pool configuration
 */
export interface ConnectionPoolConfig {
  min: number;
  max: number;
  acquireTimeoutMillis?: number;
  idleTimeoutMillis?: number;
  evictionRunIntervalMillis?: number;
}

/**
 * Pooled connection with metadata
 */
interface PooledConnection<T> {
  connection: T;
  lastUsed: number;
  isActive: boolean;
}

/**
 * Generic connection pool implementation
 */
export class GenericConnectionPool<T> implements ConnectionPool<T> {
  private connections: PooledConnection<T>[] = [];
  private waitingQueue: Array<(connection: T) => void> = [];
  private evictionTimer?: NodeJS.Timeout;

  constructor(
    private factory: () => Promise<T>,
    private destroyer: (connection: T) => Promise<void>,
    private config: ConnectionPoolConfig = CONNECTION_POOL,
    private name: string = 'ConnectionPool'
  ) {
    this.initializePool();
    this.startEvictionTimer();
  }

  /**
   * Initialize pool with minimum connections
   */
  private async initializePool(): Promise<void> {
    const correlationId = generateCorrelationId();
    try {
      for (let i = 0; i < this.config.min; i++) {
        const connection = await this.factory();
        this.connections.push({
          connection,
          lastUsed: Date.now(),
          isActive: false
        });
      }
      console.log(`[${correlationId}] [${this.name}] Initialized pool with ${this.connections.length} connections`);
    } catch (error) {
      console.error(`[${correlationId}] [${this.name}] Failed to initialize pool:`, error);
    }
  }

  /**
   * Acquire a connection from the pool
   */
  async acquire(): Promise<T> {
    const correlationId = generateCorrelationId();

    // Find an idle connection
    const idleConnection = this.connections.find(conn => !conn.isActive);

    if (idleConnection) {
      idleConnection.isActive = true;
      idleConnection.lastUsed = Date.now();
      console.log(`[${correlationId}] [${this.name}] Acquired existing connection (active: ${this.getActiveCount()}/${this.config.max})`);
      return idleConnection.connection;
    }

    // If we haven't reached max, create a new connection
    if (this.connections.length < this.config.max) {
      try {
        const newConnection = await this.factory();
        const pooledConnection: PooledConnection<T> = {
          connection: newConnection,
          lastUsed: Date.now(),
          isActive: true
        };
        this.connections.push(pooledConnection);
        console.log(`[${correlationId}] [${this.name}] Created new connection (active: ${this.getActiveCount()}/${this.config.max})`);
        return newConnection;
      } catch (error) {
        console.error(`[${correlationId}] [${this.name}] Failed to create connection:`, error);
        throw error;
      }
    }

    // Wait for a connection to become available
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const index = this.waitingQueue.findIndex(cb => cb === resolveCallback);
        if (index !== -1) {
          this.waitingQueue.splice(index, 1);
        }
        reject(new Error(`Connection acquisition timeout after ${this.config.acquireTimeoutMillis}ms`));
      }, this.config.acquireTimeoutMillis || TIMEOUT.EXTERNAL_API);

      const resolveCallback = (connection: T) => {
        clearTimeout(timeout);
        resolve(connection);
      };

      this.waitingQueue.push(resolveCallback);
      console.log(`[${correlationId}] [${this.name}] Added to waiting queue (position: ${this.waitingQueue.length})`);
    });
  }

  /**
   * Release a connection back to the pool
   */
  release(connection: T): void {
    const pooledConnection = this.connections.find(
      conn => conn.connection === connection && conn.isActive
    );

    if (pooledConnection) {
      pooledConnection.isActive = false;
      pooledConnection.lastUsed = Date.now();

      // If there's a waiting request, assign this connection
      if (this.waitingQueue.length > 0) {
        const nextCallback = this.waitingQueue.shift();
        if (nextCallback) {
          pooledConnection.isActive = true;
          nextCallback(connection);
        }
      }

      console.log(`[${this.name}] Released connection (active: ${this.getActiveCount()}, waiting: ${this.waitingQueue.length})`);
    }
  }

  /**
   * Close all connections in the pool
   */
  async close(): Promise<void> {
    if (this.evictionTimer) {
      clearInterval(this.evictionTimer);
    }

    const closePromises = this.connections.map(async (pooled) => {
      try {
        await this.destroyer(pooled.connection);
      } catch (error) {
        console.error(`[${this.name}] Error destroying connection:`, error);
      }
    });

    await Promise.all(closePromises);
    this.connections = [];
    this.waitingQueue = [];
    console.log(`[${this.name}] Closed all connections`);
  }

  /**
   * Get pool statistics
   */
  getStats(): PoolStats {
    return {
      total: this.connections.length,
      active: this.getActiveCount(),
      idle: this.connections.length - this.getActiveCount(),
      waiting: this.waitingQueue.length,
      max: this.config.max,
      min: this.config.min
    };
  }

  /**
   * Get count of active connections
   */
  private getActiveCount(): number {
    return this.connections.filter(conn => conn.isActive).length;
  }

  /**
   * Start eviction timer for idle connections
   */
  private startEvictionTimer(): void {
    const interval = this.config.evictionRunIntervalMillis || 60000;

    this.evictionTimer = setInterval(() => {
      this.evictIdleConnections();
    }, interval);
  }

  /**
   * Evict idle connections above minimum
   */
  private evictIdleConnections(): void {
    const now = Date.now();
    const idleTimeout = this.config.idleTimeoutMillis || CONNECTION_POOL.idleTimeoutMillis;

    // Find idle connections that can be removed
    const evictableConnections = this.connections.filter(
      conn =>
        !conn.isActive &&
        conn.lastUsed < now - idleTimeout &&
        this.connections.length > this.config.min
    );

    if (evictableConnections.length > 0) {
      evictableConnections.forEach(async (pooled) => {
        try {
          await this.destroyer(pooled.connection);
          const index = this.connections.indexOf(pooled);
          if (index !== -1) {
            this.connections.splice(index, 1);
          }
          console.log(`[${this.name}] Evicted idle connection (remaining: ${this.connections.length})`);
        } catch (error) {
          console.error(`[${this.name}] Error evicting connection:`, error);
        }
      });
    }
  }
}

/**
 * HTTP connection pool configuration
 */
export interface HttpConnectionPoolConfig extends ConnectionPoolConfig {
  keepAlive?: boolean;
  keepAliveMsecs?: number;
  maxSockets?: number;
  maxFreeSockets?: number;
  timeout?: number;
}

/**
 * HTTP connection pool for Node.js fetch
 */
export class HttpConnectionPool {
  private pools: Map<string, GenericConnectionPool<Request>> = new Map();

  constructor(private config: HttpConnectionPoolConfig = CONNECTION_POOL) {}

  /**
   * Get or create a connection pool for a specific origin
   */
  getPool(origin: string): GenericConnectionPool<Request> {
    if (!this.pools.has(origin)) {
      const pool = new GenericConnectionPool<Request>(
        async () => {
          // For HTTP, we don't actually pool connections
          // This is managed by the Node.js HTTP agent
          return new Request(origin);
        },
        async (_connection) => {
          // Cleanup if needed
        },
        this.config,
        `HttpPool-${origin}`
      );

      this.pools.set(origin, pool);
    }

    return this.pools.get(origin)!;
  }

  /**
   * Close all pools
   */
  async close(): Promise<void> {
    const closePromises = Array.from(this.pools.values()).map(pool => pool.close());
    await Promise.all(closePromises);
    this.pools.clear();
  }

  /**
   * Get stats for all pools
   */
  getAllStats(): Map<string, PoolStats> {
    const stats = new Map<string, PoolStats>();
    this.pools.forEach((pool, origin) => {
      stats.set(origin, pool.getStats());
    });
    return stats;
  }
}

/**
 * Global HTTP connection pool instance
 */
export const globalHttpPool = new HttpConnectionPool();

/**
 * PostgreSQL connection pool using pg
 */
export class PostgresConnectionPool {
  private pool?: any; // pg.Pool

  constructor(
    private connectionString: string,
    private config: ConnectionPoolConfig = CONNECTION_POOL
  ) {}

  /**
   * Initialize the PostgreSQL pool
   */
  async initialize(): Promise<void> {
    try {
      // Dynamic import of pg
      const pg = await import('pg');
      const { Pool } = pg;

      this.pool = new Pool({
        connectionString: this.connectionString,
        min: this.config.min,
        max: this.config.max,
        connectionTimeoutMillis: this.config.acquireTimeoutMillis || TIMEOUT.DATABASE_QUERY,
        idleTimeoutMillis: this.config.idleTimeoutMillis,
        allowExitOnIdle: true
      });

      console.log(`[PostgresPool] Initialized with min: ${this.config.min}, max: ${this.config.max}`);
    } catch (error) {
      console.error('[PostgresPool] Failed to initialize:', error);
      throw error;
    }
  }

  /**
   * Get a connection from the pool
   */
  async getConnection(): Promise<any> {
    if (!this.pool) {
      await this.initialize();
    }

    return this.pool.connect();
  }

  /**
   * Execute a query
   */
  async query(sql: string, params?: unknown[]): Promise<any> {
    if (!this.pool) {
      await this.initialize();
    }

    return this.pool.query(sql, params);
  }

  /**
   * Close the pool
   */
  async close(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
      this.pool = undefined;
      console.log('[PostgresPool] Closed pool');
    }
  }

  /**
   * Get pool statistics
   */
  getStats(): PoolStats {
    if (!this.pool) {
      return {
        total: 0,
        active: 0,
        idle: 0,
        waiting: 0,
        max: this.config.max,
        min: this.config.min
      };
    }

    return {
      total: this.pool.totalCount || 0,
      active: this.pool.waitingCount || 0,
      idle: (this.pool.totalCount || 0) - (this.pool.waitingCount || 0),
      waiting: this.pool.waitingCount || 0,
      max: this.config.max,
      min: this.config.min
    };
  }
}

/**
 * Global connection pool registry
 */
export class ConnectionPoolRegistry {
  private postgresPools: Map<string, PostgresConnectionPool> = new Map();
  private httpPools: Map<string, HttpConnectionPool> = new Map();

  /**
   * Get or create a PostgreSQL connection pool
   */
  getPostgresPool(connectionString: string): PostgresConnectionPool {
    if (!this.postgresPools.has(connectionString)) {
      const pool = new PostgresConnectionPool(connectionString);
      this.postgresPools.set(connectionString, pool);
    }

    return this.postgresPools.get(connectionString)!;
  }

  /**
   * Get or create an HTTP connection pool
   */
  getHttpPool(origin: string): HttpConnectionPool {
    if (!this.httpPools.has(origin)) {
      const pool = new HttpConnectionPool();
      this.httpPools.set(origin, pool);
    }

    return this.httpPools.get(origin)!;
  }

  /**
   * Close all pools
   */
  async closeAll(): Promise<void> {
    const closePromises = [
      ...Array.from(this.postgresPools.values()).map(pool => pool.close()),
      ...Array.from(this.httpPools.values()).map(pool => pool.close())
    ];

    await Promise.all(closePromises);

    this.postgresPools.clear();
    this.httpPools.clear();

    console.log('[ConnectionPoolRegistry] Closed all pools');
  }

  /**
   * Get stats for all pools
   */
  getAllStats(): {
    postgres: Map<string, PoolStats>;
    http: Map<string, PoolStats>;
  } {
    const postgresStats = new Map<string, PoolStats>();
    this.postgresPools.forEach((pool, key) => {
      postgresStats.set(key, pool.getStats());
    });

    const httpStats = new Map<string, PoolStats>();
    this.httpPools.forEach((pool, key) => {
      const poolStats = pool.getAllStats();
      poolStats.forEach((stats, origin) => {
        httpStats.set(`${key}:${origin}`, stats);
      });
    });

    return {
      postgres: postgresStats,
      http: httpStats
    };
  }
}

/**
 * Global connection pool registry instance
 */
export const globalPoolRegistry = new ConnectionPoolRegistry();
