/**
 * Connection pool manager for external services
 * Provides reusable connection pools for HTTP, PostgreSQL, and Redis
 */
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
 * Generic connection pool implementation
 */
export declare class GenericConnectionPool<T> implements ConnectionPool<T> {
    private factory;
    private destroyer;
    private config;
    private name;
    private connections;
    private waitingQueue;
    private evictionTimer?;
    constructor(factory: () => Promise<T>, destroyer: (connection: T) => Promise<void>, config?: ConnectionPoolConfig, name?: string);
    /**
     * Initialize pool with minimum connections
     */
    private initializePool;
    /**
     * Acquire a connection from the pool
     */
    acquire(): Promise<T>;
    /**
     * Release a connection back to the pool
     */
    release(connection: T): void;
    /**
     * Close all connections in the pool
     */
    close(): Promise<void>;
    /**
     * Get pool statistics
     */
    getStats(): PoolStats;
    /**
     * Get count of active connections
     */
    private getActiveCount;
    /**
     * Start eviction timer for idle connections
     */
    private startEvictionTimer;
    /**
     * Evict idle connections above minimum
     */
    private evictIdleConnections;
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
export declare class HttpConnectionPool {
    private config;
    private pools;
    constructor(config?: HttpConnectionPoolConfig);
    /**
     * Get or create a connection pool for a specific origin
     */
    getPool(origin: string): GenericConnectionPool<Request>;
    /**
     * Close all pools
     */
    close(): Promise<void>;
    /**
     * Get stats for all pools
     */
    getAllStats(): Map<string, PoolStats>;
}
/**
 * Global HTTP connection pool instance
 */
export declare const globalHttpPool: HttpConnectionPool;
/**
 * PostgreSQL connection pool using pg
 */
export declare class PostgresConnectionPool {
    private connectionString;
    private config;
    private pool?;
    constructor(connectionString: string, config?: ConnectionPoolConfig);
    /**
     * Initialize the PostgreSQL pool
     */
    initialize(): Promise<void>;
    /**
     * Get a connection from the pool
     */
    getConnection(): Promise<any>;
    /**
     * Execute a query
     */
    query(sql: string, params?: unknown[]): Promise<any>;
    /**
     * Close the pool
     */
    close(): Promise<void>;
    /**
     * Get pool statistics
     */
    getStats(): PoolStats;
}
/**
 * Global connection pool registry
 */
export declare class ConnectionPoolRegistry {
    private postgresPools;
    private httpPools;
    /**
     * Get or create a PostgreSQL connection pool
     */
    getPostgresPool(connectionString: string): PostgresConnectionPool;
    /**
     * Get or create an HTTP connection pool
     */
    getHttpPool(origin: string): HttpConnectionPool;
    /**
     * Close all pools
     */
    closeAll(): Promise<void>;
    /**
     * Get stats for all pools
     */
    getAllStats(): {
        postgres: Map<string, PoolStats>;
        http: Map<string, PoolStats>;
    };
}
/**
 * Global connection pool registry instance
 */
export declare const globalPoolRegistry: ConnectionPoolRegistry;
//# sourceMappingURL=connection-pool.d.ts.map