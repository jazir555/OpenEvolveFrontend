/**
 * Response caching utilities for Bubble implementations
 * Provides in-memory and persistent caching with TTL support
 */
/**
 * Cache configuration
 */
export interface CacheConfig {
    maxSize?: number;
    defaultTtl?: number;
    cleanupInterval?: number;
}
/**
 * Cache statistics
 */
export interface CacheStats {
    size: number;
    hits: number;
    misses: number;
    hitRate: number;
    evictions: number;
}
/**
 * Generic in-memory cache with TTL support
 */
export declare class InMemoryCache<T> {
    private config;
    private cache;
    private hits;
    private misses;
    private evictions;
    private cleanupTimer?;
    constructor(config?: CacheConfig);
    /**
     * Set a value in the cache
     * @param key - Cache key
     * @param value - Value to cache
     * @param ttl - Time to live in milliseconds (optional)
     */
    set(key: string, value: T, ttl?: number): void;
    /**
     * Get a value from the cache
     * @param key - Cache key
     * @returns Cached value or undefined if not found or expired
     */
    get(key: string): T | undefined;
    /**
     * Check if a key exists in the cache and is not expired
     * @param key - Cache key
     * @returns True if key exists and is not expired
     */
    has(key: string): boolean;
    /**
     * Delete a key from the cache
     * @param key - Cache key
     * @returns True if key was deleted
     */
    delete(key: string): boolean;
    /**
     * Clear all entries from the cache
     */
    clear(): void;
    /**
     * Get cache statistics
     * @returns Cache statistics
     */
    getStats(): CacheStats;
    /**
     * Get all cache keys (including expired)
     * @returns Array of cache keys
     */
    keys(): string[];
    /**
     * Get cache size (number of entries)
     * @returns Number of entries
     */
    size(): number;
    /**
     * Clean up expired entries
     */
    cleanup(): void;
    /**
     * Start automatic cleanup timer
     */
    private startCleanupTimer;
    /**
     * Stop automatic cleanup timer
     */
    stopCleanup(): void;
    /**
     * Evict the oldest entry from the cache
     */
    private evictOldest;
    /**
     * Destroy the cache and cleanup resources
     */
    destroy(): void;
}
/**
 * Cache key generator for common patterns
 */
export declare class CacheKeyBuilder {
    private parts;
    /**
     * Add a prefix to the cache key
     * @param prefix - Prefix to add
     * @returns This builder for chaining
     */
    withPrefix(prefix: string): this;
    /**
     * Add a component to the cache key
     * @param component - Component to add
     * @returns This builder for chaining
     */
    withComponent(component: string): this;
    /**
     * Add parameters to the cache key
     * @param params - Parameters object
     * @returns This builder for chaining
     */
    withParams(params: Record<string, unknown>): this;
    /**
     * Build the final cache key
     * @returns Cache key string
     */
    build(): string;
    /**
     * Reset the builder
     * @returns This builder for chaining
     */
    reset(): this;
}
/**
 * Cached response wrapper
 */
export interface CachedResponse<T> {
    data: T;
    cached: boolean;
    cachedAt?: number;
    age?: number;
}
/**
 * Cache decorator for async functions
 * @param cache - Cache instance
 * @param keyFn - Function to generate cache key
 * @param ttl - Optional TTL override
 * @returns Decorated function
 */
export declare function cached<T extends (...args: unknown[]) => Promise<unknown>>(cache: InMemoryCache<ReturnType<T>>, keyFn: (...args: Parameters<T>) => string, ttl?: number): T;
/**
 * Cache with multiple tiers (L1: in-memory, L2: persistent)
 */
export declare class MultiTierCache<T> {
    private l1Cache;
    private l2Cache?;
    constructor(l1Cache: InMemoryCache<T>, l2Cache?: InMemoryCache<T> | undefined);
    /**
     * Set a value in all cache tiers
     * @param key - Cache key
     * @param value - Value to cache
     * @param ttl - Time to live in milliseconds (optional)
     */
    set(key: string, value: T, ttl?: number): void;
    /**
     * Get a value from cache tiers (L1 first, then L2)
     * @param key - Cache key
     * @returns Cached value or undefined if not found
     */
    get(key: string): T | undefined;
    /**
     * Delete a key from all cache tiers
     * @param key - Cache key
     * @returns True if key was deleted from any tier
     */
    delete(key: string): boolean;
    /**
     * Clear all cache tiers
     */
    clear(): void;
    /**
     * Get combined statistics from all tiers
     * @returns Combined cache statistics
     */
    getStats(): {
        l1: CacheStats;
        l2?: CacheStats;
        combined: CacheStats;
    };
}
/**
 * Global cache instances for common use cases
 */
export declare const globalCaches: {
    /** Cache for HTTP responses */
    http: InMemoryCache<string>;
    /** Cache for database query results */
    database: InMemoryCache<unknown[]>;
    /** Cache for API responses */
    api: InMemoryCache<unknown>;
    /** Cache for user data */
    userData: InMemoryCache<unknown>;
};
/**
 * Helper function to create a cache key from parts
 * @param parts - Parts to join into a cache key
 * @returns Cache key string
 */
export declare function createCacheKey(...parts: (string | number | boolean)[]): string;
/**
 * Helper function to hash a cache key
 * @param key - Cache key to hash
 * @returns Hashed cache key
 */
export declare function hashCacheKey(key: string): string;
//# sourceMappingURL=cache.d.ts.map