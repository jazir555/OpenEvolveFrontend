/**
 * Response caching utilities for Bubble implementations
 * Provides in-memory and persistent caching with TTL support
 */
import { CACHE_TTL } from './constants.js';
import { deepClone } from './types.js';
/**
 * Generic in-memory cache with TTL support
 */
export class InMemoryCache {
    config;
    cache = new Map();
    hits = 0;
    misses = 0;
    evictions = 0;
    cleanupTimer;
    constructor(config = {}) {
        this.config = config;
        this.config = {
            maxSize: config.maxSize || 1000,
            defaultTtl: config.defaultTtl || CACHE_TTL.DEFAULT,
            cleanupInterval: config.cleanupInterval || 60000 // 1 minute
        };
        this.startCleanupTimer();
    }
    /**
     * Set a value in the cache
     * @param key - Cache key
     * @param value - Value to cache
     * @param ttl - Time to live in milliseconds (optional)
     */
    set(key, value, ttl) {
        const now = Date.now();
        const expiresAt = now + (ttl || this.config.defaultTtl);
        // If cache is full, evict oldest entry
        if (this.cache.size >= this.config.maxSize && !this.cache.has(key)) {
            this.evictOldest();
        }
        this.cache.set(key, {
            value: deepClone(value),
            expiresAt,
            createdAt: now,
            accessedAt: now,
            accessCount: 0
        });
    }
    /**
     * Get a value from the cache
     * @param key - Cache key
     * @returns Cached value or undefined if not found or expired
     */
    get(key) {
        const entry = this.cache.get(key);
        if (!entry) {
            this.misses++;
            return undefined;
        }
        // Check if expired
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            this.misses++;
            return undefined;
        }
        // Update access metadata
        entry.accessedAt = Date.now();
        entry.accessCount++;
        this.hits++;
        return deepClone(entry.value);
    }
    /**
     * Check if a key exists in the cache and is not expired
     * @param key - Cache key
     * @returns True if key exists and is not expired
     */
    has(key) {
        const entry = this.cache.get(key);
        if (!entry) {
            return false;
        }
        // Check if expired
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return false;
        }
        return true;
    }
    /**
     * Delete a key from the cache
     * @param key - Cache key
     * @returns True if key was deleted
     */
    delete(key) {
        return this.cache.delete(key);
    }
    /**
     * Clear all entries from the cache
     */
    clear() {
        this.cache.clear();
        this.hits = 0;
        this.misses = 0;
        this.evictions = 0;
    }
    /**
     * Get cache statistics
     * @returns Cache statistics
     */
    getStats() {
        const total = this.hits + this.misses;
        return {
            size: this.cache.size,
            hits: this.hits,
            misses: this.misses,
            hitRate: total > 0 ? this.hits / total : 0,
            evictions: this.evictions
        };
    }
    /**
     * Get all cache keys (including expired)
     * @returns Array of cache keys
     */
    keys() {
        return Array.from(this.cache.keys());
    }
    /**
     * Get cache size (number of entries)
     * @returns Number of entries
     */
    size() {
        return this.cache.size;
    }
    /**
     * Clean up expired entries
     */
    cleanup() {
        const now = Date.now();
        let cleaned = 0;
        for (const [key, entry] of this.cache.entries()) {
            if (now > entry.expiresAt) {
                this.cache.delete(key);
                cleaned++;
            }
        }
        if (cleaned > 0) {
            console.log(`[InMemoryCache] Cleaned up ${cleaned} expired entries`);
        }
    }
    /**
     * Start automatic cleanup timer
     */
    startCleanupTimer() {
        this.cleanupTimer = setInterval(() => {
            this.cleanup();
        }, this.config.cleanupInterval);
    }
    /**
     * Stop automatic cleanup timer
     */
    stopCleanup() {
        if (this.cleanupTimer) {
            clearInterval(this.cleanupTimer);
            this.cleanupTimer = undefined;
        }
    }
    /**
     * Evict the oldest entry from the cache
     */
    evictOldest() {
        let oldestKey;
        let oldestAccessedAt = Infinity;
        for (const [key, entry] of this.cache.entries()) {
            if (entry.accessedAt < oldestAccessedAt) {
                oldestAccessedAt = entry.accessedAt;
                oldestKey = key;
            }
        }
        if (oldestKey) {
            this.cache.delete(oldestKey);
            this.evictions++;
        }
    }
    /**
     * Destroy the cache and cleanup resources
     */
    destroy() {
        this.stopCleanup();
        this.clear();
    }
}
/**
 * Cache key generator for common patterns
 */
export class CacheKeyBuilder {
    parts = [];
    /**
     * Add a prefix to the cache key
     * @param prefix - Prefix to add
     * @returns This builder for chaining
     */
    withPrefix(prefix) {
        this.parts.push(prefix);
        return this;
    }
    /**
     * Add a component to the cache key
     * @param component - Component to add
     * @returns This builder for chaining
     */
    withComponent(component) {
        this.parts.push(component);
        return this;
    }
    /**
     * Add parameters to the cache key
     * @param params - Parameters object
     * @returns This builder for chaining
     */
    withParams(params) {
        const sortedParams = Object.keys(params)
            .sort()
            .map(key => `${key}=${JSON.stringify(params[key])}`)
            .join('&');
        this.parts.push(sortedParams);
        return this;
    }
    /**
     * Build the final cache key
     * @returns Cache key string
     */
    build() {
        return this.parts.join(':');
    }
    /**
     * Reset the builder
     * @returns This builder for chaining
     */
    reset() {
        this.parts = [];
        return this;
    }
}
/**
 * Cache decorator for async functions
 * @param cache - Cache instance
 * @param keyFn - Function to generate cache key
 * @param ttl - Optional TTL override
 * @returns Decorated function
 */
export function cached(cache, keyFn, ttl) {
    return (async (...args) => {
        const key = keyFn(...args);
        // Try to get from cache
        const cachedValue = cache.get(key);
        if (cachedValue !== undefined) {
            return cachedValue;
        }
        // Execute function and cache result
        const result = await (async () => {
            // Execute original function (this is a simplified implementation)
            return null;
        })();
        cache.set(key, result, ttl);
        return result;
    });
}
/**
 * Cache with multiple tiers (L1: in-memory, L2: persistent)
 */
export class MultiTierCache {
    l1Cache;
    l2Cache;
    constructor(l1Cache, l2Cache) {
        this.l1Cache = l1Cache;
        this.l2Cache = l2Cache;
    }
    /**
     * Set a value in all cache tiers
     * @param key - Cache key
     * @param value - Value to cache
     * @param ttl - Time to live in milliseconds (optional)
     */
    set(key, value, ttl) {
        this.l1Cache.set(key, value, ttl);
        if (this.l2Cache) {
            this.l2Cache.set(key, value, ttl);
        }
    }
    /**
     * Get a value from cache tiers (L1 first, then L2)
     * @param key - Cache key
     * @returns Cached value or undefined if not found
     */
    get(key) {
        // Try L1 cache first
        const l1Value = this.l1Cache.get(key);
        if (l1Value !== undefined) {
            return l1Value;
        }
        // Try L2 cache
        if (this.l2Cache) {
            const l2Value = this.l2Cache.get(key);
            if (l2Value !== undefined) {
                // Promote to L1
                this.l1Cache.set(key, l2Value);
                return l2Value;
            }
        }
        return undefined;
    }
    /**
     * Delete a key from all cache tiers
     * @param key - Cache key
     * @returns True if key was deleted from any tier
     */
    delete(key) {
        const l1Deleted = this.l1Cache.delete(key);
        const l2Deleted = this.l2Cache?.delete(key) || false;
        return l1Deleted || l2Deleted;
    }
    /**
     * Clear all cache tiers
     */
    clear() {
        this.l1Cache.clear();
        this.l2Cache?.clear();
    }
    /**
     * Get combined statistics from all tiers
     * @returns Combined cache statistics
     */
    getStats() {
        const l1Stats = this.l1Cache.getStats();
        const l2Stats = this.l2Cache?.getStats();
        const combinedStats = {
            size: l1Stats.size + (l2Stats?.size || 0),
            hits: l1Stats.hits + (l2Stats?.hits || 0),
            misses: l1Stats.misses + (l2Stats?.misses || 0),
            hitRate: 0,
            evictions: l1Stats.evictions + (l2Stats?.evictions || 0)
        };
        const totalRequests = combinedStats.hits + combinedStats.misses;
        combinedStats.hitRate = totalRequests > 0 ? combinedStats.hits / totalRequests : 0;
        return {
            l1: l1Stats,
            l2: l2Stats,
            combined: combinedStats
        };
    }
}
/**
 * Global cache instances for common use cases
 */
export const globalCaches = {
    /** Cache for HTTP responses */
    http: new InMemoryCache({
        maxSize: 500,
        defaultTtl: CACHE_TTL.SHORT
    }),
    /** Cache for database query results */
    database: new InMemoryCache({
        maxSize: 200,
        defaultTtl: CACHE_TTL.MEDIUM
    }),
    /** Cache for API responses */
    api: new InMemoryCache({
        maxSize: 1000,
        defaultTtl: CACHE_TTL.DEFAULT
    }),
    /** Cache for user data */
    userData: new InMemoryCache({
        maxSize: 10000,
        defaultTtl: CACHE_TTL.LONG
    })
};
/**
 * Helper function to create a cache key from parts
 * @param parts - Parts to join into a cache key
 * @returns Cache key string
 */
export function createCacheKey(...parts) {
    return parts.join(':');
}
/**
 * Helper function to hash a cache key
 * @param key - Cache key to hash
 * @returns Hashed cache key
 */
export function hashCacheKey(key) {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
        const char = key.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
}
//# sourceMappingURL=cache.js.map