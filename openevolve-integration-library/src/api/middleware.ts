import { Middleware } from './types';
import { createIntegrationError } from './errors';

/**
 * Middleware that logs execution details
 */
export const loggingMiddleware: Middleware = async (context, next) => {
  const startTime = Date.now();
  console.log(`[OpenEvolve] [${context.executionId}] Executing ${context.integration}...`, {
    inputs: context.inputs,
    options: context.options,
  });

  try {
    const result = await next();
    const duration = Date.now() - startTime;
    console.log(`[OpenEvolve] [${context.executionId}] ${context.integration} completed in ${duration}ms`);
    return result;
  } catch (error) {
    const duration = Date.now() - startTime;
    const integrationError = createIntegrationError(context.integration, error);
    console.error(`[OpenEvolve] [${context.executionId}] ${context.integration} failed after ${duration}ms:`, integrationError);
    throw integrationError;
  }
};


/**
 * Stable stringify for cache keys with circular reference detection
 */
function stableStringify(obj: any, seen: Set<any> = new Set()): string {
  if (obj === null || typeof obj !== 'object') {
    return JSON.stringify(obj);
  }

  if (seen.has(obj)) {
    return '"[Circular]"';
  }

  seen.add(obj);

  try {
    if (Array.isArray(obj)) {
      return '[' + obj.map(item => stableStringify(item, seen)).join(',') + ']';
    }

    const keys = Object.keys(obj).sort();
    return '{' + keys.map(key => `${JSON.stringify(key)}:${stableStringify(obj[key], seen)}`).join(',') + '}';
  } finally {
    seen.delete(obj);
  }
}

/**
 * Basic in-memory caching middleware
 */
export const createCachingMiddleware = (
  ttlMs: number = 60000,
  maxSize: number = 100
): Middleware & { clear: () => void } => {
  const cache = new Map<string, { result: any; expires: number }>();

  const cleanup = () => {
    const now = Date.now();
    for (const [key, value] of cache.entries()) {
      if (value.expires <= now) {
        cache.delete(key);
      }
    }
  };

  const middleware: any = async (context: any, next: any) => {
    // Only cache GET-like execution operations (heuristics)
    const isCacheable = 
      context.integration === 'knowledge' || 
      context.inputs?.operation === 'query' || 
      context.inputs?.operation === 'list' ||
      context.inputs?.operation === 'status' ||
      context.inputs?.operation === 'stats';

    if (!isCacheable || context.options?.bypassCache) {
      return await next();
    }

    const cacheKey = stableStringify({
      integration: context.integration,
      inputs: context.inputs,
      // Include specific options that might affect the result
      options: {
        limit: context.options?.metadata?.limit,
        offset: context.options?.metadata?.offset
      }
    });

    const now = Date.now();
    const cached = cache.get(cacheKey);
    if (cached && cached.expires > now) {
      return cached.result;
    }

    try {
      const result = await next();
      
      // Check size limit before adding
      if (cache.size >= maxSize) {
        cleanup();
        if (cache.size >= maxSize) {
          // Evict oldest entry (Map maintains insertion order)
          const firstKey = cache.keys().next().value;
          if (firstKey) cache.delete(firstKey);
        }
      }

      cache.set(cacheKey, {
        result,
        expires: now + ttlMs
      });

      return result;
    } catch (error) {
      // Don't cache errors, just rethrow them wrapped
      throw createIntegrationError(context.integration, error);
    }
  };


  middleware.clear = () => {
    cache.clear();
  };

  return middleware as Middleware & { clear: () => void };
};
