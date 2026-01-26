"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createCachingMiddleware = exports.loggingMiddleware = void 0;
const errors_1 = require("./errors");
const loggingMiddleware = async (context, next) => {
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
    }
    catch (error) {
        const duration = Date.now() - startTime;
        const integrationError = (0, errors_1.createIntegrationError)(context.integration, error);
        console.error(`[OpenEvolve] [${context.executionId}] ${context.integration} failed after ${duration}ms:`, integrationError);
        throw integrationError;
    }
};
exports.loggingMiddleware = loggingMiddleware;
function stableStringify(obj, seen = new Set()) {
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
    }
    finally {
        seen.delete(obj);
    }
}
const createCachingMiddleware = (ttlMs = 60000, maxSize = 100) => {
    const cache = new Map();
    const cleanup = () => {
        const now = Date.now();
        for (const [key, value] of cache.entries()) {
            if (value.expires <= now) {
                cache.delete(key);
            }
        }
    };
    const middleware = async (context, next) => {
        const isCacheable = context.integration === 'knowledge' ||
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
            if (cache.size >= maxSize) {
                cleanup();
                if (cache.size >= maxSize) {
                    const firstKey = cache.keys().next().value;
                    if (firstKey)
                        cache.delete(firstKey);
                }
            }
            cache.set(cacheKey, {
                result,
                expires: now + ttlMs
            });
            return result;
        }
        catch (error) {
            throw (0, errors_1.createIntegrationError)(context.integration, error);
        }
    };
    middleware.clear = () => {
        cache.clear();
    };
    return middleware;
};
exports.createCachingMiddleware = createCachingMiddleware;
//# sourceMappingURL=middleware.js.map