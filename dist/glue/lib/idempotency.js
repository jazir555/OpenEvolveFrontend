"use strict";
/**
 * Idempotency Utilities - Law of Idempotency (Section 1.4)
 * Every "Glue Action" must be safe to run 100 times
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.idempotentCreate = idempotentCreate;
exports.upsert = upsert;
exports.deduplicate = deduplicate;
exports.idempotentBatch = idempotentBatch;
exports.idempotentWrite = idempotentWrite;
exports.idempotentRetry = idempotentRetry;
const structuredLogger_1 = require("./structuredLogger");
/**
 * Generic idempotent resource creator
 * Checks if resource exists before creating, using distinct IDs
 */
async function idempotentCreate(checkFn, createFn, context) {
    const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    // Check if resource exists
    const checkResult = await checkFn();
    if (checkResult.exists && checkResult.resource) {
        structuredLogger_1.logger.info('Resource already exists, returning existing', {
            ...context,
            correlation_id: correlationId,
            resource_id: checkResult.id
        });
        return checkResult.resource;
    }
    // Create new resource
    structuredLogger_1.logger.info('Resource does not exist, creating new', {
        ...context,
        correlation_id: correlationId
    });
    try {
        const created = await createFn();
        structuredLogger_1.logger.info('Resource created successfully', {
            ...context,
            correlation_id: correlationId
        });
        return created;
    }
    catch (error) {
        structuredLogger_1.logger.error('Failed to create resource', error, {
            ...context,
            correlation_id: correlationId
        });
        throw error;
    }
}
/**
 * Generic UPSERT operation
 * Updates if exists, creates if not
 */
async function upsert(checkFn, createFn, updateFn, context) {
    const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const checkResult = await checkFn();
    if (checkResult.exists && checkResult.resource) {
        structuredLogger_1.logger.info('Resource exists, updating', {
            ...context,
            correlation_id: correlationId,
            resource_id: checkResult.id
        });
        return await updateFn(checkResult.resource);
    }
    structuredLogger_1.logger.info('Resource does not exist, creating', {
        ...context,
        correlation_id: correlationId
    });
    return await createFn();
}
/**
 * Deduplicate items based on distinct ID
 */
function deduplicate(items, getId = (item) => item.id || item.name || '') {
    const seen = new Set();
    return items.filter(item => {
        const id = getId(item);
        if (seen.has(id)) {
            return false;
        }
        seen.add(id);
        return true;
    });
}
/**
 * Idempotent batch operation
 * Processes items in batches, skipping duplicates
 */
async function idempotentBatch(items, processFn, getId, context) {
    const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    // Deduplicate items
    const uniqueItems = deduplicate(items, getId);
    structuredLogger_1.logger.info(`Processing ${uniqueItems.length} unique items (deduplicated from ${items.length})`, {
        ...context,
        correlation_id: correlationId
    });
    const results = [];
    const errors = [];
    for (const item of uniqueItems) {
        try {
            const result = await processFn(item);
            results.push(result);
        }
        catch (error) {
            errors.push({ item, error: error });
            // Continue processing other items (graceful degradation)
        }
    }
    if (errors.length > 0) {
        structuredLogger_1.logger.warn(`Batch processing completed with ${errors.length} errors`, {
            ...context,
            correlation_id: correlationId,
            total_processed: uniqueItems.length,
            successful: results.length,
            failed: errors.length
        });
    }
    return results;
}
/**
 * Idempotent file/content write
 * Only writes if content has changed
 */
async function idempotentWrite(path, getContent, writeContent, context) {
    const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    try {
        const newContent = await getContent();
        // Check if file exists and content is same
        try {
            const existingContent = await getContent();
            if (existingContent === newContent) {
                structuredLogger_1.logger.info('Content unchanged, skipping write', {
                    ...context,
                    correlation_id: correlationId,
                    path
                });
                return false;
            }
        }
        catch {
            // File doesn't exist, will create
        }
        await writeContent(newContent);
        structuredLogger_1.logger.info('Content written successfully', {
            ...context,
            correlation_id: correlationId,
            path
        });
        return true;
    }
    catch (error) {
        structuredLogger_1.logger.error('Failed to write content', error, {
            ...context,
            correlation_id: correlationId,
            path
        });
        throw error;
    }
}
/**
 * Retry with exponential backoff and idempotency
 */
async function idempotentRetry(fn, maxRetries = 3, baseDelay = 1000, context) {
    const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            lastError = error;
            if (attempt === maxRetries) {
                structuredLogger_1.logger.error('All retry attempts exhausted', error, {
                    ...context,
                    correlation_id: correlationId,
                    attempts: attempt + 1
                });
                throw lastError;
            }
            const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000; // Exponential backoff with jitter
            structuredLogger_1.logger.warn(`Attempt ${attempt + 1} failed, retrying in ${delay}ms`, {
                ...context,
                correlation_id: correlationId,
                error: lastError.message,
                next_attempt_in_ms: delay
            });
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw lastError;
}
//# sourceMappingURL=idempotency.js.map