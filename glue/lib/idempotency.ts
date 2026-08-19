/**
 * Idempotency Utilities - Law of Idempotency (Section 1.4)
 * Every "Glue Action" must be safe to run 100 times
 */

import { logger, LogContext } from './structuredLogger';

export interface IdempotencyCheckResult {
  exists: boolean;
  resource?: any;
  id?: string;
}

/**
 * Generic idempotent resource creator
 * Checks if resource exists before creating, using distinct IDs
 */
export async function idempotentCreate<T>(
  checkFn: () => Promise<IdempotencyCheckResult>,
  createFn: () => Promise<T>,
  context: LogContext
): Promise<T> {
  const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  // Check if resource exists
  const checkResult = await checkFn();

  if (checkResult.exists && checkResult.resource) {
    logger.info('Resource already exists, returning existing', {
      ...context,
      correlation_id: correlationId,
      resource_id: checkResult.id
    });
    return checkResult.resource as T;
  }

  // Create new resource
  logger.info('Resource does not exist, creating new', {
    ...context,
    correlation_id: correlationId
  });

  try {
    const created = await createFn();
    logger.info('Resource created successfully', {
      ...context,
      correlation_id: correlationId
    });
    return created;
  } catch (error) {
    logger.error('Failed to create resource', error as Error, {
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
export async function upsert<T>(
  checkFn: () => Promise<IdempotencyCheckResult>,
  createFn: () => Promise<T>,
  updateFn: (resource: T) => Promise<T>,
  context: LogContext
): Promise<T> {
  const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  const checkResult = await checkFn();

  if (checkResult.exists && checkResult.resource) {
    logger.info('Resource exists, updating', {
      ...context,
      correlation_id: correlationId,
      resource_id: checkResult.id
    });
    return await updateFn(checkResult.resource as T);
  }

  logger.info('Resource does not exist, creating', {
    ...context,
    correlation_id: correlationId
  });

  return await createFn();
}

/**
 * Deduplicate items based on distinct ID
 */
export function deduplicate<T extends { id?: string; name?: string }>(
  items: T[],
  getId: (item: T) => string = (item) => item.id || item.name || ''
): T[] {
  const seen = new Set<string>();
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
export async function idempotentBatch<T, R>(
  items: T[],
  processFn: (item: T) => Promise<R>,
  getId: (item: T) => string,
  context: LogContext
): Promise<R[]> {
  const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  // Deduplicate items
  const uniqueItems = deduplicate(
    items as Array<{ id?: string; name?: string }>,
    getId as (item: { id?: string; name?: string }) => string
  ) as T[];

  logger.info(`Processing ${uniqueItems.length} unique items (deduplicated from ${items.length})`, {
    ...context,
    correlation_id: correlationId
  });

  const results: R[] = [];
  const errors: Array<{ item: T; error: Error }> = [];

  for (const item of uniqueItems) {
    try {
      const result = await processFn(item);
      results.push(result);
    } catch (error) {
      errors.push({ item, error: error as Error });
      // Continue processing other items (graceful degradation)
    }
  }

  if (errors.length > 0) {
    logger.warn(`Batch processing completed with ${errors.length} errors`, {
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
export async function idempotentWrite(
  path: string,
  getContent: () => Promise<string>,
  writeContent: (content: string) => Promise<void>,
  context: LogContext
): Promise<boolean> {
  const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  try {
    const newContent = await getContent();

    // Check if file exists and content is same
    try {
      const existingContent = await getContent();
      if (existingContent === newContent) {
        logger.info('Content unchanged, skipping write', {
          ...context,
          correlation_id: correlationId,
          path
        });
        return false;
      }
    } catch {
      // File doesn't exist, will create
    }

    await writeContent(newContent);
    logger.info('Content written successfully', {
      ...context,
      correlation_id: correlationId,
      path
    });
    return true;
  } catch (error) {
    logger.error('Failed to write content', error as Error, {
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
export async function idempotentRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000,
  context: LogContext
): Promise<T> {
  const correlationId = context.correlation_id || `cid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries) {
        logger.error('All retry attempts exhausted', error as Error, {
          ...context,
          correlation_id: correlationId,
          attempts: attempt + 1
        });
        throw lastError;
      }

      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000; // Exponential backoff with jitter

      logger.warn(`Attempt ${attempt + 1} failed, retrying in ${delay}ms`, {
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
