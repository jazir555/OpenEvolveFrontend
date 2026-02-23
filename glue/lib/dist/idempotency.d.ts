/**
 * Idempotency Utilities - Law of Idempotency (Section 1.4)
 * Every "Glue Action" must be safe to run 100 times
 */
import { LogContext } from './structuredLogger';
export interface IdempotencyCheckResult {
    exists: boolean;
    resource?: any;
    id?: string;
}
/**
 * Generic idempotent resource creator
 * Checks if resource exists before creating, using distinct IDs
 */
export declare function idempotentCreate<T>(checkFn: () => Promise<IdempotencyCheckResult>, createFn: () => Promise<T>, context: LogContext): Promise<T>;
/**
 * Generic UPSERT operation
 * Updates if exists, creates if not
 */
export declare function upsert<T>(checkFn: () => Promise<IdempotencyCheckResult>, createFn: () => Promise<T>, updateFn: (resource: T) => Promise<T>, context: LogContext): Promise<T>;
/**
 * Deduplicate items based on distinct ID
 */
export declare function deduplicate<T extends {
    id?: string;
    name?: string;
}>(items: T[], getId?: (item: T) => string): T[];
/**
 * Idempotent batch operation
 * Processes items in batches, skipping duplicates
 */
export declare function idempotentBatch<T, R>(items: T[], processFn: (item: T) => Promise<R>, getId: (item: T) => string, context: LogContext): Promise<R[]>;
/**
 * Idempotent file/content write
 * Only writes if content has changed
 */
export declare function idempotentWrite(path: string, getContent: () => Promise<string>, writeContent: (content: string) => Promise<void>, context: LogContext): Promise<boolean>;
/**
 * Retry with exponential backoff and idempotency
 */
export declare function idempotentRetry<T>(fn: () => Promise<T>, maxRetries: number | undefined, baseDelay: number | undefined, context: LogContext): Promise<T>;
//# sourceMappingURL=idempotency.d.ts.map