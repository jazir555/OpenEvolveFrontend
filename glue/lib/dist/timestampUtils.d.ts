/**
 * UTC Timestamp Utilities - Law of UTC (Section 1.6)
 * All Glue Code runs in UTC
 * Ingest timestamps → Convert to UTC ISO-8601 → Process
 */
export type TimestampInput = Date | string | number;
/**
 * Convert any timestamp to UTC ISO-8601 string
 */
export declare function toUtcIso(timestamp?: TimestampInput): string;
/**
 * Get current UTC timestamp as ISO-8601 string
 */
export declare function nowUtc(): string;
/**
 * Parse timestamp and return UTC Date object
 */
export declare function parseUtc(timestamp: TimestampInput): Date;
/**
 * Check if timestamp is valid
 */
export declare function isValidTimestamp(timestamp: any): boolean;
/**
 * Calculate duration between two timestamps in milliseconds
 */
export declare function durationMs(start: TimestampInput, end?: TimestampInput): number;
/**
 * Format duration in human-readable form
 */
export declare function formatDuration(ms: number): string;
/**
 * Add milliseconds to timestamp
 */
export declare function addMs(timestamp: TimestampInput, ms: number): string;
/**
 * Add seconds to timestamp
 */
export declare function addSeconds(timestamp: TimestampInput, seconds: number): string;
/**
 * Add minutes to timestamp
 */
export declare function addMinutes(timestamp: TimestampInput, minutes: number): string;
/**
 * Add hours to timestamp
 */
export declare function addHours(timestamp: TimestampInput, hours: number): string;
/**
 * Add days to timestamp
 */
export declare function addDays(timestamp: TimestampInput, days: number): string;
/**
 * Check if timestamp is in the past
 */
export declare function isPast(timestamp: TimestampInput): boolean;
/**
 * Check if timestamp is in the future
 */
export declare function isFuture(timestamp: TimestampInput): boolean;
/**
 * Compare two timestamps
 * Returns: -1 if first < second, 0 if equal, 1 if first > second
 */
export declare function compareTimestamps(first: TimestampInput, second: TimestampInput): number;
/**
 * Get Unix timestamp (seconds since epoch)
 */
export declare function toUnix(timestamp?: TimestampInput): number;
/**
 * Create timestamp from Unix timestamp
 */
export declare function fromUnix(unix: number): string;
/**
 * Truncate timestamp to seconds (remove milliseconds)
 */
export declare function truncateToSeconds(timestamp: TimestampInput): string;
/**
 * Extract date part from timestamp (YYYY-MM-DD)
 */
export declare function toDatePart(timestamp: TimestampInput): string;
/**
 * Extract time part from timestamp (HH:mm:ss)
 */
export declare function toTimePart(timestamp: TimestampInput): string;
/**
 * Format timestamp for display in user's local timezone
 * (But keep storage in UTC)
 */
export declare function toLocalDisplay(timestamp: TimestampInput): string;
/**
 * Validate timestamp is in ISO-8601 UTC format
 */
export declare function isValidIsoUtc(timestamp: string): boolean;
/**
 * Ensure timestamp is in UTC ISO-8601 format
 * If not, convert it
 */
export declare function ensureUtcIso(timestamp: TimestampInput): string;
/**
 * Timer class for measuring operation duration
 */
export declare class Timer {
    private startTime;
    private endTime?;
    constructor();
    /**
     * Stop the timer
     */
    stop(): number;
    /**
     * Get elapsed milliseconds
     */
    elapsed(): number;
    /**
     * Get elapsed as human-readable string
     */
    elapsedFormatted(): string;
    /**
     * Get start timestamp as UTC ISO
     */
    getStartTimestamp(): string;
    /**
     * Get end timestamp as UTC ISO (if stopped)
     */
    getEndTimestamp(): string | null;
}
/**
 * Create a new timer
 */
export declare function startTimer(): Timer;
//# sourceMappingURL=timestampUtils.d.ts.map