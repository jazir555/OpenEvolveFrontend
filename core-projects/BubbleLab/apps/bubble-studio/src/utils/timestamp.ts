/**
 * Timestamp Utilities
 *
 * Per CLAUDE.md LAW OF UTC:
 * - All Glue Code runs in UTC
 * - Ingest timestamps → Convert to UTC ISO-8601 → Process
 *
 * This utility ensures consistent UTC timestamp handling across the application.
 */

/**
 * Get current timestamp in UTC ISO-8601 format
 *
 * @returns Current UTC timestamp in ISO-8601 format (e.g., "2025-01-19T12:00:00.000Z")
 *
 * @example
 * ```typescript
 * const timestamp = getCurrentTimestamp();
 * // "2025-01-19T12:00:00.000Z"
 * ```
 */
export function getCurrentTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Convert a Date object or timestamp to UTC ISO-8601 format
 *
 * @param date - Date object, timestamp (ms), or ISO string
 * @returns UTC timestamp in ISO-8601 format
 *
 * @example
 * ```typescript
 * const timestamp1 = toUtcISO(new Date());
 * const timestamp2 = toUtcISO(1705689600000);
 * const timestamp3 = toUtcISO("2025-01-19T12:00:00.000Z");
 * ```
 */
export function toUtcISO(date: Date | number | string): string {
  if (typeof date === 'number') {
    return new Date(date).toISOString();
  }
  if (date instanceof Date) {
    return date.toISOString();
  }
  // Assume it's already a string, validate and return
  const parsed = new Date(date);
  if (isNaN(parsed.getTime())) {
    throw new Error(`Invalid date: ${date}`);
  }
  return parsed.toISOString();
}

/**
 * Validate if a string is a valid UTC ISO-8601 timestamp
 *
 * @param timestamp - Timestamp string to validate
 * @returns true if valid UTC ISO-8601 format
 *
 * @example
 * ```typescript
 * isValidUtcISO("2025-01-19T12:00:00.000Z"); // true
 * isValidUtcISO("2025-01-19T12:00:00"); // false (missing Z)
 * ```
 */
export function isValidUtcISO(timestamp: string): boolean {
  if (!timestamp) return false;
  const parsed = new Date(timestamp);
  return !isNaN(parsed.getTime()) && timestamp.endsWith('Z');
}

/**
 * Get current time in milliseconds since epoch (for performance measurements)
 *
 * Note: For timestamps that will be stored or transmitted, use getCurrentTimestamp() instead.
 * This is only for duration calculations and performance metrics.
 *
 * @returns Milliseconds since epoch
 *
 * @example
 * ```typescript
 * const start = getCurrentTimeMs();
 * // ... do work ...
 * const duration = getCurrentTimeMs() - start;
 * ```
 */
export function getCurrentTimeMs(): number {
  return Date.now();
}

/**
 * Calculate duration between two UTC timestamps
 *
 * @param startTimestamp - Start timestamp in UTC ISO-8601 format
 * @param endTimestamp - End timestamp in UTC ISO-8601 format (defaults to now)
 * @returns Duration in milliseconds
 *
 * @example
 * ```typescript
 * const start = getCurrentTimestamp();
 * // ... do work ...
 * const duration = calculateDuration(start);
 * console.log(`Operation took ${duration}ms`);
 * ```
 */
export function calculateDuration(
  startTimestamp: string,
  endTimestamp?: string
): number {
  const start = new Date(startTimestamp).getTime();
  const end = endTimestamp
    ? new Date(endTimestamp).getTime()
    : Date.now();

  if (isNaN(start) || isNaN(end)) {
    throw new Error('Invalid timestamp format');
  }

  return end - start;
}

/**
 * Add duration to a timestamp
 *
 * @param timestamp - Base timestamp in UTC ISO-8601 format
 * @param milliseconds - Duration to add in milliseconds
 * @returns New timestamp in UTC ISO-8601 format
 *
 * @example
 * ```typescript
 * const now = getCurrentTimestamp();
 * const future = addDuration(now, 60000); // Add 1 minute
 * ```
 */
export function addDuration(timestamp: string, milliseconds: number): string {
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) {
    throw new Error('Invalid timestamp format');
  }
  return new Date(date.getTime() + milliseconds).toISOString();
}
