/**
 * UTC Timestamp Utilities - Law of UTC (Section 1.6)
 * All Glue Code runs in UTC
 * Ingest timestamps → Convert to UTC ISO-8601 → Process
 */

export type TimestampInput = Date | string | number;

/**
 * Convert any timestamp to UTC ISO-8601 string
 */
export function toUtcIso(timestamp: TimestampInput = new Date()): string {
  let date: Date;

  if (timestamp instanceof Date) {
    date = timestamp;
  } else if (typeof timestamp === 'string') {
    // Parse and ensure we treat it as UTC
    date = new Date(timestamp);
  } else if (typeof timestamp === 'number') {
    date = new Date(timestamp);
  } else {
    date = new Date();
  }

  // Convert to UTC ISO-8601
  return date.toISOString();
}

/**
 * Get current UTC timestamp as ISO-8601 string
 */
export function nowUtc(): string {
  return new Date().toISOString();
}

/**
 * Parse timestamp and return UTC Date object
 */
export function parseUtc(timestamp: TimestampInput): Date {
  let date: Date;

  if (timestamp instanceof Date) {
    // Convert local date to UTC
    date = new Date(timestamp.toISOString());
  } else if (typeof timestamp === 'string') {
    date = new Date(timestamp);
  } else if (typeof timestamp === 'number') {
    date = new Date(timestamp);
  } else {
    date = new Date();
  }

  return date;
}

/**
 * Check if timestamp is valid
 */
export function isValidTimestamp(timestamp: any): boolean {
  if (!timestamp) return false;

  try {
    const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
    return !isNaN(date.getTime());
  } catch {
    return false;
  }
}

/**
 * Calculate duration between two timestamps in milliseconds
 */
export function durationMs(start: TimestampInput, end: TimestampInput = new Date()): number {
  const startDate = parseUtc(start);
  const endDate = parseUtc(end);
  return endDate.getTime() - startDate.getTime();
}

/**
 * Format duration in human-readable form
 */
export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h ${minutes % 60}m`;
  } if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

/**
 * Add milliseconds to timestamp
 */
export function addMs(timestamp: TimestampInput, ms: number): string {
  const date = parseUtc(timestamp);
  date.setTime(date.getTime() + ms);
  return date.toISOString();
}

/**
 * Add seconds to timestamp
 */
export function addSeconds(timestamp: TimestampInput, seconds: number): string {
  return addMs(timestamp, seconds * 1000);
}

/**
 * Add minutes to timestamp
 */
export function addMinutes(timestamp: TimestampInput, minutes: number): string {
  return addMs(timestamp, minutes * 60 * 1000);
}

/**
 * Add hours to timestamp
 */
export function addHours(timestamp: TimestampInput, hours: number): string {
  return addMs(timestamp, hours * 60 * 60 * 1000);
}

/**
 * Add days to timestamp
 */
export function addDays(timestamp: TimestampInput, days: number): string {
  return addMs(timestamp, days * 24 * 60 * 60 * 1000);
}

/**
 * Check if timestamp is in the past
 */
export function isPast(timestamp: TimestampInput): boolean {
  const date = parseUtc(timestamp);
  return date.getTime() < Date.now();
}

/**
 * Check if timestamp is in the future
 */
export function isFuture(timestamp: TimestampInput): boolean {
  const date = parseUtc(timestamp);
  return date.getTime() > Date.now();
}

/**
 * Compare two timestamps
 * Returns: -1 if first < second, 0 if equal, 1 if first > second
 */
export function compareTimestamps(first: TimestampInput, second: TimestampInput): number {
  const firstDate = parseUtc(first);
  const secondDate = parseUtc(second);

  if (firstDate.getTime() < secondDate.getTime()) return -1;
  if (firstDate.getTime() > secondDate.getTime()) return 1;
  return 0;
}

/**
 * Get Unix timestamp (seconds since epoch)
 */
export function toUnix(timestamp: TimestampInput = new Date()): number {
  const date = parseUtc(timestamp);
  return Math.floor(date.getTime() / 1000);
}

/**
 * Create timestamp from Unix timestamp
 */
export function fromUnix(unix: number): string {
  return new Date(unix * 1000).toISOString();
}

/**
 * Truncate timestamp to seconds (remove milliseconds)
 */
export function truncateToSeconds(timestamp: TimestampInput): string {
  const iso = toUtcIso(timestamp);
  return `${iso.substring(0, 19)}Z`;
}

/**
 * Extract date part from timestamp (YYYY-MM-DD)
 */
export function toDatePart(timestamp: TimestampInput): string {
  const iso = toUtcIso(timestamp);
  return iso.substring(0, 10);
}

/**
 * Extract time part from timestamp (HH:mm:ss)
 */
export function toTimePart(timestamp: TimestampInput): string {
  const iso = toUtcIso(timestamp);
  return iso.substring(11, 19);
}

/**
 * Format timestamp for display in user's local timezone
 * (But keep storage in UTC)
 */
export function toLocalDisplay(timestamp: TimestampInput): string {
  const date = parseUtc(timestamp);
  return date.toLocaleString();
}

/**
 * Validate timestamp is in ISO-8601 UTC format
 */
export function isValidIsoUtc(timestamp: string): boolean {
  if (!timestamp) return false;

  // ISO-8601 UTC format should end with 'Z'
  if (!timestamp.endsWith('Z') && !timestamp.includes('+00:00')) {
    return false;
  }

  return isValidTimestamp(timestamp);
}

/**
 * Ensure timestamp is in UTC ISO-8601 format
 * If not, convert it
 */
export function ensureUtcIso(timestamp: TimestampInput): string {
  if (typeof timestamp === 'string' && isValidIsoUtc(timestamp)) {
    return timestamp;
  }

  return toUtcIso(timestamp);
}

/**
 * Timer class for measuring operation duration
 */
export class Timer {
  private startTime: number;
  private endTime?: number;

  constructor() {
    this.startTime = Date.now();
  }

  /**
   * Stop the timer
   */
  stop(): number {
    this.endTime = Date.now();
    return this.elapsed();
  }

  /**
   * Get elapsed milliseconds
   */
  elapsed(): number {
    const end = this.endTime || Date.now();
    return end - this.startTime;
  }

  /**
   * Get elapsed as human-readable string
   */
  elapsedFormatted(): string {
    return formatDuration(this.elapsed());
  }

  /**
   * Get start timestamp as UTC ISO
   */
  getStartTimestamp(): string {
    return new Date(this.startTime).toISOString();
  }

  /**
   * Get end timestamp as UTC ISO (if stopped)
   */
  getEndTimestamp(): string | null {
    if (!this.endTime) return null;
    return new Date(this.endTime).toISOString();
  }
}

/**
 * Create a new timer
 */
export function startTimer(): Timer {
  return new Timer();
}
