/**
 * Monitoring formatting helpers.
 *
 * The OpenEvolve monitoring endpoints return loosely-typed metric bags
 * (`Record<string, number>` / `Record<string, unknown>`) where the unit is only
 * encoded in the key name (`cpu_percent`, `memory_bytes`, `uptime_seconds`, ...).
 * These helpers infer a sensible unit from the key and reuse the existing app
 * utilities so monitoring output looks like the rest of the studio.
 */

import { formatDateTime, formatDuration, formatRelativeTime } from '@/utils/date';
import { formatBytes, formatPercentage } from '@/utils/format';

/** Placeholder used whenever a value is missing or unparsable. */
export const EMPTY_VALUE = '—';

/** Locale-aware number with at most two fraction digits. */
export function formatMetricNumber(value: number): string {
  if (!Number.isFinite(value)) return EMPTY_VALUE;
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/** True when a metric key looks like a 0-100 utilisation percentage. */
export function isPercentKey(key: string): boolean {
  const lower = key.toLowerCase();
  return lower.includes('percent') || lower.endsWith('_pct');
}

/**
 * Format a numeric metric, inferring its unit from the metric name
 * (`*_percent`, `*_bytes`, `*_ms`, `*_seconds`, ...).
 */
export function formatMetricValue(name: string, value: number): string {
  if (!Number.isFinite(value)) return EMPTY_VALUE;

  const key = name.toLowerCase();
  if (isPercentKey(key)) return formatPercentage(value);
  if (key.includes('bytes')) return formatBytes(value);
  if (key.endsWith('_ms') || key.includes('millisecond')) {
    return `${formatMetricNumber(value)} ms`;
  }
  if (key.includes('seconds') || key.includes('uptime') || key.includes('duration')) {
    return formatDuration(value);
  }
  return formatMetricNumber(value);
}

/** `cpu_percent` -> `Cpu Percent`. */
export function humanizeKey(key: string): string {
  const spaced = key.replace(/[_\-.]+/g, ' ').trim();
  if (spaced.length === 0) return key;
  return spaced.replace(/\b[a-z]/g, (char) => char.toUpperCase());
}

/** Absolute timestamp, tolerant of missing or invalid values. */
export function formatTimestamp(value?: string | null): string {
  if (!value) return EMPTY_VALUE;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return formatDateTime(date);
}

/** Relative timestamp ("3 minutes ago"), tolerant of missing values. */
export function formatSince(value?: string | null): string {
  if (!value) return EMPTY_VALUE;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return formatRelativeTime(date);
}

/** Uptime in seconds -> `1h 12m`. */
export function formatUptime(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return EMPTY_VALUE;
  }
  return formatDuration(seconds);
}

/** Service `execution_time` (seconds) -> `120 ms` / `3s`. */
export function formatExecutionTime(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return EMPTY_VALUE;
  }
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return formatDuration(seconds);
}

/** Best-effort rendering of an untyped `recent_metrics` field. */
export function formatUnknownValue(value: unknown): string {
  if (value === null || value === undefined) return EMPTY_VALUE;
  if (typeof value === 'number') return formatMetricNumber(value);
  if (typeof value === 'string') return value.length > 0 ? value : EMPTY_VALUE;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Tone for a 0-100 utilisation value (green / yellow / red thresholds). */
export function utilizationTone(value: number): 'green' | 'yellow' | 'red' {
  if (value >= 90) return 'red';
  if (value >= 75) return 'yellow';
  return 'green';
}

/**
 * Numeric entries of a loosely-typed metric bag (`system.system`, `workflow`),
 * dropping anything that is not a finite number so the render path stays safe
 * even if the backend adds string/`null` fields.
 */
export function toNumericEntries(
  bag?: Record<string, unknown> | null
): Array<[string, number]> {
  if (!bag) return [];
  return Object.entries(bag).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === 'number' && Number.isFinite(entry[1])
  );
}

/**
 * Entries of a nested metric bag (`recent_metrics`), normalising every value to
 * a plain record so the caller can always iterate its fields.
 */
export function toRecordEntries(
  bag?: Record<string, unknown> | null
): Array<[string, Record<string, unknown>]> {
  if (!bag) return [];
  return Object.entries(bag).map(([key, value]): [string, Record<string, unknown>] => [
    key,
    value !== null && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {},
  ]);
}
