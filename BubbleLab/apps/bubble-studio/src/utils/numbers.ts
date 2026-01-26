/**
 * Number Utilities
 * Number formatting and manipulation
 */

// Re-export common formatting functions from format.ts
export { formatNumber, formatCurrency, formatPercentage, formatBytes } from './format';

/**
 * Format number with thousand separators (alias)
 */
export function formatNumberLocale(num: number, locale = 'en-US'): string {
  return new Intl.NumberFormat(locale).format(num);
}

/**
 * Format number as currency with locale support (extended version)
 */
export function formatCurrencyLocale(
  amount: number,
  currency = 'USD',
  locale = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(amount);
}

/**
 * Format number with compact notation (K, M, B, T)
 */
export function formatCompactNumber(num: number): string {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(num);
}

/**
 * Parse string to number safely
 */
export function parseNumber(value: string, defaultValue = 0): number {
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

/**
 * Check if value is a valid number
 */
export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

/**
 * Check if value is finite
 */
export function isFiniteNumber(value: unknown): boolean {
  return isNumber(value) && isFinite(value);
}

/**
 * Check if value is integer
 */
export function isInteger(value: unknown): boolean {
  return isNumber(value) && Number.isInteger(value);
}

/**
 * Check if value is positive
 */
export function isPositive(value: number): boolean {
  return value > 0;
}

/**
 * Check if value is negative
 */
export function isNegative(value: number): boolean {
  return value < 0;
}

/**
 * Convert string to bytes
 */
export function parseBytes(value: string): number {
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const match = value.match(/^([\d.]+)\s*([A-Z]+)?$/i);

  if (!match) return 0;

  const num = parseFloat(match[1]);
  const unit = match[2]?.toUpperCase() || 'B';
  const exponent = units.indexOf(unit);

  return exponent >= 0 ? num * Math.pow(1024, exponent) : 0;
}

/**
 * Convert bytes to human readable
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Generate range of numbers
 */
export function range(start: number, end: number, step = 1): number[] {
  const result: number[] = [];
  for (let i = start; i < end; i += step) {
    result.push(i);
  }
  return result;
}

/**
 * Pad number with zeros
 */
export function padZero(num: number, length = 2): string {
  return num.toString().padStart(length, '0');
}

/**
 * Convert to ordinal (1st, 2nd, 3rd, etc.)
 */
export function toOrdinal(num: number): string {
  const suffixes = ['th', 'st', 'nd', 'rd'];
  const value = num % 100;
  const suffix = suffixes[(value - 20) % 10] || suffixes[value] || suffixes[0];
  return `${num}${suffix}`;
}

/**
 * Round to nearest multiple
 */
export function roundToMultiple(num: number, multiple: number): number {
  return Math.round(num / multiple) * multiple;
}

/**
 * Floor to nearest multiple
 */
export function floorToMultiple(num: number, multiple: number): number {
  return Math.floor(num / multiple) * multiple;
}

/**
 * Ceil to nearest multiple
 */
export function ceilToMultiple(num: number, multiple: number): number {
  return Math.ceil(num / multiple) * multiple;
}
