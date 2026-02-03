/**
 * Array Utilities
 * Helper functions for array manipulation
 */

/**
 * Remove duplicates from array
 */
export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array));
}

/**
 * Group array by key
 */
export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((result, item) => {
    const groupKey = String(item[key]);
    if (!result[groupKey]) {
      result[groupKey] = [];
    }
    result[groupKey].push(item);
    return result;
  }, {} as Record<string, T[]>);
}

/**
 * Chunk array into smaller arrays
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

/**
 * Shuffle array
 */
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Sort array by key
 */
export function sortBy<T>(array: T[], key: keyof T, direction: 'asc' | 'desc' = 'asc'): T[] {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];

    if (aVal < bVal) return direction === 'asc' ? -1 : 1;
    if (aVal > bVal) return direction === 'asc' ? 1 : -1;
    return 0;
  });
}

/**
 * Find intersection of two arrays
 */
export function intersection<T>(a: T[], b: T[]): T[] {
  return a.filter((item) => b.includes(item));
}

/**
 * Find difference of two arrays
 */
export function difference<T>(a: T[], b: T[]): T[] {
  return a.filter((item) => !b.includes(item));
}

/**
 * Flatten nested array
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce((acc: T[], item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : [item]);
  }, [] as T[]);
}

/**
 * Partition array by predicate
 */
export function partition<T>(array: T[], predicate: (item: T) => boolean): [T[], T[]] {
  return array.reduce(
    (acc, item) => {
      acc[predicate(item) ? 0 : 1].push(item);
      return acc;
    },
    [[], []] as [T[], T[]]
  );
}

/**
 * Get random item from array
 */
export function sample<T>(array: T[]): T | undefined {
  return array[Math.floor(Math.random() * array.length)];
}

/**
 * Get n random items from array
 */
export function sampleSize<T>(array: T[], n: number): T[] {
  const shuffled = shuffle(array);
  return shuffled.slice(0, n);
}

/**
 * Find item by predicate
 */
export function find<T>(array: T[], predicate: (item: T) => boolean): T | undefined {
  return array.find(predicate);
}

/**
 * Check if array includes item
 */
export function includes<T>(array: T[], item: T): boolean {
  return array.includes(item);
}

/**
 * Sum array of numbers
 */
export function sum(array: number[]): number {
  return array.reduce((acc, val) => acc + val, 0);
}

/**
 * Get average of array of numbers
 */
export function average(array: number[]): number {
  return sum(array) / array.length;
}

/**
 * Get max value in array
 */
export function max(array: number[]): number {
  return Math.max(...array);
}

/**
 * Get min value in array
 */
export function min(array: number[]): number {
  return Math.min(...array);
}
