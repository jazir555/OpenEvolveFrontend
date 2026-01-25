import { ClassValue } from 'clsx';
/**
 * Utility function to merge Tailwind CSS classes
 */
export declare function cn(...inputs: ClassValue[]): string;
/**
 * Format a date to a localized string
 */
export declare function formatDate(date: string | Date): string;
/**
 * Format a number with units
 */
export declare function formatNumber(value: number, units?: number): string;
/**
 * Truncate text to a specified length
 */
export declare function truncate(text: string, length: number): string;
/**
 * Generate a unique ID
 */
export declare function generateId(): string;
/**
 * Sleep for a specified number of milliseconds
 */
export declare function sleep(ms: number): Promise<void>;
/**
 * Debounce a function
 */
export declare function debounce<T extends (...args: unknown[]) => unknown>(func: T, wait: number): (...args: Parameters<T>) => void;
/**
 * Throttle a function
 */
export declare function throttle<T extends (...args: unknown[]) => unknown>(func: T, limit: number): (...args: Parameters<T>) => void;
