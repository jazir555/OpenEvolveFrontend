/**
 * Utility functions for OpenEvolve plugin components
 */
/**
 * cn - Class name utility for merging Tailwind classes
 * Combines clsx-like functionality with conditional classes
 */
export declare function cn(...classes: (string | undefined | null | false)[]): string;
/**
 * formatNumber - Format numbers with separators
 */
export declare function formatNumber(num: number): string;
/**
 * formatBytes - Format bytes to human readable size
 */
export declare function formatBytes(bytes: number, decimals?: number): string;
/**
 * formatDate - Format date to readable string
 */
export declare function formatDate(date: Date | string): string;
/**
 * truncate - Truncate string to max length
 */
export declare function truncate(str: string, maxLength: number): string;
/**
 * debounce - Debounce function execution
 */
export declare function debounce<T extends (...args: any[]) => any>(func: T, wait: number): (...args: Parameters<T>) => void;
/**
 * throttle - Throttle function execution
 */
export declare function throttle<T extends (...args: any[]) => any>(func: T, limit: number): (...args: Parameters<T>) => void;
/**
 * generateId - Generate unique ID
 */
export declare function generateId(): string;
/**
 * isValidUrl - Validate URL format
 */
export declare function isValidUrl(url: string): boolean;
/**
 * deepClone - Deep clone object
 */
export declare function deepClone<T>(obj: T): T;
/**
 * isEmpty - Check if value is empty
 */
export declare function isEmpty(value: any): boolean;
