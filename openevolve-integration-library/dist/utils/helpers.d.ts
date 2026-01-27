import type { ValidationResult, ParameterSchema } from '../api/types';
export declare function validateInputs(inputs: any, schema: ParameterSchema): ValidationResult;
export declare function deepMerge<T extends object>(target: T, source: Partial<T>): T;
export declare function generateId(): string;
export declare function formatDuration(ms: number): string;
export declare function retryWithBackoff<T>(fn: () => Promise<T>, maxRetries?: number, baseDelay?: number, shouldRetry?: (error: any) => boolean, onRetry?: (error: any, attempt: number, delay: number) => void): Promise<T>;
export declare function sleep(ms: number): Promise<void>;
export declare function debounce<T extends (...args: any[]) => any>(fn: T, delay: number): (...args: Parameters<T>) => void;
export declare function throttle<T extends (...args: any[]) => any>(fn: T, limit: number): (...args: Parameters<T>) => void;
export declare function parseDuration(duration: string): number;
export declare function isPlainObject(value: any): boolean;
export declare function deepClone<T>(obj: T): T;
export declare function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K>;
export declare function omit<T extends object, K extends keyof T>(obj: T, keys: K[]): Omit<T, K>;
//# sourceMappingURL=helpers.d.ts.map