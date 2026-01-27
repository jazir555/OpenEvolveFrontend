/**
 * Sanitizes error messages to remove sensitive credential information
 */
/**
 * Safely extracts error message without exposing sensitive data
 */
export declare function getSafeErrorMessage(error: unknown): string;
/**
 * Sanitizes error stack traces
 */
export declare function sanitizeErrorStack(error: Error | undefined): string | undefined;
/**
 * Sanitizes error messages
 */
export declare function sanitizeErrorMessage(message: string): string;
//# sourceMappingURL=error-sanitizer.d.ts.map