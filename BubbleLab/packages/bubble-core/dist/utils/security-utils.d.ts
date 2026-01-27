/**
 * SECURITY UTILITIES
 *
 * Centralized security functions for tool bubbles.
 * Provides input sanitization, output encoding, rate limiting, and threat detection.
 *
 * Features:
 * - SQL injection prevention
 * - XSS prevention
 * - Path traversal prevention
 * - Command injection prevention
 * - Rate limiting
 * - Input validation
 * - Output sanitization
 */
/**
 * Sanitize SQL query to prevent SQL injection attacks
 * Uses a whitelist-based approach to only allow safe SQL patterns
 */
export declare function sanitizeSQLQuery(query: string): {
    isSafe: boolean;
    sanitized?: string;
    reason?: string;
};
/**
 * Validate and sanitize file path to prevent path traversal attacks
 */
export declare function sanitizeFilePath(filePath: string, allowedPaths?: string[]): {
    isSafe: boolean;
    sanitized?: string;
    reason?: string;
};
/**
 * Sanitize HTML to prevent XSS attacks
 */
export declare function sanitizeHTML(html: string): string;
/**
 * Sanitize text output to prevent injection attacks
 */
export declare function sanitizeOutput(text: string): string;
/**
 * Validate email address format
 */
export declare function validateEmail(email: string): {
    isValid: boolean;
    reason?: string;
};
/**
 * Validate URL format and safety
 */
export declare function validateURL(url: string): {
    isValid: boolean;
    sanitized?: string;
    reason?: string;
};
/**
 * Rate limiter for preventing abuse
 */
export declare class RateLimiter {
    private requests;
    private readonly maxRequests;
    private readonly windowMs;
    constructor(maxRequests?: number, windowMs?: number);
    /**
     * Check if a request should be rate limited
     */
    checkLimit(identifier: string): {
        allowed: boolean;
        remaining: number;
        resetTime: number;
    };
    /**
     * Reset rate limit for a specific identifier
     */
    reset(identifier: string): void;
    /**
     * Clear all rate limit data
     */
    clear(): void;
}
/**
 * Input size limiter to prevent DoS attacks
 */
export declare class SizeLimiter {
    private readonly maxSize;
    constructor(maxSize?: number);
    /**
     * Check if input size exceeds limit
     */
    checkSize(input: string): {
        withinLimit: boolean;
        size: number;
        maxSize: number;
    };
    /**
     * Truncate input to maximum size
     */
    truncate(input: string): string;
}
/**
 * Timeout manager for operations
 */
export declare class TimeoutManager {
    /**
     * Execute a function with a timeout
     */
    static withTimeout<T>(fn: () => Promise<T>, timeoutMs: number, errorMessage?: string): Promise<T>;
}
/**
 * Command injection prevention
 */
export declare function sanitizeCommand(command: string): {
    isSafe: boolean;
    reason?: string;
};
/**
 * Validate and sanitize JSON data
 */
export declare function sanitizeJSON(data: unknown): {
    isValid: boolean;
    sanitized?: unknown;
    reason?: string;
};
/**
 * Validate and sanitize regex pattern to prevent ReDoS attacks
 */
export declare function sanitizeRegex(pattern: string): {
    isSafe: boolean;
    reason?: string;
};
/**
 * Validate file type to prevent malicious file uploads
 */
export declare function validateFileType(filename: string, allowedExtensions: string[]): {
    isValid: boolean;
    reason?: string;
};
//# sourceMappingURL=security-utils.d.ts.map