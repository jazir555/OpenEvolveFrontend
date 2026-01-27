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
import { z } from 'zod';
/**
 * Sanitize SQL query to prevent SQL injection attacks
 * Uses a whitelist-based approach to only allow safe SQL patterns
 */
export function sanitizeSQLQuery(query) {
    if (!query || typeof query !== 'string') {
        return { isSafe: false, reason: 'Query must be a non-empty string' };
    }
    const trimmedQuery = query.trim();
    // Check for dangerous SQL patterns
    const dangerousPatterns = [
        // SQL comments (used to bypass filters)
        /--/g,
        /\/\*/g,
        /\*\//g,
        /;/g, // Statement separator (allows multiple queries)
        // UNION-based injection
        /\bUNION\s+SELECT/gi,
        /\bUNION\s+ALL\s+SELECT/gi,
        // Time-based injection
        /\bWAITFOR\s+DELAY/gi,
        /\bSLEEP\s*\(/gi,
        /\bBENCHMARK\s*\(/gi,
        // Boolean-based injection
        /\bAND\s+1\s*=\s*1/gi,
        /\bOR\s+1\s*=\s*1/gi,
        // Stacked queries
        /\bEXEC\s*\(/gi,
        /\bEXECUTE\s*\(/gi,
        // Stored procedures
        /\bXP_CMDSHELL/gi,
        /\bSP_OACREATE/gi,
        // Hex encoding
        /0x[0-9a-f]+/gi,
        // Char encoding
        /\bCHAR\s*\(/gi,
        /\bASCII\s*\(/gi,
    ];
    for (const pattern of dangerousPatterns) {
        if (pattern.test(trimmedQuery)) {
            return {
                isSafe: false,
                reason: `Query contains dangerous SQL pattern: ${pattern.source}`,
            };
        }
    }
    // Additional checks for parameterized queries
    // Check for quote escaping attempts
    const singleQuoteCount = (trimmedQuery.match(/'/g) || []).length;
    if (singleQuoteCount % 2 !== 0 && !trimmedQuery.includes('?') && !trimmedQuery.includes('$')) {
        return {
            isSafe: false,
            reason: 'Query contains unbalanced quotes (possible injection attempt)',
        };
    }
    // Check for unusual whitespace patterns (used to bypass filters)
    if (/\s{10,}/.test(trimmedQuery)) {
        return {
            isSafe: false,
            reason: 'Query contains excessive whitespace (possible obfuscation)',
        };
    }
    // Validate query structure
    // Must start with a safe keyword
    const safeStartPatterns = [
        /^\s*SELECT\b/i,
        /^\s*WITH\b/i,
        /^\s*EXPLAIN\b/i,
        /^\s*ANALYZE\b/i,
        /^\s*SHOW\b/i,
        /^\s*DESCRIBE\b/i,
        /^\s*DESC\b/i,
    ];
    const startsSafely = safeStartPatterns.some((pattern) => pattern.test(trimmedQuery));
    if (!startsSafely) {
        return {
            isSafe: false,
            reason: 'Query must start with SELECT, WITH, EXPLAIN, ANALYZE, SHOW, or DESCRIBE',
        };
    }
    // Check for database version disclosure attempts
    if (/\b@@VERSION\b/i.test(trimmedQuery) || /\bVERSION\s*\(\s*\)\b/i.test(trimmedQuery)) {
        return {
            isSafe: false,
            reason: 'Query attempts to disclose database version',
        };
    }
    // If we made it here, the query is considered safe
    return { isSafe: true, sanitized: trimmedQuery };
}
/**
 * Validate and sanitize file path to prevent path traversal attacks
 */
export function sanitizeFilePath(filePath, allowedPaths = []) {
    if (!filePath || typeof filePath !== 'string') {
        return { isSafe: false, reason: 'File path must be a non-empty string' };
    }
    // Check for path traversal attempts
    const traversalPatterns = [
        /\.\.\//, // Unix traversal
        /\.\.\\/, // Windows traversal
        /\.\.%2f/i, // URL encoded traversal
        /\.\.%5c/i, // URL encoded traversal
        /%2e%2e/i, // Double dot URL encoded
        /\.\.{2,}/, // Multiple dots
    ];
    for (const pattern of traversalPatterns) {
        if (pattern.test(filePath)) {
            return {
                isSafe: false,
                reason: 'File path contains path traversal sequence',
            };
        }
    }
    // Check for absolute paths (unless explicitly allowed)
    const isAbsolute = filePath.startsWith('/') || /^[a-zA-Z]:/.test(filePath);
    if (isAbsolute && allowedPaths.length === 0) {
        return {
            isSafe: false,
            reason: 'Absolute paths are not allowed',
        };
    }
    // If allowed paths are specified, validate against them
    if (allowedPaths.length > 0) {
        const normalizedPath = filePath.replace(/\\/g, '/');
        const isAllowed = allowedPaths.some((allowed) => {
            const normalizedAllowed = allowed.replace(/\\/g, '/');
            return normalizedPath.startsWith(normalizedAllowed);
        });
        if (!isAllowed) {
            return {
                isSafe: false,
                reason: 'File path is not in the allowed directory list',
            };
        }
    }
    // Check for suspicious characters
    const suspiciousChars = /[<>:"|?*\x00-\x1f]/;
    if (suspiciousChars.test(filePath)) {
        return {
            isSafe: false,
            reason: 'File path contains invalid or suspicious characters',
        };
    }
    // Check for null bytes
    if (filePath.includes('\0')) {
        return {
            isSafe: false,
            reason: 'File path contains null bytes',
        };
    }
    return { isSafe: true, sanitized: filePath };
}
/**
 * Sanitize HTML to prevent XSS attacks
 */
export function sanitizeHTML(html) {
    if (!html || typeof html !== 'string') {
        return '';
    }
    return html
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;')
        .replace(/\//g, '&#x2F;');
}
/**
 * Sanitize text output to prevent injection attacks
 */
export function sanitizeOutput(text) {
    if (!text || typeof text !== 'string') {
        return '';
    }
    // Remove null bytes and control characters (except newline, tab, carriage return)
    return text
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
        .replace(/\0/g, '');
}
/**
 * Validate email address format
 */
export function validateEmail(email) {
    if (!email || typeof email !== 'string') {
        return { isValid: false, reason: 'Email must be a non-empty string' };
    }
    // Basic email format validation
    const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
    if (!emailRegex.test(email)) {
        return { isValid: false, reason: 'Invalid email format' };
    }
    // Check for suspicious patterns
    if (email.includes('..')) {
        return { isValid: false, reason: 'Email contains consecutive dots' };
    }
    if (email.startsWith('.') || email.endsWith('.')) {
        return { isValid: false, reason: 'Email starts or ends with a dot' };
    }
    if (email.length > 254) {
        return { isValid: false, reason: 'Email is too long (max 254 characters)' };
    }
    const [localPart, domain] = email.split('@');
    if (localPart.length > 64) {
        return { isValid: false, reason: 'Local part is too long (max 64 characters)' };
    }
    if (domain.length > 253) {
        return { isValid: false, reason: 'Domain is too long (max 253 characters)' };
    }
    return { isValid: true };
}
/**
 * Validate URL format and safety
 */
export function validateURL(url) {
    if (!url || typeof url !== 'string') {
        return { isValid: false, reason: 'URL must be a non-empty string' };
    }
    try {
        const parsed = new URL(url);
        // Only allow http and https protocols
        if (!['http:', 'https:'].includes(parsed.protocol)) {
            return {
                isValid: false,
                reason: `Protocol ${parsed.protocol} is not allowed. Only http and https are supported.`,
            };
        }
        // Check for internal IP addresses in URL
        const hostname = parsed.hostname;
        const internalIPPatterns = [
            /^127\./, // Loopback
            /^10\./, // Private Class A
            /^172\.(1[6-9]|2\d|3[01])\./, // Private Class B
            /^192\.168\./, // Private Class C
            /^localhost$/i,
            /^0\./, // Current network
        ];
        if (internalIPPatterns.some((pattern) => pattern.test(hostname))) {
            return {
                isValid: false,
                reason: 'URL contains internal or private IP address',
            };
        }
        // Check for file:// protocol
        if (url.toLowerCase().startsWith('file://')) {
            return {
                isValid: false,
                reason: 'file:// protocol is not allowed',
            };
        }
        // Check for javascript: protocol
        if (url.toLowerCase().startsWith('javascript:')) {
            return {
                isValid: false,
                reason: 'javascript: protocol is not allowed',
            };
        }
        // Check for data: protocol
        if (url.toLowerCase().startsWith('data:')) {
            return {
                isValid: false,
                reason: 'data: protocol is not allowed',
            };
        }
        return { isValid: true, sanitized: parsed.href };
    }
    catch (error) {
        return {
            isValid: false,
            reason: 'Invalid URL format',
        };
    }
}
/**
 * Rate limiter for preventing abuse
 */
export class RateLimiter {
    requests = new Map();
    maxRequests;
    windowMs;
    constructor(maxRequests = 100, windowMs = 60000) {
        this.maxRequests = maxRequests;
        this.windowMs = windowMs;
    }
    /**
     * Check if a request should be rate limited
     */
    checkLimit(identifier) {
        const now = Date.now();
        const windowStart = now - this.windowMs;
        // Get existing requests for this identifier
        let timestamps = this.requests.get(identifier) || [];
        // Filter out old timestamps outside the window
        timestamps = timestamps.filter((timestamp) => timestamp > windowStart);
        // Check if limit exceeded
        if (timestamps.length >= this.maxRequests) {
            const oldestTimestamp = timestamps[0];
            const resetTime = oldestTimestamp + this.windowMs;
            return {
                allowed: false,
                remaining: 0,
                resetTime,
            };
        }
        // Add current request timestamp
        timestamps.push(now);
        this.requests.set(identifier, timestamps);
        return {
            allowed: true,
            remaining: this.maxRequests - timestamps.length,
            resetTime: now + this.windowMs,
        };
    }
    /**
     * Reset rate limit for a specific identifier
     */
    reset(identifier) {
        this.requests.delete(identifier);
    }
    /**
     * Clear all rate limit data
     */
    clear() {
        this.requests.clear();
    }
}
/**
 * Input size limiter to prevent DoS attacks
 */
export class SizeLimiter {
    maxSize;
    constructor(maxSize = 10 * 1024 * 1024) {
        this.maxSize = maxSize;
    }
    /**
     * Check if input size exceeds limit
     */
    checkSize(input) {
        const size = Buffer.byteLength(input, 'utf8');
        return {
            withinLimit: size <= this.maxSize,
            size,
            maxSize: this.maxSize,
        };
    }
    /**
     * Truncate input to maximum size
     */
    truncate(input) {
        const { size } = this.checkSize(input);
        if (size <= this.maxSize) {
            return input;
        }
        // Truncate to max size (in bytes, not characters)
        let truncated = '';
        let currentSize = 0;
        for (const char of input) {
            const charSize = Buffer.byteLength(char, 'utf8');
            if (currentSize + charSize > this.maxSize) {
                break;
            }
            truncated += char;
            currentSize += charSize;
        }
        return truncated;
    }
}
/**
 * Timeout manager for operations
 */
export class TimeoutManager {
    /**
     * Execute a function with a timeout
     */
    static async withTimeout(fn, timeoutMs, errorMessage = 'Operation timed out') {
        return Promise.race([
            fn(),
            new Promise((_, reject) => setTimeout(() => reject(new Error(errorMessage)), timeoutMs)),
        ]);
    }
}
/**
 * Command injection prevention
 */
export function sanitizeCommand(command) {
    if (!command || typeof command !== 'string') {
        return { isSafe: false, reason: 'Command must be a non-empty string' };
    }
    // Dangerous command patterns
    const dangerousPatterns = [
        /;/, // Command separator
        /\|/, // Pipe
        /&/, // Background execution
        /\$/, // Variable expansion
        /`/, // Command substitution
        /\$\(/, // Command substitution
        /</, // Input redirection
        />/, // Output redirection
        /\n/, // Newline (command separator)
        /\r/, // Carriage return
        /\t/, // Tab (command separator in some contexts)
        /\\/, // Escape character
    ];
    for (const pattern of dangerousPatterns) {
        if (pattern.test(command)) {
            return {
                isSafe: false,
                reason: `Command contains dangerous character: ${pattern.source}`,
            };
        }
    }
    return { isSafe: true };
}
/**
 * Validate and sanitize JSON data
 */
export function sanitizeJSON(data) {
    try {
        // Check for circular references
        const seen = new WeakSet();
        const detectCircular = (obj) => {
            if (obj && typeof obj === 'object') {
                if (seen.has(obj)) {
                    return true;
                }
                seen.add(obj);
                for (const value of Object.values(obj)) {
                    if (detectCircular(value)) {
                        return true;
                    }
                }
            }
            return false;
        };
        if (detectCircular(data)) {
            return {
                isValid: false,
                reason: 'JSON data contains circular references',
            };
        }
        // Validate with Zod
        const safeJSON = z.unknown().safeParse(data);
        if (!safeJSON.success) {
            return {
                isValid: false,
                reason: 'JSON data is invalid',
            };
        }
        return { isValid: true, sanitized: safeJSON.data };
    }
    catch (error) {
        return {
            isValid: false,
            reason: error instanceof Error ? error.message : 'Unknown error',
        };
    }
}
/**
 * Validate and sanitize regex pattern to prevent ReDoS attacks
 */
export function sanitizeRegex(pattern) {
    if (!pattern || typeof pattern !== 'string') {
        return { isSafe: false, reason: 'Pattern must be a non-empty string' };
    }
    // Check for dangerous regex patterns that can cause ReDoS
    const dangerousPatterns = [
        /\(\.\*\*\)\*/, // Catastrophic backtracking
        /\(\.\+\+\)\+/, // Catastrophic backtracking
        /\(([a-z]+)\+\1\)+/, // Nested quantifiers
        /.{10,}/, // Excessive wildcards
    ];
    for (const dangerousPattern of dangerousPatterns) {
        if (dangerousPattern.test(pattern)) {
            return {
                isSafe: false,
                reason: 'Pattern contains dangerous regex syntax that may cause ReDoS',
            };
        }
    }
    // Try to compile the pattern to check validity
    try {
        new RegExp(pattern);
    }
    catch (error) {
        return {
            isSafe: false,
            reason: 'Invalid regex pattern',
        };
    }
    return { isSafe: true };
}
/**
 * Validate file type to prevent malicious file uploads
 */
export function validateFileType(filename, allowedExtensions) {
    if (!filename || typeof filename !== 'string') {
        return { isValid: false, reason: 'Filename must be a non-empty string' };
    }
    const extension = filename.slice(filename.lastIndexOf('.')).toLowerCase();
    if (!extension) {
        return { isValid: false, reason: 'File has no extension' };
    }
    if (!allowedExtensions.includes(extension)) {
        return {
            isValid: false,
            reason: `File type ${extension} is not allowed. Allowed types: ${allowedExtensions.join(', ')}`,
        };
    }
    // Check for double extensions (used to hide executable files)
    const parts = filename.split('.');
    if (parts.length > 2) {
        // Check if the file has multiple executable extensions
        const executableExtensions = ['.exe', '.bat', '.cmd', '.sh', '.php', '.asp', '.jsp'];
        const hasExecutableExtension = parts.some((part) => executableExtensions.includes(`.${part.toLowerCase()}`));
        if (hasExecutableExtension) {
            return {
                isValid: false,
                reason: 'File has multiple extensions including executable type',
            };
        }
    }
    return { isValid: true };
}
//# sourceMappingURL=security-utils.js.map