/**
 * Common constants for Bubble implementations
 * Provides centralized configuration values and magic numbers
 */
/**
 * Default timeout values (in milliseconds)
 */
export const TIMEOUT = {
    /** Default timeout for HTTP requests */
    HTTP_REQUEST: 30000,
    /** Default timeout for database queries */
    DATABASE_QUERY: 30000,
    /** Default timeout for external API calls */
    EXTERNAL_API: 60000,
    /** Default timeout for file operations */
    FILE_OPERATION: 120000,
    /** Minimum allowed timeout */
    MIN: 1000,
    /** Maximum allowed timeout */
    MAX: 300000,
};
/**
 * Default retry configuration
 */
export const RETRY = {
    /** Default number of retry attempts */
    MAX_ATTEMPTS: 3,
    /** Base delay for exponential backoff (ms) */
    BASE_DELAY_MS: 1000,
    /** Maximum delay between retries (ms) */
    MAX_DELAY_MS: 30000,
    /** Exponential backoff multiplier */
    BACKOFF_MULTIPLIER: 2,
    /** Jitter amount (0-1) to prevent thundering herd */
    JITTER_AMOUNT: 0.1,
};
/**
 * Default pagination values
 */
export const PAGINATION = {
    /** Default page size */
    DEFAULT_LIMIT: 50,
    /** Maximum page size */
    MAX_LIMIT: 1000,
    /** Minimum page size */
    MIN_LIMIT: 1,
    /** Default offset */
    DEFAULT_OFFSET: 0,
};
/**
 * File size limits (in bytes)
 */
export const FILE_SIZE = {
    /** Maximum upload size for most files (10MB) */
    MAX_UPLOAD: 10 * 1024 * 1024,
    /** Maximum upload size for large files (50MB) */
    MAX_UPLOAD_LARGE: 50 * 1024 * 1024,
    /** Maximum upload size for documents (5MB) */
    MAX_UPLOAD_DOCUMENT: 5 * 1024 * 1024,
    /** Maximum upload size for images (2MB) */
    MAX_UPLOAD_IMAGE: 2 * 1024 * 1024,
    /** Chunk size for streaming (64KB) */
    CHUNK_SIZE: 64 * 1024,
};
/**
 * HTTP status codes
 */
export const HTTP_STATUS = {
    /** Continue */
    CONTINUE: 100,
    /** Switching Protocols */
    SWITCHING_PROTOCOLS: 101,
    /** OK */
    OK: 200,
    /** Created */
    CREATED: 201,
    /** Accepted */
    ACCEPTED: 202,
    /** No Content */
    NO_CONTENT: 204,
    /** Moved Permanently */
    MOVED_PERMANENTLY: 301,
    /** Found */
    FOUND: 302,
    /** Not Modified */
    NOT_MODIFIED: 304,
    /** Bad Request */
    BAD_REQUEST: 400,
    /** Unauthorized */
    UNAUTHORIZED: 401,
    /** Forbidden */
    FORBIDDEN: 403,
    /** Not Found */
    NOT_FOUND: 404,
    /** Method Not Allowed */
    METHOD_NOT_ALLOWED: 405,
    /** Request Timeout */
    REQUEST_TIMEOUT: 408,
    /** Conflict */
    CONFLICT: 409,
    /** Too Many Requests */
    TOO_MANY_REQUESTS: 429,
    /** Internal Server Error */
    INTERNAL_SERVER_ERROR: 500,
    /** Not Implemented */
    NOT_IMPLEMENTED: 501,
    /** Bad Gateway */
    BAD_GATEWAY: 502,
    /** Service Unavailable */
    SERVICE_UNAVAILABLE: 503,
    /** Gateway Timeout */
    GATEWAY_TIMEOUT: 504,
};
/**
 * Regular expression patterns
 */
export const REGEX = {
    /** Email validation */
    EMAIL: /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/,
    /** URL validation */
    URL: /^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/,
    /** ISO 8601 timestamp */
    ISO_TIMESTAMP: /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$/,
    /** UUID v4 */
    UUID_V4: /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    /** Hex color code */
    HEX_COLOR: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
    /** Phone number (E.164 format) */
    PHONE_E164: /^\+[1-9]\d{1,14}$/,
    /** Alphanumeric with spaces */
    ALPHANUMERIC_SPACE: /^[a-zA-Z0-9\s]+$/,
    /** Safe filename */
    SAFE_FILENAME: /^[a-zA-Z0-9._-]+$/,
    /** Channel ID (e.g., Slack) */
    CHANNEL_ID: /^[A-Z0-9]+$/,
};
/**
 * String length limits
 */
export const STRING_LENGTH = {
    /** Maximum email length (RFC 5321) */
    MAX_EMAIL: 254,
    /** Maximum URL length */
    MAX_URL: 2048,
    /** Maximum filename length */
    MAX_FILENAME: 255,
    /** Maximum file path length */
    MAX_FILEPATH: 4096,
    /** Maximum description length */
    MAX_DESCRIPTION: 5000,
    /** Maximum name length */
    MAX_NAME: 255,
    /** Maximum message length */
    MAX_MESSAGE: 10000,
    /** Minimum password length */
    MIN_PASSWORD: 8,
    /** Maximum password length */
    MAX_PASSWORD: 128,
};
/**
 * Date/time formats
 */
export const DATE_FORMAT = {
    /** ISO 8601 format */
    ISO_8601: "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
    /** Human-readable date */
    DATE_ONLY: 'yyyy-MM-dd',
    /** Human-readable time */
    TIME_ONLY: 'HH:mm:ss',
    /** Human-readable datetime */
    DATETIME: 'yyyy-MM-dd HH:mm:ss',
};
/**
 * Cache TTL values (in milliseconds)
 */
export const CACHE_TTL = {
    /** Very short cache (1 minute) */
    VERY_SHORT: 60 * 1000,
    /** Short cache (5 minutes) */
    SHORT: 5 * 60 * 1000,
    /** Medium cache (15 minutes) */
    MEDIUM: 15 * 60 * 1000,
    /** Long cache (1 hour) */
    LONG: 60 * 60 * 1000,
    /** Very long cache (24 hours) */
    VERY_LONG: 24 * 60 * 60 * 1000,
    /** Default cache TTL */
    DEFAULT: 5 * 60 * 1000,
};
/**
 * Rate limiting values
 */
export const RATE_LIMIT = {
    /** Default rate limit (requests per minute) */
    DEFAULT_RPM: 60,
    /** Default rate limit (requests per second) */
    DEFAULT_RPS: 10,
    /** Strict rate limit (requests per minute) */
    STRICT_RPM: 10,
    /** Permissive rate limit (requests per minute) */
    PERMISSIVE_RPM: 600,
};
/**
 * Connection pool configuration
 */
export const CONNECTION_POOL = {
    /** Minimum pool size */
    min: 2,
    /** Maximum pool size */
    max: 10,
    /** Connection timeout (ms) */
    acquireTimeoutMillis: 10000,
    /** Idle timeout (ms) */
    idleTimeoutMillis: 30000,
    /** Cleanup interval (ms) */
    evictionRunIntervalMillis: 60000,
};
/**
 * Compression settings
 */
export const COMPRESSION = {
    /** Minimum size to compress (bytes) */
    MIN_SIZE: 10240, // 10KB
    /** Compression level (0-9) */
    LEVEL: 6,
    /** Gzip compression threshold */
    GZIP_THRESHOLD: 10240, // 10KB
};
/**
 * Batch operation sizes
 */
export const BATCH_SIZE = {
    /** Default batch size for bulk operations */
    DEFAULT: 100,
    /** Maximum batch size for bulk operations */
    MAX: 1000,
    /** Minimum batch size */
    MIN: 1,
    /** Batch size for database operations */
    DATABASE: 500,
    /** Batch size for API calls */
    API: 100,
};
/**
 * Security constants
 */
export const SECURITY = {
    /** Maximum login attempts */
    MAX_LOGIN_ATTEMPTS: 5,
    /** Account lockout duration (ms) */
    LOCKOUT_DURATION: 15 * 60 * 1000, // 15 minutes
    /** Token expiration (ms) */
    TOKEN_EXPIRATION: 60 * 60 * 1000, // 1 hour
    /** Refresh token expiration (ms) */
    REFRESH_TOKEN_EXPIRATION: 7 * 24 * 60 * 60 * 1000, // 7 days
    /** Minimum password strength score */
    MIN_PASSWORD_STRENGTH: 2,
    /** Salt rounds for bcrypt */
    BCRYPT_SALT_ROUNDS: 10,
};
/**
 * Environment names
 */
export const ENVIRONMENT = {
    DEVELOPMENT: 'development',
    TESTING: 'testing',
    STAGING: 'staging',
    PRODUCTION: 'production',
};
/**
 * Log levels
 */
export const LOG_LEVEL = {
    DEBUG: 'debug',
    INFO: 'info',
    WARN: 'warn',
    ERROR: 'error',
    FATAL: 'fatal',
};
/**
 * Common HTTP headers
 */
export const HTTP_HEADER = {
    /** Content-Type header */
    CONTENT_TYPE: 'Content-Type',
    /** Authorization header */
    AUTHORIZATION: 'Authorization',
    /** User-Agent header */
    USER_AGENT: 'User-Agent',
    /** Accept header */
    ACCEPT: 'Accept',
    /** Accept-Encoding header */
    ACCEPT_ENCODING: 'Accept-Encoding',
    /** X-Request-ID header */
    X_REQUEST_ID: 'X-Request-ID',
    /** X-Correlation-ID header */
    X_CORRELATION_ID: 'X-Correlation-ID',
};
/**
 * Common MIME types
 */
export const MIME_TYPE = {
    /** JSON */
    JSON: 'application/json',
    /** Form-encoded */
    FORM_URLENCODED: 'application/x-www-form-urlencoded',
    /** Multipart form data */
    MULTIPART_FORM_DATA: 'multipart/form-data',
    /** Plain text */
    TEXT: 'text/plain',
    /** HTML */
    HTML: 'text/html',
    /** XML */
    XML: 'application/xml',
    /** PDF */
    PDF: 'application/pdf',
    /** ZIP */
    ZIP: 'application/zip',
    /** JPEG image */
    JPEG: 'image/jpeg',
    /** PNG image */
    PNG: 'image/png',
    /** GIF image */
    GIF: 'image/gif',
    /** SVG image */
    SVG: 'image/svg+xml',
    /** MP4 video */
    MP4: 'video/mp4',
    /** MP3 audio */
    MP3: 'audio/mpeg',
};
/**
 * Error codes
 */
export const ERROR_CODE = {
    /** Validation error */
    VALIDATION_ERROR: 'VALIDATION_ERROR',
    /** Authentication error */
    AUTH_ERROR: 'AUTH_ERROR',
    /** Authorization error */
    AUTHZ_ERROR: 'AUTHZ_ERROR',
    /** Not found error */
    NOT_FOUND: 'NOT_FOUND',
    /** Conflict error */
    CONFLICT: 'CONFLICT',
    /** Rate limit error */
    RATE_LIMIT: 'RATE_LIMIT',
    /** Network error */
    NETWORK_ERROR: 'NETWORK_ERROR',
    /** Timeout error */
    TIMEOUT: 'TIMEOUT',
    /** Configuration error */
    CONFIG_ERROR: 'CONFIG_ERROR',
    /** External service error */
    EXTERNAL_SERVICE_ERROR: 'EXTERNAL_SERVICE_ERROR',
    /** Internal server error */
    INTERNAL_ERROR: 'INTERNAL_ERROR',
};
/**
 * Currency codes (ISO 4217)
 */
export const CURRENCY = {
    USD: 'USD',
    EUR: 'EUR',
    GBP: 'GBP',
    JPY: 'JPY',
    CAD: 'CAD',
    AUD: 'AUD',
    CHF: 'CHF',
    CNY: 'CNY',
    INR: 'INR',
};
/**
 * Timezone names
 */
export const TIMEZONE = {
    UTC: 'UTC',
    /** Eastern Time */
    ET: 'America/New_York',
    /** Pacific Time */
    PT: 'America/Los_Angeles',
    /** Central European Time */
    CET: 'Europe/Paris',
    /** Eastern European Time */
    EET: 'Europe/Helsinki',
    /** Japan Standard Time */
    JST: 'Asia/Tokyo',
    /** China Standard Time */
    CST: 'Asia/Shanghai',
    /** Australian Eastern Time */
    AET: 'Australia/Sydney',
};
/**
 * Helper function to get timeout value with fallback
 */
export function getTimeout(customTimeout, defaultTimeout = TIMEOUT.HTTP_REQUEST) {
    if (customTimeout !== undefined) {
        if (customTimeout < TIMEOUT.MIN) {
            console.warn(`Timeout ${customTimeout}ms is below minimum, using ${TIMEOUT.MIN}ms`);
            return TIMEOUT.MIN;
        }
        if (customTimeout > TIMEOUT.MAX) {
            console.warn(`Timeout ${customTimeout}ms is above maximum, using ${TIMEOUT.MAX}ms`);
            return TIMEOUT.MAX;
        }
        return customTimeout;
    }
    return defaultTimeout;
}
/**
 * Helper function to validate pagination limit
 */
export function validateLimit(limit) {
    if (limit === undefined) {
        return PAGINATION.DEFAULT_LIMIT;
    }
    if (limit < PAGINATION.MIN_LIMIT) {
        console.warn(`Limit ${limit} is below minimum, using ${PAGINATION.MIN_LIMIT}`);
        return PAGINATION.MIN_LIMIT;
    }
    if (limit > PAGINATION.MAX_LIMIT) {
        console.warn(`Limit ${limit} is above maximum, using ${PAGINATION.MAX_LIMIT}`);
        return PAGINATION.MAX_LIMIT;
    }
    return limit;
}
//# sourceMappingURL=constants.js.map