/**
 * Common constants for Bubble implementations
 * Provides centralized configuration values and magic numbers
 */
/**
 * Default timeout values (in milliseconds)
 */
export declare const TIMEOUT: {
    /** Default timeout for HTTP requests */
    readonly HTTP_REQUEST: 30000;
    /** Default timeout for database queries */
    readonly DATABASE_QUERY: 30000;
    /** Default timeout for external API calls */
    readonly EXTERNAL_API: 60000;
    /** Default timeout for file operations */
    readonly FILE_OPERATION: 120000;
    /** Minimum allowed timeout */
    readonly MIN: 1000;
    /** Maximum allowed timeout */
    readonly MAX: 300000;
};
/**
 * Default retry configuration
 */
export declare const RETRY: {
    /** Default number of retry attempts */
    readonly MAX_ATTEMPTS: 3;
    /** Base delay for exponential backoff (ms) */
    readonly BASE_DELAY_MS: 1000;
    /** Maximum delay between retries (ms) */
    readonly MAX_DELAY_MS: 30000;
    /** Exponential backoff multiplier */
    readonly BACKOFF_MULTIPLIER: 2;
    /** Jitter amount (0-1) to prevent thundering herd */
    readonly JITTER_AMOUNT: 0.1;
};
/**
 * Default pagination values
 */
export declare const PAGINATION: {
    /** Default page size */
    readonly DEFAULT_LIMIT: 50;
    /** Maximum page size */
    readonly MAX_LIMIT: 1000;
    /** Minimum page size */
    readonly MIN_LIMIT: 1;
    /** Default offset */
    readonly DEFAULT_OFFSET: 0;
};
/**
 * File size limits (in bytes)
 */
export declare const FILE_SIZE: {
    /** Maximum upload size for most files (10MB) */
    readonly MAX_UPLOAD: number;
    /** Maximum upload size for large files (50MB) */
    readonly MAX_UPLOAD_LARGE: number;
    /** Maximum upload size for documents (5MB) */
    readonly MAX_UPLOAD_DOCUMENT: number;
    /** Maximum upload size for images (2MB) */
    readonly MAX_UPLOAD_IMAGE: number;
    /** Chunk size for streaming (64KB) */
    readonly CHUNK_SIZE: number;
};
/**
 * HTTP status codes
 */
export declare const HTTP_STATUS: {
    /** Continue */
    readonly CONTINUE: 100;
    /** Switching Protocols */
    readonly SWITCHING_PROTOCOLS: 101;
    /** OK */
    readonly OK: 200;
    /** Created */
    readonly CREATED: 201;
    /** Accepted */
    readonly ACCEPTED: 202;
    /** No Content */
    readonly NO_CONTENT: 204;
    /** Moved Permanently */
    readonly MOVED_PERMANENTLY: 301;
    /** Found */
    readonly FOUND: 302;
    /** Not Modified */
    readonly NOT_MODIFIED: 304;
    /** Bad Request */
    readonly BAD_REQUEST: 400;
    /** Unauthorized */
    readonly UNAUTHORIZED: 401;
    /** Forbidden */
    readonly FORBIDDEN: 403;
    /** Not Found */
    readonly NOT_FOUND: 404;
    /** Method Not Allowed */
    readonly METHOD_NOT_ALLOWED: 405;
    /** Request Timeout */
    readonly REQUEST_TIMEOUT: 408;
    /** Conflict */
    readonly CONFLICT: 409;
    /** Too Many Requests */
    readonly TOO_MANY_REQUESTS: 429;
    /** Internal Server Error */
    readonly INTERNAL_SERVER_ERROR: 500;
    /** Not Implemented */
    readonly NOT_IMPLEMENTED: 501;
    /** Bad Gateway */
    readonly BAD_GATEWAY: 502;
    /** Service Unavailable */
    readonly SERVICE_UNAVAILABLE: 503;
    /** Gateway Timeout */
    readonly GATEWAY_TIMEOUT: 504;
};
/**
 * Regular expression patterns
 */
export declare const REGEX: {
    /** Email validation */
    readonly EMAIL: RegExp;
    /** URL validation */
    readonly URL: RegExp;
    /** ISO 8601 timestamp */
    readonly ISO_TIMESTAMP: RegExp;
    /** UUID v4 */
    readonly UUID_V4: RegExp;
    /** Hex color code */
    readonly HEX_COLOR: RegExp;
    /** Phone number (E.164 format) */
    readonly PHONE_E164: RegExp;
    /** Alphanumeric with spaces */
    readonly ALPHANUMERIC_SPACE: RegExp;
    /** Safe filename */
    readonly SAFE_FILENAME: RegExp;
    /** Channel ID (e.g., Slack) */
    readonly CHANNEL_ID: RegExp;
};
/**
 * String length limits
 */
export declare const STRING_LENGTH: {
    /** Maximum email length (RFC 5321) */
    readonly MAX_EMAIL: 254;
    /** Maximum URL length */
    readonly MAX_URL: 2048;
    /** Maximum filename length */
    readonly MAX_FILENAME: 255;
    /** Maximum file path length */
    readonly MAX_FILEPATH: 4096;
    /** Maximum description length */
    readonly MAX_DESCRIPTION: 5000;
    /** Maximum name length */
    readonly MAX_NAME: 255;
    /** Maximum message length */
    readonly MAX_MESSAGE: 10000;
    /** Minimum password length */
    readonly MIN_PASSWORD: 8;
    /** Maximum password length */
    readonly MAX_PASSWORD: 128;
};
/**
 * Date/time formats
 */
export declare const DATE_FORMAT: {
    /** ISO 8601 format */
    readonly ISO_8601: "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'";
    /** Human-readable date */
    readonly DATE_ONLY: "yyyy-MM-dd";
    /** Human-readable time */
    readonly TIME_ONLY: "HH:mm:ss";
    /** Human-readable datetime */
    readonly DATETIME: "yyyy-MM-dd HH:mm:ss";
};
/**
 * Cache TTL values (in milliseconds)
 */
export declare const CACHE_TTL: {
    /** Very short cache (1 minute) */
    readonly VERY_SHORT: number;
    /** Short cache (5 minutes) */
    readonly SHORT: number;
    /** Medium cache (15 minutes) */
    readonly MEDIUM: number;
    /** Long cache (1 hour) */
    readonly LONG: number;
    /** Very long cache (24 hours) */
    readonly VERY_LONG: number;
    /** Default cache TTL */
    readonly DEFAULT: number;
};
/**
 * Rate limiting values
 */
export declare const RATE_LIMIT: {
    /** Default rate limit (requests per minute) */
    readonly DEFAULT_RPM: 60;
    /** Default rate limit (requests per second) */
    readonly DEFAULT_RPS: 10;
    /** Strict rate limit (requests per minute) */
    readonly STRICT_RPM: 10;
    /** Permissive rate limit (requests per minute) */
    readonly PERMISSIVE_RPM: 600;
};
/**
 * Connection pool configuration
 */
export declare const CONNECTION_POOL: {
    /** Minimum pool size */
    readonly min: 2;
    /** Maximum pool size */
    readonly max: 10;
    /** Connection timeout (ms) */
    readonly acquireTimeoutMillis: 10000;
    /** Idle timeout (ms) */
    readonly idleTimeoutMillis: 30000;
    /** Cleanup interval (ms) */
    readonly evictionRunIntervalMillis: 60000;
};
/**
 * Compression settings
 */
export declare const COMPRESSION: {
    /** Minimum size to compress (bytes) */
    readonly MIN_SIZE: 10240;
    /** Compression level (0-9) */
    readonly LEVEL: 6;
    /** Gzip compression threshold */
    readonly GZIP_THRESHOLD: 10240;
};
/**
 * Batch operation sizes
 */
export declare const BATCH_SIZE: {
    /** Default batch size for bulk operations */
    readonly DEFAULT: 100;
    /** Maximum batch size for bulk operations */
    readonly MAX: 1000;
    /** Minimum batch size */
    readonly MIN: 1;
    /** Batch size for database operations */
    readonly DATABASE: 500;
    /** Batch size for API calls */
    readonly API: 100;
};
/**
 * Security constants
 */
export declare const SECURITY: {
    /** Maximum login attempts */
    readonly MAX_LOGIN_ATTEMPTS: 5;
    /** Account lockout duration (ms) */
    readonly LOCKOUT_DURATION: number;
    /** Token expiration (ms) */
    readonly TOKEN_EXPIRATION: number;
    /** Refresh token expiration (ms) */
    readonly REFRESH_TOKEN_EXPIRATION: number;
    /** Minimum password strength score */
    readonly MIN_PASSWORD_STRENGTH: 2;
    /** Salt rounds for bcrypt */
    readonly BCRYPT_SALT_ROUNDS: 10;
};
/**
 * Environment names
 */
export declare const ENVIRONMENT: {
    readonly DEVELOPMENT: "development";
    readonly TESTING: "testing";
    readonly STAGING: "staging";
    readonly PRODUCTION: "production";
};
/**
 * Log levels
 */
export declare const LOG_LEVEL: {
    readonly DEBUG: "debug";
    readonly INFO: "info";
    readonly WARN: "warn";
    readonly ERROR: "error";
    readonly FATAL: "fatal";
};
/**
 * Common HTTP headers
 */
export declare const HTTP_HEADER: {
    /** Content-Type header */
    readonly CONTENT_TYPE: "Content-Type";
    /** Authorization header */
    readonly AUTHORIZATION: "Authorization";
    /** User-Agent header */
    readonly USER_AGENT: "User-Agent";
    /** Accept header */
    readonly ACCEPT: "Accept";
    /** Accept-Encoding header */
    readonly ACCEPT_ENCODING: "Accept-Encoding";
    /** X-Request-ID header */
    readonly X_REQUEST_ID: "X-Request-ID";
    /** X-Correlation-ID header */
    readonly X_CORRELATION_ID: "X-Correlation-ID";
};
/**
 * Common MIME types
 */
export declare const MIME_TYPE: {
    /** JSON */
    readonly JSON: "application/json";
    /** Form-encoded */
    readonly FORM_URLENCODED: "application/x-www-form-urlencoded";
    /** Multipart form data */
    readonly MULTIPART_FORM_DATA: "multipart/form-data";
    /** Plain text */
    readonly TEXT: "text/plain";
    /** HTML */
    readonly HTML: "text/html";
    /** XML */
    readonly XML: "application/xml";
    /** PDF */
    readonly PDF: "application/pdf";
    /** ZIP */
    readonly ZIP: "application/zip";
    /** JPEG image */
    readonly JPEG: "image/jpeg";
    /** PNG image */
    readonly PNG: "image/png";
    /** GIF image */
    readonly GIF: "image/gif";
    /** SVG image */
    readonly SVG: "image/svg+xml";
    /** MP4 video */
    readonly MP4: "video/mp4";
    /** MP3 audio */
    readonly MP3: "audio/mpeg";
};
/**
 * Error codes
 */
export declare const ERROR_CODE: {
    /** Validation error */
    readonly VALIDATION_ERROR: "VALIDATION_ERROR";
    /** Authentication error */
    readonly AUTH_ERROR: "AUTH_ERROR";
    /** Authorization error */
    readonly AUTHZ_ERROR: "AUTHZ_ERROR";
    /** Not found error */
    readonly NOT_FOUND: "NOT_FOUND";
    /** Conflict error */
    readonly CONFLICT: "CONFLICT";
    /** Rate limit error */
    readonly RATE_LIMIT: "RATE_LIMIT";
    /** Network error */
    readonly NETWORK_ERROR: "NETWORK_ERROR";
    /** Timeout error */
    readonly TIMEOUT: "TIMEOUT";
    /** Configuration error */
    readonly CONFIG_ERROR: "CONFIG_ERROR";
    /** External service error */
    readonly EXTERNAL_SERVICE_ERROR: "EXTERNAL_SERVICE_ERROR";
    /** Internal server error */
    readonly INTERNAL_ERROR: "INTERNAL_ERROR";
};
/**
 * Currency codes (ISO 4217)
 */
export declare const CURRENCY: {
    readonly USD: "USD";
    readonly EUR: "EUR";
    readonly GBP: "GBP";
    readonly JPY: "JPY";
    readonly CAD: "CAD";
    readonly AUD: "AUD";
    readonly CHF: "CHF";
    readonly CNY: "CNY";
    readonly INR: "INR";
};
/**
 * Timezone names
 */
export declare const TIMEZONE: {
    readonly UTC: "UTC";
    /** Eastern Time */
    readonly ET: "America/New_York";
    /** Pacific Time */
    readonly PT: "America/Los_Angeles";
    /** Central European Time */
    readonly CET: "Europe/Paris";
    /** Eastern European Time */
    readonly EET: "Europe/Helsinki";
    /** Japan Standard Time */
    readonly JST: "Asia/Tokyo";
    /** China Standard Time */
    readonly CST: "Asia/Shanghai";
    /** Australian Eastern Time */
    readonly AET: "Australia/Sydney";
};
/**
 * Helper function to get timeout value with fallback
 */
export declare function getTimeout(customTimeout?: number, defaultTimeout?: number): number;
/**
 * Helper function to validate pagination limit
 */
export declare function validateLimit(limit?: number): number;
//# sourceMappingURL=constants.d.ts.map