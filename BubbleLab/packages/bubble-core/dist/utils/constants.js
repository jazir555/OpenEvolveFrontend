/**
 * Shared Constants for BubbleLab Bubbles
 *
 * This file contains all magic numbers and configuration values
 * that should be used consistently across all bubble implementations.
 */
// ============================================
// HTTP Timeout Constants (milliseconds)
// ============================================
export const HTTP_TIMEOUT_DEFAULT = 30000; // 30 seconds
export const HTTP_TIMEOUT_SHORT = 5000; // 5 seconds
export const HTTP_TIMEOUT_LONG = 60000; // 60 seconds
export const HTTP_TIMEOUT_UPLOAD = 300000; // 5 minutes (for file uploads)
// ============================================
// Retry Constants
// ============================================
export const RETRY_DEFAULT_ATTEMPTS = 3;
export const RETRY_MAX_ATTEMPTS = 5;
export const RETRY_MIN_DELAY_MS = 1000; // 1 second
export const RETRY_MAX_DELAY_MS = 10000; // 10 seconds
export const RETRY_BACKOFF_MULTIPLIER = 2; // Exponential backoff
// ============================================
// Pagination Constants
// ============================================
export const PAGE_SIZE_DEFAULT = 50;
export const PAGE_SIZE_MAX = 500;
export const PAGE_SIZE_MIN = 10;
export const PAGE_SIZE_SMALL = 25;
export const OFFSET_DEFAULT = 0;
// ============================================
// Buffer Sizes (bytes)
// ============================================
export const BUFFER_SIZE_TINY = 512; // 0.5 KB
export const BUFFER_SIZE_SMALL = 1024; // 1 KB
export const BUFFER_SIZE_MEDIUM = 4096; // 4 KB
export const BUFFER_SIZE_LARGE = 8192; // 8 KB
export const BUFFER_SIZE_XLARGE = 65536; // 64 KB
export const BUFFER_SIZE_HUGE = 1048576; // 1 MB
// ============================================
// HTTP Status Codes
// ============================================
export const HTTP_STATUS_OK = 200;
export const HTTP_STATUS_CREATED = 201;
export const HTTP_STATUS_ACCEPTED = 202;
export const HTTP_STATUS_NO_CONTENT = 204;
export const HTTP_STATUS_MOVED_PERMANENTLY = 301;
export const HTTP_STATUS_FOUND = 302;
export const HTTP_STATUS_BAD_REQUEST = 400;
export const HTTP_STATUS_UNAUTHORIZED = 401;
export const HTTP_STATUS_FORBIDDEN = 403;
export const HTTP_STATUS_NOT_FOUND = 404;
export const HTTP_STATUS_METHOD_NOT_ALLOWED = 405;
export const HTTP_STATUS_CONFLICT = 409;
export const HTTP_STATUS_UNPROCESSABLE_ENTITY = 422;
export const HTTP_STATUS_REQUEST_TIMEOUT = 408;
export const HTTP_STATUS_TOO_MANY_REQUESTS = 429;
export const HTTP_STATUS_INTERNAL_ERROR = 500;
export const HTTP_STATUS_NOT_IMPLEMENTED = 501;
export const HTTP_STATUS_BAD_GATEWAY = 502;
export const HTTP_STATUS_SERVICE_UNAVAILABLE = 503;
export const HTTP_STATUS_GATEWAY_TIMEOUT = 504;
// ============================================
// File Size Limits (bytes)
// ============================================
export const MAX_FILE_SIZE_TINY = 100 * 1024; // 100 KB
export const MAX_FILE_SIZE_SMALL = 1024 * 1024; // 1 MB
export const MAX_FILE_SIZE_MEDIUM = 10 * 1024 * 1024; // 10 MB
export const MAX_FILE_SIZE_LARGE = 100 * 1024 * 1024; // 100 MB
export const MAX_FILE_SIZE_XLARGE = 1024 * 1024 * 1024; // 1 GB
// ============================================
// Rate Limiting
// ============================================
export const RATE_LIMIT_DEFAULT = 100; // requests per minute
export const RATE_LIMIT_BURST = 10; // burst requests
export const RATE_LIMIT_STRICT = 10; // strict rate limit
export const RATE_LIMIT_GENEROUS = 1000; // generous rate limit
// ============================================
// Time Intervals (milliseconds)
// ============================================
export const SECOND_MS = 1000;
export const MINUTE_MS = 60 * SECOND_MS;
export const HOUR_MS = 60 * MINUTE_MS;
export const DAY_MS = 24 * HOUR_MS;
export const WEEK_MS = 7 * DAY_MS;
// ============================================
// String Length Limits
// ============================================
export const MAX_STRING_LENGTH_SHORT = 50;
export const MAX_STRING_LENGTH_MEDIUM = 255;
export const MAX_STRING_LENGTH_LONG = 1000;
export const MAX_STRING_LENGTH_XLONG = 5000;
export const MAX_STRING_LENGTH_TEXT = 65000; // MySQL TEXT limit
// ============================================
// Array Sizes
// ============================================
export const MAX_ARRAY_SIZE_SMALL = 100;
export const MAX_ARRAY_SIZE_MEDIUM = 1000;
export const MAX_ARRAY_SIZE_LARGE = 10000;
// ============================================
// Common Delays (milliseconds)
// ============================================
export const DELAY_INSTANT = 0;
export const DELAY_VERY_SHORT = 100; // 0.1 seconds
export const DELAY_SHORT = 500; // 0.5 seconds
export const DELAY_MEDIUM = 1000; // 1 second
export const DELAY_LONG = 2000; // 2 seconds
export const DELAY_VERY_LONG = 5000; // 5 seconds
// ============================================
// Retryable HTTP Status Codes
// ============================================
export const RETRYABLE_STATUS_CODES = new Set([
    HTTP_STATUS_REQUEST_TIMEOUT, // 408
    HTTP_STATUS_TOO_MANY_REQUESTS, // 429
    HTTP_STATUS_INTERNAL_ERROR, // 500
    HTTP_STATUS_BAD_GATEWAY, // 502
    HTTP_STATUS_SERVICE_UNAVAILABLE, // 503
    HTTP_STATUS_GATEWAY_TIMEOUT, // 504
]);
// ============================================
// Common MIME Types
// ============================================
export const MIME_TYPES = {
    JSON: 'application/json',
    TEXT: 'text/plain',
    HTML: 'text/html',
    XML: 'application/xml',
    PDF: 'application/pdf',
    FORM_URLENCODED: 'application/x-www-form-urlencoded',
    MULTIPART_FORM_DATA: 'multipart/form-data',
};
// ============================================
// Common HTTP Headers
// ============================================
export const HEADERS = {
    ACCEPT: 'Accept',
    AUTHORIZATION: 'Authorization',
    CONTENT_TYPE: 'Content-Type',
    USER_AGENT: 'User-Agent',
    ACCEPT_ENCODING: 'Accept-Encoding',
    CACHE_CONTROL: 'Cache-Control',
};
// ============================================
// Validation Constraints
// ============================================
export const VALIDATION = {
    MIN_PASSWORD_LENGTH: 8,
    MAX_PASSWORD_LENGTH: 128,
    MIN_USERNAME_LENGTH: 3,
    MAX_USERNAME_LENGTH: 50,
    EMAIL_REGEX: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    URL_REGEX: /^https?:\/\/.+/,
    SLACK_CHANNEL_ID_REGEX: /^[A-Z0-9]+$/,
    SLACK_USER_ID_REGEX: /^U[A-Z0-9]+$/,
};
// ============================================
// Cache Durations (milliseconds)
// ============================================
export const CACHE_DURATION_TTL_SHORT = 5 * MINUTE_MS; // 5 minutes
export const CACHE_DURATION_TTL_MEDIUM = 30 * MINUTE_MS; // 30 minutes
export const CACHE_DURATION_TTL_LONG = 2 * HOUR_MS; // 2 hours
export const CACHE_DURATION_TTL_XLONG = 24 * HOUR_MS; // 24 hours
// ============================================
// Regular Expression Patterns
// ============================================
export const PATTERNS = {
    EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    URL: /^https?:\/\/.+/,
    SLACK_CHANNEL: /^#[\w-]+$/,
    SLACK_CHANNEL_ID: /^[A-Z0-9]+$/,
    SLACK_USER_ID: /^U[A-Z0-9]+$/,
    GITHUB_REPO: /^[\w-]+\/[\w-]+$/,
    NOTION_ID: /^[a-f0-9]{32}$/,
    UUID: /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i,
};
// ============================================
// Error Messages
// ============================================
export const ERROR_MESSAGES = {
    INVALID_INPUT: 'Invalid input provided',
    UNAUTHORIZED: 'Unauthorized access',
    FORBIDDEN: 'Access forbidden',
    NOT_FOUND: 'Resource not found',
    RATE_LIMITED: 'Rate limit exceeded',
    TIMEOUT: 'Request timeout',
    NETWORK_ERROR: 'Network error occurred',
    INVALID_RESPONSE: 'Invalid response received',
};
// ============================================
// Logging Levels
// ============================================
export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["DEBUG"] = 0] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
})(LogLevel || (LogLevel = {}));
// ============================================
// Chart Constants
// ============================================
export const CHART = {
    // Default dimensions
    DEFAULT_WIDTH: 800,
    DEFAULT_HEIGHT: 600,
    MIN_WIDTH: 400,
    MIN_HEIGHT: 300,
    // Size categories
    SIZE_SMALL: { width: 400, height: 300 },
    SIZE_SQUARE: { width: 400, height: 400 },
    SIZE_RADAR: { width: 450, height: 450 },
    SIZE_MEDIUM: { width: 600, height: 400 },
    // Data thresholds
    DATA_DENSITY_THRESHOLD: 50,
    // Border widths
    BORDER_WIDTH_THIN: 1,
    BORDER_WIDTH_THICK: 2,
    // Default bubble radius
    BUBBLE_RADIUS_DEFAULT: 5,
    // Sample size for column detection
    COLUMN_DETECTION_SAMPLE_SIZE: 10,
    // Color opacity values
    OPACITY_DEFAULT: '0.8',
    OPACITY_SOLID: '1',
};
// ============================================
// Chart Color Palettes
// ============================================
export const CHART_COLORS = {
    DEFAULT: [
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 205, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
    ],
    VIRIDIS: [
        'rgba(68, 1, 84, 0.8)',
        'rgba(59, 82, 139, 0.8)',
        'rgba(33, 145, 140, 0.8)',
        'rgba(94, 201, 98, 0.8)',
        'rgba(253, 231, 37, 0.8)',
    ],
    PLASMA: [
        'rgba(13, 8, 135, 0.8)',
        'rgba(126, 3, 168, 0.8)',
        'rgba(204, 71, 120, 0.8)',
        'rgba(248, 149, 64, 0.8)',
        'rgba(240, 249, 33, 0.8)',
    ],
    INFERNO: [
        'rgba(0, 0, 4, 0.8)',
        'rgba(87, 15, 108, 0.8)',
        'rgba(204, 71, 120, 0.8)',
        'rgba(251, 150, 66, 0.8)',
        'rgba(252, 254, 164, 0.8)',
    ],
    MAGMA: [
        'rgba(3, 0, 71, 0.8)',
        'rgba(94, 19, 145, 0.8)',
        'rgba(212, 71, 104, 0.8)',
        'rgba(249, 145, 56, 0.8)',
        'rgba(252, 253, 191, 0.8)',
    ],
    BLUES: [
        'rgba(8, 48, 107, 0.8)',
        'rgba(8, 81, 156, 0.8)',
        'rgba(33, 113, 181, 0.8)',
        'rgba(66, 146, 198, 0.8)',
        'rgba(107, 174, 214, 0.8)',
        'rgba(158, 202, 225, 0.8)',
    ],
    GREENS: [
        'rgba(0, 68, 27, 0.8)',
        'rgba(0, 109, 44, 0.8)',
        'rgba(35, 139, 69, 0.8)',
        'rgba(74, 172, 98, 0.8)',
        'rgba(140, 202, 140, 0.8)',
        'rgba(199, 233, 192, 0.8)',
    ],
    REDS: [
        'rgba(103, 0, 13, 0.8)',
        'rgba(178, 24, 43, 0.8)',
        'rgba(214, 96, 77, 0.8)',
        'rgba(244, 165, 130, 0.8)',
        'rgba(253, 219, 199, 0.8)',
    ],
    ORANGES: [
        'rgba(79, 30, 9, 0.8)',
        'rgba(148, 58, 12, 0.8)',
        'rgba(230, 126, 34, 0.8)',
        'rgba(253, 187, 132, 0.8)',
        'rgba(254, 230, 206, 0.8)',
    ],
    CATEGORICAL: [
        'rgba(31, 119, 180, 0.8)',
        'rgba(255, 127, 14, 0.8)',
        'rgba(44, 160, 44, 0.8)',
        'rgba(214, 39, 40, 0.8)',
        'rgba(148, 103, 189, 0.8)',
        'rgba(140, 86, 75, 0.8)',
        'rgba(227, 119, 194, 0.8)',
        'rgba(127, 127, 127, 0.8)',
        'rgba(188, 189, 34, 0.8)',
        'rgba(23, 190, 207, 0.8)',
    ],
};
export const CHART_COLOR_SCHEMES = {
    DEFAULT: 'default',
    VIRIDIS: 'viridis',
    PLASMA: 'plasma',
    INFERNO: 'inferno',
    MAGMA: 'magma',
    BLUES: 'blues',
    GREENS: 'greens',
    REDS: 'reds',
    ORANGES: 'oranges',
    CATEGORICAL: 'categorical',
};
export const CHART_TYPES = {
    LINE: 'line',
    BAR: 'bar',
    PIE: 'pie',
    DOUGHNUT: 'doughnut',
    RADAR: 'radar',
    POLAR_AREA: 'polarArea',
    SCATTER: 'scatter',
    BUBBLE: 'bubble',
};
// ============================================
// AI Agent Configuration
// ============================================
export const AI_AGENT = {
    // Temperature settings
    TEMPERATURE_MIN: 0,
    TEMPERATURE_MAX: 2,
    TEMPERATURE_DEFAULT: 1,
    TEMPERATURE_LOW: 0.3,
    TEMPERATURE_MEDIUM: 0.7,
    TEMPERATURE_HIGH: 1.0,
    // Token limits
    MAX_TOKENS_DEFAULT: 64000,
    MAX_TOKENS_MIN: 1,
    MAX_TOKENS_LOW: 4000,
    MAX_TOKENS_MEDIUM: 16000,
    MAX_TOKENS_HIGH: 32000,
    // Retry configuration
    MAX_RETRIES_DEFAULT: 3,
    MAX_RETRIES_MAX: 10,
    MAX_RETRIES_MIN: 0,
    // Iteration limits
    MIN_ITERATIONS: 5,
    MAX_ITERATIONS_DEFAULT: 50,
    MAX_ITERATIONS_MIN: 1,
    MAX_ITERATIONS_MAX: 100,
    // Cache configuration
    CACHE_SIZE: 1000,
    CACHE_TTL_MS: 3600000, // 1 hour
    CACHE_TTL_SHORT_MS: 300000, // 5 minutes
    CACHE_TTL_LONG_MS: 7200000, // 2 hours
    CLEANUP_INTERVAL_MS: 300000, // 5 minutes
    // Retry delays
    RETRY_BASE_DELAY_MS: 1000,
    RETRY_JITTER_FACTOR: 0.25,
    RETRY_MAX_DELAY_MS: 10000,
    RETRY_BACKOFF_MULTIPLIER: 2,
    // Image fetch limits
    IMAGE_FETCH_TIMEOUT_MS: 10000, // 10 seconds
    IMAGE_MAX_SIZE_BYTES: 10 * 1024 * 1024, // 10 MB
    IMAGE_MIN_WIDTH: 100,
    IMAGE_MIN_HEIGHT: 100,
    // Message truncation
    MESSAGE_PREVIEW_LENGTH: 100,
    MESSAGE_MAX_LENGTH: 10000,
    CONTEXT_WINDOW_DEFAULT: 8000,
    // Streaming
    STREAMING_CHUNK_SIZE: 1000,
    STREAMING_TIMEOUT_MS: 60000,
    // Tool execution
    TOOL_TIMEOUT_MS: 30000,
    TOOL_MAX_RETRIES: 3,
    // Thinking budgets for reasoning models
    THINKING_BUDGET_LOW: 1025,
    THINKING_BUDGET_MEDIUM: 5000,
    THINKING_BUDGET_HIGH: 10000,
};
// ============================================
// Reddit API Configuration
// ============================================
export const REDDIT = {
    // API limits
    MAX_POSTS_DEFAULT: 100,
    MAX_POSTS_MIN: 1,
    MAX_POSTS_MAX: 1000,
    MAX_COMMENTS_DEFAULT: 50,
    MAX_COMMENTS_MIN: 1,
    MAX_COMMENTS_MAX: 500,
    MIN_POSTS: 1,
    MIN_COMMENTS: 1,
    // Timeouts
    REQUEST_TIMEOUT_MS: 10000,
    RETRY_DELAY_MS: 1000,
    CONNECTION_TIMEOUT_MS: 5000,
    // Retry configuration
    MAX_RETRIES: 3,
    RETRY_BACKOFF_MS: 2000,
    // URL limits
    MAX_URL_LENGTH: 2000,
    // Score thresholds
    MIN_SCORE: 0,
    MIN_UPVOTE_RATIO: 0.0,
    MAX_UPVOTE_RATIO: 1.0,
    MIN_UPVOTES: 0,
    // Rate limiting
    RATE_LIMIT_REQUESTS: 60,
    RATE_LIMIT_WINDOW_MS: 60000, // 1 minute
    // Post filtering
    POST_AGE_MAX_DAYS: 365,
    COMMENT_DEPTH_MAX: 10,
};
// ============================================
// Document Generation Configuration
// ============================================
export const DOCUMENT_GENERATION = {
    // Chunk sizes
    CHUNK_SIZE_DEFAULT: 4000,
    CHUNK_SIZE_MIN: 1000,
    CHUNK_SIZE_MAX: 8000,
    CHUNK_OVERHEAD: 200,
    CHUNK_OVERLAP: 200,
    // Generation limits
    MAX_SECTIONS: 50,
    MAX_SECTIONS_MIN: 1,
    MAX_ITERATIONS: 10,
    MAX_ITERATIONS_MIN: 1,
    MAX_ITERATIONS_MAX: 50,
    // Timeouts
    GENERATION_TIMEOUT_MS: 30000,
    CHUNK_TIMEOUT_MS: 5000,
    SECTION_TIMEOUT_MS: 10000,
    // Retry configuration
    MAX_RETRIES: 3,
    RETRY_DELAY_MS: 1000,
    // Content limits
    MIN_SECTION_LENGTH: 200,
    MAX_SECTION_LENGTH: 5000,
    TARGET_SECTION_LENGTH: 2000,
    // Templates
    TEMPLATE_CACHE_SIZE: 100,
    TEMPLATE_MAX_SIZE: 50000,
};
// ============================================
// PDF Generation Configuration
// ============================================
export const PDF_GENERATION = {
    // Page dimensions (points)
    PAGE_WIDTH_DEFAULT: 595, // A4 in points
    PAGE_HEIGHT_DEFAULT: 842,
    PAGE_WIDTH_LETTER: 612,
    PAGE_HEIGHT_LETTER: 792,
    MARGIN_DEFAULT: 50,
    MARGIN_NARROW: 25,
    MARGIN_WIDE: 75,
    // Font sizes
    FONT_SIZE_TITLE: 24,
    FONT_SIZE_SUBTITLE: 20,
    FONT_SIZE_HEADING: 18,
    FONT_SIZE_SUBHEADING: 16,
    FONT_SIZE_BODY: 12,
    FONT_SIZE_SMALL: 10,
    FONT_SIZE_FOOTNOTE: 8,
    FONT_SIZE_MIN: 8,
    FONT_SIZE_MAX: 72,
    // Line spacing
    LINE_SPACING_SINGLE: 1.0,
    LINE_SPACING_DOUBLE: 2.0,
    LINE_SPACING_1_5: 1.5,
    // Colors
    COLOR_BLACK: '#000000',
    COLOR_WHITE: '#FFFFFF',
    COLOR_GRAY_LIGHT: '#CCCCCC',
    COLOR_GRAY_DARK: '#333333',
    // Limits
    MAX_FILE_SIZE_MB: 100,
    MAX_PAGES: 1000,
    MAX_IMAGES: 500,
    // Image constraints
    IMAGE_MAX_WIDTH: 500,
    IMAGE_MAX_HEIGHT: 500,
    IMAGE_DPI_DEFAULT: 150,
    IMAGE_DPI_HIGH: 300,
    // Table settings
    TABLE_BORDER_WIDTH: 1,
    TABLE_CELL_PADDING: 8,
    TABLE_HEADER_BG_COLOR: '#DDDDDD',
};
// ============================================
// GitHub API Configuration
// ============================================
export const GITHUB_API = {
    // API endpoints
    API_BASE_URL: 'https://api.github.com',
    API_VERSION: '2022-11-28',
    // Pagination
    PER_PAGE_DEFAULT: 30,
    PER_PAGE_MAX: 100,
    PER_PAGE_MIN: 1,
    PER_PAGE_SMALL: 10,
    // Rate limiting
    RATE_LIMIT_DEFAULT: 5000, // requests per hour
    RATE_LIMIT_AUTH: 5000,
    RATE_LIMIT_UNAUTH: 60,
    // Timeouts
    REQUEST_TIMEOUT_MS: 10000,
    RETRY_DELAY_MS: 1000,
    CONNECTION_TIMEOUT_MS: 5000,
    // Retry configuration
    MAX_RETRIES: 3,
    RETRY_BACKOFF_MS: 2000,
    // Repository limits
    MAX_REPOS_PER_PAGE: 100,
    MAX_REPOS_SEARCH: 1000,
    // Commit limits
    MAX_COMMITS_PER_PAGE: 100,
    MAX_COMMITS_FETCH: 500,
    // Issue/PR limits
    MAX_ISSUES_PER_PAGE: 100,
    MAX_COMMENTS_PER_PAGE: 100,
    // File size limits
    MAX_FILE_SIZE_RAW: 1 * 1024 * 1024, // 1 MB for raw file viewing
};
// ============================================
// Stripe Configuration
// ============================================
export const STRIPE = {
    // API version
    API_VERSION: '2023-10-16',
    // Currency
    DEFAULT_CURRENCY: 'usd',
    MIN_AMOUNT: 50, // $0.50 in cents
    MAX_AMOUNT: 99999999, // $999,999.99 in cents
    // Payment intents
    PAYMENT_METHOD_TYPES: ['card', 'us_bank_account'],
    DEFAULT_PAYMENT_METHOD_TYPE: 'card',
    // Limits
    MAX_LINE_ITEMS: 100,
    MAX_DESCRIPTION_LENGTH: 5000,
    MAX_METADATA_KEYS: 50,
    MAX_METADATA_VALUE_LENGTH: 500,
    // Timeouts
    REQUEST_TIMEOUT_MS: 30000,
    RETRY_DELAY_MS: 1000,
    WEBHOOK_TIMEOUT_MS: 30000,
    // Retry configuration
    MAX_RETRIES: 3,
    MAX_RETRIES_IDEMPOTENCY: 10,
    // Webhook limits
    WEBHOOK_MAX_PAYLOAD_SIZE: 500 * 1024, // 500 KB
    WEBHOOK_MAX_EVENTS: 100,
    // Subscription limits
    MAX_TRIAL_PERIOD_DAYS: 365,
    MIN_TRIAL_PERIOD_DAYS: 0,
};
// ============================================
// Document Parsing Configuration
// ============================================
export const DOCUMENT_PARSING = {
    // Chunk sizes
    CHUNK_SIZE_DEFAULT: 2000,
    CHUNK_SIZE_MIN: 500,
    CHUNK_SIZE_MAX: 5000,
    CHUNK_OVERLAP: 200,
    CHUNK_OVERLAP_MIN: 50,
    CHUNK_OVERLAP_MAX: 500,
    // Parsing limits
    MAX_FILE_SIZE_MB: 100,
    MAX_PAGES: 1000,
    MAX_TEXT_LENGTH: 1000000, // 1 million characters
    // Timeout
    PARSE_TIMEOUT_MS: 30000,
    PAGE_PARSE_TIMEOUT_MS: 5000,
    // Retry configuration
    MAX_RETRIES: 3,
    RETRY_DELAY_MS: 1000,
    // Text extraction
    MIN_TEXT_LENGTH: 100,
    MAX_TEXT_LENGTH_PER_PAGE: 10000,
    // OCR fallback
    OCR_CONFIDENCE_THRESHOLD: 0.7,
    OCR_MIN_TEXT_LENGTH: 50,
    // Supported formats
    SUPPORTED_IMAGE_FORMATS: ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'],
    SUPPORTED_PDF_FORMATS: ['application/pdf'],
};
// ============================================
// OCR Configuration
// ============================================
export const OCR = {
    // Image preprocessing
    DPI_DEFAULT: 300,
    DPI_MIN: 150,
    DPI_MAX: 600,
    DOWNSCALE_FACTOR: 2,
    // Processing
    TIMEOUT_MS: 60000,
    TIMEOUT_PER_PAGE_MS: 10000,
    MAX_PAGES: 1000,
    MAX_CONCURRENT_PAGES: 5,
    // Confidence thresholds
    CONFIDENCE_THRESHOLD: 0.7,
    CONFIDENCE_MIN: 0.5,
    CONFIDENCE_HIGH: 0.9,
    MIN_TEXT_LENGTH: 10,
    MIN_WORD_LENGTH: 2,
    // Language settings
    DEFAULT_LANGUAGE: 'eng',
    MAX_LANGUAGES: 10,
    SUPPORTED_LANGUAGES: ['eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'kor'],
    // Image preprocessing
    PREPROCESSING_DOWNSCALE_WIDTH: 2000,
    PREPROCESSING_DOWNSCALE_HEIGHT: 2000,
    CONTRAST_ENHANCEMENT: true,
    NOISE_REDUCTION: true,
    // Text extraction
    EXTRACT_WORDS: true,
    EXTRACT_CONFIDENCE: true,
    EXTRACT_BBOX: true,
    EXTRACT_LINES: true,
    // Performance
    CACHE_ENABLED: true,
    CACHE_SIZE: 1000,
    CACHE_TTL_MS: 3600000, // 1 hour
};
// ============================================
// Hephaestus Configuration
// ============================================
export const HEPHAEUSTUS = {
    // MCP timeouts
    MCP_TIMEOUT_MS: 30000,
    MCP_TIMEOUT_SHORT_MS: 10000,
    MCP_TIMEOUT_LONG_MS: 60000,
    MCP_RETRY_DELAY_MS: 1000,
    MCP_MAX_RETRIES: 3,
    // Delegation limits
    MAX_DELEGATION_DEPTH: 5,
    MAX_DELEGATION_DEPTH_MIN: 1,
    MAX_DELEGATION_DEPTH_MAX: 10,
    MAX_SUBTASKS: 10,
    MAX_SUBTASKS_MIN: 1,
    MAX_SUBTASKS_MAX: 50,
    // Result limits
    MAX_RESULT_SIZE: 1000000, // 1 MB
    MAX_RESULT_SIZE_MIN: 100000, // 100 KB
    MAX_RESULT_SIZE_MAX: 10000000, // 10 MB
    // Task execution
    TASK_TIMEOUT_MS: 300000, // 5 minutes
    TASK_TIMEOUT_DEFAULT_MS: 60000, // 1 minute
    SUBTASK_TIMEOUT_MS: 30000,
    // Retry configuration
    MAX_RETRIES: 3,
    RETRY_DELAY_MS: 1000,
    RETRY_BACKOFF_MULTIPLIER: 2,
    // Queue management
    MAX_QUEUE_SIZE: 1000,
    MAX_CONCURRENT_TASKS: 10,
    TASK_PRIORITY_LEVELS: 5,
    // Resource limits
    MAX_MEMORY_MB: 512,
    MAX_CPU_PERCENT: 80,
};
// ============================================
// Default Values
// ============================================
export const DEFAULTS = {
    TIMEOUT: HTTP_TIMEOUT_DEFAULT,
    RETRY_ATTEMPTS: RETRY_DEFAULT_ATTEMPTS,
    PAGE_SIZE: PAGE_SIZE_DEFAULT,
    CACHE_TTL: CACHE_DURATION_TTL_MEDIUM,
    LOG_LEVEL: LogLevel.INFO,
    CHART_WIDTH: CHART.DEFAULT_WIDTH,
    CHART_HEIGHT: CHART.DEFAULT_HEIGHT,
    AI_TEMPERATURE: AI_AGENT.TEMPERATURE_DEFAULT,
    AI_MAX_TOKENS: AI_AGENT.MAX_TOKENS_DEFAULT,
    PDF_MARGIN: PDF_GENERATION.MARGIN_DEFAULT,
};
//# sourceMappingURL=constants.js.map