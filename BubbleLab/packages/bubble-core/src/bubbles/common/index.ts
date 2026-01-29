/**
 * Common utilities for Bubble implementations
 * Exports all shared validators, error handlers, retry logic, types, and constants
 */

// Validators (export all, but ValidationError comes from error-handlers)
export {
  validateEmail,
  validateUrl,
  validateTimestamp,
  validateNonEmptyString,
  validateNumberRange,
  validateArrayLength,
  validateRequiredProperties,
  sanitizeString,
  validateFilePath,
  createNonEmptyStringSchema,
  createEmailSchema,
  createUrlSchema,
  batchValidate,
  EMAIL_REGEX,
  URL_REGEX,
  ISO_TIMESTAMP_REGEX
} from './validators.js';

// Error handlers (export all, including the feature-rich ValidationError)
export * from './error-handlers.js';

// Retry logic
export * from './retry.js';

// Types (export all except conflicting ones that are defined in their specialized modules)
export type {
  Result,
  CredentialType,
  Credential,
  RequestOptions,
  PaginationOptions,
  PaginatedResponse,
  SortOptions,
  FilterOptions,
  QueryOptions,
  DateRange,
  TimeRange,
  Coordinate,
  BoundingBox,
  Address,
  Money,
  PersonName,
  ContactInfo,
  UserProfile,
  OperationMetadata,
  CacheEntry,
  RateLimitConfig,
  RetryConfig,
  CircuitBreakerConfig,
  ResilienceConfig,
  HttpMethod,
  HttpRequest,
  HttpResponse,
  ApiErrorResponse,
  FileMetadata,
  UploadProgressCallback,
  DownloadProgressCallback,
  AsyncIteratorResult
} from './types.js';
export {
  ok,
  err,
  unwrap,
  isResult,
  isOk,
  isErr,
  isPlainObject,
  isIsoTimestamp,
  isNonEmptyString,
  isPositiveNumber,
  isArray,
  createMoneySchema,
  createCoordinateSchema,
  createPersonNameSchema,
  deepClone,
  deepMerge
} from './types.js';

// Constants
export * from './constants.js';

// Connection pool management (includes ConnectionPoolConfig)
export * from './connection-pool.js';

// Caching utilities (includes CacheConfig)
export * from './cache.js';
