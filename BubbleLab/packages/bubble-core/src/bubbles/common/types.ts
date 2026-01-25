/**
 * Common type definitions for Bubble implementations
 * Provides shared types, interfaces, and type guards
 */

import { z } from 'zod';

/**
 * Generic result type for operations that can fail
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Create a successful result
 */
export function ok<T>(data: T): Result<T> {
  return { success: true, data };
}

/**
 * Create a failed result
 */
export function err<E = Error>(error: E): Result<never, E> {
  return { success: false, error };
}

/**
 * Unwrap a result, throwing if it's an error
 */
export function unwrap<T>(result: Result<T>): T {
  if (result.success) {
    return result.data;
  }
  throw result.error;
}

/**
 * Common credential types
 */
export enum CredentialType {
  API_KEY = 'api_key',
  OAUTH_TOKEN = 'oauth_token',
  BASIC_AUTH = 'basic_auth',
  BEARER_TOKEN = 'bearer_token',
  DATABASE_CRED = 'database_cred',
  CUSTOM_AUTH_KEY = 'custom_auth_key',
  SLACK_CRED = 'slack_cred',
  STRIPE_CRED = 'stripe_cred',
  AIRTABLE_CRED = 'airtable_cred',
  GMAIL_CRED = 'gmail_cred',
  GOOGLE_CALENDAR_CRED = 'google_calendar_cred',
  SHEETS_CRED = 'sheets_cred',
  NOTION_CRED = 'notion_cred',
  POSTGRES_CRED = 'postgres_cred',
  REDIS_CRED = 'redis_cred',
  MONGODB_CRED = 'mongodb_cred',
  S3_CRED = 's3_cred',
  AWS_CRED = 'aws_cred',
}

/**
 * Credential interface
 */
export interface Credential {
  type: CredentialType;
  value: string;
  expiresAt?: Date;
  metadata?: Record<string, unknown>;
}

/**
 * Generic request options
 */
export interface RequestOptions {
  timeout?: number;
  headers?: Record<string, string>;
  retries?: number;
  signal?: AbortSignal;
}

/**
 * Pagination options
 */
export interface PaginationOptions {
  limit?: number;
  offset?: number;
  cursor?: string;
  page?: number;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    total?: number;
    limit: number;
    offset: number;
    hasMore: boolean;
    nextCursor?: string;
  };
}

/**
 * Sort options
 */
export interface SortOptions {
  field: string;
  direction: 'asc' | 'desc';
}

/**
 * Filter options (generic key-value pairs)
 */
export type FilterOptions = Record<string, unknown>;

/**
 * Common query options combining pagination, sort, and filter
 */
export interface QueryOptions extends PaginationOptions {
  sort?: SortOptions;
  filter?: FilterOptions;
}

/**
 * Date range for queries
 */
export interface DateRange {
  start: Date;
  end: Date;
}

/**
 * Time range for queries
 */
export interface TimeRange {
  start: string; // ISO 8601 timestamp
  end: string; // ISO 8601 timestamp
}

/**
 * Geographic coordinate
 */
export interface Coordinate {
  latitude: number;
  longitude: number;
}

/**
 * Geographic bounding box
 */
export interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

/**
 * Address information
 */
export interface Address {
  street?: string;
  city?: string;
  state?: string;
  postalCode?: string;
  country?: string;
  latitude?: number;
  longitude?: number;
}

/**
 * Money/Amount representation (avoiding floating point errors)
 */
export interface Money {
  amount: number; // In minor units (cents)
  currency: string; // ISO 4217 currency code (e.g., 'USD')
}

/**
 * Person's name
 */
export interface PersonName {
  prefix?: string;
  firstName?: string;
  middleName?: string;
  lastName?: string;
  suffix?: string;
  fullName?: string;
}

/**
 * Contact information
 */
export interface ContactInfo {
  email?: string;
  phone?: string;
  website?: string;
}

/**
 * User profile
 */
export interface UserProfile {
  id: string;
  name: PersonName;
  contact?: ContactInfo;
  avatar?: string;
  timezone?: string;
  locale?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Metadata for operations
 */
export interface OperationMetadata {
  correlationId: string;
  operation: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  success: boolean;
  error?: string;
  retryCount?: number;
}

/**
 * Cache entry with TTL
 */
export interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  createdAt: number;
  accessedAt: number;
  accessCount: number;
}

/**
 * Cache configuration
 */
export interface CacheConfig {
  maxSize?: number;
  defaultTtl?: number;
  cleanupInterval?: number;
}

/**
 * Connection pool configuration
 */
export interface ConnectionPoolConfig {
  min: number;
  max: number;
  acquireTimeoutMillis?: number;
  idleTimeoutMillis?: number;
  evictionRunIntervalMillis?: number;
}

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  maxRequests: number;
  perMilliseconds: number;
  burst?: number;
}

/**
 * Retry configuration
 */
export interface RetryConfig {
  maxAttempts: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
  jitter: boolean;
}

/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeoutMs: number;
  monitoringPeriodMs: number;
}

/**
 * Resilience configuration (combines retry, circuit breaker, rate limiting)
 */
export interface ResilienceConfig {
  retry?: RetryConfig;
  circuitBreaker?: CircuitBreakerConfig;
  rateLimit?: RateLimitConfig;
  timeout?: number;
}

/**
 * HTTP methods
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD' | 'OPTIONS';

/**
 * HTTP request
 */
export interface HttpRequest {
  url: string;
  method: HttpMethod;
  headers?: Record<string, string>;
  body?: string | Record<string, unknown>;
  timeout?: number;
}

/**
 * HTTP response
 */
export interface HttpResponse<T = unknown> {
  status: number;
  statusText: string;
  headers: Record<string, string>;
  body: string;
  json?: T;
  duration: number;
}

/**
 * API error response
 */
export interface ApiErrorResponse {
  error: string;
  message: string;
  code?: string;
  statusCode: number;
  details?: Record<string, unknown>;
  timestamp: string;
  requestId?: string;
}

/**
 * File metadata
 */
export interface FileMetadata {
  name: string;
  size: number;
  mimeType: string;
  createdAt: Date;
  modifiedAt: Date;
  checksum?: string;
}

/**
 * Upload progress callback
 */
export type UploadProgressCallback = (progress: {
  loaded: number;
  total: number;
  percentage: number;
}) => void;

/**
 * Download progress callback
 */
export type DownloadProgressCallback = (progress: {
  loaded: number;
  total: number;
  percentage: number;
}) => void;

/**
 * Generic async iterator result
 */
export interface AsyncIteratorResult<T> {
  done: boolean;
  value?: T;
}

/**
 * Type guard to check if a value is a Result
 */
export function isResult<T>(value: unknown): value is Result<T> {
  return (
    typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    typeof (value as Result<T>).success === 'boolean'
  );
}

/**
 * Type guard to check if a value is a successful Result
 */
export function isOk<T>(value: unknown): value is { success: true; data: T } {
  return isResult<T>(value) && value.success === true;
}

/**
 * Type guard to check if a value is an error Result
 */
export function isErr(value: unknown): value is { success: false; error: Error } {
  return isResult(value) && value.success === false;
}

/**
 * Type guard to check if a value is a plain object
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    !(value instanceof Date) &&
    !(value instanceof RegExp) &&
    !(value instanceof Error)
  );
}

/**
 * Type guard to check if a string is a valid ISO 8601 timestamp
 */
export function isIsoTimestamp(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$/;
  return isoRegex.test(value);
}

/**
 * Type guard to check if a value is a non-empty string
 */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Type guard to check if a value is a positive number
 */
export function isPositiveNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && value > 0;
}

/**
 * Type guard to check if a value is an array
 */
export function isArray<T = unknown>(value: unknown): value is T[] {
  return Array.isArray(value);
}

/**
 * Create a Zod schema for Money type
 */
export function createMoneySchema() {
  return z.object({
    amount: z.number().int().min(0),
    currency: z.string().length(3).regex(/^[A-Z]{3}$/)
  });
}

/**
 * Create a Zod schema for Coordinate type
 */
export function createCoordinateSchema() {
  return z.object({
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180)
  });
}

/**
 * Create a Zod schema for PersonName type
 */
export function createPersonNameSchema() {
  return z.object({
    prefix: z.string().optional(),
    firstName: z.string().optional(),
    middleName: z.string().optional(),
    lastName: z.string().optional(),
    suffix: z.string().optional(),
    fullName: z.string().optional()
  });
}

/**
 * Deep clone an object (simple implementation)
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => deepClone(item)) as T;
  }

  const clonedObj = {} as T;
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      clonedObj[key] = deepClone(obj[key]);
    }
  }

  return clonedObj;
}

/**
 * Merge two objects deeply (simple implementation)
 */
export function deepMerge<T extends Record<string, unknown>>(target: T, source: Partial<T>): T {
  const result = { ...target };

  for (const key in source) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      const sourceValue = source[key];
      const targetValue = result[key];

      if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
        result[key] = deepMerge(
          targetValue as Record<string, unknown>,
          sourceValue as Record<string, unknown>
        ) as T[Extract<keyof T, string>];
      } else {
        result[key] = sourceValue as T[Extract<keyof T, string>];
      }
    }
  }

  return result;
}
