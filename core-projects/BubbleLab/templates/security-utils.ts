/**
 * Security Utilities for BubbleLab Workflows
 * Purpose: Centralized security functions for Wave 2 security fixes
 *
 * Provides:
 * - Environment variable validation
 * - API key authentication
 * - Rate limiting
 * - Input validation and sanitization
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - SQL injection prevention helpers
 * - Command injection prevention helpers
 */

import { z } from 'zod';
import crypto from 'crypto';

// ============================================================================
// Input Validation Schemas
// ============================================================================

export const SecuritySchemas = {
  // Container/Resource identifiers
  containerId: z.string().regex(/^[a-f0-9]{12,}$/, 'Invalid container ID format'),
  containerName: z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid container name'),

  // Database identifiers
  databaseName: z.string().min(1).max(63).regex(/^[a-zA-Z0-9_]+$/, 'Invalid database name'),
  backupId: z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid backup ID format'),
  tableName: z.string().min(1).max(63).regex(/^[a-zA-Z0-9_]+$/, 'Invalid table name'),

  // Service identifiers
  serviceName: z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid service name'),
  namespace: z.string().min(1).max(253).regex(/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/, 'Invalid namespace format'),

  // URLs and endpoints
  url: z.string().url('Invalid URL format'),
  apiUrl: z.string().url().refine(url => url.startsWith('https://') || process.env.NODE_ENV === 'development', {
    message: 'API URL must use HTTPS in production'
  }),

  // Authentication
  apiKey: z.string().min(32).max(256),
  token: z.string().min(20).max(4096),

  // Common fields
  timestamp: z.string().datetime(),
  correlationId: z.string().regex(/^[a-f0-9]{32}$/, 'Invalid correlation ID format'),
  email: z.string().email('Invalid email format'),

  // Numeric fields
  port: z.number().int().min(1).max(65535),
  percentage: z.number().min(0).max(100),
  count: z.number().int().min(0),

  // Text fields
  message: z.string().min(1).max(4096),
  description: z.string().min(1).max(10000),
};

// ============================================================================
// Environment Variable Validation
// ============================================================================

export interface EnvValidationConfig {
  required: string[];
  optional?: string[];
  schemas?: Record<string, z.ZodSchema>;
}

export function validateEnvironment(config: EnvValidationConfig): void {
  const missing = config.required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    throw new Error(
      `CRITICAL: Missing required environment variables: ${missing.join(', ')}. ` +
      `Set them and restart.`
    );
  }

  // Validate formats if schemas provided
  if (config.schemas) {
    for (const [key, schema] of Object.entries(config.schemas)) {
      const value = process.env[key];
      if (value) {
        try {
          schema.parse(value);
        } catch (error) {
          throw new Error(`CRITICAL: Environment variable ${key} has invalid format.`);
        }
      }
    }
  }
}

// ============================================================================
// API Key Authentication
// ============================================================================

export interface AuthContext {
  readonly correlationId: string;
  readonly authenticated: boolean;
  readonly ip?: string;
  readonly userAgent?: string;
}

export function authenticateRequest(
  providedKey: string | undefined,
  expectedKey: string | undefined,
  context: { correlationId: string; ip?: string; userAgent?: string }
): AuthContext {
  if (!expectedKey) {
    throw new Error('CRITICAL: API_KEY environment variable not configured.');
  }

  const isAuthenticated = providedKey === expectedKey;

  const authContext: AuthContext = {
    correlationId: context.correlationId,
    authenticated: isAuthenticated,
    ip: context.ip,
    userAgent: context.userAgent,
  };

  return authContext;
}

export function requireAuthentication(authContext: AuthContext): void {
  if (!authContext.authenticated) {
    throw new Error('Unauthorized: Invalid API key');
  }
}

// ============================================================================
// Rate Limiting
// ============================================================================

export interface RateLimitConfig {
  readonly maxRequests: number;
  readonly windowMs: number;
}

export class RateLimiter {
  private static requests = new Map<string, { count: number; resetTime: number }>();

  constructor(private config: RateLimitConfig) {}

  checkLimit(identifier: string): boolean {
    const now = Date.now();
    const key = identifier || 'anonymous';

    let record = RateLimiter.requests.get(key);

    if (!record || now > record.resetTime) {
      record = {
        count: 0,
        resetTime: now + this.config.windowMs
      };
      RateLimiter.requests.set(key, record);
    }

    record.count++;

    return record.count <= this.config.maxRequests;
  }

  getRemainingRequests(identifier: string): number {
    const record = RateLimiter.requests.get(identifier);
    if (!record) return this.config.maxRequests;

    const now = Date.now();
    if (now > record.resetTime) return this.config.maxRequests;

    return Math.max(0, this.config.maxRequests - record.count);
  }

  static cleanup(): void {
    const now = Date.now();
    for (const [key, record] of RateLimiter.requests.entries()) {
      if (now > record.resetTime) {
        RateLimiter.requests.delete(key);
      }
    }
  }
}

// Clean up expired rate limit entries every 5 minutes
if (typeof setInterval !== 'undefined') {
  setInterval(() => RateLimiter.cleanup(), 5 * 60 * 1000);
}

// ============================================================================
// Input Validation and Sanitization
// ============================================================================

export class InputValidator {
  static validateContainerId(id: string): string {
    try {
      SecuritySchemas.containerId.parse(id);
      return id;
    } catch (error) {
      throw new Error('Invalid container ID format');
    }
  }

  static validateContainerName(name: string): string {
    try {
      SecuritySchemas.containerName.parse(name);
      return name;
    } catch (error) {
      return 'invalid-name';
    }
  }

  static validateDatabaseName(dbName: string): string {
    try {
      SecuritySchemas.databaseName.parse(dbName);
      return dbName;
    } catch (error) {
      throw new Error('Invalid database name format');
    }
  }

  static validateBackupId(backupId: string): string {
    try {
      SecuritySchemas.backupId.parse(backupId);
      return backupId;
    } catch (error) {
      throw new Error('Invalid backup ID format');
    }
  }

  static validateServiceName(serviceName: string): string {
    try {
      SecuritySchemas.serviceName.parse(serviceName);
      return serviceName;
    } catch (error) {
      throw new Error('Invalid service name format');
    }
  }

  static validateUrl(url: string): string {
    try {
      SecuritySchemas.url.parse(url);
      return url;
    } catch (error) {
      throw new Error('Invalid URL format');
    }
  }

  static validateApiKey(apiKey: string): string {
    try {
      SecuritySchemas.apiKey.parse(apiKey);
      return apiKey;
    } catch (error) {
      throw new Error('Invalid API key format');
    }
  }

  static sanitizeString(input: string, maxLength: number = 1000): string {
    if (typeof input !== 'string') return '';

    // Remove null bytes and control characters except newlines and tabs
    let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');

    // Truncate to max length
    if (sanitized.length > maxLength) {
      sanitized = sanitized.substring(0, maxLength);
    }

    return sanitized;
  }

  static sanitizeNumber(input: unknown, min: number = 0, max: number = Number.MAX_SAFE_INTEGER): number {
    const num = typeof input === 'number' ? input : parseInt(String(input), 10);

    if (isNaN(num)) {
      throw new Error('Invalid number format');
    }

    return Math.max(min, Math.min(max, num));
  }
}

// ============================================================================
// Error Message Sanitization
// ============================================================================

export function sanitizeError(error: unknown): string {
  if (error instanceof Error) {
    // Remove stack traces and internal file paths
    let sanitized = error.message;

    // Remove file paths
    sanitized = sanitized.replace(/\/[a-zA-Z0-9_\-\/]+\.(ts|js|tsx|jsx):\d+:\d+/g, '[internal]');
    sanitized = sanitized.replace(/at .+/g, '');
    sanitized = sanitized.replace(/    at .+/g, '');

    // Remove potential secrets (basic heuristic)
    sanitized = sanitized.replace(/password["\s:=]+[^\s"]+/gi, 'password=[REDACTED]');
    sanitized = sanitized.replace(/token["\s:=]+[^\s"]+/gi, 'token=[REDACTED]');
    sanitized = sanitized.replace(/key["\s:=]+[^\s"]+/gi, 'key=[REDACTED]');
    sanitized = sanitized.replace(/secret["\s:=]+[^\s"]+/gi, 'secret=[REDACTED]');

    return sanitized;
  }

  return 'Unknown error';
}

// ============================================================================
// Structured Logging
// ============================================================================

export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

export interface LogContext {
  readonly correlationId: string;
  readonly [key: string]: unknown;
}

export class StructuredLogger {
  constructor(private serviceName: string) {}

  private log(level: LogLevel, data: Record<string, unknown>, error?: unknown): void {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      service: this.serviceName,
      ...data,
      ...(error && { error: sanitizeError(error) }),
    };

    console.log(JSON.stringify(logEntry));
  }

  info(data: Record<string, unknown>, error?: unknown): void {
    this.log('info', data, error);
  }

  warn(data: Record<string, unknown>, error?: unknown): void {
    this.log('warn', data, error);
  }

  error(data: Record<string, unknown>, error?: unknown): void {
    this.log('error', data, error);
  }

  debug(data: Record<string, unknown>, error?: unknown): void {
    if (process.env.LOG_LEVEL === 'debug') {
      this.log('debug', data, error);
    }
  }

  child(context: LogContext): StructuredLogger {
    const childLogger = new StructuredLogger(this.serviceName);
    childLogger.info = (data: Record<string, unknown>, error?: unknown) => {
      this.log('info', { ...context, ...data }, error);
    };
    childLogger.warn = (data: Record<string, unknown>, error?: unknown) => {
      this.log('warn', { ...context, ...data }, error);
    };
    childLogger.error = (data: Record<string, unknown>, error?: unknown) => {
      this.log('error', { ...context, ...data }, error);
    };
    childLogger.debug = (data: Record<string, unknown>, error?: unknown) => {
      if (process.env.LOG_LEVEL === 'debug') {
        this.log('debug', { ...context, ...data }, error);
      }
    };
    return childLogger;
  }
}

// ============================================================================
// Correlation ID Management
// ============================================================================

export function generateCorrelationId(): string {
  return crypto.randomBytes(16).toString('hex');
}

export function extractCorrelationId(headers: Record<string, string | undefined>): string {
  const provided = headers['x-correlation-id'] || headers['correlation-id'];
  if (provided) {
    try {
      SecuritySchemas.correlationId.parse(provided);
      return provided;
    } catch {
      // Invalid format, generate new one
    }
  }
  return generateCorrelationId();
}

// ============================================================================
// SQL Injection Prevention Helpers
// ============================================================================

export function validateSqlIdentifier(identifier: string, type: 'table' | 'column' | 'database'): string {
  const schema = type === 'table' ? SecuritySchemas.tableName :
                 type === 'database' ? SecuritySchemas.databaseName :
                 z.string().min(1).max(63).regex(/^[a-zA-Z0-9_]+$/, 'Invalid column name');

  try {
    schema.parse(identifier);
    return identifier;
  } catch (error) {
    throw new Error(`Invalid ${type} identifier format`);
  }
}

export function buildParameterizedQuery(baseQuery: string, params: unknown[]): {
  query: string;
  params: unknown[];
} {
  // Ensure parameter placeholders match params array
  const placeholderCount = (baseQuery.match(/\$\d+/g) || []).length;

  if (placeholderCount !== params.length) {
    throw new Error(
      `Parameter count mismatch: Expected ${placeholderCount} parameters, got ${params.length}`
    );
  }

  return { query: baseQuery, params };
}

// ============================================================================
// Command Injection Prevention Helpers
// ============================================================================

export function validateCommandArgument(arg: string, allowPattern?: RegExp): string {
  // Basic sanitization - allow only safe characters
  const safeChars = /^[a-zA-Z0-9._-]+$/;

  if (allowPattern) {
    if (!allowPattern.test(arg)) {
      throw new Error('Invalid command argument format');
    }
  } else if (!safeChars.test(arg)) {
    throw new Error('Invalid command argument format');
  }

  return arg;
}

export function sanitizeContainerCommand(containerId: string): string {
  // Validate container ID format before using in commands
  try {
    SecuritySchemas.containerId.parse(containerId);
    return containerId;
  } catch (error) {
    throw new Error('Invalid container ID for command execution');
  }
}

// ============================================================================
// Common Webhook Payload Validation
// ============================================================================

export const WebhookPayloadSchema = z.object({
  headers: z.record(z.string()).optional(),
  body: z.unknown().optional(),
  query: z.record(z.string()).optional(),
});

export type WebhookPayload = z.infer<typeof WebhookPayloadSchema>;

export function validateWebhookPayload(payload: unknown): WebhookPayload {
  try {
    return WebhookPayloadSchema.parse(payload);
  } catch (error) {
    throw new Error('Invalid webhook payload format');
  }
}

// ============================================================================
// Export All Utilities
// ============================================================================

export const SecurityUtils = {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  extractCorrelationId,
  validateSqlIdentifier,
  buildParameterizedQuery,
  validateCommandArgument,
  sanitizeContainerCommand,
  validateWebhookPayload,
  SecuritySchemas,
};

export default SecurityUtils;
