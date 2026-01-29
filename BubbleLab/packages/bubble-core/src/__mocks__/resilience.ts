/**
 * Mock Resilience Module for Testing
 *
 * Provides mock implementations of resilience patterns to avoid
 * dependency on external integrations/openevolve module during testing
 */

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

export enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  halfOpenAttempts: number;
}

export class CircuitBreaker {
  constructor(_config: CircuitBreakerConfig) {
    this.validateConfig();
  }

  private validateConfig(): void {
    // Mock validation
  }

  async execute<T>(_operation: string, fn: () => Promise<T>): Promise<T> {
    return await fn();
  }

  async onSuccess(): Promise<void> {
    // Mock success handler
  }

  async onFailure(): Promise<void> {
    // Mock failure handler
  }

  getState(): CircuitBreakerState {
    return CircuitBreakerState.CLOSED;
  }

  getStats() {
    return {
      state: CircuitBreakerState.CLOSED,
      failureCount: 0,
      successCount: 0,
    };
  }

  reset(): void {
    // Mock reset
  }
}

// ============================================================================
// RESILIENCE WRAPPER
// ============================================================================

export interface ResilienceConfig {
  retry?: {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    jitterMultiplier?: number;
  };
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  timeoutMs?: number;
  circuitBreaker?: CircuitBreakerConfig;
  deduplication?: {
    enabled?: boolean;
    ttl?: number;
    cacheResult?: boolean;
  };
  deadLetterQueue?: {
    enabled?: boolean;
    maxSize?: number;
  };
}

export const DEFAULT_RESILIENCE_CONFIG: ResilienceConfig = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 10000,
  timeoutMs: 30000,
  circuitBreaker: {
    failureThreshold: 5,
    successThreshold: 2,
    timeout: 60000,
    halfOpenAttempts: 3,
  },
};

export class ResilienceWrapper {
  private config: ResilienceConfig;
  private circuitBreaker: CircuitBreaker;
  private deduplicatorStats: Map<string, number> = new Map();
  private deadLetterQueue: Array<{ error: string; timestamp: number; data: any }> = [];

  constructor(config: ResilienceConfig = {}) {
    this.config = { ...DEFAULT_RESILIENCE_CONFIG, ...config };
    this.circuitBreaker = new CircuitBreaker(
      this.config.circuitBreaker || DEFAULT_RESILIENCE_CONFIG.circuitBreaker!
    );
  }

  async execute<T>(
    operation: string,
    fn: () => Promise<T>,
    _options?: { timeout?: number; retries?: number }
  ): Promise<T> {
    return await this.circuitBreaker.execute(operation, fn);
  }

  getCircuitBreaker(): CircuitBreaker {
    return this.circuitBreaker;
  }

  getCircuitBreakerState() {
    return this.circuitBreaker.getState();
  }

  getCircuitBreakerStats() {
    return this.circuitBreaker.getStats();
  }

  async resetCircuitBreaker(): Promise<void> {
    this.circuitBreaker.reset();
  }

  getDeduplicatorStats() {
    return {
      totalProcessed: Array.from(this.deduplicatorStats.values()).reduce((a, b) => a + b, 0),
      duplicates: this.deduplicatorStats.get('duplicates') || 0,
      byKey: Object.fromEntries(this.deduplicatorStats),
    };
  }

  getDeadLetterEntries() {
    return [...this.deadLetterQueue];
  }

  clearDeadLetterQueue(): void {
    this.deadLetterQueue = [];
  }
}

// ============================================================================
// RATE LIMITER
// ============================================================================

export interface RateLimiterConfig {
  requestsPerSecond?: number;
  maxRequests?: number;
  windowMs?: number;
  burstCapacity?: number;
}

export class RateLimiter {
  private requestCount: number = 0;
  private lastReset: number = Date.now();

  constructor(private config: RateLimiterConfig) {
    // Mock rate limiter
  }

  checkLimit(_key: string): boolean {
    const now = Date.now();
    const elapsed = now - this.lastReset;

    // Reset counter every second or window
    const windowMs = this.config.windowMs || 1000;
    if (elapsed >= windowMs) {
      this.requestCount = 0;
      this.lastReset = now;
    }

    const maxRequests = this.config.maxRequests || this.config.requestsPerSecond || 5;
    if (this.requestCount < maxRequests) {
      this.requestCount++;
      return true;
    }

    return false;
  }

  async acquire(): Promise<boolean> {
    return true;
  }

  async release(): Promise<void> {
    // Mock release
  }

  getStats() {
    const maxRequests = this.config.maxRequests || this.config.requestsPerSecond || 5;
    return {
      available: maxRequests,
      used: 0,
    };
  }
}

// ============================================================================
// INPUT VALIDATOR
// ============================================================================

export class InputValidator {
  static validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  static validateURL(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  static sanitizeString(input: string, maxLength: number = 1000): string {
    if (!input) return '';
    return input.substring(0, maxLength).trim();
  }

  static validateFilePath(filePath: string): boolean {
    // Prevent path traversal
    const normalized = filePath.replace(/\\/g, '/');
    return !normalized.includes('..') && !normalized.includes('~');
  }
}

// ============================================================================
// STRUCTURED LOGGER
// ============================================================================

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export interface LogContext {
  correlationId?: string;
  [key: string]: any;
}

export class StructuredLogger {
  private context: LogContext;

  constructor(context: LogContext = {}) {
    this.context = context;
  }

  debug(_message: string, _meta?: any): void {
    // Mock debug log
  }

  info(_message: string, _meta?: any): void {
    // Mock info log
  }

  warn(_message: string, _meta?: any): void {
    // Mock warn log
  }

  error(_message: string, _error?: Error | any, _meta?: any): void {
    // Mock error log
  }

  withContext(context: LogContext): StructuredLogger {
    return new StructuredLogger({ ...this.context, ...context });
  }

  static create(context?: LogContext): StructuredLogger {
    return new StructuredLogger(context);
  }
}

// ============================================================================
// ERROR SANITIZATION
// ============================================================================

export function sanitizeError(error: any): Error {
  if (error instanceof Error) {
    return error;
  }

  if (typeof error === 'string') {
    return new Error(error);
  }

  if (error && typeof error === 'object') {
    return new Error(error.message || JSON.stringify(error));
  }

  return new Error('Unknown error');
}

// ============================================================================
// CORRELATION ID GENERATOR
// ============================================================================

export function generateCorrelationId(): string {
  return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}

// ============================================================================
// ERROR CLASSIFICATION
// ============================================================================

export function isTransientError(error: unknown): boolean {
  if (error instanceof Error) {
    const errorMessage = error.message.toLowerCase();

    // Network errors
    if (
      errorMessage.includes('ECONNREFUSED') ||
      errorMessage.includes('ECONNRESET') ||
      errorMessage.includes('ETIMEDOUT') ||
      errorMessage.includes('ENOTFOUND') ||
      errorMessage.includes('EAI_AGAIN')
    ) {
      return true;
    }

    // HTTP errors that might be transient
    if (
      errorMessage.includes('503') || // Service Unavailable
      errorMessage.includes('502') || // Bad Gateway
      errorMessage.includes('504') || // Gateway Timeout
      errorMessage.includes('429')    // Too Many Requests
    ) {
      return true;
    }

    // Specific error messages
    if (
      errorMessage.includes('timeout') ||
      errorMessage.includes('temporary') ||
      errorMessage.includes('transient') ||
      errorMessage.includes('rate limit')
    ) {
      return true;
    }
  }
  return false;
}

// ============================================================================
// MOCK FACTORY
// ============================================================================

export function createMockResilienceWrapper(config?: ResilienceConfig): ResilienceWrapper {
  return new ResilienceWrapper(config);
}

export function createMockCircuitBreaker(config?: CircuitBreakerConfig): CircuitBreaker {
  return new CircuitBreaker(config || DEFAULT_RESILIENCE_CONFIG.circuitBreaker!);
}

export function createMockRateLimiter(config?: RateLimiterConfig): RateLimiter {
  return new RateLimiter(config || { requestsPerSecond: 10 });
}

export function createMockLogger(context?: LogContext): StructuredLogger {
  return StructuredLogger.create(context);
}
