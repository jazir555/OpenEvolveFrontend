/**
 * Resilience Patterns for OpenEvolve Integration
 *
 * Implements Federation Constitution requirements:
 * - Circuit Breaker pattern for fault tolerance
 * - Exponential Backoff Retry with jitter for transient failures
 * - Request Deduplication for idempotency
 * - Dead Letter Queue for permanent failures
 * - Rate Limiting for API protection
 * - Input Validation for security
 * - Structured Logging for observability
 */

import crypto from 'crypto';

// ============================================================================
// CIRCUIT BREAKER (Thread-safe with state locking)
// ============================================================================

/**
 * Circuit Breaker States
 */
export enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

/**
 * Circuit Breaker Configuration
 */
export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  halfOpenAttempts: number;
}

/**
 * Thread-safe Circuit Breaker Implementation
 * Prevents cascading failures by temporarily disabling failing services
 */
export class CircuitBreaker {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenAttemptCount = 0;
  private stateLock = false; // Simple mutex for state transitions

  constructor(private config: CircuitBreakerConfig) {
    this.validateConfig();
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async execute<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    // Acquire lock for state check
    await this.acquireStateLock();

    try {
      if (this.state === CircuitBreakerState.OPEN) {
        if (this.shouldAttemptReset()) {
          this.transitionTo(CircuitBreakerState.HALF_OPEN);
          this.halfOpenAttemptCount = 0;
        } else {
          throw new Error(
            `Circuit breaker is OPEN for operation: ${operation}. ` +
            `Last failure: ${new Date(this.lastFailureTime).toISOString()}. ` +
            `Retry after: ${new Date(this.lastFailureTime + this.config.timeout).toISOString()}`
          );
        }
      }

      return await fn();
    } finally {
      this.releaseStateLock();
    }
  }

  /**
   * Record successful operation
   */
  async onSuccess(): Promise<void> {
    await this.acquireStateLock();

    try {
      this.failureCount = 0;

      if (this.state === CircuitBreakerState.HALF_OPEN) {
        this.halfOpenAttemptCount++;

        if (this.halfOpenAttemptCount >= this.config.halfOpenAttempts) {
          this.transitionTo(CircuitBreakerState.CLOSED);
        }
      }
    } finally {
      this.releaseStateLock();
    }
  }

  /**
   * Record failed operation
   */
  async onFailure(): Promise<void> {
    await this.acquireStateLock();

    try {
      this.failureCount++;
      this.lastFailureTime = Date.now();

      if (this.failureCount >= this.config.failureThreshold) {
        this.transitionTo(CircuitBreakerState.OPEN);
      }
    } finally {
      this.releaseStateLock();
    }
  }

  /**
   * Get current circuit breaker state
   */
  getState(): CircuitBreakerState {
    return this.state;
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
      halfOpenAttemptCount: this.halfOpenAttemptCount,
    };
  }

  /**
   * Reset circuit breaker to CLOSED state
   */
  async reset(): Promise<void> {
    await this.acquireStateLock();

    try {
      this.state = CircuitBreakerState.CLOSED;
      this.failureCount = 0;
      this.successCount = 0;
      this.lastFailureTime = 0;
      this.halfOpenAttemptCount = 0;
    } finally {
      this.releaseStateLock();
    }
  }

  private transitionTo(newState: CircuitBreakerState): void {
    const oldState = this.state;
    this.state = newState;

    // Log state transition (in production, use proper logger)
    console.log(`[CircuitBreaker] State transition: ${oldState} -> ${newState}`);
  }

  private shouldAttemptReset(): boolean {
    return Date.now() - this.lastFailureTime > this.config.timeout;
  }

  private async acquireStateLock(): Promise<void> {
    // Simple spinlock implementation
    while (this.stateLock) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    this.stateLock = true;
  }

  private releaseStateLock(): void {
    this.stateLock = false;
  }

  private validateConfig(): void {
    if (this.config.failureThreshold <= 0) {
      throw new Error('failureThreshold must be positive');
    }
    if (this.config.successThreshold <= 0) {
      throw new Error('successThreshold must be positive');
    }
    if (this.config.timeout <= 0) {
      throw new Error('timeout must be positive');
    }
    if (this.config.halfOpenAttempts <= 0) {
      throw new Error('halfOpenAttempts must be positive');
    }
  }
}

// ============================================================================
// EXPONENTIAL BACKOFF RETRY WITH JITTER
// ============================================================================

/**
 * Retry Configuration
 */
export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  jitterMultiplier: number;
}

/**
 * Error classification for retry logic
 */
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

/**
 * Retry with exponential backoff and jitter
 * Implements Federation Constitution requirement for fault tolerance
 */
export async function retryWithBackoff<T>(
  operation: string,
  fn: () => Promise<T>,
  config: RetryConfig
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt < config.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry if error is permanent
      if (!isTransientError(lastError)) {
        throw lastError;
      }

      // Don't retry on last attempt
      if (attempt === config.maxRetries - 1) {
        throw lastError;
      }

      // Calculate delay with exponential backoff and jitter
      const exponentialDelay = config.baseDelay * Math.pow(2, attempt);
      const jitter = Math.random() * config.jitterMultiplier * exponentialDelay;
      const delay = Math.min(exponentialDelay + jitter, config.maxDelay);

      console.log(
        `[Retry] Operation "${operation}" failed (attempt ${attempt + 1}/${config.maxRetries}). ` +
        `Retrying in ${Math.round(delay)}ms. Error: ${lastError.message}`
      );

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

// ============================================================================
// REQUEST DEDUPLICATION
// ============================================================================

/**
 * Request deduplicator for idempotency
 * Ensures that identical in-flight requests are not duplicated
 */
export class RequestDeduplicator {
  private pendingRequests = new Map<string, Promise<any>>();
  private completedRequests = new Map<string, { result: any; timestamp: number }>();
  private readonly ttl: number; // Time to cache completed results

  constructor(ttl = 60000) {
    this.ttl = ttl;
    this.startCleanupTimer();
  }

  /**
   * Execute operation with deduplication
   * If a request with the same key is in flight, return its promise
   */
  async execute<T>(
    key: string,
    operation: () => Promise<T>,
    options?: { cacheResult?: boolean; skipCache?: boolean }
  ): Promise<T> {
    // Check if we have a cached result
    if (options?.cacheResult && !options?.skipCache) {
      const cached = this.completedRequests.get(key);
      if (cached && Date.now() - cached.timestamp < this.ttl) {
        console.log(`[Deduplicator] Cache hit for key: ${key}`);
        return cached.result as T;
      }
    }

    // Check if request is in flight
    const inFlight = this.pendingRequests.get(key);
    if (inFlight) {
      console.log(`[Deduplicator] Request in flight for key: ${key}. Waiting...`);
      return inFlight as T;
    }

    // Execute new request
    const promise = operation()
      .then(result => {
        // Cache result if enabled
        if (options?.cacheResult) {
          this.completedRequests.set(key, { result, timestamp: Date.now() });
        }
        return result;
      })
      .finally(() => {
        // Clean up pending request
        this.pendingRequests.delete(key);
      });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  /**
   * Clear cached results
   */
  clearCache(): void {
    this.completedRequests.clear();
  }

  /**
   * Clear cache for specific key
   */
  clearKey(key: string): void {
    this.completedRequests.delete(key);
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      pendingRequests: this.pendingRequests.size,
      completedRequests: this.completedRequests.size,
    };
  }

  /**
   * Start periodic cleanup of expired cache entries
   */
  private startCleanupTimer(): void {
    setInterval(() => {
      const now = Date.now();
      for (const [key, value] of this.completedRequests.entries()) {
        if (now - value.timestamp > this.ttl) {
          this.completedRequests.delete(key);
        }
      }
    }, this.ttl / 2); // Cleanup twice per TTL
  }
}

// ============================================================================
// DEAD LETTER QUEUE
// ============================================================================

/**
 * Dead Letter Queue Entry
 */
export interface DeadLetterEntry {
  id: string;
  operation: string;
  input: unknown;
  error: Error;
  timestamp: number;
  retryCount: number;
}

/**
 * In-memory Dead Letter Queue
 * In production, this should persist to a database or message queue
 */
export class DeadLetterQueue {
  private queue: DeadLetterEntry[] = [];
  private readonly maxSize: number;

  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
  }

  /**
   * Add failed operation to DLQ
   */
  async push(
    operation: string,
    input: unknown,
    error: Error,
    retryCount = 0
  ): Promise<void> {
    const entry: DeadLetterEntry = {
      id: `${operation}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      operation,
      input,
      error,
      timestamp: Date.now(),
      retryCount,
    };

    // Evict oldest entry if queue is full
    if (this.queue.length >= this.maxSize) {
      this.queue.shift();
    }

    this.queue.push(entry);

    console.error(
      `[DLQ] Added to dead letter queue: ${entry.id}. ` +
      `Operation: ${operation}, Error: ${error.message}`
    );
  }

  /**
   * Consume entries from DLQ
   */
  async consume(limit = 10): Promise<DeadLetterEntry[]> {
    return this.queue.splice(0, limit);
  }

  /**
   * Get all entries
   */
  getAll(): DeadLetterEntry[] {
    return [...this.queue];
  }

  /**
   * Clear queue
   */
  clear(): void {
    this.queue = [];
  }

  /**
   * Get queue size
   */
  size(): number {
    return this.queue.length;
  }
}

// ============================================================================
// COMBINED RESILIENCE WRAPPER
// ============================================================================

/**
 * Combined resilience configuration
 */
export interface ResilienceConfig {
  circuitBreaker?: CircuitBreakerConfig;
  retry?: RetryConfig;
  deduplication?: {
    enabled: boolean;
    ttl?: number;
    cacheResult?: boolean;
  };
  deadLetterQueue?: {
    enabled: boolean;
    maxSize?: number;
  };
}

/**
 * Resilience wrapper that combines all patterns
 */
export class ResilienceWrapper {
  private circuitBreaker?: CircuitBreaker;
  private deduplicator?: RequestDeduplicator;
  private dlq?: DeadLetterQueue;
  private retryConfig: RetryConfig;

  constructor(
    private operationName: string,
    config: ResilienceConfig
  ) {
    // Initialize circuit breaker
    if (config.circuitBreaker) {
      this.circuitBreaker = new CircuitBreaker(config.circuitBreaker);
    }

    // Initialize deduplicator
    if (config.deduplication?.enabled) {
      this.deduplicator = new RequestDeduplicator(config.deduplication.ttl);
    }

    // Initialize DLQ
    if (config.deadLetterQueue?.enabled) {
      this.dlq = new DeadLetterQueue(config.deadLetterQueue.maxSize);
    }

    // Set retry config (with defaults)
    this.retryConfig = config.retry || {
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 30000,
      jitterMultiplier: 0.1,
    };
  }

  /**
   * Execute operation with full resilience protection
   */
  async execute<T>(
    key: string,
    operation: () => Promise<T>,
    input?: unknown
  ): Promise<T> {
    const wrappedOperation = async () => {
      try {
        // Circuit breaker protection
        if (this.circuitBreaker) {
          return await this.circuitBreaker.execute(this.operationName, async () => {
            const result = await operation();
            await this.circuitBreaker!.onSuccess();
            return result;
          });
        }

        return await operation();
      } catch (error) {
        // Record failure in circuit breaker
        if (this.circuitBreaker) {
          await this.circuitBreaker.onFailure();
        }

        // Add to DLQ for permanent failures
        if (this.dlq && !isTransientError(error)) {
          await this.dlq.push(this.operationName, input, error as Error);
        }

        throw error;
      }
    };

    // Apply retry logic
    const retriedOperation = () => retryWithBackoff(
      this.operationName,
      wrappedOperation,
      this.retryConfig
    );

    // Apply deduplication
    if (this.deduplicator) {
      return this.deduplicator.execute(
        key,
        retriedOperation,
        { cacheResult: true }
      );
    }

    return retriedOperation();
  }

  /**
   * Get circuit breaker state
   */
  getCircuitBreakerState(): CircuitBreakerState | undefined {
    return this.circuitBreaker?.getState();
  }

  /**
   * Get circuit breaker stats
   */
  getCircuitBreakerStats() {
    return this.circuitBreaker?.getStats();
  }

  /**
   * Get deduplicator stats
   */
  getDeduplicatorStats() {
    return this.deduplicator?.getStats();
  }

  /**
   * Get DLQ entries
   */
  getDeadLetterEntries(): DeadLetterEntry[] {
    return this.dlq?.getAll() || [];
  }

  /**
   * Clear DLQ
   */
  clearDeadLetterQueue(): void {
    this.dlq?.clear();
  }

  /**
   * Reset circuit breaker
   */
  async resetCircuitBreaker(): Promise<void> {
    if (this.circuitBreaker) {
      await this.circuitBreaker.reset();
    }
  }
}

// ============================================================================
// RATE LIMITING (Token Bucket Implementation)
// ============================================================================

export interface RateLimitConfig {
  readonly maxRequests: number;
  readonly windowMs: number;
}

export class RateLimiter {
  private requests: Map<string, { count: number; resetTime: number }>;

  constructor(private config: RateLimitConfig) {
    this.requests = new Map();
  }

  checkLimit(identifier: string): boolean {
    const now = Date.now();
    const key = identifier || 'anonymous';

    let record = this.requests.get(key);

    if (!record || now > record.resetTime) {
      record = {
        count: 0,
        resetTime: now + this.config.windowMs
      };
      this.requests.set(key, record);
    }

    record.count++;

    return record.count <= this.config.maxRequests;
  }

  getRemainingRequests(identifier: string): number {
    const record = this.requests.get(identifier);
    if (!record) return this.config.maxRequests;

    const now = Date.now();
    if (now > record.resetTime) return this.config.maxRequests;

    return Math.max(0, this.config.maxRequests - record.count);
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, record] of this.requests.entries()) {
      if (now > record.resetTime) {
        this.requests.delete(key);
      }
    }
  }
}

// ============================================================================
// INPUT VALIDATION
// ============================================================================

export class InputValidator {
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

  static validateApiKey(apiKey: string): string {
    if (!apiKey || typeof apiKey !== 'string') {
      throw new Error('Invalid API key format');
    }
    return apiKey;
  }
}

// ============================================================================
// STRUCTURED LOGGING
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
      ...(error ? { error: sanitizeError(error) } : {}),
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
// ERROR SANITIZATION
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
// CORRELATION ID MANAGEMENT
// ============================================================================

export function generateCorrelationId(): string {
  return crypto.randomBytes(16).toString('hex');
}

// ============================================================================
// DEFAULT CONFIGURATIONS
// ============================================================================

export const DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfig = {
  failureThreshold: 5,
  successThreshold: 2,
  timeout: 60000, // 1 minute
  halfOpenAttempts: 3,
};

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  jitterMultiplier: 0.1,
};

export const DEFAULT_RESILIENCE_CONFIG: ResilienceConfig = {
  circuitBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG,
  retry: DEFAULT_RETRY_CONFIG,
  deduplication: {
    enabled: true,
    ttl: 60000,
    cacheResult: true,
  },
  deadLetterQueue: {
    enabled: true,
    maxSize: 1000,
  },
};
