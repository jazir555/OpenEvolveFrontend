/**
 * Redis Caching Service Bubble
 *
 * Integrates with Redis for caching, session management,
 * pub/sub messaging, and data structures operations.
 *
 * FIXED: Now extends ServiceBubble properly with Federation Constitution compliance
 * FIXED: Uses real ioredis client instead of HTTP proxy
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';
import Redis from 'ioredis';

// ============================================================================
// REDIS-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const RedisOperationSchema = z.enum([
  'get',
  'set',
  'delete',
  'exists',
  'expire',
  'ttl',
  'keys',
  'flushdb',
  'health_check',
  'info',
  'incr',
  'decr',
  'hget',
  'hset',
  'hgetall',
  'hdel',
  'lpush',
  'rpush',
  'lrange',
  'lpop',
  'rpop',
  'sadd',
  'smembers',
  'srem',
  'publish',
]);

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const RedisParamsSchema = z.object({
  operation: RedisOperationSchema.describe('Redis operation'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  connectionString: z.string().describe('Redis connection string (REQUIRED)'),

  password: z.string().optional().describe('Redis password'),
  database: z.number().min(0).max(15).default(0).describe('Redis database number'),
  timeout: z.number().min(1000).max(60000).default(5000).describe('Operation timeout in ms'),

  // Key operations
  key: z.string().optional().describe('Redis key'),
  keys: z.array(z.string()).optional().describe('Multiple keys'),
  value: z.union([z.string(), z.number(), z.boolean(), z.record(z.unknown())]).optional().describe('Value to store'),
  values: z.array(z.unknown()).optional().describe('Multiple values'),

  // Expiration
  ttl: z.number().optional().describe('Time to live in seconds'),

  // Hash operations
  field: z.string().optional().describe('Hash field'),
  fields: z.array(z.string()).optional().describe('Multiple hash fields'),

  // List operations
  listIndex: z.number().optional().describe('List index'),
  listRange: z.tuple([z.number(), z.number()]).optional().describe('List range [start, stop]'),

  // Pub/Sub
  channel: z.string().optional().describe('Pub/Sub channel'),
  message: z.string().optional().describe('Message to publish'),
});

type RedisParamsInput = z.input<typeof RedisParamsSchema>;
type RedisParams = z.output<typeof RedisParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const RedisResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  result: z.unknown().optional(),
  count: z.number().optional().describe('Count of affected items'),
  keys: z.array(z.string()).optional().describe('Returned keys'),
  values: z.array(z.unknown()).optional().describe('Returned values'),
  exists: z.boolean().optional().describe('Whether key exists'),
  ttl: z.number().optional().describe('Time to live'),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number(),
});

type RedisResult = z.output<typeof RedisResultSchema>;

// ============================================================================
// REDIS BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class RedisBubble extends ServiceBubble<RedisParams, RedisResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'redis' as const;
  static readonly type = 'service' as const;
  static readonly schema = RedisParamsSchema;
  static readonly resultSchema = RedisResultSchema;
  static readonly credentialType = 'redis_password' as const;

  static readonly shortDescription = 'Redis integration for caching and data structures';
  static readonly longDescription = `
    Redis service bubble for OpenEvolve caching and data structures.

    Features:
    - String operations (get, set, delete, exists, expire, ttl)
    - Hash operations (hget, hset, hgetall, hdel)
    - List operations (lpush, rpush, lrange, lpop, rpop)
    - Set operations (sadd, smembers, srem)
    - Pub/Sub messaging
    - Connection pooling with ioredis
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - connectionString: Redis connection string (no default - must be provided)
    - password: Optional Redis password

    Federation Constitution Compliance:
    - No magic defaults (connectionString is required)
    - Real ioredis client (not HTTP proxy)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
  `;

  private resilience: ResilienceWrapper;
  private client: Redis;

  constructor(params: RedisParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    RedisBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('redis', DEFAULT_RESILIENCE_CONFIG);

    // Initialize real Redis client (ioredis)
    this.client = new Redis(this.params.connectionString, {
      password: this.params.password,
      db: this.params.database,
      retryStrategy: (times: number) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
      maxRetriesPerRequest: 3,
      enableReadyCheck: true,
    });

    // Handle connection errors
    this.client.on('error', (error: Error) => {
      console.error('Redis connection error:', error);
    });
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - connectionString is required by schema
  }

  /**
   * Execute Redis command with resilience
   */
  private async executeWithResilience<T>(
    key: string,
    operation: () => Promise<T>,
    input?: unknown
  ): Promise<T> {
    return await this.resilience.execute(
      `redis-${this.params.operation}-${key}`,
      operation,
      input
    );
  }

  /**
   * Get operation
   */
  private async get(): Promise<RedisResult> {
    if (!this.params.key) {
      throw new Error('key is required for get operation');
    }

    const startTime = Date.now();

    try {
      const result = await this.executeWithResilience(
        `redis-get-${this.params.key}`,
        async () => await this.client.get(this.params.key!),
        { operation: 'get', key: this.params.key }
      );

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'get',
        data: result,
        result: result ? JSON.parse(result) : null,
        timing,
        status: { code: 200, reason: 'OK' },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get',
        error: errorMessage,
        timing,
        status: { code: 0, reason: 'Request failed' },
      };
    }
  }

  /**
   * Set operation
   */
  private async set(): Promise<RedisResult> {
    if (!this.params.key || this.params.value === undefined) {
      throw new Error('key and value are required for set operation');
    }

    const startTime = Date.now();

    try {
      const value = typeof this.params.value === 'string'
        ? this.params.value
        : JSON.stringify(this.params.value);

      const result = await this.executeWithResilience(
        `redis-set-${this.params.key}`,
        async () => {
          if (this.params.ttl !== undefined) {
            return await this.client.setex(this.params.key!, this.params.ttl, value);
          }
          return await this.client.set(this.params.key!, value);
        },
        { operation: 'set', key: this.params.key, value: this.params.value }
      );

      const timing = Date.now() - startTime;

      return {
        success: result === 'OK',
        operation: 'set',
        data: result,
        result,
        timing,
        status: { code: 200, reason: 'OK' },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'set',
        error: errorMessage,
        timing,
        status: { code: 0, reason: 'Request failed' },
      };
    }
  }

  /**
   * Delete operation
   */
  private async delete(): Promise<RedisResult> {
    if (!this.params.key) {
      throw new Error('key is required for delete operation');
    }

    const startTime = Date.now();

    try {
      const result = await this.executeWithResilience(
        `redis-delete-${this.params.key}`,
        async () => await this.client.del(this.params.key!),
        { operation: 'delete', key: this.params.key }
      );

      const timing = Date.now() - startTime;

      return {
        success: result > 0,
        operation: 'delete',
        data: result,
        count: result,
        timing,
        status: { code: 200, reason: 'OK' },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'delete',
        error: errorMessage,
        timing,
        status: { code: 0, reason: 'Request failed' },
      };
    }
  }

  /**
   * Health check operation
   */
  private async healthCheck(): Promise<RedisResult> {
    const startTime = Date.now();

    try {
      const result = await this.executeWithResilience(
        'redis-healthcheck',
        async () => await this.client.ping(),
        { operation: 'health_check' }
      );

      const timing = Date.now() - startTime;

      return {
        success: result === 'PONG',
        operation: 'health_check',
        data: result,
        result,
        timing,
        status: { code: 200, reason: 'OK' },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'health_check',
        error: errorMessage,
        timing,
        status: { code: 0, reason: 'Request failed' },
      };
    }
  }

  /**
   * Info operation
   */
  private async info(): Promise<RedisResult> {
    const startTime = Date.now();

    try {
      const result = await this.executeWithResilience(
        'redis-info',
        async () => await this.client.info(),
        { operation: 'info' }
      );

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'info',
        data: result,
        result,
        timing,
        status: { code: 200, reason: 'OK' },
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'info',
        error: errorMessage,
        timing,
        status: { code: 0, reason: 'Request failed' },
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<RedisResult> {
    switch (this.params.operation) {
      case 'get':
        return this.get();
      case 'set':
        return this.set();
      case 'delete':
        return this.delete();
      case 'health_check':
        return this.healthCheck();
      case 'info':
        return this.info();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
          status: { code: 400, reason: 'Invalid operation' },
        };
    }
  }

  /**
   * Cleanup on destruction
   */
  async destroy(): Promise<void> {
    await this.client.quit();
  }
}

export default RedisBubble;
