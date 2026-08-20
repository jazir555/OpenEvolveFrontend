import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { createClient } from 'redis';

/**
 * Redis Bubble - In-Memory Data Store Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. set - Store a key-value pair with optional TTL
 * 2. get - Retrieve a value by key
 * 3. delete - Delete one or more keys
 * 4. exists - Check if one or more keys exist
 * 5. expire - Set a TTL on a key
 * 6. incr - Increment a numeric value
 * 7. decr - Decrement a numeric value
 * 8. hset - Set a field in a hash
 * 9. hget - Get a field from a hash
 * 10. hgetall - Get all fields and values from a hash
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const SetParamsSchema = z.object({
  operation: z.literal('set'),
  key: z.string().min(1, 'Key is required'),
  value: z.union([z.string(), z.number(), z.boolean(), z.record(z.unknown())]).describe('Value to store'),
  ttl: z.number().int().nonnegative().optional().describe('Time to live in seconds'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetParamsSchema = z.object({
  operation: z.literal('get'),
  key: z.string().min(1, 'Key is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteParamsSchema = z.object({
  operation: z.literal('delete'),
  keys: z.array(z.string()).min(1, 'At least one key is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ExistsParamsSchema = z.object({
  operation: z.literal('exists'),
  keys: z.array(z.string()).min(1, 'At least one key is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ExpireParamsSchema = z.object({
  operation: z.literal('expire'),
  key: z.string().min(1, 'Key is required'),
  ttl: z.number().int().positive().describe('Time to live in seconds'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const IncrParamsSchema = z.object({
  operation: z.literal('incr'),
  key: z.string().min(1, 'Key is required'),
  amount: z.number().int().default(1).describe('Amount to increment (default: 1)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DecrParamsSchema = z.object({
  operation: z.literal('decr'),
  key: z.string().min(1, 'Key is required'),
  amount: z.number().int().default(1).describe('Amount to decrement (default: 1)'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const HSetParamsSchema = z.object({
  operation: z.literal('hset'),
  key: z.string().min(1, 'Hash key is required'),
  field: z.string().min(1, 'Field name is required'),
  value: z.union([z.string(), z.number(), z.boolean()]).describe('Field value'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const HGetParamsSchema = z.object({
  operation: z.literal('hget'),
  key: z.string().min(1, 'Hash key is required'),
  field: z.string().min(1, 'Field name is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const HGetAllParamsSchema = z.object({
  operation: z.literal('hgetall'),
  key: z.string().min(1, 'Hash key is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const RedisBubbleParamsSchema = z.discriminatedUnion('operation', [
  SetParamsSchema,
  GetParamsSchema,
  DeleteParamsSchema,
  ExistsParamsSchema,
  ExpireParamsSchema,
  IncrParamsSchema,
  DecrParamsSchema,
  HSetParamsSchema,
  HGetParamsSchema,
  HGetAllParamsSchema,
]);

type RedisBubbleParams = z.input<typeof RedisBubbleParamsSchema>;

// Result schema
const RedisBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string(),
  meta: z.object({
    operation: z.string(),
    key: z.string().optional(),
  }),
});

type RedisBubbleResult = z.output<typeof RedisBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class RedisBubble extends ServiceBubble<RedisBubbleParams, RedisBubbleResult> {
  static readonly service = 'redis';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'redis';
  static readonly type = 'service' as const;
  static readonly schema = RedisBubbleParamsSchema;
  static readonly resultSchema = RedisBubbleResultSchema;
  static readonly shortDescription = 'In-memory data store for caching and real-time applications';
  static readonly longDescription = `
    Redis Bubble for high-performance key-value storage and caching.

    Features:
    - Ultra-fast in-memory operations
    - Key-value pairs with optional TTL
    - Hash data structures for complex objects
    - Atomic increments and decrements
    - Support for strings, numbers, booleans, and JSON
    - Persistence options and replication

    Use cases:
    - Caching frequently accessed data
    - Session storage
    - Rate limiting
    - Real-time leaderboards
    - Pub/sub messaging
    - Queue management
  `;
  static readonly alias = 'cache';

  private client: ReturnType<typeof createClient> | null = null;

  constructor(
    params: RedisBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.REDIS_CRED;
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Redis credentials are required');
    }
    return credentials[CredentialType.REDIS_CRED];
  }

  public async testCredential(): Promise<boolean> {
    try {
      const client = this.getClient();
      await client.ping();
      return true;
    } catch (error) {
      console.error('[Redis] Credential test failed:', error);
      return false;
    }
  }

  private getClient(): ReturnType<typeof createClient> {
    if (!this.client) {
      const credential = this.chooseCredential();
      if (!credential) {
        throw new Error('Redis credentials not found');
      }

      // Parse credential (expected format: JSON string with url)
      let config: any;
      try {
        config = typeof credential === 'string' ? JSON.parse(credential) : credential;
      } catch {
        throw new Error('Invalid Redis credentials format. Expected JSON string.');
      }

      if (!config.url) {
        throw new Error('Redis URL is required in credentials');
      }

      this.client = createClient({
        url: config.url,
      });

      // Connect the client
      this.client.connect().catch((err) => {
        console.error('[Redis] Connection failed:', err);
      });

      console.log('[Redis] Client initialized successfully');
    }

    return this.client;
  }

  protected async performAction(context?: BubbleContext): Promise<RedisBubbleResult> {
    void context;

    try {
      const client = this.getClient();
      const operation = this.params.operation;
      let result: any;

      console.log(`[Redis] Executing operation: ${operation}`);

      switch (operation) {
        case 'set':
          result = await this.set(client);
          break;

        case 'get':
          result = await this.get(client);
          break;

        case 'delete':
          result = await this.delete(client);
          break;

        case 'exists':
          result = await this.exists(client);
          break;

        case 'expire':
          result = await this.expire(client);
          break;

        case 'incr':
          result = await this.incr(client);
          break;

        case 'decr':
          result = await this.decr(client);
          break;

        case 'hset':
          result = await this.hset(client);
          break;

        case 'hget':
          result = await this.hget(client);
          break;

        case 'hgetall':
          result = await this.hgetall(client);
          break;

        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      return {
        success: true,
        data: result,
        error: '', // Empty string for successful operations,
        meta: {
          operation,
          key: this.extractKey(),
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Redis] Operation failed:`, errorMessage);

      return {
        success: false,
        data: null,
        error: errorMessage,
        meta: {
          operation: this.params.operation,
          key: this.extractKey(),
        },
      };
    }
  }

  private async set(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof SetParamsSchema>;

    // Convert value to string
    let valueStr: string;
    if (typeof params.value === 'object') {
      valueStr = JSON.stringify(params.value);
    } else {
      valueStr = String(params.value);
    }

    if (params.ttl) {
      await client.setEx(params.key, params.ttl, valueStr);
      console.log(`[Redis] Set key ${params.key} with TTL ${params.ttl}s`);
    } else {
      await client.set(params.key, valueStr);
      console.log(`[Redis] Set key ${params.key}`);
    }

    return {
      key: params.key,
      value: params.value,
      ttl: params.ttl,
      status: 'set',
    };
  }

  private async get(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof GetParamsSchema>;

    const value = await client.get(params.key);

    if (value === null) {
      console.log(`[Redis] Key ${params.key} not found`);
      return {
        key: params.key,
        value: null,
        exists: false,
      };
    }

    // Try to parse as JSON
    let parsedValue: any = value;
    try {
      parsedValue = JSON.parse(value);
    } catch {
      // Not JSON, return as-is
    }

    console.log(`[Redis] Got key ${params.key}`);

    return {
      key: params.key,
      value: parsedValue,
      exists: true,
    };
  }

  private async delete(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof DeleteParamsSchema>;

    const deleted = await client.del(params.keys);

    console.log(`[Redis] Deleted ${deleted} keys`);

    return {
      deletedCount: deleted,
      keys: params.keys,
    };
  }

  private async exists(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof ExistsParamsSchema>;

    const count = await client.exists(params.keys);

    console.log(`[Redis] ${count} keys exist`);

    return {
      existsCount: count,
      totalCount: params.keys.length,
      keys: params.keys,
    };
  }

  private async expire(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof ExpireParamsSchema>;

    const result = await client.expire(params.key, params.ttl);

    console.log(`[Redis] Set TTL ${params.ttl}s on key ${params.key}: ${result ? 'success' : 'failed'}`);

    return {
      key: params.key,
      ttl: params.ttl,
      success: result === 1,
    };
  }

  private async incr(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof IncrParamsSchema>;

    let newValue: number;
    if (params.amount === 1) {
      newValue = await client.incr(params.key);
    } else {
      newValue = await client.incrBy(params.key, params.amount);
    }

    console.log(`[Redis] Incremented ${params.key} by ${params.amount}, new value: ${newValue}`);

    return {
      key: params.key,
      value: newValue,
      incremented: params.amount,
    };
  }

  private async decr(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof DecrParamsSchema>;

    let newValue: number;
    if (params.amount === 1) {
      newValue = await client.decr(params.key);
    } else {
      newValue = await client.decrBy(params.key, params.amount);
    }

    console.log(`[Redis] Decremented ${params.key} by ${params.amount}, new value: ${newValue}`);

    return {
      key: params.key,
      value: newValue,
      decremented: params.amount,
    };
  }

  private async hset(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof HSetParamsSchema>;

    await client.hSet(params.key, params.field, String(params.value));

    console.log(`[Redis] Set hash ${params.key} field ${params.field}`);

    return {
      key: params.key,
      field: params.field,
      value: params.value,
      status: 'set',
    };
  }

  private async hget(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof HGetParamsSchema>;

    const value = await client.hGet(params.key, params.field);

    if (value === null) {
      console.log(`[Redis] Hash ${params.key} field ${params.field} not found`);
      return {
        key: params.key,
        field: params.field,
        value: null,
        exists: false,
      };
    }

    console.log(`[Redis] Got hash ${params.key} field ${params.field}`);

    return {
      key: params.key,
      field: params.field,
      value,
      exists: true,
    };
  }

  private async hgetall(client: ReturnType<typeof createClient>): Promise<any> {
    const params = this.params as z.output<typeof HGetAllParamsSchema>;

    const hash = await client.hGetAll(params.key);

    console.log(`[Redis] Got hash ${params.key} with ${Object.keys(hash).length} fields`);

    return {
      key: params.key,
      fields: hash,
      count: Object.keys(hash).length,
    };
  }

  private extractKey(): string | undefined {
    const params = this.params as any;
    return params.key || params.keys?.[0];
  }

  async cleanup(): Promise<void> {
    if (this.client) {
      await this.client.quit();
      this.client = null;
      console.log('[Redis] Client disconnected');
    }
  }
}

