import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * RedisBubble - Redis key-value store operations
 */
export class RedisBubble extends ServiceBubble<RedisParams, RedisResult> {
  bubbleName = 'redis';
  type = 'service';
  alias = 'Redis';
  credentialType = 'redis_api_key';

  params = {
    host: z.string().min(1),
    port: z.number().int().positive().default(6379),
    password: z.string().optional(),
    db: z.number().int().default(0),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const Redis = await import('ioredis');
    this.client = new Redis({
      host: this.params.host,
      port: this.params.port,
      password: this.params.password,
      db: this.params.db,
      connectTimeout: this.params.timeout
    });
  }

  async set(params: { key: string; value: string; expiration?: number }): Promise<RedisResult> {
    try {
      if (params.expiration) {
        await this.client.setex(params.key, params.expiration, params.value);
      } else {
        await this.client.set(params.key, params.value);
      }
      return { success: true, key: params.key };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async get(params: { key: string }): Promise<RedisResult> {
    try {
      const value = await this.client.get(params.key);
      return { success: true, value };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async delete(params: { keys: string[] }): Promise<RedisResult> {
    try {
      const count = await this.client.del(...params.keys);
      return { success: true, deleted: count };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async exists(params: { keys: string[] }): Promise<RedisResult> {
    try {
      const count = await this.client.exists(...params.keys);
      return { success: true, exists: count };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async expire(params: { key: string; seconds: number }): Promise<RedisResult> {
    try {
      await this.client.expire(params.key, params.seconds);
      return { success: true, key: params.key };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async incr(params: { key: string }): Promise<RedisResult> {
    try {
      const value = await this.client.incr(params.key);
      return { success: true, value };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async decr(params: { key: string }): Promise<RedisResult> {
    try {
      const value = await this.client.decr(params.key);
      return { success: true, value };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async hset(params: { key: string; field: string; value: string }): Promise<RedisResult> {
    try {
      await this.client.hset(params.key, params.field, params.value);
      return { success: true, key: params.key };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async hget(params: { key: string; field: string }): Promise<RedisResult> {
    try {
      const value = await this.client.hget(params.key, params.field);
      return { success: true, value };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async hgetAll(params: { key: string }): Promise<RedisResult> {
    try {
      const value = await this.client.hgetall(params.key);
      return { success: true, value };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async lpush(params: { key: string; values: string[] }): Promise<RedisResult> {
    try {
      const length = await this.client.lpush(params.key, ...params.values);
      return { success: true, length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async lrange(params: { key: string; start: number; stop: number }): Promise<RedisResult> {
    try {
      const values = await this.client.lrange(params.key, params.start, params.stop);
      return { success: true, values };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sadd(params: { key: string; members: string[] }): Promise<RedisResult> {
    try {
      const count = await this.client.sadd(params.key, ...params.members);
      return { success: true, added: count };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async smembers(params: { key: string }): Promise<RedisResult> {
    try {
      const members = await this.client.smembers(params.key);
      return { success: true, members };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface RedisParams {
  host: string;
  port?: number;
  password?: string;
  db?: number;
  timeout?: number;
}

export interface RedisResult {
  success: boolean;
  key?: string;
  value?: string | any;
  deleted?: number;
  exists?: number;
  added?: number;
  length?: number;
  members?: string[];
  values?: string[];
  error?: string;
}
