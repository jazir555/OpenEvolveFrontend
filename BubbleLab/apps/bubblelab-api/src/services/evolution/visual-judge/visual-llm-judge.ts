import { createHash } from 'crypto';
import { CredentialType } from '@bubblelab/shared-schemas';
import { AccessibilityAgent } from './agents/AccessibilityAgent.js';
import { BrandAgent } from './agents/BrandAgent.js';
import { ConversionAgent } from './agents/ConversionAgent.js';
import { LayoutAgent } from './agents/LayoutAgent.js';
import { cacheService } from '../../cache.js';
import type {
  JudgeAggregateResult,
  JudgeEvaluation,
  JudgeInput,
} from './types.js';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

class RateLimiter {
  private queue = Promise.resolve();
  private lastRun = 0;

  constructor(private readonly minIntervalMs: number) {}

  async schedule<T>(fn: () => Promise<T>): Promise<T> {
    const task = async () => {
      const now = Date.now();
      const wait = Math.max(0, this.minIntervalMs - (now - this.lastRun));
      if (wait > 0) {
        await sleep(wait);
      }
      this.lastRun = Date.now();
      return await fn();
    };

    this.queue = this.queue.then(task, task) as Promise<void>;
    return this.queue as Promise<T>;
  }
}

const defaultWeights = {
  LayoutAgent: 0.35,
  AccessibilityAgent: 0.2,
  BrandAgent: 0.2,
  ConversionAgent: 0.25,
};

export type VisualJudgeConfig = {
  weights?: Record<string, number>;
  cacheTtlMs?: number;
  maxRetries?: number;
  rateLimitMsByProvider?: Record<string, number>;
};

export class VisualLLMJudge {
  private readonly agents = [
    new LayoutAgent(),
    new AccessibilityAgent(),
    new BrandAgent(),
    new ConversionAgent(),
  ];
  private readonly rateLimiters = new Map<string, RateLimiter>();

  constructor(private readonly config: VisualJudgeConfig = {}) {}

  private getLimiter(provider: string): RateLimiter {
    const defaultRateMs = 150;
    const limit = this.config.rateLimitMsByProvider?.[provider] ?? defaultRateMs;
    if (!this.rateLimiters.has(provider)) {
      this.rateLimiters.set(provider, new RateLimiter(limit));
    }
    return this.rateLimiters.get(provider)!;
  }

  private cacheKey(agentName: string, input: JudgeInput): string {
    const hash = createHash('sha256');
    hash.update(agentName);
    hash.update(JSON.stringify(input));
    return hash.digest('hex');
  }

  private getCached(agentName: string, input: JudgeInput): JudgeEvaluation | null {
    const key = this.cacheKey(agentName, input);
    return cacheService.getJson<JudgeEvaluation>(`visual-judge:${key}`);
  }

  private setCached(agentName: string, input: JudgeInput, result: JudgeEvaluation) {
    const ttl = this.config.cacheTtlMs ?? 15 * 60 * 1000;
    const key = this.cacheKey(agentName, input);
    cacheService.setJson(`visual-judge:${key}`, result, ttl);
  }

  private async withRetries<T>(fn: () => Promise<T>): Promise<T> {
    const maxRetries = this.config.maxRetries ?? 2;
    let attempt = 0;
    let lastError: Error | undefined;

    while (attempt <= maxRetries) {
      try {
        return await fn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        const delay = Math.min(2000, 200 * 2 ** attempt);
        await sleep(delay);
      }
      attempt += 1;
    }

    throw lastError ?? new Error('Visual judge failed');
  }

  async evaluate(
    input: JudgeInput,
    credentials: Record<CredentialType, string>,
    weightsOverride?: Record<string, number>
  ): Promise<JudgeAggregateResult> {
    const weights = {
      ...defaultWeights,
      ...(this.config.weights ?? {}),
      ...(weightsOverride ?? {}),
    };

    const results = await Promise.all(
      this.agents.map(async (agent) => {
        const agentName = agent.agent;
        const cached = this.getCached(agentName, input);
        if (cached) return cached;

        const provider = agent.providerName;
        const limiter = this.getLimiter(provider);
        const result = await limiter.schedule(() =>
          this.withRetries(() => agent.evaluate(input, credentials))
        );
        this.setCached(agentName, input, result);
        return result;
      })
    );

    const totalWeight = results.reduce(
      (sum, result) => sum + (weights[result.agent] ?? 0),
      0
    );
    const score =
      results.reduce(
        (sum, result) => sum + result.score * (weights[result.agent] ?? 0),
        0
      ) / (totalWeight || 1);

    return {
      score: Number(score.toFixed(4)),
      weights,
      agents: results,
    };
  }

  async evaluateBatch(
    inputs: JudgeInput[],
    credentials: Record<CredentialType, string>,
    maxConcurrency = 3
  ): Promise<JudgeAggregateResult[]> {
    const queue = [...inputs];
    const results: JudgeAggregateResult[] = [];

    const worker = async () => {
      while (queue.length) {
        const input = queue.shift();
        if (!input) return;
        const result = await this.evaluate(input, credentials);
        results.push(result);
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(maxConcurrency, inputs.length) }, () =>
        worker()
      )
    );

    return results;
  }
}
