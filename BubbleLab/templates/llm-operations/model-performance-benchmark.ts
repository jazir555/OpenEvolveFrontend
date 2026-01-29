/**
 * Model Performance Benchmark
 * Purpose: Benchmark model performance across various tasks
 * Category: LLM Operations
 * Event Type: schedule/cron
 * Schedule: 0 3 * * 0 (Weekly on Sunday at 3 AM)
 *
 * Required Credentials:
 * - OPENAI_API_KEY: For GPT models
 * - ANTHROPIC_API_KEY: For Claude models
 * - GOOGLE_API_KEY: For Gemini models
 * - POSTGRES_CONNECTION_STRING: To store benchmark results
 * - API_KEY: API key for authentication (required)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - API key authentication
 * - Input validation for all user inputs (Zod schemas)
 * - Rate limiting
 * - SQL injection prevention with parameterized queries
 * - Error message sanitization
 * - Structured logging with correlation IDs
 */

import {
  BubbleFlow,
  AIAgentBubble,
  PostgreSQLBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  buildParameterizedQuery,
} from '../security-utils';

// Input validation schemas
const ModelSchema = z.string().min(1).max(100);
const ApiKeySchema = z.string().min(32).max(256);
const BenchmarkTaskSchema = z.object({
  name: z.string().min(1).max(255),
  prompt: z.string().min(1).max(10000),
  category: z.enum(['reasoning', 'coding', 'writing', 'analysis', 'math']),
  expectedLatency: z.number().int().min(0).max(60000),
});

interface BenchmarkTask {
  name: string;
  prompt: string;
  category: 'reasoning' | 'coding' | 'writing' | 'analysis' | 'math';
  expectedLatency: number;
}

interface ModelBenchmark {
  model: string;
  task: string;
  latency: number;
  tokenThroughput: number;
  quality: number;
  cost: number;
}

interface BenchmarkResult {
  timestamp: string;
  tasks: number;
  models: number;
  benchmarks: ModelBenchmark[];
  rankings: {
    fastest: string;
    highestQuality: string;
    mostCostEffective: string;
  };
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});

export class ModelPerformanceBenchmark extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 3 * * 0';
  readonly name = 'Model Performance Benchmark';
  readonly description = 'Benchmark model performance across various tasks';

  private logger = new StructuredLogger('model-performance-benchmark');
  private rateLimiter = new RateLimiter({
    maxRequests: 1, // Benchmarks are expensive, limit to once per day
    windowMs: 86400000, // 24 hours
  });

  async handle(payload: CronEvent): Promise<BenchmarkResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 1 benchmark per 24 hours.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting model performance benchmark',
    });

    // Define models to benchmark
    const models = [
      'openai/gpt-4',
      'openai/gpt-4-turbo',
      'openai/gpt-3.5-turbo',
      'anthropic/claude-sonnet-4-20250514',
      'anthropic/claude-3-5-sonnet-20241022',
      'google/gemini-2.5-flash',
    ].filter(m => {
      try {
        ModelSchema.parse(m);
        return true;
      } catch {
        return false;
      }
    });

    // Define benchmark tasks
    const tasks: BenchmarkTask[] = [
      {
        name: 'Complex Reasoning',
        prompt: 'Explain the concept of recursion in computer science with three practical examples.',
        category: 'reasoning',
        expectedLatency: 5000,
      },
      {
        name: 'Code Generation',
        prompt: 'Write a Python function that implements a binary search algorithm with proper error handling.',
        category: 'coding',
        expectedLatency: 3000,
      },
      {
        name: 'Creative Writing',
        prompt: 'Write a short story about a robot learning to paint. Keep it under 200 words.',
        category: 'writing',
        expectedLatency: 4000,
      },
      {
        name: 'Data Analysis',
        prompt: 'Analyze the pros and cons of microservices vs monolithic architecture for a startup with 10 engineers.',
        category: 'analysis',
        expectedLatency: 5000,
      },
      {
        name: 'Math Problem',
        prompt: 'Calculate the compound interest on $10,000 at 5% annual rate compounded monthly for 3 years.',
        category: 'math',
        expectedLatency: 2000,
      },
    ];

    const allBenchmarks: ModelBenchmark[] = [];

    // Run benchmarks
    for (const model of models) {
      for (const task of tasks) {
        try {
          const startTime = Date.now();

          // Sanitize prompt
          const sanitizedPrompt = InputValidator.sanitizeString(task.prompt, 10000);

          // Execute task
          const agent = new AIAgentBubble({
            model: { model },
            systemPrompt: 'You are a helpful AI assistant.',
            message: sanitizedPrompt,
          });

          const response = await agent.action();

          const latency = Date.now() - startTime;
          const tokens = response.usage?.total_tokens || 0;
          const tokenThroughput = parseFloat((tokens / (latency / 1000)).toFixed(2));

          // Calculate cost (estimated pricing per 1M tokens)
          const costPerMillion = this.getModelCost(model);
          const cost = (tokens / 1000000) * costPerMillion;

          // Evaluate quality (simplified - would use more sophisticated metrics)
          const quality = await this.evaluateQuality(response.data.response, task.category);

          allBenchmarks.push({
            model,
            task: InputValidator.sanitizeString(task.name, 255),
            latency: InputValidator.sanitizeNumber(latency, 0, 60000),
            tokenThroughput: InputValidator.sanitizeNumber(tokenThroughput, 0, 10000),
            quality: InputValidator.sanitizeNumber(quality, 0, 100),
            cost: InputValidator.sanitizeNumber(cost, 0, 100),
          });

        } catch (error) {
          this.logger.warn({
            msg: 'Benchmark failed',
            model,
            task: task.name,
          }, error);
        }
      }
    }

    // Calculate rankings
    const rankings = this.calculateRankings(allBenchmarks);

    const result: BenchmarkResult = {
      timestamp,
      tasks: tasks.length,
      models: models.length,
      benchmarks: allBenchmarks,
      rankings,
      correlationId,
    };

    // Store results
    // Security: SQL injection prevention - use parameterized queries
    for (const benchmark of allBenchmarks) {
      const storeBenchmarkQuery = buildParameterizedQuery(
        `
          INSERT INTO model_benchmarks (
            timestamp, model, task, latency, token_throughput, quality, cost
          )
          VALUES ($1, $2, $3, $4, $5, $6, $7)
        `,
        [
          timestamp,
          benchmark.model,
          benchmark.task,
          benchmark.latency,
          benchmark.tokenThroughput,
          benchmark.quality,
          benchmark.cost,
        ]
      );

      try {
        const storeBenchmark = new PostgreSQLBubble({
          connectionString: process.env.POSTGRES_CONNECTION_STRING,
          query: storeBenchmarkQuery.query,
          params: storeBenchmarkQuery.params,
        });

        await storeBenchmark.action();
      } catch (error) {
        this.logger.error({
          msg: 'Failed to store benchmark',
          model: benchmark.model,
          task: benchmark.task,
        }, error);
        // Don't throw - storage failure shouldn't break the workflow
      }
    }

    this.logger.info({
      msg: 'Model performance benchmark completed',
      totalBenchmarks: allBenchmarks.length,
      fastest: rankings.fastest,
      highestQuality: rankings.highestQuality,
      mostCostEffective: rankings.mostCostEffective,
    });

    return result;
  }

  private getModelCost(model: string): number {
    // Estimated cost per 1M tokens (input + output average)
    const costs: Record<string, number> = {
      'openai/gpt-4': 30,
      'openai/gpt-4-turbo': 10,
      'openai/gpt-3.5-turbo': 0.5,
      'anthropic/claude-sonnet-4-20250514': 15,
      'anthropic/claude-3-5-sonnet-20241022': 3,
      'google/gemini-2.5-flash': 0.5,
    };

    return InputValidator.sanitizeNumber(costs[model] || 10, 0, 100);
  }

  private async evaluateQuality(response: string, category: string): Promise<number> {
    // Simplified quality evaluation
    let score = 50; // Base score

    const sanitizedResponse = InputValidator.sanitizeString(response, 10000);

    // Length check
    if (sanitizedResponse.length > 50) score += 10;
    if (sanitizedResponse.length > 200) score += 10;

    // Category-specific checks
    switch (category) {
      case 'coding':
        if (sanitizedResponse.includes('def ') || sanitizedResponse.includes('function')) score += 20;
        if (sanitizedResponse.includes('error') || sanitizedResponse.includes('exception')) score += 10;
        break;

      case 'math':
        if (/\d+/.test(sanitizedResponse)) score += 20;
        if (sanitizedResponse.includes('$') || sanitizedResponse.includes('percent')) score += 10;
        break;

      case 'reasoning':
      case 'analysis':
        if (sanitizedResponse.split('.').length > 3) score += 15;
        if (sanitizedResponse.includes('because') || sanitizedResponse.includes('therefore')) score += 15;
        break;

      case 'writing':
        if (sanitizedResponse.split(' ').length > 100) score += 15;
        if (/[.!?]/.test(sanitizedResponse)) score += 15;
        break;
    }

    return InputValidator.sanitizeNumber(Math.min(100, score), 0, 100);
  }

  private calculateRankings(benchmarks: ModelBenchmark[]): {
    fastest: string;
    highestQuality: string;
    mostCostEffective: string;
  } {
    // Group by model
    const modelStats = new Map<string, {
      totalLatency: number;
      totalQuality: number;
      totalCost: number;
      count: number;
    }>();

    for (const b of benchmarks) {
      if (!modelStats.has(b.model)) {
        modelStats.set(b.model, {
          totalLatency: 0,
          totalQuality: 0,
          totalCost: 0,
          count: 0,
        });
      }

      const stats = modelStats.get(b.model)!;
      stats.totalLatency += b.latency;
      stats.totalQuality += b.quality;
      stats.totalCost += b.cost;
      stats.count++;
    }

    // Calculate averages
    const averages = new Map<string, {
      avgLatency: number;
      avgQuality: number;
      avgCost: number;
    }>();

    for (const [model, stats] of modelStats.entries()) {
      averages.set(model, {
        avgLatency: stats.totalLatency / stats.count,
        avgQuality: stats.totalQuality / stats.count,
        avgCost: stats.totalCost / stats.count,
      });
    }

    // Find rankings
    let fastest = '';
    let highestQuality = '';
    let mostCostEffective = '';

    let minLatency = Infinity;
    let maxQuality = -Infinity;
    let bestQualityPerCost = -Infinity;

    for (const [model, avg] of averages.entries()) {
      if (avg.avgLatency < minLatency) {
        minLatency = avg.avgLatency;
        fastest = model;
      }

      if (avg.avgQuality > maxQuality) {
        maxQuality = avg.avgQuality;
        highestQuality = model;
      }

      const qualityPerCost = avg.avgQuality / avg.avgCost;
      if (qualityPerCost > bestQualityPerCost) {
        bestQualityPerCost = qualityPerCost;
        mostCostEffective = model;
      }
    }

    return {
      fastest,
      highestQuality,
      mostCostEffective,
    };
  }
}

export default ModelPerformanceBenchmark;
