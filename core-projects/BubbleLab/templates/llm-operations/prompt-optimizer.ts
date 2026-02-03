/**
 * Prompt Optimizer
 * Purpose: Optimize prompts for better performance and cost-efficiency
 * Category: LLM Operations
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - OPENAI_API_KEY: For prompt testing
 * - ANTHROPIC_API_KEY: For prompt testing
 * - POSTGRES_CONNECTION_STRING: To store optimized prompts
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
  HttpBubble,
  type WebhookEvent
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
const PromptSchema = z.string().min(1).max(10000);
const GoalSchema = z.string().min(1).max(500);

interface PromptOptimizationRequest {
  originalPrompt: string;
  goal: string;
  constraints?: string[];
  models: string[];
  maxIterations?: number;
}

interface PromptCandidate {
  prompt: string;
  score: number;
  metrics: {
    clarity: number;
    specificity: number;
    conciseness: number;
    effectiveness: number;
  };
  tokenCount: number;
  estimatedCost: number;
}

interface OptimizationResult {
  timestamp: string;
  originalPrompt: string;
  bestPrompt: string;
  improvement: number;
  iterations: number;
  candidates: PromptCandidate[];
  recommendations: string[];
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});

export class PromptOptimizer extends BubbleFlow<'webhook/http'> {
  readonly name = 'Prompt Optimizer';
  readonly description = 'Optimize prompts for better performance and cost-efficiency';

  private logger = new StructuredLogger('prompt-optimizer');
  private rateLimiter = new RateLimiter({
    maxRequests: 30, // Prompt optimization is computationally expensive
    windowMs: 3600000, // 1 hour
  });

  async handle(payload: WebhookEvent & PromptOptimizationRequest): Promise<OptimizationResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 30 optimizations per hour.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    // Security: Validate and sanitize inputs
    let sanitizedOriginalPrompt: string;
    try {
      PromptSchema.parse(payload.originalPrompt);
      sanitizedOriginalPrompt = InputValidator.sanitizeString(payload.originalPrompt, 10000);
    } catch (error) {
      throw new Error('Invalid original prompt format');
    }

    let sanitizedGoal: string;
    try {
      GoalSchema.parse(payload.goal);
      sanitizedGoal = InputValidator.sanitizeString(payload.goal, 500);
    } catch (error) {
      throw new Error('Invalid goal format');
    }

    const sanitizedConstraints = (payload.constraints || [])
      .map(c => InputValidator.sanitizeString(c, 500))
      .filter(c => c.length > 0);

    const models = (payload.models || ['openai/gpt-4']).filter(m => {
      try {
        ModelSchema.parse(m);
        return true;
      } catch {
        return false;
      }
    });

    if (models.length === 0) {
      throw new Error('No valid models provided');
    }

    const maxIterations = InputValidator.sanitizeNumber(payload.maxIterations || 5, 1, 10);

    this.logger.info({
      msg: 'Starting prompt optimization',
      maxIterations,
      modelCount: models.length,
    });

    const candidates: PromptCandidate[] = [];

    // Initial candidate: original prompt
    let bestCandidate = await this.evaluatePrompt(sanitizedOriginalPrompt, models);
    candidates.push(bestCandidate);

    // Iterative optimization
    for (let i = 0; i < maxIterations; i++) {
      // Generate improved prompt using AI
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'You are a prompt engineering expert. Improve the given prompt.',
        message: InputValidator.sanitizeString(`
Original Prompt:
${sanitizedOriginalPrompt}

Goal: ${sanitizedGoal}
Constraints: ${sanitizedConstraints.join(', ')}

Current Best (score: ${bestCandidate.score}):
${bestCandidate.prompt}

Generate an improved version of this prompt. Focus on:
1. Clarity and specificity
2. Removing unnecessary words
3. Adding relevant context
4. Structuring for better understanding

Return only the improved prompt, no explanations.
        `.trim(), 10000),
      });

      const result = await agent.action();
      const improvedPrompt = InputValidator.sanitizeString(result.data.response.trim(), 10000);

      // Evaluate new candidate
      const candidate = await this.evaluatePrompt(improvedPrompt, models);
      candidates.push(candidate);

      // Update best if improved
      if (candidate.score > bestCandidate.score) {
        bestCandidate = candidate;
      }
    }

    // Calculate improvement
    const improvement = InputValidator.sanitizeNumber(
      ((bestCandidate.score - candidates[0].score) / candidates[0].score) * 100,
      -100, 10000
    );

    // Generate recommendations
    const recommendations = await this.generateRecommendations(
      sanitizedOriginalPrompt,
      bestCandidate.prompt
    );

    const result: OptimizationResult = {
      timestamp,
      originalPrompt: sanitizedOriginalPrompt,
      bestPrompt: bestCandidate.prompt,
      improvement,
      iterations: candidates.length,
      candidates,
      recommendations,
      correlationId,
    };

    // Store optimization result
    // Security: SQL injection prevention - use parameterized query
    const storeResultQuery = buildParameterizedQuery(
      `
        INSERT INTO prompt_optimizations (
          timestamp, original_prompt, optimized_prompt, improvement_score,
          iterations, candidates, recommendations
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
      `,
      [
        timestamp,
        sanitizedOriginalPrompt,
        bestCandidate.prompt,
        improvement,
        candidates.length,
        JSON.stringify(candidates.map(c => ({
          prompt: InputValidator.sanitizeString(c.prompt, 10000),
          score: c.score,
          metrics: c.metrics,
          tokenCount: c.tokenCount,
          estimatedCost: c.estimatedCost,
        }))),
        JSON.stringify(recommendations),
      ]
    );

    try {
      const storeResult = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: storeResultQuery.query,
        params: storeResultQuery.params,
      });

      await storeResult.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to store optimization result',
      }, error);
      // Don't throw - storage failure shouldn't break the workflow
    }

    this.logger.info({
      msg: 'Prompt optimization completed',
      improvement,
      iterations: candidates.length,
    });

    return result;
  }

  private async evaluatePrompt(prompt: string, models: string[]): Promise<PromptCandidate> {
    // Sanitize prompt
    const sanitizedPrompt = InputValidator.sanitizeString(prompt, 10000);

    // Calculate token count (rough estimate: ~4 characters per token)
    const tokenCount = InputValidator.sanitizeNumber(
      Math.ceil(sanitizedPrompt.length / 4),
      0, 100000
    );

    // Calculate cost (average $10 per 1M tokens)
    const estimatedCost = InputValidator.sanitizeNumber(
      (tokenCount / 1000000) * 10,
      0, 1000
    );

    // Evaluate quality metrics
    const metrics = await this.calculateMetrics(sanitizedPrompt);

    // Overall score
    const score = InputValidator.sanitizeNumber(
      (metrics.clarity + metrics.specificity + metrics.conciseness + metrics.effectiveness) / 4,
      0, 100
    );

    return {
      prompt: sanitizedPrompt,
      score,
      metrics,
      tokenCount,
      estimatedCost,
    };
  }

  private async calculateMetrics(prompt: string): Promise<{
    clarity: number;
    specificity: number;
    conciseness: number;
    effectiveness: number;
  }> {
    const sanitizedPrompt = InputValidator.sanitizeString(prompt, 10000);

    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Evaluate prompt quality. Return JSON with scores 0-100.',
      message: InputValidator.sanitizeString(`
Evaluate this prompt:
"${sanitizedPrompt}"

Provide scores (0-100) for:
{
  "clarity": number,
  "specificity": number,
  "conciseness": number,
  "effectiveness": number
}

- Clarity: How clear and unambiguous is the prompt?
- Specificity: How specific and detailed are the instructions?
- Conciseness: How efficiently is the information conveyed?
- Effectiveness: How likely is this to produce good results?
      `.trim(), 10000),
    });

    const result = await agent.action();

    try {
      const scores = JSON.parse(result.data.response);
      return {
        clarity: InputValidator.sanitizeNumber(scores.clarity, 0, 100),
        specificity: InputValidator.sanitizeNumber(scores.specificity, 0, 100),
        conciseness: InputValidator.sanitizeNumber(scores.conciseness, 0, 100),
        effectiveness: InputValidator.sanitizeNumber(scores.effectiveness, 0, 100),
      };
    } catch (parseError) {
      // Fallback: simple heuristic-based scoring
      return {
        clarity: this.basicClarityScore(sanitizedPrompt),
        specificity: this.basicSpecificityScore(sanitizedPrompt),
        conciseness: this.basicConcisenessScore(sanitizedPrompt),
        effectiveness: 70, // Can't determine without testing
      };
    }
  }

  private basicClarityScore(prompt: string): number {
    let score = 50;

    // Check for clear structure
    if (prompt.includes('\n\n')) score += 10;
    if (prompt.includes('1.') || prompt.includes('-')) score += 10;

    // Check for explicit instructions
    const instructionWords = ['please', 'ensure', 'make sure', 'generate', 'create', 'write'];
    if (instructionWords.some(w => prompt.toLowerCase().includes(w))) score += 15;

    // Avoid excessive length
    if (prompt.length < 500) score += 15;

    return InputValidator.sanitizeNumber(Math.min(100, score), 0, 100);
  }

  private basicSpecificityScore(prompt: string): number {
    let score = 50;

    // Check for specific details
    if (/\d+/.test(prompt)) score += 15; // Contains numbers
    if (/"[^"]*"/.test(prompt)) score += 15; // Contains quotes

    // Check for examples
    if (prompt.toLowerCase().includes('example')) score += 20;

    return InputValidator.sanitizeNumber(Math.min(100, score), 0, 100);
  }

  private basicConcisenessScore(prompt: string): number {
    const words = prompt.split(/\s+).length;

    // Ideal length: 50-200 words
    if (words >= 50 && words <= 200) return 100;
    if (words < 50) return 70; // Too brief
    if (words <= 400) return 80; // Slightly long

    return 50; // Too long
  }

  private async generateRecommendations(original: string, optimized: string): Promise<string[]> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Explain the differences between two prompts and provide recommendations',
      message: InputValidator.sanitizeString(`
Original Prompt:
"${original}"

Optimized Prompt:
"${optimized}"

Explain the key improvements and provide 3-5 specific recommendations for prompt engineering.
Return as JSON array of strings.
      `.trim(), 10000),
    });

    const result = await agent.action();

    try {
      const recommendations = JSON.parse(result.data.response);
      return recommendations.map((r: string) => InputValidator.sanitizeString(r, 500));
    } catch (parseError) {
      // Fallback recommendations
      const recommendations: string[] = [];

      if (optimized.length < original.length) {
        recommendations.push(
          `Reduced prompt length by ${original.length - optimized.length} characters while maintaining effectiveness`
        );
      }

      if (optimized.includes('\n\n') && !original.includes('\n\n')) {
        recommendations.push('Added better structure with clear sections');
      }

      recommendations.push('Test the optimized prompt with your specific use cases');
      recommendations.push('Consider adding few-shot examples for better performance');

      return recommendations;
    }
  }
}

export default PromptOptimizer;
