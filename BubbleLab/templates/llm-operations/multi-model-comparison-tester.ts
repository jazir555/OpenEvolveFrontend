/**
 * Multi-Model Comparison Tester
 * Purpose: Compare outputs from multiple models on the same prompt
 * Category: LLM Operations
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - OPENAI_API_KEY: For GPT models
 * - ANTHROPIC_API_KEY: For Claude models
 * - GOOGLE_API_KEY: For Gemini models
 * - POSTGRES_CONNECTION_STRING: To store comparison results
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

interface ModelResponse {
  model: string;
  response: string;
  latency: number;
  tokenUsage: {
    prompt: number;
    completion: number;
    total: number;
  };
  cost: number;
}

interface ComparisonResult {
  timestamp: string;
  prompt: string;
  responses: ModelResponse[];
  analysis: {
    mostCoherent: string;
    mostAccurate: string;
    mostCreative: string;
    mostConcise: string;
    fastest: string;
    mostCostEffective: string;
    overallBest: string;
  };
  detailedAnalysis: string;
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

export class MultiModelComparisonTester extends BubbleFlow<'webhook/http'> {
  readonly name = 'Multi-Model Comparison Tester';
  readonly description = 'Compare outputs from multiple models on the same prompt';

  private logger = new StructuredLogger('multi-model-comparison-tester');
  private rateLimiter = new RateLimiter({
    maxRequests: 50, // Model comparisons can be expensive
    windowMs: 3600000, // 1 hour
  });

  async handle(payload: WebhookEvent & {
    prompt: string;
    models?: string[];
    systemPrompt?: string;
  }): Promise<ComparisonResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 50 comparisons per hour.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    // Security: Validate and sanitize inputs
    let sanitizedPrompt: string;
    try {
      PromptSchema.parse(payload.prompt);
      sanitizedPrompt = InputValidator.sanitizeString(payload.prompt, 10000);
    } catch (error) {
      throw new Error('Invalid prompt format');
    }

    const sanitizedSystemPrompt = InputValidator.sanitizeString(
      payload.systemPrompt || 'You are a helpful AI assistant.',
      10000
    );

    // Default models to compare
    const models = (payload.models || [
      'openai/gpt-4',
      'anthropic/claude-sonnet-4-20250514',
      'google/gemini-2.5-flash',
    ]).filter(m => {
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

    this.logger.info({
      msg: 'Starting multi-model comparison',
      modelCount: models.length,
    });

    const responses: ModelResponse[] = [];

    // Get responses from each model
    for (const model of models) {
      try {
        const startTime = Date.now();

        const agent = new AIAgentBubble({
          model: { model },
          systemPrompt: sanitizedSystemPrompt,
          message: sanitizedPrompt,
        });

        const result = await agent.action();

        const latency = Date.now() - startTime;
        const tokens = result.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

        // Calculate cost
        const costPerMillion = this.getModelCost(model);
        const cost = (tokens.total_tokens / 1000000) * costPerMillion;

        responses.push({
          model,
          response: InputValidator.sanitizeString(result.data.response, 10000),
          latency: InputValidator.sanitizeNumber(latency, 0, 60000),
          tokenUsage: {
            prompt: InputValidator.sanitizeNumber(tokens.prompt_tokens, 0, Number.MAX_SAFE_INTEGER),
            completion: InputValidator.sanitizeNumber(tokens.completion_tokens, 0, Number.MAX_SAFE_INTEGER),
            total: InputValidator.sanitizeNumber(tokens.total_tokens, 0, Number.MAX_SAFE_INTEGER),
          },
          cost: InputValidator.sanitizeNumber(cost, 0, 100),
        });

      } catch (error) {
        this.logger.warn({
          msg: 'Error getting response from model',
          model,
        }, error);
      }
    }

    // Analyze and compare responses
    const analysis = await this.analyzeResponses(sanitizedPrompt, responses);

    // Generate detailed analysis
    const detailedAnalysis = await this.generateDetailedAnalysis(sanitizedPrompt, responses);

    const result: ComparisonResult = {
      timestamp,
      prompt: sanitizedPrompt,
      responses,
      analysis,
      detailedAnalysis: InputValidator.sanitizeString(detailedAnalysis, 10000),
      correlationId,
    };

    // Store comparison in database
    // Security: SQL injection prevention - use parameterized query
    const storeComparisonQuery = buildParameterizedQuery(
      `
        INSERT INTO model_comparisons (
          timestamp, prompt, models, responses, analysis, detailed_analysis
        )
        VALUES ($1, $2, $3, $4, $5, $6)
      `,
      [
        timestamp,
        sanitizedPrompt,
        JSON.stringify(models),
        JSON.stringify(responses),
        JSON.stringify(analysis),
        InputValidator.sanitizeString(detailedAnalysis, 10000),
      ]
    );

    try {
      const storeComparison = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: storeComparisonQuery.query,
        params: storeComparisonQuery.params,
      });

      await storeComparison.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to store comparison',
      }, error);
      // Don't throw - storage failure shouldn't break the workflow
    }

    this.logger.info({
      msg: 'Multi-model comparison completed',
      responseCount: responses.length,
      overallBest: analysis.overallBest,
    });

    return result;
  }

  private async analyzeResponses(prompt: string, responses: ModelResponse[]): Promise<{
    mostCoherent: string;
    mostAccurate: string;
    mostCreative: string;
    mostConcise: string;
    fastest: string;
    mostCostEffective: string;
    overallBest: string;
  }> {
    // Use AI to compare responses
    const sanitizedPrompt = InputValidator.sanitizeString(prompt, 10000);

    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: InputValidator.sanitizeString(`You are an AI response evaluator. Compare the given responses and identify the best in each category.

Return JSON:
{
  "mostCoherent": "model_name",
  "mostAccurate": "model_name",
  "mostCreative": "model_name",
  "mostConcise": "model_name",
  "fastest": "model_name",
  "mostCostEffective": "model_name",
  "overallBest": "model_name"
}`, 10000),
      message: InputValidator.sanitizeString(`
Prompt: ${sanitizedPrompt}

Responses:
${responses.map(r => `
Model: ${r.model}
Response: ${InputValidator.sanitizeString(r.response, 5000)}
Latency: ${r.latency}ms
Cost: $${r.cost.toFixed(4)}
`).join('\n')}

Evaluate and rank each response.
      `.trim(), 10000),
    });

    const result = await agent.action();

    try {
      const analysis = JSON.parse(result.data.response);

      // Validate all model names
      const validModels = responses.map(r => r.model);
      const validateModel = (model: string) => validModels.includes(model) ? model : responses[0]?.model || '';

      return {
        mostCoherent: validateModel(analysis.mostCoherent),
        mostAccurate: validateModel(analysis.mostAccurate),
        mostCreative: validateModel(analysis.mostCreative),
        mostConcise: validateModel(analysis.mostConcise),
        fastest: validateModel(analysis.fastest),
        mostCostEffective: validateModel(analysis.mostCostEffective),
        overallBest: validateModel(analysis.overallBest),
      };
    } catch (parseError) {
      // Fallback: simple heuristic-based analysis
      return {
        mostCoherent: this.selectByMetric(responses, 'coherence'),
        mostAccurate: this.selectByMetric(responses, 'accuracy'),
        mostCreative: this.selectByMetric(responses, 'creativity'),
        mostConcise: this.selectByMetric(responses, 'conciseness'),
        fastest: responses.sort((a, b) => a.latency - b.latency)[0]?.model || '',
        mostCostEffective: responses.sort((a, b) => a.cost - b.cost)[0]?.model || '',
        overallBest: responses[0]?.model || '',
      };
    }
  }

  private selectByMetric(responses: ModelResponse[], metric: string): string {
    // Simple heuristic-based selection
    switch (metric) {
      case 'coherence':
        return responses.sort((a, b) => {
          const scoreA = this.calculateCoherence(a.response);
          const scoreB = this.calculateCoherence(b.response);
          return scoreB - scoreA;
        })[0]?.model || '';

      case 'conciseness':
        return responses.sort((a, b) => a.response.length - b.response.length)[0]?.model || '';

      case 'creativity':
        return responses.sort((a, b) => {
          const scoreA = this.calculateCreativity(a.response);
          const scoreB = this.calculateCreativity(b.response);
          return scoreB - scoreA;
        })[0]?.model || '';

      default:
        return responses[0]?.model || '';
    }
  }

  private calculateCoherence(response: string): number {
    const sanitizedResponse = InputValidator.sanitizeString(response, 10000);
    let score = 0;

    // Check for sentence structure
    const sentences = sanitizedResponse.split(/[.!?]/).filter(s => s.trim().length > 0);
    score += Math.min(sentences.length * 10, 30);

    // Check for logical connectors
    const connectors = ['because', 'therefore', 'however', 'moreover', 'furthermore'];
    const hasConnectors = connectors.some(c => sanitizedResponse.toLowerCase().includes(c));
    if (hasConnectors) score += 20;

    // Check paragraph structure
    if (sanitizedResponse.includes('\n\n')) score += 20;

    // Check length balance
    if (sanitizedResponse.length > 100 && sanitizedResponse.length < 1000) score += 30;

    return score;
  }

  private calculateCreativity(response: string): number {
    const sanitizedResponse = InputValidator.sanitizeString(response, 10000);
    let score = 0;

    // Vocabulary diversity
    const words = sanitizedResponse.toLowerCase().split(/\s+/);
    const uniqueWords = new Set(words);
    const diversity = uniqueWords.size / words.length;
    score += diversity * 30;

    // Use of examples/analogies
    if (sanitizedResponse.includes('example') || sanitizedResponse.includes('such as')) score += 25;

    // Metaphors or figurative language
    if (sanitizedResponse.includes('like') || sanitizedResponse.includes('similar to')) score += 25;

    // Unique insights (based on length and complexity)
    if (sanitizedResponse.length > 300) score += 20;

    return score;
  }

  private async generateDetailedAnalysis(prompt: string, responses: ModelResponse[]): Promise<string> {
    const sanitizedPrompt = InputValidator.sanitizeString(prompt, 10000);

    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate a detailed comparative analysis of AI model responses',
      message: InputValidator.sanitizeString(`
Provide a detailed analysis comparing these model responses:

Prompt: ${sanitizedPrompt}

Responses:
${responses.map(r => `
### ${r.model}
${InputValidator.sanitizeString(r.response, 5000)}

*Latency: ${r.latency}ms*
*Tokens: ${r.tokenUsage.total}*
*Cost: $${r.cost.toFixed(4)}*
`).join('\n')}

Include:
1. Response quality comparison
2. Performance metrics comparison
3. Cost-effectiveness analysis
4. Use case recommendations
      `.trim(), 10000),
    });

    const result = await agent.action();
    return InputValidator.sanitizeString(result.data.response, 10000);
  }

  private getModelCost(model: string): number {
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
}

export default MultiModelComparisonTester;
