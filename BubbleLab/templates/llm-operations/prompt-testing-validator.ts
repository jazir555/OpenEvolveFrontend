/**
 * Prompt Testing Validator
 * Purpose: Test and validate prompts across multiple models
 * Category: LLM Operations
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - OPENAI_API_KEY: For GPT models
 * - ANTHROPIC_API_KEY: For Claude models
 * - GOOGLE_API_KEY: For Gemini models
 * - POSTGRES_CONNECTION_STRING: To store test results
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
import crypto from 'crypto';
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
const TestSuiteSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid test suite name');
const ModelSchema = z.string().min(1).max(100);
const ApiKeySchema = z.string().min(32).max(256);
const PromptSchema = z.string().min(1).max(10000);

interface PromptTestCase {
  name: string;
  prompt: string;
  expectedOutput: string;
  evaluationCriteria: string[];
}

interface PromptTestResult {
  testCase: string;
  model: string;
  response: string;
  evaluation: {
    relevance: number;
    accuracy: number;
    completeness: number;
    overallScore: number;
  };
  tokenUsage: {
    prompt: number;
    completion: number;
    total: number;
  };
  latency: number;
}

interface PromptValidationResult {
  timestamp: string;
  testSuite: string;
  totalTests: number;
  results: PromptTestResult[];
  bestModel: string;
  averageScore: number;
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

export class PromptTestingValidator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Prompt Testing Validator';
  readonly description = 'Test and validate prompts across multiple models';

  private logger = new StructuredLogger('prompt-testing-validator');
  private rateLimiter = new RateLimiter({
    maxRequests: 50, // Prompt testing can be expensive
    windowMs: 3600000, // 1 hour
  });

  async handle(payload: WebhookEvent & {
    testSuite: string;
    testCases: PromptTestCase[];
    models: string[];
  }): Promise<PromptValidationResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();

    // Security: Validate and sanitize inputs
    let testSuite: string;
    try {
      testSuite = TestSuiteSchema.parse(payload.testSuite);
    } catch (error) {
      throw new Error('Invalid test suite name format');
    }

    // Validate models
    const models = payload.models.filter(m => {
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

    // Validate test cases
    const testCases = payload.testCases.filter(tc => {
      try {
        PromptSchema.parse(tc.prompt);
        return tc.name.length > 0 && tc.name.length <= 255;
      } catch {
        return false;
      }
    });

    if (testCases.length === 0) {
      throw new Error('No valid test cases provided');
    }

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 50 tests per hour.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting prompt testing validation',
      testSuite,
      testCount: testCases.length,
      modelCount: models.length,
    });

    const allResults: PromptTestResult[] = [];

    // Test each prompt against each model
    for (const testCase of testCases) {
      for (const model of models) {
        const startTime = Date.now();

        try {
          // Sanitize prompt
          const sanitizedPrompt = InputValidator.sanitizeString(testCase.prompt, 10000);

          // Execute prompt
          const agent = new AIAgentBubble({
            model: { model },
            systemPrompt: 'You are a helpful AI assistant.',
            message: sanitizedPrompt,
          });

          const response = await agent.action();

          // Evaluate response
          const evaluation = await this.evaluateResponse(
            response.data.response,
            testCase.expectedOutput,
            testCase.evaluationCriteria
          );

          const result: PromptTestResult = {
            testCase: InputValidator.sanitizeString(testCase.name, 255),
            model,
            response: InputValidator.sanitizeString(response.data.response, 10000),
            evaluation,
            tokenUsage: {
              prompt: response.usage?.prompt_tokens || 0,
              completion: response.usage?.completion_tokens || 0,
              total: response.usage?.total_tokens || 0,
            },
            latency: Date.now() - startTime,
          };

          allResults.push(result);

        } catch (error) {
          this.logger.warn({
            msg: 'Error testing prompt',
            testCase: testCase.name,
            model,
          }, error);
        }
      }
    }

    // Calculate statistics
    const modelScores = new Map<string, number[]>();

    for (const result of allResults) {
      if (!modelScores.has(result.model)) {
        modelScores.set(result.model, []);
      }
      modelScores.get(result.model)!.push(result.evaluation.overallScore);
    }

    const averageScores = new Map<string, number>();
    for (const [model, scores] of modelScores.entries()) {
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      averageScores.set(model, avg);
    }

    const bestModel = [...averageScores.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || '';
    const averageScore = [...averageScores.values()].reduce((a, b) => a + b, 0) / averageScores.size;

    const result: PromptValidationResult = {
      timestamp,
      testSuite,
      totalTests: allResults.length,
      results: allResults,
      bestModel,
      averageScore,
      correlationId,
    };

    // Store results in database
    // Security: SQL injection prevention - use parameterized queries
    for (const testResult of allResults) {
      const storeResultQuery = buildParameterizedQuery(
        `
          INSERT INTO prompt_tests (
            timestamp, test_suite, test_case, model, response,
            relevance, accuracy, completeness, overall_score,
            prompt_tokens, completion_tokens, total_tokens, latency
          )
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        `,
        [
          timestamp,
          testSuite,
          testResult.testCase,
          testResult.model,
          testResult.response,
          testResult.evaluation.relevance,
          testResult.evaluation.accuracy,
          testResult.evaluation.completeness,
          testResult.evaluation.overallScore,
          testResult.tokenUsage.prompt,
          testResult.tokenUsage.completion,
          testResult.tokenUsage.total,
          testResult.latency,
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
          msg: 'Failed to store test result',
          testCase: testResult.testCase,
          model: testResult.model,
        }, error);
        // Don't throw - storage failure shouldn't break the workflow
      }
    }

    this.logger.info({
      msg: 'Prompt testing validation completed',
      totalTests: result.totalTests,
      bestModel,
      averageScore,
    });

    return result;
  }

  private async evaluateResponse(
    response: string,
    expected: string,
    criteria: string[]
  ): Promise<{
    relevance: number;
    accuracy: number;
    completeness: number;
    overallScore: number;
  }> {
    // Sanitize inputs
    const sanitizedResponse = InputValidator.sanitizeString(response, 10000);
    const sanitizedExpected = InputValidator.sanitizeString(expected, 10000);

    // Use AI to evaluate response quality
    const evaluator = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Evaluate AI response quality. Return JSON with scores 0-100.',
      message: InputValidator.sanitizeString(`
Evaluate this response:

Response: ${sanitizedResponse}

Expected: ${sanitizedExpected}

Criteria: ${criteria.join(', ')}

Provide JSON scores (0-100):
{
  "relevance": number,
  "accuracy": number,
  "completeness": number
}
      `.trim(), 10000),
    });

    const evaluation = await evaluator.action();

    try {
      const scores = JSON.parse(evaluation.data.response);
      const overallScore = (scores.relevance + scores.accuracy + scores.completeness) / 3;

      return {
        relevance: InputValidator.sanitizeNumber(scores.relevance, 0, 100),
        accuracy: InputValidator.sanitizeNumber(scores.accuracy, 0, 100),
        completeness: InputValidator.sanitizeNumber(scores.completeness, 0, 100),
        overallScore: InputValidator.sanitizeNumber(overallScore, 0, 100),
      };
    } catch (parseError) {
      // Fallback: simple similarity check
      const relevance = this.calculateSimilarity(sanitizedResponse, sanitizedExpected);
      return {
        relevance,
        accuracy: relevance,
        completeness: Math.min(100, (response.length / expected.length) * 100),
        overallScore: relevance,
      };
    }
  }

  private calculateSimilarity(str1: string, str2: string): number {
    const words1 = new Set(str1.toLowerCase().split(/\s+/));
    const words2 = new Set(str2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    return (intersection.size / union.size) * 100;
  }
}

export default PromptTestingValidator;
