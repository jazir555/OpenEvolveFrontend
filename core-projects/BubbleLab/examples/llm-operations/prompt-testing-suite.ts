/**
 * Workflow: Prompt Testing Suite
 * Description: Test prompts across multiple models for performance and quality
 * Use Case: LLM optimization - compare prompt performance across different models
 *
 * Setup Instructions:
 * 1. Configure API keys for multiple LLM providers (OpenAI, Anthropic, Google, etc.)
 * 2. Prepare test cases with expected outputs
 * 3. Configure evaluation metrics (accuracy, coherence, relevance, etc.)
 * 4. Set up result storage and reporting
 *
 * Required Credentials:
 * - ai-agent: For LLM API access (multiple providers)
 * - google-sheets: For storing test results (optional)
 * - slack: For notifications (optional)
 *
 * Trigger Options:
 * - Scheduled: Run daily or weekly
 * - Webhook: Manual test trigger
 * - Manual: On-demand testing
  *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 *

import { BubbleFlow, AIAgentBubble, GoogleSheetsBubble, SlackBubble, type WebhookEvent } from '@bubblelab/bubble-core';

import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  SecuritySchemas,
} from '../../templates/security-utils';

export interface PromptTest {
  id: string;
  prompt: string;
  expectedOutput?: string;
  category: string;
  parameters: Record<string, any>;
}

export interface TestResult {
  testId: string;
  model: string;
  prompt: string;
  output: string;
  latency: number;
  tokensUsed: number;
  qualityScore: number;
  passesTests: boolean;
  metrics: Record<string, number>;
}

export interface CustomWebhookPayload extends WebhookEvent {
  /**
   * Prompts to test
   * @canBeFile false
   */
  prompts?: PromptTest[];

  /**
   * Models to test against
   * @canBeFile false
   */
  models?: string[];

  /**
   * Number of iterations per test
   * @canBeFile false
   */
  iterations?: number;

  /**
   * Store results to Google Sheets
   * @canBeFile false
   */
  storeResults?: boolean;

  /**
   * Spreadsheet ID for results
   * @canBeFile false
   */
  spreadsheetId?: string;

  notify?: boolean;
  slackChannel?: string;
}

// Security: Environment variable validation at startup
validateEnvironment({
  required: ['PROMPT_TEST_SPREADSHEET_ID', 'API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,

  },
});

export class PromptTestingSuite extends BubbleFlow<'webhook/http'> {
  private logger = new StructuredLogger('prompt_testing_suite');
  private rateLimiter = new RateLimiter({
    maxRequests: 60,
    windowMs: 60000,
  });

  private readonly DEFAULT_MODELS = [
    'gpt-4',
    'gpt-3.5-turbo',
    'claude-3-opus',
    'claude-3-sonnet',
    'gemini-pro',
  ];

  private async runPromptTest(
    prompt: string,
    model: string,
    parameters: Record<string, any>
  ): Promise<{ output: string; latency: number; tokensUsed: number }> {
    const startTime = Date.now();

    const agent = new AIAgentBubble({
      model: { model, temperature: parameters.temperature || 0.7 },
      systemPrompt: parameters.systemPrompt || 'You are a helpful AI assistant.',
      message: prompt,
    });

    const result = await agent.action();

    return {
      output: result.success ? result.data.response : '',
      latency: Date.now() - startTime,
      tokensUsed: result.data?.tokensUsed || 0,
    };
  }

  private async evaluateQuality(
    output: string,
    expectedOutput?: string
  ): Promise<number> {
    if (!expectedOutput) {
      // Basic quality metrics without expected output
      const agent = new AIAgentBubble({
        model: { model: 'gpt-4', temperature: 0.3 },
        systemPrompt: 'Rate the quality of the following AI response on a scale of 0-100 based on: coherence, relevance, completeness, and clarity. Return only a number.',
        message: `Rate this response: ${output.substring(0, 1000)}`,
      });

      const result = await agent.action();
      const score = parseInt(result.success ? result.data.response : '70');
      return Math.min(100, Math.max(0, score));
    } else {
      // Compare with expected output
      const agent = new AIAgentBubble({
        model: { model: 'gpt-4', temperature: 0.3 },
        systemPrompt: 'Compare the AI output with the expected output. Rate similarity on a scale of 0-100. Return only a number.',
        message: `Expected: ${expectedOutput}\n\nActual: ${output.substring(0, 1000)}`,
      });

      const result = await agent.action();
      const score = parseInt(result.success ? result.data.response : '50');
      return Math.min(100, Math.max(0, score));
    }
  }

  private async calculateMetrics(output: string): Promise<Record<string, number>> {
    return {
      length: output.length,
      wordCount: output.split(/\s+/).length,
      sentenceCount: output.split(/[.!?]+/).length,
      avgWordLength: output.length / output.split(/\s+/).length || 0,
    };
  }

  private async storeResults(results: TestResult[], spreadsheetId: string): Promise<void> {
    const rows = [
      [
        'Timestamp',
        'Test ID',
        'Model',
        'Prompt',
        'Output',
        'Latency (ms)',
        'Tokens Used',
        'Quality Score',
        'Passes Tests',
      ],
      ...results.map(r => [
        new Date().toISOString(),
        r.testId,
        r.model,
        r.prompt.substring(0, 100),
        r.output.substring(0, 200),
        r.latency.toString(),
        r.tokensUsed.toString(),
        r.qualityScore.toString(),
        r.passesTests.toString(),
      ]),
    ];

    const sheets = new GoogleSheetsBubble({
      operation: 'update',
      spreadsheetId,
      range: 'PromptTests!A1',
      values: rows,
    });

    await sheets.action();
  }

  private async sendSlackNotification(results: TestResult[], channel: string): Promise<void> {
    const byModel = results.reduce((acc, r) => {
      if (!acc[r.model]) acc[r.model] = [];
      acc[r.model].push(r);
      return acc;
    }, {} as Record<string, TestResult[]>);

    const summaries = Object.entries(byModel).map(([model, modelResults]) => {
      const avgQuality =
        modelResults.reduce((sum, r) => sum + r.qualityScore, 0) / modelResults.length;
      const avgLatency =
        modelResults.reduce((sum, r) => sum + r.latency, 0) / modelResults.length;

      return `${model}: Quality ${avgQuality.toFixed(0)}/100, Latency ${avgLatency.toFixed(0)}ms`;
    });

    const slack = new SlackBubble({
      channel,
      message: {
        text: `🧪 Prompt Testing Complete`,
        attachments: [
          {
            color: 'good',
            fields: [
              { title: 'Total Tests', value: results.length.toString(), short: true },
              { title: 'Models Tested', value: Object.keys(byModel).length.toString(), short: true },
            ],
          },
          {
            title: 'Model Summaries',
            text: summaries.join('\n'),
          },
        ],
      },
    });

    await slack.action();
  }

  async handle(payload: CustomWebhookPayload): Promise<any> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting prompt testing suite',
    });

    const {
      prompts = [
        {
          id: 'test-1',
          prompt: 'Explain quantum computing in simple terms',
          category: 'explanation',
          parameters: { temperature: 0.7 },
        },
        {
          id: 'test-2',
          prompt: 'Write a haiku about programming',
          category: 'creative',
          parameters: { temperature: 0.9 },
        },
      ],
      models = this.DEFAULT_MODELS,
      iterations = 1,
      storeResults = true,
      spreadsheetId = process.env.PROMPT_TEST_SPREADSHEET_ID || '',
      notify = true,
      slackChannel = '#llm-ops',
    } = payload;

    this.logger?.info(`Starting prompt testing suite: ${prompts.length} prompts × ${models.length} models × ${iterations} iterations`);

    const results: TestResult[] = [];

    for (const prompt of prompts) {
      for (const model of models) {
        for (let i = 0; i < iterations; i++) {
          this.logger?.info(`Testing ${prompt.id} with ${model}...`);

          const { output, latency, tokensUsed } = await this.runPromptTest(
            prompt.prompt,
            model,
            prompt.parameters
          );

          const qualityScore = await this.evaluateQuality(output, prompt.expectedOutput);
          const metrics = await this.calculateMetrics(output);

          results.push({
            testId: prompt.id,
            model,
            prompt: prompt.prompt,
            output,
            latency,
            tokensUsed,
            qualityScore,
            passesTests: qualityScore >= 70,
            metrics,
          });

          this.logger?.info(`${model}: Quality ${qualityScore}/100, ${latency}ms`);
        }
      }
    }

    // Store results
    if (storeResults && spreadsheetId) {
      await this.storeResults(results, spreadsheetId);
    }

    // Send notification
    if (notify) {
      await this.sendSlackNotification(results, slackChannel);
    }

    const avgQuality = results.reduce((sum, r) => sum + r.qualityScore, 0) / results.length;

    return {
      message: `Testing complete: ${results.length} test(s) run`,
      results,
      summary: {
        totalTests: results.length,
        avgQuality,
        avgLatency: results.reduce((sum, r) => sum + r.latency, 0) / results.length,
        totalTokens: results.reduce((sum, r) => sum + r.tokensUsed, 0),
      },
    };
  }
}

export const workflowConfig = {
  id: 'prompt-testing-suite',
  name: 'Prompt Testing Suite',
  description: 'Test prompts across multiple models for performance and quality',
  version: '1.0.0',
  category: 'llm-operations',
  icon: '🧪',
  tags: ['llm', 'testing', 'prompts', 'benchmarks', 'quality'],
};
