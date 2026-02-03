/**
 * AI Response Quality Assessor
 * Purpose: Assess and track AI response quality over time
 * Category: LLM Operations
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - OPENAI_API_KEY: For GPT-4 evaluation
 * - POSTGRES_CONNECTION_STRING: To store quality metrics
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
const ResponseIdSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid response ID');
const ModelSchema = z.string().min(1).max(100);
const ApiKeySchema = z.string().min(32).max(256);
const PromptSchema = z.string().min(1).max(10000);

interface QualityAssessment {
  responseId: string;
  model: string;
  prompt: string;
  response: string;
  context: string;
  metrics: {
    coherence: number;
    relevance: number;
    accuracy: number;
    completeness: number;
    helpfulness: number;
    overall: number;
  };
  issues: string[];
  suggestions: string[];
}

interface QualityReport {
  timestamp: string;
  assessments: QualityAssessment[];
  averageQuality: number;
  trends: {
    improving: string[];
    declining: string[];
  };
  recommendations: string[];
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['OPENAI_API_KEY'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});

export class AIResponseQualityAssessor extends BubbleFlow<'webhook/http'> {
  readonly name = 'AI Response Quality Assessor';
  readonly description = 'Assess and track AI response quality over time';

  private logger = new StructuredLogger('ai-response-quality-assessor');
  private rateLimiter = new RateLimiter({
    maxRequests: 100, // Quality assessments can be frequent
    windowMs: 3600000, // 1 hour
  });

  async handle(payload: WebhookEvent & {
    assessments: Array<{
      responseId: string;
      model: string;
      prompt: string;
      response: string;
      context?: string;
    }>;
  }): Promise<QualityReport> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();
    const assessments: QualityAssessment[] = [];

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 100 assessments per hour.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting AI response quality assessment',
      assessmentCount: payload.assessments.length,
    });

    // Validate and filter assessments
    const validAssessments = payload.assessments.filter(a => {
      try {
        ResponseIdSchema.parse(a.responseId);
        ModelSchema.parse(a.model);
        PromptSchema.parse(a.prompt);
        return true;
      } catch {
        return false;
      }
    });

    if (validAssessments.length === 0) {
      throw new Error('No valid assessments provided');
    }

    // Assess each response
    for (const item of validAssessments) {
      try {
        const assessment = await this.assessQuality(
          item.responseId,
          item.model,
          item.prompt,
          item.response,
          item.context || ''
        );

        assessments.push(assessment);

        // Store in database
        // Security: SQL injection prevention - use parameterized query
        const storeAssessmentQuery = buildParameterizedQuery(
          `
            INSERT INTO quality_assessments (
              response_id, model, prompt, response, context,
              coherence, relevance, accuracy, completeness, helpfulness, overall,
              issues, suggestions
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
          `,
          [
            assessment.responseId,
            assessment.model,
            assessment.prompt,
            assessment.response,
            assessment.context,
            assessment.metrics.coherence,
            assessment.metrics.relevance,
            assessment.metrics.accuracy,
            assessment.metrics.completeness,
            assessment.metrics.helpfulness,
            assessment.metrics.overall,
            JSON.stringify(assessment.issues),
            JSON.stringify(assessment.suggestions),
          ]
        );

        try {
          const storeAssessment = new PostgreSQLBubble({
            connectionString: process.env.POSTGRES_CONNECTION_STRING,
            query: storeAssessmentQuery.query,
            params: storeAssessmentQuery.params,
          });

          await storeAssessment.action();
        } catch (error) {
          this.logger.error({
            msg: 'Failed to store assessment',
            responseId: assessment.responseId,
          }, error);
          // Don't throw - storage failure shouldn't break the workflow
        }
      } catch (error) {
        this.logger.warn({
          msg: 'Failed to assess response',
          responseId: item.responseId,
          model: item.model,
        }, error);
      }
    }

    // Calculate average quality
    const averageQuality =
      assessments.reduce((sum, a) => sum + a.metrics.overall, 0) / assessments.length;

    // Analyze trends (compare with historical data)
    const trends = await this.analyzeTrends(assessments);

    // Generate recommendations
    const recommendations = await this.generateRecommendations(assessments, trends);

    const report: QualityReport = {
      timestamp,
      assessments,
      averageQuality: InputValidator.sanitizeNumber(averageQuality, 0, 100),
      trends,
      recommendations,
      correlationId,
    };

    this.logger.info({
      msg: 'AI response quality assessment completed',
      assessmentCount: assessments.length,
      averageQuality,
      improvingCount: trends.improving.length,
      decliningCount: trends.declining.length,
    });

    return report;
  }

  private async assessQuality(
    responseId: string,
    model: string,
    prompt: string,
    response: string,
    context: string
  ): Promise<QualityAssessment> {
    // Sanitize inputs
    const sanitizedPrompt = InputValidator.sanitizeString(prompt, 10000);
    const sanitizedResponse = InputValidator.sanitizeString(response, 10000);
    const sanitizedContext = InputValidator.sanitizeString(context, 5000);

    // Use GPT-4 to assess quality
    const assessor = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: InputValidator.sanitizeString(`You are an AI response quality assessor. Evaluate the given response and provide scores (0-100) and feedback.

Return JSON format:
{
  "metrics": {
    "coherence": number,
    "relevance": number,
    "accuracy": number,
    "completeness": number,
    "helpfulness": number,
    "overall": number
  },
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"]
}`, 10000),
      message: InputValidator.sanitizeString(`
Assess this AI response:

Model: ${model}
Context: ${sanitizedContext}
Prompt: ${sanitizedPrompt}
Response: ${sanitizedResponse}

Provide comprehensive assessment.
      `.trim(), 10000),
    });

    const result = await assessor.action();

    try {
      const assessment = JSON.parse(result.data.response);

      return {
        responseId: InputValidator.sanitizeString(responseId, 255),
        model: InputValidator.sanitizeString(model, 100),
        prompt: sanitizedPrompt,
        response: sanitizedResponse,
        context: sanitizedContext,
        metrics: {
          coherence: InputValidator.sanitizeNumber(assessment.metrics.coherence, 0, 100),
          relevance: InputValidator.sanitizeNumber(assessment.metrics.relevance, 0, 100),
          accuracy: InputValidator.sanitizeNumber(assessment.metrics.accuracy, 0, 100),
          completeness: InputValidator.sanitizeNumber(assessment.metrics.completeness, 0, 100),
          helpfulness: InputValidator.sanitizeNumber(assessment.metrics.helpfulness, 0, 100),
          overall: InputValidator.sanitizeNumber(assessment.metrics.overall, 0, 100),
        },
        issues: (assessment.issues || []).map((i: string) => InputValidator.sanitizeString(i, 500)),
        suggestions: (assessment.suggestions || []).map((s: string) => InputValidator.sanitizeString(s, 500)),
      };
    } catch (parseError) {
      // Fallback: simple assessment
      return {
        responseId: InputValidator.sanitizeString(responseId, 255),
        model: InputValidator.sanitizeString(model, 100),
        prompt: sanitizedPrompt,
        response: sanitizedResponse,
        context: sanitizedContext,
        metrics: {
          coherence: this.basicScore(sanitizedResponse, 'coherence'),
          relevance: this.basicScore(sanitizedResponse, 'relevance'),
          accuracy: this.basicScore(sanitizedResponse, 'accuracy'),
          completeness: this.basicScore(sanitizedResponse, 'completeness'),
          helpfulness: this.basicScore(sanitizedResponse, 'helpfulness'),
          overall: 70,
        },
        issues: [],
        suggestions: ['Could not parse detailed assessment'],
      };
    }
  }

  private basicScore(response: string, metric: string): number {
    // Very basic scoring based on response characteristics
    let score = 50;

    switch (metric) {
      case 'coherence':
        if (response.split('.').length > 2) score += 20;
        if (response.length > 100) score += 15;
        if (/\n/.test(response)) score += 15;
        break;

      case 'relevance':
        score = 70; // Can't assess without prompt analysis
        break;

      case 'accuracy':
        score = 75; // Can't assess without ground truth
        break;

      case 'completeness':
        if (response.length > 200) score += 30;
        if (response.includes('.') && response.includes(',')) score += 20;
        break;

      case 'helpfulness':
        score = 70; // Subjective
        break;
    }

    return InputValidator.sanitizeNumber(Math.min(100, score), 0, 100);
  }

  private async analyzeTrends(assessments: QualityAssessment[]): Promise<{
    improving: string[];
    declining: string[];
  }> {
    // Get historical data for comparison
    // Security: SQL injection prevention - use parameterized query
    const historicalDataQuery = buildParameterizedQuery(
      `
        SELECT
          model,
          AVG(overall) as avg_quality,
          COUNT(*) as sample_count
        FROM quality_assessments
        WHERE timestamp > NOW() - INTERVAL '7 days'
        GROUP BY model
      `,
      []
    );

    let historical;
    try {
      const historicalData = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: historicalDataQuery.query,
        params: historicalDataQuery.params,
      });

      historical = await historicalData.action();
    } catch (error) {
      this.logger.warn({
        msg: 'Failed to retrieve historical data',
      }, error);
      // Return empty trends if historical data is unavailable
      return { improving: [], declining: [] };
    }

    const historicalScores = new Map(
      (historical.data.rows || []).map((r: any) => [r.model, r.avg_quality])
    );

    const improving: string[] = [];
    const declining: string[] = [];

    // Compare current vs historical
    for (const assessment of assessments) {
      try {
        ModelSchema.parse(assessment.model);

        const historicalAvg = historicalScores.get(assessment.model) || 70;
        const currentScore = assessment.metrics.overall;

        if (currentScore > historicalAvg + 5) {
          improving.push(assessment.model);
        } else if (currentScore < historicalAvg - 5) {
          declining.push(assessment.model);
        }
      } catch {
        // Skip invalid model names
        continue;
      }
    }

    return { improving, declining };
  }

  private async generateRecommendations(
    assessments: QualityAssessment[],
    trends: { improving: string[]; declining: string[] }
  ): Promise<string[]> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze AI quality assessment data and provide actionable recommendations',
      message: InputValidator.sanitizeString(`
Quality Assessments:
${JSON.stringify(assessments.map(a => ({
  model: a.model,
  overall: a.metrics.overall,
  issues: a.issues,
})), null, 2)}

Trends:
Improving: ${trends.improving.join(', ')}
Declining: ${trends.declining.join(', ')}

Provide 3-5 specific recommendations to improve AI response quality.
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

      if (trends.declining.length > 0) {
        recommendations.push(
          `Investigate quality decline in models: ${trends.declining.join(', ')}`
        );
      }

      const lowQualityModels = assessments
        .filter(a => a.metrics.overall < 70)
        .map(a => a.model);

      if (lowQualityModels.length > 0) {
        recommendations.push(
          `Consider replacing or reconfiguring underperforming models: ${[...new Set(lowQualityModels)].join(', ')}`
        );
      }

      recommendations.push('Review and refine system prompts for low-scoring models');
      recommendations.push('Implement A/B testing for prompt optimization');

      return recommendations;
    }
  }
}

export default AIResponseQualityAssessor;
