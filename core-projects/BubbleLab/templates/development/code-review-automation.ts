/**
 * Code Review Automation
 * Purpose: Automated code review with AI analysis and PR feedback
 * Category: Development Workflow
 * Event Type: webhook/http
 *
 * Required Credentials:
 * - GITHUB_PAT: GitHub personal access token
 * - OPENAI_API_KEY: For AI code analysis
 * - API_KEY: API key for authentication (required)
 * - SLACK_WEBHOOK_URL: Slack webhook for notifications (optional)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - API key authentication
 * - Input validation for PR data (XSS prevention)
 * - Rate limiting
 * - Error message sanitization
 * - Structured logging with correlation IDs
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
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
  SecuritySchemas,
} from '../security-utils';

// Input validation schemas
const RepositorySchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid repository format');
const BranchNameSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_\-\/]+$/, 'Invalid branch name');

interface PullRequestData {
  number: number;
  title: string;
  body: string;
  author: string;
  baseBranch: string;
  headBranch: string;
  repository: string;
  changedFiles: number;
  additions: number;
  deletions: number;
}

interface CodeReviewResult {
  prNumber: number;
  repository: string;
  timestamp: string;
  overallScore: number;
  issues: {
    critical: string[];
    warning: string[];
    suggestion: string[];
  };
  approved: boolean;
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['GITHUB_PAT', 'OPENAI_API_KEY', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    GITHUB_PAT: SecuritySchemas.token,
  },
});

export class CodeReviewAutomation extends BubbleFlow<'webhook/http'> {
  readonly name = 'Code Review Automation';
  readonly description = 'Automated code review with AI analysis and PR feedback';

  // Security: Rate limiter
  private rateLimiter = new RateLimiter({ maxRequests: 50, windowMs: 60000 });
  private logger = new StructuredLogger('code-review-automation');

  async handle(payload: WebhookEvent & PullRequestData): Promise<CodeReviewResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({ msg: 'Rate limit exceeded', correlationId });
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({ msg: 'Starting code review', correlationId });

    // Security: Validate input
    let sanitizedRepository: string;
    try {
      sanitizedRepository = RepositorySchema.parse(payload.repository);
    } catch (error) {
      this.logger.error({ msg: 'Invalid repository format', correlationId }, error);
      throw new Error('Invalid repository format');
    }

    const { number, title, author, changedFiles } = payload;

    // Step 1: Get PR diff
    const getDiff = new HttpBubble({
      url: `https://api.github.com/repos/${sanitizedRepository}/pulls/${number}/files`,
      method: 'GET',
      headers: {
        'Authorization': `token ${process.env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      timeout: 30000,
    });

    let diffResult;
    try {
      diffResult = await getDiff.action();
    } catch (error) {
      this.logger.error({ msg: 'Failed to fetch PR diff', correlationId, repository: sanitizedRepository }, error);
      throw new Error('Failed to retrieve PR diff');
    }
    const files = diffResult.data;

    // Step 2: Analyze code with AI
    const codeAnalysisPrompt = InputValidator.sanitizeString(`
Review this pull request for:
1. Security vulnerabilities
2. Code quality issues
3. Performance problems
4. Best practices violations
5. Test coverage

PR Title: ${InputValidator.sanitizeString(title, 200)}
Changed Files: ${files.length}

Files:
${files.map((file: any) => `
${InputValidator.sanitizeString(file.filename, 255)}
+${file.additions} -${file.deletions}
Patch:
${InputValidator.sanitizeString(file.patch || '', 5000)}
`).join('\n')}

Provide review in JSON format:
{
  "overallScore": number (0-100),
  "issues": {
    "critical": ["issue1", "issue2"],
    "warning": ["warning1", "warning2"],
    "suggestion": ["suggestion1", "suggestion2"]
  },
  "approved": boolean,
  "summary": "brief summary"
}
    `.trim(), 10000);

    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'You are a senior code reviewer. Provide constructive, actionable feedback.',
      message: codeAnalysisPrompt,
    });

    let analysis;
    try {
      analysis = await agent.action();
    } catch (error) {
      this.logger.error({ msg: 'AI analysis failed', correlationId }, error);
      throw new Error('Failed to analyze code with AI');
    }

    let reviewData;
    try {
      reviewData = JSON.parse(analysis.data.response);
    } catch (parseError) {
      this.logger.warn({ msg: 'Could not parse AI response as JSON', correlationId });
      // Fallback if AI response is not valid JSON
      reviewData = {
        overallScore: 70,
        issues: {
          critical: [],
          warning: ['Could not parse AI response'],
          suggestion: ['Manual review required'],
        },
        approved: false,
        summary: InputValidator.sanitizeString(analysis.data.response, 500),
      };
    }

    // Step 3: Post review as comment
    const commentBody = InputValidator.sanitizeString(`
## 🤖 Automated Code Review

**Overall Score:** ${reviewData.overallScore}/100
**Status:** ${reviewData.approved ? '✅ Approved' : '❌ Needs Changes'}

### Summary
${InputValidator.sanitizeString(reviewData.summary || '', 1000)}

### Issues

${reviewData.issues.critical.length > 0 ? `
#### 🚨 Critical Issues
${reviewData.issues.critical.map((issue: string) => `- ${InputValidator.sanitizeString(issue, 500)}`).join('\n')}
` : ''}

${reviewData.issues.warning.length > 0 ? `
#### ⚠️ Warnings
${reviewData.issues.warning.map((issue: string) => `- ${InputValidator.sanitizeString(issue, 500)}`).join('\n')}
` : ''}

${reviewData.issues.suggestion.length > 0 ? `
#### 💡 Suggestions
${reviewData.issues.suggestion.map((issue: string) => `- ${InputValidator.sanitizeString(issue, 500)}`).join('\n')}
` : ''}

---
*This review was automatically generated by BubbleLab Code Review Automation*
    `.trim(), 10000);

    const postComment = new HttpBubble({
      url: `https://api.github.com/repos/${sanitizedRepository}/issues/${number}/comments`,
      method: 'POST',
      headers: {
        'Authorization': `token ${process.env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      body: {
        body: commentBody,
      },
      timeout: 10000,
    });

    try {
      await postComment.action();
    } catch (error) {
      this.logger.error({ msg: 'Failed to post comment', correlationId, prNumber: number }, error);
      throw new Error('Failed to post review comment');
    }

    // Step 4: Add label if not approved
    if (!reviewData.approved) {
      const addLabel = new HttpBubble({
        url: `https://api.github.com/repos/${sanitizedRepository}/issues/${number}/labels`,
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GITHUB_PAT}`,
          'Accept': 'application/vnd.github.v3+json',
        },
        body: {
          labels: ['review-required'],
        },
        timeout: 10000,
      });

      try {
        await addLabel.action();
      } catch (error) {
        this.logger.warn({ msg: 'Failed to add label', correlationId, prNumber: number }, error);
      }
    }

    const result: CodeReviewResult = {
      prNumber: number,
      repository: sanitizedRepository,
      timestamp,
      overallScore: reviewData.overallScore,
      issues: reviewData.issues,
      approved: reviewData.approved,
      correlationId,
    };

    // Step 5: Send notification
    if (!result.approved && process.env.SLACK_WEBHOOK_URL) {
      try {
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message: InputValidator.sanitizeString(`🔍 Code Review Alert\n\nPR #${number} by ${InputValidator.sanitizeString(author, 100)}\nRepository: ${sanitizedRepository}\nScore: ${result.overallScore}/100\n\nCritical Issues: ${result.issues.critical.length}`, 2000),
        });

        await slack.action();
      } catch (error) {
        this.logger.warn({ msg: 'Slack notification failed', correlationId }, error);
      }
    }

    this.logger.info({
      msg: 'Code review completed',
      correlationId,
      prNumber: number,
      approved: result.approved,
      score: result.overallScore,
    });

    return result;
  }
}

export default CodeReviewAutomation;
