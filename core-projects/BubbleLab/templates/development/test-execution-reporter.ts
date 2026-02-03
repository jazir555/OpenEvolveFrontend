/**
 * Test Execution Reporter
 * Purpose: Execute test suites and generate comprehensive reports
 * Category: Development Workflow
 * Event Type: schedule/cron
 * Schedule: 0 2 * * * (Daily at 2 AM)
 *
 * Required Credentials:
 * - GITHUB_PAT: GitHub API access
 * - POSTGRES_CONNECTION_STRING: Test results database
 * - API_KEY: API key for authentication (required)
 * - SLACK_WEBHOOK_URL: Slack webhook for notifications (optional)
 * - GMAIL_CRED: For email reports (optional)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - API key authentication
 * - Input validation (Zod schemas)
 * - Rate limiting
 * - SQL injection prevention (parameterized queries)
 * - Error message sanitization
 * - Structured logging with correlation IDs
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  GmailBubble,
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
  SecuritySchemas,
  validateSqlIdentifier,
  buildParameterizedQuery,
} from '../security-utils';

interface TestSuite {
  name: string;
  type: 'unit' | 'integration' | 'e2e';
  framework: string;
  command: string;
}

interface TestResult {
  suite: string;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  coverage?: number;
}

interface TestReportResult {
  timestamp: string;
  totalSuites: number;
  totalTests: number;
  totalPassed: number;
  totalFailed: number;
  coverage: number;
  duration: number;
  results: TestResult[];
  correlationId: string;
}

// Input validation schemas
const TestSuiteNameSchema = z.string().min(1).max(100).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid test suite name');
const TestFrameworkSchema = z.string().min(1).max(50).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid framework name');

// Security: Environment variable validation
validateEnvironment({
  required: ['GITHUB_PAT', 'POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL', 'GMAIL_CRED', 'CI_API_URL'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    GITHUB_PAT: SecuritySchemas.token,
    POSTGRES_CONNECTION_STRING: z.string().min(20),
  },
});

export class TestExecutionReporter extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 2 * * *';
  readonly name = 'Test Execution Reporter';
  readonly description = 'Execute test suites and generate comprehensive reports';

  // Security: Rate limiter
  private rateLimiter = new RateLimiter({ maxRequests: 30, windowMs: 60000 });
  private logger = new StructuredLogger('test-execution-reporter');

  async handle(payload: CronEvent): Promise<TestReportResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    const timestamp = new Date().toISOString();
    const startTime = Date.now();

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

    this.logger.info({ msg: 'Starting test execution', correlationId });

    // Define test suites
    const testSuites: TestSuite[] = [
      { name: 'backend-unit', type: 'unit', framework: 'pytest', command: 'pytest tests/unit/' },
      { name: 'backend-integration', type: 'integration', framework: 'pytest', command: 'pytest tests/integration/' },
      { name: 'frontend-unit', type: 'unit', framework: 'jest', command: 'npm test -- --coverage' },
      { name: 'e2e-tests', type: 'e2e', framework: 'cypress', command: 'npm run test:e2e' },
    ];

    const allResults: TestResult[] = [];
    let totalTests = 0;
    let totalPassed = 0;
    let totalFailed = 0;
    let totalCoverage = 0;

    // Step 1: Execute each test suite
    for (const suite of testSuites) {
      try {
        // Security: Validate test suite name
        const sanitizedName = TestSuiteNameSchema.parse(suite.name);

        const executeTests = new HttpBubble({
          url: `${process.env.CI_API_URL}/tests/run`,
          method: 'POST',
          body: {
            suite: sanitizedName,
            command: InputValidator.sanitizeString(suite.command, 500),
            framework: TestFrameworkSchema.parse(suite.framework),
          },
          timeout: 600000, // 10 minutes max per suite
        });

        const result = await executeTests.action();
        const testData = result.data;

        const testResult: TestResult = {
          suite: sanitizedName,
          passed: InputValidator.sanitizeNumber(testData.passed || 0, 0),
          failed: InputValidator.sanitizeNumber(testData.failed || 0, 0),
          skipped: InputValidator.sanitizeNumber(testData.skipped || 0, 0),
          duration: InputValidator.sanitizeNumber(testData.duration || 0, 0),
          coverage: testData.coverage ? InputValidator.sanitizeNumber(testData.coverage, 0, 100) : undefined,
        };

        allResults.push(testResult);
        totalTests += testResult.passed + testResult.failed + testResult.skipped;
        totalPassed += testResult.passed;
        totalFailed += testResult.failed;
        if (testResult.coverage) {
          totalCoverage += testResult.coverage;
        }

      } catch (error) {
        this.logger.error({
          msg: 'Test suite failed',
          correlationId,
          suite: suite.name,
        }, error);
        allResults.push({
          suite: TestSuiteNameSchema.parse(suite.name),
          passed: 0,
          failed: 1,
          skipped: 0,
          duration: 0,
        });
        totalFailed++;
      }
    }

    const duration = Date.now() - startTime;
    const avgCoverage = totalCoverage / testSuites.length;

    const reportResult: TestReportResult = {
      timestamp,
      totalSuites: testSuites.length,
      totalTests,
      totalPassed,
      totalFailed,
      coverage: avgCoverage,
      duration,
      results: allResults,
      correlationId,
    };

    // Step 2: Analyze results with AI
    let analysis;
    try {
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'Analyze test results and provide insights, trends, and recommendations',
        message: InputValidator.sanitizeString(JSON.stringify(reportResult, null, 2), 10000),
      });

      const agentResult = await agent.action();
      analysis = agentResult.data.response;
    } catch (error) {
      this.logger.warn({ msg: 'AI analysis failed', correlationId }, error);
      analysis = 'Analysis unavailable';
    }

    // Step 3: Store results in database (SQL injection protection)
    try {
      // Security: Validate table name
      const tableName = validateSqlIdentifier('test_reports', 'table');

      // Security: Use parameterized query to prevent SQL injection
      const { query, params } = buildParameterizedQuery(
        `
        INSERT INTO ${tableName}
        (timestamp, total_tests, passed, failed, coverage, duration, results, analysis)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        `,
        [
          timestamp,
          InputValidator.sanitizeNumber(totalTests, 0),
          InputValidator.sanitizeNumber(totalPassed, 0),
          InputValidator.sanitizeNumber(totalFailed, 0),
          InputValidator.sanitizeNumber(avgCoverage, 0, 100),
          InputValidator.sanitizeNumber(duration, 0),
          JSON.stringify(allResults),
          InputValidator.sanitizeString(analysis, 5000),
        ]
      );

      const storeResults = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query,
        params,
      });

      await storeResults.action();
    } catch (error) {
      this.logger.error({ msg: 'Failed to store results', correlationId }, error);
      throw new Error('Failed to store test results in database');
    }

    // Step 4: Create HTML report
    const htmlReport = `
<!DOCTYPE html>
<html>
<head>
  <title>Test Report - ${timestamp}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .passed { color: green; }
    .failed { color: red; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
  </style>
</head>
<body>
  <h1>Test Report</h1>
  <p><strong>Generated:</strong> ${timestamp}</p>

  <div class="summary">
    <h2>Summary</h2>
    <p>Total Tests: <strong>${totalTests}</strong></p>
    <p class="passed">Passed: <strong>${totalPassed}</strong></p>
    <p class="failed">Failed: <strong>${totalFailed}</strong></p>
    <p>Coverage: <strong>${avgCoverage.toFixed(1)}%</strong></p>
    <p>Duration: <strong>${(duration / 1000).toFixed(2)}s</strong></p>
  </div>

  <h2>Test Suites</h2>
  <table>
    <tr>
      <th>Suite</th>
      <th>Passed</th>
      <th>Failed</th>
      <th>Skipped</th>
      <th>Duration</th>
      <th>Coverage</th>
    </tr>
    ${allResults.map(r => `
    <tr>
      <td>${InputValidator.sanitizeString(r.suite, 100)}</td>
      <td class="passed">${r.passed}</td>
      <td class="failed">${r.failed}</td>
      <td>${r.skipped}</td>
      <td>${(r.duration / 1000).toFixed(2)}s</td>
      <td>${r.coverage ? r.coverage.toFixed(1) + '%' : 'N/A'}</td>
    </tr>
    `).join('')}
  </table>

  <h2>AI Analysis</h2>
  <div style="background: #e3f2fd; padding: 20px; border-radius: 8px;">
    <pre>${InputValidator.sanitizeString(analysis, 2000)}</pre>
  </div>
</body>
</html>
    `.trim();

    // Step 5: Send notifications
    const statusEmoji = totalFailed === 0 ? '✅' : '❌';
    const message = InputValidator.sanitizeString(`
${statusEmoji} Test Execution Report

Total: ${totalTests} tests
Passed: ${totalPassed} ✅
Failed: ${totalFailed} ${totalFailed > 0 ? '❌' : ''}
Coverage: ${avgCoverage.toFixed(1)}%
Duration: ${(duration / 1000).toFixed(2)}s

${allResults.map(r => `
${InputValidator.sanitizeString(r.suite, 100)}: ${r.passed}/${r.passed + r.failed} passed${r.coverage ? ` (${r.coverage.toFixed(1)}% coverage)` : ''}
`).join('\n')}

AI Analysis:
${InputValidator.sanitizeString(analysis.substring(0, 500), 500)}...
    `.trim(), 5000);

    if (process.env.SLACK_WEBHOOK_URL) {
      try {
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message,
        });

        await slack.action();
      } catch (error) {
        this.logger.warn({ msg: 'Slack notification failed', correlationId }, error);
      }
    }

    if (process.env.GMAIL_CRED && totalFailed > 0) {
      try {
        const email = new GmailBubble({
          to: 'team@openevolve.com',
          subject: `Test Report: ${totalFailed} failures`,
          body: htmlReport,
        });

        await email.action();
      } catch (error) {
        this.logger.warn({ msg: 'Email notification failed', correlationId }, error);
      }
    }

    this.logger.info({
      msg: 'Test execution completed',
      correlationId,
      totalTests,
      totalFailed,
      coverage: avgCoverage,
    });

    return reportResult;
  }
}

export default TestExecutionReporter;
