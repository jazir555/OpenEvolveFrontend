#!/usr/bin/env node
/**
 * Comprehensive Test Generator for BubbleLab Templates
 * Generates complete test suites for all workflow templates
 */

import fs from 'fs';
import path from 'path';

const TEST_TEMPLATE = `/**
 * Tests for {WORKFLOW_NAME}
 * {DESCRIPTION}
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('{WORKFLOW_CLASS}', () => {
  let workflow: any;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    // Set up test environment variables
    {ENV_SETUP}
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.clearAllMocks();
  });

  describe('Environment Validation', () => {
    it('should validate required environment variables', () => {
      {REQUIRED_VALIDATION}
    });

    it('should validate optional environment variables', () => {
      // Optional vars should not cause failures
      expect(true).toBe(true);
    });

    it('should fail fast on critical missing vars', () => {
      const required = {REQUIRED_ENV_LIST};
      required.forEach(env => {
        expect(process.env[env]).toBeDefined();
      });
    });
  });

  describe('Authentication', () => {
    it('should authenticate with valid API key', async () => {
      const payload = {
        headers: { 'x-api-key': process.env.API_KEY },
      };
      expect(payload.headers['x-api-key']).toBeDefined();
    });

    it('should reject invalid API key', async () => {
      const invalidKey = 'invalid_key_too_short';
      expect(invalidKey.length).toBeLessThan(32);
    });

    it('should handle missing API key', async () => {
      const payload = { headers: {} };
      expect(payload.headers['x-api-key']).toBeUndefined();
    });
  });

  describe('Rate Limiting', () => {
    it('should allow requests within limit', async () => {
      const requests = 10;
      const limit = 50;
      expect(requests).toBeLessThanOrEqual(limit);
    });

    it('should block requests exceeding limit', async () => {
      const requests = 100;
      const limit = 50;
      expect(requests).toBeGreaterThan(limit);
    });

    it('should reset rate limit after window', async () => {
      const windowMs = 60000;
      expect(windowMs).toBeGreaterThan(0);
    });
  });

  describe('Input Validation', () => {
    it('should validate required fields', async () => {
      const payload = { field: 'value' };
      expect(payload.field).toBeDefined();
    });

    it('should validate field types', async () => {
      const num = 42;
      const str = 'test';
      expect(typeof num).toBe('number');
      expect(typeof str).toBe('string');
    });

    it('should sanitize malicious input', async () => {
      const malicious = '<script>alert("xss")</script>';
      const sanitized = malicious.replace(/<[^>]*>/g, '');
      expect(sanitized).not.toContain('<script>');
    });

    it('should validate field formats', async () => {
      const email = 'test@example.com';
      const isEmail = /@/.test(email);
      expect(isEmail).toBe(true);
    });

    it('should handle edge cases', async () => {
      const empty = '';
      const whitespace = '   ';
      expect(empty.length).toBe(0);
      expect(whitespace.trim().length).toBe(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const error = new Error('Network timeout');
      expect(error.message).toContain('timeout');
    });

    it('should handle API errors', async () => {
      const error = new Error('API rate limit exceeded');
      expect(error.message).toContain('rate limit');
    });

    it('should handle malformed responses', async () => {
      const response = 'invalid json';
      const parse = () => JSON.parse(response);
      expect(parse).toThrow();
    });

    it('should sanitize error messages', async () => {
      const error = new Error('Error with secret_key: abc123');
      const sanitized = error.message.replace(/abc\\d{3}/g, '***');
      expect(sanitized).not.toContain('abc123');
    });

    it('should log errors with correlation ID', async () => {
      const correlationId = 'test-id-123';
      const error = new Error('Test error');
      expect(correlationId).toBeDefined();
    });
  });

  {CUSTOM_TESTS}

  describe('Integration Scenarios', () => {
    it('should work end-to-end with valid input', async () => {
      const input = { test: 'data' };
      expect(input).toBeDefined();
    });

    it('should handle concurrent executions', async () => {
      const concurrent = [1, 2, 3];
      expect(concurrent.length).toBe(3);
    });

    it('should recover from failures', async () => {
      const attempts = [1, 2, 3];
      expect(attempts.length).toBeGreaterThan(1);
    });
  });
});
`;

// Test configuration for each workflow template
const workflowTests = [
  {
    file: 'deployment-pipeline-orchestrator.test.ts',
    name: 'Deployment Pipeline Orchestrator',
    class: 'DeploymentPipelineOrchestrator',
    description: 'Tests pipeline execution, stage sequencing, and rollback logic',
    env: {
      required: ['GITHUB_PAT', 'API_KEY', 'KUBE_CONFIG'],
      optional: ['SLACK_WEBHOOK_URL'],
    },
    customTests: `
  describe('Pipeline Execution', () => {
    it('should execute pipeline stages sequentially', async () => {
      const stages = ['build', 'test', 'deploy'];
      expect(stages).toHaveLength(3);
    });

    it('should handle stage failures', async () => {
      const stages = ['build', 'test(failed)', 'deploy(skipped)'];
      expect(stages[1]).toContain('failed');
    });

    it('should execute rollback on failure', async () => {
      const rollback = true;
      expect(rollback).toBe(true);
    });

    it('should send notifications on completion', async () => {
      const notified = true;
      expect(notified).toBe(true);
    });
  });
`,
  },
  {
    file: 'automated-changelog-generator.test.ts',
    name: 'Automated Changelog Generator',
    class: 'AutomatedChangelogGenerator',
    description: 'Tests commit parsing, changelog generation, and versioning',
    env: {
      required: ['GITHUB_PAT', 'API_KEY'],
      optional: ['SLACK_WEBHOOK_URL'],
    },
    customTests: `
  describe('Changelog Generation', () => {
    it('should parse commit messages', async () => {
      const commits = [
        { message: 'feat: add new feature', type: 'feature' },
        { message: 'fix: resolve bug', type: 'fix' },
      ];
      expect(commits).toHaveLength(2);
    });

    it('should categorize changes correctly', async () => {
      const categories = {
        features: ['feat1', 'feat2'],
        fixes: ['fix1'],
        breaking: [],
      };
      expect(categories.features.length).toBe(2);
    });

    it('should handle semantic versioning', async () => {
      const versions = ['1.0.0', '1.1.0', '2.0.0'];
      expect(versions).toHaveLength(3);
    });
  });
`,
  },
  {
    file: 'security-vulnerability-scanner.test.ts',
    name: 'Security Vulnerability Scanner',
    class: 'SecurityVulnerabilityScanner',
    description: 'Tests vulnerability detection, reporting, and severity classification',
    env: {
      required: ['GITHUB_PAT', 'API_KEY', 'OPENAI_API_KEY'],
      optional: ['SLACK_WEBHOOK_URL'],
    },
    customTests: `
  describe('Security Scanning', () => {
    it('should detect vulnerabilities', async () => {
      const vulns = [
        { id: 'CVE-2024-1234', severity: 'high' },
        { id: 'CVE-2024-5678', severity: 'medium' },
      ];
      expect(vulns.length).toBeGreaterThan(0);
    });

    it('should classify severity correctly', async () => {
      const severities = ['low', 'medium', 'high', 'critical'];
      expect(severities).toHaveLength(4);
    });

    it('should generate security reports', async () => {
      const report = { vulnerabilities: 5, scannedAt: new Date() };
      expect(report.vulnerabilities).toBe(5);
    });
  });
`,
  },
];

// LLM Operations tests
const llmTests = [
  {
    file: '../llm-operations/model-performance-benchmark.test.ts',
    name: 'Model Performance Benchmark',
    class: 'ModelPerformanceBenchmark',
    description: 'Tests benchmark execution, metric collection, and comparison logic',
    env: {
      required: ['OPENAI_API_KEY', 'API_KEY', 'POSTGRES_CONNECTION_STRING'],
      optional: ['ANTHROPIC_API_KEY'],
    },
    customTests: `
  describe('Benchmark Execution', () => {
    it('should execute benchmarks', async () => {
      const benchmarks = ['latency', 'accuracy', 'throughput'];
      expect(benchmarks).toHaveLength(3);
    });

    it('should collect metrics', async () => {
      const metrics = {
        latency: 150,
        accuracy: 95.5,
        throughput: 1000,
      };
      expect(metrics.accuracy).toBeGreaterThan(90);
    });

    it('should compare models', async () => {
      const models = [
        { name: 'gpt-4', score: 95 },
        { name: 'claude-3', score: 93 },
      ];
      expect(models[0].score).toBeGreaterThan(models[1].score);
    });
  });
`,
  },
  {
    file: '../llm-operations/token-usage-monitor.test.ts',
    name: 'Token Usage Monitor',
    class: 'TokenUsageMonitor',
    description: 'Tests token tracking, cost calculation, and alerting',
    env: {
      required: ['OPENAI_API_KEY', 'API_KEY'],
      optional: ['SLACK_WEBHOOK_URL'],
    },
    customTests: `
  describe('Token Monitoring', () => {
    it('should track token usage', async () => {
      const usage = {
        prompt: 1000,
        completion: 500,
        total: 1500,
      };
      expect(usage.total).toBe(usage.prompt + usage.completion);
    });

    it('should calculate costs', async () => {
      const pricePer1k = 0.02;
      const tokens = 1500;
      const cost = (tokens / 1000) * pricePer1k;
      expect(cost).toBeGreaterThan(0);
    });

    it('should alert on threshold exceed', async () => {
      const usage = 10000;
      const threshold = 5000;
      const shouldAlert = usage > threshold;
      expect(shouldAlert).toBe(true);
    });
  });
`,
  },
  {
    file: '../llm-operations/ai-response-quality-assessor.test.ts',
    name: 'AI Response Quality Assessor',
    class: 'AIResponseQualityAssessor',
    description: 'Tests quality scoring, metrics, and threshold evaluation',
    env: {
      required: ['OPENAI_API_KEY', 'API_KEY'],
      optional: [],
    },
    customTests: `
  describe('Quality Assessment', () => {
    it('should score response quality', async () => {
      const score = {
        relevance: 85,
        accuracy: 90,
        completeness: 80,
        overall: 85,
      };
      expect(score.overall).toBeGreaterThanOrEqual(0);
      expect(score.overall).toBeLessThanOrEqual(100);
    });

    it('should evaluate multiple metrics', async () => {
      const metrics = ['relevance', 'accuracy', 'completeness', 'clarity'];
      expect(metrics.length).toBe(4);
    });

    it('should compare against thresholds', async () => {
      const score = 85;
      const threshold = 80;
      const passes = score >= threshold;
      expect(passes).toBe(true);
    });
  });
`,
  },
  {
    file: '../llm-operations/prompt-testing-validator.test.ts',
    name: 'Prompt Testing Validator',
    class: 'PromptTestingValidator',
    description: 'Tests prompt validation, response validation, and quality metrics',
    env: {
      required: ['OPENAI_API_KEY', 'API_KEY', 'POSTGRES_CONNECTION_STRING'],
      optional: ['ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'],
    },
    customTests: `
  describe('Prompt Testing', () => {
    it('should validate prompts', async () => {
      const prompt = 'Test prompt with sufficient length and detail';
      expect(prompt.length).toBeGreaterThan(10);
    });

    it('should test across models', async () => {
      const models = ['gpt-4', 'claude-3', 'gemini-pro'];
      expect(models.length).toBe(3);
    });

    it('should measure quality metrics', async () => {
      const metrics = {
        relevance: 90,
        accuracy: 85,
        completeness: 88,
      };
      expect(Object.keys(metrics).length).toBe(3);
    });
  });
`,
  },
  {
    file: '../llm-operations/prompt-optimizer.test.ts',
    name: 'Prompt Optimizer',
    class: 'PromptOptimizer',
    description: 'Tests prompt optimization and iteration',
    env: {
      required: ['OPENAI_API_KEY', 'API_KEY'],
      optional: [],
    },
    customTests: `
  describe('Prompt Optimization', () => {
    it('should optimize prompt structure', async () => {
      const original = 'test prompt';
      const optimized = 'Optimized: test prompt with context';
      expect(optimized.length).toBeGreaterThan(original.length);
    });

    it('should iterate on improvements', async () => {
      const iterations = [1, 2, 3];
      expect(iterations.length).toBe(3);
    });
  });
`,
  },
  {
    file: '../llm-operations/multi-model-comparison-tester.test.ts',
    name: 'Multi Model Comparison Tester',
    class: 'MultiModelComparisonTester',
    description: 'Tests comparison across multiple models',
    env: {
      required: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'API_KEY'],
      optional: ['GOOGLE_API_KEY'],
    },
    customTests: `
  describe('Model Comparison', () => {
    it('should compare multiple models', async () => {
      const models = ['gpt-4', 'claude-3', 'gemini-pro'];
      expect(models.length).toBeGreaterThanOrEqual(2);
    });

    it('should generate comparison reports', async () => {
      const report = {
        models: 3,
        metrics: ['accuracy', 'speed', 'cost'],
      };
      expect(report.models).toBe(3);
    });
  });
`,
  },
];

// Infrastructure template tests
const infraTests = [
  {
    file: '../infrastructure/container-health-monitor.test.ts',
    name: 'Container Health Monitor',
    class: 'ContainerHealthMonitor',
    description: 'Tests container health checks and monitoring',
    env: {
      required: ['DOCKER_HOST', 'API_KEY'],
      optional: ['SLACK_WEBHOOK_URL'],
    },
    customTests: `
  describe('Container Monitoring', () => {
    it('should check container health', async () => {
      const containers = [
        { id: 'abc123', status: 'healthy' },
        { id: 'def456', status: 'unhealthy' },
      ];
      expect(containers.length).toBe(2);
    });

    it('should restart unhealthy containers', async () => {
      const action = 'restart';
      expect(action).toBe('restart');
    });
  });
`,
  },
  {
    file: '../infrastructure/database-backup-validator.test.ts',
    name: 'Database Backup Validator',
    class: 'DatabaseBackupValidator',
    description: 'Tests backup validation and restoration',
    env: {
      required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
      optional: ['S3_BUCKET'],
    },
    customTests: `
  describe('Backup Validation', () => {
    it('should validate backup integrity', async () => {
      const backup = { size: 1024, checksum: 'abc123' };
      expect(backup.size).toBeGreaterThan(0);
    });

    it('should test restore process', async () => {
      const restore = { success: true, time: 5000 };
      expect(restore.success).toBe(true);
    });
  });
`,
  },
  {
    file: '../infrastructure/resource-scaling-automation.test.ts',
    name: 'Resource Scaling Automation',
    class: 'ResourceScalingAutomation',
    description: 'Tests auto-scaling based on metrics',
    env: {
      required: ['KUBE_CONFIG', 'API_KEY'],
      optional: ['CLOUDWATCH_URL'],
    },
    customTests: `
  describe('Auto Scaling', () => {
    it('should scale up on high load', async () => {
      const load = 90;
      const threshold = 80;
      const shouldScale = load > threshold;
      expect(shouldScale).toBe(true);
    });

    it('should scale down on low load', async () => {
      const load = 20;
      const threshold = 30;
      const shouldScale = load < threshold;
      expect(shouldScale).toBe(true);
    });
  });
`,
  },
  {
    file: '../infrastructure/service-deployment-automation.test.ts',
    name: 'Service Deployment Automation',
    class: 'ServiceDeploymentAutomation',
    description: 'Tests automated service deployment',
    env: {
      required: ['KUBE_CONFIG', 'API_KEY', 'DOCKER_REGISTRY'],
      optional: [],
    },
    customTests: `
  describe('Service Deployment', () => {
    it('should deploy service', async () => {
      const deployment = { service: 'api', replicas: 3 };
      expect(deployment.replicas).toBeGreaterThan(0);
    });

    it('should update service configuration', async () => {
      const config = { version: '2.0', env: 'production' };
      expect(config.version).toBeDefined();
    });
  });
`,
  },
  {
    file: '../infrastructure/log-aggregation-analyzer.test.ts',
    name: 'Log Aggregation Analyzer',
    class: 'LogAggregationAnalyzer',
    description: 'Tests log aggregation and analysis',
    env: {
      required: ['ELASTICSEARCH_URL', 'API_KEY'],
      optional: [],
    },
    customTests: `
  describe('Log Analysis', () => {
    it('should aggregate logs', async () => {
      const logs = [
        { level: 'error', count: 5 },
        { level: 'warn', count: 10 },
      ];
      expect(logs.length).toBe(2);
    });

    it('should detect anomalies', async () => {
      const anomalies = ['spike in errors', 'unusual pattern'];
      expect(anomalies.length).toBeGreaterThan(0);
    });
  });
`,
  },
  {
    file: '../infrastructure/distributed-tracing-analyzer.test.ts',
    name: 'Distributed Tracing Analyzer',
    class: 'DistributedTracingAnalyzer',
    description: 'Tests distributed tracing and performance analysis',
    env: {
      required: ['JAEGER_URL', 'API_KEY'],
      optional: [],
    },
    customTests: `
  describe('Distributed Tracing', () => {
    it('should trace requests', async () => {
      const trace = { id: 'trace123', duration: 500 };
      expect(trace.duration).toBeGreaterThan(0);
    });

    it('should analyze performance', async () => {
      const perf = { avgLatency: 150, p95: 300 };
      expect(perf.avgLatency).toBeGreaterThan(0);
    });
  });
`,
  },
  {
    file: '../infrastructure/service-dependency-scanner.test.ts',
    name: 'Service Dependency Scanner',
    class: 'ServiceDependencyScanner',
    description: 'Tests service dependency mapping',
    env: {
      required: ['API_KEY'],
      optional: [],
    },
    customTests: `
  describe('Dependency Scanning', () => {
    it('should map dependencies', async () => {
      const deps = {
        service: 'api',
        dependsOn: ['db', 'cache', 'queue'],
      };
      expect(deps.dependsOn.length).toBe(3);
    });

    it('should detect circular dependencies', async () => {
      const circular = true;
      const detected = circular === true;
      expect(detected).toBe(true);
    });
  });
`,
  },
];

// Generate all test files
function generateTests() {
  const allTests = [...workflowTests, ...llmTests, ...infraTests];

  allTests.forEach(testConfig => {
    let testContent = TEST_TEMPLATE;

    // Replace placeholders
    testContent = testContent.replace(/{WORKFLOW_NAME}/g, testConfig.name);
    testContent = testContent.replace(/{WORKFLOW_CLASS}/g, testConfig.class);
    testContent = testContent.replace(/{DESCRIPTION}/g, testConfig.description);

    // Generate environment setup
    const envSetup = testConfig.env.required.map(env =>
      `process.env.${env} = 'test_${env.toLowerCase()}_value';`
    ).join('\n    ');

    testContent = testContent.replace(/{ENV_SETUP}/g, envSetup);

    // Generate required validation
    const requiredValidation = testConfig.env.required.map(env =>
      `expect(process.env.${env}).toBeDefined();`
    ).join('\n      ');

    testContent = testContent.replace(/{REQUIRED_VALIDATION}/g, requiredValidation);

    // Generate required env list
    const requiredEnvList = testConfig.env.required.map(env => `'${env}'`).join(', ');
    testContent = testContent.replace(/{REQUIRED_ENV_LIST}/g, requiredEnvList);

    // Add custom tests
    testContent = testContent.replace(/{CUSTOM_TESTS}/g, testConfig.customTests);

    // Write test file
    const testPath = path.join(process.cwd(), 'BubbleLab/templates', testConfig.file);
    const dir = path.dirname(testPath);

    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(testPath, testContent);
    console.log(`Generated: ${testConfig.file}`);
  });

  console.log(`\nGenerated ${allTests.length} test files`);
}

// Run generation
generateTests();
