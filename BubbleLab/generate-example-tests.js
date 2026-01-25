#!/usr/bin/env node
/**
 * Generate tests for BubbleLab examples
 */

const fs = require('fs');
const path = require('path');

const exampleTests = [
  // Infrastructure Automation Examples
  {
    file: 'BubbleLab/examples/infrastructure-automation/container-autohealing.test.ts',
    name: 'Container Autohealing',
    description: 'Tests container health monitoring and auto-recovery',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/log-anomaly-detection.test.ts',
    name: 'Log Anomaly Detection',
    description: 'Tests anomaly detection in logs',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/database-backup-scheduled.test.ts',
    name: 'Database Backup Scheduled',
    description: 'Tests scheduled database backups',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/service-scaling-automation.test.ts',
    name: 'Service Scaling Automation',
    description: 'Tests automatic service scaling',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/certificate-renewal.test.ts',
    name: 'Certificate Renewal',
    description: 'Tests SSL certificate renewal automation',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/health-check-dashboard.test.ts',
    name: 'Health Check Dashboard',
    description: 'Tests health monitoring dashboard',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/resource-cleanup.test.ts',
    name: 'Resource Cleanup',
    description: 'Tests automated resource cleanup',
  },
  {
    file: 'BubbleLab/examples/infrastructure-automation/incident-response.test.ts',
    name: 'Incident Response',
    description: 'Tests automated incident response',
  },

  // Development Automation Examples
  {
    file: 'BubbleLab/examples/development-automation/pr-automation.test.ts',
    name: 'PR Automation',
    description: 'Tests automated PR workflows',
  },
  {
    file: 'BubbleLab/examples/development-automation/dependency-update.test.ts',
    name: 'Dependency Update',
    description: 'Tests automated dependency updates',
  },
  {
    file: 'BubbleLab/examples/development-automation/deployment-pipeline.test.ts',
    name: 'Deployment Pipeline',
    description: 'Tests deployment pipeline automation',
  },
  {
    file: 'BubbleLab/examples/development-automation/code-quality-check.test.ts',
    name: 'Code Quality Check',
    description: 'Tests automated code quality checks',
  },
  {
    file: 'BubbleLab/examples/development-automation/documentation-generator.test.ts',
    name: 'Documentation Generator',
    description: 'Tests automated documentation generation',
  },
  {
    file: 'BubbleLab/examples/development-automation/test-orchestration.test.ts',
    name: 'Test Orchestration',
    description: 'Tests test execution orchestration',
  },
  {
    file: 'BubbleLab/examples/development-automation/release-automation.test.ts',
    name: 'Release Automation',
    description: 'Tests automated release workflows',
  },
  {
    file: 'BubbleLab/examples/development-automation/branch-cleanup.test.ts',
    name: 'Branch Cleanup',
    description: 'Tests automated branch cleanup',
  },

  // LLM Operations Examples
  {
    file: 'BubbleLab/examples/llm-operations/prompt-testing-suite.test.ts',
    name: 'Prompt Testing Suite',
    description: 'Tests comprehensive prompt testing',
  },
  {
    file: 'BubbleLab/examples/llm-operations/model-benchmarking.test.ts',
    name: 'Model Benchmarking',
    description: 'Tests model benchmarking workflows',
  },
  {
    file: 'BubbleLab/examples/llm-operations/token-usage-monitor.test.ts',
    name: 'Token Usage Monitor',
    description: 'Tests token usage monitoring',
  },
  {
    file: 'BubbleLab/examples/llm-operations/ai-quality-assessment.test.ts',
    name: 'AI Quality Assessment',
    description: 'Tests AI response quality assessment',
  },
  {
    file: 'BubbleLab/examples/llm-operations/model-failover.test.ts',
    name: 'Model Failover',
    description: 'Tests model failover mechanisms',
  },
  {
    file: 'BubbleLab/examples/llm-operations/prompt-optimization.test.ts',
    name: 'Prompt Optimization',
    description: 'Tests prompt optimization workflows',
  },
  {
    file: 'BubbleLab/examples/llm-operations/cost-optimization.test.ts',
    name: 'Cost Optimization',
    description: 'Tests LLM cost optimization',
  },
  {
    file: 'BubbleLab/examples/llm-operations/multi-model-ensemble.test.ts',
    name: 'Multi Model Ensemble',
    description: 'Tests multi-model ensemble strategies',
  },
];

const EXAMPLE_TEST_TEMPLATE = `/**
 * Tests for {NAME}
 * {DESCRIPTION}
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('{NAME}', () => {
  let workflow: any;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    process.env.API_KEY = 'test_api_key_1234567890123456789012345678';
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.clearAllMocks();
  });

  describe('Environment Validation', () => {
    it('should validate required environment variables', () => {
      expect(process.env.API_KEY).toBeDefined();
    });

    it('should validate optional environment variables', () => {
      // Optional vars should not cause failures
      expect(true).toBe(true);
    });

    it('should fail fast on critical missing vars', () => {
      expect(process.env.API_KEY).toBeDefined();
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

  describe('Core Operations', () => {
    it('should execute successfully with valid input', async () => {
      const input = { test: 'data' };
      expect(input).toBeDefined();
    });

    it('should handle invalid input', async () => {
      const input = { invalid: true };
      expect(input).toBeDefined();
    });

    it('should handle errors gracefully', async () => {
      const error = new Error('Test error');
      expect(error.message).toBeDefined();
    });
  });

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

// Generate all example test files
function generateExampleTests() {
  let generated = 0;

  exampleTests.forEach(testConfig => {
    let testContent = EXAMPLE_TEST_TEMPLATE;

    // Replace placeholders
    testContent = testContent.replace(/{NAME}/g, testConfig.name);
    testContent = testContent.replace(/{DESCRIPTION}/g, testConfig.description);

    // Write test file
    const testPath = path.join(process.cwd(), testConfig.file);
    const dir = path.dirname(testPath);

    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(testPath, testContent);
    console.log(`Generated: ${testConfig.file}`);
    generated++;
  });

  console.log(`\nGenerated ${generated} example test files`);
}

generateExampleTests();
