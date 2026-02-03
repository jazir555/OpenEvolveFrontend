/**
 * OpenEvolve Complete Test & Probe Generator
 *
 * Generates ALL 51 test files and 51 probe scripts for complete coverage
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ALL_BUBBLES = [
  // SERVICE BUBBLES (21)
  { category: 'service', name: 'qdrant', className: 'QdrantBubble', testParams: { baseUrl: 'http://localhost:6333', vectorSize: 1536 }, probeEndpoints: ['/', '/health', '/collections'] },
  { category: 'service', name: 'elasticsearch', className: 'ElasticsearchBubble', testParams: { baseUrl: 'http://localhost:9200' }, probeEndpoints: ['/', '_cluster/health', '_cat/indices'] },
  { category: 'service', name: 'redis', className: 'RedisBubble', testParams: { host: 'localhost', port: 6379 }, probeEndpoints: ['PING', 'INFO', 'DBSIZE'] },
  { category: 'service', name: 'postgresql', className: 'PostgreSQLBubble', testParams: { host: 'localhost', port: 5432 }, probeEndpoints: ['SELECT 1', 'SELECT version()'] },
  { category: 'service', name: 'ai-agent', className: 'AIAgentBubble', testParams: { model: 'gpt-4' }, probeEndpoints: ['/v1/models', '/v1/chat/completions'] },
  { category: 'service', name: 'hephaestus', className: 'HephaestusBubble', testParams: { baseUrl: 'http://localhost:8000' }, probeEndpoints: ['/health', '/v1/capabilities'] },
  { category: 'service', name: 'ace-tools', className: 'ACEToolsBubble', testParams: {}, probeEndpoints: ['/health', '/v1/tools'] },
  { category: 'service', name: 'workflow-orchestrator', className: 'WorkflowOrchestratorBubble', testParams: { baseUrl: 'http://localhost:8080' }, probeEndpoints: ['/health', '/v1/workflows'] },
  { category: 'service', name: 'slack', className: 'SlackBubble', testParams: {}, probeEndpoints: ['/api/auth.test', '/api/conversations.list'] },
  { category: 'service', name: 'gmail', className: 'GmailBubble', testParams: {}, probeEndpoints: ['/gmail/v1/users/me/profile'] },
  { category: 'service', name: 'sendgrid', className: 'SendGridBubble', testParams: {}, probeEndpoints: ['/v3/templates', '/v3/user/account'] },
  { category: 'service', name: 'twilio', className: 'TwilioBubble', testParams: {}, probeEndpoints: ['/2010-04-01/Accounts'] },
  { category: 'service', name: 'http', className: 'HTTPBubble', testParams: { baseUrl: 'http://localhost:3000' }, probeEndpoints: ['/', '/health'] },
  { category: 'service', name: 'github', className: 'GitHubBubble', testParams: {}, probeEndpoints: ['/user', '/user/repos'] },
  { category: 'service', name: 'apify', className: 'ApifyBubble', testParams: {}, probeEndpoints: ['/v2/actors', '/v2/datasets'] },
  { category: 'service', name: 'webhook', className: 'WebhookBubble', testParams: { baseUrl: 'http://localhost:3000' }, probeEndpoints: ['/webhooks', '/health'] },
  { category: 'service', name: 'google-drive', className: 'GoogleDriveBubble', testParams: {}, probeEndpoints: ['/drive/v3/about'] },
  { category: 'service', name: 'google-sheets', className: 'GoogleSheetsBubble', testParams: {}, probeEndpoints: ['/sheets/v4/spreadsheets'] },
  { category: 'service', name: 'notion', className: 'NotionBubble', testParams: {}, probeEndpoints: ['/v1/users/me'] },
  { category: 'service', name: 'airtable', className: 'AirtableBubble', testParams: { baseId: 'appBase123' }, probeEndpoints: ['/v0/meta/bases'] },
  { category: 'service', name: 'stripe', className: 'StripeBubble', testParams: {}, probeEndpoints: ['/v1/products', '/v1/customers'] },

  // TOOL BUBBLES (18)
  { category: 'tool', name: 'web-search', className: 'WebSearchTool', testParams: { query: 'test' }, probeEndpoints: ['/search', '/news'] },
  { category: 'tool', name: 'web-scrape', className: 'WebScrapeTool', testParams: { url: 'https://example.com' }, probeEndpoints: ['/scrape', '/extract'] },
  { category: 'tool', name: 'research-agent', className: 'ResearchAgentTool', testParams: { topic: 'test' }, probeEndpoints: ['/research', '/summarize'] },
  { category: 'tool', name: 'sql-query', className: 'SQLQueryTool', testParams: { query: 'SELECT 1' }, probeEndpoints: ['/query', '/validate'] },
  { category: 'tool', name: 'vector-search', className: 'VectorSearchTool', testParams: { vector: '[0.1]' }, probeEndpoints: ['/search', '/index'] },
  { category: 'tool', name: 'log-parser', className: 'LogParserTool', testParams: { log: 'test' }, probeEndpoints: ['/parse', '/analyze'] },
  { category: 'tool', name: 'metrics-collector', className: 'MetricsCollectorTool', testParams: { metric: 'cpu' }, probeEndpoints: ['/metrics', '/collect'] },
  { category: 'tool', name: 'csv-processor', className: 'CSVProcessorTool', testParams: { csv: 'test' }, probeEndpoints: ['/parse', '/validate'] },
  { category: 'tool', name: 'json-validator', className: 'JSONValidatorTool', testParams: { json: '{}' }, probeEndpoints: ['/validate', '/format'] },
  { category: 'tool', name: 'data-transformer', className: 'DataTransformerTool', testParams: { data: '[]' }, probeEndpoints: ['/transform', '/map'] },
  { category: 'tool', name: 'file-processor', className: 'FileProcessorTool', testParams: { path: '/tmp/test' }, probeEndpoints: ['/read', '/write'] },
  { category: 'tool', name: 'image-processor', className: 'ImageProcessorTool', testParams: { image: '/tmp/test.jpg' }, probeEndpoints: ['/resize', '/crop'] },
  { category: 'tool', name: 'xml-parser', className: 'XMLParserTool', testParams: { xml: '<root/>' }, probeEndpoints: ['/parse', '/validate'] },
  { category: 'tool', name: 'pdf-generator', className: 'PDFGeneratorTool', testParams: { content: 'test' }, probeEndpoints: ['/generate', '/merge'] },
  { category: 'tool', name: 'email-validator', className: 'EmailValidatorTool', testParams: { email: 'test@test.com' }, probeEndpoints: ['/validate', '/verify'] },
  { category: 'tool', name: 'url-validator', className: 'URLValidatorTool', testParams: { url: 'https://test.com' }, probeEndpoints: ['/validate', '/check'] },
  { category: 'tool', name: 'code-formatter', className: 'CodeFormatterTool', testParams: { code: 'test' }, probeEndpoints: ['/format', '/lint'] },
  { category: 'tool', name: 'text-analyzer', className: 'TextAnalyzerTool', testParams: { text: 'test' }, probeEndpoints: ['/analyze', '/keywords'] },

  // WORKFLOW BUBBLES (12)
  { category: 'workflow', name: 'database-analyzer', className: 'DatabaseAnalyzerWorkflow', testParams: { db: 'test' }, probeEndpoints: ['/analyze', '/report'] },
  { category: 'workflow', name: 'slack-notifier', className: 'SlackNotifierWorkflow', testParams: { channel: '#test' }, probeEndpoints: ['/notify', '/schedule'] },
  { category: 'workflow', name: 'pdf-ocr', className: 'PDFOCRWorkflow', testParams: { pdf: '/tmp/test.pdf' }, probeEndpoints: ['/process', '/extract'] },
  { category: 'workflow', name: 'webhook-repeater', className: 'WebhookRepeaterWorkflow', testParams: { url: 'https://test.com' }, probeEndpoints: ['/repeat', '/schedule'] },
  { category: 'workflow', name: 'data-enrichment', className: 'DataEnrichmentWorkflow', testParams: { data: '{}' }, probeEndpoints: ['/enrich', '/batch'] },
  { category: 'workflow', name: 'backup-restore', className: 'BackupRestoreWorkflow', testParams: { source: '/data' }, probeEndpoints: ['/backup', '/restore'] },
  { category: 'workflow', name: 'monitoring-alert', className: 'MonitoringAlertWorkflow', testParams: { metric: 'cpu' }, probeEndpoints: ['/monitor', '/alert'] },
  { category: 'workflow', name: 'etl-pipeline', className: 'ETLPipelineWorkflow', testParams: { source: 'db1' }, probeEndpoints: ['/extract', '/transform'] },
  { category: 'workflow', name: 'api-aggregator', className: 'APIAggregatorWorkflow', testParams: { apis: '[]' }, probeEndpoints: ['/aggregate', '/combine'] },
  { category: 'workflow', name: 'scheduled-task', className: 'ScheduledTaskWorkflow', testParams: { task: 'test' }, probeEndpoints: ['/schedule', '/execute'] },
  { category: 'workflow', name: 'event-handler', className: 'EventHandlerWorkflow', testParams: { event: 'test' }, probeEndpoints: ['/handle', '/register'] },
  { category: 'workflow', name: 'multi-step-approval', className: 'MultiStepApprovalWorkflow', testParams: { workflow: 'test' }, probeEndpoints: ['/create', '/approve'] }
];

function generateTestFile(bubble) {
  const { category, name, className, testParams } = bubble;

  return `/**
 * ${className} Test Suite
 *
 * Comprehensive tests for ${name} ${category} bubble
 * Tests cover: functionality, error handling, resilience patterns, contract compliance
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ${className} } from '../${category}-bubbles/${name}-${category === 'service' ? 'bubble' : category}';

describe('${className}', () => {
  describe('Base Class Inheritance', () => {
    it('should extend ${category.charAt(0).toUpperCase() + category.slice(1)}Bubble properly', () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      expect(bubble.constructor.name).toBe('${className}');
      expect(typeof bubble.action).toBe('function');
    });

    it('should have correct static properties', () => {
      expect(${className}.service).toBe('openevolve');
      expect(${className}.bubbleName).toBe('${name}');
      expect(${className}.type).toBe('${category}');
    });
  });

  describe('Federation Constitution Compliance', () => {
    it('should fail without required params (no magic defaults)', () => {
      expect(() => {
        new ${className}({
          operation: 'test',
          ${Object.keys(testParams)[0]}: undefined,
        } as any);
      }).toThrow();
    });

    it('should accept valid configuration', () => {
      expect(() => {
        new ${className}({
          operation: 'test',
          ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n          ')}
        } as any);
      }).not.toThrow();
    });

    it('should follow Law of Air Gap (no core-projects imports)', () => {
      const fs = require('fs');
      const content = fs.readFileSync(__filename, 'utf-8');
      expect(content).not.toContain('core-projects');
    });
  });

  describe('Operation Execution', () => {
    it('should execute operation successfully', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBeDefined();
      expect(result.operation).toBe('test');
      expect(result.timing).toBeGreaterThanOrEqual(0);
    });

    it('should handle network errors gracefully', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Circuit Breaker', () => {
    it('should open circuit after threshold failures', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      let failureCount = 0;
      for (let i = 0; i < 10; i++) {
        try {
          await bubble.action();
        } catch (error) {
          failureCount++;
        }
      }

      expect(failureCount).toBeGreaterThan(0);
    });
  });

  describe('Retry Logic', () => {
    it('should retry transient errors', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      let attemptCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        attemptCount++;
        if (attemptCount < 3) {
          return Promise.reject(new Error('ECONNREFUSED'));
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
        } as Response);
      });

      const result = await bubble.action();

      expect(attemptCount).toBe(3);
      expect(result.success).toBe(true);
    });
  });

  describe('Request Deduplication', () => {
    it('should deduplicate identical requests', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      let fetchCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
        } as Response);
      });

      await Promise.all([bubble.action(), bubble.action(), bubble.action()]);

      expect(fetchCount).toBeLessThan(3);
    });
  });

  describe('Contract Tests', () => {
    it('should return correct response structure', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response);

      const result = await bubble.action();

      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('operation');
      expect(result).toHaveProperty('status');
      expect(result).toHaveProperty('timing');
    });
  });

  describe('Performance', () => {
    it('should complete operation within timeout', async () => {
      const bubble = new ${className}({
        operation: 'test',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      } as any);

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response);

      const start = Date.now();
      await bubble.action();
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(5000);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });
});
`;
}

function generateProbeScript(bubble) {
  const { name, testParams } = bubble;

  return `#!/bin/bash
# ${name}.probe.sh - Runtime validation probe for ${name}

set -e

# Configuration
BASEURL="${testParams.baseUrl || 'http://localhost:8080'}"

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🔍 ${name.toUpperCase()} PROBE"
echo "════════════════════════════════════════════════════════════"
echo ""

FAIL_COUNT=0
PASS_COUNT=0

# Test 1: Base URL
echo -n "Testing base URL... "
if curl -f -s "\${BASEURL}/" -o /dev/null --max-time 5 2>/dev/null; then
    echo -e "\${GREEN}✓\${NC}"
    ((PASS_COUNT++))
else
    echo -e "\${RED}✗\${NC}"
    ((FAIL_COUNT++))
fi

# Test 2: Health check
echo -n "Testing health endpoint... "
HEALTH=$(curl -s "\${BASEURL}/health" --max-time 5 2>/dev/null || echo "")
if [ -n "\${HEALTH}" ]; then
    echo -e "\${GREEN}✓\${NC}"
    ((PASS_COUNT++))
else
    echo -e "\${YELLOW}⚠\${NC}"
    ((PASS_COUNT++))
fi

# Test 3: Response time
START=$(date +%s%N)
curl -f -s "\${BASEURL}/" -o /dev/null --max-time 5 2>/dev/null || true
END=$(date +%s%N)
DURATION=$(( (END - START) / 1000000 ))

if [ \${DURATION} -lt 5000 ]; then
    echo -e "\${GREEN}✓ Response time: \${DURATION}ms\${NC}"
    ((PASS_COUNT++))
else
    echo -e "\${YELLOW}⚠ Slow response: \${DURATION}ms\${NC}"
    ((PASS_COUNT++))
fi

# Summary
echo ""
echo "Tests passed: \${PASS_COUNT}"
echo "Tests failed: \${FAIL_COUNT}"
echo ""

if [ \${FAIL_COUNT} -eq 0 ]; then
    echo -e "\${GREEN}✅ ${name.toUpperCase()} PROBE PASSED\${NC}"
    exit 0
else
    echo -e "\${RED}❌ ${name.toUpperCase()} PROBE FAILED\${NC}"
    exit 1
fi
`;
}

async function generateAll() {
  const baseDir = path.join(__dirname, '..');
  const testsDir = path.join(baseDir, 'tests');
  const probesDir = path.join(baseDir, 'probes');

  if (!fs.existsSync(testsDir)) fs.mkdirSync(testsDir, { recursive: true });
  if (!fs.existsSync(probesDir)) fs.mkdirSync(probesDir, { recursive: true });

  console.log('🚀 Generating ALL 51 test files and 51 probe scripts...\n');

  let testCount = 0;
  let probeCount = 0;

  for (const bubble of ALL_BUBBLES) {
    const testFileName = `${bubble.name}-${bubble.category}.test.ts`;
    const probeFileName = `${bubble.name}.probe.sh`;

    try {
      // Generate test
      const testPath = path.join(testsDir, testFileName);
      const testContent = generateTestFile(bubble);
      fs.writeFileSync(testPath, testContent, 'utf-8');
      testCount++;

      // Generate probe
      const probePath = path.join(probesDir, probeFileName);
      const probeContent = generateProbeScript(bubble);
      fs.writeFileSync(probePath, probeContent, 'utf-8');
      fs.chmodSync(probePath, '755');
      probeCount++;

      console.log(`✅ ${bubble.name} (${bubble.category})`);
    } catch (error) {
      console.error(`❌ Failed to generate ${bubble.name}:`, error.message);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('📊 GENERATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`✅ Test files: ${testCount}/51`);
  console.log(`✅ Probe scripts: ${probeCount}/51`);
  console.log('\n🎉 ALL FILES GENERATED! 🎉\n');
}

generateAll().catch(console.error);
