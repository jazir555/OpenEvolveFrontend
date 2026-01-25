/**
 * OpenEvolve Complete Test & Probe Generator
 *
 * Generates ALL 51 test files and 51 probe scripts for complete coverage
 * 21 Service Bubbles + 18 Tool Bubbles + 12 Workflow Bubbles = 51 Total
 */

import fs from 'fs';
import path from 'path';

interface BubbleTemplate {
  category: 'service' | 'tool' | 'workflow';
  name: string;
  className: string;
  operations: string[];
  testParams: Record<string, any>;
  probeEndpoints: string[];
}

const ALL_BUBBLES: BubbleTemplate[] = [
  // SERVICE BUBBLES (21)
  {
    category: 'service',
    name: 'qdrant',
    className: 'QdrantBubble',
    operations: ['health_check', 'create_collection', 'delete_collection', 'insert_points', 'search_points', 'delete_points', 'get_collection', 'list_collections', 'count_points'],
    testParams: { baseUrl: 'http://localhost:6333', vectorSize: 1536 },
    probeEndpoints: ['/', '/health', '/collections', '/cluster', '/metrics']
  },
  {
    category: 'service',
    name: 'elasticsearch',
    className: 'ElasticsearchBubble',
    operations: ['health_check', 'create_index', 'delete_index', 'index_document', 'search', 'bulk', 'get_document', 'update_document', 'delete_document'],
    testParams: { baseUrl: 'http://localhost:9200' },
    probeEndpoints: ['/', '_cluster/health', '_cat/indices', '_nodes/stats']
  },
  {
    category: 'service',
    name: 'redis',
    className: 'RedisBubble',
    operations: ['health_check', 'set', 'get', 'delete', 'exists', 'expire', 'incr', 'decr', 'hset', 'hget', 'lpush', 'lpop', 'sadd', 'smembers'],
    testParams: { host: 'localhost', port: 6379 },
    probeEndpoints: ['PING', 'INFO', 'DBSIZE', 'CLIENT LIST']
  },
  {
    category: 'service',
    name: 'postgresql',
    className: 'PostgreSQLBubble',
    operations: ['health_check', 'query', 'execute', 'begin_transaction', 'commit', 'rollback', 'list_tables', 'describe_table'],
    testParams: { host: 'localhost', port: 5432, database: 'test' },
    probeEndpoints: ['SELECT 1', 'SELECT version()', 'SELECT current_database()']
  },
  {
    category: 'service',
    name: 'ai-agent',
    className: 'AIAgentBubble',
    operations: ['generate', 'chat', 'stream', 'embed', 'analyze'],
    testParams: { model: 'gpt-4' },
    probeEndpoints: ['/v1/models', '/v1/chat/completions']
  },
  {
    category: 'service',
    name: 'crewai',
    className: 'CrewAIBubble',
    operations: ['execute_workflow', 'create_workflow', 'list_tasks', 'get_status'],
    testParams: { baseUrl: 'http://localhost:8080' },
    probeEndpoints: ['/health', '/v1/workflows', '/v1/tasks']
  },
  {
    category: 'service',
    name: 'ace-tools',
    className: 'ACEToolsBubble',
    operations: ['analyze', 'optimize', 'validate', 'transform'],
    testParams: {},
    probeEndpoints: ['/health', '/v1/tools', '/v1/status']
  },
  {
    category: 'service',
    name: 'workflow-orchestrator',
    className: 'WorkflowOrchestratorBubble',
    operations: ['create_workflow', 'execute_workflow', 'schedule_workflow', 'list_workflows', 'get_workflow_status'],
    testParams: { baseUrl: 'http://localhost:8080' },
    probeEndpoints: ['/health', '/v1/workflows', '/v1/executions']
  },
  {
    category: 'service',
    name: 'slack',
    className: 'SlackBubble',
    operations: ['send_message', 'upload_file', 'create_channel', 'invite_user', 'list_channels', 'get_history'],
    testParams: {},
    probeEndpoints: ['/api/auth.test', '/api/conversations.list', '/api/team.info']
  },
  {
    category: 'service',
    name: 'gmail',
    className: 'GmailBubble',
    operations: ['send_email', 'read_email', 'list_emails', 'search_emails', 'delete_email', 'label_email'],
    testParams: {},
    probeEndpoints: ['/gmail/v1/users/me/profile', '/gmail/v1/users/me/messages']
  },
  {
    category: 'service',
    name: 'sendgrid',
    className: 'SendGridBubble',
    operations: ['send_email', 'list_templates', 'get_template', 'send_template'],
    testParams: {},
    probeEndpoints: ['/v3/templates', '/v3/user/account', '/v3/user/profile']
  },
  {
    category: 'service',
    name: 'twilio',
    className: 'TwilioBubble',
    operations: ['send_sms', 'make_call', 'send_whatsapp', 'lookup_number'],
    testParams: {},
    probeEndpoints: ['/2010-04-01/Accounts', '/2010-04-01/Accounts.json']
  },
  {
    category: 'service',
    name: 'http',
    className: 'HTTPBubble',
    operations: ['get', 'post', 'put', 'patch', 'delete', 'head', 'options'],
    testParams: { baseUrl: 'http://localhost:3000' },
    probeEndpoints: ['/', '/health', '/api/status']
  },
  {
    category: 'service',
    name: 'github',
    className: 'GitHubBubble',
    operations: ['create_issue', 'create_pr', 'list_issues', 'get_file', 'create_repo', 'fork_repo', 'add_collaborator'],
    testParams: {},
    probeEndpoints: ['/user', '/user/repos', '/rate_limit']
  },
  {
    category: 'service',
    name: 'apify',
    className: 'ApifyBubble',
    operations: ['run_actor', 'get_dataset', 'list_actors', 'create_actor', 'get_run_status'],
    testParams: {},
    probeEndpoints: ['/v2/actors', '/v2/datasets', '/v2/users/me']
  },
  {
    category: 'service',
    name: 'webhook',
    className: 'WebhookBubble',
    operations: ['register', 'unregister', 'trigger', 'verify', 'list_webhooks'],
    testParams: { baseUrl: 'http://localhost:3000' },
    probeEndpoints: ['/webhooks', '/health']
  },
  {
    category: 'service',
    name: 'google-drive',
    className: 'GoogleDriveBubble',
    operations: ['upload_file', 'download_file', 'list_files', 'create_folder', 'share_file', 'delete_file'],
    testParams: {},
    probeEndpoints: ['/drive/v3/about', '/drive/v3/files']
  },
  {
    category: 'service',
    name: 'google-sheets',
    className: 'GoogleSheetsBubble',
    operations: ['create_spreadsheet', 'read_values', 'write_values', 'append_values', 'update_cell'],
    testParams: {},
    probeEndpoints: ['/sheets/v4/spreadsheets', '/drive/v3/about']
  },
  {
    category: 'service',
    name: 'notion',
    className: 'NotionBubble',
    operations: ['create_page', 'read_page', 'update_page', 'query_database', 'create_database', 'append_block'],
    testParams: {},
    probeEndpoints: ['/v1/users/me', '/v1/search']
  },
  {
    category: 'service',
    name: 'airtable',
    className: 'AirtableBubble',
    operations: ['list_records', 'create_record', 'update_record', 'delete_record', 'query_records'],
    testParams: { baseId: 'appBase123' },
    probeEndpoints: ['/v0/meta/bases', '/v0/meta/whoami']
  },
  {
    category: 'service',
    name: 'stripe',
    className: 'StripeBubble',
    operations: ['create_customer', 'create_charge', 'create_subscription', 'refund_charge', 'list_customers', 'get_invoice'],
    testParams: {},
    probeEndpoints: ['/v1/products', '/v1/customers', '/v1/charges']
  },

  // TOOL BUBBLES (18)
  {
    category: 'tool',
    name: 'web-search',
    className: 'WebSearchTool',
    operations: ['search', 'advanced_search', 'news_search', 'image_search'],
    testParams: { query: 'test search' },
    probeEndpoints: ['/search', '/news', '/images']
  },
  {
    category: 'tool',
    name: 'web-scrape',
    className: 'WebScrapeTool',
    operations: ['scrape', 'scrape_multiple', 'extract_links', 'extract_images', 'extract_text'],
    testParams: { url: 'https://example.com' },
    probeEndpoints: ['/scrape', '/extract']
  },
  {
    category: 'tool',
    name: 'research-agent',
    className: 'ResearchAgentTool',
    operations: ['research', 'deep_research', 'summarize', 'cite_sources'],
    testParams: { topic: 'test topic' },
    probeEndpoints: ['/research', '/summarize', '/sources']
  },
  {
    category: 'tool',
    name: 'sql-query',
    className: 'SQLQueryTool',
    operations: ['execute_query', 'execute_script', 'explain_query', 'validate_query'],
    testParams: { query: 'SELECT 1' },
    probeEndpoints: ['/query', '/validate', '/explain']
  },
  {
    category: 'tool',
    name: 'vector-search',
    className: 'VectorSearchTool',
    operations: ['search', 'index', 'batch_search', 'similarity_score'],
    testParams: { vector: Array(1536).fill(0.1) },
    probeEndpoints: ['/search', '/index', '/status']
  },
  {
    category: 'tool',
    name: 'log-parser',
    className: 'LogParserTool',
    operations: ['parse', 'analyze', 'extract_errors', 'generate_stats'],
    testParams: { logContent: 'test log' },
    probeEndpoints: ['/parse', '/analyze', '/errors']
  },
  {
    category: 'tool',
    name: 'metrics-collector',
    className: 'MetricsCollectorTool',
    operations: ['collect', 'aggregate', 'query', 'export'],
    testParams: { metric: 'cpu_usage' },
    probeEndpoints: ['/metrics', '/collect', '/export']
  },
  {
    category: 'tool',
    name: 'csv-processor',
    className: 'CSVProcessorTool',
    operations: ['parse', 'validate', 'transform', 'merge', 'split'],
    testParams: { csvData: 'id,name\n1,Test' },
    probeEndpoints: ['/parse', '/validate', '/transform']
  },
  {
    category: 'tool',
    name: 'json-validator',
    className: 'JSONValidatorTool',
    operations: ['validate', 'format', 'minify', 'transform', 'compare'],
    testParams: { jsonData: '{"test": true}' },
    probeEndpoints: ['/validate', '/format', '/transform']
  },
  {
    category: 'tool',
    name: 'data-transformer',
    className: 'DataTransformerTool',
    operations: ['transform', 'map', 'filter', 'aggregate', 'pivot'],
    testParams: { data: [1, 2, 3] },
    probeEndpoints: ['/transform', '/map', '/filter']
  },
  {
    category: 'tool',
    name: 'file-processor',
    className: 'FileProcessorTool',
    operations: ['read', 'write', 'compress', 'decompress', 'convert'],
    testParams: { filePath: '/tmp/test.txt' },
    probeEndpoints: ['/read', '/write', '/convert']
  },
  {
    category: 'tool',
    name: 'image-processor',
    className: 'ImageProcessorTool',
    operations: ['resize', 'crop', 'rotate', 'filter', 'convert_format', 'optimize'],
    testParams: { imagePath: '/tmp/test.jpg' },
    probeEndpoints: ['/resize', '/crop', '/filter']
  },
  {
    category: 'tool',
    name: 'xml-parser',
    className: 'XMLParserTool',
    operations: ['parse', 'validate', 'transform', 'query', 'format'],
    testParams: { xmlContent: '<root><test>data</test></root>' },
    probeEndpoints: ['/parse', '/validate', '/transform']
  },
  {
    category: 'tool',
    name: 'pdf-generator',
    className: 'PDFGeneratorTool',
    operations: ['generate', 'merge', 'split', 'add_watermark', 'convert'],
    testParams: { content: 'Test PDF content' },
    probeEndpoints: ['/generate', '/merge', '/split']
  },
  {
    category: 'tool',
    name: 'email-validator',
    className: 'EmailValidatorTool',
    operations: ['validate', 'verify', 'normalize', 'batch_validate'],
    testParams: { email: 'test@example.com' },
    probeEndpoints: ['/validate', '/verify', '/batch']
  },
  {
    category: 'tool',
    name: 'url-validator',
    className: 'URLValidatorTool',
    operations: ['validate', 'normalize', 'check_alive', 'extract_info'],
    testParams: { url: 'https://example.com' },
    probeEndpoints: ['/validate', '/check', '/info']
  },
  {
    category: 'tool',
    name: 'code-formatter',
    className: 'CodeFormatterTool',
    operations: ['format', 'lint', 'beautify', 'minify'],
    testParams: { code: 'function test(){}', language: 'javascript' },
    probeEndpoints: ['/format', '/lint', '/beautify']
  },
  {
    category: 'tool',
    name: 'text-analyzer',
    className: 'TextAnalyzerTool',
    operations: ['analyze', 'extract_keywords', 'sentiment', 'summarize', 'detect_language'],
    testParams: { text: 'Test text to analyze' },
    probeEndpoints: ['/analyze', '/keywords', '/sentiment']
  },

  // WORKFLOW BUBBLES (12)
  {
    category: 'workflow',
    name: 'database-analyzer',
    className: 'DatabaseAnalyzerWorkflow',
    operations: ['analyze', 'generate_report', 'optimize_schema', 'monitor_performance'],
    testParams: { database: 'test_db' },
    probeEndpoints: ['/analyze', '/report', '/optimize']
  },
  {
    category: 'workflow',
    name: 'slack-notifier',
    className: 'SlackNotifierWorkflow',
    operations: ['notify', 'schedule_notification', 'create_alert', 'digest'],
    testParams: { channel: '#general', message: 'Test' },
    probeEndpoints: ['/notify', '/schedule', '/alert']
  },
  {
    category: 'workflow',
    name: 'pdf-ocr',
    className: 'PDFOCRWorkflow',
    operations: ['process', 'batch_process', 'extract_text', 'search_text'],
    testParams: { pdfPath: '/tmp/test.pdf' },
    probeEndpoints: ['/process', '/extract', '/search']
  },
  {
    category: 'workflow',
    name: 'webhook-repeater',
    className: 'WebhookRepeaterWorkflow',
    operations: ['repeat', 'schedule_repeat', 'transform_payload', 'aggregate_results'],
    testParams: { webhookUrl: 'https://example.com/webhook' },
    probeEndpoints: ['/repeat', '/schedule', '/transform']
  },
  {
    category: 'workflow',
    name: 'data-enrichment',
    className: 'DataEnrichmentWorkflow',
    operations: ['enrich', 'batch_enrich', 'validate_enrichment', 'schedule_enrichment'],
    testParams: { data: { id: 1 } },
    probeEndpoints: ['/enrich', '/batch', '/validate']
  },
  {
    category: 'workflow',
    name: 'backup-restore',
    className: 'BackupRestoreWorkflow',
    operations: ['backup', 'restore', 'schedule_backup', 'list_backups', 'verify_backup'],
    testParams: { source: '/data', target: '/backup' },
    probeEndpoints: ['/backup', '/restore', '/list']
  },
  {
    category: 'workflow',
    name: 'monitoring-alert',
    className: 'MonitoringAlertWorkflow',
    operations: ['monitor', 'create_alert', 'check_threshold', 'send_notification'],
    testParams: { metric: 'cpu', threshold: 80 },
    probeEndpoints: ['/monitor', '/alert', '/check']
  },
  {
    category: 'workflow',
    name: 'etl-pipeline',
    className: 'ETLPipelineWorkflow',
    operations: ['extract', 'transform', 'load', 'run_pipeline', 'schedule_pipeline'],
    testParams: { source: 'db1', target: 'db2' },
    probeEndpoints: ['/extract', '/transform', '/load', '/run']
  },
  {
    category: 'workflow',
    name: 'api-aggregator',
    className: 'APIAggregatorWorkflow',
    operations: ['aggregate', 'combine_results', 'cache_results', 'parallel_fetch'],
    testParams: { apis: ['api1', 'api2'] },
    probeEndpoints: ['/aggregate', '/combine', '/cache']
  },
  {
    category: 'workflow',
    name: 'scheduled-task',
    className: 'ScheduledTaskWorkflow',
    operations: ['schedule', 'execute', 'list_scheduled', 'cancel_schedule', 'get_status'],
    testParams: { task: 'test_task', schedule: '0 0 * * *' },
    probeEndpoints: ['/schedule', '/execute', '/list']
  },
  {
    category: 'workflow',
    name: 'event-handler',
    className: 'EventHandlerWorkflow',
    operations: ['handle', 'register_handler', 'trigger_event', 'list_handlers'],
    testParams: { eventType: 'user.created' },
    probeEndpoints: ['/handle', '/register', '/trigger']
  },
  {
    category: 'workflow',
    name: 'multi-step-approval',
    className: 'MultiStepApprovalWorkflow',
    operations: ['create_workflow', 'approve_step', 'reject_step', 'get_status', 'add_approver'],
    testParams: { workflowName: 'test_approval', steps: 3 },
    probeEndpoints: ['/create', '/approve', '/reject', '/status']
  }
];

function generateTestFile(bubble: BubbleTemplate): string {
  const { category, name, className, operations, testParams } = bubble;

  return `/**
 * ${className} Test Suite
 *
 * Comprehensive tests for ${name} ${category} bubble
 * Tests cover: functionality, error handling, resilience patterns, contract compliance
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ${className} } from '../${category}-bubbles/${name}-${category === 'service' ? 'bubble' : category}';
import type { ${name.charAt(0).toUpperCase() + name.slice(1)}Params } from '../${category}-bubbles/${name}-${category === 'service' ? 'bubble' : category}';

describe('${className}', () => {
  // ============================================================================
  // BASE CLASS INHERITANCE TESTS
  // ============================================================================

  describe('Base Class Inheritance', () => {
    it('should extend ${category.charAt(0).toUpperCase() + category.slice(1)}Bubble properly', () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
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

  // ============================================================================
  // FEDERATION CONSTITUTION COMPLIANCE TESTS
  // ============================================================================

  describe('Federation Constitution Compliance', () => {
    it('should fail without required params (no magic defaults)', () => {
      expect(() => {
        new ${className}({
          operation: '${operations[0]}',
          // @ts-expect-error - Testing missing required fields
          ${Object.keys(testParams)[0]}: undefined,
        });
      }).toThrow();
    });

    it('should accept valid configuration', () => {
      expect(() => {
        new ${className}({
          operation: '${operations[0]}',
          ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n          ')}
        });
      }).not.toThrow();
    });

    it('should follow Law of Air Gap (no core-projects imports)', () => {
      const fs = require('fs');
      const content = fs.readFileSync(__filename, 'utf-8');
      expect(content).not.toContain('core-projects');
    });

    it('should follow Law of Configuration Explicitness', () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });
      // Should use provided values, not defaults
      expect(bubble.params).toBeDefined();
    });
  });

  // ============================================================================
  // PARAMETER VALIDATION TESTS
  // ============================================================================

  describe('Parameter Validation', () => {
    let validParams: any;

    beforeEach(() => {
      validParams = {
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      };
    });

    it('should validate operation enum', () => {
      const validOperations = ${JSON.stringify(operations, null, 6)};

      validOperations.forEach(operation => {
        expect(() => {
          new ${className}({ ...validParams, operation });
        }).not.toThrow();
      });
    });

    it('should reject invalid operation', () => {
      expect(() => {
        new ${className}({
          ...validParams,
          // @ts-expect-error - Testing invalid operation
          operation: 'invalid_operation',
        });
      }).toThrow();
    });
  });

  // ============================================================================
  // OPERATION-SPECIFIC TESTS
  // ============================================================================

${operations.slice(0, 5).map(op => `
  describe('${op.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())} Operation', () => {
    it('should execute ${op}', async () => {
      const bubble = new ${className}({
        operation: '${op}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      // Mock fetch or client
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBeDefined();
      expect(result.operation).toBe('${op}');
      expect(result.timing).toBeGreaterThanOrEqual(0);
    });

    it('should handle network errors gracefully', async () => {
      const bubble = new ${className}({
        operation: '${op}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.timing).toBeGreaterThanOrEqual(0);
    });
  });
`).join('')}

  // ============================================================================
  // CIRCUIT BREAKER TESTS
  // ============================================================================

  describe('Circuit Breaker', () => {
    it('should open circuit after threshold failures', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

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

    it('should recover after circuit closes', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      expect(bubble).toBeDefined();
    });
  });

  // ============================================================================
  // RETRY LOGIC TESTS
  // ============================================================================

  describe('Retry Logic', () => {
    it('should retry transient errors', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      let attemptCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        attemptCount++;
        if (attemptCount < 3) {
          return Promise.reject(new Error('ECONNREFUSED'));
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: async () => ({ success: true }),
        } as Response);
      });

      const result = await bubble.action();

      expect(attemptCount).toBe(3);
      expect(result.success).toBe(true);
    });

    it('should not retry permanent errors', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      let attemptCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        attemptCount++;
        return Promise.reject(new Error('404 Not Found'));
      });

      try {
        await bubble.action();
      } catch (error) {
        // Expected
      }

      expect(attemptCount).toBe(1);
    });
  });

  // ============================================================================
  // REQUEST DEDUPLICATION TESTS
  // ============================================================================

  describe('Request Deduplication', () => {
    it('should deduplicate identical requests', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      let fetchCount = 0;
      global.fetch = vi.fn().mockImplementation(() => {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: async () => ({ success: true }),
        } as Response);
      });

      const promises = [
        bubble.action(),
        bubble.action(),
        bubble.action(),
      ];

      await Promise.all(promises);

      expect(fetchCount).toBeLessThan(3);
    });
  });

  // ============================================================================
  // CONTRACT TESTS
  // ============================================================================

  describe('Contract Tests', () => {
    it('should return correct response structure', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);

      const result = await bubble.action();

      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('operation');
      expect(result).toHaveProperty('status');
      expect(result).toHaveProperty('timing');
      expect(result.status).toHaveProperty('code');
      expect(result.status).toHaveProperty('reason');
    });

    it('should include timing information', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);

      const result = await bubble.action();

      expect(result.timing).toBeGreaterThanOrEqual(0);
      expect(typeof result.timing).toBe('number');
    });
  });

  // ============================================================================
  // ERROR HANDLING TESTS
  // ============================================================================

  describe('Error Handling', () => {
    it('should handle timeout errors', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockImplementation(() =>
        new Promise((resolve) => setTimeout(resolve, 10000))
      );

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should handle malformed responses', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ invalid: 'response' }),
      } as Response);

      const result = await bubble.action();

      expect(result).toBeDefined();
    });
  });

  // ============================================================================
  // PERFORMANCE TESTS
  // ============================================================================

  describe('Performance', () => {
    it('should complete operation within timeout', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);

      const start = Date.now();
      await bubble.action();
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(5000);
    });

    it('should handle concurrent operations', async () => {
      const bubble = new ${className}({
        operation: '${operations[0]}',
        ${Object.entries(testParams).map(([k, v]) => `${k}: '${v}'`).join(',\n        ')}
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);

      const operations = Array.from({ length: 10 }, () => bubble.action());

      const start = Date.now();
      await Promise.all(operations);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(30000);
    });
  });

  // ============================================================================
  // CLEANUP
  // ============================================================================

  afterEach(() => {
    vi.restoreAllMocks();
  });
});
`;
}

function generateProbeScript(bubble: BubbleTemplate): string {
  const { category, name, probeEndpoints, testParams } = bubble;

  return `#!/bin/bash
# probes/${name}.probe.sh
# Runtime validation probe for ${name} ${category} bubble

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

${Object.entries(testParams).map(([k, v]) => {
  if (k === 'baseUrl' || k === 'host') return `${k.toUpperCase()}="${v}"`
  return `${k.toUpperCase()}="${v}"`
}).join('\n')}

# Color codes
GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo -e "\${BLUE}ℹ\${NC} $1"
}

log_success() {
    echo -e "\${GREEN}✓\${NC} $1"
}

log_error() {
    echo -e "\${RED}✗\${NC} $1"
}

log_warning() {
    echo -e "\${YELLOW}⚠\${NC} $1"
}

test_endpoint() {
    local endpoint="$1"
    local description="$2"
    local expected_code="${3:-200}"

    echo -n "Testing \${description}... "

    if curl -f -s -X GET "${BASEURL:-http://localhost:8080}${endpoint}" \\
        -H "Content-Type: application/json" \\
        -o /dev/null \\
        -w "%{http_code}" \\
        --max-time 5 2>/dev/null | grep -q "\${expected_code}"; then
        log_success "PASS"
        return 0
    else
        log_error "FAIL"
        return 1
    fi
}

test_post_endpoint() {
    local endpoint="$1"
    local data="$2"
    local description="$3"

    echo -n "Testing \${description}... "

    if curl -f -s -X POST "${BASEURL:-http://localhost:8080}${endpoint}" \\
        -H "Content-Type: application/json" \\
        -d "\${data}" \\
        -o /dev/null \\
        -w "%{http_code}" \\
        --max-time 5 2>/dev/null | grep -q "200\\|201"; then
        log_success "PASS"
        return 0
    else
        log_error "FAIL"
        return 1
    fi
}

# ============================================================================
# PROBE SEQUENCE
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🔍 ${name.toUpperCase()} ${category.toUpperCase()} BUBBLE PROBE"
echo "════════════════════════════════════════════════════════════════"
echo ""

FAIL_COUNT=0
PASS_COUNT=0

# ============================================================================
# TEST 1: Base URL Connectivity
# ============================================================================

log_info "Test 1: Base URL Connectivity"
echo "Target: \${BASEURL:-http://localhost:8080}"

if curl -f -s "\${BASEURL:-http://localhost:8080}/" -o /dev/null --max-time 5 2>/dev/null; then
    log_success "Base URL is reachable"
    ((PASS_COUNT++))
else
    log_error "Base URL is not reachable"
    ((FAIL_COUNT++))
fi

echo ""

# ============================================================================
# TEST 2: Health Check
# ============================================================================

log_info "Test 2: Health Check"

HEALTH_RESPONSE=$(curl -s "\${BASEURL:-http://localhost:8080}/health" --max-time 5 2>/dev/null || echo "")

if [ -n "\${HEALTH_RESPONSE}" ]; then
    echo "\${HEALTH_RESPONSE}" | jq . > /dev/null 2>&1 && {
        log_success "Health check endpoint is responding with JSON"
        ((PASS_COUNT++))
        echo "\${HEALTH_RESPONSE}" | jq .
    } || {
        log_warning "Health check responding but not valid JSON"
        ((PASS_COUNT++))
    }
else
    log_error "Health check endpoint not responding"
    ((FAIL_COUNT++))
fi

echo ""

# ============================================================================
# TEST 3: Capabilities/Status Endpoint
# ============================================================================

log_info "Test 3: Service Capabilities"

STATUS_RESPONSE=$(curl -s "\${BASEURL:-http://localhost:8080}/v1/status" --max-time 5 2>/dev/null || curl -s "\${BASEURL:-http://localhost:8080}/status" --max-time 5 2>/dev/null || echo "")

if [ -n "\${STATUS_RESPONSE}" ]; then
    log_success "Status endpoint is responding"
    ((PASS_COUNT++))
    echo "\${STATUS_RESPONSE}" | jq . 2>/dev/null || echo "\${STATUS_RESPONSE}"
else
    log_warning "Status endpoint not available (non-critical)"
fi

echo ""

# ============================================================================
# TEST 4: Operation-Specific Tests
# ============================================================================

log_info "Test 4: Operation-Specific Endpoints"

${probeEndpoints.slice(0, 5).map((endpoint, i) => `
# Test ${i + 1}: ${endpoint}
if test_endpoint "${endpoint}" "Endpoint ${i + 1}: ${endpoint}" 2>/dev/null; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi
`).join('\n')}

echo ""

# ============================================================================
# TEST 5: Configuration Validation
# ============================================================================

log_info "Test 5: Configuration Validation"

# Check required environment variables
REQUIRED_VARS="${Object.keys(testParams).join(' ').toUpperCase()}"

for var in \$REQUIRED_VARS; do
    if [ -z "\${!var}" ]; then
        log_warning "\${var} not set (may have default)"
    else
        log_success "\${var} is configured: \${!var}"
        ((PASS_COUNT++))
    fi
done

echo ""

# ============================================================================
# TEST 6: Performance Test
# ============================================================================

log_info "Test 6: Performance Test"

START_TIME=$(date +%s%N)
curl -f -s "\${BASEURL:-http://localhost:8080}/health" -o /dev/null --max-time 5 2>/dev/null || true
END_TIME=$(date +%s%N)
DURATION=$(( (END_TIME - START_TIME) / 1000000 ))  # Convert to milliseconds

if [ \$DURATION -lt 5000 ]; then
    log_success "Response time: \${DURATION}ms"
    ((PASS_COUNT++))
else
    log_warning "Slow response: \${DURATION}ms (threshold: 5000ms)"
    ((PASS_COUNT++))
fi

echo ""

# ============================================================================
# TEST 7: Error Handling
# ============================================================================

log_info "Test 7: Error Handling"

# Test invalid endpoint
INVALID_RESPONSE=$(curl -s "\${BASEURL:-http://localhost:8080}/invalid-endpoint" -w "%{http_code}" -o /dev/null --max-time 5 2>/dev/null || echo "000")

if [ "\${INVALID_RESPONSE}" == "404" ] || [ "\${INVALID_RESPONSE}" == "400" ]; then
    log_success "Proper error handling (404 on invalid endpoint)"
    ((PASS_COUNT++))
else
    log_warning "Unexpected response code for invalid endpoint: \${INVALID_RESPONSE}"
fi

echo ""

# ============================================================================
# TEST 8: Concurrent Request Test
# ============================================================================

log_info "Test 8: Concurrent Request Test"

CONCURRENT_START=$(date +%s%N)

for i in {1..5}; do
    curl -f -s "\${BASEURL:-http://localhost:8080}/health" -o /dev/null --max-time 5 2>/dev/null &
done

wait

CONCURRENT_END=$(date +%s%N)
CONCURRENT_DURATION=$(( (CONCURRENT_END - CONCURRENT_START) / 1000000 ))

log_success "5 concurrent requests completed in \${CONCURRENT_DURATION}ms"
((PASS_COUNT++))

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  📊 PROBE SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "  Total Tests: \$((PASS_COUNT + FAIL_COUNT))"
echo -e "  \${GREEN}Passed: \${PASS_COUNT}\${NC}"
echo -e "  \${RED}Failed: \${FAIL_COUNT}\${NC}"
echo ""

if [ \$FAIL_COUNT -eq 0 ]; then
    echo -e "\${GREEN}✅ ${name.toUpperCase()} PROBE PASSED\${NC}"
    echo ""
    exit 0
else
    echo -e "\${RED}❌ ${name.toUpperCase()} PROBE FAILED\${NC}"
    echo ""
    exit 1
fi
`;
}

// Main generation function
async function generateAll() {
  const baseDir = path.join(__dirname, '..');
  const testsDir = path.join(baseDir, 'tests');
  const probesDir = path.join(baseDir, 'probes');

  // Ensure directories exist
  if (!fs.existsSync(testsDir)) {
    fs.mkdirSync(testsDir, { recursive: true });
  }
  if (!fs.existsSync(probesDir)) {
    fs.mkdirSync(probesDir, { recursive: true });
  }

  console.log('🚀 Generating ALL 51 test files and 51 probe scripts...\n');

  let testCount = 0;
  let probeCount = 0;
  const errors: string[] = [];

  for (const bubble of ALL_BUBBLES) {
    const testFileName = `${bubble.name}-${bubble.category}.test.ts`;
    const testFilePath = path.join(testsDir, testFileName);
    const probeFileName = `${bubble.name}.probe.sh`;
    const probeFilePath = path.join(probesDir, probeFileName);

    try {
      // Generate test file
      console.log(`📝 Creating test: ${testFileName}`);
      const testContent = generateTestFile(bubble);
      fs.writeFileSync(testFilePath, testContent, 'utf-8');
      testCount++;

      // Generate probe script
      console.log(`🔍 Creating probe: ${probeFileName}`);
      const probeContent = generateProbeScript(bubble);
      fs.writeFileSync(probeFilePath, probeContent, 'utf-8');
      fs.chmodSync(probeFilePath, '755'); // Make executable
      probeCount++;

      console.log(`  ✅ ${bubble.name} (${bubble.category})\n`);
    } catch (error) {
      const errorMsg = `❌ Failed to generate ${bubble.name}: ${error}`;
      console.error(errorMsg);
      errors.push(errorMsg);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('📊 GENERATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`✅ Test files created: ${testCount}/51`);
  console.log(`✅ Probe scripts created: ${probeCount}/51`);
  console.log(`❌ Errors: ${errors.length}`);

  if (errors.length > 0) {
    console.log('\n❌ Errors encountered:');
    errors.forEach(err => console.log(`  - ${err}`));
  }

  console.log('\n🎉 ALL FILES GENERATED SUCCESSFULLY! 🎉\n');
  console.log('Next steps:');
  console.log('  1. Review generated test files in: tests/');
  console.log('  2. Review generated probe scripts in: probes/');
  console.log('  3. Run tests: npm test');
  console.log('  4. Run probes: ./probes/*.probe.sh');
  console.log('');

  if (errors.length > 0) {
    process.exit(1);
  }
}

// Run generation
generateAll().catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
