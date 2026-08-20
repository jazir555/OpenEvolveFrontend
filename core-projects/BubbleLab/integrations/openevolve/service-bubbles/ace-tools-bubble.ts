/**
 * ACE MCP Tools Service Bubble
 *
 * Integrates with OpenEvolve ACE (Adversarial Context Engine) MCP tools
 * for advanced analytics, verification, and security operations.
 */

import { z } from 'zod';
import { HttpBubble, AIAgentBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { checkOpenEvolveHealth } from './openevolve-health';

const ACEOperationSchema = z.enum([
  'analytics',
  'verification',
  'security_scan',
  'edge_case_analysis',
  'red_team_test',
  'blue_team_defense',
  'knowledge_extraction',
  'workflow_integration',
  'health_check',
  'benchmark',
  'metrics',
]);

const ACEToolsParamsSchema = z.object({
  operation: ACEOperationSchema.describe('ACE operation'),
  baseUrl: z.string().url().default('http://localhost:8000').describe('ACE API URL'),
  timeout: z.number().min(1000).max(300000).default(60000),

  // Analytics
  metricType: z.string().optional().describe('Type of metric to analyze'),
  timeRange: z.string().optional().describe('Time range for analytics'),

  // Verification
  componentId: z.string().optional().describe('Component ID to verify'),
  verificationLevel: z.enum(['basic', 'thorough', 'exhaustive']).default('thorough'),

  // Security
  scanDepth: z.enum(['quick', 'standard', 'deep']).default('standard'),
  vulnerabilityTypes: z.array(z.string()).optional().describe('Vulnerability types to scan'),

  // Edge cases
  functionId: z.string().optional().describe('Function ID for edge case analysis'),
  parameterSpace: z.record(z.unknown()).optional().describe('Parameter space to explore'),

  // Red/Blue team
  attackVector: z.string().optional().describe('Attack vector'),
  defenseStrategy: z.string().optional().describe('Defense strategy'),
  testDuration: z.number().optional().describe('Test duration in seconds'),

  // Knowledge extraction
  workflowId: z.string().optional().describe('Workflow ID'),
  extractionDepth: z.number().min(1).max(10).default(3),

  // General
  parameters: z.record(z.unknown()).optional(),
  outputFormat: z.enum(['json', 'yaml', 'xml', 'text']).default('json'),
});

type ACEToolsParamsInput = z.input<typeof ACEToolsParamsSchema>;
type ACEToolsParams = z.output<typeof ACEToolsParamsSchema>;

const ACEToolsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  analytics: z.record(z.unknown()).optional(),
  verification: z.object({
    passed: z.boolean(),
    issues: z.array(z.unknown()),
    coverage: z.number(),
  }).optional(),
  security: z.object({
    vulnerabilities: z.array(z.unknown()),
    score: z.number(),
    recommendations: z.array(z.string()),
  }).optional(),
  edgeCases: z.array(z.unknown()).optional(),
  knowledgeArtifacts: z.array(z.unknown()).optional(),
  metrics: z.record(z.number()).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type ACEToolsResult = z.output<typeof ACEToolsResultSchema>;

export class ACEToolsBubble {
  private http: HttpBubble;
  private aiAgent: AIAgentBubble;
  private params: ACEToolsParams;
  private context?: BubbleContext;

  constructor(params: ACEToolsParamsInput, context?: BubbleContext) {
    this.params = ACEToolsParamsSchema.parse(params);
    this.context = context;

    this.http = new HttpBubble({
      url: this.params.baseUrl,
      method: 'GET',
      timeout: this.params.timeout,
    }, context);

    this.aiAgent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'You are an ACE (Adversarial Context Engine) analyzer for OpenEvolve.',
    }, context);
  }

  private async request(endpoint: string, body?: unknown): Promise<ACEToolsResult> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${this.params.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: body ? JSON.stringify(body) : undefined,
      });

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: this.params.operation,
        data,
        analytics: data.analytics,
        verification: data.verification,
        security: data.security,
        edgeCases: data.edge_cases,
        knowledgeArtifacts: data.knowledge_artifacts,
        metrics: data.metrics,
        error: response.ok ? undefined : data.error,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: this.params.operation,
        error: errorMessage,
        timing,
      };
    }
  }

  public async analytics(): Promise<ACEToolsResult> {
    return this.request('/api/ace/analytics', {
      metric_type: this.params.metricType,
      time_range: this.params.timeRange,
      parameters: this.params.parameters,
    });
  }

  public async verification(): Promise<ACEToolsResult> {
    if (!this.params.componentId) {
      throw new Error('componentId is required for verification operation');
    }

    return this.request('/api/ace/verification', {
      component_id: this.params.componentId,
      level: this.params.verificationLevel,
      parameters: this.params.parameters,
    });
  }

  public async securityScan(): Promise<ACEToolsResult> {
    return this.request('/api/ace/security', {
      depth: this.params.scanDepth,
      vulnerability_types: this.params.vulnerabilityTypes,
      parameters: this.params.parameters,
    });
  }

  public async edgeCaseAnalysis(): Promise<ACEToolsResult> {
    if (!this.params.functionId) {
      throw new Error('functionId is required for edge_case_analysis operation');
    }

    return this.request('/api/ace/edge-cases', {
      function_id: this.params.functionId,
      parameter_space: this.params.parameterSpace,
      parameters: this.params.parameters,
    });
  }

  public async redTeamTest(): Promise<ACEToolsResult> {
    return this.request('/api/ace/red-team', {
      attack_vector: this.params.attackVector,
      duration: this.params.testDuration,
      parameters: this.params.parameters,
    });
  }

  public async blueTeamDefense(): Promise<ACEToolsResult> {
    return this.request('/api/ace/blue-team', {
      defense_strategy: this.params.defenseStrategy,
      duration: this.params.testDuration,
      parameters: this.params.parameters,
    });
  }

  public async knowledgeExtraction(): Promise<ACEToolsResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for knowledge_extraction operation');
    }

    return this.request('/api/ace/knowledge-extract', {
      workflow_id: this.params.workflowId,
      depth: this.params.extractionDepth,
      parameters: this.params.parameters,
    });
  }

  public async workflowIntegration(): Promise<ACEToolsResult> {
    return this.request('/api/ace/workflow-integrate', {
      parameters: this.params.parameters,
    });
  }

  public async benchmark(): Promise<ACEToolsResult> {
    return this.request('/api/ace/benchmark', {
      parameters: this.params.parameters,
    });
  }

  public async metrics(): Promise<ACEToolsResult> {
    return this.request('/api/ace/metrics', {
      parameters: this.params.parameters,
    });
  }

  public async healthCheck(): Promise<ACEToolsResult> {
    const startTime = Date.now();
    const health = await checkOpenEvolveHealth();
    const timing = Date.now() - startTime;
    return {
      success: health.ok,
      operation: 'health_check',
      data: health.data,
      error: health.error,
      timing,
    };
  }

  public async action(): Promise<ACEToolsResult> {
    switch (this.params.operation) {
      case 'analytics':
        return this.analytics();
      case 'verification':
        return this.verification();
      case 'security_scan':
        return this.securityScan();
      case 'edge_case_analysis':
        return this.edgeCaseAnalysis();
      case 'red_team_test':
        return this.redTeamTest();
      case 'blue_team_defense':
        return this.blueTeamDefense();
      case 'knowledge_extraction':
        return this.knowledgeExtraction();
      case 'workflow_integration':
        return this.workflowIntegration();
      case 'benchmark':
        return this.benchmark();
      case 'metrics':
        return this.metrics();
      case 'health_check':
        return this.healthCheck();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default ACEToolsBubble;
