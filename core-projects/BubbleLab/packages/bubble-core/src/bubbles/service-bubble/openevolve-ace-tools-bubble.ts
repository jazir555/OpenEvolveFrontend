import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

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

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const ACEToolsParamsSchema = z.object({
  operation: ACEOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  metric_type: z.string().optional(),
  time_range: z.string().optional(),

  component_id: z.string().optional(),
  verification_level: z.enum(['basic', 'thorough', 'exhaustive']).default('thorough'),

  scan_depth: z.enum(['quick', 'standard', 'deep']).default('standard'),
  vulnerability_types: z.array(z.string()).optional(),

  function_id: z.string().optional(),
  parameter_space: z.record(z.unknown()).optional(),

  attack_vector: z.string().optional(),
  defense_strategy: z.string().optional(),
  test_duration: z.number().optional(),

  workflow_id: z.string().optional(),
  extraction_depth: z.number().min(1).max(10).default(3),

  parameters: z.record(z.unknown()).optional(),
  output_format: z.enum(['json', 'yaml', 'xml', 'text']).default('json'),
});

type ACEToolsParams = z.input<typeof ACEToolsParamsSchema> & ServiceBubbleParams;

const ACEToolsResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type ACEToolsResult = z.output<typeof ACEToolsResultSchema> & BubbleOperationResult;

export class OpenEvolveAceToolsBubble extends ServiceBubble<
  ACEToolsParams,
  ACEToolsResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-ace-tools' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = ACEToolsParamsSchema;
  static readonly resultSchema = ACEToolsResultSchema;
  static readonly shortDescription =
    'OpenEvolve ACE tools for analytics, verification, and security checks';
  static readonly longDescription = `
    ACE tools bubble for advanced analytics and verification workflows.
  `;
  static readonly alias = 'openevolve-ace-tools';

  constructor(params: ACEToolsParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<ACEToolsResult> {
    const startTime = Date.now();
    try {
      switch (((this.params.operation as string) as string)) {
        case 'analytics':
          return await this.request('/api/ace/analytics', {
            metric_type: this.params.metric_type,
            time_range: this.params.time_range,
            parameters: this.params.parameters,
          }, startTime);
        case 'verification':
          return await this.request('/api/ace/verification', {
            component_id: this.requireParam('component_id'),
            level: this.params.verification_level,
            parameters: this.params.parameters,
          }, startTime);
        case 'security_scan':
          return await this.request('/api/ace/security', {
            depth: this.params.scan_depth,
            vulnerability_types: this.params.vulnerability_types,
            parameters: this.params.parameters,
          }, startTime);
        case 'edge_case_analysis':
          return await this.request('/api/ace/edge-cases', {
            function_id: this.requireParam('function_id'),
            parameter_space: this.params.parameter_space,
            parameters: this.params.parameters,
          }, startTime);
        case 'red_team_test':
          return await this.request('/api/ace/red-team', {
            attack_vector: this.params.attack_vector,
            duration: this.params.test_duration,
            parameters: this.params.parameters,
          }, startTime);
        case 'blue_team_defense':
          return await this.request('/api/ace/blue-team', {
            defense_strategy: this.params.defense_strategy,
            duration: this.params.test_duration,
            parameters: this.params.parameters,
          }, startTime);
        case 'knowledge_extraction':
          return await this.request('/api/ace/knowledge-extract', {
            workflow_id: this.requireParam('workflow_id'),
            depth: this.params.extraction_depth,
            parameters: this.params.parameters,
          }, startTime);
        case 'workflow_integration':
          return await this.request('/api/ace/workflow-integrate', {
            parameters: this.params.parameters,
          }, startTime);
        case 'benchmark':
          return await this.request('/api/ace/benchmark', {
            parameters: this.params.parameters,
          }, startTime);
        case 'metrics':
          return await this.request('/api/ace/metrics', {
            parameters: this.params.parameters,
          }, startTime);
        case 'health_check':
          return await this.request('/api/ace/health', undefined, startTime, 'GET');
        default:
          return {
            success: false,
            operation: ((this.params.operation as string) as string),
            error: `Unsupported operation: ${((this.params.operation as string) as string)}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireParam(key: 'component_id' | 'function_id' | 'workflow_id'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${((this.params.operation as string) as string)}`);
    }
    return value;
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.headers) {
      Object.assign(headers, this.params.headers);
    }
    if (this.params.auth_token) {
      const headerName = this.params.auth_header || 'Authorization';
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' &&
        !this.params.auth_token.startsWith('Bearer ')
          ? `Bearer ${this.params.auth_token}`
          : this.params.auth_token;
    }
    return headers;
  }

  private async request(
    endpoint: string,
    body: unknown,
    startTime: number,
    method: 'GET' | 'POST' = 'POST'
  ): Promise<ACEToolsResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveAceToolsBubble;
