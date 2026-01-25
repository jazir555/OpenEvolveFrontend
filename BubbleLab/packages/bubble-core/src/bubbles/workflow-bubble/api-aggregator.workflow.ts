import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';

const APIAggregatorParamsSchema = z.object({
  apis: z.array(z.object({
    name: z.string(),
    url: z.string().url(),
    method: z.enum(['GET', 'POST', 'PUT', 'PATCH']).default('GET'),
    headers: z.record(z.string()).optional(),
    body: z.unknown().optional(),
    timeout: z.number().int().positive().default(30000),
  })),
  aggregationStrategy: z.enum(['parallel', 'sequential', 'batch']).default('parallel'),
  mergeStrategy: z.enum(['concat', 'merge', 'zip']).default('merge'),
  errorHandling: z.enum(['fail', 'continue', 'partial']).default('continue'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

type APIAggregatorParams = z.input<typeof APIAggregatorParamsSchema>;

const APIAggregatorResultSchema = z.object({
  success: z.boolean(),
  error: z.string(),
  results: z.array(z.object({
    api: z.string(),
    success: z.boolean(),
    data: z.unknown().optional(),
    error: z.string().optional(),
    responseTime: z.number(),
  })),
  mergedData: z.unknown().optional(),
  totalResponseTime: z.number(),
});

export class APIAggregatorWorkflow extends WorkflowBubble<APIAggregatorParams, z.infer<typeof APIAggregatorResultSchema>> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName: BubbleName = 'api-aggregator-workflow';
  static readonly schema = APIAggregatorParamsSchema;
  static readonly resultSchema = APIAggregatorResultSchema;
  static readonly shortDescription = 'Aggregate multiple API calls into unified response';
  static readonly longDescription = 'Calls multiple APIs in parallel or sequence and merges results into unified response.';
  static readonly alias = 'aggregate-apis';

  constructor(params: APIAggregatorParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction() {
    const startTime = Date.now();
    console.log('[APIAggregator] Aggregating API calls');

    try {
      const results: z.infer<typeof APIAggregatorResultSchema>['results'] = [];

      if (this.params.aggregationStrategy === 'parallel') {
        const promises = this.params.apis.map(api => this.callAPI({
          name: api.name,
          url: api.url,
          method: api.method || 'GET',
          timeout: api.timeout || 30000,
          headers: api.headers,
          body: api.body,
        }));
        const apiResults = await Promise.all(promises);
        results.push(...apiResults);
      } else {
        for (const api of this.params.apis) {
          const result = await this.callAPI({
            name: api.name,
            url: api.url,
            method: api.method || 'GET',
            timeout: api.timeout || 30000,
            headers: api.headers,
            body: api.body,
          });
          results.push(result);

          if (!result.success && this.params.errorHandling === 'fail') {
            break;
          }
        }
      }

      const mergedData = this.mergeResults(results);
      const totalResponseTime = Date.now() - startTime;

      return {
        success: true,
        error: '',
        results,
        mergedData,
        totalResponseTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        results: [],
        totalResponseTime: Date.now() - startTime,
      };
    }
  }

  private async callAPI(apiConfig: {
    name: string;
    url: string;
    method: 'GET' | 'POST' | 'PUT' | 'PATCH';
    timeout: number;
    headers?: Record<string, string>;
    body?: unknown;
  }) {
    const startTime = Date.now();

    try {
      const httpBubble = new HttpBubble({
        url: apiConfig.url,
        method: apiConfig.method,
        headers: apiConfig.headers,
        body: apiConfig.body as string | Record<string, unknown> | undefined,
        timeout: apiConfig.timeout,
        credentials: this.params.credentials,
      }, this.context);

      const result = await httpBubble.action();

      return {
        api: apiConfig.name,
        success: result.success,
        data: result.data.json as string | Record<string, unknown> | undefined,
        error: result.error,
        responseTime: Date.now() - startTime,
      };
    } catch (error) {
      return {
        api: apiConfig.name,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        responseTime: Date.now() - startTime,
      };
    }
  }

  private mergeResults(results: z.infer<typeof APIAggregatorResultSchema>['results']) {
    if (this.params.mergeStrategy === 'concat') {
      return results.flatMap(r => r.success && r.data ? [r.data] : []);
    } else if (this.params.mergeStrategy === 'merge') {
      const merged: Record<string, unknown> = {};
      for (const r of results) {
        if (r.success && r.data && typeof r.data === 'object') {
          Object.assign(merged, r.data);
        }
      }
      return merged;
    }
    return results;
  }
}
