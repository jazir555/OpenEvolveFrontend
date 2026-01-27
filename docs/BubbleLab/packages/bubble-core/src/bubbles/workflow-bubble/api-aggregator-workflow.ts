import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * APIAggregatorWorkflow - Multiple API aggregation and dispatch
 */
export class APIAggregatorWorkflow extends WorkflowBubble<APIAggregatorParams, APIAggregatorResult> {
  bubbleName = 'api-aggregator';
  type = 'workflow';
  alias = 'api-aggregator';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<APIAggregatorResult> {
    const steps = [];

    try {
      // Step 1: Aggregate
      const step1Result = await this.aggregate(input);
      steps.push({
        step: 1,
        name: 'aggregate',
        status: 'completed',
        result: step1Result
      });

      // Step 2: Merge
      const step2Result = await this.merge({ ...input, responses: step1Result });
      steps.push({
        step: 2,
        name: 'merge',
        status: 'completed',
        result: step2Result
      });

      // Step 3: Dispatch
      const step3Result = await this.dispatch({ ...input, merged: step2Result });
      steps.push({
        step: 3,
        name: 'dispatch',
        status: 'completed',
        result: step3Result
      });

      return { success: true, steps };
    } catch (error: any) {
      return { success: false, error: error.message, steps };
    }
  }

  async aggregate(params: { apis: string[] }): Promise<APIAggregatorResult> {
    try {
      const responses = await Promise.all(
        params.apis.map(api =>
          Promise.resolve({
            api,
            status: 200,
            data: { value: Math.random() * 100 },
            latency: Math.floor(Math.random() * 1000)
          })
        )
      );
      return { success: true, responses };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async merge(params: { responses: any[] }): Promise<APIAggregatorResult> {
    try {
      const merged = {
        totalCount: params.responses.length,
        successful: params.responses.filter(r => r.status === 200).length,
        data: params.responses.map(r => r.data),
        averageLatency: params.responses.reduce((sum, r) => sum + r.latency, 0) / params.responses.length
      };
      return { success: true, merged };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async dispatch(params: { merged: any; target: string }): Promise<APIAggregatorResult> {
    try {
      const dispatched = {
        target: params.target,
        payload: params.merged,
        dispatchedAt: new Date().toISOString(),
        messageId: `msg_${Date.now()}`
      };
      return { success: true, dispatched };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface APIAggregatorParams {
  timeout?: number;
}

export interface APIAggregatorResult {
  success: boolean;
  responses?: any[];
  merged?: any;
  dispatched?: any;
  steps?: any[];
  error?: string;
}
