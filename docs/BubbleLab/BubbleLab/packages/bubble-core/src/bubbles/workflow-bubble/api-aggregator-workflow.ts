import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * APIAggregatorWorkflow - apiaggregator workflow
 */
export class APIAggregatorWorkflow extends WorkflowBubble<APIAggregatorParams, APIAggregatorResult> {
  bubbleName = 'apiaggregator';
  type = 'workflow';
  alias = 'apiaggregator';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<APIAggregatorResult> {
    const steps = [];

    try {
      // Step 1: aggregate
      const step1Result = await this.aggregate(input);
      steps.push({
        step: 1,
        name: 'aggregate',
        status: 'completed',
        result: step1Result
      });
      // Step 2: merge
      const step2Result = await this.merge(input);
      steps.push({
        step: 2,
        name: 'merge',
        status: 'completed',
        result: step2Result
      });
      // Step 3: dispatch
      const step3Result = await this.dispatch(input);
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

  async aggregate(params: any): Promise<any> {
    try {
      // Implementation for aggregate
      const result = await this.client.aggregate(params);
      return { success: true, result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
  async merge(params: any): Promise<any> {
    try {
      // Implementation for merge
      const result = await this.client.merge(params);
      return { success: true, result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
  async dispatch(params: any): Promise<any> {
    try {
      // Implementation for dispatch
      const result = await this.client.dispatch(params);
      return { success: true, result };
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
  steps?: any[];
  error?: string;
}
