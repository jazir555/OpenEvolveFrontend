import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebhookRepeaterWorkflow - Webhook reception and retry logic
 */
export class WebhookRepeaterWorkflow extends WorkflowBubble<WebhookRepeaterParams, WebhookRepeaterResult> {
  bubbleName = 'webhook-repeater';
  type = 'workflow';
  alias = 'webhook-repeater';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<WebhookRepeaterResult> {
    const steps = [];

    try {
      // Step 1: Receive
      const step1Result = await this.receive(input);
      steps.push({
        step: 1,
        name: 'receive',
        status: 'completed',
        result: step1Result
      });

      // Step 2: Retry
      const step2Result = await this.retry({ ...input, webhook: step1Result });
      steps.push({
        step: 2,
        name: 'retry',
        status: 'completed',
        result: step2Result
      });

      // Step 3: Dispatch
      const step3Result = await this.dispatch({ ...input, attempts: step2Result });
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

  async receive(params: { payload: any }): Promise<WebhookRepeaterResult> {
    try {
      const webhook = {
        id: `wh_${Date.now()}`,
        payload: params.payload,
        receivedAt: new Date().toISOString(),
        attempts: 0
      };
      return { success: true, webhook };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async retry(params: { webhook: any; maxRetries?: number }): Promise<WebhookRepeaterResult> {
    try {
      const maxRetries = params.maxRetries || 3;
      const attempts = [];
      for (let i = 0; i < maxRetries; i++) {
        attempts.push({
          attempt: i + 1,
          timestamp: new Date().toISOString(),
          success: i === 2 // Simulate success on 3rd attempt
        });
      }
      return { success: true, attempts };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async dispatch(params: { webhook: any; attempts: any[] }): Promise<WebhookRepeaterResult> {
    try {
      const dispatched = {
        webhookId: params.webhook.id,
        successfulAttempt: params.attempts.find(a => a.success),
        dispatchedAt: new Date().toISOString()
      };
      return { success: true, dispatched };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface WebhookRepeaterParams {
  timeout?: number;
}

export interface WebhookRepeaterResult {
  success: boolean;
  webhook?: any;
  attempts?: any[];
  dispatched?: any;
  steps?: any[];
  error?: string;
}
