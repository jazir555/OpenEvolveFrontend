import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * EventHandlerWorkflow - Event routing and transformation
 */
export class EventHandlerWorkflow extends WorkflowBubble<EventHandlerParams, EventHandlerResult> {
  bubbleName = 'event-handler';
  type = 'workflow';
  alias = 'event-handler';

  params = {
    timeout: z.number().int().positive().default(300000)
  };

  async execute(input: any): Promise<EventHandlerResult> {
    const steps = [];

    try {
      // Step 1: Route
      const step1Result = await this.route(input);
      steps.push({
        step: 1,
        name: 'route',
        status: 'completed',
        result: step1Result
      });

      // Step 2: Handle
      const step2Result = await this.handle({ ...input, routing: step1Result });
      steps.push({
        step: 2,
        name: 'handle',
        status: 'completed',
        result: step2Result
      });

      // Step 3: Transform
      const step3Result = await this.transform({ ...input, handled: step2Result });
      steps.push({
        step: 3,
        name: 'transform',
        status: 'completed',
        result: step3Result
      });

      return { success: true, steps };
    } catch (error: any) {
      return { success: false, error: error.message, steps };
    }
  }

  async route(params: { event: any; rules?: any[] }): Promise<EventHandlerResult> {
    try {
      const routing = {
        eventType: params.event.type || 'unknown',
        handler: `handler_${params.event.type}`,
        priority: params.event.priority || 'normal',
        routedAt: new Date().toISOString()
      };
      return { success: true, routing };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async handle(params: { event: any; routing: any }): Promise<EventHandlerResult> {
    try {
      const handled = {
        eventId: params.event.id || `evt_${Date.now()}`,
        handler: params.routing.handler,
        status: 'processed',
        handledAt: new Date().toISOString(),
        result: { message: 'Event processed successfully' }
      };
      return { success: true, handled };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transform(params: { handled: any; outputFormat?: string }): Promise<EventHandlerResult> {
    try {
      const transformed = {
        original: params.handled,
        format: params.outputFormat || 'json',
        data: JSON.stringify(params.handled),
        transformedAt: new Date().toISOString()
      };
      return { success: true, transformed };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface EventHandlerParams {
  timeout?: number;
}

export interface EventHandlerResult {
  success: boolean;
  routing?: any;
  handled?: any;
  transformed?: any;
  steps?: any[];
  error?: string;
}
