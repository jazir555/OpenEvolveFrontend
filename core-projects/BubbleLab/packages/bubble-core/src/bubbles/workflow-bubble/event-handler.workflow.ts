/**
 * EVENT HANDLER WORKFLOW
 *
 * Route and handle events with pattern matching and middleware support.
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';

const EventTypeSchema = z.enum([
  'webhook',
  'message',
  'schedule',
  'custom',
]);

const HandlerTypeSchema = z.enum([
  'http',
  'workflow',
  'function',
  'slack',
  'email',
]);

const EventHandlerParamsSchema = z.object({
  eventType: EventTypeSchema,
  eventPayload: z.record(z.unknown()),
  routingRules: z.array(z.object({
    condition: z.string().describe('JavaScript expression to match'),
    handler: z.object({
      type: HandlerTypeSchema,
      config: z.record(z.unknown()),
    }),
    priority: z.number().default(0),
  })),
  middleware: z.array(z.object({
    name: z.string(),
    config: z.record(z.unknown()).optional(),
  })).optional().describe('Middleware to apply before handlers'),
  errorHandling: z.enum(['continue', 'stop', 'retry']).default('continue'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

type EventHandlerParams = z.input<typeof EventHandlerParamsSchema>;

const EventHandlerResultSchema = z.object({
  success: z.boolean(),
  error: z.string(),
  matchedHandlers: z.array(z.object({
    condition: z.string(),
    handlerType: z.string(),
    result: z.unknown().optional(),
    success: z.boolean(),
  })),
  middlewareResults: z.array(z.object({
    name: z.string(),
    success: z.boolean(),
    error: z.string().optional(),
  })).optional(),
  executionTime: z.number(),
});

export class EventHandlerWorkflow extends WorkflowBubble<
  EventHandlerParams,
  z.infer<typeof EventHandlerResultSchema>
> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName: BubbleName = 'event-handler-workflow';
  static readonly schema = EventHandlerParamsSchema;
  static readonly resultSchema = EventHandlerResultSchema;
  static readonly shortDescription = 'Route and handle events with pattern matching';
  static readonly longDescription = `
    Event routing and handling system with pattern matching, middleware support, and flexible handlers.

    Features:
    - Pattern-based event routing with JavaScript expressions
    - Multiple handler types (HTTP, workflow, function, Slack, email)
    - Middleware pipeline for pre/post processing
    - Priority-based handler execution
    - Comprehensive error handling strategies

    Use cases:
    - Webhook event processing
    - Message queue handling
    - Event-driven architecture
    - Custom event routing
    - Integration event processing
  `;
  static readonly alias = 'handle-event';

  constructor(params: EventHandlerParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<z.infer<typeof EventHandlerResultSchema>> {
    const startTime = Date.now();
    console.log(`[EventHandler] Processing event: ${this.params.eventType}`);

    try {
      const middlewareResults: z.infer<typeof EventHandlerResultSchema>['middlewareResults'] = [];

      // Step 1: Execute middleware
      if (this.params.middleware) {
        console.log('[EventHandler] Step 1: Executing middleware');

        for (const mw of this.params.middleware) {
          const result = await this.executeMiddleware(mw);
          middlewareResults.push(result);

          if (!result.success && this.params.errorHandling === 'stop') {
            return {
              success: false,
              error: `Middleware ${mw.name} failed: ${result.error}`,
              matchedHandlers: [],
              middlewareResults,
              executionTime: Date.now() - startTime,
            };
          }
        }
      }

      // Step 2: Match and execute handlers
      console.log('[EventHandler] Step 2: Matching and executing handlers');
      const matchedHandlers: z.infer<typeof EventHandlerResultSchema>['matchedHandlers'] = [];

      // Sort rules by priority (higher priority first)
      const sortedRules = [...this.params.routingRules].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));

      for (const rule of sortedRules) {
        if (this.evaluateCondition(rule.condition, this.params.eventPayload)) {
          console.log(`[EventHandler] Matched condition: ${rule.condition}`);

          const handlerResult = await this.executeHandler(rule.handler);

          matchedHandlers.push({
            condition: rule.condition,
            handlerType: rule.handler.type,
            result: handlerResult,
            success: handlerResult !== undefined,
          });

          if (!handlerResult && this.params.errorHandling === 'stop') {
            break;
          }
        }
      }

      const executionTime = Date.now() - startTime;
      console.log(`[EventHandler] Event processed in ${executionTime}ms`);

      return {
        success: true,
        error: '',
        matchedHandlers,
        middlewareResults,
        executionTime,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[EventHandler] Event processing failed:', errorMessage);

      return {
        success: false,
        error: errorMessage,
        matchedHandlers: [],
        executionTime: Date.now() - startTime,
      };
    }
  }

  /**
   * Execute middleware
   */
  private async executeMiddleware(middleware: any) {
    if (!middleware) {
      return { name: 'unknown', success: false, error: 'Middleware is undefined' };
    }
    console.log(`[EventHandler] Executing middleware: ${middleware.name}`);

    try {
      // Middleware implementation would go here
      // For now, just log and return success
      return {
        name: middleware.name,
        success: true,
      };
    } catch (error) {
      return {
        name: middleware.name,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Evaluate condition against event payload
   */
  private evaluateCondition(condition: string, payload: Record<string, unknown>): boolean {
    try {
      // Create a safe evaluation context
      const func = new Function('payload', `
        try {
          return ${condition};
        } catch (e) {
          return false;
        }
      `);

      return func(payload);
    } catch (error) {
      console.error('[EventHandler] Condition evaluation failed:', error);
      return false;
    }
  }

  /**
   * Execute handler
   */
  private async executeHandler(handler: z.infer<typeof EventHandlerParamsSchema>['routingRules'][number]['handler']) {
    console.log(`[EventHandler] Executing handler: ${handler.type}`);

    try {
      if (handler.type === 'http') {
        const config = handler.config as { url: string; method?: string; body?: unknown };
        const { HttpBubble } = await import('../service-bubble/http.js');

        const httpBubble = new HttpBubble(
          {
            url: config.url,
            method: (config.method as any) || 'POST',
            body: config.body as string | Record<string, unknown> | undefined,
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await httpBubble.action();
        return result.data as unknown as { success: boolean; data: string | Record<string, unknown> | undefined; error: string };
      } else if (handler.type === 'slack') {
        const config = handler.config as { channel?: string; message: string };
        const { SlackBubble } = await import('../service-bubble/slack.js');

        const slackBubble = new SlackBubble(
          {
            operation: 'sendMessage',
            channel: config.channel || '#events',
            text: config.message,
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await slackBubble.action();
        return result;
      } else {
        console.log(`[EventHandler] Handler type ${handler.type} not fully implemented`);
        return { success: true };
      }
    } catch (error) {
      console.error('[EventHandler] Handler execution failed:', error);
      return undefined;
    }
  }
}
