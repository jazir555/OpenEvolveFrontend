import type { IWorkflowBubble, BubbleOperationResult, ServiceBubbleParams, BubbleContext } from '@bubblelab/bubble-core';
import { BaseBubble } from './base-bubble-class.js';
/**
 * WorkflowBubble - Higher-level abstraction that orchestrates ServiceBubbles
 * to create common, reusable workflow patterns.
 *
 * Key principles:
 * - User-friendly parameter names with clear purpose
 * - TypeScript type safety with helpful intellisense
 * - Composable patterns that reduce BubbleFlow complexity
 * - Error handling and validation at workflow level
 */
export declare abstract class WorkflowBubble<TParams extends ServiceBubbleParams = ServiceBubbleParams, TResult extends BubbleOperationResult = BubbleOperationResult> extends BaseBubble<TParams, TResult> implements IWorkflowBubble<TResult> {
    readonly type: "workflow";
    constructor(params: unknown, context?: BubbleContext, instanceId?: string);
    /**
     * Get the current parameters
     */
    get currentParams(): TParams;
    /**
     * Get the current context
     */
    get currentContext(): BubbleContext | undefined;
}
//# sourceMappingURL=workflow-bubble-class.d.ts.map