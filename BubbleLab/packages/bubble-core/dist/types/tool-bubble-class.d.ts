import { z } from 'zod';
import type { IToolBubble, ServiceBubbleParams, BubbleContext, BubbleOperationResult, BubbleResult } from '@bubblelab/bubble-core';
import { BaseBubble } from './base-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
export interface LangGraphTool {
    name: string;
    description: string;
    schema: z.ZodSchema;
    func<TResult extends BubbleOperationResult = BubbleOperationResult>(params: unknown): Promise<BubbleResult<TResult>>;
}
/**
 * Abstract base class for all tool bubbles that can be converted to LangGraph tools
 */
export declare abstract class ToolBubble<TParams extends ServiceBubbleParams = ServiceBubbleParams, TResult extends BubbleOperationResult = BubbleOperationResult> extends BaseBubble<TParams, TResult> implements IToolBubble<TResult> {
    readonly type: "tool";
    constructor(params: unknown, context?: BubbleContext, instanceId?: string);
    static toolAgent(credentials?: Partial<Record<CredentialType, string>>, config?: Record<string, unknown>, context?: BubbleContext): LangGraphTool;
}
//# sourceMappingURL=tool-bubble-class.d.ts.map