import { z } from 'zod';
import type { IBubble, BubbleContext, BubbleResult, BubbleOperationResult } from '@bubblelab/bubble-core';
/**
 * Abstract base class for all bubble types
 * Implements common properties and methods defined in IBubble interface
 */
export declare abstract class BaseBubble<TParams = unknown, TResult extends BubbleOperationResult = BubbleOperationResult> implements IBubble<TResult> {
    readonly name: string;
    readonly schema: z.ZodObject<z.ZodRawShape>;
    readonly resultSchema: z.ZodObject<z.ZodRawShape>;
    readonly shortDescription: string;
    readonly longDescription: string;
    readonly alias?: string;
    abstract readonly type: 'service' | 'workflow' | 'tool' | 'ui' | 'infra';
    protected readonly params: TParams;
    protected context?: BubbleContext;
    previousResult: BubbleResult<BubbleOperationResult> | undefined;
    protected readonly instanceId?: string;
    constructor(params: unknown, context?: BubbleContext, instanceId?: string);
    /**
     * Compute child context based on dependency graph and current unique id.
     * Finds the node matching currentUniqueId, then determines this child's unique id as:
     * - If instanceId is provided: `${currentUniqueId}.${this.name}#${instanceId}`
     * - Otherwise: `${currentUniqueId}.${this.name}#k` for the next ordinal k
     * Assigns the variableId from the dependency graph if present, otherwise keeps parent's variableId.
     */
    private computeChildContext;
    saveResult<R extends BubbleOperationResult>(result: BubbleResult<R>): void;
    clearSavedResult(): void;
    /**
     * Override toJSON to prevent credential leaking via JSON.stringify or console.log
     * Only exposes safe metadata, never params which may contain credentials
     */
    toJSON(): Record<string, unknown>;
    /**
     * Execute the bubble - just runs the action
     */
    action(): Promise<BubbleResult<TResult>>;
    /**
     * Generate mock result data based on the result schema
     * Useful for testing and development when you need sample data
     */
    generateMockResult(): BubbleResult<TResult>;
    /**
     * Generate mock result with a specific seed for reproducible results
     * Useful for consistent testing scenarios
     */
    generateMockResultWithSeed(seed: number): BubbleResult<TResult>;
    /**
     * Perform the actual bubble action - must be implemented by subclasses
     */
    protected abstract performAction(context?: BubbleContext): Promise<TResult>;
}
//# sourceMappingURL=base-bubble-class.d.ts.map