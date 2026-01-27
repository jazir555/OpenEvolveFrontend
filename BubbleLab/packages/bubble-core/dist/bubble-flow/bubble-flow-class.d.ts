import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
import type { BubbleFlowOperationResult } from '../types/bubble.js';
import type { BubbleLogger } from '../logging/BubbleLogger.js';
export declare abstract class BubbleFlow<TEventType extends keyof BubbleTriggerEventRegistry> {
    readonly name: string;
    readonly description: string;
    protected logger?: BubbleLogger;
    private __currentInvocationCallSiteKey?;
    /**
     * Cron schedule expression for schedule/cron event types.
     * Required for flows that extend BubbleFlow<'schedule/cron'>.
     * Uses standard 5-part cron format: minute hour day month day-of-week
     *
     * @example
     * ```typescript
     * readonly cronSchedule = '0 0 * * *'; // Daily at midnight
     * readonly cronSchedule = '0 9 * * 1-5'; // Weekdays at 9am
     * readonly cronSchedule = '*\/15 * * * *'; // Every 15 minutes
     * ```
     *
     * Note: This property is enforced by the ESLint rule 'bubble-core/require-cron-schedule'
     * for flows with event type 'schedule/cron'.
     */
    readonly cronSchedule?: string;
    constructor(name: string, description: string, logger?: BubbleLogger);
    abstract handle(payload: BubbleTriggerEventRegistry[TEventType]): Promise<BubbleFlowOperationResult>;
    /**
     * Get the logger instance if available
     */
    getLogger(): BubbleLogger | undefined;
    /**
     * Set a logger for this flow instance
     */
    setLogger(logger: BubbleLogger): void;
    __setInvocationCallSiteKey(key?: string): string | undefined;
    __restoreInvocationCallSiteKey(previous?: string | undefined): void;
    __getInvocationCallSiteKey(): string | undefined;
    __computeInvocationVariableId(originalVariableId?: number): number | undefined;
}
//# sourceMappingURL=bubble-flow-class.d.ts.map