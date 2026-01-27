import { hashToVariableId } from '@bubblelab/shared-schemas';
export class BubbleFlow {
    name;
    description;
    logger;
    __currentInvocationCallSiteKey;
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
    cronSchedule;
    constructor(name, description, logger) {
        this.name = name;
        this.description = description;
        this.logger = logger;
    }
    /**
     * Get the logger instance if available
     */
    getLogger() {
        return this.logger;
    }
    /**
     * Set a logger for this flow instance
     */
    setLogger(logger) {
        this.logger = logger;
    }
    __setInvocationCallSiteKey(key) {
        const previous = this.__currentInvocationCallSiteKey;
        this.__currentInvocationCallSiteKey = key || undefined;
        return previous;
    }
    __restoreInvocationCallSiteKey(previous) {
        this.__currentInvocationCallSiteKey = previous || undefined;
    }
    __getInvocationCallSiteKey() {
        return this.__currentInvocationCallSiteKey;
    }
    __computeInvocationVariableId(originalVariableId) {
        if (typeof originalVariableId !== 'number' ||
            !this.__currentInvocationCallSiteKey) {
            return originalVariableId;
        }
        return hashToVariableId(`${originalVariableId}:${this.__currentInvocationCallSiteKey}`);
    }
}
//# sourceMappingURL=bubble-flow-class.js.map