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
export class WorkflowBubble extends BaseBubble {
    type = 'workflow';
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    /**
     * Get the current parameters
     */
    get currentParams() {
        return this.params;
    }
    /**
     * Get the current context
     */
    get currentContext() {
        return this.context;
    }
}
//# sourceMappingURL=workflow-bubble-class.js.map