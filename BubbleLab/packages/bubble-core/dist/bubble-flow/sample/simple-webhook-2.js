import { BubbleFlow } from '../bubble-flow-class.js';
// Import all services
import * as bubbles from '@bubblelab/bubble-core';
export class TestBubbleFlow extends BubbleFlow {
    constructor() {
        super('test-flow', 'A flow that handles webhook events');
    }
    async handle(payload) {
        // Type assertion to access your custom fields
        const customPayload = payload;
        const { userId, requestId, customData } = customPayload;
        const result = await new bubbles.AIAgentBubble({
            message: `Hello user ${userId}, priority: ${customData.priority}`,
            model: {
                model: 'google/gemini-2.5-flash',
            },
        }).action();
        return {
            message: `Response from ${payload.path} (Request: ${requestId}): ${result.data?.response ?? 'No response'}`,
        };
    }
}
//# sourceMappingURL=simple-webhook-2.js.map