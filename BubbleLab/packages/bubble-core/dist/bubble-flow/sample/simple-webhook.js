import { BubbleFlow, AIAgentBubble } from '@bubblelab/bubble-core';
export class TestBubbleFlow extends BubbleFlow {
    constructor() {
        super('test-flow', 'A flow that handles webhook events');
    }
    async handle(payload) {
        const result = await new AIAgentBubble({
            message: 'Hello, how are you?',
            model: {
                model: 'google/gemini-2.5-flash',
            },
        }).action();
        return {
            message: `Response from ${payload.path}: ${result.data?.response ?? 'No response'}`,
        };
    }
}
//# sourceMappingURL=simple-webhook.js.map