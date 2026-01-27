import { BubbleFlow } from '@bubblelab/bubble-core';
// This is a test bubble flow that is used to test the bubble flow system
export class TestBubbleFlow extends BubbleFlow {
    constructor() {
        super('test-flow', 'A flow that handles webhook events');
    }
    async handle(payload) {
        return {
            message: `Response from ${payload.path}, ${payload.timestamp}, ${payload.type}: Hello!${payload.body?.name ?? 'there'}! Welcome to Nodex!`,
        };
    }
}
//# sourceMappingURL=sanitytest.js.map