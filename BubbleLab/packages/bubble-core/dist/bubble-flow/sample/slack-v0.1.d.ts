import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
import { BubbleFlow } from '@bubblelab/bubble-core';
export interface Output {
    message: string;
}
export declare class TestBubbleFlow extends BubbleFlow<'slack/bot_mentioned'> {
    constructor();
    handle(payload: BubbleTriggerEventRegistry['slack/bot_mentioned']): Promise<Output>;
}
//# sourceMappingURL=slack-v0.1.d.ts.map