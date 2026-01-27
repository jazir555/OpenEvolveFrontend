import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
import { BubbleFlow } from '@bubblelab/bubble-core';
export interface Output {
    message: string;
}
export declare class TestBubbleFlow extends BubbleFlow<'webhook/http'> {
    constructor();
    handle(payload: BubbleTriggerEventRegistry['webhook/http']): Promise<Output>;
}
//# sourceMappingURL=slackagenttest.d.ts.map