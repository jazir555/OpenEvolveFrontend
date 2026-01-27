import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
import { BubbleFlow } from '@bubblelab/bubble-core';
export interface Output {
    success: boolean;
    directAnswer?: string;
    insights?: string[];
    recommendations?: string[];
    slackMessageId?: string;
    error?: string;
}
export declare class DataAnalystFlow extends BubbleFlow<'slack/bot_mentioned'> {
    constructor();
    handle(payload: BubbleTriggerEventRegistry['slack/bot_mentioned']): Promise<Output>;
}
//# sourceMappingURL=data-analyst-flow.d.ts.map