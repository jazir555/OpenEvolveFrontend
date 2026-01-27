import type { WebhookEvent } from '@bubblelab/shared-schemas';
import { BubbleFlow } from '../bubble-flow-class.js';
export interface Output {
    message: string;
}
export interface CustomWebhookPayload extends WebhookEvent {
    userId: string;
    requestId: string;
    customData: {
        source: string;
        priority: 'low' | 'medium' | 'high';
        metadata: Record<string, any>;
    };
}
export declare class TestBubbleFlow extends BubbleFlow<'webhook/http'> {
    constructor();
    handle(payload: CustomWebhookPayload): Promise<Output>;
}
//# sourceMappingURL=simple-webhook-2.d.ts.map