import { BubbleFlow } from '@bubblelab/bubble-core';
import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
export declare class AIAnalysisFlow extends BubbleFlow<'webhook/http'> {
    constructor();
    handle(payload: BubbleTriggerEventRegistry['webhook/http']): Promise<{
        success: boolean;
        input: {};
        analysis: {
            analysis: string;
            sentiment: string;
            wordCount: number;
        };
        timestamp: string;
        error?: undefined;
    } | {
        success: boolean;
        error: string;
        input?: undefined;
        analysis?: undefined;
        timestamp?: undefined;
    }>;
}
//# sourceMappingURL=error-ts.d.ts.map