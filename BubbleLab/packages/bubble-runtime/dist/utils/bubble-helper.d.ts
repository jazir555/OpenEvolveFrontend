import { BubbleFactory } from '@bubblelab/bubble-core';
import { BubbleName, BubbleNodeType } from '@bubblelab/shared-schemas';
/**
 * Build a lookup map from className to bubble metadata
 */
export declare function buildClassNameLookup(factory: BubbleFactory): Map<string, {
    bubbleName: BubbleName;
    className: string;
    nodeType: BubbleNodeType;
}>;
//# sourceMappingURL=bubble-helper.d.ts.map