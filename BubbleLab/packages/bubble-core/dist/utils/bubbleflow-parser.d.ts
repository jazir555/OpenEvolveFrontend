import type { BubbleFactory } from '../bubble-factory.js';
import type { BubbleName, ParsedBubble, BubbleParameter } from '@bubblelab/shared-schemas';
export interface BubbleFlowParseResult {
    success: boolean;
    bubbles: Record<string, ParsedBubble>;
    errors?: string[];
}
/**
 * Parses a BubbleFlow TypeScript code and extracts all bubble instantiations
 * with their parameters using the BubbleRegistry for accurate mapping.
 */
export declare function parseBubbleFlow(code: string, bubbleFactory: BubbleFactory): BubbleFlowParseResult;
/**
 * Reconstructs a BubbleFlow code string from parsed bubbles and parameters.
 */
export declare function reconstructBubbleFlow(originalCode: string, bubbleParameters: Record<string, ParsedBubble>, bubbleFactory: BubbleFactory): {
    success: boolean;
    code?: string;
    errors?: string[];
};
/**
 * Validates bubble parameters against their schema from the BubbleRegistry
 */
export declare function validateBubbleParameters(bubbleName: BubbleName, bubbleFactory: BubbleFactory, _parameters: BubbleParameter[]): {
    valid: boolean;
    errors?: string[];
};
/**
 * Gets available parameter schema for a bubble from the registry
 */
export declare function getBubbleParameterSchema(bubbleName: BubbleName, bubbleFactory: BubbleFactory): unknown;
//# sourceMappingURL=bubbleflow-parser.d.ts.map