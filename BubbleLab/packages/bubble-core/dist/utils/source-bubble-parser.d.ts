import type { BubbleName } from '@bubblelab/shared-schemas';
import type { BubbleClassWithMetadata } from '../bubble-factory.js';
export interface ParsedInstanceInfo {
    variableName: string;
    isAnonymous: boolean;
    startLine?: number;
    endLine?: number;
    tools?: BubbleName[];
}
export type ParsedInstancesByBubble = Map<BubbleName, ParsedInstanceInfo[]>;
/**
 * Build a className -> { bubbleName } map from the factory registry
 */
export declare function buildClassNameLookup(registry: Map<BubbleName, BubbleClassWithMetadata<any>>): Map<string, {
    bubbleName: BubbleName;
    className: string;
}>;
/**
 * Parse a source file and extract all bubble instantiations with variable names and locations.
 * Works for patterns:
 *  - const v = new Class({...})
 *  - await new Class({...}).action()
 */
export declare function parseBubbleInstancesFromSource(sourceCode: string, classNameLookup: Map<string, {
    bubbleName: BubbleName;
    className: string;
}>, options?: {
    debug?: boolean;
    filePath?: string;
}): ParsedInstancesByBubble;
//# sourceMappingURL=source-bubble-parser.d.ts.map