import type { BubbleFactory } from '../bubble-factory.js';
import type { ParsedBubble } from '@bubblelab/shared-schemas';
export interface VariableTypeInfo {
    name: string;
    type: string;
    line: number;
    column: number;
}
export interface ValidationResult {
    valid: boolean;
    errors?: string[];
    bubbleParameters?: Record<string, ParsedBubble>;
    variableTypes?: VariableTypeInfo[];
}
export declare function validateBubbleFlow(code: string, bubbleFactory: BubbleFactory): Promise<ValidationResult>;
//# sourceMappingURL=bubbleflow-validation.d.ts.map