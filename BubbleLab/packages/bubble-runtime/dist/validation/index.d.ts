import type { ParsedBubbleWithInfo, CredentialType, BubbleTrigger, ParsedWorkflow } from '@bubblelab/shared-schemas';
import { BubbleFactory } from '@bubblelab/bubble-core';
export interface ValidationResult {
    valid: boolean;
    errors?: string[];
    syntaxErrors?: string[];
    lintErrors?: string[];
}
export interface ValidationAndExtractionResult extends ValidationResult {
    bubbleParameters?: Record<number, ParsedBubbleWithInfo>;
    workflow?: ParsedWorkflow;
    inputSchema?: Record<string, unknown>;
    trigger?: BubbleTrigger;
    requiredCredentials?: Record<string, CredentialType[]>;
}
/**
 * Validates a BubbleFlow TypeScript code
 * This focuses purely on validation without extraction
 *
 * @param code - The TypeScript code to validate
 * @returns ValidationResult with success status and errors
 */
export declare function validateBubbleFlow(code: string, requireLintErrors?: boolean): Promise<ValidationResult>;
/**
 * Validates a BubbleFlow TypeScript code and extracts bubble parameters
 * This is the main entry point for bubble runtime validation with extraction
 *
 * @param code - The TypeScript code to validate
 * @returns ValidationAndExtractionResult with success status, errors, and extracted parameters
 */
export declare function validateAndExtract(code: string, bubbleFactory: BubbleFactory, requireLintErrors?: boolean): Promise<ValidationAndExtractionResult>;
//# sourceMappingURL=index.d.ts.map