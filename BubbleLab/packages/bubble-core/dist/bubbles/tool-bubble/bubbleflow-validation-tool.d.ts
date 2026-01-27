/**
 * BUBBLEFLOW VALIDATION TOOL
 *
 * A tool bubble that validates BubbleFlow TypeScript code for syntax, type errors,
 * and bubble structure. Takes TypeScript bubbleflow code as input and outputs
 * validation results including any errors and information about detected bubbles.
 *
 * Features:
 * - TypeScript syntax validation
 * - BubbleFlow class structure validation
 * - Bubble instantiation parsing and analysis
 * - Detailed error reporting with line numbers
 * - Bubble count and type information
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
/**
 * Define the parameters schema using Zod
 * This schema validates and types the input parameters for the validation tool
 */
declare const BubbleFlowValidationToolParamsSchema: z.ZodObject<{
    code: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        includeDetails: z.ZodDefault<z.ZodBoolean>;
        strictMode: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        includeDetails: boolean;
        strictMode: boolean;
    }, {
        includeDetails?: boolean | undefined;
        strictMode?: boolean | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    options?: {
        includeDetails: boolean;
        strictMode: boolean;
    } | undefined;
    credentials?: Record<string, string> | undefined;
    config?: Record<string, unknown> | undefined;
}, {
    code: string;
    options?: {
        includeDetails?: boolean | undefined;
        strictMode?: boolean | undefined;
    } | undefined;
    credentials?: Record<string, string> | undefined;
    config?: Record<string, unknown> | undefined;
}>;
/**
 * Type definitions derived from schemas
 */
type BubbleFlowValidationToolParamsInput = z.input<typeof BubbleFlowValidationToolParamsSchema>;
type BubbleFlowValidationToolParams = z.output<typeof BubbleFlowValidationToolParamsSchema>;
type BubbleFlowValidationToolResult = z.output<typeof BubbleFlowValidationToolResultSchema>;
/**
 * Define the result schema
 * This schema defines what the validation tool returns
 */
declare const BubbleFlowValidationToolResultSchema: z.ZodObject<{
    valid: z.ZodBoolean;
    errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bubbleCount: z.ZodOptional<z.ZodNumber>;
    bubbles: z.ZodOptional<z.ZodArray<z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodString;
        className: z.ZodString;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        parameterCount: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        bubbleName: string;
        variableName: string;
        className: string;
        hasAwait: boolean;
        hasActionCall: boolean;
        parameterCount: number;
    }, {
        bubbleName: string;
        variableName: string;
        className: string;
        hasAwait: boolean;
        hasActionCall: boolean;
        parameterCount: number;
    }>, "many">>;
    variableTypes: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        type: z.ZodString;
        line: z.ZodNumber;
        column: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        type: string;
        name: string;
        line: number;
        column: number;
    }, {
        type: string;
        name: string;
        line: number;
        column: number;
    }>, "many">>;
    metadata: z.ZodObject<{
        validatedAt: z.ZodString;
        codeLength: z.ZodNumber;
        strictMode: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
    }, {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    valid: boolean;
    success: boolean;
    metadata: {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
    };
    bubbles?: {
        bubbleName: string;
        variableName: string;
        className: string;
        hasAwait: boolean;
        hasActionCall: boolean;
        parameterCount: number;
    }[] | undefined;
    errors?: string[] | undefined;
    bubbleCount?: number | undefined;
    variableTypes?: {
        type: string;
        name: string;
        line: number;
        column: number;
    }[] | undefined;
}, {
    error: string;
    valid: boolean;
    success: boolean;
    metadata: {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
    };
    bubbles?: {
        bubbleName: string;
        variableName: string;
        className: string;
        hasAwait: boolean;
        hasActionCall: boolean;
        parameterCount: number;
    }[] | undefined;
    errors?: string[] | undefined;
    bubbleCount?: number | undefined;
    variableTypes?: {
        type: string;
        name: string;
        line: number;
        column: number;
    }[] | undefined;
}>;
/**
 * BubbleFlow Validation Tool
 * Validates TypeScript BubbleFlow code and provides detailed analysis
 */
export declare class BubbleFlowValidationTool extends ToolBubble<BubbleFlowValidationToolParams, BubbleFlowValidationToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName = "bubbleflow-validation-tool";
    static readonly schema: z.ZodObject<{
        code: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            includeDetails: z.ZodDefault<z.ZodBoolean>;
            strictMode: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            includeDetails: boolean;
            strictMode: boolean;
        }, {
            includeDetails?: boolean | undefined;
            strictMode?: boolean | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        options?: {
            includeDetails: boolean;
            strictMode: boolean;
        } | undefined;
        credentials?: Record<string, string> | undefined;
        config?: Record<string, unknown> | undefined;
    }, {
        code: string;
        options?: {
            includeDetails?: boolean | undefined;
            strictMode?: boolean | undefined;
        } | undefined;
        credentials?: Record<string, string> | undefined;
        config?: Record<string, unknown> | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        valid: z.ZodBoolean;
        errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        bubbleCount: z.ZodOptional<z.ZodNumber>;
        bubbles: z.ZodOptional<z.ZodArray<z.ZodObject<{
            variableName: z.ZodString;
            bubbleName: z.ZodString;
            className: z.ZodString;
            hasAwait: z.ZodBoolean;
            hasActionCall: z.ZodBoolean;
            parameterCount: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            bubbleName: string;
            variableName: string;
            className: string;
            hasAwait: boolean;
            hasActionCall: boolean;
            parameterCount: number;
        }, {
            bubbleName: string;
            variableName: string;
            className: string;
            hasAwait: boolean;
            hasActionCall: boolean;
            parameterCount: number;
        }>, "many">>;
        variableTypes: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodString;
            line: z.ZodNumber;
            column: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            line: number;
            column: number;
        }, {
            type: string;
            name: string;
            line: number;
            column: number;
        }>, "many">>;
        metadata: z.ZodObject<{
            validatedAt: z.ZodString;
            codeLength: z.ZodNumber;
            strictMode: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            strictMode: boolean;
            validatedAt: string;
            codeLength: number;
        }, {
            strictMode: boolean;
            validatedAt: string;
            codeLength: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        valid: boolean;
        success: boolean;
        metadata: {
            strictMode: boolean;
            validatedAt: string;
            codeLength: number;
        };
        bubbles?: {
            bubbleName: string;
            variableName: string;
            className: string;
            hasAwait: boolean;
            hasActionCall: boolean;
            parameterCount: number;
        }[] | undefined;
        errors?: string[] | undefined;
        bubbleCount?: number | undefined;
        variableTypes?: {
            type: string;
            name: string;
            line: number;
            column: number;
        }[] | undefined;
    }, {
        error: string;
        valid: boolean;
        success: boolean;
        metadata: {
            strictMode: boolean;
            validatedAt: string;
            codeLength: number;
        };
        bubbles?: {
            bubbleName: string;
            variableName: string;
            className: string;
            hasAwait: boolean;
            hasActionCall: boolean;
            parameterCount: number;
        }[] | undefined;
        errors?: string[] | undefined;
        bubbleCount?: number | undefined;
        variableTypes?: {
            type: string;
            name: string;
            line: number;
            column: number;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Validates BubbleFlow TypeScript code for syntax and structure";
    static readonly longDescription = "\n    A comprehensive validation tool for BubbleFlow TypeScript code.\n    \n    What it does:\n    - Validates TypeScript syntax and compilation\n    - Checks BubbleFlow class structure and requirements\n    - Parses and analyzes bubble instantiations\n    - Provides detailed error reporting with line numbers\n    - Returns metadata about detected bubbles\n    \n    How it works:\n    - Uses TypeScript compiler API for syntax validation\n    - Validates that code extends BubbleFlow and has handle method\n    - Parses bubble instantiations using AST analysis\n    - Maps bubble class names to registered bubble types\n    \n    Use cases:\n    - When an AI agent needs to validate user-provided BubbleFlow code\n    - When checking code before execution or deployment\n    - When providing feedback on BubbleFlow implementation\n    - When analyzing bubble usage patterns in code\n  ";
    static readonly alias = "validate-bubbleflow";
    private bubbleFactory;
    private initializationPromise;
    /**
     * Constructor
     */
    constructor(params: BubbleFlowValidationToolParamsInput, context?: BubbleContext);
    private initializeBubbleFactory;
    /**
     * Main action method - performs BubbleFlow validation
     */
    performAction(context?: BubbleContext): Promise<BubbleFlowValidationToolResult>;
}
export {};
/**
 * REGISTRATION AND USAGE
 *
 * 1. Register the tool in the BubbleFactory:
 * ```typescript
 * import { BubbleFlowValidationTool } from './services/bubbleflow-validation-tool.js';
 *
 * // In registerDefaults() method:
 * this.register('bubbleflow-validation', BubbleFlowValidationTool as BubbleClassWithMetadata);
 * ```
 *
 * 2. Direct usage:
 * ```typescript
 * const tool = new BubbleFlowValidationTool({
 *   code: 'export class TestFlow extends BubbleFlow<"webhook/http"> { ... }',
 *   options: { includeDetails: true }
 * });
 * const result = await tool.action();
 * ```
 *
 * 3. AI Agent usage:
 * ```typescript
 * // AI agent calls with:
 * {
 *   "code": "export class TestFlow extends BubbleFlow<'webhook/http'> { ... }",
 *   "options": { "includeDetails": true, "strictMode": true }
 * }
 * ```
 */
//# sourceMappingURL=bubbleflow-validation-tool.d.ts.map