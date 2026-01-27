import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GetBubbleDetailsToolParamsSchema: z.ZodObject<{
    bubbleName: z.ZodString;
    includeInputSchema: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    bubbleName: string;
    includeInputSchema: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    bubbleName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeInputSchema?: boolean | undefined;
}>;
type GetBubbleDetailsToolParamsInput = z.input<typeof GetBubbleDetailsToolParamsSchema>;
type GetBubbleDetailsToolParams = z.output<typeof GetBubbleDetailsToolParamsSchema>;
type GetBubbleDetailsToolResult = z.output<typeof GetBubbleDetailsToolResultSchema>;
declare const GetBubbleDetailsToolResultSchema: z.ZodObject<{
    name: z.ZodString;
    alias: z.ZodOptional<z.ZodString>;
    inputSchema: z.ZodOptional<z.ZodString>;
    outputSchema: z.ZodString;
    usageExample: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    name: string;
    success: boolean;
    outputSchema: string;
    usageExample: string;
    alias?: string | undefined;
    inputSchema?: string | undefined;
}, {
    error: string;
    name: string;
    success: boolean;
    outputSchema: string;
    usageExample: string;
    alias?: string | undefined;
    inputSchema?: string | undefined;
}>;
export declare class GetBubbleDetailsTool extends ToolBubble<GetBubbleDetailsToolParams, GetBubbleDetailsToolResult> {
    static readonly type: "tool";
    static readonly bubbleName = "get-bubble-details-tool";
    static readonly schema: z.ZodObject<{
        bubbleName: z.ZodString;
        includeInputSchema: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        bubbleName: string;
        includeInputSchema: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        bubbleName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeInputSchema?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        name: z.ZodString;
        alias: z.ZodOptional<z.ZodString>;
        inputSchema: z.ZodOptional<z.ZodString>;
        outputSchema: z.ZodString;
        usageExample: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        name: string;
        success: boolean;
        outputSchema: string;
        usageExample: string;
        alias?: string | undefined;
        inputSchema?: string | undefined;
    }, {
        error: string;
        name: string;
        success: boolean;
        outputSchema: string;
        usageExample: string;
        alias?: string | undefined;
        inputSchema?: string | undefined;
    }>;
    static readonly shortDescription = "Provides detailed information about a specific bubble, including schema, parameters, and documentation";
    static readonly longDescription = "\n    A tool bubble that retrieves comprehensive information about any registered bubble in the system.\n    \n    Returns detailed information including:\n    - Complete schema with parameter types and descriptions\n    - Result schema for expected outputs\n    - Credential requirements\n    - AI-formatted documentation\n    - Usage examples\n    \n    Use cases:\n    - AI agent understanding of specific bubble capabilities\n    - Parameter validation before bubble instantiation\n    - Documentation generation and help systems\n    - Dynamic form generation for bubble configuration\n  ";
    static readonly alias = "details";
    private factory;
    constructor(params: GetBubbleDetailsToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<GetBubbleDetailsToolResult>;
    private generateOutputSchemaString;
    private generateTypeInfo;
    private generateUsageExample;
    private isDiscriminatedUnion;
    private generateOperationExamples;
    private generateSingleExample;
    private formatOperationComment;
    private getResultSchemaOption;
    private getFirstResultSchemaOption;
    /**
     * Extracts the description from a Zod schema type
     */
    private getParameterDescription;
    private generateExampleParams;
    private generateExampleValue;
    /**
     * Checks if a key represents a credential parameter that should be omitted from examples
     */
    private isCredentialKey;
    private toCamelCase;
    private toPascalCase;
}
export {};
//# sourceMappingURL=get-bubble-details-tool.d.ts.map