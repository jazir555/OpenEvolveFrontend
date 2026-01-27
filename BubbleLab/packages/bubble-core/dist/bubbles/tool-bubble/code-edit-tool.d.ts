/**
 * EDIT BUBBLEFLOW TOOL
 *
 * A tool bubble that applies code edits to BubbleFlow files using Morph Fast Apply.
 * This tool uses the Morph API via OpenRouter to intelligently merge code changes
 * specified by an AI agent into existing BubbleFlow code, following the Fast Apply
 * pattern used in Cursor.
 *
 * Features:
 * - Intelligent code merging using Morph Fast Apply model
 * - Support for lazy edits with "// ... existing code ..." markers
 * - Minimal context repetition for efficient edits
 * - Automatic validation after edits
 * - Detailed diff reporting
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
/**
 * Define the parameters schema using Zod
 * This schema validates and types the input parameters for the edit tool
 */
declare const EditBubbleFlowToolParamsSchema: z.ZodObject<{
    initialCode: z.ZodEffects<z.ZodString, string, string>;
    instructions: z.ZodString;
    codeEdit: z.ZodEffects<z.ZodString, string, string>;
    morphModel: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    initialCode: string;
    instructions: string;
    codeEdit: string;
    credentials?: Record<string, string> | undefined;
    config?: Record<string, unknown> | undefined;
    morphModel?: string | undefined;
}, {
    initialCode: string;
    instructions: string;
    codeEdit: string;
    credentials?: Record<string, string> | undefined;
    config?: Record<string, unknown> | undefined;
    morphModel?: string | undefined;
}>;
/**
 * Type definitions derived from schemas
 */
type EditBubbleFlowToolParams = z.output<typeof EditBubbleFlowToolParamsSchema>;
type EditBubbleFlowToolResult = z.output<typeof EditBubbleFlowToolResultSchema>;
/**
 * Define the result schema
 * This schema defines what the edit tool returns
 */
declare const EditBubbleFlowToolResultSchema: z.ZodObject<{
    mergedCode: z.ZodString;
    applied: z.ZodBoolean;
    diff: z.ZodOptional<z.ZodString>;
    metadata: z.ZodObject<{
        editedAt: z.ZodString;
        originalLength: z.ZodNumber;
        finalLength: z.ZodNumber;
        morphModel: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        originalLength: number;
        morphModel: string;
        editedAt: string;
        finalLength: number;
    }, {
        originalLength: number;
        morphModel: string;
        editedAt: string;
        finalLength: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata: {
        originalLength: number;
        morphModel: string;
        editedAt: string;
        finalLength: number;
    };
    mergedCode: string;
    applied: boolean;
    diff?: string | undefined;
}, {
    error: string;
    success: boolean;
    metadata: {
        originalLength: number;
        morphModel: string;
        editedAt: string;
        finalLength: number;
    };
    mergedCode: string;
    applied: boolean;
    diff?: string | undefined;
}>;
/**
 * Edit BubbleFlow Tool
 * Applies code edits using Morph Fast Apply API via AIAgent
 */
export declare class EditBubbleFlowTool extends ToolBubble<EditBubbleFlowToolParams, EditBubbleFlowToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName = "code-edit-tool";
    static readonly schema: z.ZodObject<{
        initialCode: z.ZodEffects<z.ZodString, string, string>;
        instructions: z.ZodString;
        codeEdit: z.ZodEffects<z.ZodString, string, string>;
        morphModel: z.ZodOptional<z.ZodDefault<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        initialCode: string;
        instructions: string;
        codeEdit: string;
        credentials?: Record<string, string> | undefined;
        config?: Record<string, unknown> | undefined;
        morphModel?: string | undefined;
    }, {
        initialCode: string;
        instructions: string;
        codeEdit: string;
        credentials?: Record<string, string> | undefined;
        config?: Record<string, unknown> | undefined;
        morphModel?: string | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        mergedCode: z.ZodString;
        applied: z.ZodBoolean;
        diff: z.ZodOptional<z.ZodString>;
        metadata: z.ZodObject<{
            editedAt: z.ZodString;
            originalLength: z.ZodNumber;
            finalLength: z.ZodNumber;
            morphModel: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            originalLength: number;
            morphModel: string;
            editedAt: string;
            finalLength: number;
        }, {
            originalLength: number;
            morphModel: string;
            editedAt: string;
            finalLength: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata: {
            originalLength: number;
            morphModel: string;
            editedAt: string;
            finalLength: number;
        };
        mergedCode: string;
        applied: boolean;
        diff?: string | undefined;
    }, {
        error: string;
        success: boolean;
        metadata: {
            originalLength: number;
            morphModel: string;
            editedAt: string;
            finalLength: number;
        };
        mergedCode: string;
        applied: boolean;
        diff?: string | undefined;
    }>;
    static readonly shortDescription = "Applies code edits to BubbleFlow files using Morph Fast Apply";
    static readonly longDescription = "\n    A tool for intelligently applying code edits to BubbleFlow TypeScript files.\n    Uses the Morph Fast Apply API via OpenRouter to merge lazy code edits into existing code.\n\n    What it does:\n    - Merges code edits specified with \"// ... existing code ...\" markers\n    - Uses Morph's apply model for intelligent code merging\n    - Minimizes context repetition for efficient edits\n    - Returns the final merged code\n\n    How it works:\n    - Takes original code, edit instructions, and code edit as input\n    - Sends to Morph API via OpenRouter using HttpBubble\n    - Receives merged code from Morph's apply model\n    - Returns the final code ready to be written to file\n\n    Use cases:\n    - When an AI agent needs to make edits to BubbleFlow code\n    - When applying multiple distinct edits to a file at once\n    - When making targeted changes without rewriting entire files\n    - When following the Cursor Fast Apply pattern for code edits\n\n    Important:\n    - The codeEdit parameter should use \"// ... existing code ...\" to mark unchanged sections\n    - The instructions parameter should be generated by the model in first person\n    - Requires OPENROUTER_CRED credential for Morph API access via OpenRouter\n  ";
    static readonly alias = "code-edit";
    /**
     * Main action method - performs code edit merging
     */
    performAction(): Promise<EditBubbleFlowToolResult>;
}
export {};
//# sourceMappingURL=code-edit-tool.d.ts.map