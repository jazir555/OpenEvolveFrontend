/**
 * TOOL BUBBLE TEMPLATE
 *
 * This template provides a starting point for creating new tool bubbles in the NodeX system.
 * Tool bubbles are specialized bubbles that provide utility functions for other bubbles,
 * particularly useful for AI agents that need to perform specific operations.
 *
 * NEW FEATURES (v2):
 * - Automatic credential injection via base class
 * - Config parameter support for runtime configuration
 * - No need to implement toAgentTool() - handled automatically
 * - IMPORTANT: credentials and config are HIDDEN from AI agents
 *   - AI agents only see your actual tool parameters (inputData, options, etc.)
 *   - credentials and config are injected at runtime and available in performAction()
 *   - This keeps the tool interface clean for AI while providing access to secrets
 *
 * To create a new tool bubble:
 * 1. Copy this template and rename it (e.g., my-custom-tool.ts)
 * 2. Replace all instances of "MyCustomTool" with your tool name
 * 3. Update the schema to define your input parameters
 * 4. Add credentials/config fields if needed (see examples below)
 * 5. Update the result schema to define your output structure
 * 6. Implement the performAction method with your tool logic
 * 7. Update all static metadata (bubbleName, descriptions, etc.)
 * 8. Register your tool in the BubbleFactory
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Define the parameters schema using Zod
 * This schema validates and types the input parameters for your tool
 *
 * Common patterns:
 * - Required string: z.string().min(1, 'Field is required')
 * - Optional string: z.string().optional()
 * - Number with range: z.number().min(0).max(100)
 * - Enum: z.enum(['option1', 'option2'])
 * - Array: z.array(z.string())
 * - Object: z.object({ key: z.string() })
 * - Credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional()
 */
declare const MyCustomToolParamsSchema: z.ZodObject<{
    inputData: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        includeDetails: z.ZodDefault<z.ZodBoolean>;
        maxResults: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        maxResults: number;
        includeDetails: boolean;
    }, {
        maxResults?: number | undefined;
        includeDetails?: boolean | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    inputData: string;
    options?: {
        maxResults: number;
        includeDetails: boolean;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
}, {
    inputData: string;
    options?: {
        maxResults?: number | undefined;
        includeDetails?: boolean | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
}>;
/**
 * Type definitions derived from schemas
 * These provide TypeScript types for compile-time type safety
 */
type MyCustomToolParamsInput = z.input<typeof MyCustomToolParamsSchema>;
type MyCustomToolParams = z.output<typeof MyCustomToolParamsSchema>;
type MyCustomToolResult = z.output<typeof MyCustomToolResultSchema>;
/**
 * Define the result schema
 * This schema defines what your tool returns
 * Always include success and error fields for consistent error handling
 */
declare const MyCustomToolResultSchema: z.ZodObject<{
    processedData: z.ZodString;
    metadata: z.ZodObject<{
        processedAt: z.ZodString;
        itemsProcessed: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processedAt: string;
        itemsProcessed: number;
    }, {
        processedAt: string;
        itemsProcessed: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata: {
        processedAt: string;
        itemsProcessed: number;
    };
    processedData: string;
}, {
    error: string;
    success: boolean;
    metadata: {
        processedAt: string;
        itemsProcessed: number;
    };
    processedData: string;
}>;
/**
 * Main tool class implementation
 * Extends ToolBubble with your parameter and result types
 */
export declare class MyCustomTool extends ToolBubble<MyCustomToolParams, MyCustomToolResult> {
    /**
     * REQUIRED STATIC METADATA
     * These fields are used by the BubbleFactory and AI agents
     */
    static readonly type: "tool";
    static readonly bubbleName = "my-custom-tool";
    static readonly schema: z.ZodObject<{
        inputData: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            includeDetails: z.ZodDefault<z.ZodBoolean>;
            maxResults: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            maxResults: number;
            includeDetails: boolean;
        }, {
            maxResults?: number | undefined;
            includeDetails?: boolean | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        inputData: string;
        options?: {
            maxResults: number;
            includeDetails: boolean;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }, {
        inputData: string;
        options?: {
            maxResults?: number | undefined;
            includeDetails?: boolean | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        processedData: z.ZodString;
        metadata: z.ZodObject<{
            processedAt: z.ZodString;
            itemsProcessed: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processedAt: string;
            itemsProcessed: number;
        }, {
            processedAt: string;
            itemsProcessed: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata: {
            processedAt: string;
            itemsProcessed: number;
        };
        processedData: string;
    }, {
        error: string;
        success: boolean;
        metadata: {
            processedAt: string;
            itemsProcessed: number;
        };
        processedData: string;
    }>;
    static readonly shortDescription = "Brief description of what your tool does";
    static readonly longDescription = "\n    A comprehensive description of your tool bubble.\n    \n    What it does:\n    - Main functionality point 1\n    - Main functionality point 2\n    \n    How it works:\n    - Implementation detail 1\n    - Implementation detail 2\n    \n    Use cases:\n    - When an AI agent needs to perform X\n    - When processing Y type of data\n    - When integrating with Z system\n  ";
    static readonly alias = "custom";
    /**
     * Constructor
     * Initialize your tool with parameters and optional context
     */
    constructor(params: MyCustomToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - this is where your tool logic goes
     * This method is called when the tool is executed
     *
     * @param context - Optional bubble context (contains metadata, credentials, etc.)
     * @returns Promise with your result object
     */
    performAction(context?: BubbleContext): Promise<MyCustomToolResult>;
    /**
     * PRIVATE HELPER METHODS
     * Add your internal logic methods here
     */
    private processData;
}
export {};
/**
 * REGISTRATION AND USAGE
 *
 * 1. Register your tool in the BubbleFactory:
 * ```typescript
 * import { MyCustomTool } from './bubbles/tool-bubble/my-custom-tool.js';
 *
 * // In registerDefaults() method:
 * this.register('my-custom-tool', MyCustomTool as BubbleClassWithMetadata);
 * ```
 *
 * 2. Direct usage:
 * ```typescript
 * const tool = new MyCustomTool({
 *   inputData: 'some data',
 *   options: { includeDetails: true },
 *   credentials: { DATABASE_CRED: 'connection_string' }
 * });
 * const result = await tool.action();
 * ```
 *
 * 3. AI Agent usage:
 * ```typescript
 * // When AI agent calls your tool, it ONLY sees these parameters:
 * {
 *   "inputData": "some data",
 *   "options": { "includeDetails": true }
 * }
 *
 * // The AI agent configuration provides credentials/config separately:
 * tools: [{
 *   name: 'my-custom-tool',
 *   credentials: { DATABASE_CRED: 'connection_string' },  // Runtime injection
 *   config: { someOption: true }  // Runtime injection
 * }]
 * ```
 *
 * IMPORTANT SECURITY FEATURE:
 * - AI agents NEVER see credentials or config in the tool schema
 * - The base ToolBubble class automatically strips these from the AI-visible schema
 * - Runtime injects credentials/config before calling performAction()
 * - This prevents AI from accidentally exposing or misusing secrets
 */
//# sourceMappingURL=tool-template.d.ts.map