import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
declare const ListBubblesToolParamsSchema: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
type ListBubblesToolParamsInput = z.input<typeof ListBubblesToolParamsSchema>;
type ListBubblesToolParams = z.output<typeof ListBubblesToolParamsSchema>;
type ListBubblesToolResult = z.output<typeof ListBubblesToolResultSchema>;
declare const ListBubblesToolResultSchema: z.ZodObject<{
    bubbles: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        alias: z.ZodOptional<z.ZodString>;
        shortDescription: z.ZodString;
        useCase: z.ZodString;
        type: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: string;
        name: string;
        shortDescription: string;
        useCase: string;
        alias?: string | undefined;
    }, {
        type: string;
        name: string;
        shortDescription: string;
        useCase: string;
        alias?: string | undefined;
    }>, "many">;
    totalCount: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    bubbles: {
        type: string;
        name: string;
        shortDescription: string;
        useCase: string;
        alias?: string | undefined;
    }[];
    totalCount: number;
}, {
    error: string;
    success: boolean;
    bubbles: {
        type: string;
        name: string;
        shortDescription: string;
        useCase: string;
        alias?: string | undefined;
    }[];
    totalCount: number;
}>;
export declare class ListBubblesTool extends ToolBubble<ListBubblesToolParams, ListBubblesToolResult> {
    static readonly bubbleName = "list-bubbles-tool";
    static readonly schema: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
    static readonly resultSchema: z.ZodObject<{
        bubbles: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            alias: z.ZodOptional<z.ZodString>;
            shortDescription: z.ZodString;
            useCase: z.ZodString;
            type: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            shortDescription: string;
            useCase: string;
            alias?: string | undefined;
        }, {
            type: string;
            name: string;
            shortDescription: string;
            useCase: string;
            alias?: string | undefined;
        }>, "many">;
        totalCount: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        bubbles: {
            type: string;
            name: string;
            shortDescription: string;
            useCase: string;
            alias?: string | undefined;
        }[];
        totalCount: number;
    }, {
        error: string;
        success: boolean;
        bubbles: {
            type: string;
            name: string;
            shortDescription: string;
            useCase: string;
            alias?: string | undefined;
        }[];
        totalCount: number;
    }>;
    static readonly shortDescription = "Lists all available bubbles in the registry";
    static readonly longDescription = "\n    A tool bubble that provides a comprehensive list of all registered bubbles in the NodeX system.\n    \n    Returns information about each bubble including:\n    - Bubble name and alias\n    - Short description\n    - Extracted use cases\n    - Bubble type (service, workflow, tool)\n    \n    Use cases:\n    - AI agent discovery of available capabilities\n    - System introspection and documentation\n    - Dynamic tool selection for workflow building\n  ";
    static readonly alias = "list";
    static readonly type = "tool";
    constructor(params?: ListBubblesToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<ListBubblesToolResult>;
    private extractUseCaseFromDescription;
}
export {};
//# sourceMappingURL=list-bubbles-tool.d.ts.map