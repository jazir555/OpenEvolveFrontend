import { z } from '@hono/zod-openapi';
import { BubbleParameterType } from './bubble-definition-schema.js';
import { CredentialType } from './types.js';
import type { BubbleName } from './types.js';
export declare const createBubbleFlowSchema: z.ZodObject<{
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    prompt: z.ZodOptional<z.ZodString>;
    code: z.ZodString;
    eventType: z.ZodString;
    webhookPath: z.ZodOptional<z.ZodString>;
    webhookActive: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    bubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
        variableId: z.ZodNumber;
        nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
        location: z.ZodObject<{
            startLine: z.ZodNumber;
            startCol: z.ZodNumber;
            endLine: z.ZodNumber;
            endCol: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }>;
        description: z.ZodOptional<z.ZodString>;
        invocationCallSiteKey: z.ZodOptional<z.ZodString>;
        clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
        isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
        containingCustomToolId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    name: string;
    eventType: string;
    description?: string | undefined;
    prompt?: string | undefined;
    webhookPath?: string | undefined;
    webhookActive?: boolean | undefined;
    bubbleParameters?: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
}, {
    code: string;
    name: string;
    eventType: string;
    description?: string | undefined;
    prompt?: string | undefined;
    webhookPath?: string | undefined;
    webhookActive?: boolean | undefined;
    bubbleParameters?: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
}>;
export declare const createEmptyBubbleFlowSchema: z.ZodObject<{
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    prompt: z.ZodString;
    eventType: z.ZodString;
    webhookPath: z.ZodOptional<z.ZodString>;
    webhookActive: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    prompt: string;
    eventType: string;
    description?: string | undefined;
    webhookPath?: string | undefined;
    webhookActive?: boolean | undefined;
}, {
    name: string;
    prompt: string;
    eventType: string;
    description?: string | undefined;
    webhookPath?: string | undefined;
    webhookActive?: boolean | undefined;
}>;
export declare const executeBubbleFlowSchema: z.ZodRecord<z.ZodString, z.ZodUnknown>;
export declare const updateBubbleFlowParametersSchema: z.ZodObject<{
    bubbleParameters: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
        variableId: z.ZodNumber;
        nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
        location: z.ZodObject<{
            startLine: z.ZodNumber;
            startCol: z.ZodNumber;
            endLine: z.ZodNumber;
            endCol: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }>;
        description: z.ZodOptional<z.ZodString>;
        invocationCallSiteKey: z.ZodOptional<z.ZodString>;
        clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
        isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
        containingCustomToolId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
    }, "strip", z.ZodTypeAny, {
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
    }, {
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
    }>]>>;
}, "strip", z.ZodTypeAny, {
    bubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
    } | {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
}, {
    bubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
    } | {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
}>;
export declare const updateBubbleFlowNameSchema: z.ZodObject<{
    name: z.ZodString;
}, "strip", z.ZodTypeAny, {
    name: string;
}, {
    name: string;
}>;
export declare const createBubbleFlowResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    message: z.ZodString;
    inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    bubbleParameters: z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
        variableId: z.ZodNumber;
        nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
        location: z.ZodObject<{
            startLine: z.ZodNumber;
            startCol: z.ZodNumber;
            endLine: z.ZodNumber;
            endCol: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }>;
        description: z.ZodOptional<z.ZodString>;
        invocationCallSiteKey: z.ZodOptional<z.ZodString>;
        clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
        isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
        containingCustomToolId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>;
    workflow: z.ZodOptional<z.ZodObject<{
        root: z.ZodArray<z.ZodType<import("./bubble-definition-schema.js").WorkflowNode, z.ZodTypeDef, import("./bubble-definition-schema.js").WorkflowNode>, "many">;
        bubbles: z.ZodRecord<z.ZodNumber, z.ZodObject<{
            variableName: z.ZodString;
            bubbleName: z.ZodType<BubbleName>;
            className: z.ZodString;
            parameters: z.ZodArray<z.ZodObject<{
                location: z.ZodOptional<z.ZodObject<{
                    startLine: z.ZodNumber;
                    startCol: z.ZodNumber;
                    endLine: z.ZodNumber;
                    endCol: z.ZodNumber;
                }, "strip", z.ZodTypeAny, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }>>;
                variableId: z.ZodOptional<z.ZodNumber>;
                name: z.ZodString;
                value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
                type: z.ZodNativeEnum<typeof BubbleParameterType>;
                source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
            }, "strip", z.ZodTypeAny, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }>, "many">;
            hasAwait: z.ZodBoolean;
            hasActionCall: z.ZodBoolean;
            dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
            dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
            variableId: z.ZodNumber;
            nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
            location: z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>;
            description: z.ZodOptional<z.ZodString>;
            invocationCallSiteKey: z.ZodOptional<z.ZodString>;
            clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
            isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
            containingCustomToolId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }, {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }>>;
    requiredCredentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>>;
    eventType: z.ZodString;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        url: z.ZodString;
        path: z.ZodString;
        active: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    id: number;
    eventType: string;
    bubbleParameters: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
    workflow?: {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
}, {
    message: string;
    id: number;
    eventType: string;
    bubbleParameters: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
    workflow?: {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
}>;
export declare const createEmptyBubbleFlowResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    message: z.ZodString;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        url: z.ZodString;
        path: z.ZodString;
        active: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    id: number;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
}, {
    message: string;
    id: number;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
}>;
export declare const bubbleFlowDetailsResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    prompt: z.ZodOptional<z.ZodString>;
    eventType: z.ZodString;
    code: z.ZodString;
    originalCode: z.ZodOptional<z.ZodString>;
    generationError: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    cron: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    cronActive: z.ZodOptional<z.ZodBoolean>;
    defaultInputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    isActive: z.ZodBoolean;
    requiredCredentials: z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>;
    displayedBubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodString;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodUnknown;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
        }, "strip", z.ZodTypeAny, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>>>;
    bubbleParameters: z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
        variableId: z.ZodNumber;
        nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
        location: z.ZodObject<{
            startLine: z.ZodNumber;
            startCol: z.ZodNumber;
            endLine: z.ZodNumber;
            endCol: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }>;
        description: z.ZodOptional<z.ZodString>;
        invocationCallSiteKey: z.ZodOptional<z.ZodString>;
        clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
        isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
        containingCustomToolId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>;
    workflow: z.ZodOptional<z.ZodObject<{
        root: z.ZodArray<z.ZodType<import("./bubble-definition-schema.js").WorkflowNode, z.ZodTypeDef, import("./bubble-definition-schema.js").WorkflowNode>, "many">;
        bubbles: z.ZodRecord<z.ZodNumber, z.ZodObject<{
            variableName: z.ZodString;
            bubbleName: z.ZodType<BubbleName>;
            className: z.ZodString;
            parameters: z.ZodArray<z.ZodObject<{
                location: z.ZodOptional<z.ZodObject<{
                    startLine: z.ZodNumber;
                    startCol: z.ZodNumber;
                    endLine: z.ZodNumber;
                    endCol: z.ZodNumber;
                }, "strip", z.ZodTypeAny, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }>>;
                variableId: z.ZodOptional<z.ZodNumber>;
                name: z.ZodString;
                value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
                type: z.ZodNativeEnum<typeof BubbleParameterType>;
                source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
            }, "strip", z.ZodTypeAny, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }>, "many">;
            hasAwait: z.ZodBoolean;
            hasActionCall: z.ZodBoolean;
            dependencies: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
            dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
            variableId: z.ZodNumber;
            nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
            location: z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>;
            description: z.ZodOptional<z.ZodString>;
            invocationCallSiteKey: z.ZodOptional<z.ZodString>;
            clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
            isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
            containingCustomToolId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }, {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
    webhook_url: z.ZodString;
}, "strip", z.ZodTypeAny, {
    code: string;
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    webhook_url: string;
    eventType: string;
    requiredCredentials: Record<string, CredentialType[]>;
    bubbleParameters: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
    isActive: boolean;
    description?: string | undefined;
    workflow?: {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    metadata?: Record<string, unknown> | undefined;
    prompt?: string | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    cron?: string | null | undefined;
    cronActive?: boolean | undefined;
    originalCode?: string | undefined;
    generationError?: string | null | undefined;
    displayedBubbleParameters?: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }> | undefined;
}, {
    code: string;
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    webhook_url: string;
    eventType: string;
    requiredCredentials: Record<string, CredentialType[]>;
    bubbleParameters: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
    isActive: boolean;
    description?: string | undefined;
    workflow?: {
        root: import("./bubble-definition-schema.js").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    metadata?: Record<string, unknown> | undefined;
    prompt?: string | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    cron?: string | null | undefined;
    cronActive?: boolean | undefined;
    originalCode?: string | undefined;
    generationError?: string | null | undefined;
    displayedBubbleParameters?: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }> | undefined;
}>;
export declare const bubbleFlowListItemSchema: z.ZodObject<{
    id: z.ZodNumber;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    eventType: z.ZodString;
    isActive: z.ZodBoolean;
    cronActive: z.ZodBoolean;
    cronSchedule: z.ZodOptional<z.ZodString>;
    webhookExecutionCount: z.ZodNumber;
    webhookFailureCount: z.ZodNumber;
    executionCount: z.ZodNumber;
    bubbles: z.ZodOptional<z.ZodArray<z.ZodObject<{
        bubbleName: z.ZodType<BubbleName>;
        className: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        className: string;
        bubbleName: BubbleName;
    }, {
        className: string;
        bubbleName: BubbleName;
    }>, "many">>;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    eventType: string;
    cronActive: boolean;
    isActive: boolean;
    webhookExecutionCount: number;
    webhookFailureCount: number;
    executionCount: number;
    description?: string | undefined;
    bubbles?: {
        className: string;
        bubbleName: BubbleName;
    }[] | undefined;
    cronSchedule?: string | undefined;
}, {
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    eventType: string;
    cronActive: boolean;
    isActive: boolean;
    webhookExecutionCount: number;
    webhookFailureCount: number;
    executionCount: number;
    description?: string | undefined;
    bubbles?: {
        className: string;
        bubbleName: BubbleName;
    }[] | undefined;
    cronSchedule?: string | undefined;
}>;
export declare const bubbleFlowListResponseSchema: z.ZodObject<{
    bubbleFlows: z.ZodDefault<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        eventType: z.ZodString;
        isActive: z.ZodBoolean;
        cronActive: z.ZodBoolean;
        cronSchedule: z.ZodOptional<z.ZodString>;
        webhookExecutionCount: z.ZodNumber;
        webhookFailureCount: z.ZodNumber;
        executionCount: z.ZodNumber;
        bubbles: z.ZodOptional<z.ZodArray<z.ZodObject<{
            bubbleName: z.ZodType<BubbleName>;
            className: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            className: string;
            bubbleName: BubbleName;
        }, {
            className: string;
            bubbleName: BubbleName;
        }>, "many">>;
        createdAt: z.ZodString;
        updatedAt: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: number;
        createdAt: string;
        updatedAt: string;
        eventType: string;
        cronActive: boolean;
        isActive: boolean;
        webhookExecutionCount: number;
        webhookFailureCount: number;
        executionCount: number;
        description?: string | undefined;
        bubbles?: {
            className: string;
            bubbleName: BubbleName;
        }[] | undefined;
        cronSchedule?: string | undefined;
    }, {
        name: string;
        id: number;
        createdAt: string;
        updatedAt: string;
        eventType: string;
        cronActive: boolean;
        isActive: boolean;
        webhookExecutionCount: number;
        webhookFailureCount: number;
        executionCount: number;
        description?: string | undefined;
        bubbles?: {
            className: string;
            bubbleName: BubbleName;
        }[] | undefined;
        cronSchedule?: string | undefined;
    }>, "many">>;
    userMonthlyUsage: z.ZodObject<{
        count: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        count: number;
    }, {
        count: number;
    }>;
}, "strip", z.ZodTypeAny, {
    bubbleFlows: {
        name: string;
        id: number;
        createdAt: string;
        updatedAt: string;
        eventType: string;
        cronActive: boolean;
        isActive: boolean;
        webhookExecutionCount: number;
        webhookFailureCount: number;
        executionCount: number;
        description?: string | undefined;
        bubbles?: {
            className: string;
            bubbleName: BubbleName;
        }[] | undefined;
        cronSchedule?: string | undefined;
    }[];
    userMonthlyUsage: {
        count: number;
    };
}, {
    userMonthlyUsage: {
        count: number;
    };
    bubbleFlows?: {
        name: string;
        id: number;
        createdAt: string;
        updatedAt: string;
        eventType: string;
        cronActive: boolean;
        isActive: boolean;
        webhookExecutionCount: number;
        webhookFailureCount: number;
        executionCount: number;
        description?: string | undefined;
        bubbles?: {
            className: string;
            bubbleName: BubbleName;
        }[] | undefined;
        cronSchedule?: string | undefined;
    }[] | undefined;
}>;
export declare const activateBubbleFlowResponseSchema: z.ZodObject<{
    success: z.ZodBoolean;
    webhookUrl: z.ZodString;
    message: z.ZodString;
}, "strip", z.ZodTypeAny, {
    message: string;
    success: boolean;
    webhookUrl: string;
}, {
    message: string;
    success: boolean;
    webhookUrl: string;
}>;
export type ActivateBubbleFlowResponse = z.infer<typeof activateBubbleFlowResponseSchema>;
export type CreateBubbleFlowResponse = z.infer<typeof createBubbleFlowResponseSchema>;
export type CreateBubbleFlowRequest = z.infer<typeof createBubbleFlowSchema>;
export type CreateEmptyBubbleFlowRequest = z.infer<typeof createEmptyBubbleFlowSchema>;
export type CreateEmptyBubbleFlowResponse = z.infer<typeof createEmptyBubbleFlowResponseSchema>;
export type ExecuteBubbleFlowRequest = z.infer<typeof executeBubbleFlowSchema>;
export type UpdateBubbleFlowParametersRequest = z.infer<typeof updateBubbleFlowParametersSchema>;
export type UpdateBubbleFlowParametersResponse = z.infer<typeof updateBubbleFlowParametersSchema>;
export type UpdateBubbleFlowNameRequest = z.infer<typeof updateBubbleFlowNameSchema>;
export type BubbleFlowDetailsResponse = z.infer<typeof bubbleFlowDetailsResponseSchema>;
export type BubbleFlowListResponse = z.infer<typeof bubbleFlowListResponseSchema>;
export type BubbleFlowListItem = z.infer<typeof bubbleFlowListItemSchema>;
//# sourceMappingURL=bubbleflow-schema.d.ts.map