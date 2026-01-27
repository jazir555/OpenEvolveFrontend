import { z } from 'zod';
import { CredentialType, BubbleName } from './types';
export declare enum BubbleParameterType {
    STRING = "string",
    NUMBER = "number",
    BOOLEAN = "boolean",
    OBJECT = "object",
    ARRAY = "array",
    ENV = "env",
    VARIABLE = "variable",
    EXPRESSION = "expression",
    UNKNOWN = "unknown"
}
export declare const CREDENTIAL_CONFIGURATION_MAP: Record<CredentialType, Record<string, BubbleParameterType>>;
export declare const BUBBLE_NAMES_WITH_CONTEXT_INJECTION: string[];
export declare const BubbleParameterTypeSchema: z.ZodNativeEnum<typeof BubbleParameterType>;
export declare const BubbleParameterSchema: z.ZodObject<{
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
    /**
     * Source of the parameter - indicates whether it came from an object literal property
     * or represents the entire first argument. Used to determine if spread pattern should be applied.
     * Ex.
     * const abc = '1234567890';
     * new GoogleDriveBubble({
     *   fileId: abc,
     * })
     * source: 'object-property',
     *
     * new GoogleDriveBubble({
     *   url: 'https://www.google.com',
     *   ...args,
     * })
     * source: 'spread',
     *
     * source = 'first-arg'
     * new GoogleDriveBubble(args)
     */
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
}>;
export type BubbleParameter = z.infer<typeof BubbleParameterSchema>;
export interface ParsedBubble {
    variableName: string;
    bubbleName: BubbleName;
    className: string;
    parameters: BubbleParameter[];
    hasAwait: boolean;
    hasActionCall: boolean;
    dependencies?: BubbleName[];
    dependencyGraph?: DependencyGraphNode;
}
export interface DependencyGraphNode {
    name: BubbleName;
    /** Optional variable name for this node instance, when available */
    variableName?: string;
    nodeType: BubbleNodeType;
    /**
     * Unique hierarchical ID path for the node within a flow.
     * Constructed as parentUniqueId + "." + bubbleName + "#" + ordinal.
     * Root nodes can omit or use empty string for the parent portion.
     */
    uniqueId?: string;
    /**
     * Variable id assigned by the parser/scope manager if available.
     * Root bubble nodes will carry their declaration variable id; synthetic/child nodes
     * inferred from dependencies may be assigned a negative synthetic id.
     */
    variableId?: number;
    dependencies: DependencyGraphNode[];
    /**
     * Custom tool functions parsed as FunctionCallWorkflowNode.
     * Used when an ai-agent has customTools with func properties containing bubble instantiations.
     */
    functionCallChildren?: FunctionCallWorkflowNode[];
}
export interface BubbleDependencySpec {
    name: BubbleName;
    tools?: BubbleName[];
}
export type BubbleNodeType = 'service' | 'tool' | 'workflow' | 'unknown';
export declare const BubbleNodeTypeSchema: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
export declare const DependencyGraphNodeSchema: z.ZodType<DependencyGraphNode>;
export declare const ParsedBubbleSchema: z.ZodObject<{
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
        /**
         * Source of the parameter - indicates whether it came from an object literal property
         * or represents the entire first argument. Used to determine if spread pattern should be applied.
         * Ex.
         * const abc = '1234567890';
         * new GoogleDriveBubble({
         *   fileId: abc,
         * })
         * source: 'object-property',
         *
         * new GoogleDriveBubble({
         *   url: 'https://www.google.com',
         *   ...args,
         * })
         * source: 'spread',
         *
         * source = 'first-arg'
         * new GoogleDriveBubble(args)
         */
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
    dependencyGraph: z.ZodOptional<z.ZodType<DependencyGraphNode, z.ZodTypeDef, DependencyGraphNode>>;
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
    dependencyGraph?: DependencyGraphNode | undefined;
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
    dependencyGraph?: DependencyGraphNode | undefined;
}>;
export declare const BubbleDependencySpecSchema: z.ZodObject<{
    name: z.ZodType<BubbleName>;
    tools: z.ZodOptional<z.ZodArray<z.ZodType<BubbleName, z.ZodTypeDef, BubbleName>, "many">>;
}, "strip", z.ZodTypeAny, {
    name: BubbleName;
    tools?: BubbleName[] | undefined;
}, {
    name: BubbleName;
    tools?: BubbleName[] | undefined;
}>;
export declare const ParsedBubbleWithInfoSchema: z.ZodObject<{
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
        /**
         * Source of the parameter - indicates whether it came from an object literal property
         * or represents the entire first argument. Used to determine if spread pattern should be applied.
         * Ex.
         * const abc = '1234567890';
         * new GoogleDriveBubble({
         *   fileId: abc,
         * })
         * source: 'object-property',
         *
         * new GoogleDriveBubble({
         *   url: 'https://www.google.com',
         *   ...args,
         * })
         * source: 'spread',
         *
         * source = 'first-arg'
         * new GoogleDriveBubble(args)
         */
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
    dependencyGraph: z.ZodOptional<z.ZodType<DependencyGraphNode, z.ZodTypeDef, DependencyGraphNode>>;
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
    dependencyGraph?: DependencyGraphNode | undefined;
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
    dependencyGraph?: DependencyGraphNode | undefined;
    invocationCallSiteKey?: string | undefined;
    clonedFromVariableId?: number | undefined;
    isInsideCustomTool?: boolean | undefined;
    containingCustomToolId?: string | undefined;
}>;
export type ParsedBubbleWithInfo = z.infer<typeof ParsedBubbleWithInfoSchema>;
export type BubbleParameterTypeInferred = z.infer<typeof BubbleParameterTypeSchema>;
export type BubbleParameterInferred = BubbleParameter;
export type BubbleNodeTypeInferred = z.infer<typeof BubbleNodeTypeSchema>;
export type DependencyGraphNodeInferred = z.infer<typeof DependencyGraphNodeSchema>;
export type ParsedBubbleInferred = z.infer<typeof ParsedBubbleSchema>;
export type BubbleDependencySpecInferred = z.infer<typeof BubbleDependencySpecSchema>;
export type ParsedBubbleWithInfoInferred = z.infer<typeof ParsedBubbleWithInfoSchema>;
export type WorkflowNodeType = 'bubble' | 'if' | 'for' | 'while' | 'try_catch' | 'variable_declaration' | 'return' | 'function_call' | 'code_block' | 'parallel_execution' | 'transformation_function';
export interface BubbleWorkflowNode {
    type: 'bubble';
    variableId: number;
}
export interface ControlFlowWorkflowNode {
    type: 'if' | 'for' | 'while';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    condition?: string;
    children: WorkflowNode[];
    elseBranch?: WorkflowNode[];
    thenTerminates?: boolean;
    elseTerminates?: boolean;
}
export interface TryCatchWorkflowNode {
    type: 'try_catch';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    children: WorkflowNode[];
    catchBlock?: WorkflowNode[];
}
export interface CodeBlockWorkflowNode {
    type: 'code_block';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    code: string;
    children: WorkflowNode[];
}
export interface VariableDeclarationBlockNode {
    type: 'variable_declaration';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    code: string;
    variables: Array<{
        name: string;
        type: 'const' | 'let' | 'var';
        hasInitializer: boolean;
    }>;
    children: WorkflowNode[];
}
export interface ReturnWorkflowNode {
    type: 'return';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    code: string;
    value?: string;
    children: WorkflowNode[];
}
export interface FunctionCallWorkflowNode {
    type: 'function_call';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    functionName: string;
    isMethodCall: boolean;
    description?: string;
    arguments?: string;
    code: string;
    variableId: number;
    variableDeclaration?: {
        variableName: string;
        variableType: 'const' | 'let' | 'var';
    };
    methodDefinition?: {
        location: {
            startLine: number;
            endLine: number;
        };
        isAsync: boolean;
        parameters: string[];
    };
    children: WorkflowNode[];
}
export interface ParallelExecutionWorkflowNode {
    type: 'parallel_execution';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    code: string;
    variableDeclaration?: {
        variableNames: string[];
        variableType: 'const' | 'let' | 'var';
    };
    children: WorkflowNode[];
}
export interface TransformationFunctionWorkflowNode {
    type: 'transformation_function';
    location: {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    };
    code: string;
    functionName: string;
    isMethodCall: boolean;
    description?: string;
    arguments?: string;
    variableId: number;
    variableDeclaration?: {
        variableName: string;
        variableType: 'const' | 'let' | 'var';
    };
    methodDefinition?: {
        location: {
            startLine: number;
            endLine: number;
        };
        isAsync: boolean;
        parameters: string[];
    };
}
export type WorkflowNode = BubbleWorkflowNode | ControlFlowWorkflowNode | TryCatchWorkflowNode | CodeBlockWorkflowNode | VariableDeclarationBlockNode | ReturnWorkflowNode | FunctionCallWorkflowNode | ParallelExecutionWorkflowNode | TransformationFunctionWorkflowNode;
export interface ParsedWorkflow {
    root: WorkflowNode[];
    bubbles: Record<number, ParsedBubbleWithInfo>;
}
export declare const WorkflowNodeTypeSchema: z.ZodEnum<["bubble", "if", "for", "while", "try_catch", "variable_declaration", "return", "function_call", "code_block", "parallel_execution", "transformation_function"]>;
export declare const LocationSchema: z.ZodObject<{
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
export declare const BubbleWorkflowNodeSchema: z.ZodType<BubbleWorkflowNode>;
export declare const ControlFlowWorkflowNodeSchema: z.ZodType<ControlFlowWorkflowNode>;
export declare const TryCatchWorkflowNodeSchema: z.ZodType<TryCatchWorkflowNode>;
export declare const CodeBlockWorkflowNodeSchema: z.ZodType<CodeBlockWorkflowNode>;
export declare const VariableDeclarationBlockNodeSchema: z.ZodType<VariableDeclarationBlockNode>;
export declare const ReturnWorkflowNodeSchema: z.ZodType<ReturnWorkflowNode>;
export declare const FunctionCallWorkflowNodeSchema: z.ZodType<FunctionCallWorkflowNode>;
export declare const ParallelExecutionWorkflowNodeSchema: z.ZodType<ParallelExecutionWorkflowNode>;
export declare const TransformationFunctionWorkflowNodeSchema: z.ZodType<TransformationFunctionWorkflowNode>;
export declare const WorkflowNodeSchema: z.ZodType<WorkflowNode>;
export declare const ParsedWorkflowSchema: z.ZodObject<{
    root: z.ZodArray<z.ZodType<WorkflowNode, z.ZodTypeDef, WorkflowNode>, "many">;
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
            /**
             * Source of the parameter - indicates whether it came from an object literal property
             * or represents the entire first argument. Used to determine if spread pattern should be applied.
             * Ex.
             * const abc = '1234567890';
             * new GoogleDriveBubble({
             *   fileId: abc,
             * })
             * source: 'object-property',
             *
             * new GoogleDriveBubble({
             *   url: 'https://www.google.com',
             *   ...args,
             * })
             * source: 'spread',
             *
             * source = 'first-arg'
             * new GoogleDriveBubble(args)
             */
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
        dependencyGraph: z.ZodOptional<z.ZodType<DependencyGraphNode, z.ZodTypeDef, DependencyGraphNode>>;
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
        dependencyGraph?: DependencyGraphNode | undefined;
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
        dependencyGraph?: DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    root: WorkflowNode[];
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
        dependencyGraph?: DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
}, {
    root: WorkflowNode[];
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
        dependencyGraph?: DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>;
}>;
//# sourceMappingURL=bubble-definition-schema.d.ts.map