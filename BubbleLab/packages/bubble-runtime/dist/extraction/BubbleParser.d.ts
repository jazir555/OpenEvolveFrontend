import type { TSESTree } from '@typescript-eslint/typescript-estree';
import type { ScopeManager } from '@bubblelab/ts-scope-manager';
import { BubbleFactory } from '@bubblelab/bubble-core';
import type { MethodInvocationInfo } from '../parse/BubbleScript';
import type { ParsedBubbleWithInfo, ParsedWorkflow } from '@bubblelab/shared-schemas';
export declare class BubbleParser {
    private bubbleScript;
    private cachedAST;
    private methodInvocationOrdinalMap;
    private invocationBubbleCloneCache;
    /**
     * Track which call expressions have been assigned an invocation index.
     * Key: `methodName:startOffset` (using AST range start position)
     * Value: the assigned invocation index
     * This prevents double-counting when the same call site is processed multiple times
     * (e.g., once in .map() callback processing, again in Promise.all resolution)
     */
    private processedCallSiteIndexes;
    /** Custom tool func ranges for marking bubbles inside custom tools */
    private customToolFuncs;
    constructor(bubbleScript: string);
    /**
     * Parse bubble dependencies from an AST using the provided factory and scope manager
     */
    parseBubblesFromAST(bubbleFactory: BubbleFactory, ast: TSESTree.Program, scopeManager: ScopeManager): {
        bubbles: Record<number, ParsedBubbleWithInfo>;
        workflow: ParsedWorkflow;
        instanceMethodsLocation: Record<string, {
            startLine: number;
            endLine: number;
            definitionStartLine: number;
            bodyStartLine: number;
            invocationLines: MethodInvocationInfo[];
        }>;
    };
    private findDependenciesForBubble;
    private buildDependencyGraph;
    /**
     * Build a JSON Schema object for the payload parameter of the top-level `handle` entrypoint.
     * Supports primitives, arrays, unions (anyOf), intersections (allOf), type literals, and
     * same-file interfaces/type aliases. Interface `extends` are ignored for now.
     */
    getPayloadJsonSchema(ast: TSESTree.Program): Record<string, unknown> | null;
    /**
     * Find the actual Function/ArrowFunction node corresponding to the handle entrypoint.
     */
    private findHandleFunctionNode;
    private findHandleInClass;
    /** Extract defaults from object destructuring of the first handle parameter */
    private extractPayloadDefaultsFromHandle;
    private getFirstParamIdentifierName;
    /** Best-effort conversion of default expression to JSON-safe value */
    private evaluateDefaultExpressionToJSON;
    /** Convert a TS type AST node into a JSON Schema object */
    private tsTypeToJsonSchema;
    private extractTypeReferenceName;
    private objectTypeToJsonSchema;
    private eventKeyToSchema;
    /** Resolve in-file interface/type alias by name to JSON Schema */
    private resolveTypeNameToJson;
    /**
     * Find the main class that extends BubbleFlow
     */
    private findMainBubbleFlowClass;
    /**
     * Extract all instance methods from a class
     */
    private findAllInstanceMethods;
    /**
     * Find all method invocations in the AST with full details
     */
    private findMethodInvocations;
    /**
     * Check if a child node is in the condition/test part of a control flow statement
     * Returns true if the child is the test/discriminant expression, false if it's in the body
     */
    private isNodeInConditionPart;
    /**
     * Helper to recursively visit child nodes for finding invocations
     */
    private visitChildNodesForInvocations;
    /**
     * Recursively visit AST nodes to find bubble instantiations
     */
    private visitNode;
    /**
     * Find the Variable object corresponding to a bubble declaration
     */
    private findVariableForBubble;
    /**
     * Add variable ID references to parameters that are variables
     */
    private addVariableReferencesToParameters;
    /**
     * Extract base variable name from expressions like "prompts[i]", "result.data"
     */
    private extractBaseVariableName;
    /**
     * Find the Variable.$id for a variable name at a specific context line
     */
    private findVariableIdByName;
    /**
     * Extract bubble information from an expression node
     */
    private extractBubbleFromExpression;
    /**
     * Extract bubble information from a NewExpression node
     */
    private extractFromNewExpression;
    /**
     * Extract parameter value and type from an expression
     */
    private extractParameterValue;
    /**
     * Find custom tools in ai-agent bubbles and populate customToolFuncs.
     * This scans the AST for ai-agent instantiations and extracts custom tool func locations.
     */
    private findCustomToolsInAIAgentBubbles;
    /**
     * Mark bubbles that are inside custom tool funcs with isInsideCustomTool flag.
     */
    private markBubblesInsideCustomTools;
    /**
     * Extract comment/description for a node by looking at preceding comments
     **/
    private extractCommentForNode;
    /**
     * Extract JSDoc info including description and @canBeFile tag from a node's preceding comments.
     * The @canBeFile tag controls whether file upload is enabled for string fields in the UI.
     */
    private extractJSDocForNode;
    /**
     * Check if a list of workflow nodes contains a terminating statement (return/throw)
     * A branch terminates if its last statement is a return or throw
     */
    private branchTerminates;
    /**
     * Build hierarchical workflow structure from AST
     */
    private buildWorkflowTree;
    /**
     * Group consecutive nodes of the same type
     * - Consecutive variable_declaration nodes → merge into one
     * - Consecutive code_block nodes → merge into one
     * - return nodes are NOT grouped (each is a distinct exit point)
     */
    private groupConsecutiveNodes;
    /**
     * Merge a group of nodes of the same type into a single node
     */
    private mergeGroup;
    /**
     * Build a workflow node from an AST statement
     */
    private buildWorkflowNodeFromStatement;
    /**
     * Build an if node from IfStatement
     */
    private buildIfNode;
    /**
     * Build a for node from ForStatement/ForInStatement/ForOfStatement
     */
    private buildForNode;
    /**
     * Build a while node from WhileStatement
     */
    private buildWhileNode;
    /**
     * Build a try-catch node from TryStatement
     */
    private buildTryCatchNode;
    /**
     * Build a code block node from a statement
     */
    private buildCodeBlockNode;
    /**
     * Find a bubble in an expression by checking if it matches any parsed bubble
     */
    private findBubbleInExpression;
    /**
     * Extract the NewExpression from an expression, handling await, .action(), etc.
     */
    private extractNewExpression;
    /**
     * Build a variable declaration node from a VariableDeclaration statement
     */
    private buildVariableDeclarationNode;
    /**
     * Build a return node from a ReturnStatement
     */
    private buildReturnNode;
    /**
     * Detect if an expression is Promise.all([...]) or Promise.all(variable)
     */
    private detectPromiseAll;
    /**
     * Detect if an expression is a function call
     */
    private detectFunctionCall;
    /**
     * Find a method definition in the class by name
     */
    private findMethodDefinition;
    /**
     * Check if a workflow node tree contains any bubbles (recursively)
     */
    private containsBubbles;
    /**
     * Build a function call node from a function call expression
     */
    private buildFunctionCallNode;
    /**
     * Extract the body of a callback function (arrow or regular function expression)
     * Handles both block statements and concise arrow functions
     */
    private extractCallbackBody;
    /**
     * Find array elements from .push() calls or .map() callbacks
     * Handles both patterns:
     * - .push(): array.push(item1, item2, ...)
     * - .map(): const promises = items.map(item => this.processItem(item))
     */
    private findArrayElements;
    /**
     * Extract expression from callback function
     */
    private extractCallbackExpression;
    /**
     * Get elements from source array (literal or variable)
     */
    private getSourceArrayElements;
    /**
     * Find all return statements in a block statement
     */
    private findReturnStatements;
    /**
     * Build a parallel execution node from Promise.all()
     */
    private buildParallelExecutionNode;
    /**
     * Get the invocation index for a method call.
     * If the same call expression (identified by its AST range) has been processed before,
     * return the same index to avoid double-counting.
     *
     * @param methodName - The name of the method being called
     * @param callExprStartOffset - Optional start offset of the CallExpression in the source.
     *                              Used to deduplicate when the same call is processed multiple times
     *                              (e.g., .map() callback processing vs Promise.all resolution)
     */
    private getNextInvocationIndex;
    private cloneWorkflowNodesForInvocation;
    private cloneWorkflowNodeForInvocation;
    private ensureClonedBubbleForInvocation;
    private cloneBubbleForInvocation;
    private cloneDependencyGraphNodeForInvocation;
    /**
     * Extract condition string from a test expression
     */
    private extractConditionString;
    /**
     * Extract location from a node
     */
    private extractLocation;
}
//# sourceMappingURL=BubbleParser.d.ts.map