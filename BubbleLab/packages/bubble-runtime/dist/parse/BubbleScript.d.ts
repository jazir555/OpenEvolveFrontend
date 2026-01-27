import type { TSESTree } from '@typescript-eslint/typescript-estree';
import type { ScopeManager, Variable } from '@bubblelab/ts-scope-manager';
import { BubbleFactory } from '@bubblelab/bubble-core';
import type { ParsedBubbleWithInfo, BubbleTrigger, ParsedWorkflow } from '@bubblelab/shared-schemas';
/**
 * Detailed information about a method invocation captured during AST parsing
 */
export interface MethodInvocationInfo {
    lineNumber: number;
    endLineNumber: number;
    columnNumber: number;
    invocationIndex: number;
    hasAwait: boolean;
    arguments: string;
    statementType: 'variable_declaration' | 'assignment' | 'return' | 'simple' | 'condition_expression' | 'nested_call_expression';
    variableName?: string;
    variableType?: 'const' | 'let' | 'var';
    destructuringPattern?: string;
    context?: 'default' | 'promise_all_element';
    /** For condition_expression: the line where the containing statement (if/while/etc) starts */
    containingStatementLine?: number;
    /** For condition_expression: the exact source range of the call expression to replace */
    callRange?: {
        start: number;
        end: number;
    };
    /** For condition_expression: the full text of the call expression (e.g., "this.validateEmail(email)") */
    callText?: string;
}
export declare class BubbleScript {
    private ast;
    private scopeManager;
    parsingErrors: string[];
    private parsedBubbles;
    private originalParsedBubbles;
    private workflow;
    private scriptVariables;
    private variableLocations;
    instanceMethodsLocation: Record<string, {
        startLine: number;
        endLine: number;
        definitionStartLine: number;
        bodyStartLine: number;
        invocationLines: MethodInvocationInfo[];
    }>;
    private bubbleScript;
    private bubbleFactory;
    currentBubbleScript: string;
    trigger: BubbleTrigger;
    /**
     * Reparse the AST and bubbles after the script has been modified
     * This is necessary when the script text changes but we need updated bubble locations
     */
    reparseAST(): void;
    /**
     * Find the matching original bubble for a given bubble.
     * Used to restore original locations for user-facing data.
     */
    private findOriginalBubble;
    constructor(bubbleScript: string, bubbleFactory: BubbleFactory);
    get bubblescript(): string;
    /** Print script with line numbers in pretty readable format */
    showScript(message: string): void;
    /**
     * Get all variable names available at a specific line (excluding globals)
     * This is like setting a debugger breakpoint at that line
     */
    getVarsForLine(lineNumber: number): Variable[];
    /**
     * Find ALL scopes that contain the given line number
     * This is crucial because variables can be in sibling scopes (like block + for)
     */
    private getAllScopesContainingLine;
    /**
     * Find the most specific scope that contains the given line number
     */
    private findScopeForLine;
    /**
     * Get all variables accessible from a scope (including parent scopes)
     * This mimics how debugger shows variables from current scope + outer scopes
     */
    private getAllAccessibleVariables;
    /**
     * Check if a variable is declared before a given line number
     * This ensures we only return variables that actually exist at the breakpoint
     */
    private isVariableDeclaredBeforeLine;
    /**
     * Check if a variable is a global/built-in (filter these out)
     */
    private isGlobalVariable;
    /**
     * Debug method: Get detailed scope info for a line
     */
    getScopeInfoForLine(lineNumber: number): {
        scopeType: string;
        variables: string[];
        allAccessible: string[];
        lineRange: string;
    } | null;
    /**
     * Build a mapping of all user-defined variables with unique IDs
     * Also cross-references with parsed bubbles
     * Fills variableLocations
     */
    private buildVariableMapping;
    /**
     * Extract precise location (line and column) for a variable
     */
    private extractVariableLocation;
    /**
     * Get Variable object by its $id
     */
    getVariableById(id: number): Variable | undefined;
    /**
     * Get all user-defined variables with their $ids
     */
    getAllVariablesWithIds(): Record<number, Variable>;
    /**
     * Get all user-defined variables in the entire script
     */
    getAllUserVariables(): string[];
    /**
     * Get the parsed AST (for debugging or further analysis)
     */
    getAST(): TSESTree.Program;
    getOriginalParsedBubbles(): Record<number, ParsedBubbleWithInfo>;
    /**
     * Get the scope manager (for advanced analysis)
     */
    getScopeManager(): ScopeManager;
    /**
     * Get the parsed bubbles with NORMALIZED locations (for internal use like injection).
     * These locations match the normalized script, not the original user script.
     */
    getParsedBubblesRaw(): Record<number, ParsedBubbleWithInfo>;
    /**
     * Get the parsed bubbles with original line numbers restored.
     * This returns a COPY of current bubbles (with clones, workflow updates, etc.)
     * with locations matching the original script that users see in their IDE.
     * Use this for frontend/user-facing data.
     */
    getParsedBubbles(): Record<number, ParsedBubbleWithInfo>;
    /**
     * Get the hierarchical workflow structure
     */
    getWorkflow(): ParsedWorkflow;
    /**
     * Get the handle method location (start and end lines)
     */
    getHandleMethodLocation(): {
        startLine: number;
        endLine: number;
        definitionStartLine: number;
        bodyStartLine: number;
    } | null;
    getInstanceMethodLocation(methodName: string): {
        startLine: number;
        endLine: number;
        definitionStartLine: number;
        bodyStartLine: number;
        invocationLines: MethodInvocationInfo[];
    } | null;
    /**
     * Get location information for a variable by its $id
     */
    getVariableLocation(variableId: number): {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    } | null;
    /**
     * Get all variable locations
     */
    getAllVariableLocations(): Record<number, {
        startLine: number;
        startCol: number;
        endLine: number;
        endCol: number;
    }>;
    resetBubbleScript(): void;
    /** Reassign variable to another value and assign to the new bubble script and return the new bubble script */
    reassignVariable(variableId: number, newValue: string): string;
    /** Inject lines of script at particular locations and return the new bubble script */
    injectLines(lines: string[], lineNumber: number): string;
    /**
     * Helper method to escape special regex characters in variable names
     */
    private escapeRegExp;
    /**
     * Build a JSON Schema object for the payload parameter of the top-level `handle` entrypoint.
     * Delegates to BubbleParser for the actual implementation.
     */
    getPayloadJsonSchema(): Record<string, unknown> | null;
    /**
     * Detect the BubbleTriggerEventRegistry key from the class extends generic.
     * Example: class X extends BubbleFlow<'slack/bot_mentioned'> {}
     * Returns the string key (e.g., 'slack/bot_mentioned') or null if not found.
     */
    getBubbleTriggerEventType(): BubbleTrigger | null;
}
//# sourceMappingURL=BubbleScript.d.ts.map