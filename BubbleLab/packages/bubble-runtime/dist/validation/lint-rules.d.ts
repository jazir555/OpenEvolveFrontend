import ts from 'typescript';
/**
 * Represents a lint error found during validation
 */
export interface LintError {
    line: number;
    column?: number;
    message: string;
}
/**
 * Context containing pre-parsed AST information for lint rules
 * This allows rules to avoid redundant AST traversals
 */
export interface LintRuleContext {
    sourceFile: ts.SourceFile;
    bubbleFlowClass: ts.ClassDeclaration | null;
    handleMethod: ts.MethodDeclaration | null;
    handleMethodBody: ts.Block | null;
    importedBubbleClasses: Set<string>;
}
/**
 * Interface for lint rules that can validate BubbleFlow code
 */
export interface LintRule {
    name: string;
    validate(context: LintRuleContext): LintError[];
}
/**
 * Registry that manages and executes all lint rules
 */
export declare class LintRuleRegistry {
    private rules;
    /**
     * Register a lint rule
     */
    register(rule: LintRule): void;
    /**
     * Execute all registered rules on the given code
     * Traverses AST once and shares context with all rules for efficiency
     */
    validateAll(sourceFile: ts.SourceFile): LintError[];
    /**
     * Get all registered rule names
     */
    getRuleNames(): string[];
}
/**
 * Lint rule that prevents throw statements directly in the handle method
 */
export declare const noThrowInHandleRule: LintRule;
/**
 * Lint rule that prevents direct bubble instantiation in the handle method
 */
export declare const noDirectBubbleInstantiationInHandleRule: LintRule;
/**
 * Lint rule that prevents credentials parameter from being used in bubble instantiations
 */
export declare const noCredentialsParameterRule: LintRule;
/**
 * Lint rule that prevents usage of process.env
 */
export declare const noProcessEnvRule: LintRule;
/**
 * Lint rule that prevents method invocations inside complex expressions
 */
export declare const noMethodInvocationInComplexExpressionRule: LintRule;
/**
 * Lint rule that prevents try-catch statements in the handle method
 * Try-catch blocks interfere with runtime instrumentation and error handling
 */
export declare const noTryCatchInHandleRule: LintRule;
/**
 * Lint rule that prevents methods from calling other methods
 * Methods should only be called from the handle method, not from other methods
 */
export declare const noMethodCallingMethodRule: LintRule;
/**
 * Lint rule that prevents usage of 'any' type
 * Using 'any' bypasses TypeScript's type checking and should be avoided
 */
export declare const noAnyTypeRule: LintRule;
/**
 * Lint rule that prevents multiple BubbleFlow classes in a single file
 * Only one class extending BubbleFlow is allowed per file for proper runtime instrumentation
 */
export declare const singleBubbleFlowClassRule: LintRule;
/**
 * Default registry instance with all rules registered
 */
export declare const defaultLintRuleRegistry: LintRuleRegistry;
//# sourceMappingURL=lint-rules.d.ts.map