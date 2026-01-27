import { BubbleScript } from '../parse/BubbleScript';
export interface LoggingInjectionOptions {
    enableLineByLineLogging: boolean;
    enableBubbleLogging: boolean;
    enableVariableLogging: boolean;
}
export declare class LoggerInjector {
    private bubbleScript;
    private options;
    constructor(bubbleScript: BubbleScript, options?: Partial<LoggingInjectionOptions>);
    /**
     * Inject comprehensive logging into the bubble script using existing analysis
     */
    injectLogging(): string;
    /**
     * Inject logging using original line numbers for traceability
     */
    injectLoggingWithOriginalLines(originalAST: any, originalHandleMethodLocation: any): string;
    /**
     * Inject statement-level logging using existing AST analysis
     */
    private injectLineLogging;
    /**
     * Inject statement-level logging using original line numbers for traceability
     */
    private injectLineLoggingWithOriginalLines;
    /**
     * Inject `const __bubbleFlowSelf = this;` at the beginning of all instance method bodies
     * This captures `this` in a lexical variable that works at any nesting level
     * This should be called BEFORE reapplyBubbleInstantiations so bubble instantiations can use __bubbleFlowSelf.logger
     */
    injectSelfCapture(): void;
    /**
     * Find all statements within the handle method using AST
     */
    private findStatementsInHandleMethod;
    private visitChildNodes;
    /**
     * Inject logging for all method invocations using pre-tracked invocation data
     * Uses rich invocation info captured during AST parsing
     */
    private injectFunctionCallLogging;
    /**
     * Inject logging before and after a method invocation line
     * Uses pre-parsed invocation data from AST analysis - no regex parsing needed
     */
    private injectLoggingForInvocation;
    private injectPromiseAllElementLogging;
}
//# sourceMappingURL=LoggerInjector.d.ts.map