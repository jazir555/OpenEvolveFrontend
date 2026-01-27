import { ExecutionResult, ParsedBubbleWithInfo, CredentialType } from '@bubblelab/shared-schemas';
import { BubbleFactory, WebhookEvent, BubbleLogger, LogLevel } from '@bubblelab/bubble-core';
import type { StreamCallback } from '@bubblelab/shared-schemas';
import type { ExecutionPlan } from './types';
import { BubbleScript } from '../parse/BubbleScript';
import { BubbleInjector } from '../injection/BubbleInjector';
export interface VariableState {
    value: unknown;
    error?: string;
}
export interface BubbleRunnerOptions {
    enableLogging?: boolean;
    logLevel?: LogLevel;
    enableLineByLineLogging?: boolean;
    enableBubbleLogging?: boolean;
    streamCallback?: StreamCallback;
    useWebhookLogger?: boolean;
    pricingTable: Record<string, {
        unit: string;
        unitCost: number;
    }>;
    userCredentialMapping?: Map<number, Set<CredentialType>>;
}
export declare class BubbleRunner {
    bubbleScript: BubbleScript;
    private currentStep;
    private savedStates;
    private plan;
    private logger;
    injector: BubbleInjector;
    private options;
    private hasInjectedLogging;
    constructor(bubbleScript: string | BubbleScript, bubbleFactory: BubbleFactory, options: BubbleRunnerOptions);
    /**
     * Creates a list of steps where length = number of parsed bubbles
     * Contains the bubble and parameters to run
     * Each step represents a continuous line range (e.g., line 1-20, 21-xxx)
     */
    private buildExecutionPlan;
    /**
     * Find the line where .action() is called for a bubble
     * Uses AST to locate the method call
     */
    private findActionCallLine;
    /**
     * Recursively search AST for .action() calls on a specific variable
     */
    private findActionCallInAST;
    getParsedBubbles(): Record<string, ParsedBubbleWithInfo>;
    /**
     * Get the ORIGINAL parsed bubbles (locations from the initial script before any rewrites)
     */
    getOriginalParsedBubbles(): Record<number, ParsedBubbleWithInfo>;
    getVariables(): string[];
    /**
     * Finds step ID, calls memorizes results on previous bubbles, and runs the script from 1 to line end
     * Executes a single step from the execution plan
     */
    runStep(stepId: number): Promise<ExecutionResult>;
    /**
     * Execute a single mini-step (bubble instantiation or execution)
     */
    private executeMiniStep;
    /**
     * Save the current execution state for a specific step
     */
    private saveState;
    /**
     * Run from step 1 to end
     */
    runAll(payload?: Partial<WebhookEvent>): Promise<ExecutionResult>;
    /**
     * Find the BubbleFlow class in module exports
     */
    private findBubbleFlowClass;
    /**
     * Check if a function is a BubbleFlow class
     */
    private isBubbleFlowClass;
    /**
     * Instantiate the flow class with appropriate constructor parameters
     */
    private instantiateFlowClass;
    /**
     * Resume execution from a specific step
     * Loads the saved state and continues execution from that point
     */
    resumeFromStep(stepId: number): Promise<ExecutionResult>;
    /**
     * Get saved state for a specific step
     */
    getSavedState(stepId: number): any | undefined;
    /**
     * Get all saved states
     */
    getAllSavedStates(): Map<number, any>;
    /**
     * Clear saved states (e.g., for fresh execution)
     */
    clearSavedStates(): void;
    getPlan(): ExecutionPlan;
    /**
     * Get the logger instance
     */
    getLogger(): BubbleLogger | undefined;
    /**
     * Get execution summary with detailed analytics
     */
    getExecutionSummary(): ReturnType<BubbleLogger['getExecutionSummary']> | null;
    /**
     * Export execution logs in various formats
     */
    exportLogs(format?: 'json' | 'csv' | 'table'): string | null;
    /**
     * Find the project root directory by looking for package.json
     */
    private findProjectRoot;
    /**
     * Dispose of resources (logger, etc.)
     */
    dispose(): void;
}
//# sourceMappingURL=BubbleRunner.d.ts.map