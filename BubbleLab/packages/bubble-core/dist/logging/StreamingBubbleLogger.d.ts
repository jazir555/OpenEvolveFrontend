import { BubbleLogger, type LoggerConfig, type LogMetadata } from './BubbleLogger.js';
import type { StreamCallback } from '@bubblelab/shared-schemas';
import { BubbleError } from '../types/bubble-errors';
interface StreamingLoggerConfig extends Partial<Omit<LoggerConfig, 'pricingTable'>> {
    pricingTable: Record<string, {
        unit: string;
        unitCost: number;
    }>;
    streamCallback?: StreamCallback;
}
/**
 * Streaming version of BubbleLogger that emits real-time events
 * Extends BubbleLogger to maintain all existing functionality
 */
export declare class StreamingBubbleLogger extends BubbleLogger {
    private streamCallback?;
    constructor(flowName: string, options?: StreamingLoggerConfig);
    /**
     * Override logLine to emit streaming events
     */
    logLine(lineNumber: number, message: string, additionalData?: Record<string, unknown>): string;
    /**
     * Override logBubbleInstantiation to emit streaming events
     */
    logBubbleInstantiation(variableId: number, bubbleName: string, variableName: string, parameters?: Record<string, unknown>): string;
    /**
     * Override logBubbleExecution to emit streaming events
     */
    logBubbleExecution(variableId: number, bubbleName: string, variableName: string, parameters?: Record<string, unknown>): string;
    logBubbleExecutionComplete(variableId: number, bubbleName: string, variableName: string, result?: unknown): string;
    /**
     * Override logFunctionCallStart to emit streaming events
     */
    logFunctionCallStart(variableId: number, functionName: string, functionInput: unknown, lineNumber?: number): string;
    /**
     * Override logFunctionCallComplete to emit streaming events
     */
    logFunctionCallComplete(variableId: number, functionName: string, functionOutput: unknown, duration: number, lineNumber?: number): string;
    /**
     * Log execution completion
     */
    logExecutionComplete(success: boolean, finalResult?: unknown, error?: string): void;
    /**
     * Override trace method to emit streaming events
     */
    trace(message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Override debug method to emit streaming events
     */
    debug(message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Override info method to emit streaming events
     */
    info(message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Override warn method to emit streaming events
     */
    warn(message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Override error method to emit streaming events
     */
    error(message: string, error?: BubbleError, metadata?: Partial<LogMetadata>): void;
    /**
     * Override fatal method to emit streaming events
     */
    fatal(message: string, error?: BubbleError, metadata?: Partial<LogMetadata>): void;
    /**
     * Override logToolCallStart to add streaming events
     */
    logToolCallStart(toolCallId: string, toolName: string, toolInput: unknown, message?: string): string;
    /**
     * Override logToolCallComplete to add streaming events
     */
    logToolCallComplete(toolCallId: string, toolName: string, toolInput: unknown, toolOutput: unknown, duration: number, message?: string): string;
    /**
     * Set or update the stream callback
     */
    setStreamCallback(callback: StreamCallback): void;
    /**
     * Clear the stream callback
     */
    clearStreamCallback(): void;
    /**
     * Emit a streaming event if callback is set
     */
    private emitStreamEvent;
    /**
     * Get current execution time in milliseconds
     */
    private getCurrentExecutionTime;
    /**
     * Get current memory usage in bytes
     */
    private getCurrentMemoryUsage;
}
export {};
//# sourceMappingURL=StreamingBubbleLogger.d.ts.map