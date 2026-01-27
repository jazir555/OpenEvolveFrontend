import { BubbleLogger, type LoggerConfig, type LogMetadata } from './BubbleLogger.js';
import type { StreamCallback } from '@bubblelab/shared-schemas';
import { BubbleError } from '../types/bubble-errors';
interface WebhookStreamLoggerConfig extends Partial<Omit<LoggerConfig, 'pricingTable'>> {
    pricingTable: Record<string, {
        unit: string;
        unitCost: number;
    }>;
    streamCallback?: StreamCallback;
}
/**
 * Webhook-optimized streaming logger for terminal-friendly output
 * Designed specifically for webhook streaming endpoints
 * Shows only essential information with truncated data for readability
 */
export declare class WebhookStreamLogger extends BubbleLogger {
    private streamCallback?;
    constructor(flowName: string, options?: WebhookStreamLoggerConfig);
    /**
     * Override logBubbleExecution to emit clean, truncated events
     */
    logBubbleExecution(variableId: number, bubbleName: string, variableName: string, parameters?: Record<string, unknown>): string;
    /**
     * Override logBubbleExecutionComplete to emit clean results
     */
    logBubbleExecutionComplete(variableId: number, bubbleName: string, variableName: string, result?: unknown): string;
    /**
     * Log execution completion with beautiful, terminal-friendly formatting
     * Makes the final result super clear and easy to read
     */
    logExecutionComplete(success: boolean, finalResult?: unknown, error?: string): void;
    /**
     * Override error method to emit clean error events
     */
    error(message: string, error?: BubbleError, metadata?: Partial<LogMetadata>): void;
    /**
     * Override warn method to emit clean warning events
     */
    warn(message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Override info method to emit clean info events
     */
    info(message: string, metadata?: Partial<LogMetadata>): void;
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
//# sourceMappingURL=WebhookStreamLogger.d.ts.map