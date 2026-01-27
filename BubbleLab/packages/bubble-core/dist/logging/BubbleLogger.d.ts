import { BubbleError } from '../types/bubble-errors';
import type { ExecutionSummary } from '@bubblelab/shared-schemas';
import { CredentialType } from '@bubblelab/shared-schemas';
export declare enum LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
}
export interface LogEntry {
    id: string;
    timestamp: number;
    level: LogLevel;
    message: string;
    metadata: LogMetadata;
    duration?: number;
}
export interface LogServiceUsage {
    usage: number;
    service: CredentialType;
    unit: string;
    subService?: string;
}
export interface LogMetadata {
    flowName: string;
    variableId?: number;
    lineNumber?: number;
    functionName?: string;
    bubbleName?: string;
    variableName?: string;
    operationType?: 'bubble_instantiation' | 'bubble_execution' | 'variable_assignment' | 'condition' | 'loop_iteration' | 'script' | 'bubble_execution_complete' | 'function_call';
    memoryUsage?: NodeJS.MemoryUsage;
    stackTrace?: string;
    additionalData?: Record<string, unknown>;
    serviceUsage?: LogServiceUsage;
}
export interface LoggerConfig {
    minLevel: LogLevel;
    enableTiming: boolean;
    enableMemoryTracking: boolean;
    enableStackTraces: boolean;
    maxLogEntries: number;
    bufferSize: number;
    flushInterval?: number;
    pricingTable: Record<string, {
        unit: string;
        unitCost: number;
    }>;
    userCredentialMapping?: Map<number, Set<CredentialType>>;
}
export declare class BubbleLogger {
    private flowName;
    private config;
    private logs;
    private startTime;
    private lastLogTime;
    private executionId;
    private lineTimings;
    private lineLogCounts;
    private peakMemoryUsage?;
    private buffer;
    private flushTimer?;
    cumulativeServiceUsageByService: Map<string, Map<number, {
        usage: number;
        service: CredentialType;
        subService?: string;
        unit: string;
    }>>;
    private bubbleStartTimes;
    private functionCallStartTimes;
    private userCredentialMapping?;
    constructor(flowName: string, config: Partial<Omit<LoggerConfig, 'pricingTable'>> & {
        pricingTable: Record<string, {
            unit: string;
            unitCost: number;
        }>;
    });
    /**
     * Log execution at a specific line with timing information
     * Returns the message for backward compatibility, use shouldLogLine() to check if logging occurred
     */
    logLine(lineNumber: number, message: string, additionalData?: Record<string, unknown>): string;
    /**
     * Check if a line should be logged (hasn't exceeded the limit)
     */
    protected shouldLogLine(lineNumber: number): boolean;
    /**
     * Log bubble instantiation
     */
    logBubbleInstantiation(variableId: number, bubbleName: string, variableName: string, parameters?: Record<string, unknown>): string;
    /**
     * Log bubble execution
     */
    logBubbleExecution(variableId: number, bubbleName: string, variableName: string, parameters?: Record<string, unknown>): string;
    /**
     * Log bubble execution completion
     */
    logBubbleExecutionComplete(variableId: number, bubbleName: string, variableName: string, result?: unknown): string;
    /**
     * Log variable assignment
     */
    logVariableAssignment(lineNumber: number, variableName: string, value: unknown): void;
    /**
     * Log control flow operations
     */
    logControlFlow(lineNumber: number, type: 'condition' | 'loop_iteration', condition: string, result?: boolean): void;
    protected getServiceUsageKey(service: LogServiceUsage, unitCost?: number): string;
    /**
     * Add token usage to cumulative tracking per model and variable ID
     */
    addServiceUsage(serviceUsage: LogServiceUsage, variableId?: number): void;
    /**
     * Log token usage
     */
    logTokenUsage(serviceUsage: LogServiceUsage, message?: string, metadata?: Partial<LogMetadata>): string;
    /**
     * Log AI agent tool call start
     */
    logToolCallStart(toolCallId: string, toolName: string, toolInput: unknown, message?: string): string;
    /**
     * Log AI agent tool call completion
     */
    logToolCallComplete(toolCallId: string, toolName: string, toolInput: unknown, toolOutput: unknown, duration: number, message?: string): string;
    /**
     * Log transformation function call start
     */
    logFunctionCallStart(variableId: number, functionName: string, functionInput: unknown, lineNumber?: number): string;
    /**
     * Log transformation function call completion
     */
    logFunctionCallComplete(variableId: number, functionName: string, functionOutput: unknown, duration: number, lineNumber?: number): string;
    /**
     * Get execution time for a function call
     */
    protected getFunctionCallExecutionTime(variableId: number): number;
    /**
     * General logging method with different levels
     */
    log(level: LogLevel, message: string, metadata?: Partial<LogMetadata>): void;
    /**
     * Convenience methods for different log levels
     */
    trace(message: string, metadata?: Partial<LogMetadata>): void;
    debug(message: string, metadata?: Partial<LogMetadata>): void;
    info(message: string, metadata?: Partial<LogMetadata>): void;
    warn(message: string, metadata?: Partial<LogMetadata>): void;
    error(message: string, error?: BubbleError, metadata?: Partial<LogMetadata>): void;
    fatal(message: string, error?: BubbleError, metadata?: Partial<LogMetadata>): void;
    /**
     * Flush buffered logs to main log array
     */
    flush(): void;
    /**
     * Get execution summary with analytics
     */
    getExecutionSummary(): ExecutionSummary;
    /**
     * Get all logs
     */
    getLogs(): LogEntry[];
    /**
     * Get logs filtered by level
     */
    getLogsByLevel(level: LogLevel): LogEntry[];
    /**
     * Get logs for a specific line number
     */
    getLogsByLine(lineNumber: number): LogEntry[];
    /**
     * Export logs in various formats
     */
    exportLogs(format: 'json' | 'csv' | 'table'): string;
    /**
     * Clean up resources
     */
    dispose(): void;
    private captureStackTrace;
    private formatAsTable;
    /**
     * Get individual bubble execution time for a specific variable ID
     */
    protected getBubbleExecutionTime(variableId: number): number;
}
//# sourceMappingURL=BubbleLogger.d.ts.map