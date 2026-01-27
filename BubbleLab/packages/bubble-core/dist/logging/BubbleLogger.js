const SHOULD_ENABLE_TOKEN_USAGE_LOGGING_IN_CONSOLE = false;
export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["TRACE"] = 0] = "TRACE";
    LogLevel[LogLevel["DEBUG"] = 1] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 2] = "INFO";
    LogLevel[LogLevel["WARN"] = 3] = "WARN";
    LogLevel[LogLevel["ERROR"] = 4] = "ERROR";
    LogLevel[LogLevel["FATAL"] = 5] = "FATAL";
})(LogLevel || (LogLevel = {}));
export class BubbleLogger {
    flowName;
    config;
    logs = [];
    startTime;
    lastLogTime;
    executionId;
    lineTimings = new Map();
    lineLogCounts = new Map();
    peakMemoryUsage;
    buffer = [];
    flushTimer;
    // Track cumulative raw usage by service, then by variable ID
    // Structure: serviceKey -> variableId -> usage data
    cumulativeServiceUsageByService = new Map();
    // Track individual bubble execution times
    bubbleStartTimes = new Map();
    // Track individual function call execution times
    functionCallStartTimes = new Map();
    // Store user credential mapping for cost calculation
    userCredentialMapping;
    constructor(flowName, config) {
        this.flowName = flowName;
        const { pricingTable, userCredentialMapping, ...restConfig } = config;
        this.config = {
            minLevel: LogLevel.INFO,
            enableTiming: true,
            enableMemoryTracking: true,
            enableStackTraces: false,
            maxLogEntries: 10000,
            bufferSize: 100,
            flushInterval: 1000,
            ...restConfig,
            pricingTable,
        };
        this.userCredentialMapping = userCredentialMapping;
        this.startTime = Date.now();
        this.lastLogTime = this.startTime;
        this.executionId = `${flowName}-${this.startTime}-${Math.random().toString(36).substring(7)}`;
        if (this.config.flushInterval) {
            this.flushTimer = setInterval(() => this.flush(), this.config.flushInterval);
        }
    }
    /**
     * Log execution at a specific line with timing information
     * Returns the message for backward compatibility, use shouldLogLine() to check if logging occurred
     */
    logLine(lineNumber, message, additionalData) {
        const MAX_LOGS_PER_LINE = 5;
        // Track how many times this line has been logged
        const logCount = this.lineLogCounts.get(lineNumber) ?? 0;
        // Only log if under the limit
        if (logCount < MAX_LOGS_PER_LINE) {
            this.log(LogLevel.INFO, message, {
                lineNumber,
                operationType: 'script',
                additionalData,
            });
            this.lineLogCounts.set(lineNumber, logCount + 1);
        }
        // Track line execution timing (always track, even if not logging)
        const now = Date.now();
        const duration = now - this.lastLogTime;
        // Update timing statistics for this line
        const stats = this.lineTimings.get(lineNumber);
        if (!stats) {
            this.lineTimings.set(lineNumber, {
                min: duration,
                max: duration,
                sum: duration,
                count: 1,
            });
        }
        else {
            stats.min = Math.min(stats.min, duration);
            stats.max = Math.max(stats.max, duration);
            stats.sum += duration;
            stats.count++;
        }
        return message;
    }
    /**
     * Check if a line should be logged (hasn't exceeded the limit)
     */
    shouldLogLine(lineNumber) {
        const MAX_LOGS_PER_LINE = 5;
        const logCount = this.lineLogCounts.get(lineNumber) ?? 0;
        return logCount < MAX_LOGS_PER_LINE;
    }
    /**
     * Log bubble instantiation
     */
    logBubbleInstantiation(variableId, bubbleName, variableName, parameters) {
        const message = parameters?.message;
        const messagePreview = message ? message.substring(0, 200) : '';
        const logMessage = `Instantiating bubble ${bubbleName} as ${variableName}${messagePreview ? ` with message: ${messagePreview}` : ''}`;
        this.log(LogLevel.DEBUG, logMessage, {
            variableId,
            bubbleName,
            variableName,
            operationType: 'bubble_instantiation',
            additionalData: { parameters },
        });
        return logMessage;
    }
    /**
     * Log bubble execution
     */
    logBubbleExecution(variableId, bubbleName, variableName, parameters) {
        // Track start time for this bubble
        this.bubbleStartTimes.set(variableId, Date.now());
        // Try to find the message from the parameters
        const message = parameters?.message;
        const messagePreview = message ? message.substring(0, 200) : '';
        const logMessage = `Executing bubble ${bubbleName} as ${variableName}${messagePreview ? ` with message: ${messagePreview}` : ''}`;
        this.log(LogLevel.DEBUG, logMessage, {
            variableId,
            bubbleName,
            variableName,
            operationType: 'bubble_execution',
            additionalData: { parameters },
        });
        return logMessage;
    }
    /**
     * Log bubble execution completion
     */
    logBubbleExecutionComplete(variableId, bubbleName, variableName, result) {
        // Calculate individual bubble execution time
        const individualExecutionTime = this.getBubbleExecutionTime(variableId);
        const logMessage = `Bubble execution completed: ${bubbleName} in ${individualExecutionTime}ms`;
        this.log(LogLevel.DEBUG, logMessage, {
            variableId,
            bubbleName,
            variableName,
            operationType: 'bubble_execution_complete',
            additionalData: { result, executionTime: individualExecutionTime },
        });
        // Clean up the start time for this bubble
        this.bubbleStartTimes.delete(variableId);
        return logMessage;
    }
    /**
     * Log variable assignment
     */
    logVariableAssignment(lineNumber, variableName, value) {
        this.log(LogLevel.TRACE, `Assigning variable ${variableName}`, {
            lineNumber,
            variableName,
            operationType: 'variable_assignment',
            additionalData: { value: typeof value === 'object' ? '[Object]' : value },
        });
    }
    /**
     * Log control flow operations
     */
    logControlFlow(lineNumber, type, condition, result) {
        this.log(LogLevel.TRACE, `${type}: ${condition}`, {
            lineNumber,
            operationType: type,
            additionalData: { condition, result },
        });
    }
    getServiceUsageKey(service, unitCost) {
        //Combine service, subservice and unit into a single string
        //Optionally include unitCost to differentiate between user and system credentials
        const baseKey = `${service.service}${service.subService ? `:${service.subService}` : ''}:${service.unit}`;
        return unitCost !== undefined ? `${baseKey}:${unitCost}` : baseKey;
    }
    /**
     * Add token usage to cumulative tracking per model and variable ID
     */
    addServiceUsage(serviceUsage, variableId) {
        const serviceKey = this.getServiceUsageKey(serviceUsage);
        // Get or create the service entry
        if (!this.cumulativeServiceUsageByService.has(serviceKey)) {
            this.cumulativeServiceUsageByService.set(serviceKey, new Map());
        }
        const serviceMap = this.cumulativeServiceUsageByService.get(serviceKey);
        // If variableId is provided, track per variable ID
        // If not provided, use 0 as a fallback (for backward compatibility)
        const varId = variableId ?? 0;
        const existing = serviceMap.get(varId) || {
            usage: 0,
            service: serviceUsage.service,
            subService: serviceUsage.subService,
            unit: serviceUsage.unit,
        };
        serviceMap.set(varId, {
            usage: existing.usage + serviceUsage.usage,
            service: existing.service,
            subService: existing.subService,
            unit: existing.unit,
        });
    }
    /**
     * Log token usage
     */
    logTokenUsage(serviceUsage, message, metadata) {
        const logMessage = message ||
            `Service usage (${this.getServiceUsageKey(serviceUsage)}): ${serviceUsage.usage} units`;
        // Add token usage to cumulative tracking per model and variable ID
        this.addServiceUsage(serviceUsage, metadata?.variableId);
        // Convert Map to object for logging (flattened for backward compatibility)
        const serviceUsageByService = {};
        for (const [serviceKey, varIdMap,] of this.cumulativeServiceUsageByService.entries()) {
            // Aggregate usage across all variable IDs for display
            let totalUsage = 0;
            for (const usageData of varIdMap.values()) {
                totalUsage += usageData.usage;
            }
            serviceUsageByService[serviceKey] = {
                usage: totalUsage,
                service: Array.from(varIdMap.values())[0]?.service,
                subService: Array.from(varIdMap.values())[0]?.subService,
                unit: Array.from(varIdMap.values())[0]?.unit,
            };
        }
        if (SHOULD_ENABLE_TOKEN_USAGE_LOGGING_IN_CONSOLE) {
            this.info(logMessage, {
                ...metadata,
                serviceUsage,
                operationType: metadata?.operationType || 'bubble_execution',
                additionalData: {
                    ...metadata?.additionalData,
                    serviceUsage,
                    variableId: metadata?.variableId,
                    cumulativeServiceUsageByService: serviceUsageByService, // Per-service breakdown
                },
            });
        }
        return logMessage;
    }
    /**
     * Log AI agent tool call start
     */
    logToolCallStart(toolCallId, toolName, toolInput, message) {
        const logMessage = message || `Starting tool call: ${toolName}`;
        this.debug(logMessage, {
            bubbleName: 'ai-agent',
            operationType: 'bubble_execution',
            additionalData: {
                toolCallId,
                toolName,
                toolInput,
                phase: 'tool_call_start',
            },
        });
        return logMessage;
    }
    /**
     * Log AI agent tool call completion
     */
    logToolCallComplete(toolCallId, toolName, toolInput, toolOutput, duration, message) {
        const logMessage = message || `Completed tool call: ${toolName} (${duration}ms)`;
        this.debug(logMessage, {
            bubbleName: 'ai-agent',
            operationType: 'bubble_execution',
            additionalData: {
                toolCallId,
                toolName,
                toolInput,
                toolOutput,
                duration,
                phase: 'tool_call_complete',
            },
        });
        return logMessage;
    }
    /**
     * Log transformation function call start
     */
    logFunctionCallStart(variableId, functionName, functionInput, lineNumber) {
        // Track start time for this function call
        this.functionCallStartTimes.set(variableId, Date.now());
        const logMessage = `Starting step: ${functionName}`;
        this.log(LogLevel.DEBUG, logMessage, {
            variableId,
            lineNumber,
            operationType: 'function_call',
            additionalData: {
                functionName,
                functionInput,
                phase: 'function_call_start',
            },
        });
        return logMessage;
    }
    /**
     * Log transformation function call completion
     */
    logFunctionCallComplete(variableId, functionName, functionOutput, duration, lineNumber) {
        const logMessage = `Completed step: ${functionName}`;
        this.log(LogLevel.DEBUG, logMessage, {
            variableId,
            lineNumber,
            operationType: 'function_call',
            additionalData: {
                functionName,
                functionOutput,
                duration,
                phase: 'function_call_complete',
            },
        });
        // Clean up the start time for this function call
        this.functionCallStartTimes.delete(variableId);
        return logMessage;
    }
    /**
     * Get execution time for a function call
     */
    getFunctionCallExecutionTime(variableId) {
        const functionCallStartTime = this.functionCallStartTimes.get(variableId);
        if (!functionCallStartTime) {
            return 0;
        }
        return Date.now() - functionCallStartTime;
    }
    /**
     * General logging method with different levels
     */
    log(level, message, metadata = {}) {
        if (level < this.config.minLevel) {
            return;
        }
        const now = Date.now();
        const duration = now - this.lastLogTime;
        const entry = {
            id: `${this.executionId}-${this.logs.length}`,
            timestamp: now,
            level,
            message,
            duration,
            metadata: {
                flowName: this.flowName,
                ...metadata,
                memoryUsage: this.config.enableMemoryTracking
                    ? process.memoryUsage()
                    : undefined,
                stackTrace: this.config.enableStackTraces
                    ? this.captureStackTrace()
                    : undefined,
            },
        };
        // Update peak memory usage
        if (this.config.enableMemoryTracking && entry.metadata.memoryUsage) {
            if (!this.peakMemoryUsage ||
                entry.metadata.memoryUsage.heapUsed > this.peakMemoryUsage.heapUsed) {
                this.peakMemoryUsage = entry.metadata.memoryUsage;
            }
        }
        this.buffer.push(entry);
        this.lastLogTime = now;
        // Auto-flush if buffer is full
        if (this.buffer.length >= this.config.bufferSize) {
            this.flush();
        }
    }
    /**
     * Convenience methods for different log levels
     */
    trace(message, metadata) {
        this.log(LogLevel.TRACE, message, metadata);
    }
    debug(message, metadata) {
        this.log(LogLevel.DEBUG, message, metadata);
    }
    info(message, metadata) {
        this.log(LogLevel.INFO, message, metadata);
    }
    warn(message, metadata) {
        this.log(LogLevel.WARN, message, metadata);
    }
    error(message, error, metadata) {
        const errorMetadata = error
            ? {
                additionalData: {
                    errorMessage: error.message,
                    errorStack: error.stack,
                    errorName: error.name,
                },
            }
            : {};
        this.log(LogLevel.ERROR, message, {
            ...metadata,
            ...errorMetadata,
            variableId: error?.variableId,
            bubbleName: error?.bubbleName,
        });
    }
    fatal(message, error, metadata) {
        const errorMetadata = error
            ? {
                additionalData: {
                    errorMessage: error.message,
                    errorStack: error.stack,
                    errorName: error.name,
                },
            }
            : {};
        this.log(LogLevel.FATAL, message, {
            ...metadata,
            ...errorMetadata,
            variableId: error?.variableId,
            bubbleName: error?.bubbleName,
        });
    }
    /**
     * Flush buffered logs to main log array
     */
    flush() {
        if (this.buffer.length === 0)
            return;
        this.logs.push(...this.buffer);
        this.buffer = [];
        // Trim logs if exceeding max entries
        if (this.logs.length > this.config.maxLogEntries) {
            this.logs = this.logs.slice(-this.config.maxLogEntries);
        }
    }
    /**
     * Get execution summary with analytics
     */
    getExecutionSummary() {
        this.flush(); // Ensure all logs are included
        const endTime = Date.now();
        const totalDuration = endTime - this.startTime;
        const MAX_LOG_ENTRIES = 1000;
        const errorLogs = this.logs.filter((log) => log.level >= LogLevel.ERROR);
        const warningLogs = this.logs.filter((log) => log.level === LogLevel.WARN);
        // Calculate average line execution time from statistics
        let totalExecutionTime = 0;
        let totalExecutionCount = 0;
        for (const stats of this.lineTimings.values()) {
            totalExecutionTime += stats.sum;
            totalExecutionCount += stats.count;
        }
        // Find slowest lines using max from statistics
        const slowestLines = [];
        for (const [lineNumber, stats] of this.lineTimings.entries()) {
            if (slowestLines.length >= MAX_LOG_ENTRIES)
                break;
            const maxDuration = stats.max;
            const logEntry = this.logs.find((log) => log.metadata.lineNumber === lineNumber && log.duration === maxDuration);
            if (logEntry) {
                slowestLines.push({
                    lineNumber,
                    duration: maxDuration,
                    message: logEntry.message,
                });
            }
        }
        slowestLines.sort((a, b) => b.duration - a.duration);
        // Extract errors and warnings with details
        const errors = errorLogs.map((log) => ({
            message: log.message,
            timestamp: log.timestamp,
            bubbleName: log.metadata.bubbleName,
            variableId: log.metadata.variableId,
            lineNumber: log.metadata.lineNumber,
            additionalData: log.metadata.additionalData,
        }));
        const warnings = warningLogs.map((log) => ({
            message: log.message,
            timestamp: log.timestamp,
            bubbleName: log.metadata.bubbleName,
            variableId: log.metadata.variableId,
            lineNumber: log.metadata.lineNumber,
            additionalData: log.metadata.additionalData,
        }));
        // If pricing table is empty or undefined, print a warning
        if (!this.config.pricingTable ||
            Object.keys(this.config.pricingTable).length === 0) {
            console.warn('Pricing table is empty, no pricing data will be available. To track cost tracking, please set the pricing table in the logger config.');
        }
        const serviceUsage = [];
        // Calculate service usage based on pricing table, per variable ID
        // Create separate entries for user credentials (unitCost = 0) vs system credentials
        // Key format: serviceKey:unitCost to differentiate between user and system credentials
        const aggregatedServiceUsage = new Map();
        for (const [serviceKey, varIdMap,] of this.cumulativeServiceUsageByService.entries()) {
            // Use the full key (service:subService:unit) to look up pricing
            // This matches the key format used in the pricing table
            const pricingKey = serviceKey; // key is already in format "service:subService:unit"
            const pricing = this.config.pricingTable[pricingKey];
            const systemUnitCost = pricing?.unitCost || 0;
            // Process each variable ID's usage separately
            for (const [variableId, usageData] of varIdMap.entries()) {
                // Check if this variable ID used a user credential for this service type
                const userCredsForVarId = this.userCredentialMapping?.get(variableId);
                const usedUserCredential = userCredsForVarId?.has(usageData.service) ?? false;
                // Set unitCost to 0 if user credential was used, otherwise use pricing table
                const unitCost = usedUserCredential ? 0 : systemUnitCost;
                const costForThisVarId = usageData.usage * unitCost;
                // Create a key that includes unitCost to separate user vs system credential usage
                const usageKey = `${serviceKey}:${unitCost}`;
                // Aggregate by usage key (serviceKey:unitCost)
                const existing = aggregatedServiceUsage.get(usageKey);
                if (existing) {
                    existing.usage += usageData.usage;
                    existing.totalCost += costForThisVarId;
                }
                else {
                    aggregatedServiceUsage.set(usageKey, {
                        service: usageData.service,
                        usage: usageData.usage,
                        subService: usageData.subService,
                        unit: usageData.unit,
                        unitCost: unitCost,
                        totalCost: costForThisVarId,
                    });
                }
            }
        }
        // Convert aggregated map to ServiceUsage array
        for (const [, aggregated] of aggregatedServiceUsage.entries()) {
            serviceUsage.push({
                service: aggregated.service,
                usage: aggregated.usage,
                subService: aggregated.subService,
                unit: aggregated.unit,
                unitCost: aggregated.unitCost,
                totalCost: aggregated.totalCost,
            });
        }
        return {
            totalDuration,
            errors,
            warnings,
            serviceUsage,
            totalCost: serviceUsage.reduce((acc, curr) => acc + curr.totalCost, 0),
        };
    }
    /**
     * Get all logs
     */
    getLogs() {
        this.flush();
        return [...this.logs];
    }
    /**
     * Get logs filtered by level
     */
    getLogsByLevel(level) {
        return this.getLogs().filter((log) => log.level === level);
    }
    /**
     * Get logs for a specific line number
     */
    getLogsByLine(lineNumber) {
        return this.getLogs().filter((log) => log.metadata.lineNumber === lineNumber);
    }
    /**
     * Export logs in various formats
     */
    exportLogs(format) {
        const logs = this.getLogs();
        switch (format) {
            case 'json': {
                return JSON.stringify({
                    executionId: this.executionId,
                    summary: this.getExecutionSummary(),
                    logs,
                }, null, 2);
            }
            case 'csv': {
                const headers = 'ID,Timestamp,Level,Message,Duration,LineNumber,Operation,BubbleName,VariableName';
                const rows = logs.map((log) => [
                    log.id,
                    new Date(log.timestamp).toISOString(),
                    LogLevel[log.level],
                    `"${log.message.replace(/"/g, '""')}"`,
                    log.duration || 0,
                    log.metadata.lineNumber || '',
                    log.metadata.operationType || '',
                    log.metadata.bubbleName || '',
                    log.metadata.variableName || '',
                ].join(','));
                return [headers, ...rows].join('\n');
            }
            case 'table':
                return this.formatAsTable(logs);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }
    /**
     * Clean up resources
     */
    dispose() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        this.flush();
    }
    captureStackTrace() {
        const stack = new Error().stack;
        return stack ? stack.split('\n').slice(3).join('\n') : '';
    }
    formatAsTable(logs) {
        if (logs.length === 0)
            return 'No logs to display';
        const headers = ['Time', 'Level', 'Line', 'Duration', 'Message'];
        const rows = logs.map((log) => [
            new Date(log.timestamp).toISOString().substring(11, 23),
            LogLevel[log.level].padEnd(5),
            (log.metadata.lineNumber || '').toString().padEnd(4),
            `${log.duration || 0}ms`.padEnd(8),
            log.message.substring(0, 80),
        ]);
        const columnWidths = headers.map((header, i) => Math.max(header.length, ...rows.map((row) => row[i].length)));
        const separator = columnWidths
            .map((width) => '-'.repeat(width))
            .join('-+-');
        const headerRow = headers
            .map((header, i) => header.padEnd(columnWidths[i]))
            .join(' | ');
        const dataRows = rows.map((row) => row.map((cell, i) => cell.padEnd(columnWidths[i])).join(' | '));
        return [headerRow, separator, ...dataRows].join('\n');
    }
    /**
     * Get individual bubble execution time for a specific variable ID
     */
    getBubbleExecutionTime(variableId) {
        const bubbleStartTime = this.bubbleStartTimes.get(variableId);
        return bubbleStartTime ? Date.now() - bubbleStartTime : 0;
    }
}
//# sourceMappingURL=BubbleLogger.js.map