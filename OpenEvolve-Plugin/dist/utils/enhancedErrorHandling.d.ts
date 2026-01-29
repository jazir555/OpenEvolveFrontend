/**
 * Enhanced Error Handling System
 * Provides sophisticated error classification, recovery strategies, and reporting
 */
/**
 * Advanced Error Classifier with ML-like pattern matching
 */
export declare class AdvancedErrorClassifier {
    private errorPatterns;
    constructor();
    /**
     * Initialize default error patterns
     */
    private initializeDefaultPatterns;
    /**
     * Add custom error pattern
     */
    addPattern(id: string, pattern: {
        pattern: RegExp | ((error: any) => boolean);
        category: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
        recoverySuggestions: string[];
    }): void;
    /**
     * Remove error pattern
     */
    removePattern(id: string): boolean;
    /**
     * Classify error using pattern matching
     */
    classifyError(error: any): {
        category: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
        recoverySuggestions: string[];
        confidence: number;
    };
    /**
     * Calculate confidence score for pattern match
     */
    private calculateConfidence;
    /**
     * Get all error patterns
     */
    getAllPatterns(): Map<string, any>;
}
/**
 * Advanced Error Recovery System
 */
export declare class AdvancedErrorRecovery {
    private recoveryStrategies;
    constructor();
    /**
     * Initialize default recovery strategies
     */
    private initializeDefaultStrategies;
    /**
     * Add custom recovery strategy
     */
    addStrategy(id: string, strategy: {
        strategy: (error: any, context: any) => Promise<boolean>;
        applicableCategories: string[];
        description: string;
    }): void;
    /**
     * Remove recovery strategy
     */
    removeStrategy(id: string): boolean;
    /**
     * Attempt error recovery using appropriate strategies
     */
    attemptRecovery(error: any, classification: {
        category: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
    }, context?: any): Promise<{
        success: boolean;
        strategiesAttempted: string[];
        finalError?: any;
    }>;
    /**
     * Get recovery strategy suggestions for error category
     */
    getRecoverySuggestions(category: string): string[];
    /**
     * Get all recovery strategies
     */
    getAllStrategies(): Map<string, any>;
}
/**
 * Advanced Error Reporting System
 */
export declare class AdvancedErrorReporter {
    private errorQueue;
    private reportingDestinations;
    constructor();
    /**
     * Initialize default reporting destinations
     */
    private initializeDefaultDestinations;
    /**
     * Add reporting destination
     */
    addDestination(id: string, destination: {
        reporter: (errorData: any) => Promise<boolean>;
        enabled: boolean;
        maxRetries: number;
    }): void;
    /**
     * Remove reporting destination
     */
    removeDestination(id: string): boolean;
    /**
     * Enable/disable reporting destination
     */
    setDestinationEnabled(id: string, enabled: boolean): boolean;
    /**
     * Queue error for reporting
     */
    queueError(errorData: any): string;
    /**
     * Process error queue
     */
    processQueue(): Promise<{
        successCount: number;
        failureCount: number;
        totalProcessed: number;
    }>;
    /**
     * Report error to all enabled destinations
     */
    private reportToDestinations;
    /**
     * Get max retries for error based on destinations
     */
    private getMaxRetriesForError;
    /**
     * Get queue status
     */
    getQueueStatus(): {
        queueSize: number;
        oldestErrorTimestamp?: number;
        destinations: Array<{
            id: string;
            enabled: boolean;
            maxRetries: number;
        }>;
    };
    /**
     * Clear error queue
     */
    clearQueue(): void;
    /**
     * Get all reporting destinations
     */
    getAllDestinations(): Map<string, any>;
}
/**
 * Comprehensive Error Handling System
 * Combines classification, recovery, and reporting
 */
export declare class ComprehensiveErrorHandler {
    private classifier;
    private recovery;
    private reporter;
    private errorHistory;
    constructor();
    /**
     * Handle error with comprehensive processing
     */
    handleError(error: any, context?: any, options?: {
        autoRecover?: boolean;
        autoReport?: boolean;
        metadata?: Record<string, any>;
    }): Promise<{
        errorId: string;
        classification: any;
        recoveryResult?: any;
        reportingResult?: any;
    }>;
    /**
     * Prepare error data for reporting
     */
    private prepareErrorData;
    /**
     * Notify error based on severity
     */
    private notifyError;
    /**
     * Process error queue
     */
    processErrorQueue(): Promise<any>;
    /**
     * Get error history
     */
    getErrorHistory(limit?: number): typeof this.errorHistory;
    /**
     * Clear error history
     */
    clearErrorHistory(): void;
    /**
     * Get error statistics
     */
    getErrorStatistics(): {
        totalErrors: number;
        bySeverity: Record<string, number>;
        byCategory: Record<string, number>;
        recoverySuccessRate: number;
        reportingSuccessRate: number;
    };
    /**
     * Get queue status
     */
    getQueueStatus(): any;
    /**
     * Add custom error pattern
     */
    addErrorPattern(id: string, pattern: any): void;
    /**
     * Add custom recovery strategy
     */
    addRecoveryStrategy(id: string, strategy: any): void;
    /**
     * Add reporting destination
     */
    addReportingDestination(id: string, destination: any): void;
    /**
     * Get all components
     */
    getComponents(): {
        classifier: AdvancedErrorClassifier;
        recovery: AdvancedErrorRecovery;
        reporter: AdvancedErrorReporter;
    };
}
