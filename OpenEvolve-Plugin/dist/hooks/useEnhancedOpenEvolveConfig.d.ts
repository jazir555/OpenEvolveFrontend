import { EnhancedOpenEvolvePluginState, EnhancedOpenEvolvePlugin } from '../types/enhanced-plugin-types';
/**
 * Custom hook for managing enhanced OpenEvolve configuration
 * Provides state management and utility functions for enhanced features
 */
export declare function useEnhancedOpenEvolveConfig(initialConfig?: Partial<EnhancedOpenEvolvePluginState>): {
    config: EnhancedOpenEvolvePluginState;
    plugin: EnhancedOpenEvolvePlugin;
    isLoading: boolean;
    error: Error | null;
    updateConfig: (updates: Partial<EnhancedOpenEvolvePluginState>) => boolean;
    resetConfig: () => boolean;
    validateAll: () => Record<string, boolean>;
    executeWithEnhancedFeatures: (goal: string, options?: {
        performanceProfile?: string;
        securityProfile?: string;
        monitoringEnabled?: boolean;
        integrationMode?: 'auto' | 'manual' | 'disabled';
    }) => Promise<{
        success: boolean;
        result?: any;
        performanceMetrics?: any;
        securityStatus?: any;
        monitoringData?: any;
        integrationResults?: any;
        error?: Error;
    }>;
};
/**
 * Custom hook for performance management
 */
export declare function usePerformanceManagement(): {
    getPerformanceMetrics: () => any;
    getMemoryUsage: () => any;
    getCacheStats: () => any;
};
/**
 * Custom hook for security management
 */
export declare function useSecurityManagement(): {
    getSecurityStatus: () => any;
};
/**
 * Custom hook for integration management
 */
export declare function useIntegrationManagement(): {
    getIntegrationStatus: () => any;
    setupIntegrations: (autoMode?: boolean) => Promise<any>;
    cleanupIntegrations: () => Promise<void>;
};
/**
 * Custom hook for monitoring management
 */
export declare function useMonitoringManagement(): {
    startMonitoring: () => any;
    stopMonitoring: () => void;
    getMonitoringData: () => any;
};
/**
 * Custom hook for error handling management
 */
export declare function useErrorHandlingManagement(): {
    handleError: (error: unknown, options?: {
        errorId?: string;
        context?: string;
        severity?: 'low' | 'medium' | 'high' | 'critical';
    }) => void;
    classifyError: (errorType: string, errorMessage: string) => string;
    logError: (errorData: {
        errorId: string;
        errorType: string;
        errorMessage: string;
        context: string;
        severity: string;
        classification: string;
        timestamp: number;
    }) => void;
    reportError: (errorData: {
        errorId: string;
        errorType: string;
        errorMessage: string;
        context: string;
        severity: string;
        classification: string;
    }) => void;
    attemptErrorRecovery: (errorId: string, errorType: string, errorMessage: string, context: string) => void;
};
/**
 * Custom hook for profile management
 */
export declare function useProfileManagement(): {
    addPerformanceProfile: (profileName: string, profileConfig: any) => boolean;
    addSecurityProfile: (profileName: string, profileConfig: any) => boolean;
    removePerformanceProfile: (profileName: string) => boolean;
    removeSecurityProfile: (profileName: string) => boolean;
};
/**
 * Custom hook for statistics management
 */
export declare function useStatisticsManagement(): {
    getExecutionStatistics: () => any;
    getErrorStatistics: () => any;
    getValidationHistory: () => any;
    clearValidationHistory: () => boolean;
};
