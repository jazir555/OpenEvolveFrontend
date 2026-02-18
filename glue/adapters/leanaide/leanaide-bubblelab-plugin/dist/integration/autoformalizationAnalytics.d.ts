/**
 * LeanAide Autoformalization System with BubbleLab Analytics Integration
 *
 * This module provides comprehensive integration between the LeanAide autoformalization system
 * and BubbleLab analytics platform, enabling advanced visualization and monitoring of
 * mathematical formalization processes.
 */
import { LeanAideVerificationProps } from './components';
export interface AutoformalizationMetrics {
    totalAttempts: number;
    successfulConversions: number;
    failedConversions: number;
    averageProcessingTime: number;
    successRate: number;
    confidenceScores: number[];
    domainDistribution: Record<string, number>;
    strategyUsage: Record<string, number>;
    errorPatterns: Record<string, number>;
    performanceByDomain: Record<string, {
        successRate: number;
        avgTime: number;
        count: number;
    }>;
}
export interface AutoformalizationEvent {
    id: string;
    timestamp: Date;
    eventType: 'conversion_start' | 'conversion_success' | 'conversion_failure' | 'verification_start' | 'verification_success' | 'verification_failure';
    input: string;
    output?: string;
    strategyUsed: string;
    domain: string;
    confidenceScore?: number;
    processingTime?: number;
    error?: string;
    metadata?: Record<string, any>;
}
export interface BubbleLabAnalyticsConfig {
    enableRealTimeTracking: boolean;
    enablePerformanceMetrics: boolean;
    enableErrorTracking: boolean;
    enableDomainAnalysis: boolean;
    enableStrategyComparison: boolean;
    retentionPeriodDays: number;
    batchSize: number;
    flushIntervalMs: number;
}
export declare function useAutoformalizationAnalytics(): {
    metrics: any;
    events: any;
    isLoading: any;
    trackEvent: any;
    getMetrics: any;
};
export interface EnhancedLeanAideVerificationProps extends LeanAideVerificationProps {
    enableAnalytics?: boolean;
    onAnalyticsEvent?: (event: AutoformalizationEvent) => void;
    strategy?: string;
    domain?: string;
}
export declare function EnhancedLeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode, className, enableAnalytics, onAnalyticsEvent, strategy, domain, }: EnhancedLeanAideVerificationProps): any;
export interface AnalyticsDashboardProps {
    className?: string;
}
export declare function AnalyticsDashboard({ className }: AnalyticsDashboardProps): any;
export interface KnowledgeGraphIntegrationProps {
    className?: string;
}
export declare function KnowledgeGraphIntegration({ className }: KnowledgeGraphIntegrationProps): any;
export interface LeanAideBubbleLabIntegrationProps {
    className?: string;
}
export declare function LeanAideBubbleLabIntegration({ className }: LeanAideBubbleLabIntegrationProps): any;
export default LeanAideBubbleLabIntegration;
