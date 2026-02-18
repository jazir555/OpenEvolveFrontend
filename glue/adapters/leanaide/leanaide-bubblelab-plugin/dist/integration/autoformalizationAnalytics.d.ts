import { type LeanAideVerificationProps } from '../components';
export type AutoformalizationStrategy = 'auto' | 'theorem' | 'definition' | 'verification' | 'query' | 'elaboration';
export interface AutoformalizationConfig {
    serverUrl?: string;
    ragbitsUrl?: string;
    apiKey?: string;
    enableAnalytics?: boolean;
    defaultDomain?: string;
    defaultStrategy?: AutoformalizationStrategy;
}
export interface AutoformalizationResult {
    success: boolean;
    mode: LeanAideVerificationProps['mode'];
    data?: unknown;
    error?: string;
    logs?: string;
    confidence?: number;
    processingTimeMs: number;
}
export interface AutoformalizationRequest {
    input: string;
    mode?: LeanAideVerificationProps['mode'];
    solutionCode?: string;
    context?: string;
}
export interface AutoformalizationEvent {
    id: string;
    timestamp: Date;
    eventType: 'conversion_start' | 'conversion_success' | 'conversion_failure' | 'verification_start' | 'verification_success' | 'verification_failure';
    input: string;
    output?: string;
    strategyUsed: AutoformalizationStrategy;
    domain: string;
    confidenceScore?: number;
    processingTime?: number;
    error?: string;
    metadata?: Record<string, unknown>;
}
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
}
export interface BubbleLabAnalyticsConfig {
    enableRealTimeTracking: boolean;
    enablePerformanceMetrics: boolean;
    enableErrorTracking: boolean;
    enableDomainAnalysis: boolean;
    enableStrategyComparison: boolean;
}
export declare const DEFAULT_ANALYTICS_CONFIG: BubbleLabAnalyticsConfig;
export declare class LeanAideAutoformalizationEngine {
    private readonly config;
    constructor(config?: AutoformalizationConfig);
    run(request: AutoformalizationRequest): Promise<AutoformalizationResult>;
}
export declare function create_leanaide_autoformalization_engine(config?: AutoformalizationConfig): LeanAideAutoformalizationEngine;
export declare function autoformalize_with_mdap_maker(input: string, config?: AutoformalizationConfig): Promise<AutoformalizationResult>;
export declare function useAutoformalizationAnalytics(): {
    events: any;
    metrics: any;
    isLoading: boolean;
    trackEvent: any;
    getMetrics: any;
    reset: any;
};
export interface EnhancedLeanAideVerificationProps extends LeanAideVerificationProps {
    enableAnalytics?: boolean;
    onAnalyticsEvent?: (event: AutoformalizationEvent) => void;
    strategy?: AutoformalizationStrategy;
    domain?: string;
}
export declare function EnhancedLeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode, className, enableAnalytics, onAnalyticsEvent, strategy, domain, }: EnhancedLeanAideVerificationProps): any;
export interface AnalyticsDashboardProps {
    className?: string;
    metrics?: AutoformalizationMetrics;
}
export declare function AnalyticsDashboard({ className, metrics: externalMetrics }: AnalyticsDashboardProps): any;
export interface KnowledgeGraphIntegrationProps {
    className?: string;
}
export declare function KnowledgeGraphIntegration({ className }: KnowledgeGraphIntegrationProps): any;
export interface LeanAideBubbleLabIntegrationProps {
    className?: string;
}
export declare function LeanAideBubbleLabIntegration({ className }: LeanAideBubbleLabIntegrationProps): any;
export default LeanAideBubbleLabIntegration;
