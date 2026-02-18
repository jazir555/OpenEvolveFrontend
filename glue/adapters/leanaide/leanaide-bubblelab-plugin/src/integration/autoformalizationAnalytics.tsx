import React, { useCallback, useMemo, useRef, useState } from 'react';
import { BarChart3, Clock, Database, Shield } from 'lucide-react';

import {
  LeanAideVerification,
  type LeanAideVerificationProps,
  RagbitsKnowledgeSearch,
} from '../components';
import {
  elaborateCode,
  initializeLeanAideClient,
  initializeRagbitsClient,
  mathQuery,
  searchKnowledge,
  translateDefinition,
  translateTheorem,
  verifySolution,
} from '../services';
import type { LeanAideTaskResponse, RagbitsSearchResult } from '../lib';

export type AutoformalizationStrategy =
  | 'auto'
  | 'theorem'
  | 'definition'
  | 'verification'
  | 'query'
  | 'elaboration';

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
  eventType:
    | 'conversion_start'
    | 'conversion_success'
    | 'conversion_failure'
    | 'verification_start'
    | 'verification_success'
    | 'verification_failure';
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

export const DEFAULT_ANALYTICS_CONFIG: BubbleLabAnalyticsConfig = {
  enableRealTimeTracking: true,
  enablePerformanceMetrics: true,
  enableErrorTracking: true,
  enableDomainAnalysis: true,
  enableStrategyComparison: true,
};

const INITIAL_METRICS: AutoformalizationMetrics = {
  totalAttempts: 0,
  successfulConversions: 0,
  failedConversions: 0,
  averageProcessingTime: 0,
  successRate: 0,
  confidenceScores: [],
  domainDistribution: {},
  strategyUsage: {},
  errorPatterns: {},
};

export class LeanAideAutoformalizationEngine {
  private readonly config: Required<Pick<AutoformalizationConfig, 'defaultDomain' | 'defaultStrategy'>> &
    AutoformalizationConfig;

  constructor(config: AutoformalizationConfig = {}) {
    this.config = {
      defaultDomain: config.defaultDomain ?? 'general',
      defaultStrategy: config.defaultStrategy ?? 'auto',
      ...config,
    };

    if (config.serverUrl || config.apiKey) {
      initializeLeanAideClient({ serverUrl: config.serverUrl, apiKey: config.apiKey });
    }

    if (config.ragbitsUrl || config.apiKey) {
      initializeRagbitsClient({ serverUrl: config.ragbitsUrl, apiKey: config.apiKey });
    }
  }

  async run(request: AutoformalizationRequest): Promise<AutoformalizationResult> {
    const start = Date.now();
    const mode = request.mode ?? 'theorem';

    try {
      let result: LeanAideTaskResponse;

      switch (mode) {
        case 'theorem':
          result = await translateTheorem(request.input, request.context);
          break;
        case 'definition':
          result = await translateDefinition(request.input, request.context);
          break;
        case 'verification':
          if (!request.solutionCode) {
            throw new Error('solutionCode is required for verification mode');
          }
          result = await verifySolution(request.input, request.solutionCode, request.context);
          break;
        case 'query':
          result = await mathQuery(request.input, request.context);
          break;
        case 'elaboration':
          result = await elaborateCode(request.input, request.context);
          break;
        default:
          throw new Error(`Unsupported mode: ${String(mode)}`);
      }

      const processingTimeMs = Date.now() - start;
      return {
        success: result.success,
        mode,
        data: result.data,
        error: result.error,
        logs: result.logs,
        confidence: typeof result.data?.confidence === 'number' ? result.data.confidence : undefined,
        processingTimeMs,
      };
    } catch (error) {
      const processingTimeMs = Date.now() - start;
      return {
        success: false,
        mode,
        error: error instanceof Error ? error.message : 'Unknown autoformalization error',
        processingTimeMs,
      };
    }
  }
}

export function create_leanaide_autoformalization_engine(
  config: AutoformalizationConfig = {}
): LeanAideAutoformalizationEngine {
  return new LeanAideAutoformalizationEngine(config);
}

export async function autoformalize_with_mdap_maker(
  input: string,
  config: AutoformalizationConfig = {}
): Promise<AutoformalizationResult> {
  const engine = new LeanAideAutoformalizationEngine(config);
  return engine.run({ input, mode: 'theorem' });
}

function applyEventToMetrics(
  previous: AutoformalizationMetrics,
  event: AutoformalizationEvent
): AutoformalizationMetrics {
  const next: AutoformalizationMetrics = {
    ...previous,
    confidenceScores: [...previous.confidenceScores],
    domainDistribution: { ...previous.domainDistribution },
    strategyUsage: { ...previous.strategyUsage },
    errorPatterns: { ...previous.errorPatterns },
  };

  if (event.eventType === 'conversion_start' || event.eventType === 'verification_start') {
    return next;
  }

  next.totalAttempts += 1;

  if (event.eventType === 'conversion_success' || event.eventType === 'verification_success') {
    next.successfulConversions += 1;
  } else {
    next.failedConversions += 1;
  }

  next.successRate = next.totalAttempts === 0 ? 0 : next.successfulConversions / next.totalAttempts;

  if (typeof event.processingTime === 'number') {
    const priorTotal = previous.averageProcessingTime * Math.max(previous.totalAttempts, 0);
    next.averageProcessingTime = (priorTotal + event.processingTime) / next.totalAttempts;
  }

  if (typeof event.confidenceScore === 'number') {
    next.confidenceScores.push(event.confidenceScore);
  }

  next.domainDistribution[event.domain] = (next.domainDistribution[event.domain] ?? 0) + 1;
  next.strategyUsage[event.strategyUsed] = (next.strategyUsage[event.strategyUsed] ?? 0) + 1;

  if (event.error) {
    const key = event.error.slice(0, 80);
    next.errorPatterns[key] = (next.errorPatterns[key] ?? 0) + 1;
  }

  return next;
}

export function useAutoformalizationAnalytics() {
  const [events, setEvents] = useState<AutoformalizationEvent[]>([]);
  const [metrics, setMetrics] = useState<AutoformalizationMetrics>(INITIAL_METRICS);

  const trackEvent = useCallback((event: AutoformalizationEvent) => {
    setEvents((previous: AutoformalizationEvent[]) => [...previous, event]);
    setMetrics((previous: AutoformalizationMetrics) => applyEventToMetrics(previous, event));
  }, []);

  const reset = useCallback(() => {
    setEvents([]);
    setMetrics(INITIAL_METRICS);
  }, []);

  const getMetrics = useCallback(() => metrics, [metrics]);

  return {
    events,
    metrics,
    isLoading: false,
    trackEvent,
    getMetrics,
    reset,
  };
}

export interface EnhancedLeanAideVerificationProps extends LeanAideVerificationProps {
  enableAnalytics?: boolean;
  onAnalyticsEvent?: (event: AutoformalizationEvent) => void;
  strategy?: AutoformalizationStrategy;
  domain?: string;
}

export function EnhancedLeanAideVerification({
  problemStatement,
  solutionCode,
  onVerificationResult,
  mode = 'verification',
  className = '',
  enableAnalytics = true,
  onAnalyticsEvent,
  strategy = 'auto',
  domain = 'general',
}: EnhancedLeanAideVerificationProps) {
  const startedAtRef = useRef<number>(Date.now());
  const { trackEvent } = useAutoformalizationAnalytics();

  const handleVerificationResult = useCallback(
    (result: LeanAideTaskResponse) => {
      const processingTime = Math.max(Date.now() - startedAtRef.current, 0);

      if (enableAnalytics) {
        const event: AutoformalizationEvent = {
          id: `event_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          timestamp: new Date(),
          eventType: result.success ? 'conversion_success' : 'conversion_failure',
          input: problemStatement,
          output: result.data ? JSON.stringify(result.data) : undefined,
          strategyUsed: strategy,
          domain,
          confidenceScore:
            typeof result.data?.confidence === 'number' ? result.data.confidence : undefined,
          processingTime,
          error: result.error,
          metadata: { mode },
        };

        trackEvent(event);
        onAnalyticsEvent?.(event);
      }

      onVerificationResult?.(result);
    },
    [
      domain,
      enableAnalytics,
      mode,
      onAnalyticsEvent,
      onVerificationResult,
      problemStatement,
      strategy,
      trackEvent,
    ]
  );

  return (
    <div className={className}>
      <LeanAideVerification
        problemStatement={problemStatement}
        solutionCode={solutionCode}
        mode={mode}
        onVerificationResult={handleVerificationResult}
      />
    </div>
  );
}

export interface AnalyticsDashboardProps {
  className?: string;
  metrics?: AutoformalizationMetrics;
}

export function AnalyticsDashboard({ className = '', metrics: externalMetrics }: AnalyticsDashboardProps) {
  const analytics = useAutoformalizationAnalytics();
  const metrics = externalMetrics ?? analytics.metrics;

  const avgConfidence = useMemo(() => {
    if (metrics.confidenceScores.length === 0) {
      return 0;
    }

    const total = metrics.confidenceScores.reduce((sum: number, value: number) => sum + value, 0);
    return total / metrics.confidenceScores.length;
  }, [metrics.confidenceScores]);

  return (
    <div className={`rounded-lg border bg-white p-6 ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-gray-900">Autoformalization Analytics</h2>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Attempts" value={String(metrics.totalAttempts)} />
        <MetricCard label="Success Rate" value={`${Math.round(metrics.successRate * 100)}%`} />
        <MetricCard label="Avg Time" value={`${Math.round(metrics.averageProcessingTime)}ms`} icon={<Clock className="h-4 w-4 text-gray-500" />} />
        <MetricCard label="Avg Confidence" value={avgConfidence.toFixed(2)} />
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  icon?: React.ReactNode;
}

function MetricCard({ label, value, icon }: MetricCardProps) {
  return (
    <div className="rounded-md border bg-gray-50 p-3">
      <div className="mb-1 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-gray-500">
        <span>{label}</span>
        {icon}
      </div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
    </div>
  );
}

export interface KnowledgeGraphIntegrationProps {
  className?: string;
}

export function KnowledgeGraphIntegration({ className = '' }: KnowledgeGraphIntegrationProps) {
  const [results, setResults] = useState<RagbitsSearchResult[]>([]);

  return (
    <div className={`rounded-lg border bg-white p-6 ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <Database className="h-5 w-5 text-indigo-600" />
        <h2 className="text-lg font-semibold text-gray-900">Knowledge Graph Search</h2>
      </div>

      <RagbitsKnowledgeSearch onResults={setResults} />

      {results.length > 0 && (
        <p className="mt-3 text-xs text-gray-500">Received {results.length} result(s) from RAGBits.</p>
      )}
    </div>
  );
}

export interface LeanAideBubbleLabIntegrationProps {
  className?: string;
}

export function LeanAideBubbleLabIntegration({ className = '' }: LeanAideBubbleLabIntegrationProps) {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'verification' | 'knowledge'>('dashboard');
  const analytics = useAutoformalizationAnalytics();

  return (
    <div className={`space-y-4 rounded-xl border bg-gray-50 p-4 ${className}`}>
      <div className="flex items-center gap-2 text-xl font-semibold text-gray-900">
        <Shield className="h-6 w-6 text-blue-600" />
        LeanAide BubbleLab Integration
      </div>

      <div className="flex gap-2">
        {['dashboard', 'verification', 'knowledge'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as 'dashboard' | 'verification' | 'knowledge')}
            className={`rounded-md px-3 py-2 text-sm font-medium ${
              activeTab === tab ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'dashboard' && <AnalyticsDashboard metrics={analytics.metrics} />}

      {activeTab === 'verification' && (
        <EnhancedLeanAideVerification
          problemStatement="Prove that for all natural numbers n, n + 0 = n"
          mode="theorem"
          strategy="auto"
          domain="arithmetic"
          enableAnalytics
          onAnalyticsEvent={analytics.trackEvent}
        />
      )}

      {activeTab === 'knowledge' && <KnowledgeGraphIntegration />}
    </div>
  );
}

export default LeanAideBubbleLabIntegration;
