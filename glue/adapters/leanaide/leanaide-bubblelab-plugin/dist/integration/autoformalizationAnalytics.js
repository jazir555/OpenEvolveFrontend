import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useMemo, useRef, useState } from 'react';
import { BarChart3, Clock, Database, Shield } from 'lucide-react';
import { LeanAideVerification, RagbitsKnowledgeSearch, } from '../components';
import { elaborateCode, initializeLeanAideClient, initializeRagbitsClient, mathQuery, translateDefinition, translateTheorem, verifySolution, } from '../services';
export const DEFAULT_ANALYTICS_CONFIG = {
    enableRealTimeTracking: true,
    enablePerformanceMetrics: true,
    enableErrorTracking: true,
    enableDomainAnalysis: true,
    enableStrategyComparison: true,
};
const INITIAL_METRICS = {
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
    constructor(config = {}) {
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
    async run(request) {
        const start = Date.now();
        const mode = request.mode ?? 'theorem';
        try {
            let result;
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
        }
        catch (error) {
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
export function create_leanaide_autoformalization_engine(config = {}) {
    return new LeanAideAutoformalizationEngine(config);
}
export async function autoformalize_with_mdap_maker(input, config = {}) {
    const engine = new LeanAideAutoformalizationEngine(config);
    return engine.run({ input, mode: 'theorem' });
}
function applyEventToMetrics(previous, event) {
    const next = {
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
    }
    else {
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
    const [events, setEvents] = useState([]);
    const [metrics, setMetrics] = useState(INITIAL_METRICS);
    const trackEvent = useCallback((event) => {
        setEvents((previous) => [...previous, event]);
        setMetrics((previous) => applyEventToMetrics(previous, event));
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
export function EnhancedLeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode = 'verification', className = '', enableAnalytics = true, onAnalyticsEvent, strategy = 'auto', domain = 'general', }) {
    const startedAtRef = useRef(Date.now());
    const { trackEvent } = useAutoformalizationAnalytics();
    const handleVerificationResult = useCallback((result) => {
        const processingTime = Math.max(Date.now() - startedAtRef.current, 0);
        if (enableAnalytics) {
            const event = {
                id: `event_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                timestamp: new Date(),
                eventType: result.success ? 'conversion_success' : 'conversion_failure',
                input: problemStatement,
                output: result.data ? JSON.stringify(result.data) : undefined,
                strategyUsed: strategy,
                domain,
                confidenceScore: typeof result.data?.confidence === 'number' ? result.data.confidence : undefined,
                processingTime,
                error: result.error,
                metadata: { mode },
            };
            trackEvent(event);
            onAnalyticsEvent?.(event);
        }
        onVerificationResult?.(result);
    }, [
        domain,
        enableAnalytics,
        mode,
        onAnalyticsEvent,
        onVerificationResult,
        problemStatement,
        strategy,
        trackEvent,
    ]);
    return (_jsx("div", { className: className, children: _jsx(LeanAideVerification, { problemStatement: problemStatement, solutionCode: solutionCode, mode: mode, onVerificationResult: handleVerificationResult }) }));
}
export function AnalyticsDashboard({ className = '', metrics: externalMetrics }) {
    const analytics = useAutoformalizationAnalytics();
    const metrics = externalMetrics ?? analytics.metrics;
    const avgConfidence = useMemo(() => {
        if (metrics.confidenceScores.length === 0) {
            return 0;
        }
        const total = metrics.confidenceScores.reduce((sum, value) => sum + value, 0);
        return total / metrics.confidenceScores.length;
    }, [metrics.confidenceScores]);
    return (_jsxs("div", { className: `rounded-lg border bg-white p-6 ${className}`, children: [_jsxs("div", { className: "mb-4 flex items-center gap-2", children: [_jsx(BarChart3, { className: "h-5 w-5 text-blue-600" }), _jsx("h2", { className: "text-lg font-semibold text-gray-900", children: "Autoformalization Analytics" })] }), _jsxs("div", { className: "grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4", children: [_jsx(MetricCard, { label: "Attempts", value: String(metrics.totalAttempts) }), _jsx(MetricCard, { label: "Success Rate", value: `${Math.round(metrics.successRate * 100)}%` }), _jsx(MetricCard, { label: "Avg Time", value: `${Math.round(metrics.averageProcessingTime)}ms`, icon: _jsx(Clock, { className: "h-4 w-4 text-gray-500" }) }), _jsx(MetricCard, { label: "Avg Confidence", value: avgConfidence.toFixed(2) })] })] }));
}
function MetricCard({ label, value, icon }) {
    return (_jsxs("div", { className: "rounded-md border bg-gray-50 p-3", children: [_jsxs("div", { className: "mb-1 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-gray-500", children: [_jsx("span", { children: label }), icon] }), _jsx("div", { className: "text-xl font-semibold text-gray-900", children: value })] }));
}
export function KnowledgeGraphIntegration({ className = '' }) {
    const [results, setResults] = useState([]);
    return (_jsxs("div", { className: `rounded-lg border bg-white p-6 ${className}`, children: [_jsxs("div", { className: "mb-4 flex items-center gap-2", children: [_jsx(Database, { className: "h-5 w-5 text-indigo-600" }), _jsx("h2", { className: "text-lg font-semibold text-gray-900", children: "Knowledge Graph Search" })] }), _jsx(RagbitsKnowledgeSearch, { onResults: setResults }), results.length > 0 && (_jsxs("p", { className: "mt-3 text-xs text-gray-500", children: ["Received ", results.length, " result(s) from RAGBits."] }))] }));
}
export function LeanAideBubbleLabIntegration({ className = '' }) {
    const [activeTab, setActiveTab] = useState('dashboard');
    const analytics = useAutoformalizationAnalytics();
    return (_jsxs("div", { className: `space-y-4 rounded-xl border bg-gray-50 p-4 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 text-xl font-semibold text-gray-900", children: [_jsx(Shield, { className: "h-6 w-6 text-blue-600" }), "LeanAide BubbleLab Integration"] }), _jsx("div", { className: "flex gap-2", children: ['dashboard', 'verification', 'knowledge'].map((tab) => (_jsx("button", { onClick: () => setActiveTab(tab), className: `rounded-md px-3 py-2 text-sm font-medium ${activeTab === tab ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'}`, children: tab.charAt(0).toUpperCase() + tab.slice(1) }, tab))) }), activeTab === 'dashboard' && _jsx(AnalyticsDashboard, { metrics: analytics.metrics }), activeTab === 'verification' && (_jsx(EnhancedLeanAideVerification, { problemStatement: "Prove that for all natural numbers n, n + 0 = n", mode: "theorem", strategy: "auto", domain: "arithmetic", enableAnalytics: true, onAnalyticsEvent: analytics.trackEvent })), activeTab === 'knowledge' && _jsx(KnowledgeGraphIntegration, {})] }));
}
export default LeanAideBubbleLabIntegration;
//# sourceMappingURL=autoformalizationAnalytics.js.map