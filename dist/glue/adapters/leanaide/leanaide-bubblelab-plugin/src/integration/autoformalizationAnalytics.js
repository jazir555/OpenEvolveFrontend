"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAideAutoformalizationEngine = exports.DEFAULT_ANALYTICS_CONFIG = void 0;
exports.create_leanaide_autoformalization_engine = create_leanaide_autoformalization_engine;
exports.autoformalize_with_mdap_maker = autoformalize_with_mdap_maker;
exports.useAutoformalizationAnalytics = useAutoformalizationAnalytics;
exports.EnhancedLeanAideVerification = EnhancedLeanAideVerification;
exports.AnalyticsDashboard = AnalyticsDashboard;
exports.KnowledgeGraphIntegration = KnowledgeGraphIntegration;
exports.LeanAideBubbleLabIntegration = LeanAideBubbleLabIntegration;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const components_1 = require("../components");
const services_1 = require("../services");
exports.DEFAULT_ANALYTICS_CONFIG = {
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
class LeanAideAutoformalizationEngine {
    constructor(config = {}) {
        this.config = {
            defaultDomain: config.defaultDomain ?? 'general',
            defaultStrategy: config.defaultStrategy ?? 'auto',
            ...config,
        };
        if (config.serverUrl || config.apiKey) {
            (0, services_1.initializeLeanAideClient)({ serverUrl: config.serverUrl, apiKey: config.apiKey });
        }
        if (config.ragbitsUrl || config.apiKey) {
            (0, services_1.initializeRagbitsClient)({ serverUrl: config.ragbitsUrl, apiKey: config.apiKey });
        }
    }
    async run(request) {
        const start = Date.now();
        const mode = request.mode ?? 'theorem';
        try {
            let result;
            switch (mode) {
                case 'theorem':
                    result = await (0, services_1.translateTheorem)(request.input, request.context);
                    break;
                case 'definition':
                    result = await (0, services_1.translateDefinition)(request.input, request.context);
                    break;
                case 'verification':
                    if (!request.solutionCode) {
                        throw new Error('solutionCode is required for verification mode');
                    }
                    result = await (0, services_1.verifySolution)(request.input, request.solutionCode, request.context);
                    break;
                case 'query':
                    result = await (0, services_1.mathQuery)(request.input, request.context);
                    break;
                case 'elaboration':
                    result = await (0, services_1.elaborateCode)(request.input, request.context);
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
exports.LeanAideAutoformalizationEngine = LeanAideAutoformalizationEngine;
function create_leanaide_autoformalization_engine(config = {}) {
    return new LeanAideAutoformalizationEngine(config);
}
async function autoformalize_with_mdap_maker(input, config = {}) {
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
function useAutoformalizationAnalytics() {
    const [events, setEvents] = (0, react_1.useState)([]);
    const [metrics, setMetrics] = (0, react_1.useState)(INITIAL_METRICS);
    const trackEvent = (0, react_1.useCallback)((event) => {
        setEvents((previous) => [...previous, event]);
        setMetrics((previous) => applyEventToMetrics(previous, event));
    }, []);
    const reset = (0, react_1.useCallback)(() => {
        setEvents([]);
        setMetrics(INITIAL_METRICS);
    }, []);
    const getMetrics = (0, react_1.useCallback)(() => metrics, [metrics]);
    return {
        events,
        metrics,
        isLoading: false,
        trackEvent,
        getMetrics,
        reset,
    };
}
function EnhancedLeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode = 'verification', className = '', enableAnalytics = true, onAnalyticsEvent, strategy = 'auto', domain = 'general', }) {
    const startedAtRef = (0, react_1.useRef)(Date.now());
    const { trackEvent } = useAutoformalizationAnalytics();
    const handleVerificationResult = (0, react_1.useCallback)((result) => {
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
    return (<div className={className}>
      <components_1.LeanAideVerification problemStatement={problemStatement} solutionCode={solutionCode} mode={mode} onVerificationResult={handleVerificationResult}/>
    </div>);
}
function AnalyticsDashboard({ className = '', metrics: externalMetrics }) {
    const analytics = useAutoformalizationAnalytics();
    const metrics = externalMetrics ?? analytics.metrics;
    const avgConfidence = (0, react_1.useMemo)(() => {
        if (metrics.confidenceScores.length === 0) {
            return 0;
        }
        const total = metrics.confidenceScores.reduce((sum, value) => sum + value, 0);
        return total / metrics.confidenceScores.length;
    }, [metrics.confidenceScores]);
    return (<div className={`rounded-lg border bg-white p-6 ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <lucide_react_1.BarChart3 className="h-5 w-5 text-blue-600"/>
        <h2 className="text-lg font-semibold text-gray-900">Autoformalization Analytics</h2>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Attempts" value={String(metrics.totalAttempts)}/>
        <MetricCard label="Success Rate" value={`${Math.round(metrics.successRate * 100)}%`}/>
        <MetricCard label="Avg Time" value={`${Math.round(metrics.averageProcessingTime)}ms`} icon={<lucide_react_1.Clock className="h-4 w-4 text-gray-500"/>}/>
        <MetricCard label="Avg Confidence" value={avgConfidence.toFixed(2)}/>
      </div>
    </div>);
}
function MetricCard({ label, value, icon }) {
    return (<div className="rounded-md border bg-gray-50 p-3">
      <div className="mb-1 flex items-center justify-between text-xs font-medium uppercase tracking-wide text-gray-500">
        <span>{label}</span>
        {icon}
      </div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
    </div>);
}
function KnowledgeGraphIntegration({ className = '' }) {
    const [results, setResults] = (0, react_1.useState)([]);
    return (<div className={`rounded-lg border bg-white p-6 ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <lucide_react_1.Database className="h-5 w-5 text-indigo-600"/>
        <h2 className="text-lg font-semibold text-gray-900">Knowledge Graph Search</h2>
      </div>

      <components_1.RagbitsKnowledgeSearch onResults={setResults}/>

      {results.length > 0 && (<p className="mt-3 text-xs text-gray-500">Received {results.length} result(s) from RAGBits.</p>)}
    </div>);
}
function LeanAideBubbleLabIntegration({ className = '' }) {
    const [activeTab, setActiveTab] = (0, react_1.useState)('dashboard');
    const analytics = useAutoformalizationAnalytics();
    return (<div className={`space-y-4 rounded-xl border bg-gray-50 p-4 ${className}`}>
      <div className="flex items-center gap-2 text-xl font-semibold text-gray-900">
        <lucide_react_1.Shield className="h-6 w-6 text-blue-600"/>
        LeanAide BubbleLab Integration
      </div>

      <div className="flex gap-2">
        {['dashboard', 'verification', 'knowledge'].map((tab) => (<button key={tab} onClick={() => setActiveTab(tab)} className={`rounded-md px-3 py-2 text-sm font-medium ${activeTab === tab ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'}`}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>))}
      </div>

      {activeTab === 'dashboard' && <AnalyticsDashboard metrics={analytics.metrics}/>}

      {activeTab === 'verification' && (<EnhancedLeanAideVerification problemStatement="Prove that for all natural numbers n, n + 0 = n" mode="theorem" strategy="auto" domain="arithmetic" enableAnalytics onAnalyticsEvent={analytics.trackEvent}/>)}

      {activeTab === 'knowledge' && <KnowledgeGraphIntegration />}
    </div>);
}
exports.default = LeanAideBubbleLabIntegration;
//# sourceMappingURL=autoformalizationAnalytics.js.map