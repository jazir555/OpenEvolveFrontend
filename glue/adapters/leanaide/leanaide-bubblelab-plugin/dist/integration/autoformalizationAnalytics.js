import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * LeanAide Autoformalization System with BubbleLab Analytics Integration
 *
 * This module provides comprehensive integration between the LeanAide autoformalization system
 * and BubbleLab analytics platform, enabling advanced visualization and monitoring of
 * mathematical formalization processes.
 */
import { useState, useEffect, useCallback } from 'react';
import { BarChart3, Activity, TrendingUp, Clock, CheckCircle, AlertTriangle, Database, Zap, Brain, Target, Shield, BarChart2, PieChart, LineChart, Settings } from 'lucide-react';
import { LeanAideVerification } from './components';
import { translateTheorem, translateDefinition, verifySolution, elaborateCode, mathQuery, searchKnowledge } from './services';
// Default configuration
const DEFAULT_ANALYTICS_CONFIG = {
    enableRealTimeTracking: true,
    enablePerformanceMetrics: true,
    enableErrorTracking: true,
    enableDomainAnalysis: true,
    enableStrategyComparison: true,
    retentionPeriodDays: 30,
    batchSize: 10,
    flushIntervalMs: 5000,
};
// Analytics service class
class BubbleLabAnalyticsService {
    constructor(config) {
        this.events = [];
        this.eventQueue = [];
        this.flushTimer = null;
        this.isInitialized = false;
        this.config = { ...DEFAULT_ANALYTICS_CONFIG, ...config };
        this.metrics = this.initializeMetrics();
    }
    initializeMetrics() {
        return {
            totalAttempts: 0,
            successfulConversions: 0,
            failedConversions: 0,
            averageProcessingTime: 0,
            successRate: 0,
            confidenceScores: [],
            domainDistribution: {},
            strategyUsage: {},
            errorPatterns: {},
            performanceByDomain: {},
        };
    }
    initialize() {
        if (this.isInitialized)
            return;
        // Set up periodic flushing
        if (this.config.flushIntervalMs > 0) {
            this.flushTimer = setInterval(() => {
                this.flushEvents();
            }, this.config.flushIntervalMs);
        }
        this.isInitialized = true;
    }
    shutdown() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
            this.flushTimer = null;
        }
        this.flushEvents();
        this.isInitialized = false;
    }
    trackEvent(event) {
        if (!this.config.enableRealTimeTracking)
            return;
        this.eventQueue.push(event);
        // Process immediately if batch is full
        if (this.eventQueue.length >= this.config.batchSize) {
            this.flushEvents();
        }
        // Update metrics
        this.updateMetrics(event);
    }
    updateMetrics(event) {
        this.metrics.totalAttempts++;
        // Update success/failure counts
        if (event.eventType.includes('success')) {
            this.metrics.successfulConversions++;
        }
        else if (event.eventType.includes('failure')) {
            this.metrics.failedConversions++;
        }
        // Update success rate
        if (this.metrics.totalAttempts > 0) {
            this.metrics.successRate = this.metrics.successfulConversions / this.metrics.totalAttempts;
        }
        // Update processing time
        if (event.processingTime !== undefined) {
            const totalTime = this.metrics.averageProcessingTime * (this.metrics.totalAttempts - 1) + event.processingTime;
            this.metrics.averageProcessingTime = totalTime / this.metrics.totalAttempts;
        }
        // Update confidence scores
        if (event.confidenceScore !== undefined) {
            this.metrics.confidenceScores.push(event.confidenceScore);
        }
        // Update domain distribution
        if (event.domain) {
            this.metrics.domainDistribution[event.domain] = (this.metrics.domainDistribution[event.domain] || 0) + 1;
            // Update performance by domain
            if (!this.metrics.performanceByDomain[event.domain]) {
                this.metrics.performanceByDomain[event.domain] = {
                    successRate: 0,
                    avgTime: 0,
                    count: 0
                };
            }
            const domainPerf = this.metrics.performanceByDomain[event.domain];
            domainPerf.count++;
            if (event.eventType.includes('success')) {
                domainPerf.successRate = (domainPerf.successRate * (domainPerf.count - 1) + 1) / domainPerf.count;
            }
            else {
                domainPerf.successRate = (domainPerf.successRate * (domainPerf.count - 1)) / domainPerf.count;
            }
            if (event.processingTime !== undefined) {
                domainPerf.avgTime = (domainPerf.avgTime * (domainPerf.count - 1) + event.processingTime) / domainPerf.count;
            }
        }
        // Update strategy usage
        if (event.strategyUsed) {
            this.metrics.strategyUsage[event.strategyUsed] = (this.metrics.strategyUsage[event.strategyUsed] || 0) + 1;
        }
        // Update error patterns
        if (event.error) {
            const errorKey = event.error.substring(0, 50); // Limit error key length
            this.metrics.errorPatterns[errorKey] = (this.metrics.errorPatterns[errorKey] || 0) + 1;
        }
    }
    flushEvents() {
        if (this.eventQueue.length === 0)
            return;
        // In a real implementation, this would send events to a backend service
        // For now, we'll just log them
        console.log('Flushing analytics events:', this.eventQueue);
        // Clear the queue
        this.eventQueue = [];
    }
    getMetrics() {
        return { ...this.metrics };
    }
    getEvents() {
        return [...this.events];
    }
    reset() {
        this.events = [];
        this.metrics = this.initializeMetrics();
        this.eventQueue = [];
    }
}
// Create a singleton instance
const analyticsService = new BubbleLabAnalyticsService();
// Hook for using analytics in components
export function useAutoformalizationAnalytics() {
    const [metrics, setMetrics] = useState(analyticsService.getMetrics());
    const [events, setEvents] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    useEffect(() => {
        analyticsService.initialize();
        // Set up metrics listener
        const interval = setInterval(() => {
            setMetrics(analyticsService.getMetrics());
        }, 1000); // Update metrics every second
        return () => {
            clearInterval(interval);
            analyticsService.shutdown();
        };
    }, []);
    const trackEvent = useCallback((event) => {
        analyticsService.trackEvent(event);
    }, []);
    const getMetrics = useCallback(() => {
        return analyticsService.getMetrics();
    }, []);
    return {
        metrics,
        events,
        isLoading,
        trackEvent,
        getMetrics,
    };
}
export function EnhancedLeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode = 'verification', className = '', enableAnalytics = true, onAnalyticsEvent, strategy = 'auto', domain = 'general', }) {
    const [processingTime, setProcessingTime] = useState(null);
    const { trackEvent } = useAutoformalizationAnalytics();
    const handleVerification = async () => {
        const startTime = Date.now();
        // Track start event
        if (enableAnalytics) {
            trackEvent({
                id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                timestamp: new Date(),
                eventType: 'conversion_start',
                input: problemStatement,
                strategyUsed: strategy,
                domain: domain,
                metadata: { mode }
            });
        }
        if (onAnalyticsEvent) {
            onAnalyticsEvent({
                id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                timestamp: new Date(),
                eventType: 'conversion_start',
                input: problemStatement,
                strategyUsed: strategy,
                domain: domain,
                metadata: { mode }
            });
        }
        try {
            // Perform verification using the original component's logic
            let result;
            switch (mode) {
                case 'theorem':
                    result = await translateTheorem(problemStatement);
                    break;
                case 'definition':
                    result = await translateDefinition(problemStatement);
                    break;
                case 'verification':
                    if (!solutionCode) {
                        throw new Error('Solution code is required for verification mode');
                    }
                    result = await verifySolution(problemStatement, solutionCode);
                    break;
                case 'query':
                    result = await mathQuery(problemStatement);
                    break;
                case 'elaboration':
                    if (!solutionCode) {
                        throw new Error('Lean code is required for elaboration mode');
                    }
                    result = await elaborateCode(solutionCode);
                    break;
                default:
                    throw new Error(`Unknown mode: ${mode}`);
            }
            const endTime = Date.now();
            const duration = endTime - startTime;
            setProcessingTime(duration);
            // Track success event
            if (enableAnalytics) {
                trackEvent({
                    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    timestamp: new Date(),
                    eventType: result.success ? 'conversion_success' : 'conversion_failure',
                    input: problemStatement,
                    output: result.data ? JSON.stringify(result.data) : undefined,
                    strategyUsed: strategy,
                    domain: domain,
                    confidenceScore: result.data?.confidence || 0.5,
                    processingTime: duration,
                    error: !result.success ? result.error : undefined,
                    metadata: { mode, success: result.success }
                });
            }
            if (onAnalyticsEvent) {
                onAnalyticsEvent({
                    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    timestamp: new Date(),
                    eventType: result.success ? 'conversion_success' : 'conversion_failure',
                    input: problemStatement,
                    output: result.data ? JSON.stringify(result.data) : undefined,
                    strategyUsed: strategy,
                    domain: domain,
                    confidenceScore: result.data?.confidence || 0.5,
                    processingTime: duration,
                    error: !result.success ? result.error : undefined,
                    metadata: { mode, success: result.success }
                });
            }
            if (onVerificationResult) {
                onVerificationResult(result);
            }
            return result;
        }
        catch (error) {
            const endTime = Date.now();
            const duration = endTime - startTime;
            setProcessingTime(duration);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            // Track failure event
            if (enableAnalytics) {
                trackEvent({
                    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    timestamp: new Date(),
                    eventType: 'conversion_failure',
                    input: problemStatement,
                    strategyUsed: strategy,
                    domain: domain,
                    processingTime: duration,
                    error: errorMessage,
                    metadata: { mode, error: errorMessage }
                });
            }
            if (onAnalyticsEvent) {
                onAnalyticsEvent({
                    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    timestamp: new Date(),
                    eventType: 'conversion_failure',
                    input: problemStatement,
                    strategyUsed: strategy,
                    domain: domain,
                    processingTime: duration,
                    error: errorMessage,
                    metadata: { mode, error: errorMessage }
                });
            }
            throw error;
        }
    };
    return (_jsxs("div", { className: `relative ${className}`, children: [_jsx(LeanAideVerification, { problemStatement: problemStatement, solutionCode: solutionCode, onVerificationResult: onVerificationResult, mode: mode, className: className }), processingTime !== null && (_jsxs("div", { className: "absolute top-2 right-2 bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full flex items-center gap-1", children: [_jsx(Clock, { className: "w-3 h-3" }), processingTime, "ms"] })), enableAnalytics && (_jsxs("div", { className: "absolute top-2 left-2 bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full flex items-center gap-1", children: [_jsx(BarChart3, { className: "w-3 h-3" }), "Analytics"] }))] }));
}
export function AnalyticsDashboard({ className = '' }) {
    const { metrics } = useAutoformalizationAnalytics();
    // Calculate derived metrics
    const successRatePercentage = Math.round(metrics.successRate * 100);
    const avgConfidence = metrics.confidenceScores.length > 0
        ? metrics.confidenceScores.reduce((a, b) => a + b, 0) / metrics.confidenceScores.length
        : 0;
    return (_jsxs("div", { className: `bg-white rounded-xl shadow-md p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 mb-6", children: [_jsx(BarChart2, { className: "w-6 h-6 text-blue-600" }), _jsx("h2", { className: "text-xl font-bold text-gray-800", children: "Autoformalization Analytics" })] }), _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6", children: [_jsx("div", { className: "bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium text-blue-600", children: "Total Attempts" }), _jsx("p", { className: "text-2xl font-bold text-blue-800", children: metrics.totalAttempts })] }), _jsx(Activity, { className: "w-8 h-8 text-blue-500" })] }) }), _jsx("div", { className: "bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium text-green-600", children: "Success Rate" }), _jsxs("p", { className: "text-2xl font-bold text-green-800", children: [successRatePercentage, "%"] })] }), _jsx(CheckCircle, { className: "w-8 h-8 text-green-500" })] }) }), _jsx("div", { className: "bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium text-purple-600", children: "Avg Time" }), _jsxs("p", { className: "text-2xl font-bold text-purple-800", children: [Math.round(metrics.averageProcessingTime), "ms"] })] }), _jsx(Clock, { className: "w-8 h-8 text-purple-500" })] }) }), _jsx("div", { className: "bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg border border-yellow-200", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium text-yellow-600", children: "Avg Confidence" }), _jsx("p", { className: "text-2xl font-bold text-yellow-800", children: avgConfidence.toFixed(2) })] }), _jsx(Target, { className: "w-8 h-8 text-yellow-500" })] }) })] }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsxs("h3", { className: "font-semibold text-gray-700 mb-3 flex items-center gap-2", children: [_jsx(PieChart, { className: "w-5 h-5" }), "Domain Distribution"] }), _jsx("div", { className: "space-y-2", children: Object.entries(metrics.domainDistribution).map(([domain, count]) => (_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { className: "text-sm text-gray-600 capitalize", children: domain }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-24 bg-gray-200 rounded-full h-2", children: _jsx("div", { className: "bg-blue-600 h-2 rounded-full", style: { width: `${(count / Math.max(1, metrics.totalAttempts)) * 100}%` } }) }), _jsx("span", { className: "text-sm font-medium text-gray-700", children: count })] })] }, domain))) })] }), _jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsxs("h3", { className: "font-semibold text-gray-700 mb-3 flex items-center gap-2", children: [_jsx(LineChart, { className: "w-5 h-5" }), "Strategy Usage"] }), _jsx("div", { className: "space-y-2", children: Object.entries(metrics.strategyUsage).map(([strategy, count]) => (_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { className: "text-sm text-gray-600 capitalize", children: strategy }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-24 bg-gray-200 rounded-full h-2", children: _jsx("div", { className: "bg-green-600 h-2 rounded-full", style: { width: `${(count / Math.max(1, metrics.totalAttempts)) * 100}%` } }) }), _jsx("span", { className: "text-sm font-medium text-gray-700", children: count })] })] }, strategy))) })] })] }), _jsxs("div", { className: "mt-6 bg-gray-50 p-4 rounded-lg border", children: [_jsxs("h3", { className: "font-semibold text-gray-700 mb-3 flex items-center gap-2", children: [_jsx(TrendingUp, { className: "w-5 h-5" }), "Performance by Domain"] }), _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "min-w-full divide-y divide-gray-200", children: [_jsx("thead", { className: "bg-gray-100", children: _jsxs("tr", { children: [_jsx("th", { className: "px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider", children: "Domain" }), _jsx("th", { className: "px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider", children: "Success Rate" }), _jsx("th", { className: "px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider", children: "Avg Time (ms)" }), _jsx("th", { className: "px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider", children: "Count" })] }) }), _jsx("tbody", { className: "divide-y divide-gray-200", children: Object.entries(metrics.performanceByDomain).map(([domain, perf]) => (_jsxs("tr", { children: [_jsx("td", { className: "px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900 capitalize", children: domain }), _jsxs("td", { className: "px-4 py-2 whitespace-nowrap text-sm text-gray-500", children: [Math.round(perf.successRate * 100), "%"] }), _jsxs("td", { className: "px-4 py-2 whitespace-nowrap text-sm text-gray-500", children: [Math.round(perf.avgTime), "ms"] }), _jsx("td", { className: "px-4 py-2 whitespace-nowrap text-sm text-gray-500", children: perf.count })] }, domain))) })] }) })] })] }));
}
export function KnowledgeGraphIntegration({ className = '' }) {
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [error, setError] = useState(null);
    const handleSearch = async () => {
        if (!searchQuery.trim())
            return;
        setIsSearching(true);
        setError(null);
        try {
            const results = await searchKnowledge({
                query: searchQuery,
                topK: 5
            });
            if (results.success) {
                setSearchResults(results.results);
            }
            else {
                setError(results.error || 'Search failed');
            }
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
        }
        finally {
            setIsSearching(false);
        }
    };
    return (_jsxs("div", { className: `bg-white rounded-xl shadow-md p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 mb-4", children: [_jsx(Database, { className: "w-6 h-6 text-indigo-600" }), _jsx("h2", { className: "text-xl font-bold text-gray-800", children: "Knowledge Graph Integration" })] }), _jsx("div", { className: "mb-4", children: _jsxs("div", { className: "flex gap-2", children: [_jsx("input", { type: "text", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value), placeholder: "Search mathematical concepts...", className: "flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500", onKeyDown: (e) => e.key === 'Enter' && handleSearch() }), _jsx("button", { onClick: handleSearch, disabled: isSearching, className: "px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2", children: isSearching ? (_jsxs(_Fragment, { children: [_jsx(Zap, { className: "w-4 h-4 animate-pulse" }), "Searching..."] })) : (_jsxs(_Fragment, { children: [_jsx(Zap, { className: "w-4 h-4" }), "Search"] })) })] }) }), error && (_jsxs("div", { className: "mb-4 p-3 bg-red-100 text-red-700 rounded-lg flex items-center gap-2", children: [_jsx(AlertTriangle, { className: "w-5 h-5" }), error] })), searchResults.length > 0 && (_jsxs("div", { className: "space-y-3", children: [_jsx("h3", { className: "font-medium text-gray-700", children: "Search Results:" }), searchResults.map((result, index) => (_jsxs("div", { className: "p-3 bg-gray-50 rounded-lg border-l-4 border-indigo-500", children: [_jsx("p", { className: "text-sm text-gray-800", children: result.content }), result.metadata && Object.keys(result.metadata).length > 0 && (_jsx("div", { className: "mt-2 text-xs text-gray-500", children: Object.entries(result.metadata).map(([key, value]) => (_jsxs("span", { className: "mr-2", children: [_jsxs("span", { className: "font-medium", children: [key, ":"] }), " ", String(value)] }, key))) })), result.score !== undefined && (_jsxs("div", { className: "mt-1 text-xs text-gray-500", children: ["Relevance: ", (result.score * 100).toFixed(1), "%"] }))] }, index)))] }))] }));
}
export function LeanAideBubbleLabIntegration({ className = '' }) {
    const [activeTab, setActiveTab] = useState('dashboard');
    return (_jsx("div", { className: `bg-gray-50 min-h-screen p-6 ${className}`, children: _jsxs("div", { className: "max-w-7xl mx-auto", children: [_jsxs("div", { className: "mb-8", children: [_jsxs("h1", { className: "text-3xl font-bold text-gray-900 flex items-center gap-3", children: [_jsx(Brain, { className: "w-8 h-8 text-blue-600" }), "LeanAide BubbleLab Integration"] }), _jsx("p", { className: "text-gray-600 mt-2", children: "Advanced mathematical formalization with real-time analytics and knowledge integration" })] }), _jsxs("div", { className: "bg-white rounded-xl shadow-lg overflow-hidden", children: [_jsx("div", { className: "border-b border-gray-200", children: _jsx("nav", { className: "flex space-x-8 px-6", children: [
                                    { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart2 },
                                    { id: 'verification', label: 'Autoformalization', icon: Shield },
                                    { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                                    { id: 'settings', label: 'Settings', icon: Settings },
                                ].map((tab) => (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${activeTab === tab.id
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`, children: [_jsx(tab.icon, { className: "w-4 h-4" }), tab.label] }, tab.id))) }) }), _jsxs("div", { className: "p-6", children: [activeTab === 'dashboard' && _jsx(AnalyticsDashboard, {}), activeTab === 'verification' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Shield, { className: "w-6 h-6" }), "Autoformalization Verification"] }), _jsx(EnhancedLeanAideVerification, { problemStatement: "Prove that for all natural numbers n, n + 0 = n", mode: "theorem", enableAnalytics: true, strategy: "auto", domain: "arithmetic" })] })), activeTab === 'knowledge' && _jsx(KnowledgeGraphIntegration, {}), activeTab === 'settings' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Settings, { className: "w-6 h-6" }), "Integration Settings"] }), _jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsx("h3", { className: "font-medium text-gray-700 mb-3", children: "Analytics Configuration" }), _jsx("p", { className: "text-gray-600", children: "Configure real-time tracking, performance metrics, and error monitoring for your autoformalization pipeline." }), _jsxs("div", { className: "mt-4 grid grid-cols-1 md:grid-cols-2 gap-4", children: [_jsxs("div", { className: "flex items-center justify-between p-3 bg-white rounded border", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Real-time Tracking" }), _jsx("p", { className: "text-sm text-gray-500", children: "Track all conversion events" })] }), _jsx("div", { className: "w-12 h-6 bg-green-500 rounded-full relative", children: _jsx("div", { className: "w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5" }) })] }), _jsxs("div", { className: "flex items-center justify-between p-3 bg-white rounded border", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Performance Metrics" }), _jsx("p", { className: "text-sm text-gray-500", children: "Monitor processing times" })] }), _jsx("div", { className: "w-12 h-6 bg-green-500 rounded-full relative", children: _jsx("div", { className: "w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5" }) })] }), _jsxs("div", { className: "flex items-center justify-between p-3 bg-white rounded border", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Error Tracking" }), _jsx("p", { className: "text-sm text-gray-500", children: "Log all conversion errors" })] }), _jsx("div", { className: "w-12 h-6 bg-green-500 rounded-full relative", children: _jsx("div", { className: "w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5" }) })] }), _jsxs("div", { className: "flex items-center justify-between p-3 bg-white rounded border", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Domain Analysis" }), _jsx("p", { className: "text-sm text-gray-500", children: "Analyze by mathematical domain" })] }), _jsx("div", { className: "w-12 h-6 bg-green-500 rounded-full relative", children: _jsx("div", { className: "w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5" }) })] })] })] })] }))] })] })] }) }));
}
// Initialize the analytics service when the module loads
analyticsService.initialize();
// Export the main integration component
export default LeanAideBubbleLabIntegration;
//# sourceMappingURL=autoformalizationAnalytics.js.map