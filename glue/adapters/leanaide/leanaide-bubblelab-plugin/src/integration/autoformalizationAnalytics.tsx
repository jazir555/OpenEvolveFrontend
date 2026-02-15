/**
 * LeanAide Autoformalization System with BubbleLab Analytics Integration
 * 
 * This module provides comprehensive integration between the LeanAide autoformalization system
 * and BubbleLab analytics platform, enabling advanced visualization and monitoring of
 * mathematical formalization processes.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  BarChart3, 
  Activity, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  Database, 
  Zap,
  Brain,
  Target,
  Award,
  Flame,
  Shield,
  Eye,
  BarChart2,
  PieChart,
  LineChart,
  Users,
  MessageSquare,
  Settings,
  Info
} from 'lucide-react';
import { toast } from 'react-toastify';
import { 
  LeanAideVerification, 
  LeanAideVerificationProps,
  RagbitsKnowledgeSearch,
  LeanAidePanel
} from './components';
import {
  initializeLeanAideClient,
  initializeRagbitsClient,
  translateTheorem,
  translateDefinition,
  verifySolution,
  elaborateCode,
  mathQuery,
  searchKnowledge,
  ingestArtifact,
  isLeanAideAvailable,
  isRagbitsAvailable,
  LeanAidePlugin,
  RagbitsPlugin
} from './services';
import { 
  LeanAideTaskResponse, 
  RagbitsSearchResponse,
  RagbitsSearchResult
} from './lib';

// Define types for analytics data
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

// Default configuration
const DEFAULT_ANALYTICS_CONFIG: BubbleLabAnalyticsConfig = {
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
  private config: BubbleLabAnalyticsConfig;
  private events: AutoformalizationEvent[] = [];
  private metrics: AutoformalizationMetrics;
  private eventQueue: AutoformalizationEvent[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private isInitialized: boolean = false;

  constructor(config?: Partial<BubbleLabAnalyticsConfig>) {
    this.config = { ...DEFAULT_ANALYTICS_CONFIG, ...config };
    this.metrics = this.initializeMetrics();
  }

  private initializeMetrics(): AutoformalizationMetrics {
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
    if (this.isInitialized) return;
    
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

  trackEvent(event: AutoformalizationEvent) {
    if (!this.config.enableRealTimeTracking) return;

    this.eventQueue.push(event);
    
    // Process immediately if batch is full
    if (this.eventQueue.length >= this.config.batchSize) {
      this.flushEvents();
    }

    // Update metrics
    this.updateMetrics(event);
  }

  private updateMetrics(event: AutoformalizationEvent) {
    this.metrics.totalAttempts++;
    
    // Update success/failure counts
    if (event.eventType.includes('success')) {
      this.metrics.successfulConversions++;
    } else if (event.eventType.includes('failure')) {
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
      } else {
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

  private flushEvents() {
    if (this.eventQueue.length === 0) return;

    // In a real implementation, this would send events to a backend service
    // For now, we'll just log them
    console.log('Flushing analytics events:', this.eventQueue);
    
    // Clear the queue
    this.eventQueue = [];
  }

  getMetrics(): AutoformalizationMetrics {
    return { ...this.metrics };
  }

  getEvents(): AutoformalizationEvent[] {
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
  const [metrics, setMetrics] = useState<AutoformalizationMetrics>(analyticsService.getMetrics());
  const [events, setEvents] = useState<AutoformalizationEvent[]>([]);
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

  const trackEvent = useCallback((event: AutoformalizationEvent) => {
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

// Enhanced LeanAide Verification with Analytics
export interface EnhancedLeanAideVerificationProps extends LeanAideVerificationProps {
  enableAnalytics?: boolean;
  onAnalyticsEvent?: (event: AutoformalizationEvent) => void;
  strategy?: string;
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
  const [processingTime, setProcessingTime] = useState<number | null>(null);
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
      let result: LeanAideTaskResponse;
      
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
    } catch (error) {
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

  return (
    <div className={`relative ${className}`}>
      <LeanAideVerification
        problemStatement={problemStatement}
        solutionCode={solutionCode}
        onVerificationResult={onVerificationResult}
        mode={mode}
        className={className}
      />
      
      {processingTime !== null && (
        <div className="absolute top-2 right-2 bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {processingTime}ms
        </div>
      )}
      
      {enableAnalytics && (
        <div className="absolute top-2 left-2 bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full flex items-center gap-1">
          <BarChart3 className="w-3 h-3" />
          Analytics
        </div>
      )}
    </div>
  );
}

// Analytics Dashboard Component
export interface AnalyticsDashboardProps {
  className?: string;
}

export function AnalyticsDashboard({ className = '' }: AnalyticsDashboardProps) {
  const { metrics } = useAutoformalizationAnalytics();
  
  // Calculate derived metrics
  const successRatePercentage = Math.round(metrics.successRate * 100);
  const avgConfidence = metrics.confidenceScores.length > 0 
    ? metrics.confidenceScores.reduce((a, b) => a + b, 0) / metrics.confidenceScores.length 
    : 0;
  
  return (
    <div className={`bg-white rounded-xl shadow-md p-6 ${className}`}>
      <div className="flex items-center gap-2 mb-6">
        <BarChart2 className="w-6 h-6 text-blue-600" />
        <h2 className="text-xl font-bold text-gray-800">Autoformalization Analytics</h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total Attempts */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600">Total Attempts</p>
              <p className="text-2xl font-bold text-blue-800">{metrics.totalAttempts}</p>
            </div>
            <Activity className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        
        {/* Success Rate */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600">Success Rate</p>
              <p className="text-2xl font-bold text-green-800">{successRatePercentage}%</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        
        {/* Avg Processing Time */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600">Avg Time</p>
              <p className="text-2xl font-bold text-purple-800">{Math.round(metrics.averageProcessingTime)}ms</p>
            </div>
            <Clock className="w-8 h-8 text-purple-500" />
          </div>
        </div>
        
        {/* Avg Confidence */}
        <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg border border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-yellow-600">Avg Confidence</p>
              <p className="text-2xl font-bold text-yellow-800">{avgConfidence.toFixed(2)}</p>
            </div>
            <Target className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Domain Distribution */}
        <div className="bg-gray-50 p-4 rounded-lg border">
          <h3 className="font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <PieChart className="w-5 h-5" />
            Domain Distribution
          </h3>
          <div className="space-y-2">
            {Object.entries(metrics.domainDistribution).map(([domain, count]) => (
              <div key={domain} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 capitalize">{domain}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${(count / Math.max(1, metrics.totalAttempts)) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-700">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Strategy Usage */}
        <div className="bg-gray-50 p-4 rounded-lg border">
          <h3 className="font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <LineChart className="w-5 h-5" />
            Strategy Usage
          </h3>
          <div className="space-y-2">
            {Object.entries(metrics.strategyUsage).map(([strategy, count]) => (
              <div key={strategy} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 capitalize">{strategy}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${(count / Math.max(1, metrics.totalAttempts)) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-700">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Performance by Domain */}
      <div className="mt-6 bg-gray-50 p-4 rounded-lg border">
        <h3 className="font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Performance by Domain
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Domain</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Success Rate</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Time (ms)</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {Object.entries(metrics.performanceByDomain).map(([domain, perf]) => (
                <tr key={domain}>
                  <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900 capitalize">{domain}</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">{Math.round(perf.successRate * 100)}%</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">{Math.round(perf.avgTime)}ms</td>
                  <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">{perf.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// Knowledge Graph Integration Component
export interface KnowledgeGraphIntegrationProps {
  className?: string;
}

export function KnowledgeGraphIntegration({ className = '' }: KnowledgeGraphIntegrationProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<RagbitsSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setError(null);

    try {
      const results = await searchKnowledge({
        query: searchQuery,
        topK: 5
      });

      if (results.success) {
        setSearchResults(results.results);
      } else {
        setError(results.error || 'Search failed');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className={`bg-white rounded-xl shadow-md p-6 ${className}`}>
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-6 h-6 text-indigo-600" />
        <h2 className="text-xl font-bold text-gray-800">Knowledge Graph Integration</h2>
      </div>
      
      <div className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search mathematical concepts..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2"
          >
            {isSearching ? (
              <>
                <Zap className="w-4 h-4 animate-pulse" />
                Searching...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Search
              </>
            )}
          </button>
        </div>
      </div>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          {error}
        </div>
      )}
      
      {searchResults.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-medium text-gray-700">Search Results:</h3>
          {searchResults.map((result, index) => (
            <div key={index} className="p-3 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
              <p className="text-sm text-gray-800">{result.content}</p>
              {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="mt-2 text-xs text-gray-500">
                  {Object.entries(result.metadata).map(([key, value]) => (
                    <span key={key} className="mr-2">
                      <span className="font-medium">{key}:</span> {String(value)}
                    </span>
                  ))}
                </div>
              )}
              {result.score !== undefined && (
                <div className="mt-1 text-xs text-gray-500">
                  Relevance: {(result.score * 100).toFixed(1)}%
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Complete Integration Dashboard
export interface LeanAideBubbleLabIntegrationProps {
  className?: string;
}

export function LeanAideBubbleLabIntegration({ className = '' }: LeanAideBubbleLabIntegrationProps) {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'verification' | 'knowledge' | 'settings'>('dashboard');

  return (
    <div className={`bg-gray-50 min-h-screen p-6 ${className}`}>
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-600" />
            LeanAide BubbleLab Integration
          </h1>
          <p className="text-gray-600 mt-2">
            Advanced mathematical formalization with real-time analytics and knowledge integration
          </p>
        </div>
        
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          {/* Navigation Tabs */}
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart2 },
                { id: 'verification', label: 'Autoformalization', icon: Shield },
                { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                { id: 'settings', label: 'Settings', icon: Settings },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
          
          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'dashboard' && <AnalyticsDashboard />}
            
            {activeTab === 'verification' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Shield className="w-6 h-6" />
                  Autoformalization Verification
                </h2>
                <EnhancedLeanAideVerification
                  problemStatement="Prove that for all natural numbers n, n + 0 = n"
                  mode="theorem"
                  enableAnalytics={true}
                  strategy="auto"
                  domain="arithmetic"
                />
              </div>
            )}
            
            {activeTab === 'knowledge' && <KnowledgeGraphIntegration />}
            
            {activeTab === 'settings' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Settings className="w-6 h-6" />
                  Integration Settings
                </h2>
                <div className="bg-gray-50 p-4 rounded-lg border">
                  <h3 className="font-medium text-gray-700 mb-3">Analytics Configuration</h3>
                  <p className="text-gray-600">
                    Configure real-time tracking, performance metrics, and error monitoring for your autoformalization pipeline.
                  </p>
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Real-time Tracking</p>
                        <p className="text-sm text-gray-500">Track all conversion events</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Performance Metrics</p>
                        <p className="text-sm text-gray-500">Monitor processing times</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Error Tracking</p>
                        <p className="text-sm text-gray-500">Log all conversion errors</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Domain Analysis</p>
                        <p className="text-sm text-gray-500">Analyze by mathematical domain</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Initialize the analytics service when the module loads
analyticsService.initialize();

// Export the main integration component
export default LeanAideBubbleLabIntegration;