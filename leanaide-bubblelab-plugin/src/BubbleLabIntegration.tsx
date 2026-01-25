/**
 * BubbleLab UI Integration for LeanAide Autoformalization System
 * 
 * This module provides the complete integration of the LeanAide autoformalization system
 * with predictive analytics into the BubbleLab UI as a comprehensive plugin system.
 */

import React, { useState, useEffect, useRef, Suspense } from 'react';
import { 
  Brain, 
  BarChart3, 
  Shield, 
  Database, 
  Settings, 
  Activity, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Zap,
  Target,
  Award,
  Flame,
  Eye,
  BarChart2,
  PieChart,
  LineChart,
  Users,
  MessageSquare,
  Puzzle,
  RefreshCw,
  Play,
  Pause,
  Square,
  Download,
  Upload,
  Filter,
  Search
} from 'lucide-react';
import { toast } from 'react-toastify';
import { 
  LeanAideBubbleLabIntegration,
  EnhancedLeanAideVerification,
  AnalyticsDashboard,
  KnowledgeGraphIntegration,
  useAutoformalizationAnalytics
} from './integration/autoformalizationAnalytics';
import { 
  LeanAidePlugin, 
  pluginRegistry, 
  PluginManager 
} from './PluginInterface';

// BubbleLab UI integration components
export interface BubbleLabIntegrationProps {
  serverUrl?: string;
  apiKey?: string;
  enableAnalytics?: boolean;
  enablePredictiveFlagging?: boolean;
  enableKnowledgeGraph?: boolean;
  className?: string;
}

export const BubbleLabLeanAideIntegration: React.FC<BubbleLabIntegrationProps> = ({
  serverUrl = 'http://localhost:3000/leanaide',
  apiKey,
  enableAnalytics = true,
  enablePredictiveFlagging = true,
  enableKnowledgeGraph = true,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'verification' | 'knowledge' | 'plugins' | 'settings'>('dashboard');
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState({
    serverUrl,
    apiKey,
    enableAnalytics,
    enablePredictiveFlagging,
    enableKnowledgeGraph,
    analyticsRefreshInterval: 5000,
    maxConcurrentRequests: 5,
    cacheEnabled: true,
    cacheTTL: 3600
  });

  // Initialize the integration
  useEffect(() => {
    const initializeIntegration = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Initialize LeanAide client if needed
        // In a real implementation, this would connect to the server
        console.log('Initializing LeanAide integration with server:', serverUrl);

        // Initialize plugins
        await initializePlugins();

        setIsInitialized(true);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to initialize LeanAide integration';
        setError(errorMessage);
        toast.error(`LeanAide integration initialization failed: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };

    initializeIntegration();
  }, [serverUrl, apiKey]);

  const initializePlugins = async () => {
    // Initialize any required plugins
    // In a real implementation, this would initialize the plugin system
    console.log('Initializing plugins...');
  };

  const handleConfigChange = (newConfig: any) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
          <h3 className="text-xl font-medium text-gray-800">Initializing LeanAide Integration</h3>
          <p className="text-gray-600">Connecting to autoformalization services...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 text-red-800 mb-4">
          <AlertTriangle className="w-5 h-5" />
          <h3 className="font-medium">Integration Error</h3>
        </div>
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reload Integration
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-gray-50 min-h-screen ${className}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">LeanAide Autoformalization</h1>
              <p className="text-gray-600">Natural Language to Lean 4 Formalization with Analytics</p>
            </div>
          </div>
          
          <div className="flex items-center gap-6 mt-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Connected</span>
            </div>
            <div>Server: {config.serverUrl}</div>
            <div>Analytics: {config.enableAnalytics ? 'Enabled' : 'Disabled'}</div>
            <div>Predictive: {config.enablePredictiveFlagging ? 'Enabled' : 'Disabled'}</div>
          </div>
        </div>

        {/* Navigation */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart3 },
                { id: 'verification', label: 'Autoformalization', icon: Shield },
                { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                { id: 'plugins', label: 'Plugin Manager', icon: Puzzle },
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
            {activeTab === 'dashboard' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                    <BarChart3 className="w-6 h-6" />
                    Analytics Dashboard
                  </h2>
                  <div className="flex items-center gap-2">
                    <button className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors">
                      <Download className="w-4 h-4" />
                      Export
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors">
                      <RefreshCw className="w-4 h-4" />
                      Refresh
                    </button>
                  </div>
                </div>
                
                <AnalyticsDashboard />
              </div>
            )}

            {activeTab === 'verification' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                    <Shield className="w-6 h-6" />
                    Autoformalization Verification
                  </h2>
                  <div className="flex items-center gap-2">
                    <button className="flex items-center gap-2 px-3 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors">
                      <Play className="w-4 h-4" />
                      Run
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 transition-colors">
                      <Plus className="w-4 h-4" />
                      New
                    </button>
                  </div>
                </div>
                
                <EnhancedLeanAideVerification
                  problemStatement=""
                  mode="theorem"
                  enableAnalytics={config.enableAnalytics}
                  strategy="auto"
                  domain="general"
                />
              </div>
            )}

            {activeTab === 'knowledge' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                    <Database className="w-6 h-6" />
                    Knowledge Graph Integration
                  </h2>
                  <div className="flex items-center gap-2">
                    <button className="flex items-center gap-2 px-3 py-2 bg-purple-100 text-purple-700 rounded-md hover:bg-purple-200 transition-colors">
                      <Search className="w-4 h-4" />
                      Search
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200 transition-colors">
                      <Upload className="w-4 h-4" />
                      Ingest
                    </button>
                  </div>
                </div>
                
                <KnowledgeGraphIntegration />
              </div>
            )}

            {activeTab === 'plugins' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Puzzle className="w-6 h-6" />
                  Plugin Manager
                </h2>
                
                <PluginManager />
              </div>
            )}

            {activeTab === 'settings' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Settings className="w-6 h-6" />
                  Integration Settings
                </h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gray-50 p-4 rounded-lg border">
                    <h3 className="font-medium text-gray-700 mb-3">Service Configuration</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Server URL
                        </label>
                        <input
                          type="text"
                          value={config.serverUrl}
                          onChange={(e) => handleConfigChange({ serverUrl: e.target.value })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          API Key
                        </label>
                        <input
                          type="password"
                          value={config.apiKey || ''}
                          onChange={(e) => handleConfigChange({ apiKey: e.target.value || undefined })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                          placeholder="Enter API key (optional)"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Max Concurrent Requests
                        </label>
                        <input
                          type="number"
                          value={config.maxConcurrentRequests}
                          onChange={(e) => handleConfigChange({ maxConcurrentRequests: parseInt(e.target.value) || 5 })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-4 rounded-lg border">
                    <h3 className="font-medium text-gray-700 mb-3">Feature Configuration</h3>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-gray-800">Analytics</p>
                          <p className="text-sm text-gray-500">Enable real-time metrics</p>
                        </div>
                        <div 
                          className={`w-12 h-6 rounded-full relative cursor-pointer ${
                            config.enableAnalytics ? 'bg-blue-500' : 'bg-gray-300'
                          }`}
                          onClick={() => handleConfigChange({ enableAnalytics: !config.enableAnalytics })}
                        >
                          <div 
                            className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                              config.enableAnalytics ? 'left-6' : 'left-0.5'
                            }`}
                          ></div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-gray-800">Predictive Flagging</p>
                          <p className="text-sm text-gray-500">Enable predictive quality control</p>
                        </div>
                        <div 
                          className={`w-12 h-6 rounded-full relative cursor-pointer ${
                            config.enablePredictiveFlagging ? 'bg-blue-500' : 'bg-gray-300'
                          }`}
                          onClick={() => handleConfigChange({ enablePredictiveFlagging: !config.enablePredictiveFlagging })}
                        >
                          <div 
                            className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                              config.enablePredictiveFlagging ? 'left-6' : 'left-0.5'
                            }`}
                          ></div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-gray-800">Knowledge Graph</p>
                          <p className="text-sm text-gray-500">Enable knowledge integration</p>
                        </div>
                        <div 
                          className={`w-12 h-6 rounded-full relative cursor-pointer ${
                            config.enableKnowledgeGraph ? 'bg-blue-500' : 'bg-gray-300'
                          }`}
                          onClick={() => handleConfigChange({ enableKnowledgeGraph: !config.enableKnowledgeGraph })}
                        >
                          <div 
                            className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                              config.enableKnowledgeGraph ? 'left-6' : 'left-0.5'
                            }`}
                          ></div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-gray-800">Caching</p>
                          <p className="text-sm text-gray-500">Enable result caching</p>
                        </div>
                        <div 
                          className={`w-12 h-6 rounded-full relative cursor-pointer ${
                            config.cacheEnabled ? 'bg-blue-500' : 'bg-gray-300'
                          }`}
                          onClick={() => handleConfigChange({ cacheEnabled: !config.cacheEnabled })}
                        >
                          <div 
                            className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                              config.cacheEnabled ? 'left-6' : 'left-0.5'
                            }`}
                          ></div>
                        </div>
                      </div>
                      
                      {config.cacheEnabled && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Cache TTL (seconds)
                          </label>
                          <input
                            type="number"
                            value={config.cacheTTL}
                            onChange={(e) => handleConfigChange({ cacheTTL: parseInt(e.target.value) || 3600 })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex justify-end gap-3 pt-4">
                  <button
                    onClick={() => {
                      setConfig({
                        serverUrl: 'http://localhost:3000/leanaide',
                        apiKey: undefined,
                        enableAnalytics: true,
                        enablePredictiveFlagging: true,
                        enableKnowledgeGraph: true,
                        analyticsRefreshInterval: 5000,
                        maxConcurrentRequests: 5,
                        cacheEnabled: true,
                        cacheTTL: 3600
                      });
                      toast.success('Settings reset to defaults');
                    }}
                    className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
                  >
                    Reset Defaults
                  </button>
                  <button
                    onClick={() => {
                      toast.success('Settings saved successfully');
                    }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Lazy-loaded component for better performance
const LazyBubbleLabIntegration = React.lazy(() => 
  import('./BubbleLabIntegration').then(module => ({ default: module.BubbleLabLeanAideIntegration }))
);

export const BubbleLabLeanAideIntegrationLazy: React.FC<BubbleLabIntegrationProps> = (props) => (
  <Suspense fallback={
    <div className="flex items-center justify-center h-96">
      <div className="flex flex-col items-center gap-4">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
        <p>Loading LeanAide Integration...</p>
      </div>
    </div>
  }>
    <LazyBubbleLabIntegration {...props} />
  </Suspense>
);

// Plugin registration for BubbleLab
export const registerBubbleLabIntegration = (): LeanAidePluginInterface => {
  return {
    id: 'bubblelab-leanaide-integration',
    name: 'BubbleLab LeanAide Integration',
    description: 'Complete integration of LeanAide autoformalization with analytics into BubbleLab UI',
    version: '1.0.0',
    category: 'integration',
    component: BubbleLabLeanAideIntegration,
    icon: <Brain className="w-5 h-5" />,
    settingsSchema: {
      type: 'object',
      properties: {
        serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' },
        apiKey: { type: 'string', default: '' },
        enableAnalytics: { type: 'boolean', default: true },
        enablePredictiveFlagging: { type: 'boolean', default: true },
        enableKnowledgeGraph: { type: 'boolean', default: true },
        analyticsRefreshInterval: { type: 'number', default: 5000 },
        maxConcurrentRequests: { type: 'number', default: 5 },
        cacheEnabled: { type: 'boolean', default: true },
        cacheTTL: { type: 'number', default: 3600 }
      }
    },
    permissions: ['network', 'storage'],
    dependencies: ['leanaide-core', 'bubblelab-core'],
    author: 'OpenEvolve',
    license: 'MIT',
    homepage: 'https://github.com/openevolve/leanaide',
    repository: 'https://github.com/openevolve/leanaide/leanaide-bubblelab-plugin',
    keywords: ['lean', 'theorem', 'prover', 'formalization', 'autoformalization', 'bubblelab', 'integration', 'analytics'],
    activationEvents: ['onView:leanaide-dashboard', 'onCommand:leanaide.open'],
    contributes: {
      views: [
        {
          id: 'leanaide-dashboard',
          name: 'LeanAide Dashboard',
          when: 'leanaide.enabled'
        },
        {
          id: 'leanaide-verification',
          name: 'Autoformalization',
          when: 'leanaide.enabled'
        },
        {
          id: 'leanaide-knowledge',
          name: 'Knowledge Graph',
          when: 'leanaide.knowledgeGraphEnabled'
        }
      ],
      commands: [
        {
          command: 'leanaide.convert',
          title: 'Convert Natural Language to Lean',
          category: 'LeanAide'
        },
        {
          command: 'leanaide.verify',
          title: 'Verify Lean Code',
          category: 'LeanAide'
        },
        {
          command: 'leanaide.searchKnowledge',
          title: 'Search Mathematical Knowledge',
          category: 'LeanAide'
        }
      ],
      configuration: {
        title: 'LeanAide Configuration',
        properties: {
          'leanaide.serverUrl': {
            type: 'string',
            default: 'http://localhost:3000/leanaide',
            description: 'URL of the LeanAide server'
          },
          'leanaide.apiKey': {
            type: 'string',
            default: '',
            description: 'API key for LeanAide server'
          },
          'leanaide.enableAnalytics': {
            type: 'boolean',
            default: true,
            description: 'Enable real-time analytics'
          },
          'leanaide.enablePredictiveFlagging': {
            type: 'boolean',
            default: true,
            description: 'Enable predictive quality control'
          }
        }
      }
    }
  };
};

// Register the integration plugin
const bubbleLabIntegrationPlugin = registerBubbleLabIntegration();
pluginRegistry.register(bubbleLabIntegrationPlugin);

// Auto-activate the integration plugin
pluginRegistry.activate('bubblelab-leanaide-integration').catch(console.error);

// Export the main integration component
export { BubbleLabLeanAideIntegration, BubbleLabLeanAideIntegrationLazy };
export default BubbleLabLeanAideIntegration;