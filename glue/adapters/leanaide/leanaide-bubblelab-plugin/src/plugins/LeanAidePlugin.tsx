/**
 * LeanAide Autoformalization Plugin for BubbleLab UI
 * 
 * This plugin integrates the complete LeanAide autoformalization system with predictive analytics
 * into the BubbleLab UI as a comprehensive plugin.
 */

import React, { useState, useEffect, useRef } from 'react';
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
  Info,
  Plus,
  Search,
  Filter,
  Download,
  Upload,
  RefreshCw,
  Play,
  Pause,
  Square
} from 'lucide-react';
import { toast } from 'react-toastify';
import { 
  LeanAideBubbleLabIntegration,
  EnhancedLeanAideVerification,
  AnalyticsDashboard,
  KnowledgeGraphIntegration,
  useAutoformalizationAnalytics
} from '../integration/autoformalizationAnalytics';

// Plugin interface definition
export interface LeanAidePluginInterface {
  id: string;
  name: string;
  description: string;
  version: string;
  category: string;
  component: React.ComponentType<any>;
  icon: React.ReactNode;
  settingsSchema?: any;
  permissions?: string[];
}

// Plugin configuration
export interface LeanAidePluginConfig {
  enableAnalytics: boolean;
  enablePredictiveFlagging: boolean;
  enableKnowledgeGraph: boolean;
  analyticsRefreshInterval: number;
  maxConcurrentRequests: number;
  cacheEnabled: boolean;
  cacheTTL: number;
  serverUrl: string;
  apiKey?: string;
}

// Default configuration
export const DEFAULT_LEANAIDE_PLUGIN_CONFIG: LeanAidePluginConfig = {
  enableAnalytics: true,
  enablePredictiveFlagging: true,
  enableKnowledgeGraph: true,
  analyticsRefreshInterval: 5000,
  maxConcurrentRequests: 5,
  cacheEnabled: true,
  cacheTTL: 3600,
  serverUrl: 'http://localhost:3000/leanaide',
  apiKey: undefined
};

// Main plugin component
export interface LeanAidePluginProps {
  config?: Partial<LeanAidePluginConfig>;
  onConfigChange?: (config: LeanAidePluginConfig) => void;
  className?: string;
}

export const LeanAidePlugin: React.FC<LeanAidePluginProps> = ({
  config: userConfig,
  onConfigChange,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'verification' | 'knowledge' | 'settings'>('dashboard');
  const [pluginConfig, setPluginConfig] = useState<LeanAidePluginConfig>({
    ...DEFAULT_LEANAIDE_PLUGIN_CONFIG,
    ...userConfig
  });
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const analyticsRef = useRef<HTMLDivElement>(null);
  
  // Initialize the plugin
  useEffect(() => {
    const initializePlugin = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Initialize LeanAide client if not already done
        if (typeof window !== 'undefined') {
          // Wait for DOM to be ready
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        setIsInitialized(true);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to initialize LeanAide plugin';
        setError(errorMessage);
        toast.error(`LeanAide plugin initialization failed: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };

    initializePlugin();
  }, []);

  // Handle config changes
  const handleConfigChange = (newConfig: LeanAidePluginConfig) => {
    setPluginConfig(newConfig);
    if (onConfigChange) {
      onConfigChange(newConfig);
    }
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="text-gray-600">Initializing LeanAide Plugin...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 text-red-800">
          <AlertTriangle className="w-5 h-5" />
          <h3 className="font-medium">Plugin Initialization Error</h3>
        </div>
        <p className="text-red-600 mt-2">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
        >
          <RefreshCw className="w-4 h-4 inline mr-2" />
          Reload Plugin
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden ${className}`}>
      {/* Plugin Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8" />
            <div>
              <h1 className="text-xl font-bold">LeanAide Autoformalization</h1>
              <p className="text-blue-100 text-sm">Natural Language to Lean 4 Formalization</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-blue-500 text-xs px-2 py-1 rounded-full">
              v1.0.0
            </span>
            {isInitialized && (
              <span className="bg-green-500 text-xs px-2 py-1 rounded-full flex items-center gap-1">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                Connected
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6">
          {[
            { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart3 },
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
              enableAnalytics={pluginConfig.enableAnalytics}
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

        {activeTab === 'settings' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Settings className="w-6 h-6" />
              Plugin Settings
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-50 p-4 rounded-lg border">
                <h3 className="font-medium text-gray-700 mb-3">Analytics Configuration</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-800">Enable Analytics</p>
                      <p className="text-sm text-gray-500">Track performance metrics</p>
                    </div>
                    <div 
                      className={`w-12 h-6 rounded-full relative cursor-pointer ${
                        pluginConfig.enableAnalytics ? 'bg-blue-500' : 'bg-gray-300'
                      }`}
                      onClick={() => handleConfigChange({
                        ...pluginConfig,
                        enableAnalytics: !pluginConfig.enableAnalytics
                      })}
                    >
                      <div 
                        className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                          pluginConfig.enableAnalytics ? 'left-6' : 'left-0.5'
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
                        pluginConfig.enablePredictiveFlagging ? 'bg-blue-500' : 'bg-gray-300'
                      }`}
                      onClick={() => handleConfigChange({
                        ...pluginConfig,
                        enablePredictiveFlagging: !pluginConfig.enablePredictiveFlagging
                      })}
                    >
                      <div 
                        className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                          pluginConfig.enablePredictiveFlagging ? 'left-6' : 'left-0.5'
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
                        pluginConfig.enableKnowledgeGraph ? 'bg-blue-500' : 'bg-gray-300'
                      }`}
                      onClick={() => handleConfigChange({
                        ...pluginConfig,
                        enableKnowledgeGraph: !pluginConfig.enableKnowledgeGraph
                      })}
                    >
                      <div 
                        className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                          pluginConfig.enableKnowledgeGraph ? 'left-6' : 'left-0.5'
                        }`}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Refresh Interval (ms)
                    </label>
                    <input
                      type="number"
                      value={pluginConfig.analyticsRefreshInterval}
                      onChange={(e) => handleConfigChange({
                        ...pluginConfig,
                        analyticsRefreshInterval: parseInt(e.target.value) || 5000
                      })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border">
                <h3 className="font-medium text-gray-700 mb-3">Connection Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Server URL
                    </label>
                    <input
                      type="text"
                      value={pluginConfig.serverUrl}
                      onChange={(e) => handleConfigChange({
                        ...pluginConfig,
                        serverUrl: e.target.value
                      })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      API Key
                    </label>
                    <input
                      type="password"
                      value={pluginConfig.apiKey || ''}
                      onChange={(e) => handleConfigChange({
                        ...pluginConfig,
                        apiKey: e.target.value || undefined
                      })}
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
                      value={pluginConfig.maxConcurrentRequests}
                      onChange={(e) => handleConfigChange({
                        ...pluginConfig,
                        maxConcurrentRequests: parseInt(e.target.value) || 5
                      })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-800">Enable Caching</p>
                      <p className="text-sm text-gray-500">Cache results for performance</p>
                    </div>
                    <div 
                      className={`w-12 h-6 rounded-full relative cursor-pointer ${
                        pluginConfig.cacheEnabled ? 'bg-blue-500' : 'bg-gray-300'
                      }`}
                      onClick={() => handleConfigChange({
                        ...pluginConfig,
                        cacheEnabled: !pluginConfig.cacheEnabled
                      })}
                    >
                      <div 
                        className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                          pluginConfig.cacheEnabled ? 'left-6' : 'left-0.5'
                        }`}
                      ></div>
                    </div>
                  </div>
                  
                  {pluginConfig.cacheEnabled && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Cache TTL (seconds)
                      </label>
                      <input
                        type="number"
                        value={pluginConfig.cacheTTL}
                        onChange={(e) => handleConfigChange({
                          ...pluginConfig,
                          cacheTTL: parseInt(e.target.value) || 3600
                        })}
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
                  // Reset to defaults
                  setPluginConfig(DEFAULT_LEANAIDE_PLUGIN_CONFIG);
                  if (onConfigChange) {
                    onConfigChange(DEFAULT_LEANAIDE_PLUGIN_CONFIG);
                  }
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
  );
};

// Plugin registration function
export function registerLeanAidePlugin(): LeanAidePluginInterface {
  return {
    id: 'leanaide-autoformalization',
    name: 'LeanAide Autoformalization',
    description: 'Convert natural language mathematical statements to formal Lean 4 code with predictive analytics',
    version: '1.0.0',
    category: 'formalization',
    component: LeanAidePlugin,
    icon: <Brain className="w-5 h-5" />,
    settingsSchema: {
      type: 'object',
      properties: {
        enableAnalytics: { type: 'boolean', default: true },
        enablePredictiveFlagging: { type: 'boolean', default: true },
        enableKnowledgeGraph: { type: 'boolean', default: true },
        analyticsRefreshInterval: { type: 'number', default: 5000 },
        maxConcurrentRequests: { type: 'number', default: 5 },
        cacheEnabled: { type: 'boolean', default: true },
        cacheTTL: { type: 'number', default: 3600 },
        serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' }
      }
    },
    permissions: ['network', 'storage']
  };
}

// Export plugin interface
export type { LeanAidePluginInterface, LeanAidePluginConfig };
export default LeanAidePlugin;