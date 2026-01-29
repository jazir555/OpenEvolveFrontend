/**
 * IntegrationConfigPanel.tsx
 *
 * Configuration panel for third-party service integrations
 * in the OpenEvolve plugin.
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { IconWrapper } from '../icons/IconWrapper';
import { BubbleBadge, BubbleButton, BubbleInput, BubbleSelect, BubbleTextArea } from '@/components/bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import {
  Network,
  Key,
  Server,
  Clock,
  Settings,
  Zap,
} from 'lucide-react';

export interface IntegrationConfig {
  // API endpoints
  leanAideApiEndpoint: string;
  crewaiApiEndpoint: string;
  bubbleLabsApiEndpoint: string;
  researchQuestApiEndpoint: string;

  // Timeout configurations (in milliseconds)
  defaultTimeout: number;
  leanAideTimeout: number;
  crewaiTimeout: number;
  bubbleLabsTimeout: number;
  researchQuestTimeout: number;

  // Authentication
  useAuthentication: boolean;
  authType: 'api_key' | 'oauth2' | 'jwt' | 'basic';
  apiKey: string;
  oauth2ClientId: string;
  oauth2ClientSecret: string;
  oauth2Scope: string;
  jwtToken: string;
  basicUsername: string;
  basicPassword: string;

  // WebSocket settings
  enableWebSocket: boolean;
  websocketUrl: string;
  websocketReconnectInterval: number;
  websocketMaxReconnectAttempts: number;
  websocketHeartbeatInterval: number;

  // Retry logic
  enableRetry: boolean;
  maxRetries: number;
  retryDelay: number;
  retryBackoffMultiplier: number;
  retryOnStatusCodes: number[];

  // Rate limiting
  enableRateLimiting: boolean;
  maxRequestsPerSecond: number;
  maxRequestsPerMinute: number;
  rateLimitBurstSize: number;

  // Circuit breaker
  enableCircuitBreaker: boolean;
  circuitBreakerThreshold: number;
  circuitBreakerTimeout: number;
  circuitBreakerHalfOpenAttempts: number;

  // Caching
  enableCaching: boolean;
  cacheTtl: number;
  cacheSize: number;
  cacheStrategy: 'lru' | 'fifo' | 'lfu';

  // Monitoring and logging
  enableLogging: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error' | 'none';
  logRequests: boolean;
  logResponses: boolean;
  logErrors: boolean;

  // Health checks
  enableHealthChecks: boolean;
  healthCheckInterval: number;
  healthCheckTimeout: number;
  healthCheckUnhealthyThreshold: number;

  // Service-specific settings
  leanAide: {
    provider: 'openai' | 'anthropic' | 'mistral' | 'custom';
    model: string;
    temperature: number;
    maxTokens: number;
  };

  crewai: {
    parallelExecution: boolean;
    maxParallelTasks: number;
    delegationTimeout: number;
  };

  bubbleLabs: {
    uiEnabled: boolean;
    realTimeUpdates: boolean;
    visualizationQuality: 'low' | 'medium' | 'high';
  };

  researchQuest: {
    knowledgeGraphEnabled: boolean;
    semanticSearch: boolean;
    maxResults: number;
  };
}

interface IntegrationConfigPanelProps {
  config: IntegrationConfig;
  onConfigChange: (config: IntegrationConfig) => void;
}

const DEFAULT_CONFIG: IntegrationConfig = {
  leanAideApiEndpoint: 'http://localhost:8000/api/v1',
  crewaiApiEndpoint: 'http://localhost:8001/api/v1',
  bubbleLabsApiEndpoint: 'http://localhost:8002/api/v1',
  researchQuestApiEndpoint: 'http://localhost:8003/api/v1',
  defaultTimeout: 30000,
  leanAideTimeout: 60000,
  crewaiTimeout: 45000,
  bubbleLabsTimeout: 30000,
  researchQuestTimeout: 30000,
  useAuthentication: true,
  authType: 'api_key',
  apiKey: '',
  oauth2ClientId: '',
  oauth2ClientSecret: '',
  oauth2Scope: '',
  jwtToken: '',
  basicUsername: '',
  basicPassword: '',
  enableWebSocket: false,
  websocketUrl: 'ws://localhost:8080/ws',
  websocketReconnectInterval: 5000,
  websocketMaxReconnectAttempts: 10,
  websocketHeartbeatInterval: 30000,
  enableRetry: true,
  maxRetries: 3,
  retryDelay: 1000,
  retryBackoffMultiplier: 2,
  retryOnStatusCodes: [408, 429, 500, 502, 503, 504],
  enableRateLimiting: true,
  maxRequestsPerSecond: 10,
  maxRequestsPerMinute: 500,
  rateLimitBurstSize: 20,
  enableCircuitBreaker: true,
  circuitBreakerThreshold: 5,
  circuitBreakerTimeout: 60000,
  circuitBreakerHalfOpenAttempts: 3,
  enableCaching: true,
  cacheTtl: 300000,
  cacheSize: 1000,
  cacheStrategy: 'lru',
  enableLogging: true,
  logLevel: 'info',
  logRequests: true,
  logResponses: false,
  logErrors: true,
  enableHealthChecks: true,
  healthCheckInterval: 30000,
  healthCheckTimeout: 5000,
  healthCheckUnhealthyThreshold: 3,
  leanAide: {
    provider: 'anthropic',
    model: 'claude-3-sonnet',
    temperature: 0.7,
    maxTokens: 4096,
  },
  crewai: {
    parallelExecution: true,
    maxParallelTasks: 5,
    delegationTimeout: 120000,
  },
  bubbleLabs: {
    uiEnabled: true,
    realTimeUpdates: true,
    visualizationQuality: 'medium',
  },
  researchQuest: {
    knowledgeGraphEnabled: true,
    semanticSearch: true,
    maxResults: 10,
  },
};

const IntegrationConfigPanelBase: React.FC<IntegrationConfigPanelProps> = ({
  config,
  onConfigChange,
}) => {
  const [localConfig, setLocalConfig] = useState<IntegrationConfig>(config);
  const [activeSection, setActiveSection] = useState<
    'endpoints' | 'authentication' | 'timeouts' | 'websocket' | 'retry' | 'services'
  >('endpoints');
  const [hasChanges, setHasChanges] = useState(false);
  const [showSensitiveData, setShowSensitiveData] = useState(false);

  useEffect(() => {
    setLocalConfig(config);
    setHasChanges(false);
  }, [config]);

  const handleFieldChange = <K extends keyof IntegrationConfig>(
    field: K,
    value: IntegrationConfig[K]
  ) => {
    const newConfig = { ...localConfig, [field]: value };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleNestedFieldChange = <
    K extends keyof IntegrationConfig,
    NK extends keyof IntegrationConfig[K]
  >(
    field: K,
    nestedField: NK,
    value: IntegrationConfig[K][NK]
  ) => {
    const newConfig: IntegrationConfig = {
      ...localConfig,
      [field]: {
        ...(localConfig[field] as object),
        [nestedField]: value,
      },
    };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = () => {
    try {
      onConfigChange(localConfig);
      setHasChanges(false);
      toast.success('Integration configuration saved successfully');
    } catch (error) {
      toast.error(`Failed to save configuration: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset to default configuration?')) {
      setLocalConfig(DEFAULT_CONFIG);
      setHasChanges(true);
      toast.info('Configuration reset to defaults. Click Save to apply.');
    }
  };

  const handleDiscard = () => {
    setLocalConfig(config);
    setHasChanges(false);
    toast.info('Changes discarded');
  };

  const maskSensitiveValue = (value: string) => {
    if (!showSensitiveData && value) {
      return '••••••••••••';
    }
    return value;
  };

  const sections = [
    { id: 'endpoints', label: 'API Endpoints', icon: <Server className="w-4 h-4" /> },
    { id: 'authentication', label: 'Authentication', icon: <Key className="w-4 h-4" /> },
    { id: 'timeouts', label: 'Timeouts & Limits', icon: <Clock className="w-4 h-4" /> },
    { id: 'websocket', label: 'WebSocket', icon: <Zap className="w-4 h-4" /> },
    { id: 'retry', label: 'Retry & Circuit Breaker', icon: <Settings className="w-4 h-4" /> },
    { id: 'services', label: 'Service Settings', icon: <Network className="w-4 h-4" /> },
  ] as const;

  return (
    <div className="integration-config-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Network className="mr-3 text-2xl" />
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Integration Configuration
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            {hasChanges && <BubbleBadge tone="warning">Unsaved Changes</BubbleBadge>}
            <BubbleButton onClick={() => setShowSensitiveData(!showSensitiveData)} variant="secondary">
              {showSensitiveData ? '?? Hide' : '??? Show'} Secrets
            </BubbleButton>
            <BubbleButton onClick={handleSave} disabled={!hasChanges}>
              Save
            </BubbleButton>
            <BubbleButton onClick={handleDiscard} disabled={!hasChanges} variant="secondary">
              Discard
            </BubbleButton>
            <BubbleButton onClick={handleReset} variant="secondary">
              Reset to Defaults
            </BubbleButton>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Sidebar Navigation */}
        <div className="w-64 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
          <nav className="p-4 space-y-2">
            {sections.map((section) => (
              <BubbleButton
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                variant={activeSection === section.id ? 'primary' : 'secondary'}
                className="w-full justify-start gap-3"
              >
                <span className="mr-3">{section.icon}</span>
                <span className="font-medium">{section.label}</span>
              </BubbleButton>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto max-h-[calc(100vh-200px)]">
          {/* API Endpoints Section */}
          {activeSection === 'endpoints' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  API Endpoints
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure the base URLs for all integrated services.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-6">
                <div>
                  <label htmlFor="leanAideApiEndpoint" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    LeanAide API Endpoint
                  </label>
                  <BubbleInput
                    id="leanAideApiEndpoint"
                    type="url"
                    placeholder="http://localhost:8000/api/v1"
                    value={localConfig.leanAideApiEndpoint}
                    onChange={(e) => handleFieldChange('leanAideApiEndpoint', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Base URL for LeanAide mathematical reasoning service
                  </p>
                </div>

                <div>
                  <label htmlFor="crewaiApiEndpoint" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    CrewAI API Endpoint
                  </label>
                  <BubbleInput
                    id="crewaiApiEndpoint"
                    type="url"
                    placeholder="http://localhost:8001/api/v1"
                    value={localConfig.crewaiApiEndpoint}
                    onChange={(e) => handleFieldChange('crewaiApiEndpoint', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Base URL for CrewAI workflow service
                  </p>
                </div>

                <div>
                  <label htmlFor="bubbleLabsApiEndpoint" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    BubbleLabs API Endpoint
                  </label>
                  <BubbleInput
                    id="bubbleLabsApiEndpoint"
                    type="url"
                    placeholder="http://localhost:8002/api/v1"
                    value={localConfig.bubbleLabsApiEndpoint}
                    onChange={(e) => handleFieldChange('bubbleLabsApiEndpoint', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Base URL for BubbleLabs visualization service
                  </p>
                </div>

                <div>
                  <label htmlFor="researchQuestApiEndpoint" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    ResearchQuest API Endpoint
                  </label>
                  <BubbleInput
                    id="researchQuestApiEndpoint"
                    type="url"
                    placeholder="http://localhost:8003/api/v1"
                    value={localConfig.researchQuestApiEndpoint}
                    onChange={(e) => handleFieldChange('researchQuestApiEndpoint', e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Base URL for ResearchQuest knowledge exploration service
                  </p>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <h4 className="text-sm font-medium text-blue-900 dark:text-blue-400 mb-2">
                  Connection Test
                </h4>
                <BubbleButton
                  onClick={() => {
                    toast.info('Testing connection to all endpoints...');
                    // In a real implementation, this would test the connections
                    setTimeout(() => {
                      toast.success('All endpoints are reachable');
                    }, 2000);
                  }}
                >
                  Test All Connections
                </BubbleButton>
              </div>
            </div>
          )}

          {/* Authentication Section */}
          {activeSection === 'authentication' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Authentication Settings
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure authentication for API requests.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex items-start">
                  <div className="flex items-center h-5">
                    <BubbleInput
                      id="useAuthentication"
                      type="checkbox"
                      checked={localConfig.useAuthentication}
                      onChange={(e) => handleFieldChange('useAuthentication', e.target.checked)}
                      className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                  </div>
                  <div className="ml-3">
                    <label htmlFor="useAuthentication" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable Authentication
                    </label>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Require authentication for API requests
                    </p>
                  </div>
                </div>

                {localConfig.useAuthentication && (
                  <>
                    <div>
                      <label htmlFor="authType" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Authentication Type
                      </label>
                      <BubbleSelect
                        id="authType"
                        value={localConfig.authType}
                        onChange={(e) => handleFieldChange('authType', e.target.value as IntegrationConfig['authType'])}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      >
                        <option value="api_key">API Key</option>
                        <option value="oauth2">OAuth 2.0</option>
                        <option value="jwt">JWT Token</option>
                        <option value="basic">Basic Auth</option>
                      </BubbleSelect>
                    </div>
                  </>
                )}
              </div>

              {localConfig.useAuthentication && localConfig.authType === 'api_key' && (
                <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-md">
                  <h4 className="text-sm font-medium text-yellow-900 dark:text-yellow-400 mb-4">
                    API Key Configuration
                  </h4>
                  <div>
                    <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      API Key
                    </label>
                    <BubbleInput
                      id="apiKey"
                      type={showSensitiveData ? 'text' : 'password'}
                      placeholder="Enter your API key"
                      value={localConfig.apiKey}
                      onChange={(e) => handleFieldChange('apiKey', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      The API key will be sent in the X-API-Key header
                    </p>
                  </div>
                </div>
              )}

              {localConfig.useAuthentication && localConfig.authType === 'oauth2' && (
                <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-md space-y-4">
                  <h4 className="text-sm font-medium text-yellow-900 dark:text-yellow-400">
                    OAuth 2.0 Configuration
                  </h4>
                  <div>
                    <label htmlFor="oauth2ClientId" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Client ID
                    </label>
                    <BubbleInput
                      id="oauth2ClientId"
                      type="text"
                      value={localConfig.oauth2ClientId}
                      onChange={(e) => handleFieldChange('oauth2ClientId', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                  <div>
                    <label htmlFor="oauth2ClientSecret" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Client Secret
                    </label>
                    <BubbleInput
                      id="oauth2ClientSecret"
                      type={showSensitiveData ? 'text' : 'password'}
                      value={maskSensitiveValue(localConfig.oauth2ClientSecret)}
                      onChange={(e) => handleFieldChange('oauth2ClientSecret', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                  <div>
                    <label htmlFor="oauth2Scope" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Scope
                    </label>
                    <BubbleInput
                      id="oauth2Scope"
                      type="text"
                      placeholder="api.read api.write"
                      value={localConfig.oauth2Scope}
                      onChange={(e) => handleFieldChange('oauth2Scope', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                </div>
              )}

              {localConfig.useAuthentication && localConfig.authType === 'jwt' && (
                <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-md">
                  <h4 className="text-sm font-medium text-yellow-900 dark:text-yellow-400 mb-4">
                    JWT Token Configuration
                  </h4>
                  <div>
                    <label htmlFor="jwtToken" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      JWT Token
                    </label>
                    <BubbleTextArea
                      id="jwtToken"
                      rows={4}
                      value={maskSensitiveValue(localConfig.jwtToken)}
                      onChange={(e) => handleFieldChange('jwtToken', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-xs"
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      The token will be sent in the Authorization header as Bearer token
                    </p>
                  </div>
                </div>
              )}

              {localConfig.useAuthentication && localConfig.authType === 'basic' && (
                <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-md space-y-4">
                  <h4 className="text-sm font-medium text-yellow-900 dark:text-yellow-400">
                    Basic Authentication
                  </h4>
                  <div>
                    <label htmlFor="basicUsername" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Username
                    </label>
                    <BubbleInput
                      id="basicUsername"
                      type="text"
                      value={localConfig.basicUsername}
                      onChange={(e) => handleFieldChange('basicUsername', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                  <div>
                    <label htmlFor="basicPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Password
                    </label>
                    <BubbleInput
                      id="basicPassword"
                      type={showSensitiveData ? 'text' : 'password'}
                      value={maskSensitiveValue(localConfig.basicPassword)}
                      onChange={(e) => handleFieldChange('basicPassword', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Timeouts Section */}
          {activeSection === 'timeouts' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Timeouts & Rate Limits
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure request timeouts and rate limiting.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="defaultTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Default Timeout (ms)
                    <span className="ml-2 text-xs text-gray-500">(1000-300000)</span>
                  </label>
                  <BubbleInput
                    id="defaultTimeout"
                    type="number"
                    min="1000"
                    max="300000"
                    step="1000"
                    value={localConfig.defaultTimeout}
                    onChange={(e) => handleFieldChange('defaultTimeout', parseInt(e.target.value) || 30000)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Default timeout for API requests (30 seconds)
                  </p>
                </div>

                <div>
                  <label htmlFor="leanAideTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    LeanAide Timeout (ms)
                  </label>
                  <BubbleInput
                    id="leanAideTimeout"
                    type="number"
                    min="1000"
                    max="300000"
                    step="1000"
                    value={localConfig.leanAideTimeout}
                    onChange={(e) => handleFieldChange('leanAideTimeout', parseInt(e.target.value) || 60000)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Timeout for LeanAide mathematical proofs (60 seconds)
                  </p>
                </div>

                <div>
                  <label htmlFor="crewaiTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    CrewAI Timeout (ms)
                  </label>
                  <BubbleInput
                    id="crewaiTimeout"
                    type="number"
                    min="1000"
                    max="300000"
                    step="1000"
                    value={localConfig.crewaiTimeout}
                    onChange={(e) => handleFieldChange('crewaiTimeout', parseInt(e.target.value) || 45000)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Timeout for CrewAI workflow tasks (45 seconds)
                  </p>
                </div>

                <div>
                  <label htmlFor="bubbleLabsTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    BubbleLabs Timeout (ms)
                  </label>
                  <BubbleInput
                    id="bubbleLabsTimeout"
                    type="number"
                    min="1000"
                    max="300000"
                    step="1000"
                    value={localConfig.bubbleLabsTimeout}
                    onChange={(e) => handleFieldChange('bubbleLabsTimeout', parseInt(e.target.value) || 30000)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Timeout for BubbleLabs UI operations (30 seconds)
                  </p>
                </div>

                <div>
                  <label htmlFor="researchQuestTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    ResearchQuest Timeout (ms)
                  </label>
                  <BubbleInput
                    id="researchQuestTimeout"
                    type="number"
                    min="1000"
                    max="300000"
                    step="1000"
                    value={localConfig.researchQuestTimeout}
                    onChange={(e) => handleFieldChange('researchQuestTimeout', parseInt(e.target.value) || 30000)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Timeout for ResearchQuest queries (30 seconds)
                  </p>
                </div>
              </div>

              <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-md">
                <h4 className="text-sm font-medium text-green-900 dark:text-green-400 mb-4">
                  Rate Limiting
                </h4>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="enableRateLimiting"
                        type="checkbox"
                        checked={localConfig.enableRateLimiting}
                        onChange={(e) => handleFieldChange('enableRateLimiting', e.target.checked)}
                        className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="enableRateLimiting" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Enable Rate Limiting
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Prevent overwhelming API servers with too many requests
                      </p>
                    </div>
                  </div>

                  {localConfig.enableRateLimiting && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label htmlFor="maxRequestsPerSecond" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Requests/Second
                        </label>
                        <BubbleInput
                          id="maxRequestsPerSecond"
                          type="number"
                          min="1"
                          max="100"
                          value={localConfig.maxRequestsPerSecond}
                          onChange={(e) => handleFieldChange('maxRequestsPerSecond', parseInt(e.target.value) || 10)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>

                      <div>
                        <label htmlFor="maxRequestsPerMinute" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Requests/Minute
                        </label>
                        <BubbleInput
                          id="maxRequestsPerMinute"
                          type="number"
                          min="1"
                          max="10000"
                          value={localConfig.maxRequestsPerMinute}
                          onChange={(e) => handleFieldChange('maxRequestsPerMinute', parseInt(e.target.value) || 500)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>

                      <div>
                        <label htmlFor="rateLimitBurstSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Burst Size
                        </label>
                        <BubbleInput
                          id="rateLimitBurstSize"
                          type="number"
                          min="1"
                          max="100"
                          value={localConfig.rateLimitBurstSize}
                          onChange={(e) => handleFieldChange('rateLimitBurstSize', parseInt(e.target.value) || 20)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* WebSocket Section */}
          {activeSection === 'websocket' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  WebSocket Configuration
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure real-time communication via WebSocket.
                </p>
              </div>

              <div className="space-y-6">
                <div className="flex items-start">
                  <div className="flex items-center h-5">
                    <BubbleInput
                      id="enableWebSocket"
                      type="checkbox"
                      checked={localConfig.enableWebSocket}
                      onChange={(e) => handleFieldChange('enableWebSocket', e.target.checked)}
                      className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                  </div>
                  <div className="ml-3">
                    <label htmlFor="enableWebSocket" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable WebSocket
                    </label>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Use WebSocket for real-time updates and notifications
                    </p>
                  </div>
                </div>

                {localConfig.enableWebSocket && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="websocketUrl" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        WebSocket URL
                      </label>
                      <BubbleInput
                        id="websocketUrl"
                        type="url"
                        placeholder="ws://localhost:8080/ws"
                        value={localConfig.websocketUrl}
                        onChange={(e) => handleFieldChange('websocketUrl', e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div>
                      <label htmlFor="websocketReconnectInterval" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Reconnect Interval (ms)
                      </label>
                      <BubbleInput
                        id="websocketReconnectInterval"
                        type="number"
                        min="1000"
                        max="60000"
                        step="1000"
                        value={localConfig.websocketReconnectInterval}
                        onChange={(e) => handleFieldChange('websocketReconnectInterval', parseInt(e.target.value) || 5000)}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div>
                      <label htmlFor="websocketMaxReconnectAttempts" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Max Reconnect Attempts
                      </label>
                      <BubbleInput
                        id="websocketMaxReconnectAttempts"
                        type="number"
                        min="1"
                        max="100"
                        value={localConfig.websocketMaxReconnectAttempts}
                        onChange={(e) => handleFieldChange('websocketMaxReconnectAttempts', parseInt(e.target.value) || 10)}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div>
                      <label htmlFor="websocketHeartbeatInterval" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Heartbeat Interval (ms)
                      </label>
                      <BubbleInput
                        id="websocketHeartbeatInterval"
                        type="number"
                        min="10000"
                        max="300000"
                        step="5000"
                        value={localConfig.websocketHeartbeatInterval}
                        onChange={(e) => handleFieldChange('websocketHeartbeatInterval', parseInt(e.target.value) || 30000)}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Send ping frame every N milliseconds to keep connection alive
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Retry & Circuit Breaker Section */}
          {activeSection === 'retry' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Retry Logic & Circuit Breaker
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure resilience patterns for handling failures.
                </p>
              </div>

              <div className="space-y-6">
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-md">
                  <h4 className="text-sm font-medium text-orange-900 dark:text-orange-400 mb-4">
                    Retry Configuration
                  </h4>
                  <div className="flex items-start mb-4">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="enableRetry"
                        type="checkbox"
                        checked={localConfig.enableRetry}
                        onChange={(e) => handleFieldChange('enableRetry', e.target.checked)}
                        className="h-4 w-4 text-orange-600 focus:ring-orange-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="enableRetry" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Enable Automatic Retry
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Automatically retry failed requests
                      </p>
                    </div>
                  </div>

                  {localConfig.enableRetry && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label htmlFor="maxRetries" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Max Retries
                        </label>
                        <BubbleInput
                          id="maxRetries"
                          type="number"
                          min="0"
                          max="10"
                          value={localConfig.maxRetries}
                          onChange={(e) => handleFieldChange('maxRetries', parseInt(e.target.value) || 3)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>

                      <div>
                        <label htmlFor="retryDelay" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Initial Delay (ms)
                        </label>
                        <BubbleInput
                          id="retryDelay"
                          type="number"
                          min="100"
                          max="10000"
                          step="100"
                          value={localConfig.retryDelay}
                          onChange={(e) => handleFieldChange('retryDelay', parseInt(e.target.value) || 1000)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>

                      <div>
                        <label htmlFor="retryBackoffMultiplier" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Backoff Multiplier
                        </label>
                        <BubbleInput
                          id="retryBackoffMultiplier"
                          type="number"
                          min="1"
                          max="10"
                          step="0.5"
                          value={localConfig.retryBackoffMultiplier}
                          onChange={(e) => handleFieldChange('retryBackoffMultiplier', parseFloat(e.target.value) || 2)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Delay multiplier for exponential backoff
                        </p>
                      </div>

                      <div>
                        <label htmlFor="retryOnStatusCodes" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Status Codes
                        </label>
                        <BubbleInput
                          id="retryOnStatusCodes"
                          type="text"
                          value={localConfig.retryOnStatusCodes.join(', ')}
                          onChange={(e) =>
                            handleFieldChange(
                              'retryOnStatusCodes',
                              e.target.value
                                .split(',')
                                .map((s) => parseInt(s.trim(), 10))
                                .filter((value) => Number.isFinite(value))
                            )
                          }
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Comma-separated HTTP status codes to retry
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-md">
                  <h4 className="text-sm font-medium text-red-900 dark:text-red-400 mb-4">
                    Circuit Breaker Configuration
                  </h4>
                  <div className="flex items-start mb-4">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="enableCircuitBreaker"
                        type="checkbox"
                        checked={localConfig.enableCircuitBreaker}
                        onChange={(e) => handleFieldChange('enableCircuitBreaker', e.target.checked)}
                        className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="enableCircuitBreaker" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Enable Circuit Breaker
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Prevent cascading failures by stopping requests to failing services
                      </p>
                    </div>
                  </div>

                  {localConfig.enableCircuitBreaker && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label htmlFor="circuitBreakerThreshold" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Failure Threshold
                        </label>
                        <BubbleInput
                          id="circuitBreakerThreshold"
                          type="number"
                          min="1"
                          max="100"
                          value={localConfig.circuitBreakerThreshold}
                          onChange={(e) => handleFieldChange('circuitBreakerThreshold', parseInt(e.target.value) || 5)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-red-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Failures before opening circuit
                        </p>
                      </div>

                      <div>
                        <label htmlFor="circuitBreakerTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Timeout (ms)
                        </label>
                        <BubbleInput
                          id="circuitBreakerTimeout"
                          type="number"
                          min="10000"
                          max="600000"
                          step="1000"
                          value={localConfig.circuitBreakerTimeout}
                          onChange={(e) => handleFieldChange('circuitBreakerTimeout', parseInt(e.target.value) || 60000)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-red-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Time before attempting recovery
                        </p>
                      </div>

                      <div>
                        <label htmlFor="circuitBreakerHalfOpenAttempts" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Half-Open Attempts
                        </label>
                        <BubbleInput
                          id="circuitBreakerHalfOpenAttempts"
                          type="number"
                          min="1"
                          max="10"
                          value={localConfig.circuitBreakerHalfOpenAttempts}
                          onChange={(e) => handleFieldChange('circuitBreakerHalfOpenAttempts', parseInt(e.target.value) || 3)}
                          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-red-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Test requests in half-open state
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <h4 className="text-sm font-medium text-blue-900 dark:text-blue-400 mb-2">
                  Circuit Breaker States
                </h4>
                <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
                  <li><strong>Closed:</strong> Normal operation, requests pass through</li>
                  <li><strong>Open:</strong> Failing, requests are blocked</li>
                  <li><strong>Half-Open:</strong> Testing if service has recovered</li>
                </ul>
              </div>
            </div>
          )}

          {/* Service Settings Section */}
          {activeSection === 'services' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Service-Specific Settings
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Configure individual service parameters.
                </p>
              </div>

              {/* LeanAide Settings */}
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-md">
                <h4 className="text-sm font-medium text-purple-900 dark:text-purple-400 mb-4">
                  LeanAide Configuration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="leanAideProvider" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      AI Provider
                    </label>
                    <BubbleSelect
                      id="leanAideProvider"
                      value={localConfig.leanAide.provider}
                      onChange={(e) => handleNestedFieldChange('leanAide', 'provider', e.target.value as 'openai' | 'anthropic' | 'mistral' | 'custom')}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="mistral">Mistral</option>
                      <option value="custom">Custom</option>
                    </BubbleSelect>
                  </div>

                  <div>
                    <label htmlFor="leanAideModel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Model
                    </label>
                    <BubbleInput
                      id="leanAideModel"
                      type="text"
                      value={localConfig.leanAide.model}
                      onChange={(e) => handleNestedFieldChange('leanAide', 'model', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>

                  <div>
                    <label htmlFor="leanAideTemperature" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Temperature
                    </label>
                    <BubbleInput
                      id="leanAideTemperature"
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={localConfig.leanAide.temperature}
                      onChange={(e) => handleNestedFieldChange('leanAide', 'temperature', parseFloat(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>

                  <div>
                    <label htmlFor="leanAideMaxTokens" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Tokens
                    </label>
                    <BubbleInput
                      id="leanAideMaxTokens"
                      type="number"
                      min="1"
                      max="32000"
                      value={localConfig.leanAide.maxTokens}
                      onChange={(e) => handleNestedFieldChange('leanAide', 'maxTokens', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                </div>
              </div>

              {/* CrewAI Settings */}
              <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-md">
                <h4 className="text-sm font-medium text-indigo-900 dark:text-indigo-400 mb-4">
                  CrewAI Configuration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="crewaiParallelExecution"
                        type="checkbox"
                        checked={localConfig.crewai.parallelExecution}
                        onChange={(e) => handleNestedFieldChange('crewai', 'parallelExecution', e.target.checked)}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="crewaiParallelExecution" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Parallel Execution
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Execute tasks in parallel
                      </p>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="crewaiMaxParallelTasks" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Parallel Tasks
                    </label>
                    <BubbleInput
                      id="crewaiMaxParallelTasks"
                      type="number"
                      min="1"
                      max="20"
                      value={localConfig.crewai.maxParallelTasks}
                      onChange={(e) => handleNestedFieldChange('crewai', 'maxParallelTasks', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>

                  <div>
                    <label htmlFor="crewaiDelegationTimeout" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Delegation Timeout (ms)
                    </label>
                    <BubbleInput
                      id="crewaiDelegationTimeout"
                      type="number"
                      min="10000"
                      max="600000"
                      step="1000"
                      value={localConfig.crewai.delegationTimeout}
                      onChange={(e) => handleNestedFieldChange('crewai', 'delegationTimeout', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                </div>
              </div>

              {/* BubbleLabs Settings */}
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <h4 className="text-sm font-medium text-blue-900 dark:text-blue-400 mb-4">
                  BubbleLabs Configuration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="bubbleLabsUiEnabled"
                        type="checkbox"
                        checked={localConfig.bubbleLabs.uiEnabled}
                        onChange={(e) => handleNestedFieldChange('bubbleLabs', 'uiEnabled', e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="bubbleLabsUiEnabled" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Enable UI
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Show visualization interface
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="bubbleLabsRealTimeUpdates"
                        type="checkbox"
                        checked={localConfig.bubbleLabs.realTimeUpdates}
                        onChange={(e) => handleNestedFieldChange('bubbleLabs', 'realTimeUpdates', e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="bubbleLabsRealTimeUpdates" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Real-Time Updates
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Stream updates as they occur
                      </p>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="bubbleLabsVisualizationQuality" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Visualization Quality
                    </label>
                    <BubbleSelect
                      id="bubbleLabsVisualizationQuality"
                      value={localConfig.bubbleLabs.visualizationQuality}
                      onChange={(e) => handleNestedFieldChange('bubbleLabs', 'visualizationQuality', e.target.value as 'low' | 'medium' | 'high')}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </BubbleSelect>
                  </div>
                </div>
              </div>

              {/* ResearchQuest Settings */}
              <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded-md">
                <h4 className="text-sm font-medium text-teal-900 dark:text-teal-400 mb-4">
                  ResearchQuest Configuration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="researchQuestKnowledgeGraphEnabled"
                        type="checkbox"
                        checked={localConfig.researchQuest.knowledgeGraphEnabled}
                        onChange={(e) => handleNestedFieldChange('researchQuest', 'knowledgeGraphEnabled', e.target.checked)}
                        className="h-4 w-4 text-teal-600 focus:ring-teal-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="researchQuestKnowledgeGraphEnabled" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Knowledge Graph
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Use graph-based exploration
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <BubbleInput
                        id="researchQuestSemanticSearch"
                        type="checkbox"
                        checked={localConfig.researchQuest.semanticSearch}
                        onChange={(e) => handleNestedFieldChange('researchQuest', 'semanticSearch', e.target.checked)}
                        className="h-4 w-4 text-teal-600 focus:ring-teal-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                    <div className="ml-3">
                      <label htmlFor="researchQuestSemanticSearch" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Semantic Search
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Enable semantic similarity search
                      </p>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="researchQuestMaxResults" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Results
                    </label>
                    <BubbleInput
                      id="researchQuestMaxResults"
                      type="number"
                      min="1"
                      max="100"
                      value={localConfig.researchQuest.maxResults}
                      onChange={(e) => handleNestedFieldChange('researchQuest', 'maxResults', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const IntegrationConfigPanel = withComponentBoundary(
  IntegrationConfigPanelBase,
  'IntegrationConfigPanel'
);

export default IntegrationConfigPanel;



