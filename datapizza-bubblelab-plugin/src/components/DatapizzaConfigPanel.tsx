// Datapizza Configuration Panel Component
// React component for configuring the Datapizza plugin

import React, { useState, useEffect } from 'react';
import { X, Save, Settings, Database, Pipeline, Cpu, Network, Shield, Eye, EyeOff } from 'lucide-react';
import { DatapizzaPluginConfig, DATAPIZZA_PIPELINE_TYPES, DATAPIZZA_DATA_DOMAINS } from '../types/plugin-types';

export interface DatapizzaConfigPanelProps {
  /** Initial configuration */
  initialConfig?: Partial<DatapizzaPluginConfig>;
  
  /** Callback when configuration is saved */
  onSave: (config: DatapizzaPluginConfig) => void;
  
  /** Callback when configuration is cancelled */
  onCancel: () => void;
  
  /** Show advanced options */
  showAdvanced?: boolean;
}

export function DatapizzaConfigPanel({
  initialConfig,
  onSave,
  onCancel,
  showAdvanced = false
}: DatapizzaConfigPanelProps) {
  const [config, setConfig] = useState<DatapizzaPluginConfig>({
    enabled: true,
    serverUrl: 'http://localhost:3000/datapizza',
    apiKey: '',
    timeout: 300,
    pipelineEnabled: true,
    autoDetectDataSources: true,
    defaultPipelineType: 'standard',
    dataProcessingConfig: {
      chunkSize: 1000,
      overlapSize: 200,
      embeddingModel: 'text-embedding-ada-002',
      vectorStoreType: 'qdrant',
      maxParallelProcesses: 4
    },
    agentConfigurations: {
      agent1: {
        enabled: true,
        maxTasks: 10,
        timeout: 60
      },
      agent2: {
        enabled: true,
        parallelExecution: true,
        maxWorkers: 4
      },
      agent3: {
        enabled: true,
        critiqueLevel: 'standard'
      }
    },
    integrateWithWorkflow: true,
    integrateWithKnowledgeGraph: true,
    integrateWithExternalSources: true,
    enableCaching: true,
    cacheTTLSeconds: 3600,
    maxProcessingTime: 300,
    showAdvancedOptions: false,
    showDebugInfo: false,
    theme: 'system'
  });

  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    if (initialConfig) {
      setConfig(prev => ({ ...prev, ...initialConfig }));
    }
  }, [initialConfig]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    
    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      setConfig(prev => ({ ...prev, [name]: checked }));
    } else if (name.startsWith('dataProcessingConfig.')) {
      const fieldName = name.replace('dataProcessingConfig.', '');
      setConfig(prev => ({
        ...prev,
        dataProcessingConfig: {
          ...prev.dataProcessingConfig,
          [fieldName]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value
        }
      }));
    } else if (name.startsWith('agentConfigurations.')) {
      const [agent, field] = name.replace('agentConfigurations.', '').split('.') as [keyof DatapizzaPluginConfig['agentConfigurations'], string];
      setConfig(prev => ({
        ...prev,
        agentConfigurations: {
          ...prev.agentConfigurations,
          [agent]: {
            ...prev.agentConfigurations[agent],
            [field]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value
          }
        }
      }));
    } else {
      setConfig(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(config);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-blue-500" />
            <h3 className="font-semibold text-lg">Datapizza Configuration</h3>
          </div>
          <button
            onClick={onCancel}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            aria-label="Close configuration panel"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-6">
          {/* Basic Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Database className="h-4 w-4" />
              Basic Configuration
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Enable Plugin
                </label>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="enabled"
                    checked={config.enabled}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Server URL
                </label>
                <input
                  type="text"
                  name="serverUrl"
                  value={config.serverUrl}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  placeholder="http://localhost:3000/datapizza"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  API Key
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? "text" : "password"}
                    name="apiKey"
                    value={config.apiKey}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white pr-10"
                    placeholder="Your API key"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-400"
                  >
                    {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Timeout (ms)
                </label>
                <input
                  type="number"
                  name="timeout"
                  value={config.timeout}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  min="100"
                  max="30000"
                />
              </div>
            </div>
          </div>

          {/* Pipeline Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Pipeline className="h-4 w-4" />
              Pipeline Configuration
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Enable Pipeline
                </label>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="pipelineEnabled"
                    checked={config.pipelineEnabled}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Auto Detect Data Sources
                </label>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="autoDetectDataSources"
                    checked={config.autoDetectDataSources}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Default Pipeline Type
                </label>
                <select
                  name="defaultPipelineType"
                  value={config.defaultPipelineType}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                >
                  {DATAPIZZA_PIPELINE_TYPES.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Data Processing Configuration */}
          {showAdvanced && (
            <div className="space-y-4">
              <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                Data Processing Configuration
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Chunk Size
                  </label>
                  <input
                    type="number"
                    name="dataProcessingConfig.chunkSize"
                    value={config.dataProcessingConfig.chunkSize}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    min="100"
                    max="10000"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Overlap Size
                  </label>
                  <input
                    type="number"
                    name="dataProcessingConfig.overlapSize"
                    value={config.dataProcessingConfig.overlapSize}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    min="0"
                    max="1000"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Embedding Model
                  </label>
                  <input
                    type="text"
                    name="dataProcessingConfig.embeddingModel"
                    value={config.dataProcessingConfig.embeddingModel}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    placeholder="text-embedding-ada-002"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Vector Store Type
                  </label>
                  <input
                    type="text"
                    name="dataProcessingConfig.vectorStoreType"
                    value={config.dataProcessingConfig.vectorStoreType}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    placeholder="qdrant"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Parallel Processes
                  </label>
                  <input
                    type="number"
                    name="dataProcessingConfig.maxParallelProcesses"
                    value={config.dataProcessingConfig.maxParallelProcesses}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    min="1"
                    max="16"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Integration Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Network className="h-4 w-4" />
              Integration Configuration
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="integrateWithWorkflow"
                  checked={config.integrateWithWorkflow}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Integrate with Workflow
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="integrateWithKnowledgeGraph"
                  checked={config.integrateWithKnowledgeGraph}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Integrate with Knowledge Graph
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="integrateWithExternalSources"
                  checked={config.integrateWithExternalSources}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Integrate with External Sources
                </label>
              </div>
            </div>
          </div>

          {/* Performance Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Performance Configuration
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="enableCaching"
                  checked={config.enableCaching}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Enable Caching
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Cache TTL (seconds)
                </label>
                <input
                  type="number"
                  name="cacheTTLSeconds"
                  value={config.cacheTTLSeconds}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  min="60"
                  max="86400"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Processing Time (seconds)
                </label>
                <input
                  type="number"
                  name="maxProcessingTime"
                  value={config.maxProcessingTime}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  min="30"
                  max="3600"
                />
              </div>
            </div>
          </div>

          {/* UI Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Eye className="h-4 w-4" />
              UI Configuration
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="showAdvancedOptions"
                  checked={config.showAdvancedOptions}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Show Advanced Options
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="showDebugInfo"
                  checked={config.showDebugInfo}
                  onChange={handleInputChange}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  Show Debug Information
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Theme
                </label>
                <select
                  name="theme"
                  value={config.theme}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="system">System</option>
                </select>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}