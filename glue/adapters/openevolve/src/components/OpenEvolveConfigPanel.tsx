/**
 * OpenEvolve Configuration Panel
 * 
 * Comprehensive React component for configuring OpenEvolve functionality
 */

import React, { useState, useEffect } from 'react';
import {
  OpenEvolvePlugin,
  OpenEvolvePluginState,
  OPENEVOLVE_PLUGIN_CONSTANTS,
} from '../types/plugin-types';
import { openevolvePlugin } from '../utils/createOpenEvolvePlugin';
import { toast } from 'react-toastify';

// Mock icons
const Settings = () => <span>⚙️</span>;
const Brain = () => <span>🧠</span>;
const Shield = () => <span>🛡️</span>;
const Puzzle = () => <span>🧩</span>;
const Network = () => <span>🌐</span>;

interface OpenEvolveConfigPanelProps {
  plugin?: OpenEvolvePlugin;
  onConfigChange?: (config: OpenEvolvePluginState) => void;
}

export const OpenEvolveConfigPanel: React.FC<OpenEvolveConfigPanelProps> = ({
  plugin = openevolvePlugin,
  onConfigChange,
}) => {
  const [config, setConfig] = useState<OpenEvolvePluginState>(plugin.getConfig());
  const [activeTab, setActiveTab] = useState<'general' | 'evolution' | 'adversarial' | 'decomposition' | 'mdap_maker'>('general');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        setIsLoading(true);
        const currentConfig = plugin.getConfig();
        setConfig(currentConfig);
        setIsLoading(false);
      } catch (error) {
        toast.error(`Failed to load configuration: ${error instanceof Error ? error.message : String(error)}`);
        setIsLoading(false);
      }
    };
    loadConfig();
  }, [plugin]);

  const handleConfigChange = async (newConfig: OpenEvolvePluginState) => {
    try {
      setIsLoading(true);
      await plugin.updateConfig(newConfig);
      setConfig(newConfig);
      onConfigChange?.(newConfig);
      toast.success('Configuration updated successfully');
      setIsLoading(false);
    } catch (error) {
      toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : String(error)}`);
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      setIsLoading(true);
      await plugin.resetConfig();
      const resetConfig = plugin.getConfig();
      setConfig(resetConfig);
      onConfigChange?.(resetConfig);
      toast.success('Configuration reset to defaults');
      setIsLoading(false);
    } catch (error) {
      toast.error(`Failed to reset configuration: ${error instanceof Error ? error.message : String(error)}`);
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="openevolve-config-panel p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600 dark:text-gray-300">Loading configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="openevolve-config-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400 mr-3" />
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Configuration
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleConfigChange(config)}
              disabled={isLoading}
              className="px-3 py-1 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
            <button
              onClick={handleReset}
              disabled={isLoading}
              className="px-3 py-1 text-sm font-medium text-white bg-yellow-600 hover:bg-yellow-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8 px-6" aria-label="Tabs">
          <button
            onClick={() => setActiveTab('general')}
            className={`px-4 py-4 font-medium text-sm flex items-center whitespace-nowrap border-b-2 ${
              activeTab === 'general'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <Settings className="mr-2 w-4 h-4" /> General
          </button>
          <button
            onClick={() => setActiveTab('evolution')}
            className={`px-4 py-4 font-medium text-sm flex items-center whitespace-nowrap border-b-2 ${
              activeTab === 'evolution'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <Brain className="mr-2 w-4 h-4" /> Evolution
          </button>
          <button
            onClick={() => setActiveTab('adversarial')}
            className={`px-4 py-4 font-medium text-sm flex items-center whitespace-nowrap border-b-2 ${
              activeTab === 'adversarial'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <Shield className="mr-2 w-4 h-4" /> Adversarial
          </button>
          <button
            onClick={() => setActiveTab('decomposition')}
            className={`px-4 py-4 font-medium text-sm flex items-center whitespace-nowrap border-b-2 ${
              activeTab === 'decomposition'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <Puzzle className="mr-2 w-4 h-4" /> Decomposition
          </button>
          <button
            onClick={() => setActiveTab('mdap_maker')}
            className={`px-4 py-4 font-medium text-sm flex items-center whitespace-nowrap border-b-2 ${
              activeTab === 'mdap_maker'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <Network className="mr-2 w-4 h-4" /> MDAP/MAKER
          </button>
        </nav>
      </div>

      {/* Configuration Content */}
      <div className="px-6 py-4">
        {/* General Configuration Tab */}
        {activeTab === 'general' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Settings className="mr-2" /> General Settings
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="defaultExecutionMethod" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Default Execution Method
                  </label>
                  <select
                    id="defaultExecutionMethod"
                    value={config.defaultExecutionMethod}
                    onChange={(e) => setConfig({ ...config, defaultExecutionMethod: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    {OPENEVOLVE_PLUGIN_CONSTANTS.EXECUTION_METHODS.map((method) => (
                      <option key={method} value={method}>{
                        method.replace('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
                      }</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="pluginStatus" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Plugin Status
                  </label>
                  <input
                    id="pluginStatus"
                    type="text"
                    value={config.status}
                    readOnly
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white cursor-not-allowed"
                  />
                </div>
              </div>

              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Plugin Metadata
                </label>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
                  <pre className="text-sm text-gray-600 dark:text-gray-300 overflow-x-auto">
                    {JSON.stringify(config.metadata, null, 2)}
                  </pre>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Settings className="mr-2" /> Execution Statistics
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Total Executions
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-gray-900 dark:text-white">
                    {config.executionHistory.length}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Success Rate
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-green-600 dark:text-green-400">
                    {config.statistics.length > 0
                      ? `${(config.statistics.filter(s => s.status === 'completed').length / config.statistics.length * 100).toFixed(1)}%`
                      : 'N/A'}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Avg Performance
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-blue-600 dark:text-blue-400">
                    {config.statistics.length > 0
                      ? (config.statistics.reduce((sum, s) => sum + s.performanceScore, 0) / config.statistics.length).toFixed(2)
                      : 'N/A'}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Avg Quality
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-purple-600 dark:text-purple-400">
                    {config.statistics.length > 0
                      ? (config.statistics.reduce((sum, s) => sum + s.qualityScore, 0) / config.statistics.length).toFixed(2)
                      : 'N/A'}
                  </dd>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Evolution Configuration Tab */}
        {activeTab === 'evolution' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Brain className="mr-2" /> Evolution Configuration
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="evolutionMode" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Evolution Mode
                  </label>
                  <select
                    id="evolutionMode"
                    value={config.evolutionConfig.evolutionMode}
                    onChange={(e) => setConfig({
                      ...config,
                      evolutionConfig: {
                        ...config.evolutionConfig,
                        evolutionMode: e.target.value as any,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    {OPENEVOLVE_PLUGIN_CONSTANTS.EVOLUTION_STRATEGIES.map((strategy) => (
                      <option key={strategy} value={strategy}>{
                        strategy.replace('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
                      }</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="maxIterations" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Iterations
                  </label>
                  <input
                    id="maxIterations"
                    type="number"
                    min="1"
                    value={config.evolutionConfig.maxIterations}
                    onChange={(e) => setConfig({
                      ...config,
                      evolutionConfig: {
                        ...config.evolutionConfig,
                        maxIterations: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="populationSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Population Size
                  </label>
                  <input
                    id="populationSize"
                    type="number"
                    min="1"
                    value={config.evolutionConfig.populationSize}
                    onChange={(e) => setConfig({
                      ...config,
                      evolutionConfig: {
                        ...config.evolutionConfig,
                        populationSize: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Temperature
                  </label>
                  <input
                    id="temperature"
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={config.evolutionConfig.temperature}
                    onChange={(e) => setConfig({
                      ...config,
                      evolutionConfig: {
                        ...config.evolutionConfig,
                        temperature: parseFloat(e.target.value),
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="modelId" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Model
                  </label>
                  <input
                    id="modelId"
                    type="text"
                    value={config.evolutionConfig.modelId}
                    onChange={(e) => setConfig({
                      ...config,
                      evolutionConfig: {
                        ...config.evolutionConfig,
                        modelId: e.target.value,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MDAP/MAKER Integration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center">
                    <input
                      id="evolutionMdapMakerEnabled"
                      type="checkbox"
                      checked={config.evolutionConfig.mdapMakerEnabled}
                      onChange={(e) => setConfig({
                        ...config,
                        evolutionConfig: {
                          ...config.evolutionConfig,
                          mdapMakerEnabled: e.target.checked,
                        },
                      })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <label htmlFor="evolutionMdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                      Enable MDAP/MAKER
                    </label>
                  </div>

                  {config.evolutionConfig.mdapMakerEnabled && (
                    <div className="flex items-center">
                      <input
                        id="evolutionMdapMakerAutoSelect"
                        type="checkbox"
                        checked={config.evolutionConfig.mdapMakerAutoSelect}
                        onChange={(e) => setConfig({
                          ...config,
                          evolutionConfig: {
                            ...config.evolutionConfig,
                            mdapMakerAutoSelect: e.target.checked,
                          },
                        })}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <label htmlFor="evolutionMdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Auto-Select for Critical Tasks
                      </label>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Adversarial Configuration Tab */}
        {activeTab === 'adversarial' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Shield className="mr-2" /> Adversarial Configuration
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="adversarialMode" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Adversarial Mode
                  </label>
                  <select
                    id="adversarialMode"
                    value={config.adversarialConfig.adversarialMode}
                    onChange={(e) => setConfig({
                      ...config,
                      adversarialConfig: {
                        ...config.adversarialConfig,
                        adversarialMode: e.target.value as any,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    {OPENEVOLVE_PLUGIN_CONSTANTS.ADVERSARIAL_STRATEGIES.map((strategy) => (
                      <option key={strategy} value={strategy}>{
                        strategy.replace('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
                      }</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="redTeamSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Red Team Size
                  </label>
                  <input
                    id="redTeamSize"
                    type="number"
                    min="1"
                    value={config.adversarialConfig.redTeamSize}
                    onChange={(e) => setConfig({
                      ...config,
                      adversarialConfig: {
                        ...config.adversarialConfig,
                        redTeamSize: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="blueTeamSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Blue Team Size
                  </label>
                  <input
                    id="blueTeamSize"
                    type="number"
                    min="1"
                    value={config.adversarialConfig.blueTeamSize}
                    onChange={(e) => setConfig({
                      ...config,
                      adversarialConfig: {
                        ...config.adversarialConfig,
                        blueTeamSize: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="maxRounds" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Rounds
                  </label>
                  <input
                    id="maxRounds"
                    type="number"
                    min="1"
                    value={config.adversarialConfig.maxRounds}
                    onChange={(e) => setConfig({
                      ...config,
                      adversarialConfig: {
                        ...config.adversarialConfig,
                        maxRounds: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="contentType" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Content Type
                  </label>
                  <select
                    id="contentType"
                    value={config.adversarialConfig.contentType}
                    onChange={(e) => setConfig({
                      ...config,
                      adversarialConfig: {
                        ...config.adversarialConfig,
                        contentType: e.target.value,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="code">Code</option>
                    <option value="text">Text</option>
                    <option value="design">Design</option>
                    <option value="strategy">Strategy</option>
                  </select>
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MDAP/MAKER Integration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center">
                    <input
                      id="adversarialMdapMakerEnabled"
                      type="checkbox"
                      checked={config.adversarialConfig.mdapMakerEnabled}
                      onChange={(e) => setConfig({
                        ...config,
                        adversarialConfig: {
                          ...config.adversarialConfig,
                          mdapMakerEnabled: e.target.checked,
                        },
                      })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <label htmlFor="adversarialMdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                      Enable MDAP/MAKER
                    </label>
                  </div>

                  {config.adversarialConfig.mdapMakerEnabled && (
                    <div className="flex items-center">
                      <input
                        id="adversarialMdapMakerAutoSelect"
                        type="checkbox"
                        checked={config.adversarialConfig.mdapMakerAutoSelect}
                        onChange={(e) => setConfig({
                          ...config,
                          adversarialConfig: {
                            ...config.adversarialConfig,
                            mdapMakerAutoSelect: e.target.checked,
                          },
                        })}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <label htmlFor="adversarialMdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Auto-Select for Critical Tasks
                      </label>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Decomposition Configuration Tab */}
        {activeTab === 'decomposition' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Puzzle className="mr-2" /> Decomposition Configuration
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="decompositionStrategy" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Decomposition Strategy
                  </label>
                  <select
                    id="decompositionStrategy"
                    value={config.decompositionConfig.decompositionStrategy}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        decompositionStrategy: e.target.value as any,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    {OPENEVOLVE_PLUGIN_CONSTANTS.DECOMPOSITION_STRATEGIES.map((strategy) => (
                      <option key={strategy} value={strategy}>{
                        strategy.replace('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
                      }</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="maxSubProblems" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Sub-Problems
                  </label>
                  <input
                    id="maxSubProblems"
                    type="number"
                    min="1"
                    value={config.decompositionConfig.maxSubProblems}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        maxSubProblems: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="granularityLevel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Granularity Level
                  </label>
                  <select
                    id="granularityLevel"
                    value={config.decompositionConfig.granularityLevel}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        granularityLevel: e.target.value,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="minSubProblemSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Min Sub-Problem Size
                  </label>
                  <input
                    id="minSubProblemSize"
                    type="number"
                    min="10"
                    value={config.decompositionConfig.minSubProblemSize}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        minSubProblemSize: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="maxSubProblemSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Sub-Problem Size
                  </label>
                  <input
                    id="maxSubProblemSize"
                    type="number"
                    min="50"
                    value={config.decompositionConfig.maxSubProblemSize}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        maxSubProblemSize: parseInt(e.target.value) || 0,
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MDAP/MAKER Integration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center">
                    <input
                      id="decompositionMdapMakerEnabled"
                      type="checkbox"
                      checked={config.decompositionConfig.mdapMakerEnabled}
                      onChange={(e) => setConfig({
                        ...config,
                        decompositionConfig: {
                          ...config.decompositionConfig,
                          mdapMakerEnabled: e.target.checked,
                        },
                      })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <label htmlFor="decompositionMdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                      Enable MDAP/MAKER
                    </label>
                  </div>

                  {config.decompositionConfig.mdapMakerEnabled && (
                    <div className="flex items-center">
                      <input
                        id="decompositionMdapMakerAutoSelect"
                        type="checkbox"
                        checked={config.decompositionConfig.mdapMakerAutoSelect}
                        onChange={(e) => setConfig({
                          ...config,
                          decompositionConfig: {
                            ...config.decompositionConfig,
                            mdapMakerAutoSelect: e.target.checked,
                          },
                        })}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <label htmlFor="decompositionMdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Auto-Select for Critical Tasks
                      </label>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* MDAP/MAKER Configuration Tab */}
        {activeTab === 'mdap_maker' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Network className="mr-2" /> MDAP/MAKER Configuration
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center">
                  <input
                    id="mdapMakerEnabled"
                    type="checkbox"
                    checked={config.mdapMaker?.enabled || false}
                    onChange={(e) => setConfig({
                      ...config,
                      mdapMaker: {
                        ...config.mdapMaker,
                        enabled: e.target.checked,
                      },
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                  />
                  <label htmlFor="mdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                    Enable MDAP/MAKER
                  </label>
                </div>

                <div className="flex items-center">
                  <input
                    id="mdapMakerAutoSelect"
                    type="checkbox"
                    checked={config.mdapMaker?.autoSelect || false}
                    onChange={(e) => setConfig({
                      ...config,
                      mdapMaker: {
                        ...config.mdapMaker,
                        autoSelect: e.target.checked,
                      },
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                  />
                  <label htmlFor="mdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                    Auto-Select for Critical Tasks
                  </label>
                </div>
              </div>

              {config.mdapMaker?.enabled && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="mdapMakerMaxDepth" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Max Depth (K)
                    </label>
                    <input
                      id="mdapMakerMaxDepth"
                      type="number"
                      min="1"
                      max="20"
                      value={config.mdapMaker.maxDepth}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          maxDepth: parseInt(e.target.value) || 0,
                        },
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      Maximum depth for MDAP tree exploration
                    </p>
                  </div>

                  <div>
                    <label htmlFor="mdapMakerKAhead" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      K-Ahead
                    </label>
                    <input
                      id="mdapMakerKAhead"
                      type="number"
                      min="1"
                      max="10"
                      value={config.mdapMaker.kAhead}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          kAhead: parseInt(e.target.value) || 0,
                        },
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      Number of steps to look ahead in planning
                    </p>
                  </div>

                  <div>
                    <label htmlFor="mdapMakerProvider" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      AI Provider
                    </label>
                    <select
                      id="mdapMakerProvider"
                      value={config.mdapMaker.provider}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          provider: e.target.value,
                        },
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="mistral">Mistral</option>
                      <option value="google">Google</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="mdapMakerModel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model
                    </label>
                    <select
                      id="mdapMakerModel"
                      value={config.mdapMaker.model}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          model: e.target.value,
                        },
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="gpt-4">GPT-4</option>
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                      <option value="claude-3-opus">Claude 3 Opus</option>
                      <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                      <option value="mistral-large">Mistral Large</option>
                    </select>
                  </div>

                  <div className="flex items-center">
                    <input
                      id="mdapMakerRedFlagging"
                      type="checkbox"
                      checked={config.mdapMaker.redFlagging}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          redFlagging: e.target.checked,
                        },
                      })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <label htmlFor="mdapMakerRedFlagging" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                      Red-Flagging
                    </label>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 ml-6">
                      Automatically flag potential issues during execution
                    </p>
                  </div>

                  <div className="flex items-center">
                    <input
                      id="mdapMakerAdaptiveK"
                      type="checkbox"
                      checked={config.mdapMaker.adaptiveK}
                      onChange={(e) => setConfig({
                        ...config,
                        mdapMaker: {
                          ...config.mdapMaker,
                          adaptiveK: e.target.checked,
                        },
                      })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <label htmlFor="mdapMakerAdaptiveK" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                      Adaptive K
                    </label>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 ml-6">
                      Dynamically adjust exploration depth based on complexity
                    </p>
                  </div>
                </div>
              )}

              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Auto-Selection Keywords
                </h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                  Tasks containing these keywords will automatically use MDAP/MAKER for enhanced reliability
                </p>
                <div className="flex flex-wrap gap-2">
                  {config.mdapMaker?.autoSelectionKeywords?.map((keyword, index) => (
                    <span key={index} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full">
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <h4 className="text-sm font-medium text-blue-800 dark:text-blue-400 mb-2">
                  About MDAP/MAKER Technology
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  MDAP/MAKER provides a zero-error guarantee (P(success) ≈ 99%+ with k=5) for critical tasks.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Export the component
