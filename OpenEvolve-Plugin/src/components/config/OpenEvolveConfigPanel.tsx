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
} from '../../types/plugin-types';
import { openevolvePlugin } from '../../utils/createOpenEvolvePlugin';
import { toast } from 'react-toastify';
import { BubbleButton, BubbleInput, BubbleSelect } from '@/components/bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import {
  Settings,
  Brain,
  Shield,
  PuzzlePiece,
  Network,
} from 'lucide-react';

interface OpenEvolveConfigPanelProps {
  plugin?: OpenEvolvePlugin;
  onConfigChange?: (config: OpenEvolvePluginState) => void;
}

const OpenEvolveConfigPanelBase: React.FC<OpenEvolveConfigPanelProps> = ({
  plugin = openevolvePlugin,
  onConfigChange,
}) => {
  const [config, setConfig] = useState<OpenEvolvePluginState>(plugin.getConfig());
  const [activeTab, setActiveTab] = useState<'general' | 'evolution' | 'adversarial' | 'decomposition' | 'mdap_maker'>('general');
  const [isLoading, setIsLoading] = useState(true);
  const [lastRecursionDepthLimit, setLastRecursionDepthLimit] = useState(
    config.decompositionConfig.recursionDepthLimit > 0
      ? config.decompositionConfig.recursionDepthLimit
      : 1
  );
  const [lastMaxSubProblems, setLastMaxSubProblems] = useState(
    config.decompositionConfig.maxSubProblems > 0
      ? config.decompositionConfig.maxSubProblems
      : 3
  );

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

  useEffect(() => {
    if (config.decompositionConfig.recursionDepthLimit > 0) {
      setLastRecursionDepthLimit(config.decompositionConfig.recursionDepthLimit);
    }
    if (config.decompositionConfig.maxSubProblems > 0) {
      setLastMaxSubProblems(config.decompositionConfig.maxSubProblems);
    }
  }, [config.decompositionConfig]);

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

  const executionHistory = config.executionHistory || [];
  const statistics = config.statistics || [];

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
            <BubbleButton onClick={() => handleConfigChange(config)} disabled={isLoading}>
              Save
            </BubbleButton>
            <BubbleButton onClick={handleReset} disabled={isLoading} variant="secondary">
              Reset
            </BubbleButton>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex flex-wrap gap-2 px-6 py-3" aria-label="Tabs">
          <BubbleButton
            onClick={() => setActiveTab('general')}
            variant={activeTab === 'general' ? 'primary' : 'secondary'}
            className="flex items-center gap-2"
          >
            <Settings className="w-4 h-4" /> General
          </BubbleButton>
          <BubbleButton
            onClick={() => setActiveTab('evolution')}
            variant={activeTab === 'evolution' ? 'primary' : 'secondary'}
            className="flex items-center gap-2"
          >
            <Brain className="w-4 h-4" /> Evolution
          </BubbleButton>
          <BubbleButton
            onClick={() => setActiveTab('adversarial')}
            variant={activeTab === 'adversarial' ? 'primary' : 'secondary'}
            className="flex items-center gap-2"
          >
            <Shield className="w-4 h-4" /> Adversarial
          </BubbleButton>
          <BubbleButton
            onClick={() => setActiveTab('decomposition')}
            variant={activeTab === 'decomposition' ? 'primary' : 'secondary'}
            className="flex items-center gap-2"
          >
            <PuzzlePiece className="w-4 h-4" /> Decomposition
          </BubbleButton>
          <BubbleButton
            onClick={() => setActiveTab('mdap_maker')}
            variant={activeTab === 'mdap_maker' ? 'primary' : 'secondary'}
            className="flex items-center gap-2"
          >
            <Network className="w-4 h-4" /> MDAP/MAKER
          </BubbleButton>
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
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>

                <div>
                  <label htmlFor="pluginStatus" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Plugin Status
                  </label>
                  <BubbleInput
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
                <IconWrapper icon={Settings} className="mr-2" /> Execution Statistics
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Total Executions
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-gray-900 dark:text-white">
                    {executionHistory.length}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Success Rate
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-green-600 dark:text-green-400">
                    {statistics.length > 0
                      ? `${(statistics.filter(s => s.status === 'completed').length / statistics.length * 100).toFixed(1)}%`
                      : 'N/A'}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Avg Performance
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-blue-600 dark:text-blue-400">
                    {statistics.length > 0
                      ? (statistics.reduce((sum, s) => sum + s.performanceScore, 0) / statistics.length).toFixed(2)
                      : 'N/A'}
                  </dd>
                </div>
                <div className="text-center">
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    Avg Quality
                  </dt>
                  <dd className="mt-1 text-3xl font-semibold text-purple-600 dark:text-purple-400">
                    {statistics.length > 0
                      ? (statistics.reduce((sum, s) => sum + s.qualityScore, 0) / statistics.length).toFixed(2)
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
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>

                <div>
                  <label htmlFor="maxIterations" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Iterations
                  </label>
                  <BubbleInput
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
                  <BubbleInput
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
                  <BubbleInput
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
                  <BubbleInput
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
                    <BubbleInput
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
                      <BubbleInput
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
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>

                <div>
                  <label htmlFor="redTeamSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Red Team Size
                  </label>
                  <BubbleInput
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
                  <BubbleInput
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
                  <BubbleInput
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
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MDAP/MAKER Integration
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center">
                    <BubbleInput
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
                      <BubbleInput
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
                <PuzzlePiece className="mr-2" /> Decomposition Configuration
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="decompositionStrategy" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Decomposition Strategy
                  </label>
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>

                <div>
                  <label htmlFor="maxSubProblems" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Sub-Problems
                  </label>
                  <BubbleInput
                    id="maxSubProblems"
                    type="number"
                    min="0"
                    value={config.decompositionConfig.maxSubProblems}
                    disabled={config.decompositionConfig.maxSubProblems === 0}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        maxSubProblems: (() => {
                          const nextValue = parseInt(e.target.value, 10) || 0;
                          if (nextValue > 0) {
                            setLastMaxSubProblems(nextValue);
                          }
                          return nextValue;
                        })(),
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <div className="mt-2 flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="maxSubProblemsUnlimited"
                      checked={config.decompositionConfig.maxSubProblems === 0}
                      onChange={(e) => setConfig({
                        ...config,
                        decompositionConfig: {
                          ...config.decompositionConfig,
                          maxSubProblems: e.target.checked ? 0 : lastMaxSubProblems || 3,
                        },
                      })}
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <label htmlFor="maxSubProblemsUnlimited" className="text-sm text-gray-600 dark:text-gray-300">
                      Unlimited
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">0 = unlimited</p>
                </div>

                <div>
                  <label htmlFor="recursionDepthLimit" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Recursion Depth Limit
                  </label>
                  <BubbleInput
                    id="recursionDepthLimit"
                    type="number"
                    min="0"
                    value={config.decompositionConfig.recursionDepthLimit}
                    disabled={config.decompositionConfig.recursionDepthLimit === 0}
                    onChange={(e) => setConfig({
                      ...config,
                      decompositionConfig: {
                        ...config.decompositionConfig,
                        recursionDepthLimit: (() => {
                          const nextValue = parseInt(e.target.value, 10) || 0;
                          if (nextValue > 0) {
                            setLastRecursionDepthLimit(nextValue);
                          }
                          return nextValue;
                        })(),
                      },
                    })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <div className="mt-2 flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="recursionDepthLimitUnlimited"
                      checked={config.decompositionConfig.recursionDepthLimit === 0}
                      onChange={(e) => setConfig({
                        ...config,
                        decompositionConfig: {
                          ...config.decompositionConfig,
                          recursionDepthLimit: e.target.checked ? 0 : lastRecursionDepthLimit || 1,
                        },
                      })}
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <label htmlFor="recursionDepthLimitUnlimited" className="text-sm text-gray-600 dark:text-gray-300">
                      Unlimited
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">0 = unlimited</p>
                </div>

                <div>
                  <label htmlFor="granularityLevel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Granularity Level
                  </label>
                  <BubbleSelect
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
                  </BubbleSelect>
                </div>

                <div>
                  <label htmlFor="minSubProblemSize" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Min Sub-Problem Size
                  </label>
                  <BubbleInput
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
                  <BubbleInput
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
                    <BubbleInput
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
                      <BubbleInput
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
                  <BubbleInput
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
                  <BubbleInput
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
                    <BubbleInput
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
                    <BubbleInput
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
                    <BubbleSelect
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
                    </BubbleSelect>
                  </div>

                  <div>
                    <label htmlFor="mdapMakerModel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model
                    </label>
                    <BubbleSelect
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
                    </BubbleSelect>
                  </div>

                  <div className="flex items-center">
                    <BubbleInput
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
                    <BubbleInput
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
export const OpenEvolveConfigPanel = withComponentBoundary(
  OpenEvolveConfigPanelBase,
  'OpenEvolveConfigPanel'
);

export default OpenEvolveConfigPanel;
