/**
 * ROMA Configuration Panel Component
 * 
 * This component provides a comprehensive UI for configuring the ROMA plugin.
 * It includes tabs for general settings, agent configuration, MCP servers, and toolkits.
 */

import React, { useState, useEffect } from 'react';
import { RomaConfigPanelProps, RomaModuleType, RomaPredictionStrategy, RomaMcpServerConfig, RomaToolkitConfig } from '../types/plugin-types';
import { toast } from 'react-toastify';
import { Settings, Server, Tool, Bot, Brain, Code, FileText, Search, Database, GitHub, Cpu, Calculator, Webhook, Plus, Trash2, Save, X } from 'lucide-react';

const RomaConfigPanel: React.FC<RomaConfigPanelProps> = ({ plugin, onConfigChange, onClose }) => {
  const [activeTab, setActiveTab] = useState<'general' | 'agents' | 'mcps' | 'toolkits' | 'mdap_maker'>('general');
  const [config, setConfig] = useState(plugin.getState());
  const [isSaving, setIsSaving] = useState(false);
  const [newMcpServer, setNewMcpServer] = useState<Partial<RomaMcpServerConfig>>({
    server_name: '',
    server_type: 'http',
    enabled: true
  });
  const [newToolkit, setNewToolkit] = useState<Partial<RomaToolkitConfig>>({
    class_name: '',
    enabled: true
  });

  // Module icons mapping
  const moduleIcons: Record<RomaModuleType, React.ReactNode> = {
    atomizer: <Brain className="w-4 h-4" />,
    planner: <FileText className="w-4 h-4" />,
    executor: <Cpu className="w-4 h-4" />,
    aggregator: <Database className="w-4 h-4" />,
    verifier: <Search className="w-4 h-4" />
  };

  // Strategy icons mapping
  const strategyIcons: Record<RomaPredictionStrategy, React.ReactNode> = {
    predict: <Bot className="w-4 h-4" />,
    chain_of_thought: <Brain className="w-4 h-4" />,
    react: <Code className="w-4 h-4" />,
    code_act: <Code className="w-4 h-4" />,
    best_of_n: <Calculator className="w-4 h-4" />,
    refine: <FileText className="w-4 h-4" />,
    parallel: <Webhook className="w-4 h-4" />,
    majority: <Database className="w-4 h-4" />
  };

  // Toolkit options
  const toolkitOptions = [
    { value: 'FileToolkit', label: 'File Operations', icon: <FileText className="w-4 h-4" /> },
    { value: 'CalculatorToolkit', label: 'Calculator', icon: <Calculator className="w-4 h-4" /> },
    { value: 'E2BToolkit', label: 'Code Execution', icon: <Code className="w-4 h-4" /> },
    { value: 'SerperToolkit', label: 'Web Search', icon: <Search className="w-4 h-4" /> },
    { value: 'CoinGeckoToolkit', label: 'CoinGecko', icon: <Database className="w-4 h-4" /> },
    { value: 'BinanceToolkit', label: 'Binance', icon: <Database className="w-4 h-4" /> },
    { value: 'DefiLlamaToolkit', label: 'DefiLlama', icon: <Database className="w-4 h-4" /> },
    { value: 'ArkhamToolkit', label: 'Arkham', icon: <Search className="w-4 h-4" /> },
    { value: 'MCPToolkit', label: 'MCP Server', icon: <Server className="w-4 h-4" /> }
  ];

  // MCP server type options
  const mcpServerTypeOptions = [
    { value: 'http', label: 'HTTP/SSE Server' },
    { value: 'stdio', label: 'Stdio Subprocess' }
  ];

  // Handle configuration changes
  const handleConfigChange = async () => {
    try {
      setIsSaving(true);
      await plugin.updateConfig(config);
      toast.success('ROMA configuration saved successfully');
      if (onConfigChange) {
        onConfigChange(config);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to save configuration';
      toast.error(`Failed to save configuration: ${errorMessage}`);
      console.error('Configuration save error:', error);
    } finally {
      setIsSaving(false);
    }
  };

  // Handle MCP server addition
  const handleAddMcpServer = async () => {
    try {
      if (!newMcpServer.server_name) {
        toast.error('Server name is required');
        return;
      }

      const mcpConfig: RomaMcpServerConfig = {
        server_name: newMcpServer.server_name!,
        server_type: newMcpServer.server_type || 'http',
        url: newMcpServer.url,
        command: newMcpServer.command,
        args: newMcpServer.args,
        headers: newMcpServer.headers,
        env: newMcpServer.env,
        use_storage: newMcpServer.use_storage,
        storage_threshold_kb: newMcpServer.storage_threshold_kb,
        enabled: newMcpServer.enabled
      };

      await plugin.addMcpServer(mcpConfig);
      setConfig(plugin.getState());
      setNewMcpServer({ server_name: '', server_type: 'http', enabled: true });
      toast.success(`MCP server ${mcpConfig.server_name} added successfully`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add MCP server';
      toast.error(`Failed to add MCP server: ${errorMessage}`);
      console.error('MCP server addition error:', error);
    }
  };

  // Handle toolkit addition
  const handleAddToolkit = async () => {
    try {
      if (!newToolkit.class_name) {
        toast.error('Toolkit class name is required');
        return;
      }

      const toolkitConfig: RomaToolkitConfig = {
        class_name: newToolkit.class_name!,
        enabled: newToolkit.enabled || true,
        toolkit_config: newToolkit.toolkit_config,
        include_tools: newToolkit.include_tools,
        exclude_tools: newToolkit.exclude_tools
      };

      await plugin.addToolkit(toolkitConfig);
      setConfig(plugin.getState());
      setNewToolkit({ class_name: '', enabled: true });
      toast.success(`Toolkit ${toolkitConfig.class_name} added successfully`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add toolkit';
      toast.error(`Failed to add toolkit: ${errorMessage}`);
      console.error('Toolkit addition error:', error);
    }
  };

  // Handle MCP server removal
  const handleRemoveMcpServer = async (serverName: string) => {
    try {
      await plugin.removeMcpServer(serverName);
      setConfig(plugin.getState());
      toast.success(`MCP server ${serverName} removed successfully`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to remove MCP server';
      toast.error(`Failed to remove MCP server: ${errorMessage}`);
      console.error('MCP server removal error:', error);
    }
  };

  // Handle toolkit removal
  const handleRemoveToolkit = async (toolkitName: string) => {
    try {
      await plugin.removeToolkit(toolkitName);
      setConfig(plugin.getState());
      toast.success(`Toolkit ${toolkitName} removed successfully`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to remove toolkit';
      toast.error(`Failed to remove toolkit: ${errorMessage}`);
      console.error('Toolkit removal error:', error);
    }
  };

  // Update agent configuration
  const handleAgentConfigChange = (module: RomaModuleType, field: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      agents: {
        ...prev.agents,
        [module]: {
          ...prev.agents?.[module],
          [field]: value
        }
      }
    }));
  };

  // Update MCP server configuration
  const handleMcpServerChange = (index: number, field: string, value: any) => {
    setConfig(prev => {
      const updatedServers = [...(prev.mcpServers || [])];
      updatedServers[index] = { ...updatedServers[index], [field]: value };
      return { ...prev, mcpServers: updatedServers };
    });
  };

  // Update toolkit configuration
  const handleToolkitChange = (index: number, field: string, value: any) => {
    setConfig(prev => {
      const executorAgent = prev.agents?.executor || {};
      const updatedToolkits = [...(executorAgent.toolkits || [])];
      updatedToolkits[index] = { ...updatedToolkits[index], [field]: value };
      return {
        ...prev,
        agents: {
          ...prev.agents,
          executor: {
            ...executorAgent,
            toolkits: updatedToolkits
          }
        }
      };
    });
  };

  return (
    <div className="roma-config-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
          <Settings className="mr-2" /> ROMA Configuration
        </h2>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          aria-label="Close"
        >
          <X className="w-6 h-6" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 mb-6 overflow-x-auto">
        <button
          onClick={() => setActiveTab('general')}
          className={`px-4 py-2 font-medium text-sm flex items-center whitespace-nowrap ${
            activeTab === 'general'
              ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          <Settings className="mr-2 w-4 h-4" /> General
        </button>
        <button
          onClick={() => setActiveTab('agents')}
          className={`px-4 py-2 font-medium text-sm flex items-center whitespace-nowrap ${
            activeTab === 'agents'
              ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          <Bot className="mr-2 w-4 h-4" /> Agents
        </button>
        <button
          onClick={() => setActiveTab('mcps')}
          className={`px-4 py-2 font-medium text-sm flex items-center whitespace-nowrap ${
            activeTab === 'mcps'
              ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          <Server className="mr-2 w-4 h-4" /> MCP Servers
        </button>
        <button
          onClick={() => setActiveTab('toolkits')}
          className={`px-4 py-2 font-medium text-sm flex items-center whitespace-nowrap ${
            activeTab === 'toolkits'
              ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          <Tool className="mr-2 w-4 h-4" /> Toolkits
        </button>
        <button
          onClick={() => setActiveTab('mdap_maker')}
          className={`px-4 py-2 font-medium text-sm flex items-center whitespace-nowrap ${
            activeTab === 'mdap_maker'
              ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          <Brain className="mr-2 w-4 h-4" /> MDAP/MAKER
        </button>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {/* General Settings Tab */}
        {activeTab === 'general' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                ROMA Server URL
              </label>
              <input
                type="text"
                value={config.serverUrl || ''}
                onChange={(e) => setConfig({ ...config, serverUrl: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                placeholder="http://localhost:8000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                API Key
              </label>
              <input
                type="password"
                value={config.apiKey || ''}
                onChange={(e) => setConfig({ ...config, apiKey: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                placeholder="Your ROMA API key"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Default Profile
                </label>
                <select
                  value={config.defaultProfile || 'general'}
                  onChange={(e) => setConfig({ ...config, defaultProfile: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                >
                  <option value="general">General</option>
                  <option value="crypto_agent">Crypto Agent</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Depth
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={config.maxDepth || 3}
                  onChange={(e) => setConfig({ ...config, maxDepth: parseInt(e.target.value) || 3 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Timeout (ms)
                </label>
                <input
                  type="number"
                  min="1000"
                  max="300000"
                  value={config.timeout || 30000}
                  onChange={(e) => setConfig({ ...config, timeout: parseInt(e.target.value) || 30000 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Cache TTL (ms)
                </label>
                <input
                  type="number"
                  min="60000"
                  max="86400000"
                  value={config.cacheTTL || 3600000}
                  onChange={(e) => setConfig({ ...config, cacheTTL: parseInt(e.target.value) || 3600000 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.enableObservability || false}
                  onChange={(e) => setConfig({ ...config, enableObservability: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable Observability</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.enableStorage || false}
                  onChange={(e) => setConfig({ ...config, enableStorage: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable Storage</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.debugMode || false}
                  onChange={(e) => setConfig({ ...config, debugMode: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Debug Mode</span>
              </label>
            </div>

            {config.enableStorage && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Storage Base Path
                </label>
                <input
                  type="text"
                  value={config.storageBasePath || './roma-storage'}
                  onChange={(e) => setConfig({ ...config, storageBasePath: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="./roma-storage"
                />
              </div>
            )}
          </div>
        )}

        {/* Agents Configuration Tab */}
        {activeTab === 'agents' && (
          <div className="space-y-6">
            {Object.entries(config.agents || {}).map(([module, agentConfig]) => (
              <div key={module} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center capitalize">
                    {moduleIcons[module as RomaModuleType]} {module}
                  </h3>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      LLM Model
                    </label>
                    <input
                      type="text"
                      value={agentConfig?.llm?.model || ''}
                      onChange={(e) => handleAgentConfigChange(module as RomaModuleType, 'llm', {
                        ...agentConfig?.llm,
                        model: e.target.value
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                      placeholder="openrouter/google/gemini-2.5-flash"
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Temperature
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="1"
                        value={agentConfig?.llm?.temperature || 0.6}
                        onChange={(e) => handleAgentConfigChange(module as RomaModuleType, 'llm', {
                          ...agentConfig?.llm,
                          temperature: parseFloat(e.target.value)
                        })}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Max Tokens
                      </label>
                      <input
                        type="number"
                        min="64"
                        max="32768"
                        value={agentConfig?.llm?.max_tokens || 4096}
                        onChange={(e) => handleAgentConfigChange(module as RomaModuleType, 'llm', {
                          ...agentConfig?.llm,
                          max_tokens: parseInt(e.target.value)
                        })}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Prediction Strategy
                    </label>
                    <select
                      value={agentConfig?.prediction_strategy || 'chain_of_thought'}
                      onChange={(e) => handleAgentConfigChange(module as RomaModuleType, 'prediction_strategy', e.target.value as RomaPredictionStrategy)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    >
                      <option value="predict">Predict</option>
                      <option value="chain_of_thought">Chain of Thought</option>
                      <option value="react">ReAct</option>
                      <option value="code_act">Code Act</option>
                      <option value="best_of_n">Best of N</option>
                      <option value="refine">Refine</option>
                      <option value="parallel">Parallel</option>
                      <option value="majority">Majority</option>
                    </select>
                  </div>

                  <div>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={agentConfig?.llm?.cache || false}
                        onChange={(e) => handleAgentConfigChange(module as RomaModuleType, 'llm', {
                          ...agentConfig?.llm,
                          cache: e.target.checked
                        })}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable Cache</span>
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* MCP Servers Tab */}
        {activeTab === 'mcps' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Plus className="mr-2" /> Add New MCP Server
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Server Name
                  </label>
                  <input
                    type="text"
                    value={newMcpServer.server_name || ''}
                    onChange={(e) => setNewMcpServer({ ...newMcpServer, server_name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="coingecko"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Server Type
                  </label>
                  <select
                    value={newMcpServer.server_type || 'http'}
                    onChange={(e) => setNewMcpServer({ ...newMcpServer, server_type: e.target.value as 'http' | 'stdio' })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="http">HTTP/SSE Server</option>
                    <option value="stdio">Stdio Subprocess</option>
                  </select>
                </div>

                {newMcpServer.server_type === 'http' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Server URL
                    </label>
                    <input
                      type="text"
                      value={newMcpServer.url || ''}
                      onChange={(e) => setNewMcpServer({ ...newMcpServer, url: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                      placeholder="https://mcp.api.coingecko.com/sse"
                    />
                  </div>
                )}

                {newMcpServer.server_type === 'stdio' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Command
                    </label>
                    <input
                      type="text"
                      value={newMcpServer.command || ''}
                      onChange={(e) => setNewMcpServer({ ...newMcpServer, command: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                      placeholder="npx"
                    />
                  </div>
                )}

                <div className="flex items-center space-x-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={newMcpServer.enabled || true}
                      onChange={(e) => setNewMcpServer({ ...newMcpServer, enabled: e.target.checked })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enabled</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={newMcpServer.use_storage || false}
                      onChange={(e) => setNewMcpServer({ ...newMcpServer, use_storage: e.target.checked })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Use Storage</span>
                  </label>
                </div>

                {newMcpServer.use_storage && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Storage Threshold (KB)
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="1000"
                      value={newMcpServer.storage_threshold_kb || 100}
                      onChange={(e) => setNewMcpServer({ 
                        ...newMcpServer, 
                        storage_threshold_kb: parseInt(e.target.value) || 100 
                      })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                )}

                <button
                  onClick={handleAddMcpServer}
                  disabled={!newMcpServer.server_name || isSaving}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Plus className="mr-2" /> Add MCP Server
                </button>
              </div>
            </div>

            {/* Existing MCP Servers */}
            {config.mcpServers && config.mcpServers.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Configured MCP Servers</h3>
                {config.mcpServers.map((server, index) => (
                  <div key={server.server_name} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-3">
                      <h4 className="font-medium text-gray-900 dark:text-white flex items-center">
                        <Server className="mr-2 w-4 h-4" /> {server.server_name}
                      </h4>
                      <button
                        onClick={() => handleRemoveMcpServer(server.server_name)}
                        className="text-red-500 hover:text-red-700"
                        aria-label="Remove MCP server"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                          Server Type
                        </label>
                        <select
                          value={server.server_type}
                          onChange={(e) => handleMcpServerChange(index, 'server_type', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                        >
                          <option value="http">HTTP/SSE Server</option>
                          <option value="stdio">Stdio Subprocess</option>
                        </select>
                      </div>

                      {server.server_type === 'http' && server.url && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Server URL
                          </label>
                          <input
                            type="text"
                            value={server.url}
                            onChange={(e) => handleMcpServerChange(index, 'url', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                          />
                        </div>
                      )}

                      {server.server_type === 'stdio' && server.command && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Command
                          </label>
                          <input
                            type="text"
                            value={server.command}
                            onChange={(e) => handleMcpServerChange(index, 'command', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                          />
                        </div>
                      )}

                      <div className="flex items-center space-x-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={server.enabled || true}
                            onChange={(e) => handleMcpServerChange(index, 'enabled', e.target.checked)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enabled</span>
                        </label>

                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={server.use_storage || false}
                            onChange={(e) => handleMcpServerChange(index, 'use_storage', e.target.checked)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Use Storage</span>
                        </label>
                      </div>

                      {server.use_storage && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Storage Threshold (KB)
                          </label>
                          <input
                            type="number"
                            min="10"
                            max="1000"
                            value={server.storage_threshold_kb || 100}
                            onChange={(e) => handleMcpServerChange(index, 'storage_threshold_kb', parseInt(e.target.value) || 100)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Toolkits Tab */}
        {activeTab === 'toolkits' && (
          <div className="space-y-6">
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Plus className="mr-2" /> Add New Toolkit
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Toolkit Type
                  </label>
                  <select
                    value={newToolkit.class_name || ''}
                    onChange={(e) => setNewToolkit({ ...newToolkit, class_name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="">Select a toolkit</option>
                    {toolkitOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={newToolkit.enabled || true}
                      onChange={(e) => setNewToolkit({ ...newToolkit, enabled: e.target.checked })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enabled</span>
                  </label>
                </div>

                <button
                  onClick={handleAddToolkit}
                  disabled={!newToolkit.class_name || isSaving}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Plus className="mr-2" /> Add Toolkit
                </button>
              </div>
            </div>

            {/* Existing Toolkits */}
            {config.agents?.executor?.toolkits && config.agents.executor.toolkits.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Configured Toolkits</h3>
                {config.agents.executor.toolkits.map((toolkit, index) => {
                  const toolkitInfo = toolkitOptions.find(t => t.value === toolkit.class_name);
                  return (
                    <div key={toolkit.class_name} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-3">
                        <h4 className="font-medium text-gray-900 dark:text-white flex items-center">
                          {toolkitInfo?.icon || <Tool className="mr-2 w-4 h-4" />} {toolkitInfo?.label || toolkit.class_name}
                        </h4>
                        <button
                          onClick={() => handleRemoveToolkit(toolkit.class_name)}
                          className="text-red-500 hover:text-red-700"
                          aria-label="Remove toolkit"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                      <div className="space-y-3">
                        <div className="flex items-center">
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              checked={toolkit.enabled || true}
                              onChange={(e) => handleToolkitChange(index, 'enabled', e.target.checked)}
                              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                            />
                            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enabled</span>
                          </label>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* MDAP/MAKER Tab */}
      {activeTab === 'mdap_maker' && (
        <div className="space-y-6">
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <Brain className="mr-2" /> MDAP/MAKER Configuration
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              ROMA-MDAP-MAKER provides zero-error guarantees through hierarchical voting and adaptive decomposition.
            </p>

            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.mdapMaker?.enabled || false}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        enabled: e.target.checked
                      }
                    }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable MDAP/MAKER</span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.mdapMaker?.autoSelect || false}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        autoSelect: e.target.checked
                      }
                    }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Auto-Select for Critical Tasks</span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Depth
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={config.mdapMaker?.maxDepth || 2}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        maxDepth: parseInt(e.target.value) || 2
                      }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Max depth for ROMA decomposition (1-10)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    K-Ahead
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={config.mdapMaker?.kAhead || 3}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        kAhead: parseInt(e.target.value) || 3
                      }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    K-ahead threshold for MAKER voting (2-10)
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.mdapMaker?.enableRedFlagging || true}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        enableRedFlagging: e.target.checked
                      }
                    }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable Red-Flagging</span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.mdapMaker?.enableAdaptiveK || true}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        enableAdaptiveK: e.target.checked
                      }
                    }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable Adaptive K</span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    AI Provider
                  </label>
                  <select
                    value={config.mdapMaker?.provider || 'openai'}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        provider: e.target.value
                      }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="google">Google</option>
                    <option value="mistral">Mistral</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Model
                  </label>
                  <select
                    value={config.mdapMaker?.model || 'gpt-4o-mini'}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      mdapMaker: {
                        ...prev.mdapMaker,
                        model: e.target.value
                      }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="claude-3-opus">Claude 3 Opus</option>
                    <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Auto-Selection Keywords
                </label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {(config.mdapMaker?.autoSelectionKeywords || []).map((keyword, index) => (
                    <span key={index} className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded dark:bg-blue-900 dark:text-blue-100">
                      {keyword}
                    </span>
                  ))}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Tasks containing these keywords will automatically use MDAP/MAKER for zero-error execution
                </p>
              </div>

              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">Performance Recommendations</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-500 dark:text-gray-400">
                        <th className="py-1 px-2">Complexity</th>
                        <th className="py-1 px-2">Recommended Depth</th>
                        <th className="py-1 px-2">Recommended K</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t border-gray-100 dark:border-gray-800">
                        <td className="py-1 px-2">Low (1-3)</td>
                        <td className="py-1 px-2">1-2</td>
                        <td className="py-1 px-2">2</td>
                      </tr>
                      <tr className="border-t border-gray-100 dark:border-gray-800">
                        <td className="py-1 px-2">Medium (4-6)</td>
                        <td className="py-1 px-2">2-3</td>
                        <td className="py-1 px-2">3</td>
                      </tr>
                      <tr className="border-t border-gray-100 dark:border-gray-800">
                        <td className="py-1 px-2">High (7-8)</td>
                        <td className="py-1 px-2">3-4</td>
                        <td className="py-1 px-2">4</td>
                      </tr>
                      <tr className="border-t border-gray-100 dark:border-gray-800">
                        <td className="py-1 px-2">Very High (9-10)</td>
                        <td className="py-1 px-2">4-5</td>
                        <td className="py-1 px-2">5</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-1 flex items-center">
                  <Brain className="mr-2 w-4 h-4" /> About MDAP/MAKER
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  <strong>ROMA-MDAP-MAKER</strong> combines:
                </p>
                <ul className="text-sm text-blue-700 dark:text-blue-300 mt-1 space-y-1">
                  <li className="flex items-start">
                    <span className="mr-2">•</span>
                    <span><strong>ROMA</strong>: Recursive Open Meta-Agents for hierarchical decomposition</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2">•</span>
                    <span><strong>MDAP</strong>: Massively Decomposed Agentic Processes for millions of LLM steps</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2">•</span>
                    <span><strong>MAKER</strong>: Maximal Agentic decomposition with first-to-ahead-by-K error correction</span>
                  </li>
                </ul>
                <p className="text-sm text-blue-700 dark:text-blue-300 mt-2">
                  <strong>Zero-Error Guarantee</strong>: P(success) ≈ 99%+ with k=5
                </p>
              </div>
            </div>
          </div>
        )}

      {/* Action Buttons */}
      <div className="flex justify-end space-x-3 mt-8">
        <button
          onClick={onClose}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          Cancel
        </button>
        <button
          onClick={handleConfigChange}
          disabled={isSaving}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSaving ? (
            <>
              <span className="mr-2">Saving...</span>
            </>
          ) : (
            <>
              <Save className="mr-2" /> Save Configuration
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default RomaConfigPanel;