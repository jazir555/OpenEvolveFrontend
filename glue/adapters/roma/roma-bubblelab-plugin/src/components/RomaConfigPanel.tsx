import React, { useMemo, useState } from 'react';
import { Bot, Plus, Save, Server, Settings, Tool, Trash2, X } from 'lucide-react';
import { toast } from 'react-toastify';

import type {
  RomaConfigPanelProps,
  RomaMcpServerConfig,
  RomaPluginConfig,
  RomaToolkitConfig,
} from '../types/plugin-types';

const RomaConfigPanel: React.FC<RomaConfigPanelProps> = ({ plugin, onConfigChange, onClose }) => {
  const [config, setConfig] = useState<RomaPluginConfig>(plugin.getState());
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [newMcpName, setNewMcpName] = useState<string>('');
  const [newToolkitName, setNewToolkitName] = useState<string>('');

  const mcpServers = useMemo(() => config.mcpServers ?? [], [config.mcpServers]);
  const toolkits = useMemo(() => config.agents?.executor?.toolkits ?? [], [config.agents?.executor?.toolkits]);

  const updateConfig = (updates: Partial<RomaPluginConfig>) => {
    setConfig((previous) => ({ ...previous, ...updates }));
  };

  const saveConfig = async () => {
    try {
      setIsSaving(true);
      await plugin.updateConfig(config);
      onConfigChange?.(config);
      toast.success('ROMA configuration saved');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to save ROMA configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const addMcpServer = async () => {
    if (!newMcpName.trim()) {
      return;
    }

    const newServer: RomaMcpServerConfig = {
      server_name: newMcpName.trim(),
      server_type: 'http',
      enabled: true,
    };

    try {
      await plugin.addMcpServer(newServer);
      setConfig(plugin.getState());
      setNewMcpName('');
      toast.success(`Added MCP server ${newServer.server_name}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to add MCP server');
    }
  };

  const removeMcpServer = async (serverName: string) => {
    try {
      await plugin.removeMcpServer(serverName);
      setConfig(plugin.getState());
      toast.success(`Removed MCP server ${serverName}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to remove MCP server');
    }
  };

  const addToolkit = async () => {
    if (!newToolkitName.trim()) {
      return;
    }

    const toolkit: RomaToolkitConfig = {
      class_name: newToolkitName.trim(),
      enabled: true,
    };

    try {
      await plugin.addToolkit(toolkit);
      setConfig(plugin.getState());
      setNewToolkitName('');
      toast.success(`Added toolkit ${toolkit.class_name}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to add toolkit');
    }
  };

  const removeToolkit = async (className: string) => {
    try {
      await plugin.removeToolkit(className);
      setConfig(plugin.getState());
      toast.success(`Removed toolkit ${className}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to remove toolkit');
    }
  };

  return (
    <div className="rounded-lg border bg-white p-6 shadow-sm dark:bg-gray-800">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-xl font-semibold text-gray-900 dark:text-white">
          <Settings className="h-5 w-5" />
          ROMA Configuration
        </h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-700 dark:hover:text-gray-200"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <Settings className="h-4 w-4" />
            General
          </h3>
          <div className="space-y-3">
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Server URL
              <input
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={config.serverUrl ?? ''}
                onChange={(event: any) => updateConfig({ serverUrl: event.target.value })}
              />
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              API Key
              <input
                type="password"
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={config.apiKey ?? ''}
                onChange={(event: any) => updateConfig({ apiKey: event.target.value })}
              />
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
              <input
                type="checkbox"
                checked={config.enableObservability ?? true}
                onChange={(event: any) => updateConfig({ enableObservability: event.target.checked })}
              />
              Enable Observability
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
              <input
                type="checkbox"
                checked={config.enableStorage ?? true}
                onChange={(event: any) => updateConfig({ enableStorage: event.target.checked })}
              />
              Enable Storage
            </label>
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <Bot className="h-4 w-4" />
            Execution
          </h3>
          <div className="space-y-3">
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Default Profile
              <input
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={config.defaultProfile ?? 'default'}
                onChange={(event: any) => updateConfig({ defaultProfile: event.target.value })}
              />
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Max Depth
              <input
                type="number"
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={config.maxDepth ?? 3}
                onChange={(event: any) => updateConfig({ maxDepth: Number(event.target.value) || 3 })}
              />
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Timeout (ms)
              <input
                type="number"
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={config.timeout ?? 30000}
                onChange={(event: any) => updateConfig({ timeout: Number(event.target.value) || 30000 })}
              />
            </label>
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <Server className="h-4 w-4" />
            MCP Servers
          </h3>
          <div className="mb-3 flex gap-2">
            <input
              className="flex-1 rounded border px-3 py-2 text-sm"
              placeholder="Server name"
              value={newMcpName}
              onChange={(event: any) => setNewMcpName(event.target.value)}
            />
            <button
              type="button"
              onClick={() => {
                void addMcpServer();
              }}
              className="rounded bg-blue-600 px-2 py-2 text-white"
              aria-label="Add MCP server"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-2">
            {mcpServers.map((server: RomaMcpServerConfig) => (
              <div key={server.server_name} className="flex items-center justify-between rounded border p-2 text-sm">
                <span>{server.server_name}</span>
                <button
                  type="button"
                  onClick={() => {
                    void removeMcpServer(server.server_name);
                  }}
                  className="text-red-600"
                  aria-label={`Remove MCP server ${server.server_name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <Tool className="h-4 w-4" />
            Toolkits
          </h3>
          <div className="mb-3 flex gap-2">
            <input
              className="flex-1 rounded border px-3 py-2 text-sm"
              placeholder="Toolkit class name"
              value={newToolkitName}
              onChange={(event: any) => setNewToolkitName(event.target.value)}
            />
            <button
              type="button"
              onClick={() => {
                void addToolkit();
              }}
              className="rounded bg-blue-600 px-2 py-2 text-white"
              aria-label="Add toolkit"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-2">
            {toolkits.map((toolkit: RomaToolkitConfig) => (
              <div key={toolkit.class_name} className="flex items-center justify-between rounded border p-2 text-sm">
                <span>{toolkit.class_name}</span>
                <button
                  type="button"
                  onClick={() => {
                    void removeToolkit(toolkit.class_name);
                  }}
                  className="text-red-600"
                  aria-label={`Remove toolkit ${toolkit.class_name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="mt-6 flex justify-end">
        <button
          type="button"
          onClick={() => {
            void saveConfig();
          }}
          disabled={isSaving}
          className="inline-flex items-center gap-2 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
        >
          <Save className="h-4 w-4" />
          {isSaving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>
    </div>
  );
};

export default RomaConfigPanel;
