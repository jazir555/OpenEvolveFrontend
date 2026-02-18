import React, { useEffect, useState } from 'react';
import { AlertTriangle, Puzzle } from 'lucide-react';

export interface LeanAidePluginInterface {
  id: string;
  name: string;
  description: string;
  version: string;
  category: string;
  component: React.ComponentType<any>;
  icon: React.ReactNode;
  settingsSchema?: Record<string, unknown>;
  permissions?: string[];
  dependencies?: string[];
  enabled?: boolean;
}

export interface LeanAidePluginConfig {
  id: string;
  enabled: boolean;
  settings: Record<string, unknown>;
  metadata: {
    installedAt: Date;
    lastUpdated: Date;
    version: string;
  };
}

export interface LeanAidePluginLifecycle {
  initialize?(): Promise<void>;
  activate?(): Promise<void>;
  deactivate?(): Promise<void>;
  dispose?(): void;
}

export class LeanAidePlugin implements LeanAidePluginInterface, LeanAidePluginLifecycle {
  public readonly id: string;
  public readonly name: string;
  public readonly description: string;
  public readonly version: string;
  public readonly category: string;
  public readonly component: React.ComponentType<any>;
  public readonly icon: React.ReactNode;
  public readonly settingsSchema?: Record<string, unknown>;
  public readonly permissions?: string[];
  public readonly dependencies?: string[];
  public enabled: boolean;

  protected config: LeanAidePluginConfig;

  constructor(plugin: LeanAidePluginInterface) {
    this.id = plugin.id;
    this.name = plugin.name;
    this.description = plugin.description;
    this.version = plugin.version;
    this.category = plugin.category;
    this.component = plugin.component;
    this.icon = plugin.icon;
    this.settingsSchema = plugin.settingsSchema;
    this.permissions = plugin.permissions;
    this.dependencies = plugin.dependencies;
    this.enabled = plugin.enabled ?? true;

    this.config = {
      id: this.id,
      enabled: this.enabled,
      settings: {},
      metadata: {
        installedAt: new Date(),
        lastUpdated: new Date(),
        version: this.version,
      },
    };
  }

  async initialize(): Promise<void> {
    return Promise.resolve();
  }

  async activate(): Promise<void> {
    this.enabled = true;
    this.config.enabled = true;
  }

  async deactivate(): Promise<void> {
    this.enabled = false;
    this.config.enabled = false;
  }

  dispose(): void {
    // no-op default
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getConfig(): LeanAidePluginConfig {
    return { ...this.config, metadata: { ...this.config.metadata } };
  }
}

type RegisteredPlugin = LeanAidePluginInterface & Partial<LeanAidePluginLifecycle>;

class LeanAidePluginRegistry {
  private readonly plugins = new Map<string, RegisteredPlugin>();
  private readonly activePlugins = new Set<string>();

  register(plugin: RegisteredPlugin): boolean {
    if (this.plugins.has(plugin.id)) {
      return false;
    }

    this.plugins.set(plugin.id, plugin);
    return true;
  }

  unregister(pluginId: string): boolean {
    this.activePlugins.delete(pluginId);
    const plugin = this.plugins.get(pluginId);
    plugin?.dispose?.();
    return this.plugins.delete(pluginId);
  }

  async activate(pluginId: string): Promise<boolean> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      return false;
    }

    await plugin.activate?.();
    this.activePlugins.add(pluginId);
    return true;
  }

  async deactivate(pluginId: string): Promise<boolean> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      return false;
    }

    await plugin.deactivate?.();
    this.activePlugins.delete(pluginId);
    return true;
  }

  getPlugin(pluginId: string): RegisteredPlugin | undefined {
    return this.plugins.get(pluginId);
  }

  getAllPlugins(): RegisteredPlugin[] {
    return Array.from(this.plugins.values());
  }

  getActivePlugins(): RegisteredPlugin[] {
    return Array.from(this.activePlugins)
      .map((id: string) => this.plugins.get(id))
      .filter((plugin): plugin is RegisteredPlugin => Boolean(plugin));
  }

  isActive(pluginId: string): boolean {
    return this.activePlugins.has(pluginId);
  }
}

export const pluginRegistry = new LeanAidePluginRegistry();

export interface PluginManagerProps {
  className?: string;
}

export const PluginManager: React.FC<PluginManagerProps> = ({ className = '' }) => {
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  const [activePluginIds, setActivePluginIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    setPlugins(pluginRegistry.getAllPlugins());
    setActivePluginIds(new Set(pluginRegistry.getActivePlugins().map((plugin: RegisteredPlugin) => plugin.id)));
  };

  useEffect(() => {
    try {
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load plugins');
    } finally {
      setLoading(false);
    }
  }, []);

  const togglePlugin = async (pluginId: string) => {
    if (activePluginIds.has(pluginId)) {
      await pluginRegistry.deactivate(pluginId);
    } else {
      await pluginRegistry.activate(pluginId);
    }

    refresh();
  };

  if (loading) {
    return <div className={className}>Loading plugins...</div>;
  }

  if (error) {
    return (
      <div className={`rounded-lg border border-red-200 bg-red-50 p-4 ${className}`}>
        <div className="flex items-center gap-2 text-sm text-red-700">
          <AlertTriangle className="h-4 w-4" />
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-lg border bg-white p-4 ${className}`}>
      <div className="mb-3 flex items-center gap-2 font-medium text-gray-900">
        <Puzzle className="h-4 w-4 text-blue-600" />
        LeanAide Plugins
      </div>

      {plugins.length === 0 ? (
        <div className="text-sm text-gray-500">No plugins registered.</div>
      ) : (
        <div className="space-y-2">
          {plugins.map((plugin: RegisteredPlugin) => (
            <div key={plugin.id} className="flex items-center justify-between rounded border p-3">
              <div>
                <div className="font-medium text-gray-900">{plugin.name}</div>
                <div className="text-xs text-gray-500">{plugin.description}</div>
              </div>

              <button
                onClick={() => {
                  void togglePlugin(plugin.id);
                }}
                className={`rounded px-2 py-1 text-xs ${
                  activePluginIds.has(plugin.id)
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                {activePluginIds.has(plugin.id) ? 'Active' : 'Inactive'}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default LeanAidePlugin;
