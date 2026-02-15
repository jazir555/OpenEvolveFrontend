/**
 * LeanAide Plugin Interface and Registration System
 * 
 * Defines the interface for LeanAide plugins and provides the registration system
 * for integrating with BubbleLab UI.
 */

import React from 'react';
import { Brain, BarChart3, Shield, Database, Settings, Activity } from 'lucide-react';

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
  dependencies?: string[];
  author?: string;
  license?: string;
  homepage?: string;
  repository?: string;
  keywords?: string[];
  activationEvents?: string[];
  contributes?: {
    views?: any[];
    commands?: any[];
    configuration?: any;
  };
}

// Plugin configuration interface
export interface LeanAidePluginConfig {
  id: string;
  enabled: boolean;
  settings: Record<string, any>;
  metadata: {
    installedAt: Date;
    lastUpdated: Date;
    version: string;
  };
}

// Plugin lifecycle interface
export interface LeanAidePluginLifecycle {
  initialize?(): Promise<void>;
  activate?(): Promise<void>;
  deactivate?(): Promise<void>;
  dispose?(): void;
}

// Main plugin class
export abstract class LeanAidePlugin implements LeanAidePluginInterface, LeanAidePluginLifecycle {
  public readonly id: string;
  public readonly name: string;
  public readonly description: string;
  public readonly version: string;
  public readonly category: string;
  public readonly component: React.ComponentType<any>;
  public readonly icon: React.ReactNode;
  public settingsSchema?: any;
  public permissions?: string[];
  public dependencies?: string[];
  public author?: string;
  public license?: string;
  public homepage?: string;
  public repository?: string;
  public keywords?: string[];
  public activationEvents?: string[];
  public contributes?: {
    views?: any[];
    commands?: any[];
    configuration?: any;
  };

  protected config: LeanAidePluginConfig;

  constructor(config: Omit<LeanAidePluginInterface, 'component' | 'icon'> & {
    component: React.ComponentType<any>;
    icon: React.ReactNode;
  }) {
    this.id = config.id;
    this.name = config.name;
    this.description = config.description;
    this.version = config.version;
    this.category = config.category;
    this.component = config.component;
    this.icon = config.icon;
    this.settingsSchema = config.settingsSchema;
    this.permissions = config.permissions;
    this.dependencies = config.dependencies;
    this.author = config.author;
    this.license = config.license;
    this.homepage = config.homepage;
    this.repository = config.repository;
    this.keywords = config.keywords;
    this.activationEvents = config.activationEvents;
    this.contributes = config.contributes;

    this.config = {
      id: this.id,
      enabled: true,
      settings: {},
      metadata: {
        installedAt: new Date(),
        lastUpdated: new Date(),
        version: this.version
      }
    };
  }

  /**
   * Initialize the plugin
   */
  async initialize(): Promise<void> {
    console.log(`Initializing plugin: ${this.name} (${this.id})`);
    // Default implementation - can be overridden by subclasses
  }

  /**
   * Activate the plugin
   */
  async activate(): Promise<void> {
    console.log(`Activating plugin: ${this.name} (${this.id})`);
    this.config.enabled = true;
    // Default implementation - can be overridden by subclasses
  }

  /**
   * Deactivate the plugin
   */
  async deactivate(): Promise<void> {
    console.log(`Deactivating plugin: ${this.name} (${this.id})`);
    this.config.enabled = false;
    // Default implementation - can be overridden by subclasses
  }

  /**
   * Dispose the plugin
   */
  dispose(): void {
    console.log(`Disposing plugin: ${this.name} (${this.id})`);
    // Default implementation - can be overridden by subclasses
  }

  /**
   * Get plugin configuration
   */
  getConfig(): LeanAidePluginConfig {
    return { ...this.config };
  }

  /**
   * Update plugin configuration
   */
  updateConfig(settings: Record<string, any>): void {
    this.config.settings = { ...this.config.settings, ...settings };
    this.config.metadata.lastUpdated = new Date();
  }

  /**
   * Check if plugin is enabled
   */
  isEnabled(): boolean {
    return this.config.enabled;
  }

  /**
   * Get plugin settings
   */
  getSettings(): Record<string, any> {
    return { ...this.config.settings };
  }
}

// Plugin registry
class LeanAidePluginRegistry {
  private plugins: Map<string, LeanAidePlugin> = new Map();
  private activePlugins: Set<string> = new Set();

  /**
   * Register a plugin
   */
  register(plugin: LeanAidePlugin): boolean {
    if (this.plugins.has(plugin.id)) {
      console.warn(`Plugin with ID ${plugin.id} already registered`);
      return false;
    }

    this.plugins.set(plugin.id, plugin);
    console.log(`Plugin registered: ${plugin.name} (${plugin.id})`);
    return true;
  }

  /**
   * Unregister a plugin
   */
  unregister(pluginId: string): boolean {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      console.warn(`Plugin with ID ${pluginId} not found`);
      return false;
    }

    // Deactivate if active
    if (this.activePlugins.has(pluginId)) {
      plugin.deactivate().catch(console.error);
      this.activePlugins.delete(pluginId);
    }

    // Dispose the plugin
    plugin.dispose();

    this.plugins.delete(pluginId);
    console.log(`Plugin unregistered: ${pluginId}`);
    return true;
  }

  /**
   * Activate a plugin
   */
  async activate(pluginId: string): Promise<boolean> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      console.error(`Plugin ${pluginId} not found`);
      return false;
    }

    try {
      await plugin.activate();
      this.activePlugins.add(pluginId);
      console.log(`Plugin activated: ${pluginId}`);
      return true;
    } catch (error) {
      console.error(`Error activating plugin ${pluginId}:`, error);
      return false;
    }
  }

  /**
   * Deactivate a plugin
   */
  async deactivate(pluginId: string): Promise<boolean> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      console.error(`Plugin ${pluginId} not found`);
      return false;
    }

    try {
      await plugin.deactivate();
      this.activePlugins.delete(pluginId);
      console.log(`Plugin deactivated: ${pluginId}`);
      return true;
    } catch (error) {
      console.error(`Error deactivating plugin ${pluginId}:`, error);
      return false;
    }
  }

  /**
   * Get a plugin by ID
   */
  getPlugin(pluginId: string): LeanAidePlugin | undefined {
    return this.plugins.get(pluginId);
  }

  /**
   * Get all registered plugins
   */
  getAllPlugins(): LeanAidePlugin[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Get all active plugins
   */
  getActivePlugins(): LeanAidePlugin[] {
    return Array.from(this.activePlugins)
      .map(id => this.plugins.get(id))
      .filter(Boolean) as LeanAidePlugin[];
  }

  /**
   * Check if a plugin is active
   */
  isActive(pluginId: string): boolean {
    return this.activePlugins.has(pluginId);
  }

  /**
   * Get plugins by category
   */
  getPluginsByCategory(category: string): LeanAidePlugin[] {
    return Array.from(this.plugins.values())
      .filter(plugin => plugin.category === category);
  }

  /**
   * Get plugin count
   */
  getPluginCount(): number {
    return this.plugins.size;
  }

  /**
   * Get active plugin count
   */
  getActivePluginCount(): number {
    return this.activePlugins.size;
  }
}

// Global plugin registry instance
export const pluginRegistry = new LeanAidePluginRegistry();

// Plugin manager component
export interface PluginManagerProps {
  className?: string;
}

export const PluginManager: React.FC<PluginManagerProps> = ({ className = '' }) => {
  const [plugins, setPlugins] = useState<LeanAidePlugin[]>([]);
  const [activePlugins, setActivePlugins] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPlugins();
  }, []);

  const loadPlugins = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get all registered plugins
      const allPlugins = pluginRegistry.getAllPlugins();
      setPlugins(allPlugins);

      // Get active plugins
      const active = pluginRegistry.getActivePlugins();
      setActivePlugins(new Set(active.map(p => p.id)));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load plugins';
      setError(message);
      console.error('Error loading plugins:', err);
    } finally {
      setLoading(false);
    }
  };

  const togglePlugin = async (pluginId: string) => {
    if (activePlugins.has(pluginId)) {
      const success = await pluginRegistry.deactivate(pluginId);
      if (success) {
        setActivePlugins(prev => {
          const newSet = new Set(prev);
          newSet.delete(pluginId);
          return newSet;
        });
      }
    } else {
      const success = await pluginRegistry.activate(pluginId);
      if (success) {
        setActivePlugins(prev => new Set(prev).add(pluginId));
      }
    }
  };

  if (loading) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="text-gray-600">Loading plugins...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 text-red-800">
          <AlertTriangle className="w-5 h-5" />
          <h3 className="font-medium">Error Loading Plugins</h3>
        </div>
        <p className="text-red-600 mt-2">{error}</p>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-md border ${className}`}>
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Puzzle className="w-6 h-6 text-blue-600" />
          Plugin Manager
        </h2>
        <p className="text-gray-600 mt-1">Manage LeanAide autoformalization plugins</p>
      </div>

      <div className="p-6">
        {plugins.length === 0 ? (
          <div className="text-center py-12">
            <Puzzle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Plugins Found</h3>
            <p className="text-gray-500">Register plugins to get started with autoformalization capabilities.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {plugins.map(plugin => (
              <div 
                key={plugin.id} 
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border"
              >
                <div className="flex items-center gap-4">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    {plugin.icon}
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900">{plugin.name}</h3>
                    <p className="text-sm text-gray-500">{plugin.description}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {plugin.version}
                      </span>
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        {plugin.category}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    activePlugins.has(plugin.id) 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {activePlugins.has(plugin.id) ? 'Active' : 'Inactive'}
                  </span>
                  
                  <button
                    onClick={() => togglePlugin(plugin.id)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                      activePlugins.has(plugin.id) ? 'bg-blue-600' : 'bg-gray-200'
                    }`}
                    disabled={!plugin.isEnabled()}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      activePlugins.has(plugin.id) ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// LeanAide Autoformalization Plugin Implementation
export class LeanAideAutoformalizationPlugin extends LeanAidePlugin {
  constructor() {
    super({
      id: 'leanaide-autoformalization',
      name: 'LeanAide Autoformalization',
      description: 'Convert natural language mathematical statements to formal Lean 4 code',
      version: '1.0.0',
      category: 'formalization',
      component: () => import('./LeanAidePlugin').then(m => m.LeanAidePlugin),
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
      permissions: ['network', 'storage'],
      dependencies: ['leanaide-core'],
      author: 'OpenEvolve',
      license: 'MIT',
      homepage: 'https://github.com/openevolve/leanaide',
      repository: 'https://github.com/openevolve/leanaide/leanaide-bubblelab-plugin',
      keywords: ['lean', 'theorem', 'prover', 'formalization', 'autoformalization', 'mathematics'],
      activationEvents: ['onCommand:leanaide.open', 'onView:leanaide'],
      contributes: {
        views: [
          {
            id: 'leanaide-dashboard',
            name: 'LeanAide Dashboard',
            when: 'leanaide.enabled'
          }
        ],
        commands: [
          {
            command: 'leanaide.convert',
            title: 'Convert Natural Language to Lean',
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
            }
          }
        }
      }
    });
  }

  async initialize(): Promise<void> {
    console.log('Initializing LeanAide Autoformalization Plugin');
    // Initialize any required services
    await super.initialize();
  }

  async activate(): Promise<void> {
    console.log('Activating LeanAide Autoformalization Plugin');
    await super.activate();
    
    // Register commands and views
    // In a real implementation, this would register UI elements
  }

  async deactivate(): Promise<void> {
    console.log('Deactivating LeanAide Autoformalization Plugin');
    await super.deactivate();
    
    // Clean up resources
  }
}

// Register the default LeanAide plugin
const leanAidePlugin = new LeanAideAutoformalizationPlugin();
pluginRegistry.register(leanAidePlugin);

// Auto-activate the plugin
pluginRegistry.activate('leanaide-autoformalization').catch(console.error);

// Export the plugin interface and registry
export type { LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle };
export { LeanAidePlugin, pluginRegistry, PluginManager };
export default LeanAidePlugin;