/**
 * Plugin Registration and Management System for LeanAide Autoformalization
 * 
 * This module provides the complete plugin registration and management system
 * for integrating LeanAide autoformalization capabilities into BubbleLab UI.
 */

import { LeanAidePluginInterface } from './plugins/LeanAidePlugin';

// Plugin registry
class PluginRegistry {
  private plugins: Map<string, LeanAidePluginInterface> = new Map();
  private activePlugins: Set<string> = new Set();

  /**
   * Register a plugin
   */
  register(plugin: LeanAidePluginInterface): boolean {
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
    const wasDeleted = this.plugins.delete(pluginId);
    this.activePlugins.delete(pluginId);
    console.log(`Plugin unregistered: ${pluginId}`);
    return wasDeleted;
  }

  /**
   * Activate a plugin
   */
  activate(pluginId: string): boolean {
    if (!this.plugins.has(pluginId)) {
      console.error(`Plugin ${pluginId} not found`);
      return false;
    }

    this.activePlugins.add(pluginId);
    console.log(`Plugin activated: ${pluginId}`);
    return true;
  }

  /**
   * Deactivate a plugin
   */
  deactivate(pluginId: string): boolean {
    const wasDeactivated = this.activePlugins.delete(pluginId);
    console.log(`Plugin deactivated: ${pluginId}`);
    return wasDeactivated;
  }

  /**
   * Get a plugin by ID
   */
  getPlugin(pluginId: string): LeanAidePluginInterface | undefined {
    return this.plugins.get(pluginId);
  }

  /**
   * Get all registered plugins
   */
  getAllPlugins(): LeanAidePluginInterface[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Get all active plugins
   */
  getActivePlugins(): LeanAidePluginInterface[] {
    return Array.from(this.activePlugins)
      .map(id => this.plugins.get(id))
      .filter(Boolean) as LeanAidePluginInterface[];
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
  getPluginsByCategory(category: string): LeanAidePluginInterface[] {
    return Array.from(this.plugins.values())
      .filter(plugin => plugin.category === category);
  }
}

// Global plugin registry instance
export const pluginRegistry = new PluginRegistry();

// Plugin manager component
export interface PluginManagerProps {
  className?: string;
}

export const PluginManager: React.FC<PluginManagerProps> = ({ className = '' }) => {
  const [plugins, setPlugins] = useState<LeanAidePluginInterface[]>([]);
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

  const togglePlugin = (pluginId: string) => {
    if (activePlugins.has(pluginId)) {
      pluginRegistry.deactivate(pluginId);
      setActivePlugins(prev => {
        const newSet = new Set(prev);
        newSet.delete(pluginId);
        return newSet;
      });
    } else {
      pluginRegistry.activate(pluginId);
      setActivePlugins(prev => new Set(prev).add(pluginId));
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

// Plugin lifecycle manager
export class PluginLifecycleManager {
  private pluginStates: Map<string, 'loading' | 'loaded' | 'active' | 'inactive' | 'error'> = new Map();

  /**
   * Initialize a plugin
   */
  async initializePlugin(pluginId: string): Promise<boolean> {
    try {
      this.pluginStates.set(pluginId, 'loading');
      
      // Get the plugin
      const plugin = pluginRegistry.getPlugin(pluginId);
      if (!plugin) {
        throw new Error(`Plugin ${pluginId} not found`);
      }

      // Perform any initialization logic here
      // For now, we'll just mark as loaded
      this.pluginStates.set(pluginId, 'loaded');
      
      console.log(`Plugin initialized: ${pluginId}`);
      return true;
    } catch (error) {
      console.error(`Error initializing plugin ${pluginId}:`, error);
      this.pluginStates.set(pluginId, 'error');
      return false;
    }
  }

  /**
   * Start a plugin
   */
  async startPlugin(pluginId: string): Promise<boolean> {
    try {
      const state = this.pluginStates.get(pluginId);
      if (!state || state === 'error') {
        await this.initializePlugin(pluginId);
      }

      if (this.pluginStates.get(pluginId) !== 'loaded') {
        throw new Error(`Plugin ${pluginId} is not in a loadable state`);
      }

      // Activate the plugin
      pluginRegistry.activate(pluginId);
      this.pluginStates.set(pluginId, 'active');
      
      console.log(`Plugin started: ${pluginId}`);
      return true;
    } catch (error) {
      console.error(`Error starting plugin ${pluginId}:`, error);
      return false;
    }
  }

  /**
   * Stop a plugin
   */
  async stopPlugin(pluginId: string): Promise<boolean> {
    try {
      pluginRegistry.deactivate(pluginId);
      this.pluginStates.set(pluginId, 'loaded'); // Go back to loaded state
      
      console.log(`Plugin stopped: ${pluginId}`);
      return true;
    } catch (error) {
      console.error(`Error stopping plugin ${pluginId}:`, error);
      return false;
    }
  }

  /**
   * Get plugin state
   */
  getPluginState(pluginId: string): 'loading' | 'loaded' | 'active' | 'inactive' | 'error' | undefined {
    return this.pluginStates.get(pluginId);
  }

  /**
   * Get all plugin states
   */
  getAllPluginStates(): Map<string, 'loading' | 'loaded' | 'active' | 'inactive' | 'error'> {
    return new Map(this.pluginStates);
  }
}

// Global lifecycle manager instance
export const pluginLifecycleManager = new PluginLifecycleManager();

// Plugin installer utility
export class PluginInstaller {
  /**
   * Install a plugin from a URL
   */
  static async installFromUrl(url: string): Promise<boolean> {
    try {
      // In a real implementation, this would download and install the plugin
      // For now, we'll just return true to simulate installation
      console.log(`Installing plugin from URL: ${url}`);
      
      // Simulate installation delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return true;
    } catch (error) {
      console.error(`Error installing plugin from URL ${url}:`, error);
      return false;
    }
  }

  /**
   * Install a plugin from a file
   */
  static async installFromFile(file: File): Promise<boolean> {
    try {
      console.log(`Installing plugin from file: ${file.name}`);
      
      // In a real implementation, this would read and install the plugin file
      // For now, we'll just return true to simulate installation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return true;
    } catch (error) {
      console.error(`Error installing plugin from file ${file.name}:`, error);
      return false;
    }
  }

  /**
   * Install a plugin from a package name
   */
  static async installFromPackage(packageName: string): Promise<boolean> {
    try {
      console.log(`Installing plugin from package: ${packageName}`);
      
      // In a real implementation, this would install from a package manager
      // For now, we'll just return true to simulate installation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return true;
    } catch (error) {
      console.error(`Error installing plugin from package ${packageName}:`, error);
      return false;
    }
  }
}

// Plugin configuration manager
export interface PluginConfiguration {
  [pluginId: string]: any;
}

export class PluginConfigurationManager {
  private configurations: PluginConfiguration = {};

  /**
   * Get plugin configuration
   */
  getConfiguration(pluginId: string): any {
    return this.configurations[pluginId];
  }

  /**
   * Set plugin configuration
   */
  setConfiguration(pluginId: string, config: any): void {
    this.configurations[pluginId] = config;
  }

  /**
   * Update plugin configuration partially
   */
  updateConfiguration(pluginId: string, updates: Partial<any>): void {
    const currentConfig = this.configurations[pluginId] || {};
    this.configurations[pluginId] = { ...currentConfig, ...updates };
  }

  /**
   * Remove plugin configuration
   */
  removeConfiguration(pluginId: string): void {
    delete this.configurations[pluginId];
  }

  /**
   * Get all configurations
   */
  getAllConfigurations(): PluginConfiguration {
    return { ...this.configurations };
  }

  /**
   * Load configurations from storage
   */
  async loadFromStorage(): Promise<void> {
    try {
      // In a real implementation, this would load from localStorage or a backend
      const stored = localStorage.getItem('leanaide_plugin_configs');
      if (stored) {
        this.configurations = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Error loading plugin configurations:', error);
    }
  }

  /**
   * Save configurations to storage
   */
  async saveToStorage(): Promise<void> {
    try {
      // In a real implementation, this would save to localStorage or a backend
      localStorage.setItem('leanaide_plugin_configs', JSON.stringify(this.configurations));
    } catch (error) {
      console.error('Error saving plugin configurations:', error);
    }
  }
}

// Global configuration manager instance
export const pluginConfigurationManager = new PluginConfigurationManager();

// Initialize the configuration manager
pluginConfigurationManager.loadFromStorage().catch(console.error);

// Export everything
export {
  PluginRegistry,
  PluginLifecycleManager,
  PluginInstaller,
  PluginConfigurationManager
};