import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// Plugin registry
class PluginRegistry {
    constructor() {
        this.plugins = new Map();
        this.activePlugins = new Set();
    }
    /**
     * Register a plugin
     */
    register(plugin) {
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
    unregister(pluginId) {
        const wasDeleted = this.plugins.delete(pluginId);
        this.activePlugins.delete(pluginId);
        console.log(`Plugin unregistered: ${pluginId}`);
        return wasDeleted;
    }
    /**
     * Activate a plugin
     */
    activate(pluginId) {
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
    deactivate(pluginId) {
        const wasDeactivated = this.activePlugins.delete(pluginId);
        console.log(`Plugin deactivated: ${pluginId}`);
        return wasDeactivated;
    }
    /**
     * Get a plugin by ID
     */
    getPlugin(pluginId) {
        return this.plugins.get(pluginId);
    }
    /**
     * Get all registered plugins
     */
    getAllPlugins() {
        return Array.from(this.plugins.values());
    }
    /**
     * Get all active plugins
     */
    getActivePlugins() {
        return Array.from(this.activePlugins)
            .map(id => this.plugins.get(id))
            .filter(Boolean);
    }
    /**
     * Check if a plugin is active
     */
    isActive(pluginId) {
        return this.activePlugins.has(pluginId);
    }
    /**
     * Get plugins by category
     */
    getPluginsByCategory(category) {
        return Array.from(this.plugins.values())
            .filter(plugin => plugin.category === category);
    }
}
// Global plugin registry instance
export const pluginRegistry = new PluginRegistry();
export const PluginManager = ({ className = '' }) => {
    const [plugins, setPlugins] = useState([]);
    const [activePlugins, setActivePlugins] = useState(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
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
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load plugins';
            setError(message);
            console.error('Error loading plugins:', err);
        }
        finally {
            setLoading(false);
        }
    };
    const togglePlugin = (pluginId) => {
        if (activePlugins.has(pluginId)) {
            pluginRegistry.deactivate(pluginId);
            setActivePlugins(prev => {
                const newSet = new Set(prev);
                newSet.delete(pluginId);
                return newSet;
            });
        }
        else {
            pluginRegistry.activate(pluginId);
            setActivePlugins(prev => new Set(prev).add(pluginId));
        }
    };
    if (loading) {
        return (_jsx("div", { className: `flex items-center justify-center h-64 ${className}`, children: _jsxs("div", { className: "flex flex-col items-center gap-4", children: [_jsx("div", { className: "animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" }), _jsx("p", { className: "text-gray-600", children: "Loading plugins..." })] }) }));
    }
    if (error) {
        return (_jsxs("div", { className: `bg-red-50 border border-red-200 rounded-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 text-red-800", children: [_jsx(AlertTriangle, { className: "w-5 h-5" }), _jsx("h3", { className: "font-medium", children: "Error Loading Plugins" })] }), _jsx("p", { className: "text-red-600 mt-2", children: error })] }));
    }
    return (_jsxs("div", { className: `bg-white rounded-lg shadow-md border ${className}`, children: [_jsxs("div", { className: "p-6 border-b border-gray-200", children: [_jsxs("h2", { className: "text-xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Puzzle, { className: "w-6 h-6 text-blue-600" }), "Plugin Manager"] }), _jsx("p", { className: "text-gray-600 mt-1", children: "Manage LeanAide autoformalization plugins" })] }), _jsx("div", { className: "p-6", children: plugins.length === 0 ? (_jsxs("div", { className: "text-center py-12", children: [_jsx(Puzzle, { className: "w-12 h-12 text-gray-400 mx-auto mb-4" }), _jsx("h3", { className: "text-lg font-medium text-gray-900 mb-2", children: "No Plugins Found" }), _jsx("p", { className: "text-gray-500", children: "Register plugins to get started with autoformalization capabilities." })] })) : (_jsx("div", { className: "space-y-4", children: plugins.map(plugin => (_jsxs("div", { className: "flex items-center justify-between p-4 bg-gray-50 rounded-lg border", children: [_jsxs("div", { className: "flex items-center gap-4", children: [_jsx("div", { className: "p-2 bg-blue-100 rounded-lg", children: plugin.icon }), _jsxs("div", { children: [_jsx("h3", { className: "font-medium text-gray-900", children: plugin.name }), _jsx("p", { className: "text-sm text-gray-500", children: plugin.description }), _jsxs("div", { className: "flex items-center gap-2 mt-1", children: [_jsx("span", { className: "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800", children: plugin.version }), _jsx("span", { className: "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800", children: plugin.category })] })] })] }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx("span", { className: `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${activePlugins.has(plugin.id)
                                            ? 'bg-green-100 text-green-800'
                                            : 'bg-red-100 text-red-800'}`, children: activePlugins.has(plugin.id) ? 'Active' : 'Inactive' }), _jsx("button", { onClick: () => togglePlugin(plugin.id), className: `relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${activePlugins.has(plugin.id) ? 'bg-blue-600' : 'bg-gray-200'}`, children: _jsx("span", { className: `inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${activePlugins.has(plugin.id) ? 'translate-x-6' : 'translate-x-1'}` }) })] })] }, plugin.id))) })) })] }));
};
// Plugin lifecycle manager
export class PluginLifecycleManager {
    constructor() {
        this.pluginStates = new Map();
    }
    /**
     * Initialize a plugin
     */
    async initializePlugin(pluginId) {
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
        }
        catch (error) {
            console.error(`Error initializing plugin ${pluginId}:`, error);
            this.pluginStates.set(pluginId, 'error');
            return false;
        }
    }
    /**
     * Start a plugin
     */
    async startPlugin(pluginId) {
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
        }
        catch (error) {
            console.error(`Error starting plugin ${pluginId}:`, error);
            return false;
        }
    }
    /**
     * Stop a plugin
     */
    async stopPlugin(pluginId) {
        try {
            pluginRegistry.deactivate(pluginId);
            this.pluginStates.set(pluginId, 'loaded'); // Go back to loaded state
            console.log(`Plugin stopped: ${pluginId}`);
            return true;
        }
        catch (error) {
            console.error(`Error stopping plugin ${pluginId}:`, error);
            return false;
        }
    }
    /**
     * Get plugin state
     */
    getPluginState(pluginId) {
        return this.pluginStates.get(pluginId);
    }
    /**
     * Get all plugin states
     */
    getAllPluginStates() {
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
    static async installFromUrl(url) {
        try {
            // In a real implementation, this would download and install the plugin
            // For now, we'll just return true to simulate installation
            console.log(`Installing plugin from URL: ${url}`);
            // Simulate installation delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            return true;
        }
        catch (error) {
            console.error(`Error installing plugin from URL ${url}:`, error);
            return false;
        }
    }
    /**
     * Install a plugin from a file
     */
    static async installFromFile(file) {
        try {
            console.log(`Installing plugin from file: ${file.name}`);
            // In a real implementation, this would read and install the plugin file
            // For now, we'll just return true to simulate installation
            await new Promise(resolve => setTimeout(resolve, 1000));
            return true;
        }
        catch (error) {
            console.error(`Error installing plugin from file ${file.name}:`, error);
            return false;
        }
    }
    /**
     * Install a plugin from a package name
     */
    static async installFromPackage(packageName) {
        try {
            console.log(`Installing plugin from package: ${packageName}`);
            // In a real implementation, this would install from a package manager
            // For now, we'll just return true to simulate installation
            await new Promise(resolve => setTimeout(resolve, 1000));
            return true;
        }
        catch (error) {
            console.error(`Error installing plugin from package ${packageName}:`, error);
            return false;
        }
    }
}
export class PluginConfigurationManager {
    constructor() {
        this.configurations = {};
    }
    /**
     * Get plugin configuration
     */
    getConfiguration(pluginId) {
        return this.configurations[pluginId];
    }
    /**
     * Set plugin configuration
     */
    setConfiguration(pluginId, config) {
        this.configurations[pluginId] = config;
    }
    /**
     * Update plugin configuration partially
     */
    updateConfiguration(pluginId, updates) {
        const currentConfig = this.configurations[pluginId] || {};
        this.configurations[pluginId] = { ...currentConfig, ...updates };
    }
    /**
     * Remove plugin configuration
     */
    removeConfiguration(pluginId) {
        delete this.configurations[pluginId];
    }
    /**
     * Get all configurations
     */
    getAllConfigurations() {
        return { ...this.configurations };
    }
    /**
     * Load configurations from storage
     */
    async loadFromStorage() {
        try {
            // In a real implementation, this would load from localStorage or a backend
            const stored = localStorage.getItem('leanaide_plugin_configs');
            if (stored) {
                this.configurations = JSON.parse(stored);
            }
        }
        catch (error) {
            console.error('Error loading plugin configurations:', error);
        }
    }
    /**
     * Save configurations to storage
     */
    async saveToStorage() {
        try {
            // In a real implementation, this would save to localStorage or a backend
            localStorage.setItem('leanaide_plugin_configs', JSON.stringify(this.configurations));
        }
        catch (error) {
            console.error('Error saving plugin configurations:', error);
        }
    }
}
// Global configuration manager instance
export const pluginConfigurationManager = new PluginConfigurationManager();
// Initialize the configuration manager
pluginConfigurationManager.loadFromStorage().catch(console.error);
// Export everything
export { PluginRegistry, PluginLifecycleManager, PluginInstaller, PluginConfigurationManager };
//# sourceMappingURL=PluginRegistry.js.map