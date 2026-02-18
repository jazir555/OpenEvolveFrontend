import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Complete Plugin System for LeanAide Autoformalization in BubbleLab
 *
 * This module provides the complete plugin architecture for integrating
 * the LeanAide autoformalization system with predictive analytics into BubbleLab UI.
 */
import React, { useState, useEffect, createContext, useContext } from 'react';
import { Brain, Puzzle } from 'lucide-react';
import { toast } from 'react-toastify';
// Plugin base class
export class LeanAidePlugin {
    constructor(config) {
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
    async initialize() {
        console.log(`Initializing plugin: ${this.name} (${this.id})`);
    }
    async activate() {
        console.log(`Activating plugin: ${this.name} (${this.id})`);
        this.config.enabled = true;
    }
    async deactivate() {
        console.log(`Deactivating plugin: ${this.name} (${this.id})`);
        this.config.enabled = false;
    }
    dispose() {
        console.log(`Disposing plugin: ${this.name} (${this.id})`);
    }
    getConfig() {
        return { ...this.config };
    }
    updateConfig(settings) {
        this.config.settings = { ...this.config.settings, ...settings };
        this.config.metadata.lastUpdated = new Date();
    }
    isEnabled() {
        return this.config.enabled;
    }
    getSettings() {
        return { ...this.config.settings };
    }
}
// Plugin Registry
class LeanAidePluginRegistry {
    constructor() {
        this.plugins = new Map();
        this.activePlugins = new Set();
    }
    register(plugin) {
        if (this.plugins.has(plugin.id)) {
            console.warn(`Plugin with ID ${plugin.id} already registered`);
            return false;
        }
        this.plugins.set(plugin.id, plugin);
        console.log(`Plugin registered: ${plugin.name} (${plugin.id})`);
        return true;
    }
    unregister(pluginId) {
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
    async activate(pluginId) {
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
        }
        catch (error) {
            console.error(`Error activating plugin ${pluginId}:`, error);
            return false;
        }
    }
    async deactivate(pluginId) {
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
        }
        catch (error) {
            console.error(`Error deactivating plugin ${pluginId}:`, error);
            return false;
        }
    }
    getPlugin(pluginId) {
        return this.plugins.get(pluginId);
    }
    getAllPlugins() {
        return Array.from(this.plugins.values());
    }
    getActivePlugins() {
        return Array.from(this.activePlugins)
            .map(id => this.plugins.get(id))
            .filter(Boolean);
    }
    isActive(pluginId) {
        return this.activePlugins.has(pluginId);
    }
    getPluginsByCategory(category) {
        return Array.from(this.plugins.values())
            .filter(plugin => plugin.category === category);
    }
    getPluginCount() {
        return this.plugins.size;
    }
    getActivePluginCount() {
        return this.activePlugins.size;
    }
}
// Global plugin registry instance
export const pluginRegistry = new LeanAidePluginRegistry();
const PluginManagerContext = createContext(undefined);
export const PluginManagerProvider = ({ children }) => {
    const [plugins, setPlugins] = useState([]);
    const [activePlugins, setActivePlugins] = useState(new Set());
    const [refreshTrigger, setRefreshTrigger] = useState(0);
    useEffect(() => {
        loadPlugins();
    }, [refreshTrigger]);
    const loadPlugins = () => {
        const allPlugins = pluginRegistry.getAllPlugins();
        setPlugins(allPlugins);
        const active = new Set(pluginRegistry.getActivePlugins().map(p => p.id));
        setActivePlugins(active);
    };
    const registerPlugin = (plugin) => {
        const success = pluginRegistry.register(plugin);
        if (success) {
            setRefreshTrigger(prev => prev + 1);
        }
        return success;
    };
    const unregisterPlugin = (pluginId) => {
        const success = pluginRegistry.unregister(pluginId);
        if (success) {
            setRefreshTrigger(prev => prev + 1);
        }
        return success;
    };
    const activatePlugin = async (pluginId) => {
        const success = await pluginRegistry.activate(pluginId);
        if (success) {
            setActivePlugins(prev => new Set(prev).add(pluginId));
        }
        return success;
    };
    const deactivatePlugin = async (pluginId) => {
        const success = await pluginRegistry.deactivate(pluginId);
        if (success) {
            setActivePlugins(prev => {
                const newSet = new Set(prev);
                newSet.delete(pluginId);
                return newSet;
            });
        }
        return success;
    };
    const getPlugin = (pluginId) => {
        return pluginRegistry.getPlugin(pluginId);
    };
    const refreshPlugins = () => {
        setRefreshTrigger(prev => prev + 1);
    };
    const value = {
        plugins,
        activePlugins,
        registerPlugin,
        unregisterPlugin,
        activatePlugin,
        deactivatePlugin,
        getPlugin,
        refreshPlugins
    };
    return (_jsx(PluginManagerContext.Provider, { value: value, children: children }));
};
export const usePluginManager = () => {
    const context = useContext(PluginManagerContext);
    if (!context) {
        throw new Error('usePluginManager must be used within a PluginManagerProvider');
    }
    return context;
};
// Plugin Manager Component
export const PluginManager = ({ className = '' }) => {
    const { plugins, activePlugins, activatePlugin, deactivatePlugin, refreshPlugins } = usePluginManager();
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('all');
    const filteredPlugins = plugins.filter(plugin => {
        const matchesSearch = plugin.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            plugin.description.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesCategory = selectedCategory === 'all' || plugin.category === selectedCategory;
        return matchesSearch && matchesCategory;
    });
    const categories = ['all', ...Array.from(new Set(plugins.map(p => p.category)))];
    return (_jsxs("div", { className: `bg-white rounded-lg shadow-md border ${className}`, children: [_jsxs("div", { className: "p-6 border-b border-gray-200", children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsxs("h2", { className: "text-xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Puzzle, { className: "w-6 h-6 text-blue-600" }), "Plugin Manager"] }), _jsxs("button", { onClick: refreshPlugins, className: "flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors", children: [_jsx(RefreshCw, { className: "w-4 h-4" }), "Refresh"] })] }), _jsxs("div", { className: "flex flex-col sm:flex-row gap-4", children: [_jsxs("div", { className: "relative flex-1", children: [_jsx(Search, { className: "absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" }), _jsx("input", { type: "text", placeholder: "Search plugins...", value: searchTerm, onChange: (e) => setSearchTerm(e.target.value), className: "w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }), _jsx("select", { value: selectedCategory, onChange: (e) => setSelectedCategory(e.target.value), className: "px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500", children: categories.map(category => (_jsx("option", { value: category, children: category.charAt(0).toUpperCase() + category.slice(1) }, category))) })] })] }), _jsx("div", { className: "p-6", children: filteredPlugins.length === 0 ? (_jsxs("div", { className: "text-center py-12", children: [_jsx(Puzzle, { className: "w-12 h-12 text-gray-400 mx-auto mb-4" }), _jsx("h3", { className: "text-lg font-medium text-gray-900 mb-2", children: "No Plugins Found" }), _jsx("p", { className: "text-gray-500", children: "Try adjusting your search or filter criteria." })] })) : (_jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4", children: filteredPlugins.map(plugin => (_jsxs("div", { className: "border rounded-lg p-4 hover:shadow-md transition-shadow", children: [_jsxs("div", { className: "flex items-start gap-3", children: [_jsx("div", { className: "p-2 bg-blue-100 rounded-lg", children: plugin.icon }), _jsxs("div", { className: "flex-1 min-w-0", children: [_jsx("h3", { className: "font-medium text-gray-900 truncate", children: plugin.name }), _jsx("p", { className: "text-sm text-gray-500 mt-1 line-clamp-2", children: plugin.description }), _jsxs("div", { className: "flex items-center gap-2 mt-2", children: [_jsxs("span", { className: "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800", children: ["v", plugin.version] }), _jsx("span", { className: "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800", children: plugin.category })] })] })] }), _jsxs("div", { className: "flex items-center justify-between mt-4", children: [_jsx("span", { className: `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${activePlugins.has(plugin.id)
                                            ? 'bg-green-100 text-green-800'
                                            : 'bg-red-100 text-red-800'}`, children: activePlugins.has(plugin.id) ? 'Active' : 'Inactive' }), _jsx("button", { onClick: async () => {
                                            if (activePlugins.has(plugin.id)) {
                                                await deactivatePlugin(plugin.id);
                                                toast.success(`Plugin ${plugin.name} deactivated`);
                                            }
                                            else {
                                                await activatePlugin(plugin.id);
                                                toast.success(`Plugin ${plugin.name} activated`);
                                            }
                                        }, className: `relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${activePlugins.has(plugin.id) ? 'bg-blue-600' : 'bg-gray-200'}`, children: _jsx("span", { className: `inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${activePlugins.has(plugin.id) ? 'translate-x-6' : 'translate-x-1'}` }) })] })] }, plugin.id))) })) })] }));
};
// Main BubbleLab Integration Plugin
export class BubbleLabLeanAideIntegrationPlugin extends LeanAidePlugin {
    constructor() {
        super({
            id: 'bubblelab-leanaide-integration',
            name: 'BubbleLab LeanAide Integration',
            description: 'Complete integration of LeanAide autoformalization with predictive analytics into BubbleLab UI',
            version: '1.0.0',
            category: 'integration',
            component: () => import('./BubbleLabIntegration').then(m => m.BubbleLabLeanAideIntegration),
            icon: _jsx(Brain, { className: "w-5 h-5" }),
            settingsSchema: {
                type: 'object',
                properties: {
                    serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' },
                    apiKey: { type: 'string', default: '' },
                    enableAnalytics: { type: 'boolean', default: true },
                    enablePredictiveFlagging: { type: 'boolean', default: true },
                    enableKnowledgeGraph: { type: 'boolean', default: true },
                    analyticsRefreshInterval: { type: 'number', default: 5000 },
                    maxConcurrentRequests: { type: 'number', default: 5 },
                    cacheEnabled: { type: 'boolean', default: true },
                    cacheTTL: { type: 'number', default: 3600 }
                }
            },
            permissions: ['network', 'storage'],
            dependencies: ['leanaide-core', 'bubblelab-core'],
            author: 'OpenEvolve',
            license: 'MIT',
            homepage: 'https://github.com/openevolve/leanaide',
            repository: 'https://github.com/openevolve/leanaide/leanaide-bubblelab-plugin',
            keywords: ['lean', 'theorem', 'prover', 'formalization', 'autoformalization', 'bubblelab', 'integration', 'analytics', 'predictive'],
            activationEvents: ['onView:leanaide-dashboard', 'onCommand:leanaide.open'],
            contributes: {
                views: [
                    {
                        id: 'leanaide-dashboard',
                        name: 'LeanAide Dashboard',
                        when: 'leanaide.enabled'
                    },
                    {
                        id: 'leanaide-verification',
                        name: 'Autoformalization',
                        when: 'leanaide.enabled'
                    },
                    {
                        id: 'leanaide-knowledge',
                        name: 'Knowledge Graph',
                        when: 'leanaide.knowledgeGraphEnabled'
                    }
                ],
                commands: [
                    {
                        command: 'leanaide.convert',
                        title: 'Convert Natural Language to Lean',
                        category: 'LeanAide'
                    },
                    {
                        command: 'leanaide.verify',
                        title: 'Verify Lean Code',
                        category: 'LeanAide'
                    },
                    {
                        command: 'leanaide.searchKnowledge',
                        title: 'Search Mathematical Knowledge',
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
                        },
                        'leanaide.enableAnalytics': {
                            type: 'boolean',
                            default: true,
                            description: 'Enable real-time analytics'
                        },
                        'leanaide.enablePredictiveFlagging': {
                            type: 'boolean',
                            default: true,
                            description: 'Enable predictive quality control'
                        }
                    }
                }
            }
        });
    }
    async initialize() {
        await super.initialize();
        console.log('BubbleLab LeanAide Integration Plugin initialized');
    }
    async activate() {
        await super.activate();
        console.log('BubbleLab LeanAide Integration Plugin activated');
    }
    async deactivate() {
        await super.deactivate();
        console.log('BubbleLab LeanAide Integration Plugin deactivated');
    }
}
// Auto-register the main integration plugin
const bubbleLabIntegrationPlugin = new BubbleLabLeanAideIntegrationPlugin();
pluginRegistry.register(bubbleLabIntegrationPlugin);
export { LeanAidePlugin, pluginRegistry, PluginManager, PluginManagerProvider, usePluginManager };
export default BubbleLabLeanAideIntegrationPlugin;
//# sourceMappingURL=PluginSystem.js.map