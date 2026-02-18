import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useEffect, useState } from 'react';
import { AlertTriangle, Puzzle } from 'lucide-react';
export class LeanAidePlugin {
    constructor(plugin) {
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
    async initialize() {
        return Promise.resolve();
    }
    async activate() {
        this.enabled = true;
        this.config.enabled = true;
    }
    async deactivate() {
        this.enabled = false;
        this.config.enabled = false;
    }
    dispose() {
        // no-op default
    }
    isEnabled() {
        return this.enabled;
    }
    getConfig() {
        return { ...this.config, metadata: { ...this.config.metadata } };
    }
}
class LeanAidePluginRegistry {
    constructor() {
        this.plugins = new Map();
        this.activePlugins = new Set();
    }
    register(plugin) {
        if (this.plugins.has(plugin.id)) {
            return false;
        }
        this.plugins.set(plugin.id, plugin);
        return true;
    }
    unregister(pluginId) {
        this.activePlugins.delete(pluginId);
        const plugin = this.plugins.get(pluginId);
        plugin?.dispose?.();
        return this.plugins.delete(pluginId);
    }
    async activate(pluginId) {
        const plugin = this.plugins.get(pluginId);
        if (!plugin) {
            return false;
        }
        await plugin.activate?.();
        this.activePlugins.add(pluginId);
        return true;
    }
    async deactivate(pluginId) {
        const plugin = this.plugins.get(pluginId);
        if (!plugin) {
            return false;
        }
        await plugin.deactivate?.();
        this.activePlugins.delete(pluginId);
        return true;
    }
    getPlugin(pluginId) {
        return this.plugins.get(pluginId);
    }
    getAllPlugins() {
        return Array.from(this.plugins.values());
    }
    getActivePlugins() {
        return Array.from(this.activePlugins)
            .map((id) => this.plugins.get(id))
            .filter((plugin) => Boolean(plugin));
    }
    isActive(pluginId) {
        return this.activePlugins.has(pluginId);
    }
}
export const pluginRegistry = new LeanAidePluginRegistry();
export const PluginManager = ({ className = '' }) => {
    const [plugins, setPlugins] = useState([]);
    const [activePluginIds, setActivePluginIds] = useState(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const refresh = () => {
        setPlugins(pluginRegistry.getAllPlugins());
        setActivePluginIds(new Set(pluginRegistry.getActivePlugins().map((plugin) => plugin.id)));
    };
    useEffect(() => {
        try {
            refresh();
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load plugins');
        }
        finally {
            setLoading(false);
        }
    }, []);
    const togglePlugin = async (pluginId) => {
        if (activePluginIds.has(pluginId)) {
            await pluginRegistry.deactivate(pluginId);
        }
        else {
            await pluginRegistry.activate(pluginId);
        }
        refresh();
    };
    if (loading) {
        return _jsx("div", { className: className, children: "Loading plugins..." });
    }
    if (error) {
        return (_jsx("div", { className: `rounded-lg border border-red-200 bg-red-50 p-4 ${className}`, children: _jsxs("div", { className: "flex items-center gap-2 text-sm text-red-700", children: [_jsx(AlertTriangle, { className: "h-4 w-4" }), error] }) }));
    }
    return (_jsxs("div", { className: `rounded-lg border bg-white p-4 ${className}`, children: [_jsxs("div", { className: "mb-3 flex items-center gap-2 font-medium text-gray-900", children: [_jsx(Puzzle, { className: "h-4 w-4 text-blue-600" }), "LeanAide Plugins"] }), plugins.length === 0 ? (_jsx("div", { className: "text-sm text-gray-500", children: "No plugins registered." })) : (_jsx("div", { className: "space-y-2", children: plugins.map((plugin) => (_jsxs("div", { className: "flex items-center justify-between rounded border p-3", children: [_jsxs("div", { children: [_jsx("div", { className: "font-medium text-gray-900", children: plugin.name }), _jsx("div", { className: "text-xs text-gray-500", children: plugin.description })] }), _jsx("button", { onClick: () => {
                                void togglePlugin(plugin.id);
                            }, className: `rounded px-2 py-1 text-xs ${activePluginIds.has(plugin.id)
                                ? 'bg-green-100 text-green-700'
                                : 'bg-gray-100 text-gray-700'}`, children: activePluginIds.has(plugin.id) ? 'Active' : 'Inactive' })] }, plugin.id))) }))] }));
};
export default LeanAidePlugin;
//# sourceMappingURL=PluginInterface.js.map