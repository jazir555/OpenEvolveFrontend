"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.PluginManager = exports.pluginRegistry = exports.LeanAidePlugin = void 0;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
class LeanAidePlugin {
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
exports.LeanAidePlugin = LeanAidePlugin;
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
exports.pluginRegistry = new LeanAidePluginRegistry();
const PluginManager = ({ className = '' }) => {
    const [plugins, setPlugins] = (0, react_1.useState)([]);
    const [activePluginIds, setActivePluginIds] = (0, react_1.useState)(new Set());
    const [loading, setLoading] = (0, react_1.useState)(true);
    const [error, setError] = (0, react_1.useState)(null);
    const refresh = () => {
        setPlugins(exports.pluginRegistry.getAllPlugins());
        setActivePluginIds(new Set(exports.pluginRegistry.getActivePlugins().map((plugin) => plugin.id)));
    };
    (0, react_1.useEffect)(() => {
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
            await exports.pluginRegistry.deactivate(pluginId);
        }
        else {
            await exports.pluginRegistry.activate(pluginId);
        }
        refresh();
    };
    if (loading) {
        return <div className={className}>Loading plugins...</div>;
    }
    if (error) {
        return (<div className={`rounded-lg border border-red-200 bg-red-50 p-4 ${className}`}>
        <div className="flex items-center gap-2 text-sm text-red-700">
          <lucide_react_1.AlertTriangle className="h-4 w-4"/>
          {error}
        </div>
      </div>);
    }
    return (<div className={`rounded-lg border bg-white p-4 ${className}`}>
      <div className="mb-3 flex items-center gap-2 font-medium text-gray-900">
        <lucide_react_1.Puzzle className="h-4 w-4 text-blue-600"/>
        LeanAide Plugins
      </div>

      {plugins.length === 0 ? (<div className="text-sm text-gray-500">No plugins registered.</div>) : (<div className="space-y-2">
          {plugins.map((plugin) => (<div key={plugin.id} className="flex items-center justify-between rounded border p-3">
              <div>
                <div className="font-medium text-gray-900">{plugin.name}</div>
                <div className="text-xs text-gray-500">{plugin.description}</div>
              </div>

              <button onClick={() => {
                    void togglePlugin(plugin.id);
                }} className={`rounded px-2 py-1 text-xs ${activePluginIds.has(plugin.id)
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-700'}`}>
                {activePluginIds.has(plugin.id) ? 'Active' : 'Inactive'}
              </button>
            </div>))}
        </div>)}
    </div>);
};
exports.PluginManager = PluginManager;
exports.default = LeanAidePlugin;
//# sourceMappingURL=PluginInterface.js.map