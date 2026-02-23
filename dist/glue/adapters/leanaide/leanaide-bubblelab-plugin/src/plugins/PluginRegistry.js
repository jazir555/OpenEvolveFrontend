"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PluginManager = exports.pluginRegistry = exports.pluginConfigurationManager = exports.pluginLifecycleManager = exports.PluginConfigurationManager = exports.PluginInstaller = exports.PluginLifecycleManager = void 0;
const PluginInterface_1 = require("../PluginInterface");
Object.defineProperty(exports, "pluginRegistry", { enumerable: true, get: function () { return PluginInterface_1.pluginRegistry; } });
Object.defineProperty(exports, "PluginManager", { enumerable: true, get: function () { return PluginInterface_1.PluginManager; } });
class PluginLifecycleManager {
    constructor() {
        this.states = new Map();
    }
    async startPlugin(pluginId) {
        const success = await PluginInterface_1.pluginRegistry.activate(pluginId);
        this.states.set(pluginId, success ? 'active' : 'error');
        return success;
    }
    async stopPlugin(pluginId) {
        const success = await PluginInterface_1.pluginRegistry.deactivate(pluginId);
        this.states.set(pluginId, success ? 'inactive' : 'error');
        return success;
    }
    getPluginState(pluginId) {
        return this.states.get(pluginId);
    }
}
exports.PluginLifecycleManager = PluginLifecycleManager;
class PluginInstaller {
    static async installFromUrl(_url) {
        return true;
    }
    static async installFromPackage(_name) {
        return true;
    }
}
exports.PluginInstaller = PluginInstaller;
class PluginConfigurationManager {
    constructor() {
        this.configurations = {};
    }
    getConfiguration(pluginId) {
        return this.configurations[pluginId] ?? {};
    }
    setConfiguration(pluginId, config) {
        this.configurations[pluginId] = { ...config };
    }
    getAllConfigurations() {
        return { ...this.configurations };
    }
}
exports.PluginConfigurationManager = PluginConfigurationManager;
exports.pluginLifecycleManager = new PluginLifecycleManager();
exports.pluginConfigurationManager = new PluginConfigurationManager();
//# sourceMappingURL=PluginRegistry.js.map