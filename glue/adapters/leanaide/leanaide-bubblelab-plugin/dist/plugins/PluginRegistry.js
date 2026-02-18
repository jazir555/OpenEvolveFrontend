import { pluginRegistry, PluginManager, } from '../PluginInterface';
export class PluginLifecycleManager {
    constructor() {
        this.states = new Map();
    }
    async startPlugin(pluginId) {
        const success = await pluginRegistry.activate(pluginId);
        this.states.set(pluginId, success ? 'active' : 'error');
        return success;
    }
    async stopPlugin(pluginId) {
        const success = await pluginRegistry.deactivate(pluginId);
        this.states.set(pluginId, success ? 'inactive' : 'error');
        return success;
    }
    getPluginState(pluginId) {
        return this.states.get(pluginId);
    }
}
export class PluginInstaller {
    static async installFromUrl(_url) {
        return true;
    }
    static async installFromPackage(_name) {
        return true;
    }
}
export class PluginConfigurationManager {
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
export const pluginLifecycleManager = new PluginLifecycleManager();
export const pluginConfigurationManager = new PluginConfigurationManager();
export { pluginRegistry, PluginManager };
//# sourceMappingURL=PluginRegistry.js.map