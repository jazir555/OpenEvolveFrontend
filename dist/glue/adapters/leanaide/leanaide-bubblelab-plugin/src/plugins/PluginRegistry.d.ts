import { pluginRegistry, PluginManager, type LeanAidePluginInterface, type LeanAidePluginLifecycle } from '../PluginInterface';
export interface PluginConfiguration {
    [pluginId: string]: Record<string, unknown>;
}
export declare class PluginLifecycleManager {
    private readonly states;
    startPlugin(pluginId: string): Promise<boolean>;
    stopPlugin(pluginId: string): Promise<boolean>;
    getPluginState(pluginId: string): 'active' | 'inactive' | 'error' | undefined;
}
export declare class PluginInstaller {
    static installFromUrl(_url: string): Promise<boolean>;
    static installFromPackage(_name: string): Promise<boolean>;
}
export declare class PluginConfigurationManager {
    private configurations;
    getConfiguration(pluginId: string): Record<string, unknown>;
    setConfiguration(pluginId: string, config: Record<string, unknown>): void;
    getAllConfigurations(): PluginConfiguration;
}
export declare const pluginLifecycleManager: PluginLifecycleManager;
export declare const pluginConfigurationManager: PluginConfigurationManager;
export type { LeanAidePluginInterface, LeanAidePluginLifecycle };
export { pluginRegistry, PluginManager };
//# sourceMappingURL=PluginRegistry.d.ts.map