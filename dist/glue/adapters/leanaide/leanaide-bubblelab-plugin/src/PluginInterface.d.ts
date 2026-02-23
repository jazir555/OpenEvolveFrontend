import React from 'react';
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
export declare class LeanAidePlugin implements LeanAidePluginInterface, LeanAidePluginLifecycle {
    readonly id: string;
    readonly name: string;
    readonly description: string;
    readonly version: string;
    readonly category: string;
    readonly component: React.ComponentType<any>;
    readonly icon: React.ReactNode;
    readonly settingsSchema?: Record<string, unknown>;
    readonly permissions?: string[];
    readonly dependencies?: string[];
    enabled: boolean;
    protected config: LeanAidePluginConfig;
    constructor(plugin: LeanAidePluginInterface);
    initialize(): Promise<void>;
    activate(): Promise<void>;
    deactivate(): Promise<void>;
    dispose(): void;
    isEnabled(): boolean;
    getConfig(): LeanAidePluginConfig;
}
type RegisteredPlugin = LeanAidePluginInterface & Partial<LeanAidePluginLifecycle>;
declare class LeanAidePluginRegistry {
    private readonly plugins;
    private readonly activePlugins;
    register(plugin: RegisteredPlugin): boolean;
    unregister(pluginId: string): boolean;
    activate(pluginId: string): Promise<boolean>;
    deactivate(pluginId: string): Promise<boolean>;
    getPlugin(pluginId: string): RegisteredPlugin | undefined;
    getAllPlugins(): RegisteredPlugin[];
    getActivePlugins(): RegisteredPlugin[];
    isActive(pluginId: string): boolean;
}
export declare const pluginRegistry: LeanAidePluginRegistry;
export interface PluginManagerProps {
    className?: string;
}
export declare const PluginManager: React.FC<PluginManagerProps>;
export default LeanAidePlugin;
//# sourceMappingURL=PluginInterface.d.ts.map