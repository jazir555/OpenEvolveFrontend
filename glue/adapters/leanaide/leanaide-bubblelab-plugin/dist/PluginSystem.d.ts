/**
 * Complete Plugin System for LeanAide Autoformalization in BubbleLab
 *
 * This module provides the complete plugin architecture for integrating
 * the LeanAide autoformalization system with predictive analytics into BubbleLab UI.
 */
import React from 'react';
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
export interface LeanAidePluginLifecycle {
    initialize?(): Promise<void>;
    activate?(): Promise<void>;
    deactivate?(): Promise<void>;
    dispose?(): void;
}
export declare abstract class LeanAidePlugin implements LeanAidePluginInterface, LeanAidePluginLifecycle {
    readonly id: string;
    readonly name: string;
    readonly description: string;
    readonly version: string;
    readonly category: string;
    readonly component: React.ComponentType<any>;
    readonly icon: React.ReactNode;
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
    protected config: LeanAidePluginConfig;
    constructor(config: Omit<LeanAidePluginInterface, 'component' | 'icon'> & {
        component: React.ComponentType<any>;
        icon: React.ReactNode;
    });
    initialize(): Promise<void>;
    activate(): Promise<void>;
    deactivate(): Promise<void>;
    dispose(): void;
    getConfig(): LeanAidePluginConfig;
    updateConfig(settings: Record<string, any>): void;
    isEnabled(): boolean;
    getSettings(): Record<string, any>;
}
declare class LeanAidePluginRegistry {
    private plugins;
    private activePlugins;
    register(plugin: LeanAidePlugin): boolean;
    unregister(pluginId: string): boolean;
    activate(pluginId: string): Promise<boolean>;
    deactivate(pluginId: string): Promise<boolean>;
    getPlugin(pluginId: string): LeanAidePlugin | undefined;
    getAllPlugins(): LeanAidePlugin[];
    getActivePlugins(): LeanAidePlugin[];
    isActive(pluginId: string): boolean;
    getPluginsByCategory(category: string): LeanAidePlugin[];
    getPluginCount(): number;
    getActivePluginCount(): number;
}
export declare const pluginRegistry: LeanAidePluginRegistry;
interface PluginManagerContextType {
    plugins: LeanAidePlugin[];
    activePlugins: Set<string>;
    registerPlugin: (plugin: LeanAidePlugin) => boolean;
    unregisterPlugin: (pluginId: string) => boolean;
    activatePlugin: (pluginId: string) => Promise<boolean>;
    deactivatePlugin: (pluginId: string) => Promise<boolean>;
    getPlugin: (pluginId: string) => LeanAidePlugin | undefined;
    refreshPlugins: () => void;
}
export declare const PluginManagerProvider: React.FC<{
    children: React.ReactNode;
}>;
export declare const usePluginManager: () => PluginManagerContextType;
export declare const PluginManager: React.FC<{
    className?: string;
}>;
export declare class BubbleLabLeanAideIntegrationPlugin extends LeanAidePlugin {
    constructor();
    initialize(): Promise<void>;
    activate(): Promise<void>;
    deactivate(): Promise<void>;
}
export type { LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle };
export { LeanAidePlugin, pluginRegistry, PluginManager, PluginManagerProvider, usePluginManager };
export default BubbleLabLeanAideIntegrationPlugin;
