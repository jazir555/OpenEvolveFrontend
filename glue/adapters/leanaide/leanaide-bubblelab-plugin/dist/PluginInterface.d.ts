/**
 * LeanAide Plugin Interface and Registration System
 *
 * Defines the interface for LeanAide plugins and provides the registration system
 * for integrating with BubbleLab UI.
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
    /**
     * Initialize the plugin
     */
    initialize(): Promise<void>;
    /**
     * Activate the plugin
     */
    activate(): Promise<void>;
    /**
     * Deactivate the plugin
     */
    deactivate(): Promise<void>;
    /**
     * Dispose the plugin
     */
    dispose(): void;
    /**
     * Get plugin configuration
     */
    getConfig(): LeanAidePluginConfig;
    /**
     * Update plugin configuration
     */
    updateConfig(settings: Record<string, any>): void;
    /**
     * Check if plugin is enabled
     */
    isEnabled(): boolean;
    /**
     * Get plugin settings
     */
    getSettings(): Record<string, any>;
}
declare class LeanAidePluginRegistry {
    private plugins;
    private activePlugins;
    /**
     * Register a plugin
     */
    register(plugin: LeanAidePlugin): boolean;
    /**
     * Unregister a plugin
     */
    unregister(pluginId: string): boolean;
    /**
     * Activate a plugin
     */
    activate(pluginId: string): Promise<boolean>;
    /**
     * Deactivate a plugin
     */
    deactivate(pluginId: string): Promise<boolean>;
    /**
     * Get a plugin by ID
     */
    getPlugin(pluginId: string): LeanAidePlugin | undefined;
    /**
     * Get all registered plugins
     */
    getAllPlugins(): LeanAidePlugin[];
    /**
     * Get all active plugins
     */
    getActivePlugins(): LeanAidePlugin[];
    /**
     * Check if a plugin is active
     */
    isActive(pluginId: string): boolean;
    /**
     * Get plugins by category
     */
    getPluginsByCategory(category: string): LeanAidePlugin[];
    /**
     * Get plugin count
     */
    getPluginCount(): number;
    /**
     * Get active plugin count
     */
    getActivePluginCount(): number;
}
export declare const pluginRegistry: LeanAidePluginRegistry;
export interface PluginManagerProps {
    className?: string;
}
export declare const PluginManager: React.FC<PluginManagerProps>;
export declare class LeanAideAutoformalizationPlugin extends LeanAidePlugin {
    constructor();
    initialize(): Promise<void>;
    activate(): Promise<void>;
    deactivate(): Promise<void>;
}
export type { LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle };
export { LeanAidePlugin, pluginRegistry, PluginManager };
export default LeanAidePlugin;
