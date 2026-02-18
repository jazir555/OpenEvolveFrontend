/**
 * LeanAide Autoformalization Plugin for BubbleLab UI
 *
 * This plugin integrates the complete LeanAide autoformalization system with predictive analytics
 * into the BubbleLab UI as a comprehensive plugin.
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
}
export interface LeanAidePluginConfig {
    enableAnalytics: boolean;
    enablePredictiveFlagging: boolean;
    enableKnowledgeGraph: boolean;
    analyticsRefreshInterval: number;
    maxConcurrentRequests: number;
    cacheEnabled: boolean;
    cacheTTL: number;
    serverUrl: string;
    apiKey?: string;
}
export declare const DEFAULT_LEANAIDE_PLUGIN_CONFIG: LeanAidePluginConfig;
export interface LeanAidePluginProps {
    config?: Partial<LeanAidePluginConfig>;
    onConfigChange?: (config: LeanAidePluginConfig) => void;
    className?: string;
}
export declare const LeanAidePlugin: React.FC<LeanAidePluginProps>;
export declare function registerLeanAidePlugin(): LeanAidePluginInterface;
export type { LeanAidePluginInterface, LeanAidePluginConfig };
export default LeanAidePlugin;
