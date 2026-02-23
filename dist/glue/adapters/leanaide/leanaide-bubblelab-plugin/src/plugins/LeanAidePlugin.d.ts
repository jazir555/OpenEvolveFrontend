import React from 'react';
import { type AutoformalizationConfig } from '../integration/autoformalizationAnalytics';
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
}
export interface LeanAidePluginConfig extends AutoformalizationConfig {
    analyticsRefreshInterval?: number;
    maxConcurrentRequests?: number;
    cacheEnabled?: boolean;
    cacheTTL?: number;
}
export declare const DEFAULT_LEANAIDE_PLUGIN_CONFIG: LeanAidePluginConfig;
export interface LeanAidePluginProps {
    config?: Partial<LeanAidePluginConfig>;
    onConfigChange?: (config: LeanAidePluginConfig) => void;
    className?: string;
}
export declare const LeanAidePlugin: ({ config, onConfigChange, className, }: LeanAidePluginProps) => JSX.Element;
export declare function registerLeanAidePlugin(): LeanAidePluginInterface;
export default LeanAidePlugin;
//# sourceMappingURL=LeanAidePlugin.d.ts.map