import React from 'react';
import { LeanAidePlugin, pluginRegistry, type LeanAidePluginConfig, type LeanAidePluginInterface, type LeanAidePluginLifecycle } from './PluginInterface';
export declare const PluginManagerProvider: ({ children }: {
    children: React.ReactNode;
}) => any;
export declare const usePluginManager: () => any;
export declare const PluginManager: React.FC<import("./PluginInterface").PluginManagerProps>;
export type { LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle };
export { LeanAidePlugin, pluginRegistry };
export default PluginManager;
