import React, { createContext, useContext, useEffect, useState } from 'react';

import {
  LeanAidePlugin,
  pluginRegistry,
  PluginManager as BasePluginManager,
  type LeanAidePluginConfig,
  type LeanAidePluginInterface,
  type LeanAidePluginLifecycle,
} from './PluginInterface';

interface PluginManagerContextType {
  plugins: LeanAidePluginInterface[];
  activePlugins: Set<string>;
  refreshPlugins: () => void;
  activatePlugin: (pluginId: string) => Promise<boolean>;
  deactivatePlugin: (pluginId: string) => Promise<boolean>;
}

const PluginManagerContext = createContext<PluginManagerContextType | undefined>(undefined);

export const PluginManagerProvider = ({ children }: { children: React.ReactNode }) => {
  const [plugins, setPlugins] = useState<LeanAidePluginInterface[]>([]);
  const [activePlugins, setActivePlugins] = useState<Set<string>>(new Set());

  const refreshPlugins = () => {
    setPlugins(pluginRegistry.getAllPlugins());
    setActivePlugins(new Set(pluginRegistry.getActivePlugins().map((plugin) => plugin.id)));
  };

  useEffect(() => {
    refreshPlugins();
  }, []);

  const activatePlugin = async (pluginId: string) => {
    const success = await pluginRegistry.activate(pluginId);
    refreshPlugins();
    return success;
  };

  const deactivatePlugin = async (pluginId: string) => {
    const success = await pluginRegistry.deactivate(pluginId);
    refreshPlugins();
    return success;
  };

  return (
    <PluginManagerContext.Provider
      value={{
        plugins,
        activePlugins,
        refreshPlugins,
        activatePlugin,
        deactivatePlugin,
      }}
    >
      {children}
    </PluginManagerContext.Provider>
  );
};

export const usePluginManager = () => {
  const context = useContext(PluginManagerContext);
  if (!context) {
    throw new Error('usePluginManager must be used inside PluginManagerProvider');
  }

  return context;
};

export const PluginManager = BasePluginManager;

export type { LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle };
export { LeanAidePlugin, pluginRegistry };

export default PluginManager;
