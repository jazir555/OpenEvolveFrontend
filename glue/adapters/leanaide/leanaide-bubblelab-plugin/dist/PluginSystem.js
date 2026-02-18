import { jsx as _jsx } from "react/jsx-runtime";
import React, { createContext, useContext, useEffect, useState } from 'react';
import { LeanAidePlugin, pluginRegistry, PluginManager as BasePluginManager, } from './PluginInterface';
const PluginManagerContext = createContext(undefined);
export const PluginManagerProvider = ({ children }) => {
    const [plugins, setPlugins] = useState([]);
    const [activePlugins, setActivePlugins] = useState(new Set());
    const refreshPlugins = () => {
        setPlugins(pluginRegistry.getAllPlugins());
        setActivePlugins(new Set(pluginRegistry.getActivePlugins().map((plugin) => plugin.id)));
    };
    useEffect(() => {
        refreshPlugins();
    }, []);
    const activatePlugin = async (pluginId) => {
        const success = await pluginRegistry.activate(pluginId);
        refreshPlugins();
        return success;
    };
    const deactivatePlugin = async (pluginId) => {
        const success = await pluginRegistry.deactivate(pluginId);
        refreshPlugins();
        return success;
    };
    return (_jsx(PluginManagerContext.Provider, { value: {
            plugins,
            activePlugins,
            refreshPlugins,
            activatePlugin,
            deactivatePlugin,
        }, children: children }));
};
export const usePluginManager = () => {
    const context = useContext(PluginManagerContext);
    if (!context) {
        throw new Error('usePluginManager must be used inside PluginManagerProvider');
    }
    return context;
};
export const PluginManager = BasePluginManager;
export { LeanAidePlugin, pluginRegistry };
export default PluginManager;
//# sourceMappingURL=PluginSystem.js.map