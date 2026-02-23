"use strict";
/**
 * useRomaConfig Hook
 *
 * React hook for ROMA plugin configuration management.
 * Provides methods to update and access ROMA configuration.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRomaConfig = useRomaConfig;
const react_1 = require("react");
const useRomaPlugin_1 = require("./useRomaPlugin");
/**
 * useRomaConfig Hook
 *
 * Provides configuration management for the ROMA plugin.
 * Uses the underlying useRomaPlugin hook for state access.
 *
 * @returns Configuration state and update methods
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { config, updateConfig } = useRomaConfig();
 *
 *   const handleUpdate = async () => {
 *     await updateConfig({ maxDepth: 5 });
 *   };
 *
 *   return (
 *     <div>
 *       <p>Max Depth: {config.maxDepth}</p>
 *       <button onClick={handleUpdate}>Update</button>
 *     </div>
 *   );
 * }
 * ```
 */
function useRomaConfig() {
    const { plugin, error } = (0, useRomaPlugin_1.useRomaPlugin)();
    const config = plugin.getState();
    /**
     * Update ROMA plugin configuration
     *
     * @param configUpdate - Partial configuration update
     * @throws RomaPluginError if configuration update fails
     */
    const updateConfig = (0, react_1.useCallback)(async (configUpdate) => {
        await plugin.updateConfig(configUpdate);
    }, [plugin]);
    return {
        config,
        updateConfig,
        isUpdating: plugin.getState().status === 'configuring',
        error
    };
}
//# sourceMappingURL=useRomaConfig.js.map