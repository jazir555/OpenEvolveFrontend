"use strict";
/**
 * useRomaPlugin Hook
 *
 * Core hook for accessing ROMA plugin instance and methods.
 * This is the primary hook that all other ROMA hooks depend on.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRomaPlugin = useRomaPlugin;
const react_1 = require("react");
const createRomaPlugin_1 = require("../utils/createRomaPlugin");
/**
 * useRomaPlugin Hook
 *
 * Provides access to the global ROMA plugin instance.
 * Uses singleton pattern - only one plugin instance exists per application.
 *
 * @returns Plugin instance, config, state, and status flags
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { plugin, isReady, isExecuting } = useRomaPlugin();
 *
 *   const handleExecute = async () => {
 *     if (!isReady) {
 *       console.error('Plugin not initialized');
 *       return;
 *     }
 *
 *     const result = await plugin.executeTask('What is 2+2?');
 *     console.log('Result:', result);
 *   };
 *
 *   return (
 *     <div>
 *       <p>Status: {isExecuting ? 'Executing...' : 'Idle'}</p>
 *       <button onClick={handleExecute} disabled={!isReady}>
 *         Execute Task
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 */
function useRomaPlugin() {
    const plugin = (0, react_1.useMemo)(() => (0, createRomaPlugin_1.getRomaPluginInstance)(), []);
    if (!plugin) {
        return {
            plugin: {},
            config: {},
            state: {},
            isReady: false,
            isExecuting: false,
            error: 'ROMA plugin not initialized. Call createRomaPlugin() first.'
        };
    }
    const state = plugin.getState();
    const isReady = plugin.isReady();
    const isExecuting = state.status === 'executing';
    const error = state.initializationError;
    return {
        plugin,
        config: state,
        state,
        isReady,
        isExecuting,
        error
    };
}
//# sourceMappingURL=useRomaPlugin.js.map