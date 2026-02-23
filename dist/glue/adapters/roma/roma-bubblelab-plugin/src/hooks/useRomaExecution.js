"use strict";
/**
 * useRomaExecution Hook
 *
 * React hook for managing ROMA task execution.
 * Provides methods to execute, cancel, and monitor tasks.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRomaExecution = useRomaExecution;
const react_1 = require("react");
const useRomaPlugin_1 = require("./useRomaPlugin");
const plugin_types_1 = require("../types/plugin-types");
/**
 * useRomaExecution Hook
 *
 * Provides task execution management for ROMA plugin.
 * Includes auto-cleanup on unmount to prevent memory leaks.
 *
 * @returns Execution methods and current state
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { executeTask, cancelExecution, currentExecution, isExecuting } = useRomaExecution();
 *   const [goal, setGoal] = useState('');
 *
 *   const handleExecute = async () => {
 *     if (!goal.trim()) return;
 *     const result = await executeTask(goal, { maxDepth: 3 });
 *     console.log('Result:', result);
 *   };
 *
 *   const handleCancel = async () => {
 *     await cancelExecution();
 *   };
 *
 *   return (
 *     <div>
 *       <input value={goal} onChange={(e) => setGoal(e.target.value)} />
 *       <button onClick={handleExecute} disabled={isExecuting || !goal}>
 *         {isExecuting ? 'Executing...' : 'Execute'}
 *       </button>
 *       <button onClick={handleCancel} disabled={!isExecuting}>
 *         Cancel
 *       </button>
 *       {currentExecution && (
 *         <div>
 *           <p>Status: {currentExecution.status}</p>
 *           <p>Time: {currentExecution.statistics?.executionTime}ms</p>
 *         </div>
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */
function useRomaExecution() {
    const { plugin, isReady, error } = (0, useRomaPlugin_1.useRomaPlugin)();
    const state = plugin.getState();
    const currentExecution = state.currentExecution;
    const isExecuting = state.status === 'executing';
    /**
     * Execute a task using ROMA
     */
    const executeTask = (0, react_1.useCallback)(async (goal, options) => {
        if (!isReady) {
            throw new plugin_types_1.RomaPluginError('Plugin not initialized. Call initialize() first.', 'PLUGIN_NOT_INITIALIZED');
        }
        return await plugin.executeTask(goal, options);
    }, [plugin, isReady]);
    /**
     * Cancel current execution
     */
    const cancelExecution = (0, react_1.useCallback)(async () => {
        if (!isReady) {
            throw new plugin_types_1.RomaPluginError('Plugin not initialized. Call initialize() first.', 'PLUGIN_NOT_INITIALIZED');
        }
        await plugin.cancelExecution();
    }, [plugin, isReady]);
    return {
        executeTask,
        cancelExecution,
        currentExecution,
        isExecuting,
        isReady,
        error
    };
}
//# sourceMappingURL=useRomaExecution.js.map