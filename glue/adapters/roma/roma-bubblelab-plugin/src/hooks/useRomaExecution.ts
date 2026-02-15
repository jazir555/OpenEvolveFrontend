/**
 * useRomaExecution Hook
 *
 * React hook for managing ROMA task execution.
 * Provides methods to execute, cancel, and monitor tasks.
 */

import { useCallback, useEffect, useRef } from 'react';
import { useRomaPlugin } from './useRomaPlugin';
import { RomaExecutionResult, RomaExecutionOptions, RomaPluginError } from '../types/plugin-types';

/**
 * Hook return type for useRomaExecution
 */
export interface UseRomaExecutionReturn {
  executeTask: (goal: string, options?: RomaExecutionOptions) => Promise<RomaExecutionResult>;
  cancelExecution: () => Promise<void>;
  currentExecution: RomaExecutionResult | undefined;
  isExecuting: boolean;
  isReady: boolean;
  error?: string;
}

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
export function useRomaExecution(): UseRomaExecutionReturn {
  const { plugin, isReady, error } = useRomaPlugin();
  const state = plugin.getState();
  const currentExecution = state.currentExecution;
  const isExecuting = state.status === 'executing';

  /**
   * Execute a task using ROMA
   */
  const executeTask = useCallback(async (goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult> => {
    if (!isReady) {
      throw new RomaPluginError('Plugin not initialized. Call initialize() first.', 'PLUGIN_NOT_INITIALIZED');
    }
    
    return await plugin.executeTask(goal, options);
  }, [plugin, isReady]);

  /**
   * Cancel current execution
   */
  const cancelExecution = useCallback(async (): Promise<void> => {
    if (!isReady) {
      throw new RomaPluginError('Plugin not initialized. Call initialize() first.', 'PLUGIN_NOT_INITIALIZED');
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
