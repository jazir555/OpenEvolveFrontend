/**
 * useRomaExecution Hook
 *
 * React hook for managing ROMA task execution.
 * Provides methods to execute, cancel, and monitor tasks.
 */
import { RomaExecutionResult, RomaExecutionOptions } from '../types/plugin-types';
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
export declare function useRomaExecution(): UseRomaExecutionReturn;
//# sourceMappingURL=useRomaExecution.d.ts.map