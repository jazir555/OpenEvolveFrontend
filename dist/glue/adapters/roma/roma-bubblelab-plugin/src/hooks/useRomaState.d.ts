/**
 * useRomaState Hook
 *
 * React hook for accessing ROMA plugin state.
 * Provides read-only access to the current plugin state.
 */
import { RomaPluginState } from '../types/plugin-types';
/**
 * Hook return type for useRomaState
 */
export interface UseRomaStateReturn {
    state: RomaPluginState;
    isReady: boolean;
    isExecuting: boolean;
    isInitializing: boolean;
    hasError: boolean;
    error?: string;
}
/**
 * useRomaState Hook
 *
 * Provides convenient access to ROMA plugin state.
 * Returns derived state flags for common use cases.
 *
 * @returns Plugin state and status flags
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { state, isReady, isExecuting } = useRomaState();
 *
 *   return (
 *     <div>
 *       <p>Status: {state.status}</p>
 *       <p>Executions: {state.executionHistory.length}</p>
 *       <p>Success Rate: {getSuccessRate()}</p>
 *     </div>
 *   );
 * }
 * ```
 */
export declare function useRomaState(): UseRomaStateReturn;
//# sourceMappingURL=useRomaState.d.ts.map