"use strict";
/**
 * useRomaState Hook
 *
 * React hook for accessing ROMA plugin state.
 * Provides read-only access to the current plugin state.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.useRomaState = useRomaState;
const useRomaPlugin_1 = require("./useRomaPlugin");
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
function useRomaState() {
    const { state, isReady, error } = (0, useRomaPlugin_1.useRomaPlugin)();
    const isExecuting = state.status === 'executing';
    const isInitializing = state.status === 'initializing';
    const hasError = !!error || state.status === 'failed';
    return {
        state,
        isReady,
        isExecuting,
        isInitializing,
        hasError,
        error
    };
}
//# sourceMappingURL=useRomaState.js.map