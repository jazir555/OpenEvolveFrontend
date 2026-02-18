/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * MathSolverStateHandler - State management for MathSolver mode
 * Handles export/import of MathSolver-specific state
 */

import type { ModeStateHandler } from '../ModeStateHandler';

// Import MathSolver types from local implementation
import type { MathSolverState } from '../../MathSolver/MathSolverMode';

/**
 * State handler for MathSolver mode.
 */
export const mathsolverStateHandler: ModeStateHandler<MathSolverState> = {
    modeName: 'mathsolver',

    getFullState(): MathSolverState | null {
        // Get current MathSolver state from global state
        // This will be called by ConfigManager during export
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__MATHSOLVER_STATE__;
        return state || null;
    },

    restoreState(state: MathSolverState | null): void {
        // Restore MathSolver state during import
        // The state will already be sanitized (processing states reset)
        if (typeof window === 'undefined') {
            return;
        }

        if (state) {
            (window as any).__MATHSOLVER_STATE__ = state;
            console.log('[MathSolver] State restored from import');
        }
    },

    renderAfterImport(): void {
        // Refresh MathSolver UI after state import
        if (typeof window === 'undefined') {
            return;
        }

        // Dispatch custom event to notify MathSolver UI of state change
        const event = new CustomEvent('mathsolver:state-restored', {
            detail: { state: (window as any).__MATHSOLVER_STATE__ }
        });
        window.dispatchEvent(event);
    },

    getEmbeddedState(): unknown | null {
        // Get any embedded state (e.g., agentic state embedded in MathSolver)
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__MATHSOLVER_STATE__;
        if (!state) {
            return null;
        }

        // Return embedded agentic state if exists
        return (state as any).agenticState || null;
    },

    restoreEmbeddedState(state: unknown | null): void {
        // Restore embedded state
        if (typeof window === 'undefined') {
            return;
        }

        const mathsolverState = (window as any).__MATHSOLVER_STATE__;
        if (mathsolverState && state) {
            (mathsolverState as any).agenticState = state;
        }
    },
};
