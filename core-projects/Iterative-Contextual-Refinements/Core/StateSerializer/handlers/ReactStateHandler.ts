/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ReactStateHandler - State management for React mode
 * Handles export/import of React mode-specific state including build artifacts
 */

import type { ModeStateHandler } from '../ModeStateHandler';

// Import React mode types from local implementation
import type { ReactModeState } from '../../React/ReactMode';

/**
 * State handler for React mode.
 */
export const reactStateHandler: ModeStateHandler<ReactModeState> = {
    modeName: 'react',

    getFullState(): ReactModeState | null {
        // Get current React mode state from global state
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__REACT_MODE_STATE__;
        return state || null;
    },

    restoreState(state: ReactModeState | null): void {
        // Restore React mode state during import
        if (typeof window === 'undefined') {
            return;
        }

        if (state) {
            (window as any).__REACT_MODE_STATE__ = state;
            console.log('[React Mode] State restored from import');
        }
    },

    renderAfterImport(): void {
        // Refresh React mode UI after state import
        if (typeof window === 'undefined') {
            return;
        }

        // Dispatch custom event to notify React mode UI of state change
        const event = new CustomEvent('react-mode:state-restored', {
            detail: { state: (window as any).__REACT_MODE_STATE__ }
        });
        window.dispatchEvent(event);
    },

    getEmbeddedState(): unknown | null {
        // Get embedded state (e.g., build artifacts, worker states)
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__REACT_MODE_STATE__;
        if (!state) {
            return null;
        }

        // Return embedded build state if exists
        return {
            buildArtifacts: (state as any).buildArtifacts || null,
            workerStates: (state as any).workerStates || null,
        };
    },

    restoreEmbeddedState(state: unknown | null): void {
        // Restore embedded state
        if (typeof window === 'undefined') {
            return;
        }

        const reactState = (window as any).__REACT_MODE_STATE__;
        if (reactState && state) {
            const embedded = state as Record<string, unknown>;
            if (embedded.buildArtifacts) {
                (reactState as any).buildArtifacts = embedded.buildArtifacts;
            }
            if (embedded.workerStates) {
                (reactState as any).workerStates = embedded.workerStates;
            }
        }
    },
};
