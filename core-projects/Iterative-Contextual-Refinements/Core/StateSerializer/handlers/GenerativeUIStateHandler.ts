/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * GenerativeUIStateHandler - State management for GenerativeUI mode
 * Handles export/import of GenerativeUI-specific state including interaction data
 */

import type { ModeStateHandler } from '../ModeStateHandler';

// Import GenerativeUI types from local implementation
import type { GenerativeUIState } from '../../GenerativeUI/GenerativeUICore';

/**
 * State handler for GenerativeUI mode.
 */
export const generativeUIStateHandler: ModeStateHandler<GenerativeUIState> = {
    modeName: 'generativeui',

    getFullState(): GenerativeUIState | null {
        // Get current GenerativeUI state from global state
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__GENERATIVEUI_STATE__;
        return state || null;
    },

    restoreState(state: GenerativeUIState | null): void {
        // Restore GenerativeUI state during import
        if (typeof window === 'undefined') {
            return;
        }

        if (state) {
            (window as any).__GENERATIVEUI_STATE__ = state;
            console.log('[GenerativeUI] State restored from import');
        }
    },

    renderAfterImport(): void {
        // Refresh GenerativeUI UI after state import
        if (typeof window === 'undefined') {
            return;
        }

        // Dispatch custom event to notify GenerativeUI UI of state change
        const event = new CustomEvent('generativeui:state-restored', {
            detail: { state: (window as any).__GENERATIVEUI_STATE__ }
        });
        window.dispatchEvent(event);
    },

    getEmbeddedState(): unknown | null {
        // Get embedded state (e.g., interaction history, heatmap data)
        if (typeof window === 'undefined') {
            return null;
        }

        const state = (window as any).__GENERATIVEUI_STATE__;
        if (!state) {
            return null;
        }

        // Return embedded interaction data if exists
        return {
            interactionHistory: (state as any).interactionHistory || null,
            heatmapData: (state as any).heatmapData || null,
        };
    },

    restoreEmbeddedState(state: unknown | null): void {
        // Restore embedded state
        if (typeof window === 'undefined') {
            return;
        }

        const generativeUIState = (window as any).__GENERATIVEUI_STATE__;
        if (generativeUIState && state) {
            const embedded = state as Record<string, unknown>;
            if (embedded.interactionHistory) {
                (generativeUIState as any).interactionHistory = embedded.interactionHistory;
            }
            if (embedded.heatmapData) {
                (generativeUIState as any).heatmapData = embedded.heatmapData;
            }
        }
    },
};
