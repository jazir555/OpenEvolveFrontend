/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver Mode Integration
 * 
 * Provides the integration layer for MathSolver as an application mode
 * in Iterative Studio.
 */

import { MathSolverCore, MathSolverState } from './MathSolverCore';
import { MATH_SOLVER_SYSTEM_PROMPT } from './MathSolverPrompts';

// Global state for MathSolver mode
let activeMathSolverCore: MathSolverCore | null = null;
let mathSolverContentContainer: HTMLElement | null = null;
let isMathSolverInitialized = false;

/**
 * Initialize MathSolver mode
 */
export function initializeMathSolverMode(
    contentContainer: HTMLElement
): void {
    mathSolverContentContainer = contentContainer;
    isMathSolverInitialized = true;
    console.log('[MathSolver] Mode initialized');
}

/**
 * Start MathSolver process
 */
export async function startMathSolverProcess(
    problemStatement: string,
    options?: {
        preferredSolver?: 'z3' | 'lean' | 'unified' | 'auto';
        useKnowledgeBase?: boolean;
        timeout?: number;
    }
): Promise<void> {
    if (!isMathSolverInitialized || !mathSolverContentContainer) {
        throw new Error('[MathSolver] Mode not initialized');
    }

    // Clear previous content
    mathSolverContentContainer.innerHTML = '';

    // Create new MathSolver instance
    activeMathSolverCore = new MathSolverCore();

    // Create and mount the UI
    const { MathSolverUI } = await import('./MathSolverUI');
    const container = document.createElement('div');
    container.style.height = '100%';
    mathSolverContentContainer.appendChild(container);

    // Render the MathSolver UI
    const React = await import('react');
    const { createRoot } = await import('react-dom/client');
    const root = createRoot(container);
    
    root.render(
        React.createElement(MathSolverUI, {
            initialProblem: problemStatement,
            onClose: () => {
                // Handle close
                console.log('[MathSolver] Mode closed');
            }
        })
    );

    // If problem statement provided, start solving
    if (problemStatement.trim()) {
        const problem = activeMathSolverCore.createProblem(problemStatement);
        await activeMathSolverCore.solve({
            problem,
            preferredSolver: options?.preferredSolver || 'auto',
            useKnowledgeBase: options?.useKnowledgeBase ?? true,
            timeout: options?.timeout || 300
        });
    }
}

/**
 * Stop MathSolver process
 */
export function stopMathSolverProcess(): void {
    if (activeMathSolverCore) {
        // Reset the core
        activeMathSolverCore.reset();
        activeMathSolverCore = null;
    }
    
    if (mathSolverContentContainer) {
        mathSolverContentContainer.innerHTML = '';
    }
    
    console.log('[MathSolver] Process stopped');
}

/**
 * Get active MathSolver state for export
 */
export function getActiveMathSolverState(): any | null {
    if (!activeMathSolverCore) return null;
    return activeMathSolverCore.exportState();
}

/**
 * Set active MathSolver state from import
 */
export function setActiveMathSolverState(state: any): void {
    if (!activeMathSolverCore) {
        activeMathSolverCore = new MathSolverCore();
    }
    activeMathSolverCore.importState(state);
}

/**
 * Check if MathSolver is running
 */
export function isMathSolverRunning(): boolean {
    return activeMathSolverCore !== null;
}

/**
 * Get MathSolver system prompt
 */
export function getMathSolverSystemPrompt(): string {
    return MATH_SOLVER_SYSTEM_PROMPT;
}

/**
 * Rehydrate MathSolver UI (for after import/state restore)
 */
export function rehydrateMathSolverUI(): void {
    if (!mathSolverContentContainer || !activeMathSolverCore) return;
    
    // Re-render the UI with existing state
    const state = activeMathSolverCore.getState();
    if (state.currentProblem) {
        // Re-render with existing problem
        console.log('[MathSolver] Rehydrating UI with problem:', state.currentProblem.statement);
    }
}

// Export for global access
export { activeMathSolverCore };
