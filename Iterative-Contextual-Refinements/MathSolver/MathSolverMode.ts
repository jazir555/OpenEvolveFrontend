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

// Toast notification helper
function showToast(message: string, type: 'info' | 'success' | 'error' = 'info'): void {
    const toast = document.createElement('div');
    toast.className = `mathsolver-toast mathsolver-toast--${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 6px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#3B82F6'};
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

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
        showToast('MathSolver mode not initialized', 'error');
        throw new Error('[MathSolver] Mode not initialized');
    }

    // Set generating flag
    const { globalState } = await import('../Core/State');
    globalState.isGenerating = true;
    globalState.isMathSolverRunning = true;

    // Show start notification
    showToast('Solving mathematical problem...', 'info');

    try {
        await runMathSolverProcess(problemStatement, options);
        // Show success notification
        showToast('Solution found!', 'success');
    } catch (error) {
        // Show error notification
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        showToast(`Error: ${errorMessage}`, 'error');
        throw error;
    } finally {
        globalState.isGenerating = false;
        globalState.isMathSolverRunning = false;
    }
}

/**
 * Internal function to run MathSolver process
 */
async function runMathSolverProcess(
    problemStatement: string,
    options?: {
        preferredSolver?: 'z3' | 'lean' | 'unified' | 'auto';
        useKnowledgeBase?: boolean;
        timeout?: number;
    }
): Promise<void> {
    if (!mathSolverContentContainer) return;

    // Clean up any existing React root first
    if (activeReactRoot) {
        try {
            activeReactRoot.unmount();
        } catch (e) {
            // Root may already be unmounted
        }
        activeReactRoot = null;
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
    activeReactRoot = root; // Store for cleanup
    
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

// Store React root for proper cleanup
let activeReactRoot: ReturnType<typeof import('react-dom/client').createRoot> | null = null;

/**
 * Stop MathSolver process
 */
export async function stopMathSolverProcess(): Promise<void> {
    if (activeMathSolverCore) {
        // Reset the core
        activeMathSolverCore.reset();
        activeMathSolverCore = null;
    }
    
    // Properly unmount React root to prevent memory leaks
    if (activeReactRoot) {
        try {
            activeReactRoot.unmount();
        } catch (e) {
            // Root may already be unmounted
        }
        activeReactRoot = null;
    }
    
    if (mathSolverContentContainer) {
        mathSolverContentContainer.innerHTML = '';
    }

    // Reset state flags
    const { globalState } = await import('../Core/State');
    globalState.isGenerating = false;
    globalState.isMathSolverRunning = false;
    
    console.log('[MathSolver] Process stopped');
}

/**
 * Get active MathSolver state for export
 */
export function getActiveMathSolverState(): ReturnType<MathSolverCore['exportState']> | null {
    if (!activeMathSolverCore) return null;
    return activeMathSolverCore.exportState();
}

/**
 * Set active MathSolver state from import
 */
export function setActiveMathSolverState(state: Parameters<MathSolverCore['importState']>[0]): void {
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
 * Render MathSolver UI into a container
 * Used when switching to MathSolver mode
 */
export async function renderMathSolverUI(container: HTMLElement): Promise<void> {
    if (!container) {
        console.error('[MathSolver] Cannot render: no container provided');
        return;
    }
    
    try {
        // Update the content container reference
        mathSolverContentContainer = container;
        
        // Clean up any existing React root
        if (activeReactRoot) {
            try {
                activeReactRoot.unmount();
            } catch (e) {
                // Root may already be unmounted
            }
            activeReactRoot = null;
        }
        
        // Clear existing content
        container.innerHTML = '';
        
        // Create MathSolver instance if not exists
        if (!activeMathSolverCore) {
            activeMathSolverCore = new MathSolverCore();
        }
        
        // Create container for React
        const reactContainer = document.createElement('div');
        reactContainer.style.height = '100%';
        container.appendChild(reactContainer);
        
        // Dynamically import React and render
        const React = await import('react');
        const { createRoot } = await import('react-dom/client');
        const { MathSolverUI } = await import('./MathSolverUI');
        
        const root = createRoot(reactContainer);
        activeReactRoot = root; // Store for cleanup
        
        const state = activeMathSolverCore.getState();
        
        root.render(
            React.createElement(MathSolverUI, {
                initialProblem: state.currentProblem?.statement || '',
                onClose: () => {
                    console.log('[MathSolver] UI closed');
                }
            })
        );
        
        console.log('[MathSolver] UI rendered');
    } catch (error) {
        console.error('[MathSolver] Failed to render UI:', error);
        // Show error in container using safe DOM manipulation
        container.innerHTML = '';
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'padding: 20px; color: #ef4444;';
        
        const heading = document.createElement('h3');
        heading.textContent = 'Failed to load MathSolver';
        errorDiv.appendChild(heading);
        
        const message = document.createElement('p');
        message.textContent = error instanceof Error ? error.message : 'Unknown error';
        errorDiv.appendChild(message);
        
        const reloadButton = document.createElement('button');
        reloadButton.textContent = 'Reload Page';
        reloadButton.addEventListener('click', () => window.location.reload());
        errorDiv.appendChild(reloadButton);
        
        container.appendChild(errorDiv);
    }
}

/**
 * Rehydrate MathSolver UI (for after import/state restore)
 */
export function rehydrateMathSolverUI(): void {
    if (!mathSolverContentContainer) return;
    
    // Re-render the UI with existing state
    if (activeMathSolverCore) {
        const state = activeMathSolverCore.getState();
        if (state.currentProblem) {
            console.log('[MathSolver] Rehydrating UI with problem:', state.currentProblem.statement);
        }
    }
    
    // Re-render the UI
    renderMathSolverUI(mathSolverContentContainer);
}

/**
 * Display MathSolver result
 * Creates and manages the results container for MathSolver output
 */
export function displayMathSolverResult(result: {
    success: boolean;
    executionTimeMs?: number;
    error?: string;
}): void {
    // Results are displayed via the MathSolverUI React component
    console.log('[MathSolver] Result:', result);
    
    // Ensure results container exists
    let resultsContainer = document.getElementById('mathsolver-results-container');
    if (!resultsContainer && mathSolverContentContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'mathsolver-results-container';
        resultsContainer.style.cssText = `
            max-height: 500px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px;
            border-radius: 8px;
            background: var(--bg-secondary, #1e1e1e);
            margin-top: 12px;
        `;
        mathSolverContentContainer.appendChild(resultsContainer);
    }
    
    // If there's additional result data to display, add it to the container
    if (resultsContainer && result && typeof result === 'object') {
        const resultElement = document.createElement('div');
        resultElement.className = 'mathsolver-result-item';
        resultElement.style.cssText = `
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            background: var(--bg-tertiary, #2a2a2a);
            border-left: 3px solid ${result.success ? '#10B981' : '#EF4444'};
        `;
        
        const statusDiv = document.createElement('div');
        statusDiv.style.cssText = 'font-weight: 600; margin-bottom: 4px;';
        const status = result.success ? '✓ Success' : '✗ Failed';
        const executionTime = result.executionTimeMs ? `(${result.executionTimeMs}ms)` : '';
        statusDiv.textContent = `${status} ${executionTime}`;
        resultElement.appendChild(statusDiv);
        
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = 'font-size: 0.9em; opacity: 0.8;';
        // Use textContent to prevent XSS
        messageDiv.textContent = result.error || 'Solution completed';
        resultElement.appendChild(messageDiv);
        
        resultsContainer.appendChild(resultElement);
        
        // Auto-scroll to bottom for new results
        resultsContainer.scrollTop = resultsContainer.scrollHeight;
    }
}

// Export for global access
export { activeMathSolverCore };
