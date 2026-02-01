/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';

/**
 * MathSolver Mode Prompts Content
 * Prompts for MathSolver mode including system prompt
 */
export const MathSolverPromptsContent: React.FC = () => {
    return (
        <div id="mathsolver-prompts-container" className="prompts-mode-container">
            {/* Main System Prompt */}
            <div className="prompt-content-pane" data-prompt-key="mathsolver-system">
                <h4 className="prompt-pane-title">MathSolver Configuration</h4>
                <div className="prompt-card">
                    <div className="prompt-card-header">
                        <span className="prompt-card-title">System Prompt</span>
                        <div className="prompt-model-selector">
                            <select className="prompt-model-select" data-agent="mathsolver">
                                <option value="">Use Global Model</option>
                            </select>
                        </div>
                    </div>
                    <div className="prompt-card-body">
                        <textarea
                            id="sys-mathsolver"
                            className="prompt-textarea"
                            rows={12}
                            placeholder="Enter the system prompt for MathSolver mode..."
                        />
                    </div>
                </div>
            </div>

            {/* Z3 Formalization Prompt */}
            <div className="prompt-content-pane" data-prompt-key="mathsolver-z3">
                <h4 className="prompt-pane-title">Z3 Formalization</h4>
                <div className="prompt-card">
                    <div className="prompt-card-header">
                        <span className="prompt-card-title">Z3 Formalization Prompt</span>
                    </div>
                    <div className="prompt-card-body">
                        <textarea
                            id="sys-mathsolver-z3"
                            className="prompt-textarea"
                            rows={8}
                            placeholder="Enter the prompt for Z3 SMT-LIB formalization..."
                        />
                    </div>
                </div>
            </div>

            {/* Lean Formalization Prompt */}
            <div className="prompt-content-pane" data-prompt-key="mathsolver-lean">
                <h4 className="prompt-pane-title">Lean Formalization</h4>
                <div className="prompt-card">
                    <div className="prompt-card-header">
                        <span className="prompt-card-title">Lean Formalization Prompt</span>
                    </div>
                    <div className="prompt-card-body">
                        <textarea
                            id="sys-mathsolver-lean"
                            className="prompt-textarea"
                            rows={8}
                            placeholder="Enter the prompt for Lean theorem formalization..."
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MathSolverPromptsContent;
