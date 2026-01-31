/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver UI Component
 * 
 * React component for the MathSolver mode interface.
 * Provides input for mathematical problems and displays solver results.
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
    MathSolverCore,
    SolverSystem,
    ConsensusLevel,
    MathProblem,
    Z3SolveResponse,
    ProveLeanResponse,
    SolveUnifiedResponse,
    MathSolverMessage
} from './index';

interface MathSolverUIProps {
    onClose?: () => void;
    initialProblem?: string;
}

export const MathSolverUI: React.FC<MathSolverUIProps> = ({ 
    onClose,
    initialProblem = ''
}) => {
    // Core state
    const [core] = useState(() => new MathSolverCore());
    const [state, setState] = useState(core.getState());
    
    // Input state
    const [problemInput, setProblemInput] = useState(initialProblem);
    const [selectedSolver, setSelectedSolver] = useState<SolverSystem>('auto');
    const [consensusLevel, setConsensusLevel] = useState<ConsensusLevel>('confidence');
    const [useKnowledge, setUseKnowledge] = useState(true);
    const [timeout, setTimeout] = useState(300);
    
    // Backend status
    const [backendStatus, setBackendStatus] = useState<{ available: boolean; details?: any } | null>(null);
    const [checkingBackend, setCheckingBackend] = useState(false);

    // Update state from core
    useEffect(() => {
        const updateState = () => setState(core.getState());
        core.on('messageAdded', updateState);
        core.on('solvingStarted', updateState);
        core.on('solvingCompleted', updateState);
        core.on('solvingError', updateState);
        
        // Check backend health on mount
        checkBackend();
        
        return () => {
            // Cleanup if needed
        };
    }, [core]);

    const checkBackend = async () => {
        setCheckingBackend(true);
        const status = await core.checkBackendHealth();
        setBackendStatus(status);
        setCheckingBackend(false);
    };

    const handleSolve = useCallback(async () => {
        if (!problemInput.trim()) return;

        const problem = core.createProblem(problemInput, {
            domain: undefined // Auto-detect
        });

        await core.solve({
            problem,
            preferredSolver: selectedSolver,
            useKnowledgeBase: useKnowledge,
            consensusLevel,
            timeout
        });
    }, [core, problemInput, selectedSolver, useKnowledge, consensusLevel, timeout]);

    const handleClear = useCallback(() => {
        core.reset();
        setProblemInput('');
        setState(core.getState());
    }, [core]);

    const renderMessage = (msg: MathSolverMessage) => {
        const isUser = msg.role === 'user';
        const isError = msg.proofStatus === 'error';
        
        return (
            <div
                key={msg.id}
                className={`math-message ${msg.role} ${isError ? 'error' : ''}`}
                style={{
                    padding: '12px 16px',
                    margin: '8px 0',
                    borderRadius: '8px',
                    backgroundColor: isUser ? '#e3f2fd' : isError ? '#ffebee' : '#f5f5f5',
                    borderLeft: `4px solid ${
                        msg.solverType === 'z3' ? '#2196f3' :
                        msg.solverType === 'lean' ? '#4caf50' :
                        msg.solverType === 'unified' ? '#9c27b0' :
                        isUser ? '#1976d2' : '#757575'
                    }`,
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                }}
            >
                <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '4px' }}>
                    {msg.role.toUpperCase()}
                    {msg.solverType && ` • ${msg.solverType.toUpperCase()}`}
                    {msg.proofStatus && ` • ${msg.proofStatus}`}
                </div>
                <div>{msg.content}</div>
            </div>
        );
    };

    const renderZ3Result = (result: Z3SolveResponse | undefined) => {
        if (!result) return null;
        
        return (
            <div className="solver-result z3-result" style={{ marginTop: '16px', padding: '16px', backgroundColor: '#e3f2fd', borderRadius: '8px' }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#1976d2' }}>Z3 Solver Result</h4>
                <div><strong>Status:</strong> <span style={{ 
                    color: result.status === 'sat' ? '#4caf50' : 
                           result.status === 'unsat' ? '#ff9800' : '#f44336'
                }}>{result.status}</span></div>
                <div><strong>Time:</strong> {result.solving_time_ms}ms</div>
                {result.model && (
                    <div style={{ marginTop: '8px' }}>
                        <strong>Model:</strong>
                        <pre style={{ backgroundColor: '#fff', padding: '8px', borderRadius: '4px', overflow: 'auto' }}>
                            {JSON.stringify(result.model, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        );
    };

    const renderLeanResult = (result: ProveLeanResponse | undefined) => {
        if (!result) return null;
        
        return (
            <div className="solver-result lean-result" style={{ marginTop: '16px', padding: '16px', backgroundColor: '#e8f5e9', borderRadius: '8px' }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#388e3c' }}>Lean Theorem Prover Result</h4>
                <div><strong>Status:</strong> <span style={{
                    color: result.success ? '#4caf50' : '#f44336'
                }}>{result.success ? 'Proved' : 'Failed'}</span></div>
                <div><strong>Time:</strong> {result.execution_time_ms}ms</div>
                {result.proof && (
                    <div style={{ marginTop: '8px' }}>
                        <strong>Proof:</strong>
                        <pre style={{ backgroundColor: '#fff', padding: '8px', borderRadius: '4px', overflow: 'auto' }}>
                            {result.proof}
                        </pre>
                    </div>
                )}
                {result.error && (
                    <div style={{ marginTop: '8px', color: '#f44336' }}>
                        <strong>Error:</strong> {result.error}
                    </div>
                )}
            </div>
        );
    };

    const renderUnifiedResult = (result: SolveUnifiedResponse | undefined) => {
        if (!result) return null;
        
        return (
            <div className="solver-result unified-result" style={{ marginTop: '16px', padding: '16px', backgroundColor: '#f3e5f5', borderRadius: '8px' }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#7b1fa2' }}>Unified Result</h4>
                <div><strong>Status:</strong> {result.result_status}</div>
                <div><strong>Primary Solver:</strong> {result.primary_solver}</div>
                <div><strong>Verified:</strong> {result.verified ? '✓' : '✗'}</div>
                <div><strong>Solving Time:</strong> {result.solving_time_ms}ms</div>
                {result.consensus_status && (
                    <div><strong>Consensus:</strong> {result.consensus_status}</div>
                )}
                {result.result && (
                    <div style={{ marginTop: '8px' }}>
                        <strong>Result Details:</strong>
                        <pre style={{ backgroundColor: '#fff', padding: '8px', borderRadius: '4px', overflow: 'auto' }}>
                            {JSON.stringify(result.result, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="math-solver-ui" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '16px' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h2 style={{ margin: 0 }}>MathSolver</h2>
                <div>
                    <span style={{ 
                        marginRight: '16px',
                        color: checkingBackend ? '#ff9800' : 
                               backendStatus?.available ? '#4caf50' : '#f44336'
                    }}>
                        {checkingBackend ? '● Checking backend...' :
                         backendStatus?.available ? '● Backend connected' : '● Backend unavailable'}
                    </span>
                    <button onClick={checkBackend} disabled={checkingBackend} style={{ marginRight: '8px' }}>
                        Refresh
                    </button>
                    {onClose && <button onClick={onClose}>Close</button>}
                </div>
            </div>

            {/* Configuration */}
            <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', flexWrap: 'wrap' }}>
                <div>
                    <label>Solver:</label>
                    <select 
                        value={selectedSolver} 
                        onChange={(e) => setSelectedSolver(e.target.value as SolverSystem)}
                        style={{ marginLeft: '8px' }}
                    >
                        <option value="auto">Auto-select</option>
                        <option value="z3">Z3 SMT</option>
                        <option value="lean">Lean</option>
                        <option value="unified">Unified (Z3+Lean)</option>
                    </select>
                </div>
                
                <div>
                    <label>Consensus:</label>
                    <select 
                        value={consensusLevel} 
                        onChange={(e) => setConsensusLevel(e.target.value as ConsensusLevel)}
                        style={{ marginLeft: '8px' }}
                    >
                        <option value="strict">Strict</option>
                        <option value="confidence">Confidence</option>
                        <option value="permissive">Permissive</option>
                    </select>
                </div>
                
                <div>
                    <label>
                        <input 
                            type="checkbox" 
                            checked={useKnowledge} 
                            onChange={(e) => setUseKnowledge(e.target.checked)}
                        />
                        Use Knowledge Base
                    </label>
                </div>
                
                <div>
                    <label>Timeout (s):</label>
                    <input 
                        type="number" 
                        value={timeout} 
                        onChange={(e) => setTimeout(parseInt(e.target.value) || 300)}
                        style={{ marginLeft: '8px', width: '80px' }}
                    />
                </div>
            </div>

            {/* Problem Input */}
            <div style={{ marginBottom: '16px' }}>
                <textarea
                    value={problemInput}
                    onChange={(e) => setProblemInput(e.target.value)}
                    placeholder="Enter your mathematical problem or theorem...\n\nExamples:\n- Prove that for all integers n, n² ≥ 0\n- Find x such that x² + 3x + 2 = 0\n- Show that the sum of two even numbers is even"
                    style={{
                        width: '100%',
                        minHeight: '120px',
                        padding: '12px',
                        fontFamily: 'monospace',
                        fontSize: '14px',
                        borderRadius: '8px',
                        border: '1px solid #ddd',
                        resize: 'vertical'
                    }}
                />
                <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                    <button 
                        onClick={handleSolve}
                        disabled={!problemInput.trim() || state.isProcessing || !backendStatus?.available}
                        style={{
                            padding: '10px 24px',
                            backgroundColor: '#1976d2',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: state.isProcessing ? 'not-allowed' : 'pointer',
                            opacity: state.isProcessing ? 0.6 : 1
                        }}
                    >
                        {state.isProcessing ? 'Solving...' : 'Solve'}
                    </button>
                    <button onClick={handleClear} style={{ padding: '10px 24px' }}>
                        Clear
                    </button>
                </div>
            </div>

            {/* Messages/History */}
            <div style={{ 
                flex: 1, 
                overflow: 'auto', 
                border: '1px solid #ddd', 
                borderRadius: '8px',
                padding: '16px',
                backgroundColor: '#fafafa'
            }}>
                {state.messages.length === 0 ? (
                    <div style={{ color: '#999', textAlign: 'center', padding: '40px' }}>
                        Enter a mathematical problem and click Solve to begin.
                        <br /><br />
                        <small>The MathSolver integrates Z3 SMT solver and Lean theorem prover for automated mathematical reasoning.</small>
                    </div>
                ) : (
                    state.messages.map(renderMessage)
                )}
            </div>

            {/* Results Summary */}
            {state.currentProblem && (
                <div style={{ marginTop: '16px' }}>
                    {renderZ3Result(state.z3Results.get(state.currentProblem.id))}
                    {renderLeanResult(state.leanResults.get(state.currentProblem.id))}
                    {renderUnifiedResult(state.unifiedResults.get(state.currentProblem.id))}
                </div>
            )}
        </div>
    );
};

export default MathSolverUI;
