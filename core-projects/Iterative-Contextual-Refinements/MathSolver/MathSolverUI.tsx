<<<<<<< HEAD
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
    MathSolverMessage,
    KnowledgeEngineStatus
} from './MathSolverCore';

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
    const [solverTimeout, setSolverTimeout] = useState(300);
    
    // Backend status
    const [backendStatus, setBackendStatus] = useState<{ available: boolean; versionCompatible?: boolean; versionError?: string; details?: any } | null>(null);
    const [checkingBackend, setCheckingBackend] = useState(false);
    
    // Knowledge engine status
    const [knowledgeStatus, setKnowledgeStatus] = useState<KnowledgeEngineStatus | null>(null);
    const [checkingKnowledge, setCheckingKnowledge] = useState(false);
    
    // Refs
    const solveButtonRef = React.useRef<HTMLButtonElement>(null);
    const isMountedRef = React.useRef(true);
    
    useEffect(() => {
        return () => {
            isMountedRef.current = false;
        };
    }, []);

    // Update state from core
    useEffect(() => {
        const updateState = () => {
            if (isMountedRef.current) {
                setState(core.getState());
            }
        };
        core.on('messageAdded', updateState);
        core.on('solvingStarted', updateState);
        core.on('solvingCompleted', updateState);
        core.on('solvingError', updateState);
        core.on('solvingCancelled', updateState);
        
        // Check backend health on mount
        checkBackend();
        
        // Check knowledge engine availability
        checkKnowledgeEngine();
        
        return () => {
            // Clean up event listeners to prevent memory leaks
            core.off('messageAdded', updateState);
            core.off('solvingStarted', updateState);
            core.off('solvingCompleted', updateState);
            core.off('solvingError', updateState);
            core.off('solvingCancelled', updateState);
        };
    }, [core]);
    
    // Keyboard shortcuts in separate effect to avoid stale closures
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Escape to cancel solving
            if (e.key === 'Escape' && core.isSolving()) {
                e.preventDefault();
                core.cancelSolve();
            }
            // Ctrl+Enter to solve
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !core.isSolving()) {
                e.preventDefault();
                // Use a ref to avoid stale closure
                solveButtonRef.current?.click();
            }
        };
        
        document.addEventListener('keydown', handleKeyDown);
        
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [core]);

    const checkBackend = async (retryCount = 0) => {
        setCheckingBackend(true);
        try {
            const status = await core.checkBackendHealth();
            if (isMountedRef.current) {
                setBackendStatus(status);
            }
        } catch (error) {
            console.error('[MathSolverUI] Backend health check failed:', error);
            // Retry up to 2 times
            if (retryCount < 2 && isMountedRef.current) {
                setTimeout(() => checkBackend(retryCount + 1), 1000 * (retryCount + 1));
                return;
            }
            if (isMountedRef.current) {
                setBackendStatus({ available: false, versionCompatible: false });
            }
        } finally {
            if (isMountedRef.current) {
                setCheckingBackend(false);
            }
        }
    };
    
    const checkKnowledgeEngine = async () => {
        setCheckingKnowledge(true);
        try {
            await core.checkKnowledgeEngineAvailability();
            if (isMountedRef.current) {
                setKnowledgeStatus(core.getKnowledgeEngineStatus());
            }
        } catch (error) {
            console.error('[MathSolverUI] Knowledge engine check failed:', error);
            if (isMountedRef.current) {
                setKnowledgeStatus({
                    available: false,
                    lastChecked: Date.now(),
                    error: error instanceof Error ? error.message : 'Unknown error'
                });
            }
        } finally {
            if (isMountedRef.current) {
                setCheckingKnowledge(false);
            }
        }
    };

    const handleSolve = useCallback(async () => {
        if (!problemInput.trim()) return;

        try {
            const problem = core.createProblem(problemInput, {
                domain: undefined // Auto-detect
            });

            await core.solve({
                problem,
                preferredSolver: selectedSolver,
                useKnowledgeBase: useKnowledge,
                consensusLevel,
                timeout: solverTimeout
            });
        } catch (error) {
            // Error is already handled by MathSolverCore and emitted via solvingError
            // This catch prevents unhandled promise rejection
            console.log('[MathSolverUI] Solve error caught (handled by core):', error);
        }
    }, [core, problemInput, selectedSolver, useKnowledge, consensusLevel, solverTimeout]);

    const handleClear = useCallback(() => {
        core.reset();
        setProblemInput('');
        setSelectedSolver('auto');
        setConsensusLevel('confidence');
        setUseKnowledge(true);
        setSolverTimeout(300);
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
                    <span 
                        style={{ 
                            marginRight: '16px',
                            color: checkingKnowledge ? '#ff9800' : 
                                   knowledgeStatus?.available === true ? '#4caf50' : 
                                   knowledgeStatus?.available === false ? '#f44336' : '#757575'
                        }}
                        title={knowledgeStatus?.error || 'Knowledge engine status'}
                    >
                        {checkingKnowledge ? '● KB...' :
                         knowledgeStatus?.available === true ? '● KB ✓' : 
                         knowledgeStatus?.available === false ? '● KB ✗' : '● KB ?'}
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
                    <label htmlFor="math-solver-select">Solver:</label>
                    <select 
                        id="math-solver-select"
                        value={selectedSolver} 
                        onChange={(e) => setSelectedSolver(e.target.value as SolverSystem)}
                        disabled={state.isProcessing}
                        aria-label="Select solver type"
                        style={{ marginLeft: '8px' }}
                    >
                        <option value="auto">Auto-select</option>
                        <option value="z3">Z3 SMT</option>
                        <option value="lean">Lean</option>
                        <option value="unified">Unified (Z3+Lean)</option>
                    </select>
                </div>
                
                <div>
                    <label htmlFor="math-consensus-select">Consensus:</label>
                    <select 
                        id="math-consensus-select"
                        value={consensusLevel} 
                        onChange={(e) => setConsensusLevel(e.target.value as ConsensusLevel)}
                        disabled={state.isProcessing}
                        aria-label="Select consensus level"
                        style={{ marginLeft: '8px' }}
                    >
                        <option value="strict">Strict</option>
                        <option value="confidence">Confidence</option>
                        <option value="permissive">Permissive</option>
                    </select>
                </div>
                
                <div>
                    <label style={{ 
                        opacity: knowledgeStatus?.available === false ? 0.6 : 1,
                        cursor: knowledgeStatus?.available === false ? 'not-allowed' : 'pointer'
                    }}>
                        <input 
                            type="checkbox" 
                            checked={useKnowledge && knowledgeStatus?.available !== false} 
                            onChange={(e) => setUseKnowledge(e.target.checked)}
                            disabled={knowledgeStatus?.available === false}
                        />
                        Use Knowledge Base
                        {knowledgeStatus?.available === false && (
                            <span style={{ 
                                color: '#f44336', 
                                fontSize: '0.75rem', 
                                marginLeft: '8px' 
                            }}>
                                (Unavailable)
                            </span>
                        )}
                        {knowledgeStatus?.available === true && (
                            <span style={{ 
                                color: '#4caf50', 
                                fontSize: '0.75rem', 
                                marginLeft: '8px' 
                            }}>
                                (Available)
                            </span>
                        )}
                    </label>
                </div>
                
                <div>
                    <label htmlFor="math-timeout-input">Timeout (s):</label>
                    <input 
                        id="math-timeout-input"
                        type="number" 
                        value={solverTimeout} 
                        onChange={(e) => setSolverTimeout(parseInt(e.target.value, 10) || 300)}
                        disabled={state.isProcessing}
                        aria-label="Solver timeout in seconds"
                        style={{ marginLeft: '8px', width: '80px' }}
                    />
                </div>
            </div>

            {/* Problem Input */}
            <div style={{ marginBottom: '16px' }}>
                <label htmlFor="math-problem-input" style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                    Mathematical Problem:
                </label>
                <textarea
                    id="math-problem-input"
                    value={problemInput}
                    onChange={(e) => setProblemInput(e.target.value)}
                    placeholder="Enter your mathematical problem or theorem...\n\nExamples:\n- Prove that for all integers n, n² ≥ 0\n- Find x such that x² + 3x + 2 = 0\n- Show that the sum of two even numbers is even"
                    disabled={state.isProcessing}
                    aria-label="Mathematical problem input"
                    aria-describedby="math-problem-help"
                    style={{
                        width: '100%',
                        minHeight: '120px',
                        padding: '12px',
                        fontFamily: 'monospace',
                        fontSize: '14px',
                        borderRadius: '8px',
                        border: '1px solid #ddd',
                        resize: 'vertical',
                        opacity: state.isProcessing ? 0.6 : 1
                    }}
                />
                <div id="math-problem-help" style={{ fontSize: '0.875rem', color: '#666', marginTop: '4px' }}>
                    Supports natural language mathematical statements, theorems, and equations.
                </div>
                <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                    <button 
                        ref={solveButtonRef}
                        onClick={handleSolve}
                        disabled={!problemInput.trim() || state.isProcessing || !backendStatus?.available}
                        aria-label={state.isProcessing ? 'Solving in progress' : 'Solve mathematical problem'}
                        aria-busy={state.isProcessing}
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
                    {state.isProcessing && (
                        <button 
                            onClick={() => core.cancelSolve()}
                            aria-label="Cancel solving"
                            style={{ 
                                padding: '10px 24px',
                                backgroundColor: '#dc2626',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Stop
                        </button>
                    )}
                    <button 
                        onClick={handleClear}
                        aria-label="Clear problem and reset"
                        disabled={state.isProcessing}
                        style={{ 
                            padding: '10px 24px',
                            opacity: state.isProcessing ? 0.6 : 1,
                            cursor: state.isProcessing ? 'not-allowed' : 'pointer'
                        }}
                    >
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
=======
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
} from './MathSolverCore';

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
    const [solverTimeout, setSolverTimeout] = useState(300);
    
    // Backend status
    const [backendStatus, setBackendStatus] = useState<{ available: boolean; versionCompatible?: boolean; versionError?: string; details?: any } | null>(null);
    const [checkingBackend, setCheckingBackend] = useState(false);
    
    // Refs
    const solveButtonRef = React.useRef<HTMLButtonElement>(null);
    const isMountedRef = React.useRef(true);
    
    useEffect(() => {
        return () => {
            isMountedRef.current = false;
        };
    }, []);

    // Update state from core
    useEffect(() => {
        const updateState = () => {
            if (isMountedRef.current) {
                setState(core.getState());
            }
        };
        core.on('messageAdded', updateState);
        core.on('solvingStarted', updateState);
        core.on('solvingCompleted', updateState);
        core.on('solvingError', updateState);
        core.on('solvingCancelled', updateState);
        
        // Check backend health on mount
        checkBackend();
        
        return () => {
            // Clean up event listeners to prevent memory leaks
            core.off('messageAdded', updateState);
            core.off('solvingStarted', updateState);
            core.off('solvingCompleted', updateState);
            core.off('solvingError', updateState);
            core.off('solvingCancelled', updateState);
        };
    }, [core]);
    
    // Keyboard shortcuts in separate effect to avoid stale closures
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Escape to cancel solving
            if (e.key === 'Escape' && core.isSolving()) {
                e.preventDefault();
                core.cancelSolve();
            }
            // Ctrl+Enter to solve
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !core.isSolving()) {
                e.preventDefault();
                // Use a ref to avoid stale closure
                solveButtonRef.current?.click();
            }
        };
        
        document.addEventListener('keydown', handleKeyDown);
        
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [core]);

    const checkBackend = async (retryCount = 0) => {
        setCheckingBackend(true);
        try {
            const status = await core.checkBackendHealth();
            if (isMountedRef.current) {
                setBackendStatus(status);
            }
        } catch (error) {
            console.error('[MathSolverUI] Backend health check failed:', error);
            // Retry up to 2 times
            if (retryCount < 2 && isMountedRef.current) {
                setTimeout(() => checkBackend(retryCount + 1), 1000 * (retryCount + 1));
                return;
            }
            if (isMountedRef.current) {
                setBackendStatus({ available: false, versionCompatible: false });
            }
        } finally {
            if (isMountedRef.current) {
                setCheckingBackend(false);
            }
        }
    };

    const handleSolve = useCallback(async () => {
        if (!problemInput.trim()) return;

        try {
            const problem = core.createProblem(problemInput, {
                domain: undefined // Auto-detect
            });

            await core.solve({
                problem,
                preferredSolver: selectedSolver,
                useKnowledgeBase: useKnowledge,
                consensusLevel,
                timeout: solverTimeout
            });
        } catch (error) {
            // Error is already handled by MathSolverCore and emitted via solvingError
            // This catch prevents unhandled promise rejection
            console.log('[MathSolverUI] Solve error caught (handled by core):', error);
        }
    }, [core, problemInput, selectedSolver, useKnowledge, consensusLevel, solverTimeout]);

    const handleClear = useCallback(() => {
        core.reset();
        setProblemInput('');
        setSelectedSolver('auto');
        setConsensusLevel('confidence');
        setUseKnowledge(true);
        setSolverTimeout(300);
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
                    <label htmlFor="math-solver-select">Solver:</label>
                    <select 
                        id="math-solver-select"
                        value={selectedSolver} 
                        onChange={(e) => setSelectedSolver(e.target.value as SolverSystem)}
                        disabled={state.isProcessing}
                        aria-label="Select solver type"
                        style={{ marginLeft: '8px' }}
                    >
                        <option value="auto">Auto-select</option>
                        <option value="z3">Z3 SMT</option>
                        <option value="lean">Lean</option>
                        <option value="unified">Unified (Z3+Lean)</option>
                    </select>
                </div>
                
                <div>
                    <label htmlFor="math-consensus-select">Consensus:</label>
                    <select 
                        id="math-consensus-select"
                        value={consensusLevel} 
                        onChange={(e) => setConsensusLevel(e.target.value as ConsensusLevel)}
                        disabled={state.isProcessing}
                        aria-label="Select consensus level"
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
                    <label htmlFor="math-timeout-input">Timeout (s):</label>
                    <input 
                        id="math-timeout-input"
                        type="number" 
                        value={solverTimeout} 
                        onChange={(e) => setSolverTimeout(parseInt(e.target.value, 10) || 300)}
                        disabled={state.isProcessing}
                        aria-label="Solver timeout in seconds"
                        style={{ marginLeft: '8px', width: '80px' }}
                    />
                </div>
            </div>

            {/* Problem Input */}
            <div style={{ marginBottom: '16px' }}>
                <label htmlFor="math-problem-input" style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                    Mathematical Problem:
                </label>
                <textarea
                    id="math-problem-input"
                    value={problemInput}
                    onChange={(e) => setProblemInput(e.target.value)}
                    placeholder="Enter your mathematical problem or theorem...\n\nExamples:\n- Prove that for all integers n, n² ≥ 0\n- Find x such that x² + 3x + 2 = 0\n- Show that the sum of two even numbers is even"
                    disabled={state.isProcessing}
                    aria-label="Mathematical problem input"
                    aria-describedby="math-problem-help"
                    style={{
                        width: '100%',
                        minHeight: '120px',
                        padding: '12px',
                        fontFamily: 'monospace',
                        fontSize: '14px',
                        borderRadius: '8px',
                        border: '1px solid #ddd',
                        resize: 'vertical',
                        opacity: state.isProcessing ? 0.6 : 1
                    }}
                />
                <div id="math-problem-help" style={{ fontSize: '0.875rem', color: '#666', marginTop: '4px' }}>
                    Supports natural language mathematical statements, theorems, and equations.
                </div>
                <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                    <button 
                        ref={solveButtonRef}
                        onClick={handleSolve}
                        disabled={!problemInput.trim() || state.isProcessing || !backendStatus?.available}
                        aria-label={state.isProcessing ? 'Solving in progress' : 'Solve mathematical problem'}
                        aria-busy={state.isProcessing}
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
                    {state.isProcessing && (
                        <button 
                            onClick={() => core.cancelSolve()}
                            aria-label="Cancel solving"
                            style={{ 
                                padding: '10px 24px',
                                backgroundColor: '#dc2626',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Stop
                        </button>
                    )}
                    <button 
                        onClick={handleClear}
                        aria-label="Clear problem and reset"
                        disabled={state.isProcessing}
                        style={{ 
                            padding: '10px 24px',
                            opacity: state.isProcessing ? 0.6 : 1,
                            cursor: state.isProcessing ? 'not-allowed' : 'pointer'
                        }}
                    >
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
>>>>>>> 5eda1a20fcb6c8612f843e21628e85c5f3699f23
