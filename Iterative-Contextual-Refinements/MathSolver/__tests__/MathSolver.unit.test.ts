/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MathSolver Unit Tests
 * 
 * Comprehensive unit tests for MathSolver utility functions and core logic.
 * 
 * API Version: 1.1.0
 */

import {
    formatProofForDisplay,
    detectDomain,
    recommendSolver,
    MathSolverCore,
    SolverSystem,
    MathProblem
} from '../index';

describe('MathSolver Unit Tests', () => {
    
    describe('formatProofForDisplay', () => {
        it('should format Z3 SMT-LIB proof with syntax highlighting markers', () => {
            const proof = `(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)`;
            const result = formatProofForDisplay(proof, 'z3');
            
            expect(result).toContain('<keyword>');
            expect(result).toContain('<comment>');
        });

        it('should format Lean proof with syntax highlighting markers', () => {
            const proof = `theorem test : true := by\n  -- This is a comment\n  trivial`;
            const result = formatProofForDisplay(proof, 'lean');
            
            expect(result).toContain('<keyword>');
            expect(result).toContain('<comment>');
        });

        it('should return proof unchanged for unknown solver', () => {
            const proof = 'some proof text';
            const result = formatProofForDisplay(proof, 'unknown' as SolverSystem);
            
            expect(result).toBe(proof);
        });
    });

    describe('detectDomain', () => {
        it('should detect algebra domain', () => {
            expect(detectDomain('x² + 3x + 2 = 0')).toBe('algebra');
            expect(detectDomain('Solve for x: 2x + 5 = 10')).toBe('algebra');
        });

        it('should detect geometry domain', () => {
            expect(detectDomain('Triangle ABC has angles')).toBe('geometry');
            expect(detectDomain('Circle with radius 5')).toBe('geometry');
        });

        it('should detect calculus domain', () => {
            expect(detectDomain('Find the derivative of x²')).toBe('calculus');
            expect(detectDomain('Calculate the integral')).toBe('calculus');
        });

        it('should detect arithmetic domain', () => {
            expect(detectDomain('Calculate 123 + 456')).toBe('arithmetic');
            expect(detectDomain('Prime number factorization')).toBe('arithmetic');
        });

        it('should detect logic domain', () => {
            expect(detectDomain('Prove that P implies Q')).toBe('logic');
            expect(detectDomain('For all x, if x > 0 then...')).toBe('logic');
        });

        it('should return other for unknown domains', () => {
            expect(detectDomain('Some random text')).toBe('other');
        });
    });

    describe('recommendSolver', () => {
        it('should recommend Z3 for algebra domain', () => {
            const problem: MathProblem = {
                id: 'test-1',
                statement: 'x² + 3x + 2 = 0',
                domain: 'algebra',
                difficulty: 'easy'
            };
            expect(recommendSolver(problem)).toBe('z3');
        });

        it('should recommend Lean for logic domain', () => {
            const problem: MathProblem = {
                id: 'test-2',
                statement: 'Prove that for all n, n + 0 = n',
                domain: 'logic',
                difficulty: 'medium'
            };
            expect(recommendSolver(problem)).toBe('lean');
        });

        it('should recommend Z3 for arithmetic domain', () => {
            const problem: MathProblem = {
                id: 'test-3',
                statement: 'Find prime numbers',
                domain: 'arithmetic',
                difficulty: 'medium'
            };
            expect(recommendSolver(problem)).toBe('z3');
        });

        it('should recommend unified for hard problems', () => {
            const problem: MathProblem = {
                id: 'test-4',
                statement: 'Complex theorem',
                domain: 'algebra',
                difficulty: 'expert'
            };
            expect(recommendSolver(problem)).toBe('unified');
        });
    });

    describe('MathSolverCore - Event System', () => {
        let core: MathSolverCore;

        beforeEach(() => {
            core = new MathSolverCore();
        });

        afterEach(() => {
            core.reset();
        });

        it('should support event subscription and emission', () => {
            const mockCallback = jest.fn();
            core.on('problemCreated', mockCallback);
            
            core.createProblem('test problem');
            
            expect(mockCallback).toHaveBeenCalledTimes(1);
            expect(mockCallback.mock.calls[0][0]).toMatchObject({
                statement: 'test problem'
            });
        });

        it('should support event unsubscription', () => {
            const mockCallback = jest.fn();
            core.on('problemCreated', mockCallback);
            core.off('problemCreated', mockCallback);
            
            core.createProblem('test problem');
            
            expect(mockCallback).not.toHaveBeenCalled();
        });

        it('should support multiple event listeners', () => {
            const callback1 = jest.fn();
            const callback2 = jest.fn();
            
            core.on('problemCreated', callback1);
            core.on('problemCreated', callback2);
            
            core.createProblem('test problem');
            
            expect(callback1).toHaveBeenCalledTimes(1);
            expect(callback2).toHaveBeenCalledTimes(1);
        });

        it('should emit stateImported event on importState', () => {
            const mockCallback = jest.fn();
            core.on('stateImported', mockCallback);
            
            core.importState({ history: [] });
            
            expect(mockCallback).toHaveBeenCalledTimes(1);
        });

        it('should emit stateReset event on reset', () => {
            const mockCallback = jest.fn();
            core.on('stateReset', mockCallback);
            
            core.reset();
            
            expect(mockCallback).toHaveBeenCalledTimes(1);
        });
    });

    describe('MathSolverCore - State Management', () => {
        let core: MathSolverCore;

        beforeEach(() => {
            core = new MathSolverCore();
        });

        it('should generate unique IDs for problems', () => {
            const problem1 = core.createProblem('problem 1');
            const problem2 = core.createProblem('problem 2');
            
            expect(problem1.id).not.toBe(problem2.id);
        });

        it('should maintain problem history', () => {
            core.createProblem('problem 1');
            core.createProblem('problem 2');
            
            const state = core.getState();
            expect(state.history).toHaveLength(2);
        });

        it('should set currentProblem on create', () => {
            const problem = core.createProblem('test problem');
            const state = core.getState();
            
            expect(state.currentProblem).toEqual(problem);
        });

        it('should clear state on reset', () => {
            core.createProblem('test');
            core.reset();
            
            const state = core.getState();
            expect(state.history).toHaveLength(0);
            expect(state.currentProblem).toBeNull();
            expect(state.messages).toHaveLength(0);
        });

        it('should export and import state correctly', () => {
            core.createProblem('test problem');
            const exported = core.exportState();
            
            const newCore = new MathSolverCore();
            newCore.importState(exported);
            
            const importedState = newCore.getState();
            expect(importedState.history).toHaveLength(1);
            expect(importedState.history[0].statement).toBe('test problem');
        });

        it('should export z3Results as array for serialization', () => {
            // Note: This would require mocking API calls to populate results
            const exported = core.exportState();
            
            expect(Array.isArray(exported.z3Results)).toBe(true);
            expect(Array.isArray(exported.leanResults)).toBe(true);
            expect(Array.isArray(exported.unifiedResults)).toBe(true);
        });
    });

    describe('MathSolverCore - Concurrent Solve Protection', () => {
        let core: MathSolverCore;

        beforeEach(() => {
            core = new MathSolverCore();
        });

        it('should prevent concurrent solve requests', async () => {
            // Set processing flag manually to simulate active solve
            const state = core.getState();
            (state as any).isProcessing = true;
            
            const problem = core.createProblem('test');
            
            await expect(core.solve({ problem })).rejects.toThrow(
                'A solve operation is already in progress'
            );
        });
    });

    describe('MathSolverCore - Cancel Solve', () => {
        let core: MathSolverCore;

        beforeEach(() => {
            core = new MathSolverCore();
        });

        it('should return false for isSolving when not processing', () => {
            expect(core.isSolving()).toBe(false);
        });

        it('should emit solvingCancelled event on cancel', () => {
            const mockCallback = jest.fn();
            core.on('solvingCancelled', mockCallback);
            
            // Start a solve (will fail due to no backend, but sets up state)
            const problem = core.createProblem('test');
            
            // Manually set up processing state
            const state = core.getState();
            (state as any).isProcessing = true;
            
            core.cancelSolve();
            
            expect(mockCallback).toHaveBeenCalled();
        });
    });

    describe('Input Validation', () => {
        it('should handle empty problem statement', () => {
            const core = new MathSolverCore();
            const problem = core.createProblem('');
            
            expect(problem.statement).toBe('');
            expect(problem.id).toBeDefined();
        });

        it('should handle very long problem statements', () => {
            const core = new MathSolverCore();
            const longText = 'x'.repeat(10000);
            const problem = core.createProblem(longText);
            
            expect(problem.statement).toBe(longText);
        });

        it('should handle special characters in problem statements', () => {
            const core = new MathSolverCore();
            const specialChars = '∀∃∈∉∧∨→↔¬≤≥≠×÷±∞∫∂∆∑∏√∛∜<>"\'&';
            const problem = core.createProblem(specialChars);
            
            expect(problem.statement).toBe(specialChars);
        });
    });
});

// Performance benchmarks
describe('MathSolver Performance', () => {
    it('should create 1000 problems quickly', () => {
        const core = new MathSolverCore();
        const start = performance.now();
        
        for (let i = 0; i < 1000; i++) {
            core.createProblem(`Problem ${i}`);
        }
        
        const duration = performance.now() - start;
        expect(duration).toBeLessThan(100); // Should complete in less than 100ms
    });

    it('should handle large state export efficiently', () => {
        const core = new MathSolverCore();
        
        // Create many problems
        for (let i = 0; i < 100; i++) {
            core.createProblem(`Problem ${i}`);
        }
        
        const start = performance.now();
        const exported = core.exportState();
        const duration = performance.now() - start;
        
        expect(duration).toBeLessThan(50);
        expect(exported.history).toHaveLength(100);
    });
});

console.log('MathSolver Unit Test Module Loaded (API v1.1.0)');
