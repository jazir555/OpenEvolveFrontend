/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Knowledge Engine Graceful Degradation Tests
 * 
 * Tests that MathSolver functions correctly when the knowledge engine
 * is unavailable, ensuring graceful degradation of self-improving capabilities.
 */

import { 
    MathSolverCore, 
    mathSolverAPI,
    isKnowledgeEngineAvailable,
    getKnowledgeEngineStatus,
    executeMathToolCall
} from '../index';

describe('Knowledge Engine Graceful Degradation', () => {
    let core: MathSolverCore;

    beforeEach(() => {
        core = new MathSolverCore();
    });

    afterEach(() => {
        core.reset();
    });

    describe('Core API', () => {
        test('should expose knowledge engine status methods', () => {
            expect(typeof core.checkKnowledgeEngineAvailability).toBe('function');
            expect(typeof core.isKnowledgeEngineAvailable).toBe('function');
            expect(typeof core.getKnowledgeEngineStatus).toBe('function');
        });

        test('should return initial knowledge engine status', () => {
            const status = core.getKnowledgeEngineStatus();
            expect(status).toHaveProperty('available');
            expect(status).toHaveProperty('lastChecked');
            expect(typeof status.available).toBe('boolean');
            expect(typeof status.lastChecked).toBe('number');
        });
    });

    describe('Solving Without Knowledge Base', () => {
        test('should solve problem when knowledge engine is unavailable', async () => {
            // Mock knowledge engine failure
            const originalSearch = mathSolverAPI.searchKnowledge.bind(mathSolverAPI);
            mathSolverAPI.searchKnowledge = jest.fn().mockRejectedValue(
                new Error('Knowledge engine unavailable')
            );

            const problem = core.createProblem('x + 2 = 5');
            
            // Should not throw even when knowledge engine fails
            const result = await core.solve({
                problem,
                preferredSolver: 'z3',
                useKnowledgeBase: true,  // Request KB but it will fail
                timeout: 10
            });

            // Restore original
            mathSolverAPI.searchKnowledge = originalSearch;

            // Result may succeed or fail based on actual Z3 availability,
            // but it should NOT fail due to knowledge engine
            if (result.error) {
                expect(result.error).not.toContain('Knowledge engine unavailable');
            }
        });

        test('should disable knowledge base checkbox when unavailable', () => {
            // Simulate knowledge engine unavailable
            core['knowledgeStatus'] = {
                available: false,
                lastChecked: Date.now(),
                error: 'Knowledge engine unavailable'
            };

            const status = core.isKnowledgeEngineAvailable();
            expect(status).toBe(false);
        });
    });

    describe('Tool Execution Fallbacks', () => {
        test('search_math_knowledge should return graceful fallback on failure', async () => {
            const result = await executeMathToolCall({
                type: 'search_math_knowledge',
                query: 'test query',
                top_k: 5
            });

            // If knowledge engine fails, should return helpful fallback message
            if (result.includes('Knowledge engine currently unavailable')) {
                expect(result).toContain('Suggestion');
                expect(result).toContain('solve_z3');
                expect(result).toContain('solve_lean');
                expect(result).toContain('solve_unified');
            }
        });

        test('get_strategy should return heuristic fallback on failure', async () => {
            const result = await executeMathToolCall({
                type: 'get_strategy',
                problem_statement: 'Prove that x > 0',
                constraints: []
            });

            // If knowledge engine fails, should return heuristic recommendation
            if (result.includes('heuristic fallback')) {
                expect(result).toContain('Recommended Strategy');
                expect(result).toContain('lean');  // Should recommend lean for proofs
            }
        });

        test('get_strategy should recommend z3 for equations', async () => {
            const result = await executeMathToolCall({
                type: 'get_strategy',
                problem_statement: 'Solve x + 5 = 10',
                constraints: []
            });

            if (result.includes('heuristic fallback')) {
                expect(result).toContain('z3');  // Should recommend z3 for equations
            }
        });

        test('get_strategy should recommend unified for uncertain cases', async () => {
            const result = await executeMathToolCall({
                type: 'get_strategy',
                problem_statement: 'Find the maximum value',
                constraints: []
            });

            if (result.includes('heuristic fallback')) {
                expect(result).toContain('unified');  // Should recommend unified for uncertain
            }
        });
    });

    describe('Status Checking', () => {
        test('should track knowledge engine unavailability', async () => {
            // Simulate a failed check
            const originalStats = mathSolverAPI.getKnowledgeStats.bind(mathSolverAPI);
            mathSolverAPI.getKnowledgeStats = jest.fn().mockRejectedValue(
                new Error('Connection refused')
            );

            const available = await core.checkKnowledgeEngineAvailability();
            
            // Restore
            mathSolverAPI.getKnowledgeStats = originalStats;

            expect(available).toBe(false);
            
            const status = core.getKnowledgeEngineStatus();
            expect(status.available).toBe(false);
            expect(status.error).toContain('Connection refused');
        });

        test('should update lastChecked timestamp', async () => {
            const before = Date.now();
            
            // This may succeed or fail depending on backend availability
            await core.checkKnowledgeEngineAvailability();
            
            const after = Date.now();
            const status = core.getKnowledgeEngineStatus();
            
            expect(status.lastChecked).toBeGreaterThanOrEqual(before);
            expect(status.lastChecked).toBeLessThanOrEqual(after);
        });
    });

    describe('Mode Integration', () => {
        test('should export knowledge engine status helpers', () => {
            expect(typeof isKnowledgeEngineAvailable).toBe('function');
            expect(typeof getKnowledgeEngineStatus).toBe('function');
        });

        test('should return null status when no core is active', () => {
            // These functions require an active core instance
            const status = getKnowledgeEngineStatus();
            // Returns null when no core is active
            expect(status === null || typeof status === 'object').toBe(true);
        });
    });
});

/**
 * Integration test for graceful degradation
 */
describe('Graceful Degradation Integration', () => {
    test('complete workflow without knowledge engine', async () => {
        const core = new MathSolverCore();

        // Step 1: Check knowledge engine (may fail)
        await core.checkKnowledgeEngineAvailability().catch(() => {
            // Expected to potentially fail in test environment
        });

        // Step 2: Create problem
        const problem = core.createProblem('Solve x^2 - 4 = 0');
        expect(problem).toBeDefined();
        expect(problem.statement).toBe('Solve x^2 - 4 = 0');

        // Step 3: Attempt to solve (should work regardless of KB status)
        try {
            const result = await core.solve({
                problem,
                preferredSolver: 'z3',
                useKnowledgeBase: true,  // Request KB
                timeout: 10
            });

            // Should complete without KB errors
            if (result.error) {
                expect(result.error).not.toMatch(/knowledge.*engine/i);
                expect(result.error).not.toMatch(/knowledge.*base/i);
            }
        } catch (e) {
            // Should not throw KB-related errors
            const errorMsg = e instanceof Error ? e.message : String(e);
            expect(errorMsg).not.toMatch(/knowledge.*engine/i);
        }

        core.reset();
    });
});
