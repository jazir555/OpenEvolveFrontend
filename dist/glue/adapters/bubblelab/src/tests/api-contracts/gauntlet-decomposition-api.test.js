"use strict";
/**
 * API Contract Tests for Gauntlet and Decomposition Endpoints
 *
 * Tests the API client contract to ensure all endpoints are properly defined
 * and follow the expected structure. Following CLAUDE.md Law of Runtime Truth.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const openevolveApi_1 = require("../../lib/openevolveApi");
(0, vitest_1.describe)('Gauntlet Execution API Contract Tests', () => {
    (0, vitest_1.describe)('executeGauntlet', () => {
        (0, vitest_1.it)('should have executeGauntlet endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.executeGauntlet).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.executeGauntlet).toBe('function');
        });
        (0, vitest_1.it)('should accept gauntlet name and payload', () => {
            const endpoint = openevolveApi_1.openevolveApi.executeGauntlet;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(2); // gauntletName, payload, optional config
        });
        (0, vitest_1.it)('should have correct signature', () => {
            const payload = {
                content: 'Test content',
                content_type: 'text_general',
                evolution_mode: 'standard',
                parameters: { max_iterations: 3 }
            };
            // Should not throw on type checking
            (0, vitest_1.expect)(() => {
                openevolveApi_1.openevolveApi.executeGauntlet('test-gauntlet', payload);
            }).not.toThrow();
        });
    });
    (0, vitest_1.describe)('getGauntletExecutionStatus', () => {
        (0, vitest_1.it)('should have getGauntletExecutionStatus endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getGauntletExecutionStatus).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.getGauntletExecutionStatus).toBe('function');
        });
        (0, vitest_1.it)('should accept execution ID', () => {
            const endpoint = openevolveApi_1.openevolveApi.getGauntletExecutionStatus;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(1); // executionId, optional config
        });
    });
    (0, vitest_1.describe)('listGauntletExecutions', () => {
        (0, vitest_1.it)('should have listGauntletExecutions endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.listGauntletExecutions).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.listGauntletExecutions).toBe('function');
        });
        (0, vitest_1.it)('should accept optional gauntlet name filter', () => {
            const endpoint = openevolveApi_1.openevolveApi.listGauntletExecutions;
            (0, vitest_1.expect)(() => {
                endpoint();
                endpoint({ apiKey: 'test' });
                endpoint({ apiKey: 'test' }, 'test-gauntlet');
            }).not.toThrow();
        });
    });
});
(0, vitest_1.describe)('Decomposition Execution API Contract Tests', () => {
    (0, vitest_1.describe)('executeDecomposition', () => {
        (0, vitest_1.it)('should have executeDecomposition endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.executeDecomposition).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.executeDecomposition).toBe('function');
        });
        (0, vitest_1.it)('should accept workflow ID and payload', () => {
            const endpoint = openevolveApi_1.openevolveApi.executeDecomposition;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(2); // workflowId, payload, optional config
            const payload = {
                problem_statement: 'Test problem',
                decomposition_method: 'hierarchical',
                granularity: 'medium',
                max_depth: 3,
                max_sub_problems: 5
            };
            (0, vitest_1.expect)(() => {
                openevolveApi_1.openevolveApi.executeDecomposition('workflow-123', payload);
            }).not.toThrow();
        });
    });
    (0, vitest_1.describe)('getDecompositionExecutionStatus', () => {
        (0, vitest_1.it)('should have getDecompositionExecutionStatus endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getDecompositionExecutionStatus).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.getDecompositionExecutionStatus).toBe('function');
        });
        (0, vitest_1.it)('should accept execution ID', () => {
            const endpoint = openevolveApi_1.openevolveApi.getDecompositionExecutionStatus;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(1);
        });
    });
    (0, vitest_1.describe)('listDecompositionExecutions', () => {
        (0, vitest_1.it)('should have listDecompositionExecutions endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.listDecompositionExecutions).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.listDecompositionExecutions).toBe('function');
        });
        (0, vitest_1.it)('should accept optional workflow ID filter', () => {
            const endpoint = openevolveApi_1.openevolveApi.listDecompositionExecutions;
            (0, vitest_1.expect)(() => {
                endpoint();
                endpoint({ apiKey: 'test' });
                endpoint({ apiKey: 'test' }, 'workflow-123');
            }).not.toThrow();
        });
    });
});
(0, vitest_1.describe)('Workflow Template Execution API Contract Tests', () => {
    (0, vitest_1.describe)('executeWorkflowTemplate', () => {
        (0, vitest_1.it)('should have executeWorkflowTemplate endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.executeWorkflowTemplate).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.executeWorkflowTemplate).toBe('function');
        });
        (0, vitest_1.it)('should accept template ID and payload', () => {
            const endpoint = openevolveApi_1.openevolveApi.executeWorkflowTemplate;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(2);
            const payload = {
                parameters: {
                    gauntlet_name: 'test-gauntlet',
                    content_value: 'Test content'
                },
                callback_url: 'https://example.com/callback'
            };
            (0, vitest_1.expect)(() => {
                openevolveApi_1.openevolveApi.executeWorkflowTemplate('gauntlet-execution', payload);
            }).not.toThrow();
        });
    });
    (0, vitest_1.describe)('getWorkflowTemplateExecutionStatus', () => {
        (0, vitest_1.it)('should have getWorkflowTemplateExecutionStatus endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getWorkflowTemplateExecutionStatus).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.getWorkflowTemplateExecutionStatus).toBe('function');
        });
        (0, vitest_1.it)('should accept execution ID', () => {
            const endpoint = openevolveApi_1.openevolveApi.getWorkflowTemplateExecutionStatus;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(1);
        });
    });
    (0, vitest_1.describe)('stopWorkflowTemplateExecution', () => {
        (0, vitest_1.it)('should have stopWorkflowTemplateExecution endpoint defined', () => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi.stopWorkflowTemplateExecution).toBeDefined();
            (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.stopWorkflowTemplateExecution).toBe('function');
        });
        (0, vitest_1.it)('should accept execution ID', () => {
            const endpoint = openevolveApi_1.openevolveApi.stopWorkflowTemplateExecution;
            (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(1);
        });
    });
});
(0, vitest_1.describe)('Unified Execution Status API Contract Tests', () => {
    (0, vitest_1.it)('should have getExecutionStatus endpoint defined', () => {
        (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getExecutionStatus).toBeDefined();
        (0, vitest_1.expect)(typeof openevolveApi_1.openevolveApi.getExecutionStatus).toBe('function');
    });
    (0, vitest_1.it)('should accept execution type and execution ID', () => {
        const endpoint = openevolveApi_1.openevolveApi.getExecutionStatus;
        (0, vitest_1.expect)(endpoint.length).toBeGreaterThanOrEqual(2);
        (0, vitest_1.expect)(() => {
            openevolveApi_1.openevolveApi.getExecutionStatus('gauntlet', 'exec-123');
            openevolveApi_1.openevolveApi.getExecutionStatus('decomposition', 'exec-456');
            openevolveApi_1.openevolveApi.getExecutionStatus('workflow-template', 'exec-789');
        }).not.toThrow();
    });
    (0, vitest_1.it)('should reject invalid execution types', () => {
        const endpoint = openevolveApi_1.openevolveApi.getExecutionStatus;
        (0, vitest_1.expect)(() => {
            // @ts-expect-error - testing invalid type
            endpoint('invalid-type', 'exec-123');
        }).toThrow();
    });
});
(0, vitest_1.describe)('API Endpoint Response Type Contracts', () => {
    (0, vitest_1.it)('should have EvolutionRunResponse type for gauntlet execution', () => {
        // Type assertion - will fail at compile time if wrong
        const assertType = (_value) => {
            return true;
        };
        (0, vitest_1.expect)(assertType).toBeDefined();
    });
    (0, vitest_1.it)('should have decomposition execution response type', () => {
        const assertType = (_value) => {
            return true;
        };
        (0, vitest_1.expect)(assertType).toBeDefined();
    });
    (0, vitest_1.it)('should have workflow template execution response type', () => {
        const assertType = (_value) => {
            return true;
        };
        (0, vitest_1.expect)(assertType).toBeDefined();
    });
});
(0, vitest_1.describe)('OpenEvolve API Adapter Extension Contract Tests', () => {
    // Note: These tests verify that the adapter has the right methods
    // Actual execution tests would require mocking the API
    (0, vitest_1.it)('should have gauntlet management methods', () => {
        const requiredMethods = [
            'getGauntlet',
            'createGauntlet',
            'updateGauntlet'
        ];
        // These methods are accessed through the adapter, not directly
        // The contract is that they exist in the API client
        requiredMethods.forEach(method => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi[method]).toBeDefined();
        });
    });
    (0, vitest_1.it)('should have gauntlet execution methods', () => {
        const requiredMethods = [
            'executeGauntlet',
            'getGauntletExecutionStatus'
        ];
        requiredMethods.forEach(method => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi[method]).toBeDefined();
        });
    });
    (0, vitest_1.it)('should have decomposition execution methods', () => {
        const requiredMethods = [
            'executeDecomposition',
            'getDecompositionExecutionStatus'
        ];
        requiredMethods.forEach(method => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi[method]).toBeDefined();
        });
    });
    (0, vitest_1.it)('should have workflow management methods', () => {
        const requiredMethods = [
            'createWorkflow',
            'getWorkflowPlan',
            'getWorkflowResults',
            'startEvolutionRun',
            'getEvolutionRun'
        ];
        requiredMethods.forEach(method => {
            (0, vitest_1.expect)(openevolveApi_1.openevolveApi[method]).toBeDefined();
        });
    });
});
//# sourceMappingURL=gauntlet-decomposition-api.test.js.map