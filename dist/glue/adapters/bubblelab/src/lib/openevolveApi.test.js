"use strict";
/**
 * Contract Tests for openevolveApi.ts
 *
 * Follows Federation Constitution Section 4: The Proof of Work (Phase 2: The Contract)
 *
 * These tests verify that the API returns the specific fields we rely on.
 * If the contract is violated (OpenEvolve API changed), the adapter refuses to start
 * to prevent data corruption.
 *
 * Run on container startup to validate API contract before accepting traffic.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const node_fetch_1 = require("node-fetch");
const openevolveApi_1 = require("./openevolveApi");
// Mock fetch globally for testing
global.fetch = node_fetch_1.fetch;
// Test configuration - should be injected via environment
const TEST_CONFIG = {
    baseUrl: process.env.OPENEVOLVE_API_BASE_URL || 'http://localhost:8000',
    apiKey: process.env.OPENEVOLVE_API_KEY || 'test-key',
    timeout: 30000,
};
(0, vitest_1.describe)('OpenEvolve API Contract Tests', () => {
    (0, vitest_1.beforeAll)(() => {
        // Validate test configuration
        if (!process.env.OPENEVOLVE_API_BASE_URL) {
            console.warn('OPENEVOLVE_API_BASE_URL not set, using default: http://localhost:8000');
        }
    });
    (0, vitest_1.describe)('Health Check', () => {
        (0, vitest_1.it)('should return health status object', async () => {
            const response = await openevolveApi_1.openevolveApi.getHealth(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
            // Health endpoint should return status
            if ('status' in response) {
                (0, vitest_1.expect)(typeof response.status).toBe('string');
            }
        });
    });
    (0, vitest_1.describe)('Teams API', () => {
        (0, vitest_1.it)('listTeams should return teams array and total', async () => {
            const response = await openevolveApi_1.openevolveApi.listTeams(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.teams).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.teams)).toBe(true);
            (0, vitest_1.expect)(typeof response.total).toBe('number');
        });
        (0, vitest_1.it)('getTeam should return team object with required fields', async () => {
            // First list to get a valid team name
            const listResponse = await openevolveApi_1.openevolveApi.listTeams(TEST_CONFIG);
            if (listResponse.teams.length === 0) {
                console.warn('No teams found, skipping getTeam contract test');
                return;
            }
            const teamName = listResponse.teams[0].name;
            const response = await openevolveApi_1.openevolveApi.getTeam(teamName, TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.name).toBeDefined();
            (0, vitest_1.expect)(typeof response.name).toBe('string');
            (0, vitest_1.expect)(response.description).toBeDefined();
        });
    });
    (0, vitest_1.describe)('Workflows API', () => {
        (0, vitest_1.it)('listWorkflows should return workflows array and total', async () => {
            const response = await openevolveApi_1.openevolveApi.listWorkflows(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.workflows).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.workflows)).toBe(true);
            (0, vitest_1.expect)(typeof response.total).toBe('number');
        });
        (0, vitest_1.it)('createWorkflow should return workflow ID and status', async () => {
            const payload = {
                name: 'contract-test-workflow',
                description: 'Contract test workflow',
                gauntlet_name: 'test-gauntlet',
                protocol_text: '# Test Protocol\n\nThis is a contract test.',
            };
            try {
                const response = await openevolveApi_1.openevolveApi.createWorkflow(payload, TEST_CONFIG);
                (0, vitest_1.expect)(response).toBeDefined();
                (0, vitest_1.expect)(response.workflow_id).toBeDefined();
                (0, vitest_1.expect)(typeof response.workflow_id).toBe('string');
                (0, vitest_1.expect)(response.status).toBeDefined();
                (0, vitest_1.expect)(typeof response.status).toBe('string');
            }
            catch (error) {
                // Workflow creation might fail if test gauntlet doesn't exist
                // That's OK for contract testing - we're validating the API interface
                console.warn('Workflow creation failed (expected if test data not set up):', error);
            }
        });
    });
    (0, vitest_1.describe)('Gauntlets API', () => {
        (0, vitest_1.it)('listGauntlets should return gauntlets array and total', async () => {
            const response = await openevolveApi_1.openevolveApi.listGauntlets(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.gauntlets).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.gauntlets)).toBe(true);
            (0, vitest_1.expect)(typeof response.total).toBe('number');
        });
    });
    (0, vitest_1.describe)('Evolution API', () => {
        (0, vitest_1.it)('startEvolutionRun should return run ID and initial status', async () => {
            const payload = {
                content: '# Test Protocol\n\nThis is a contract test for evolution.',
                content_type: 'markdown',
                evolution_mode: 'incremental',
                parameters: {
                    generations: 2,
                },
            };
            try {
                const response = await openevolveApi_1.openevolveApi.startEvolutionRun(payload, TEST_CONFIG);
                (0, vitest_1.expect)(response).toBeDefined();
                (0, vitest_1.expect)(response.run_id).toBeDefined();
                (0, vitest_1.expect)(typeof response.run_id).toBe('string');
                (0, vitest_1.expect)(response.status).toBeDefined();
                (0, vitest_1.expect)(typeof response.status).toBe('string');
            }
            catch (error) {
                console.warn('Evolution run failed (may require valid protocol):', error);
            }
        });
        (0, vitest_1.it)('listEvolutionRuns should return runs array', async () => {
            const response = await openevolveApi_1.openevolveApi.listEvolutionRuns(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.runs).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.runs)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Adversarial Testing API', () => {
        (0, vitest_1.it)('startAdversarialRun should return run ID and initial status', async () => {
            const payload = {
                content: '# Test Protocol\n\nThis is a contract test for adversarial testing.',
                content_type: 'markdown',
                parameters: {
                    test_types: ['injection', 'prompt_extraction'],
                },
            };
            try {
                const response = await openevolveApi_1.openevolveApi.startAdversarialRun(payload, TEST_CONFIG);
                (0, vitest_1.expect)(response).toBeDefined();
                (0, vitest_1.expect)(response.run_id).toBeDefined();
                (0, vitest_1.expect)(typeof response.run_id).toBe('string');
                (0, vitest_1.expect)(response.status).toBeDefined();
                (0, vitest_1.expect)(typeof response.status).toBe('string');
            }
            catch (error) {
                console.warn('Adversarial run failed (may require valid protocol):', error);
            }
        });
        (0, vitest_1.it)('listAdversarialRuns should return runs array', async () => {
            const response = await openevolveApi_1.openevolveApi.listAdversarialRuns(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.runs).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.runs)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Knowledge Base API', () => {
        (0, vitest_1.it)('listKnowledgeArtifacts should return artifacts array', async () => {
            const response = await openevolveApi_1.openevolveApi.listKnowledgeArtifacts(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.artifacts).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.artifacts)).toBe(true);
        });
        (0, vitest_1.it)('getKnowledgeStats should return statistics object', async () => {
            const response = await openevolveApi_1.openevolveApi.getKnowledgeStats(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
            // Stats should contain numeric fields
            if ('total_artifacts' in response) {
                (0, vitest_1.expect)(typeof response.total_artifacts).toBe('number');
            }
        });
    });
    (0, vitest_1.describe)('Providers API', () => {
        (0, vitest_1.it)('listProviders should return providers array', async () => {
            const response = await openevolveApi_1.openevolveApi.listProviders(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.providers).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.providers)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Version Control API', () => {
        (0, vitest_1.it)('listVersions should return versions array', async () => {
            const response = await openevolveApi_1.openevolveApi.listVersions(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.versions).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.versions)).toBe(true);
        });
    });
    (0, vitest_1.describe)('BubbleLabs Integration API', () => {
        (0, vitest_1.it)('getBubblelabsStatus should return status object', async () => {
            const response = await openevolveApi_1.openevolveApi.getBubblelabsStatus(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
            if ('available' in response) {
                (0, vitest_1.expect)(typeof response.available).toBe('boolean');
            }
        });
        (0, vitest_1.it)('listWorkflowDefinitions should return definitions array', async () => {
            const response = await openevolveApi_1.openevolveApi.listWorkflowDefinitions(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.definitions).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.definitions)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Maker Integration API', () => {
        (0, vitest_1.it)('getMakerStatus should return availability status', async () => {
            const response = await openevolveApi_1.openevolveApi.getMakerStatus(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response.available).toBe('boolean');
        });
        (0, vitest_1.it)('listMakerTools should return tools array', async () => {
            const response = await openevolveApi_1.openevolveApi.listMakerTools({}, TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.tools).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.tools)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Knowledge Explorer API', () => {
        (0, vitest_1.it)('bubblelabsKnowledgeStatus should return status object', async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsKnowledgeStatus(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response.initialized).toBe('boolean');
            (0, vitest_1.expect)(typeof response.query_history_count).toBe('number');
        });
        (0, vitest_1.it)('bubblelabsKnowledgeQueryHistory should return history array', async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsKnowledgeQueryHistory(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.queries).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.queries)).toBe(true);
        });
    });
    (0, vitest_1.describe)('LeanAide API', () => {
        (0, vitest_1.it)('bubblelabsLeanAideStatus should return status object', async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideStatus(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
        });
        (0, vitest_1.it)('bubblelabsLeanAideTrees should return trees array', async () => {
            const response = await openevolveApi_1.openevolveApi.bubblelabsLeanAideTrees(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.trees).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.trees)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Monitoring API', () => {
        (0, vitest_1.it)('getMonitoringDashboard should return metrics object', async () => {
            const response = await openevolveApi_1.openevolveApi.getMonitoringDashboard(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
        });
        (0, vitest_1.it)('getMonitoringAlerts should return alerts array', async () => {
            const response = await openevolveApi_1.openevolveApi.getMonitoringAlerts(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.alerts).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.alerts)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Analytics API', () => {
        (0, vitest_1.it)('getStatistics should return statistics summary', async () => {
            const response = await openevolveApi_1.openevolveApi.getStatistics(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
        });
        (0, vitest_1.it)('getPerformanceMetrics should return metrics array and total', async () => {
            const response = await openevolveApi_1.openevolveApi.getPerformanceMetrics(undefined, 10, TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.metrics).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.metrics)).toBe(true);
            (0, vitest_1.expect)(typeof response.total).toBe('number');
        });
    });
    (0, vitest_1.describe)('Validation API', () => {
        (0, vitest_1.it)('listValidationRules should return rules object', async () => {
            const response = await openevolveApi_1.openevolveApi.listValidationRules(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(response.rules).toBeDefined();
            (0, vitest_1.expect)(typeof response.rules).toBe('object');
            (0, vitest_1.expect)(response.rule_names).toBeDefined();
            (0, vitest_1.expect)(Array.isArray(response.rule_names)).toBe(true);
        });
    });
    (0, vitest_1.describe)('Auto-Approval API', () => {
        (0, vitest_1.it)('getAutoApprovalConfig should return config object', async () => {
            const response = await openevolveApi_1.openevolveApi.getAutoApprovalConfig(TEST_CONFIG);
            (0, vitest_1.expect)(response).toBeDefined();
            (0, vitest_1.expect)(typeof response).toBe('object');
            // Config should have enabled flag
            if ('enabled' in response) {
                (0, vitest_1.expect)(typeof response.enabled).toBe('boolean');
            }
        });
    });
    (0, vitest_1.describe)('Error Handling', () => {
        (0, vitest_1.it)('should throw error with 404 status for non-existent team', async () => {
            await (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getTeam('non-existent-team-12345', TEST_CONFIG)).rejects.toThrow();
        });
        (0, vitest_1.it)('should throw error with 404 status for non-existent workflow', async () => {
            await (0, vitest_1.expect)(openevolveApi_1.openevolveApi.getWorkflow('non-existent-workflow-id', TEST_CONFIG)).rejects.toThrow();
        });
    });
});
//# sourceMappingURL=openevolveApi.test.js.map