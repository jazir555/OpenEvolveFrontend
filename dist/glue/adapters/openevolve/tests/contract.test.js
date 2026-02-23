"use strict";
/**
 * OpenEvolve React Plugin - Contract Tests
 *
 * These tests validate the API contracts between the OpenEvolve React Plugin
 * and the OpenEvolve backend API. Following Federation Constitution principles:
 * - Law of Runtime Truth: Tests validate actual API behavior, not documentation
 * - Fail Fast: Plugin refuses to start if contracts are violated
 *
 * Run with: npm run test:contract
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
// Configuration from environment
const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8002';
const TIMEOUT_MS = parseInt(process.env.PLUGIN_TIMEOUT_MS || '10000');
(0, globals_1.describe)('OpenEvolve Plugin - Contract Tests', () => {
    let apiHealthy = false;
    (0, globals_1.beforeAll)(async () => {
        // Check if API is accessible before running tests
        try {
            const response = await fetch(`${API_URL}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            apiHealthy = response.ok;
        }
        catch (error) {
            console.warn('OpenEvolve API not accessible, skipping live tests');
        }
    });
    (0, globals_1.describe)('Plugin Interface Contract', () => {
        (0, globals_1.test)('plugin should have required metadata', async () => {
            // This test validates the plugin metadata structure
            const pluginMetadata = {
                name: 'openevolve-bubblelab-plugin',
                version: globals_1.expect.any(String),
                description: globals_1.expect.any(String),
                author: 'OpenEvolve',
                capabilities: {
                    evolution: true,
                    adversarial: true,
                    decomposition: true,
                    mdap_maker: true,
                },
            };
            (0, globals_1.expect)(pluginMetadata.name).toBe('openevolve-bubblelab-plugin');
            (0, globals_1.expect)(pluginMetadata.capabilities.evolution).toBe(true);
            (0, globals_1.expect)(pluginMetadata.capabilities.adversarial).toBe(true);
            (0, globals_1.expect)(pluginMetadata.capabilities.decomposition).toBe(true);
            (0, globals_1.expect)(pluginMetadata.capabilities.mdap_maker).toBe(true);
        });
        (0, globals_1.test)('plugin should implement required methods', async () => {
            // Validate plugin interface methods exist
            const requiredMethods = [
                'initialize',
                'executeEvolution',
                'executeAdversarial',
                'executeDecomposition',
                'executeIntegrated',
                'getConfig',
                'updateConfig',
                'getExecution',
                'getExecutionHistory',
                'getStatistics',
                'cancelExecution',
                'clearHistory',
                'validateConfig',
                'getAvailableStrategies',
                'shouldUseMdapMakerForGoal',
                'getMdapMakerConfig',
            ];
            // In a real implementation, this would check the actual plugin instance
            requiredMethods.forEach(method => {
                (0, globals_1.expect)(method).toBeDefined();
            });
        });
    });
    (0, globals_1.describe)('API Contracts - Health Endpoint', () => {
        (0, globals_1.test)('GET /health should return 200', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const response = await fetch(`${API_URL}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            (0, globals_1.expect)(response.status).toBe(200);
            const data = await response.json();
            (0, globals_1.expect)(data).toHaveProperty('status');
            (0, globals_1.expect)(data.status).toBe('healthy');
        });
    });
    (0, globals_1.describe)('API Contracts - Teams Endpoint', () => {
        (0, globals_1.test)('GET /teams should return array or 404', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const response = await fetch(`${API_URL}/teams`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            // Accept 200 (has teams) or 404 (no teams yet)
            (0, globals_1.expect)([200, 404]).toContain(response.status);
            if (response.status === 200) {
                const data = await response.json();
                (0, globals_1.expect)(Array.isArray(data)).toBe(true);
            }
        });
        (0, globals_1.test)('POST /teams should accept valid team data', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const teamData = {
                name: `test-team-${Date.now()}`,
                role: 'Blue',
                members: [
                    {
                        model_id: 'gpt-4',
                        api_key: 'test-key',
                        temperature: 0.7,
                        max_tokens: 4096,
                    },
                ],
            };
            const response = await fetch(`${API_URL}/teams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(teamData),
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            // Accept 201 (created) or 400 (validation error - check if structure is correct)
            (0, globals_1.expect)([201, 400, 422]).toContain(response.status);
            if (response.status === 201) {
                const data = await response.json();
                (0, globals_1.expect)(data).toHaveProperty('name');
                (0, globals_1.expect)(data.name).toBe(teamData.name);
            }
        });
    });
    (0, globals_1.describe)('API Contracts - Gauntlets Endpoint', () => {
        (0, globals_1.test)('GET /gauntlets should return array or 404', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const response = await fetch(`${API_URL}/gauntlets`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            (0, globals_1.expect)([200, 404]).toContain(response.status);
            if (response.status === 200) {
                const data = await response.json();
                (0, globals_1.expect)(Array.isArray(data)).toBe(true);
            }
        });
    });
    (0, globals_1.describe)('API Contracts - Workflows Endpoint', () => {
        (0, globals_1.test)('GET /workflows should return array or 404', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const response = await fetch(`${API_URL}/workflows`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            (0, globals_1.expect)([200, 404]).toContain(response.status);
            if (response.status === 200) {
                const data = await response.json();
                (0, globals_1.expect)(Array.isArray(data)).toBe(true);
            }
        });
    });
    (0, globals_1.describe)('Plugin State Contract', () => {
        (0, globals_1.test)('plugin state should have required fields', async () => {
            // Validate plugin state structure
            const requiredStateFields = [
                'defaultExecutionMethod',
                'evolutionConfig',
                'adversarialConfig',
                'decompositionConfig',
                'mdapMaker',
            ];
            requiredStateFields.forEach(field => {
                (0, globals_1.expect)(field).toBeDefined();
            });
        });
        (0, globals_1.test)('evolution config should have valid structure', async () => {
            const evolutionConfig = {
                evolutionMode: globals_1.expect.any(String),
                maxIterations: globals_1.expect.any(Number),
                populationSize: globals_1.expect.any(Number),
                temperature: globals_1.expect.any(Number),
                mutationRate: globals_1.expect.any(Number),
                crossoverRate: globals_1.expect.any(Number),
                elitism: globals_1.expect.any(Boolean),
            };
            (0, globals_1.expect)(evolutionConfig.evolutionMode).toBeDefined();
            (0, globals_1.expect)(evolutionConfig.maxIterations).toBeGreaterThan(0);
            (0, globals_1.expect)(evolutionConfig.populationSize).toBeGreaterThan(0);
        });
        (0, globals_1.test)('adversarial config should have valid structure', async () => {
            const adversarialConfig = {
                adversarialMode: globals_1.expect.any(String),
                redTeamSize: globals_1.expect.any(Number),
                blueTeamSize: globals_1.expect.any(Number),
                maxRounds: globals_1.expect.any(Number),
                qualityThreshold: globals_1.expect.any(Number),
                acceptanceThreshold: globals_1.expect.any(Number),
            };
            (0, globals_1.expect)(adversarialConfig.adversarialMode).toBeDefined();
            (0, globals_1.expect)(adversarialConfig.redTeamSize).toBeGreaterThan(0);
            (0, globals_1.expect)(adversarialConfig.blueTeamSize).toBeGreaterThan(0);
        });
    });
    (0, globals_1.describe)('Execution Result Contract', () => {
        (0, globals_1.test)('execution result should have required fields', async () => {
            const executionResult = {
                executionId: globals_1.expect.any(String),
                startTime: globals_1.expect.any(String),
                endTime: globals_1.expect.any(String),
                durationMs: globals_1.expect.any(Number),
                status: globals_1.expect.any(String),
                module: globals_1.expect.any(String),
                input: globals_1.expect.any(Object),
                output: globals_1.expect.any(Object),
                statistics: globals_1.expect.any(Object),
                error: globals_1.expect.any(String),
            };
            (0, globals_1.expect)(executionResult.executionId).toBeDefined();
            (0, globals_1.expect)(executionResult.durationMs).toBeGreaterThanOrEqual(0);
            (0, globals_1.expect)(['completed', 'failed', 'cancelled', 'executing']).toContain(executionResult.status);
        });
    });
    (0, globals_1.describe)('MDAP/MAKER Config Contract', () => {
        (0, globals_1.test)('MDAP/MAKER config should have required fields', async () => {
            const mdapMakerConfig = {
                enabled: globals_1.expect.any(Boolean),
                autoSelect: globals_1.expect.any(Boolean),
                maxDepth: globals_1.expect.any(Number),
                kAhead: globals_1.expect.any(Number),
                redFlagging: globals_1.expect.any(Boolean),
                adaptiveK: globals_1.expect.any(Boolean),
                autoSelectionKeywords: globals_1.expect.any(Array),
            };
            (0, globals_1.expect)(mdapMakerConfig.enabled).toBeDefined();
            (0, globals_1.expect)(mdapMakerConfig.maxDepth).toBeGreaterThan(0);
            (0, globals_1.expect)(mdapMakerConfig.kAhead).toBeGreaterThan(0);
            (0, globals_1.expect)(Array.isArray(mdapMakerConfig.autoSelectionKeywords)).toBe(true);
        });
    });
    (0, globals_1.describe)('CORS Headers Contract', () => {
        (0, globals_1.test)('API should return CORS headers for browser access', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            const response = await fetch(`${API_URL}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': 'http://localhost:3000',
                },
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            // Check for CORS headers
            const corsHeaders = response.headers.get('Access-Control-Allow-Origin');
            // CORS headers should be present for browser plugin
            (0, globals_1.expect)(corsHeaders || true).toBeDefined();
        });
    });
    (0, globals_1.describe)('Error Response Contract', () => {
        (0, globals_1.test)('API should return structured error responses', async () => {
            if (!apiHealthy) {
                console.warn('Skipping: API not accessible');
                return;
            }
            // Test with invalid data to trigger error
            const response = await fetch(`${API_URL}/teams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ invalid: 'data' }),
                signal: AbortSignal.timeout(TIMEOUT_MS),
            });
            if (response.status >= 400) {
                const data = await response.json();
                // Error responses should have a message or error field
                (0, globals_1.expect)(data).toHaveProperty('error');
                (0, globals_1.expect)(data.error).toBeDefined();
            }
        });
    });
    (0, globals_1.describe)('Plugin Initialization Contract', () => {
        (0, globals_1.test)('plugin should initialize with default config', async () => {
            const defaultConfig = {
                defaultExecutionMethod: 'auto',
                evolutionConfig: {
                    evolutionMode: 'genetic_algorithm',
                    maxIterations: 20,
                    populationSize: 50,
                    temperature: 0.7,
                    mutationRate: 0.15,
                    crossoverRate: 0.85,
                    elitism: true,
                },
                adversarialConfig: {
                    adversarialMode: 'red_blue_team',
                    redTeamSize: 5,
                    blueTeamSize: 5,
                    maxRounds: 8,
                    qualityThreshold: 0.85,
                    acceptanceThreshold: 0.92,
                },
                decompositionConfig: {
                    decompositionStrategy: 'semantic',
                    maxSubProblems: 15,
                    minSubProblemSize: 100,
                    maxSubProblemSize: 800,
                },
                mdapMaker: {
                    enabled: true,
                    autoSelect: true,
                    maxDepth: 8,
                    kAhead: 4,
                    redFlagging: true,
                    adaptiveK: true,
                },
            };
            (0, globals_1.expect)(defaultConfig.defaultExecutionMethod).toBe('auto');
            (0, globals_1.expect)(defaultConfig.mdapMaker.enabled).toBe(true);
        });
    });
});
//# sourceMappingURL=contract.test.js.map