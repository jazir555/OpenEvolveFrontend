"use strict";
/**
 * OpenEvolve Main Adapter Contract Tests
 *
 * Comprehensive contract tests for the OpenEvolve orchestration adapter.
 * These tests validate that the adapter correctly orchestrates all
 * integrated systems and maintains the canonical schema contracts.
 *
 * Environment Variables:
 *   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (required)
 *   TIMEOUT_MS - Request timeout in milliseconds (default: 5000)
 *   SKIP_INTEGRATION_TESTS - Skip integration tests if true
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const axios_1 = __importDefault(require("axios"));
// Configuration from environment
const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8002';
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || '5000', 10);
const SKIP_INTEGRATION_TESTS = process.env.SKIP_INTEGRATION_TESTS === 'true';
// Create axios instance with defaults
const api = axios_1.default.create({
    baseURL: API_URL,
    timeout: TIMEOUT_MS,
    headers: {
        'Content-Type': 'application/json',
    },
});
// Test utilities
const generateTestId = () => {
    return `test-${Date.now()}-${Math.random().toString(36).substring(7) `;
};

const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

describe('OpenEvolve API - Health and Root Endpoints', () => {
  test('GET /health should return healthy status', async () => {
    const response = await api.get('/health');

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('status', 'healthy');
    expect(response.data).toHaveProperty('timestamp');

    // Validate timestamp format (ISO 8601 UTC)
    const timestamp = new Date(response.data.timestamp);
    expect(timestamp.toISOString()).toBe(response.data.timestamp);
  });

  test('GET / should return API information', async () => {
    const response = await api.get('/');

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message');
    expect(response.data).toHaveProperty('version');
    expect(response.data).toHaveProperty('docs');
  });
});

describe('OpenEvolve Teams API', () => {
  const testTeamId = generateTestId();
  const testTeam: Team = {
    name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
    role: 'Blue',
    members: [
      {
        model_id: 'gpt-4',
        api_key: '',
        api_base: 'http://localhost:8001',
        temperature: 0.7,
        max_tokens: 4096,
      },
    ],
    description: 'Test team for contract validation',
  };

  test('POST /openevolve/teams should create a new team', async () => {
    const response = await api.post('/openevolve/teams', testTeam);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message', 'Team created');
    expect(response.data).toHaveProperty('team_name', testTeam.name);
  });

  test('GET /openevolve/teams should list all teams', async () => {
    const response = await api.get('/openevolve/teams');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.data)).toBe(true);

    // Find our test team
    const createdTeam = response.data.find((t: any) => t.name === testTeam.name);
    expect(createdTeam).toBeDefined();
    expect(createdTeam).toHaveProperty('role', testTeam.role);
    expect(createdTeam).toHaveProperty('member_count');
  });

  test('GET /openevolve/teams/{team_name} should return team details', async () => {
    const response = await api.get(` / openevolve / teams / $;
    {
        testTeam.name;
    }
    `);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('name', testTeam.name);
    expect(response.data).toHaveProperty('role', testTeam.role);
    expect(response.data).toHaveProperty('members');
    expect(Array.isArray(response.data.members)).toBe(true);
    expect(response.data.members[0]).toHaveProperty('model_id');
    expect(response.data.members[0]).toHaveProperty('temperature');
    expect(response.data.members[0]).toHaveProperty('max_tokens');
  });

  test('PUT /openevolve/teams/{team_name} should update team', async () => {
    const updatedTeam = {
      ...testTeam,
      description: 'Updated test team description',
    };

    const response = await api.put(` / openevolve / teams / $;
    {
        testTeam.name;
    }
    `, updatedTeam);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message', 'Team updated');
  });

  test('DELETE /openevolve/teams/{team_name} should delete team', async () => {
    const response = await api.delete(` / openevolve / teams / $;
    {
        testTeam.name;
    }
    `);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message', 'Team deleted');
  });

  test('GET /openevolve/teams/{team_name} should return 404 for deleted team', async () => {
    try {
      await api.get(` / openevolve / teams / $;
    {
        testTeam.name;
    }
    `);
      expect(true).toBe(false); // Should not reach here
    } catch (error) {
      const axiosError = error as AxiosError;
      expect(axiosError.response?.status).toBe(404);
    }
  });
});

describe('OpenEvolve Gauntlets API', () => {
  const testGauntletId = generateTestId();
  const testTeamId = generateTestId();

  beforeAll(async () => {
    // Create a test team first
    const testTeam: Team = {
      name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
      role: 'Red',
      members: [
        {
          model_id: 'claude-3-opus',
          api_key: '',
          api_base: 'http://localhost:8001',
          temperature: 0.7,
          max_tokens: 4096,
        },
      ],
    };

    await api.post('/openevolve/teams', testTeam);
  });

  const testGauntlet: Gauntlet = {
    name: `;
    globals_1.test - gauntlet - $;
    {
        testGauntletId;
    }
    `,
    team_name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
    rounds: [
      {
        round_number: 1,
        quorum_required_approvals: 2,
        quorum_from_panel_size: 3,
        min_overall_confidence: 0.8,
      },
    ],
    description: 'Test gauntlet for contract validation',
  };

  test('POST /openevolve/gauntlets should create a new gauntlet', async () => {
    const response = await api.post('/openevolve/gauntlets', testGauntlet);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message', 'Gauntlet created');
    expect(response.data).toHaveProperty('gauntlet_name', testGauntlet.name);
  });

  test('GET /openevolve/gauntlets should list all gauntlets', async () => {
    const response = await api.get('/openevolve/gauntlets');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.data)).toBe(true);

    const createdGauntlet = response.data.find((g: any) => g.name === testGauntlet.name);
    expect(createdGauntlet).toBeDefined();
    expect(createdGauntlet).toHaveProperty('team_name', testGauntlet.team_name);
    expect(createdGauntlet).toHaveProperty('round_count');
  });

  test('GET /openevolve/gauntlets/{gauntlet_name} should return gauntlet details', async () => {
    const response = await api.get(` / openevolve / gauntlets / $;
    {
        testGauntlet.name;
    }
    `);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('name', testGauntlet.name);
    expect(response.data).toHaveProperty('team_name', testGauntlet.team_name);
    expect(response.data).toHaveProperty('rounds');
    expect(Array.isArray(response.data.rounds)).toBe(true);
    expect(response.data.rounds[0]).toHaveProperty('round_number', 1);
    expect(response.data.rounds[0]).toHaveProperty('quorum_required_approvals');
    expect(response.data.rounds[0]).toHaveProperty('min_overall_confidence');
  });

  test('DELETE /openevolve/gauntlets/{gauntlet_name} should delete gauntlet', async () => {
    const response = await api.delete(` / openevolve / gauntlets / $;
    {
        testGauntlet.name;
    }
    `);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message', 'Gauntlet deleted');
  });
});

describe('OpenEvolve Workflow Orchestration', () => {
  const testWorkflowId = generateTestId();
  const testTeamId = generateTestId();
  const testGauntletId = generateTestId();

  beforeAll(async () => {
    // Create test team and gauntlet
    const testTeam: Team = {
      name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
      role: 'Blue',
      members: [
        {
          model_id: 'gpt-4',
          api_key: '',
          api_base: 'http://localhost:8001',
          temperature: 0.7,
          max_tokens: 4096,
        },
      ],
    };

    await api.post('/openevolve/teams', testTeam);

    const testGauntlet: Gauntlet = {
      name: `;
    globals_1.test - gauntlet - $;
    {
        testGauntletId;
    }
    `,
      team_name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
      rounds: [
        {
          round_number: 1,
          quorum_required_approvals: 1,
          quorum_from_panel_size: 1,
          min_overall_confidence: 0.5,
        },
      ],
    };

    await api.post('/openevolve/gauntlets', testGauntlet);
  });

  const testWorkflow: WorkflowDefinition = {
    workflow_id: `;
    globals_1.test - workflow - $;
    {
        testWorkflowId;
    }
    `,
    name: 'Test Workflow',
    description: 'Test workflow for contract validation',
    problem_statement: 'Solve the test problem',
    max_refinement_loops: 2,
    auto_approval_enabled: true,
    sub_problems: [
      {
        id: 'test-sub-1',
        description: 'First test sub-problem',
        dependencies: [],
        solver_team_name: `;
    globals_1.test - team - $;
    {
        testTeamId;
    }
    `,
        gold_team_gauntlet_name: `;
    globals_1.test - gauntlet - $;
    {
        testGauntletId;
    }
    `,
      },
    ],
  };

  test('POST /openevolve/workflows should create a new workflow', async () => {
    const response = await api.post('/openevolve/workflows', testWorkflow);

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('message');
    expect(response.data).toHaveProperty('workflow_id', testWorkflow.workflow_id);
  });

  test('GET /openevolve/workflows should list all workflows', async () => {
    const response = await api.get('/openevolve/workflows');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.data)).toBe(true);

    const createdWorkflow = response.data.find((w: any) => w.workflow_id === testWorkflow.workflow_id);
    expect(createdWorkflow).toBeDefined();
    expect(createdWorkflow).toHaveProperty('name', testWorkflow.name);
    expect(createdWorkflow).toHaveProperty('status');
  });

  test('GET /openevolve/workflows/{workflow_id}/status should return workflow state', async () => {
    const response = await api.get(` / openevolve / workflows / $;
    {
        testWorkflow.workflow_id;
    }
    /status`;
    ;
    (0, globals_1.expect)(response.status).toBe(200);
    (0, globals_1.expect)(response.data).toHaveProperty('workflow_id', testWorkflow.workflow_id);
    (0, globals_1.expect)(response.data).toHaveProperty('status');
    (0, globals_1.expect)(response.data).toHaveProperty('current_stage');
    (0, globals_1.expect)(response.data).toHaveProperty('progress');
    (0, globals_1.expect)(response.data).toHaveProperty('start_time');
    // Validate timestamp format
    const startTime = new Date(response.data.start_time);
    (0, globals_1.expect)(startTime.toISOString()).toBe(response.data.start_time);
};
(0, globals_1.test)('Workflow state should have valid progress value', async () => {
    const response = await api.get(`/openevolve/workflows/${testWorkflow.workflow_id}/status`);
    (0, globals_1.expect)(response.data.progress).toBeGreaterThanOrEqual(0);
    (0, globals_1.expect)(response.data.progress).toBeLessThanOrEqual(1);
});
(0, globals_1.test)('DELETE /openevolve/workflows/{workflow_id} should cancel/delete workflow', async () => {
    const response = await api.delete(`/openevolve/workflows/${testWorkflow.workflow_id}`);
    (0, globals_1.expect)(response.status).toBe(200);
    (0, globals_1.expect)(response.data).toHaveProperty('message');
});
;
(0, globals_1.describe)('OpenEvolve Canonical Schema Validation', () => {
    (0, globals_1.test)('Team should have required canonical fields', async () => {
        const testTeam = {
            name: `canonical-test-team-${Date.now()}`,
            role: 'Gold',
            members: [
                {
                    model_id: 'test-model',
                    api_key: '',
                    api_base: 'http://localhost:8001',
                    temperature: 0.7,
                    max_tokens: 4096,
                },
            ],
        };
        const response = await api.post('/openevolve/teams', testTeam);
        (0, globals_1.expect)(response.status).toBe(200);
        const getResponse = await api.get(`/openevolve/teams/${testTeam.name}`);
        (0, globals_1.expect)(getResponse.data).toHaveProperty('name');
        (0, globals_1.expect)(getResponse.data).toHaveProperty('role');
        (0, globals_1.expect)(getResponse.data).toHaveProperty('members');
        // Validate role is one of the canonical values
        (0, globals_1.expect)(['Blue', 'Red', 'Gold']).toContain(getResponse.data.role);
    });
    (0, globals_1.test)('Gauntlet should have required canonical fields', async () => {
        const testTeamName = `canonical-test-team-${Date.now()}`;
        const testGauntletName = `canonical-test-gauntlet-${Date.now()}`;
        // Create team first
        await api.post('/openevolve/teams', {
            name: testTeamName,
            role: 'Red',
            members: [
                {
                    model_id: 'test-model',
                    api_key: '',
                    api_base: 'http://localhost:8001',
                    temperature: 0.7,
                    max_tokens: 4096,
                },
            ],
        });
        const testGauntlet = {
            name: testGauntletName,
            team_name: testTeamName,
            rounds: [
                {
                    round_number: 1,
                    quorum_required_approvals: 1,
                    quorum_from_panel_size: 1,
                    min_overall_confidence: 0.5,
                },
            ],
        };
        const response = await api.post('/openevolve/gauntlets', testGauntlet);
        (0, globals_1.expect)(response.status).toBe(200);
        const getResponse = await api.get(`/openevolve/gauntlets/${testGauntletName}`);
        (0, globals_1.expect)(getResponse.data).toHaveProperty('name');
        (0, globals_1.expect)(getResponse.data).toHaveProperty('team_name');
        (0, globals_1.expect)(getResponse.data).toHaveProperty('rounds');
        // Validate round structure
        (0, globals_1.expect)(getResponse.data.rounds[0]).toHaveProperty('round_number');
        (0, globals_1.expect)(getResponse.data.rounds[0]).toHaveProperty('quorum_required_approvals');
        (0, globals_1.expect)(getResponse.data.rounds[0]).toHaveProperty('quorum_from_panel_size');
        (0, globals_1.expect)(getResponse.data.rounds[0]).toHaveProperty('min_overall_confidence');
    });
});
globals_1.describe.skipIf(SKIP_INTEGRATION_TESTS)('OpenEvolve Integration Coordination', () => {
    (0, globals_1.test)('Adapter should report integration health status', async () => {
        const response = await api.get('/openevolve/integrations/health');
        (0, globals_1.expect)(response.status).toBe(200);
        (0, globals_1.expect)(response.data).toHaveProperty('integrations');
        (0, globals_1.expect)(Array.isArray(response.data.integrations)).toBe(true);
        // Each integration should have name and status
        if (response.data.integrations.length > 0) {
            (0, globals_1.expect)(response.data.integrations[0]).toHaveProperty('name');
            (0, globals_1.expect)(response.data.integrations[0]).toHaveProperty('status');
            (0, globals_1.expect)(['healthy', 'unhealthy', 'unknown']).toContain(response.data.integrations[0].status);
        }
    });
    (0, globals_1.test)('Adapter should list available adapters', async () => {
        const response = await api.get('/openevolve/integrations/adapters');
        (0, globals_1.expect)(response.status).toBe(200);
        (0, globals_1.expect)(Array.isArray(response.data)).toBe(true);
        // Should include known adapters
        const adapterNames = response.data.map((a) => a.name);
        const expectedAdapters = ['z3', 'leanaide', 'ragbits', 'vectordb', 'graphiti', 'karateclub'];
        // At least some expected adapters should be present
        const foundAdapters = expectedAdapters.filter(a => adapterNames.includes(a));
        (0, globals_1.expect)(foundAdapters.length).toBeGreaterThan(0);
    });
});
(0, globals_1.describe)('OpenEvolve Error Handling', () => {
    (0, globals_1.test)('GET non-existent team should return 404', async () => {
        try {
            await api.get('/openevolve/teams/non-existent-team-12345');
            (0, globals_1.expect)(true).toBe(false);
        }
        catch (error) {
            const axiosError = error;
            (0, globals_1.expect)(axiosError.response?.status).toBe(404);
            (0, globals_1.expect)(axiosError.response?.data).toHaveProperty('detail');
        }
    });
    (0, globals_1.test)('GET non-existent gauntlet should return 404', async () => {
        try {
            await api.get('/openevolve/gauntlets/non-existent-gauntlet-12345');
            (0, globals_1.expect)(true).toBe(false);
        }
        catch (error) {
            const axiosError = error;
            (0, globals_1.expect)(axiosError.response?.status).toBe(404);
            (0, globals_1.expect)(axiosError.response?.data).toHaveProperty('detail');
        }
    });
    (0, globals_1.test)('POST duplicate team should return 400', async () => {
        const teamName = `duplicate-test-team-${Date.now()}`;
        const testTeam = {
            name: teamName,
            role: 'Blue',
            members: [
                {
                    model_id: 'test-model',
                    api_key: '',
                    api_base: 'http://localhost:8001',
                    temperature: 0.7,
                    max_tokens: 4096,
                },
            ],
        };
        // Create first team
        await api.post('/openevolve/teams', testTeam);
        // Try to create duplicate
        try {
            await api.post('/openevolve/teams', testTeam);
            (0, globals_1.expect)(true).toBe(false);
        }
        catch (error) {
            const axiosError = error;
            (0, globals_1.expect)(axiosError.response?.status).toBe(400);
            (0, globals_1.expect)(axiosError.response?.data).toHaveProperty('detail');
        }
    });
    (0, globals_1.test)('Invalid team role should return 400', async () => {
        const invalidTeam = {
            name: `invalid-team-${Date.now()}`,
            role: 'InvalidRole',
            members: [
                {
                    model_id: 'test-model',
                    api_key: '',
                    api_base: 'http://localhost:8001',
                    temperature: 0.7,
                    max_tokens: 4096,
                },
            ],
        };
        try {
            await api.post('/openevolve/teams', invalidTeam);
            (0, globals_1.expect)(true).toBe(false);
        }
        catch (error) {
            const axiosError = error;
            (0, globals_1.expect)(axiosError.response?.status).toBe(422); // Unprocessable Entity
        }
    });
});
(0, globals_1.describe)('OpenEvolve UTC Timestamp Compliance', () => {
    (0, globals_1.test)('All timestamps should be in UTC ISO-8601 format', async () => {
        const response = await api.get('/health');
        (0, globals_1.expect)(response.status).toBe(200);
        (0, globals_1.expect)(response.data).toHaveProperty('timestamp');
        const timestamp = response.data.timestamp;
        const parsedDate = new Date(timestamp);
        // Should be valid ISO-8601
        (0, globals_1.expect)(parsedDate.toISOString()).toBe(timestamp);
        // Should end with Z (UTC indicator)
        (0, globals_1.expect)(timestamp.endsWith('Z')).toBe(true);
    });
});
//# sourceMappingURL=contract.test.js.map