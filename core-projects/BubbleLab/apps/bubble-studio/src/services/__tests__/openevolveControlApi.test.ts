import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockGet, mockPost, mockPut, mockDelete } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
  mockPut: vi.fn(),
  mockDelete: vi.fn(),
}));

vi.mock('@/lib/api', () => ({
  ApiClient: vi.fn().mockImplementation(() => ({
    get: mockGet,
    post: mockPost,
    put: mockPut,
    delete: mockDelete,
  })),
}));

vi.mock('@/utils/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import { openevolveApi } from '../openevolveApi';

describe('openevolveApi control plane', () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPost.mockReset();
    mockPut.mockReset();
    mockDelete.mockReset();
  });

  it('fetches control catalog', async () => {
    const response = {
      success: true,
      components: {
        openevolve_workflows: ['status', 'list_instances'],
      },
    };
    mockGet.mockResolvedValue(response);

    const result = await openevolveApi.getControlCatalog();

    expect(mockGet).toHaveBeenCalledWith('/bubblelabs/control/catalog');
    expect(result).toEqual(response);
  });

  it('refreshes auto discovery', async () => {
    const response = {
      success: true,
      discovered_components: 3,
      discovered_actions: 12,
    };
    mockPost.mockResolvedValue(response);

    const result = await openevolveApi.discoverControlComponents(true);

    expect(mockPost).toHaveBeenCalledWith('/bubblelabs/control/discover', { force: true });
    expect(result).toEqual(response);
  });

  it('executes control action with payload', async () => {
    const response = {
      success: true,
      component: 'openevolve_workflows',
      action: 'status',
      result: {
        success: true,
      },
    };
    mockPost.mockResolvedValue(response);

    const payload = { instance_id: 'inst-123' };
    const result = await openevolveApi.executeControlAction(
      'openevolve_workflows',
      'get_instance_status',
      payload
    );

    expect(mockPost).toHaveBeenCalledWith('/bubblelabs/control/execute', {
      component: 'openevolve_workflows',
      action: 'get_instance_status',
      payload,
    });
    expect(result).toEqual(response);
  });

  it('creates BubbleLabs workflow definition', async () => {
    const response = { definition_id: 'def-123' };
    mockPost.mockResolvedValue(response);

    const payload = {
      name: 'e2e-workflow',
      description: 'Workflow for integration test',
      workflow_type: 'evolution',
      parameters: { max_iterations: 1 },
    };

    const result = await openevolveApi.createBubblelabsWorkflowDefinition(payload);

    expect(mockPost).toHaveBeenCalledWith('/bubblelabs/workflow-definitions', payload);
    expect(result).toEqual(response);
  });

  it('syncs BubbleLabs workflow instance parameters', async () => {
    const response = {
      message: 'Parameters synced successfully (2 updated)',
      instance_id: 'inst-123',
      updated_count: 2,
    };
    mockPost.mockResolvedValue(response);

    const payload = {
      parameters: {
        max_iterations: 1,
        population_size: 2,
      },
    };

    const result = await openevolveApi.syncBubblelabsWorkflowInstanceParameters('inst-123', payload);

    expect(mockPost).toHaveBeenCalledWith(
      '/bubblelabs/workflow-instances/inst-123/parameters',
      payload
    );
    expect(result).toEqual(response);
  });

  it('starts BubbleLabs workflow instance', async () => {
    const response = { message: 'Workflow started', instance_id: 'inst-123', status: 'pending' };
    mockPost.mockResolvedValue(response);

    const result = await openevolveApi.startBubblelabsWorkflowInstance('inst-123');

    expect(mockPost).toHaveBeenCalledWith('/bubblelabs/workflow-instances/inst-123/start', {});
    expect(result).toEqual(response);
  });

  it('runs full BubbleLabs lifecycle from definition to terminal status', async () => {
    mockPost
      .mockResolvedValueOnce({ definition_id: 'def-100' })
      .mockResolvedValueOnce({ instance_id: 'inst-200' })
      .mockResolvedValueOnce({
        message: 'Parameters synced successfully (1 updated)',
        instance_id: 'inst-200',
        updated_count: 1,
      })
      .mockResolvedValueOnce({
        message: 'Workflow started',
        instance_id: 'inst-200',
        status: 'running',
      });

    mockGet
      .mockResolvedValueOnce({
        status: {
          instance_id: 'inst-200',
          status: 'running',
          current_stage: 'evolving',
          progress: 30,
        },
        parameters: {},
      })
      .mockResolvedValueOnce({
        status: {
          instance_id: 'inst-200',
          status: 'completed',
          current_stage: 'done',
          progress: 100,
        },
        parameters: {},
      });

    const definition = await openevolveApi.createBubblelabsWorkflowDefinition({
      name: 'workflow-e2e',
      description: 'integration flow',
      workflow_type: 'evolution',
      parameters: { max_iterations: 1 },
    });
    expect(definition.definition_id).toBe('def-100');

    const instance = await openevolveApi.createBubblelabsWorkflowInstance({
      definition_id: definition.definition_id,
      instance_name: 'instance-e2e',
      inputs: { problem_statement: 'Improve candidate solution quality.' },
      parameters: {},
    });
    expect(instance.instance_id).toBe('inst-200');

    const syncResult = await openevolveApi.syncBubblelabsWorkflowInstanceParameters(
      instance.instance_id,
      { parameters: { max_iterations: 1 } }
    );
    expect(syncResult.updated_count).toBe(1);

    await openevolveApi.startBubblelabsWorkflowInstance(instance.instance_id);

    const terminalStatuses = new Set(['completed', 'failed', 'cancelled', 'stopped']);
    let terminalStatus: string | undefined;
    let attempts = 0;

    while (!terminalStatus && attempts < 5) {
      const detail = await openevolveApi.getBubblelabsWorkflowInstance(instance.instance_id);
      const status = detail.status.status.toLowerCase();
      if (terminalStatuses.has(status)) {
        terminalStatus = status;
      }
      attempts += 1;
    }

    expect(terminalStatus).toBe('completed');
    expect(attempts).toBe(2);
    expect(mockPost).toHaveBeenNthCalledWith(1, '/bubblelabs/workflow-definitions', {
      name: 'workflow-e2e',
      description: 'integration flow',
      workflow_type: 'evolution',
      parameters: { max_iterations: 1 },
    });
    expect(mockPost).toHaveBeenNthCalledWith(2, '/bubblelabs/workflow-instances', {
      definition_id: 'def-100',
      instance_name: 'instance-e2e',
      inputs: { problem_statement: 'Improve candidate solution quality.' },
      parameters: {},
    });
    expect(mockGet).toHaveBeenNthCalledWith(1, '/bubblelabs/workflow-instances/inst-200');
    expect(mockGet).toHaveBeenNthCalledWith(2, '/bubblelabs/workflow-instances/inst-200');
  });
});
