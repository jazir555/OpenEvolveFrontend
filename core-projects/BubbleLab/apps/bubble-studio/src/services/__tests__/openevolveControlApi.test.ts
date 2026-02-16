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
});
