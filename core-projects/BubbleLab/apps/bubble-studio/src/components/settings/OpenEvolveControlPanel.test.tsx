import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const {
  mockGetControlCatalog,
  mockDiscoverControlComponents,
  mockExecuteControlAction,
  mockListDefinitions,
  mockListInstances,
  mockCreateDefinition,
  mockCreateInstance,
  mockSyncParameters,
  mockStartInstance,
  mockPauseInstance,
  mockResumeInstance,
  mockStopInstance,
  mockCancelInstance,
  mockRestartInstance,
  mockDeleteInstance,
  mockRefreshStatus,
} = vi.hoisted(() => ({
  mockGetControlCatalog: vi.fn(),
  mockDiscoverControlComponents: vi.fn(),
  mockExecuteControlAction: vi.fn(),
  mockListDefinitions: vi.fn(),
  mockListInstances: vi.fn(),
  mockCreateDefinition: vi.fn(),
  mockCreateInstance: vi.fn(),
  mockSyncParameters: vi.fn(),
  mockStartInstance: vi.fn(),
  mockPauseInstance: vi.fn(),
  mockResumeInstance: vi.fn(),
  mockStopInstance: vi.fn(),
  mockCancelInstance: vi.fn(),
  mockRestartInstance: vi.fn(),
  mockDeleteInstance: vi.fn(),
  mockRefreshStatus: vi.fn(),
}));

vi.mock('@/services/openevolveApi', () => ({
  openevolveApi: {
    getControlCatalog: mockGetControlCatalog,
    discoverControlComponents: mockDiscoverControlComponents,
    executeControlAction: mockExecuteControlAction,
    listBubblelabsWorkflowDefinitions: mockListDefinitions,
    listBubblelabsWorkflowInstances: mockListInstances,
    createBubblelabsWorkflowDefinition: mockCreateDefinition,
    createBubblelabsWorkflowInstance: mockCreateInstance,
    syncBubblelabsWorkflowInstanceParameters: mockSyncParameters,
    startBubblelabsWorkflowInstance: mockStartInstance,
    pauseBubblelabsWorkflowInstance: mockPauseInstance,
    resumeBubblelabsWorkflowInstance: mockResumeInstance,
    stopBubblelabsWorkflowInstance: mockStopInstance,
    cancelBubblelabsWorkflowInstance: mockCancelInstance,
    restartBubblelabsWorkflowInstance: mockRestartInstance,
    deleteBubblelabsWorkflowInstance: mockDeleteInstance,
  },
}));

vi.mock('@/hooks/useBubblelabsWorkflowPolling', () => ({
  useBubblelabsWorkflowPolling: () => ({
    detail: null,
    status: 'created',
    isLoading: false,
    isPolling: false,
    errorMessage: null,
    refresh: mockRefreshStatus,
  }),
  isTerminalBubblelabsWorkflowStatus: (status?: string) =>
    !!status &&
    new Set(['completed', 'failed', 'cancelled', 'stopped']).has(status.toLowerCase()),
}));

import { OpenEvolveControlPanel } from './OpenEvolveControlPanel';

describe('OpenEvolveControlPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockGetControlCatalog.mockResolvedValue({
      success: true,
      components: {
        openevolve_workflows: ['status'],
      },
    });
    mockDiscoverControlComponents.mockResolvedValue({ success: true });
    mockExecuteControlAction.mockResolvedValue({ success: true, result: {} });
    mockListDefinitions.mockResolvedValue({
      definitions: [{ id: 'def-1', name: 'Workflow A', workflow_type: 'evolution' }],
    });
    mockListInstances.mockResolvedValue({
      instances: [
        {
          instance_id: 'inst-1',
          workflow_type: 'evolution',
          status: 'created',
          current_stage: 'created',
          problem_statement: 'test',
          progress: 0,
        },
      ],
    });
    mockCreateDefinition.mockResolvedValue({ definition_id: 'def-2' });
    mockCreateInstance.mockResolvedValue({ instance_id: 'inst-2' });
    mockSyncParameters.mockResolvedValue({
      message: 'Parameters synced successfully (1 updated)',
      instance_id: 'inst-1',
      updated_count: 1,
    });
    mockStartInstance.mockResolvedValue({
      message: 'Workflow started',
      instance_id: 'inst-1',
      status: 'running',
    });
    mockPauseInstance.mockResolvedValue({ message: 'Workflow paused' });
    mockResumeInstance.mockResolvedValue({ message: 'Workflow resumed' });
    mockStopInstance.mockResolvedValue({ message: 'Workflow stopped' });
    mockCancelInstance.mockResolvedValue({ message: 'Workflow cancelled' });
    mockRestartInstance.mockResolvedValue({ message: 'Workflow restarted' });
    mockDeleteInstance.mockResolvedValue({ message: 'Workflow instance deleted' });
    mockRefreshStatus.mockResolvedValue(null);
  });

  it('wires BubbleLabs workflow lifecycle actions through service methods', async () => {
    const user = userEvent.setup();
    render(<OpenEvolveControlPanel />);

    await waitFor(() => {
      expect(mockGetControlCatalog).toHaveBeenCalledTimes(1);
      expect(mockListDefinitions).toHaveBeenCalledTimes(1);
      expect(mockListInstances).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole('button', { name: 'Create Definition' }));
    expect(mockCreateDefinition).toHaveBeenCalledWith({
      name: 'OpenEvolve Workflow',
      description: 'Managed from Bubble Studio',
      workflow_type: 'evolution',
      parameters: {},
    });

    await user.click(screen.getByRole('button', { name: 'Create Instance' }));
    expect(mockCreateInstance).toHaveBeenCalledWith({
      definition_id: 'def-2',
      instance_name: 'bubble-control-instance',
      inputs: {
        problem_statement: 'Generate and evaluate an initial OpenEvolve population.',
      },
      parameters: {},
    });

    await user.click(screen.getByRole('button', { name: 'Sync Parameters' }));
    expect(mockSyncParameters).toHaveBeenCalledWith('inst-2', {
      parameters: {
        max_iterations: 1,
      },
    });

    await user.click(screen.getByRole('button', { name: 'Start' }));
    expect(mockStartInstance).toHaveBeenCalledWith('inst-1');
  });
});
