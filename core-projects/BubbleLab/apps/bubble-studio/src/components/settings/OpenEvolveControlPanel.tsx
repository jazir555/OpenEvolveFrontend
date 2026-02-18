import { useEffect, useMemo, useState } from 'react';
import {
  openevolveApi,
  type BubbleLabsControlCatalogResponse,
} from '@/services/openevolveApi';
import {
  isTerminalBubblelabsWorkflowStatus,
  useBubblelabsWorkflowPolling,
} from '@/hooks/useBubblelabsWorkflowPolling';

type PayloadTemplateMap = Record<string, Record<string, Record<string, unknown>>>;

const PAYLOAD_TEMPLATES: PayloadTemplateMap = {
  openevolve_workflows: {
    status: {},
    create_definition: {
      name: 'OpenEvolve Workflow',
      description: 'Managed from Bubble Studio',
      workflow_type: 'evolution',
      parameters: {},
    },
    list_definitions: {},
    get_definition: {
      definition_id: '',
    },
    create_instance: {
      definition_id: '',
      instance_name: 'bubble-control-instance',
      inputs: {},
      parameters: {},
    },
    list_instances: {},
    get_instance_status: {
      instance_id: '',
    },
    start_instance: {
      instance_id: '',
    },
    pause_instance: {
      instance_id: '',
    },
    resume_instance: {
      instance_id: '',
    },
    stop_instance: {
      instance_id: '',
    },
    cancel_instance: {
      instance_id: '',
    },
    restart_instance: {
      instance_id: '',
    },
    delete_instance: {
      instance_id: '',
    },
    sync_parameters: {
      instance_id: '',
      parameters: {},
    },
  },
};

const BUBBLELABS_WORKFLOW_TYPES = [
  'evolution',
  'adversarial',
  'sovereign',
  'web3',
  'rag',
  'default',
] as const;

const OPENEVOLVE_API_KEY_STORAGE = 'openevolve_api_key';

function getPayloadTemplate(component: string, action: string): Record<string, unknown> {
  return PAYLOAD_TEMPLATES[component]?.[action] ?? {};
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function parseJsonObject(text: string, errorMessage: string): Record<string, unknown> {
  if (!text.trim()) {
    return {};
  }

  const parsed = JSON.parse(text);
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error(errorMessage);
  }

  return parsed as Record<string, unknown>;
}

const readStoredApiKey = (): string => {
  try {
    return globalThis.localStorage?.getItem(OPENEVOLVE_API_KEY_STORAGE) ?? '';
  } catch {
    return '';
  }
};

export function OpenEvolveControlPanel() {
  const [catalog, setCatalog] = useState<BubbleLabsControlCatalogResponse | null>(null);
  const [apiKey, setApiKey] = useState<string>(readStoredApiKey);
  const [selectedComponent, setSelectedComponent] = useState('');
  const [selectedAction, setSelectedAction] = useState('');
  const [payloadText, setPayloadText] = useState('{}');
  const [resultText, setResultText] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(false);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  const [isWorkflowDataLoading, setIsWorkflowDataLoading] = useState(false);
  const [isWorkflowActionLoading, setIsWorkflowActionLoading] = useState(false);
  const [workflowDefinitions, setWorkflowDefinitions] = useState<
    Array<{ id: string; name: string; workflow_type: string }>
  >([]);
  const [workflowInstances, setWorkflowInstances] = useState<
    Array<{ instance_id: string; status: string; workflow_type: string; progress?: number }>
  >([]);
  const [selectedDefinitionId, setSelectedDefinitionId] = useState('');
  const [selectedInstanceId, setSelectedInstanceId] = useState('');
  const [newDefinitionName, setNewDefinitionName] = useState('OpenEvolve Workflow');
  const [newDefinitionDescription, setNewDefinitionDescription] = useState(
    'Managed from Bubble Studio'
  );
  const [newDefinitionType, setNewDefinitionType] = useState<string>('evolution');
  const [newDefinitionParametersText, setNewDefinitionParametersText] = useState('{}');
  const [newInstanceName, setNewInstanceName] = useState('bubble-control-instance');
  const [newInstanceProblemStatement, setNewInstanceProblemStatement] = useState(
    'Generate and evaluate an initial OpenEvolve population.'
  );
  const [newInstanceParametersText, setNewInstanceParametersText] = useState('{}');
  const [syncParametersText, setSyncParametersText] = useState('{\n  "max_iterations": 1\n}');

  const componentOptions = useMemo(
    () => Object.keys(catalog?.components ?? {}).sort(),
    [catalog]
  );

  const actionOptions = useMemo(() => {
    if (!selectedComponent || !catalog?.components[selectedComponent]) {
      return [];
    }
    return [...catalog.components[selectedComponent]].sort();
  }, [catalog, selectedComponent]);

  const selectedInstanceSummary = useMemo(
    () =>
      workflowInstances.find(
        (instance) => instance.instance_id === selectedInstanceId
      ) ?? null,
    [selectedInstanceId, workflowInstances]
  );

  const {
    detail: polledInstanceDetail,
    status: polledInstanceStatus,
    isLoading: isPollingStatus,
    isPolling,
    errorMessage: pollingErrorMessage,
    refresh: refreshInstanceStatus,
  } = useBubblelabsWorkflowPolling({
    instanceId: selectedInstanceId || null,
    enabled: Boolean(selectedInstanceId),
    intervalMs: 2000,
    stopOnTerminal: true,
  });

  const loadCatalog = async () => {
    setIsCatalogLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getControlCatalog();
      setCatalog(response);
      if (!response.success) {
        setErrorMessage('Failed to load control catalog');
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to load control catalog');
    } finally {
      setIsCatalogLoading(false);
    }
  };

  const loadWorkflowLifecycleData = async () => {
    setIsWorkflowDataLoading(true);
    setErrorMessage(null);
    try {
      const [definitionsResponse, instancesResponse] = await Promise.all([
        openevolveApi.listBubblelabsWorkflowDefinitions(),
        openevolveApi.listBubblelabsWorkflowInstances(),
      ]);

      const definitions = definitionsResponse.definitions ?? [];
      const instances = instancesResponse.instances ?? [];
      setWorkflowDefinitions(definitions);
      setWorkflowInstances(instances);

      setSelectedDefinitionId((current) => {
        if (current && definitions.some((definition) => definition.id === current)) {
          return current;
        }
        return definitions[0]?.id ?? '';
      });

      setSelectedInstanceId((current) => {
        if (current && instances.some((instance) => instance.instance_id === current)) {
          return current;
        }
        return instances[0]?.instance_id ?? '';
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to load workflow lifecycle data'
      );
    } finally {
      setIsWorkflowDataLoading(false);
    }
  };

  useEffect(() => {
    void loadCatalog();
    void loadWorkflowLifecycleData();
  }, []);

  useEffect(() => {
    if (pollingErrorMessage) {
      setErrorMessage(pollingErrorMessage);
    }
  }, [pollingErrorMessage]);

  useEffect(() => {
    try {
      if (apiKey.trim().length > 0) {
        globalThis.localStorage?.setItem(OPENEVOLVE_API_KEY_STORAGE, apiKey.trim());
      } else {
        globalThis.localStorage?.removeItem(OPENEVOLVE_API_KEY_STORAGE);
      }
    } catch {
      // ignore localStorage access errors
    }
  }, [apiKey]);

  useEffect(() => {
    if (componentOptions.length === 0) {
      setSelectedComponent('');
      return;
    }
    if (!componentOptions.includes(selectedComponent)) {
      setSelectedComponent(componentOptions[0]);
    }
  }, [componentOptions, selectedComponent]);

  useEffect(() => {
    if (actionOptions.length === 0) {
      setSelectedAction('');
      return;
    }
    if (!actionOptions.includes(selectedAction)) {
      setSelectedAction(actionOptions[0]);
    }
  }, [actionOptions, selectedAction]);

  useEffect(() => {
    if (!selectedComponent || !selectedAction) {
      setPayloadText('{}');
      return;
    }
    setPayloadText(prettyJson(getPayloadTemplate(selectedComponent, selectedAction)));
  }, [selectedComponent, selectedAction]);

  const executeWorkflowLifecycleAction = async (
    action: 'start' | 'pause' | 'resume' | 'stop' | 'cancel' | 'restart' | 'delete'
  ) => {
    if (!selectedInstanceId) {
      setErrorMessage('Select a workflow instance first');
      return;
    }

    setIsWorkflowActionLoading(true);
    setErrorMessage(null);
    try {
      let response: Record<string, unknown>;
      switch (action) {
        case 'start':
          response = await openevolveApi.startBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'pause':
          response = await openevolveApi.pauseBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'resume':
          response = await openevolveApi.resumeBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'stop':
          response = await openevolveApi.stopBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'cancel':
          response = await openevolveApi.cancelBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'restart':
          response = await openevolveApi.restartBubblelabsWorkflowInstance(selectedInstanceId);
          break;
        case 'delete':
          response = await openevolveApi.deleteBubblelabsWorkflowInstance(selectedInstanceId);
          break;
      }

      setResultText(prettyJson(response));
      await loadWorkflowLifecycleData();
      if (action !== 'delete' && selectedInstanceId) {
        await refreshInstanceStatus();
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to execute workflow lifecycle action'
      );
    } finally {
      setIsWorkflowActionLoading(false);
    }
  };

  const handleCreateDefinition = async () => {
    setIsWorkflowActionLoading(true);
    setErrorMessage(null);
    try {
      const parameters = parseJsonObject(
        newDefinitionParametersText,
        'Definition parameters must be a JSON object'
      );

      const response = await openevolveApi.createBubblelabsWorkflowDefinition({
        name: newDefinitionName.trim() || 'OpenEvolve Workflow',
        description: newDefinitionDescription.trim() || 'Managed from Bubble Studio',
        workflow_type: newDefinitionType,
        parameters,
      });

      setResultText(prettyJson(response));
      await loadWorkflowLifecycleData();
      if (response.definition_id) {
        setSelectedDefinitionId(response.definition_id);
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to create workflow definition'
      );
    } finally {
      setIsWorkflowActionLoading(false);
    }
  };

  const handleCreateInstance = async () => {
    if (!selectedDefinitionId) {
      setErrorMessage('Select a workflow definition first');
      return;
    }

    setIsWorkflowActionLoading(true);
    setErrorMessage(null);
    try {
      const parameters = parseJsonObject(
        newInstanceParametersText,
        'Instance parameters must be a JSON object'
      );

      const response = await openevolveApi.createBubblelabsWorkflowInstance({
        definition_id: selectedDefinitionId,
        instance_name: newInstanceName.trim() || 'bubble-control-instance',
        inputs: {
          problem_statement: newInstanceProblemStatement,
        },
        parameters,
      });

      setResultText(prettyJson(response));
      await loadWorkflowLifecycleData();
      if (response.instance_id) {
        setSelectedInstanceId(response.instance_id);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to create workflow instance');
    } finally {
      setIsWorkflowActionLoading(false);
    }
  };

  const handleSyncParameters = async () => {
    if (!selectedInstanceId) {
      setErrorMessage('Select a workflow instance first');
      return;
    }

    setIsWorkflowActionLoading(true);
    setErrorMessage(null);
    try {
      const parameters = parseJsonObject(
        syncParametersText,
        'Sync parameters payload must be a JSON object'
      );

      const response = await openevolveApi.syncBubblelabsWorkflowInstanceParameters(
        selectedInstanceId,
        { parameters }
      );
      setResultText(prettyJson(response));
      await refreshInstanceStatus();
      await loadWorkflowLifecycleData();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to sync workflow parameters');
    } finally {
      setIsWorkflowActionLoading(false);
    }
  };

  const handleDiscover = async () => {
    setIsDiscovering(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.discoverControlComponents(true);
      setResultText(prettyJson(response));
      await loadCatalog();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to refresh discovery');
    } finally {
      setIsDiscovering(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedComponent || !selectedAction) {
      setErrorMessage('Select a component and action');
      return;
    }

    setIsExecuting(true);
    setErrorMessage(null);
    try {
      const payload = parseJsonObject(payloadText, 'Payload must be a JSON object');
      const response = await openevolveApi.executeControlAction(
        selectedComponent,
        selectedAction,
        payload
      );

      setResultText(prettyJson(response));
      if (!response.success) {
        setErrorMessage(response.error || 'Control action failed');
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to execute control action'
      );
    } finally {
      setIsExecuting(false);
    }
  };

  const currentStatus = polledInstanceStatus ?? selectedInstanceSummary?.status ?? null;

  return (
    <section className="rounded-xl border border-[#2a2a2a] bg-[#111111] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-white">OpenEvolve Control Plane</h2>
          <p className="mt-1 text-sm text-gray-400">
            Manage integrated components through BubbleLabs unified catalog and action runner.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void loadCatalog()}
            disabled={isCatalogLoading}
            className="rounded-md border border-[#3a3a3a] px-3 py-2 text-sm text-gray-200 hover:bg-[#1b1b1b] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isCatalogLoading ? 'Refreshing...' : 'Refresh Catalog'}
          </button>
          <button
            type="button"
            onClick={() => void handleDiscover()}
            disabled={isDiscovering}
            className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isDiscovering ? 'Discovering...' : 'Discover Integrations'}
          </button>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-500">
        Components discovered: {componentOptions.length}
      </div>

      <label className="mt-4 block">
        <span className="mb-2 block text-sm text-gray-300">OpenEvolve API Key</span>
        <input
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          type="password"
          placeholder="Set API key for X-API-Key protected endpoints"
          className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
        />
      </label>

      {errorMessage && (
        <div className="mt-4 rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      <section className="mt-5 rounded-lg border border-[#2d2d2d] bg-[#0d0d0d] p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-base font-medium text-white">BubbleLabs Workflow Lifecycle</h3>
          <button
            type="button"
            onClick={() => void loadWorkflowLifecycleData()}
            disabled={isWorkflowDataLoading}
            className="rounded border border-[#3a3a3a] px-3 py-1.5 text-xs text-gray-200 hover:bg-[#1b1b1b] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isWorkflowDataLoading ? 'Refreshing...' : 'Refresh Definitions + Instances'}
          </button>
        </div>

        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div className="space-y-3 rounded border border-[#2b2b2b] bg-[#111] p-3">
            <div className="text-sm font-medium text-gray-100">Create Definition</div>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Name</span>
              <input
                value={newDefinitionName}
                onChange={(event) => setNewDefinitionName(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Description</span>
              <input
                value={newDefinitionDescription}
                onChange={(event) => setNewDefinitionDescription(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Workflow Type</span>
              <select
                value={newDefinitionType}
                onChange={(event) => setNewDefinitionType(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              >
                {BUBBLELABS_WORKFLOW_TYPES.map((workflowType) => (
                  <option key={workflowType} value={workflowType}>
                    {workflowType}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Parameters (JSON)</span>
              <textarea
                value={newDefinitionParametersText}
                onChange={(event) => setNewDefinitionParametersText(event.target.value)}
                rows={5}
                spellCheck={false}
                className="w-full rounded-md border border-[#303030] bg-[#0b0b0b] px-3 py-2 font-mono text-xs text-gray-100"
              />
            </label>
            <button
              type="button"
              onClick={() => void handleCreateDefinition()}
              disabled={isWorkflowActionLoading}
              className="rounded-md bg-emerald-600 px-3 py-2 text-xs font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Create Definition
            </button>
          </div>

          <div className="space-y-3 rounded border border-[#2b2b2b] bg-[#111] p-3">
            <div className="text-sm font-medium text-gray-100">Create Instance</div>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Definition</span>
              <select
                value={selectedDefinitionId}
                onChange={(event) => setSelectedDefinitionId(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              >
                {workflowDefinitions.length === 0 && (
                  <option value="">No definitions available</option>
                )}
                {workflowDefinitions.map((definition) => (
                  <option key={definition.id} value={definition.id}>
                    {definition.name} ({definition.workflow_type})
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Instance Name</span>
              <input
                value={newInstanceName}
                onChange={(event) => setNewInstanceName(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Problem Statement</span>
              <textarea
                value={newInstanceProblemStatement}
                onChange={(event) => setNewInstanceProblemStatement(event.target.value)}
                rows={3}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Initial Parameters (JSON)</span>
              <textarea
                value={newInstanceParametersText}
                onChange={(event) => setNewInstanceParametersText(event.target.value)}
                rows={4}
                spellCheck={false}
                className="w-full rounded-md border border-[#303030] bg-[#0b0b0b] px-3 py-2 font-mono text-xs text-gray-100"
              />
            </label>
            <button
              type="button"
              onClick={() => void handleCreateInstance()}
              disabled={!selectedDefinitionId || isWorkflowActionLoading}
              className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Create Instance
            </button>
          </div>
        </div>

        <div className="mt-4 rounded border border-[#2b2b2b] bg-[#111] p-3">
          <div className="grid gap-3 md:grid-cols-[1fr_auto]">
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Instance</span>
              <select
                value={selectedInstanceId}
                onChange={(event) => setSelectedInstanceId(event.target.value)}
                className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
              >
                {workflowInstances.length === 0 && <option value="">No instances available</option>}
                {workflowInstances.map((instance) => (
                  <option key={instance.instance_id} value={instance.instance_id}>
                    {instance.instance_id} ({instance.workflow_type})
                  </option>
                ))}
              </select>
            </label>
            <div className="rounded border border-[#2e2e2e] bg-[#0d0d0d] px-3 py-2 text-xs text-gray-300">
              <div>Status: {currentStatus ?? 'n/a'}</div>
              <div>
                Progress:{' '}
                {polledInstanceDetail?.status?.progress ??
                  selectedInstanceSummary?.progress ??
                  0}
                %
              </div>
              <div>
                Polling:{' '}
                {isPolling
                  ? 'active'
                  : currentStatus && isTerminalBubblelabsWorkflowStatus(currentStatus)
                    ? 'stopped (terminal)'
                    : 'idle'}
                {isPollingStatus ? ' (updating...)' : ''}
              </div>
            </div>
          </div>

          <div className="mt-3">
            <label className="block">
              <span className="mb-1 block text-xs text-gray-400">Sync Parameters (JSON)</span>
              <textarea
                value={syncParametersText}
                onChange={(event) => setSyncParametersText(event.target.value)}
                rows={4}
                spellCheck={false}
                className="w-full rounded-md border border-[#303030] bg-[#0b0b0b] px-3 py-2 font-mono text-xs text-gray-100"
              />
            </label>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void handleSyncParameters()}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded border border-[#3a3a3a] px-2.5 py-1.5 text-xs text-gray-100 hover:bg-[#1a1a1a] disabled:cursor-not-allowed disabled:opacity-50"
            >
              Sync Parameters
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('start')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-emerald-700 px-2.5 py-1.5 text-xs text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Start
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('pause')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-amber-700 px-2.5 py-1.5 text-xs text-white hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Pause
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('resume')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-sky-700 px-2.5 py-1.5 text-xs text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Resume
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('stop')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-rose-700 px-2.5 py-1.5 text-xs text-white hover:bg-rose-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Stop
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('cancel')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-fuchsia-700 px-2.5 py-1.5 text-xs text-white hover:bg-fuchsia-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('restart')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-violet-700 px-2.5 py-1.5 text-xs text-white hover:bg-violet-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Restart
            </button>
            <button
              type="button"
              onClick={() => void executeWorkflowLifecycleAction('delete')}
              disabled={!selectedInstanceId || isWorkflowActionLoading}
              className="rounded bg-red-800 px-2.5 py-1.5 text-xs text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Delete
            </button>
          </div>
        </div>
      </section>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <label className="block">
          <span className="mb-2 block text-sm text-gray-300">Component</span>
          <select
            value={selectedComponent}
            onChange={(event) => setSelectedComponent(event.target.value)}
            className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
          >
            {componentOptions.length === 0 && <option value="">No components available</option>}
            {componentOptions.map((component) => (
              <option key={component} value={component}>
                {component}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-2 block text-sm text-gray-300">Action</span>
          <select
            value={selectedAction}
            onChange={(event) => setSelectedAction(event.target.value)}
            className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
          >
            {actionOptions.length === 0 && <option value="">No actions available</option>}
            {actionOptions.map((action) => (
              <option key={action} value={action}>
                {action}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm text-gray-300">Payload (JSON)</span>
          <button
            type="button"
            onClick={() =>
              setPayloadText(prettyJson(getPayloadTemplate(selectedComponent, selectedAction)))
            }
            className="rounded border border-[#303030] px-2 py-1 text-xs text-gray-300 hover:bg-[#1b1b1b]"
          >
            Reset Template
          </button>
        </div>
        <textarea
          value={payloadText}
          onChange={(event) => setPayloadText(event.target.value)}
          rows={8}
          spellCheck={false}
          className="w-full rounded-md border border-[#303030] bg-[#0b0b0b] px-3 py-2 font-mono text-xs text-gray-100"
        />
      </div>

      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={() => void handleExecute()}
          disabled={!selectedComponent || !selectedAction || isExecuting}
          className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isExecuting ? 'Executing...' : 'Execute Action'}
        </button>
      </div>

      <div className="mt-5">
        <div className="mb-2 text-sm text-gray-300">Result</div>
        <pre className="max-h-96 overflow-auto rounded-md border border-[#303030] bg-[#0b0b0b] p-3 text-xs text-gray-100">
          {resultText || 'No action executed yet.'}
        </pre>
      </div>
    </section>
  );
}
