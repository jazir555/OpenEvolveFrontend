import { useEffect, useMemo, useState } from 'react';
import {
  openevolveApi,
  type BubbleLabsControlCatalogResponse,
} from '@/services/openevolveApi';

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

const OPENEVOLVE_API_KEY_STORAGE = 'openevolve_api_key';

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

  useEffect(() => {
    void loadCatalog();
  }, []);

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
      let payload: Record<string, unknown> = {};
      if (payloadText.trim().length > 0) {
        const parsed = JSON.parse(payloadText);
        if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
          throw new Error('Payload must be a JSON object');
        }
        payload = parsed as Record<string, unknown>;
      }

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

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <label className="block">
          <span className="mb-2 block text-sm text-gray-300">Component</span>
          <select
            value={selectedComponent}
            onChange={(event) => setSelectedComponent(event.target.value)}
            className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
          >
            {componentOptions.length === 0 && (
              <option value="">No components available</option>
            )}
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
