import { createFileRoute } from '@tanstack/react-router';
import { Tab } from '@headlessui/react';
import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useConfigStore } from '@/stores/configStore';
import type { LLMProvider } from '@/types/api';
import {
  useLeanAideRawResponse,
  useLeanAideBenchmarkStart,
  useLeanAideBenchmark,
} from '@/hooks/use-leanaide-api';
import type { LeanAideAnyPayload } from '@/services/leanaideApi';

export const Route = createFileRoute('/leanaide/')({
  component: LeanAidePage,
});

// ==================== Shared helpers ====================

function JsonView({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <p className="text-sm text-gray-500">No response yet.</p>;
  }
  return (
    <pre className="overflow-auto rounded-lg bg-gray-900 p-4 text-xs text-gray-100">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function TabButton({ label }: { label: string }) {
  return (
    <Tab className="rounded-md px-3 py-2 text-sm font-medium text-gray-600 ui-selected:bg-white ui-selected:text-gray-900 ui-selected:shadow">
      {label}
    </Tab>
  );
}

// ==================== Home Tab ====================

function HomeTab() {
  const provider = useConfigStore((s) => s.provider);
  const modelLeanAide = useConfigStore((s) => s.model_leanaide);
  const temperature = useConfigStore((s) => s.temperature);
  const setLLMProvider = useConfigStore((s) => s.setLLMProvider);
  const setModelLeanAide = useConfigStore((s) => s.setModelLeanAide);
  const setTemperature = useConfigStore((s) => s.setTemperature);

  return (
    <div className="max-w-xl space-y-4">
      <p className="text-sm text-gray-600">
        Configure the LLM used by LeanAide for generation and verification.
      </p>
      <label className="block text-sm font-medium text-gray-700">
        Provider
        <input
          className="mt-1 w-full rounded-md border border-gray-300 p-2"
          value={String(provider)}
          onChange={(e) => setLLMProvider(e.target.value as LLMProvider)}
        />
      </label>
      <label className="block text-sm font-medium text-gray-700">
        Model
        <input
          className="mt-1 w-full rounded-md border border-gray-300 p-2"
          value={modelLeanAide}
          onChange={(e) => setModelLeanAide(e.target.value)}
        />
      </label>
      <label className="block text-sm font-medium text-gray-700">
        Temperature: {temperature}
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          className="mt-1 w-full"
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
      </label>
    </div>
  );
}

// ==================== Server Response Tab ====================

function ServerResponseTab({
  result,
  onResult,
}: {
  result: unknown;
  onResult: (r: unknown) => void;
}) {
  const [task, setTask] = useState('prove_for_formalization');
  const [theorem, setTheorem] = useState('');
  const raw = useLeanAideRawResponse();

  const send = async () => {
    const payload: LeanAideAnyPayload = { task, theorem };
    const res = await raw.mutateAsync(payload);
    onResult(res);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <input
          className="rounded-md border border-gray-300 p-2"
          placeholder="task (e.g. prove_for_formalization)"
          value={task}
          onChange={(e) => setTask(e.target.value)}
        />
        <input
          className="min-w-[240px] flex-1 rounded-md border border-gray-300 p-2"
          placeholder="theorem / body JSON string"
          value={theorem}
          onChange={(e) => setTheorem(e.target.value)}
        />
        <button
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          onClick={send}
          disabled={raw.isPending}
        >
          {raw.isPending ? <Loader2 className="inline h-4 w-4 animate-spin" /> : 'Send POST /'}
        </button>
      </div>
      <JsonView value={result} />
    </div>
  );
}

// ==================== Token Response Tab ====================

function TokenResponseTab({ result }: { result: unknown }) {
  const data = (result ?? {}) as Record<string, unknown>;
  const known = [
    'usage',
    'token_usage',
    'tokens',
    'prompt_tokens',
    'completion_tokens',
    'total_tokens',
  ];
  const meta = Object.fromEntries(
    Object.entries(data).filter(([k]) =>
      known.some((key) => k.toLowerCase().includes(key))
    )
  );

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Token usage / response metadata extracted from the last{' '}
        <code>POST /</code> response.
      </p>
      <JsonView value={Object.keys(meta).length ? meta : data} />
    </div>
  );
}

// ==================== Structured JSON Tab ====================

function StructuredJsonTab() {
  const [body, setBody] = useState(
    JSON.stringify({ task: 'lean_from_json_structured', data: {} }, null, 2)
  );
  const structured = useLeanAideRawResponse();

  const send = async () => {
    let parsed: LeanAideAnyPayload;
    try {
      parsed = JSON.parse(body) as LeanAideAnyPayload;
    } catch {
      parsed = { task: 'lean_from_json_structured' };
    }
    await structured.mutateAsync(parsed);
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Send a structured-proof task (<code>lean_from_json_structured</code>) to{' '}
        <code>POST /</code>.
      </p>
      <textarea
        className="h-48 w-full rounded-md border border-gray-300 p-2 font-mono text-xs"
        value={body}
        onChange={(e) => setBody(e.target.value)}
      />
      <button
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        onClick={send}
        disabled={structured.isPending}
      >
        {structured.isPending ? (
          <Loader2 className="inline h-4 w-4 animate-spin" />
        ) : (
          'Send structured task'
        )}
      </button>
      <JsonView value={structured.data} />
    </div>
  );
}

// ==================== Benchmark Tab ====================

function BenchmarkTab() {
  const start = useLeanAideBenchmarkStart();
  const [id, setId] = useState<string | null>(null);
  const results = useLeanAideBenchmark(id);

  const run = async () => {
    const res = await start.mutateAsync();
    setId(res.benchmark_id);
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-amber-700">
        Note: benchmarking is mocked/demo — the LeanAide server does not yet
        implement it. Results below come from the BubbleLab proxy mock.
      </p>
      <button
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        onClick={run}
        disabled={start.isPending}
      >
        {start.isPending ? (
          <Loader2 className="inline h-4 w-4 animate-spin" />
        ) : (
          'Start benchmark'
        )}
      </button>
      {id && (
        <div className="space-y-2">
          <p className="text-sm text-gray-600">
            Benchmark id: <code>{id}</code>{' '}
            {results.isFetching && '(polling…)'}
          </p>
          <JsonView value={results.data} />
        </div>
      )}
    </div>
  );
}

// ==================== Logs Display Tab ====================

function LogsDisplayTab({ result }: { result: unknown }) {
  const data = (result ?? {}) as Record<string, unknown>;
  const logs =
    typeof data.logs === 'string'
      ? data.logs
      : data.logs !== undefined
        ? JSON.stringify(data.logs, null, 2)
        : 'No logs in the last response.';

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        The <code>logs</code> string returned inline by <code>POST /</code>.
      </p>
      <pre className="overflow-auto rounded-lg bg-gray-900 p-4 text-xs text-gray-100">
        {logs}
      </pre>
    </div>
  );
}

// ==================== Page ====================

function LeanAidePage() {
  const [rawResult, setRawResult] = useState<unknown>(null);

  return (
    <div className="p-6">
      <h1 className="mb-4 text-2xl font-bold text-gray-900">LeanAide</h1>
      <Tab.Group>
        <Tab.List className="mb-4 flex gap-2 rounded-lg bg-gray-100 p-2">
          <TabButton label="Home" />
          <TabButton label="Server Response" />
          <TabButton label="Token Response" />
          <TabButton label="Structured JSON" />
          <TabButton label="Benchmark" />
          <TabButton label="Logs" />
        </Tab.List>
        <Tab.Panels>
          <Tab.Panel>
            <HomeTab />
          </Tab.Panel>
          <Tab.Panel>
            <ServerResponseTab result={rawResult} onResult={setRawResult} />
          </Tab.Panel>
          <Tab.Panel>
            <TokenResponseTab result={rawResult} />
          </Tab.Panel>
          <Tab.Panel>
            <StructuredJsonTab />
          </Tab.Panel>
          <Tab.Panel>
            <BenchmarkTab />
          </Tab.Panel>
          <Tab.Panel>
            <LogsDisplayTab result={rawResult} />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
}
