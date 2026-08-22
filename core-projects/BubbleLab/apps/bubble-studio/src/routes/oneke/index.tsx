import { createFileRoute } from '@tanstack/react-router';
import { Tab } from '@headlessui/react';
import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import {
  useOneKESchemas,
  useOneKECases,
  useOneKEExtract,
  useOneKEResult,
} from '@/hooks/use-oneke-api';
import type {
  OneKETask,
  OneKEMode,
  OneKEExtractPayload,
  OneKEResult,
} from '@/services/onekeApi';

export const Route = createFileRoute('/oneke/')({
  component: OneKEPage,
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

// ==================== Extract Tab ====================

function ExtractTab() {
  const [task, setTask] = useState<OneKETask>('NER');
  const [mode, setMode] = useState<OneKEMode>('quick');
  const [modelName, setModelName] = useState('gpt-4o-mini');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [text, setText] = useState('');
  const [fileRef, setFileRef] = useState('');
  const [instruction, setInstruction] = useState('');
  const [constraint, setConstraint] = useState('');

  const schemas = useOneKESchemas();
  const cases = useOneKECases();
  const extract = useOneKEExtract();
  const [result, setResult] = useState<OneKEResult | null>(null);

  const send = async () => {
    const payload: OneKEExtractPayload = {
      task,
      mode,
      model_name: modelName,
      base_url: baseUrl || undefined,
      api_key: apiKey || undefined,
      text: text || undefined,
      file_ref: fileRef || undefined,
      instruction: instruction || undefined,
      constraint: constraint || undefined,
    };
    const res = await extract.mutateAsync(payload);
    setResult(res);
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Configure and run OneKE knowledge extraction. Provide raw text or a file
        path; choose a task and mode.
      </p>

      <div className="grid max-w-3xl grid-cols-2 gap-3">
        <label className="block text-sm font-medium text-gray-700">
          Task
          <select
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={task}
            onChange={(e) => setTask(e.target.value as OneKETask)}
          >
            <option>NER</option>
            <option>RE</option>
            <option>EE</option>
            <option>Triple</option>
            <option>Base</option>
          </select>
        </label>
        <label className="block text-sm font-medium text-gray-700">
          Mode
          <select
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={mode}
            onChange={(e) => setMode(e.target.value as OneKEMode)}
          >
            <option>quick</option>
            <option>agent</option>
            <option>customized</option>
          </select>
        </label>
        <label className="block text-sm font-medium text-gray-700">
          Model name
          <input
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          />
        </label>
        <label className="block text-sm font-medium text-gray-700">
          Base URL (optional)
          <input
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        </label>
        <label className="col-span-2 block text-sm font-medium text-gray-700">
          API key (optional)
          <input
            type="password"
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </label>
        <label className="col-span-2 block text-sm font-medium text-gray-700">
          Text input
          <textarea
            className="mt-1 h-24 w-full rounded-md border border-gray-300 p-2 font-mono text-xs"
            placeholder="Raw text to extract from…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </label>
        <label className="col-span-2 block text-sm font-medium text-gray-700">
          File reference (optional path)
          <input
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={fileRef}
            onChange={(e) => setFileRef(e.target.value)}
          />
        </label>
        <label className="col-span-2 block text-sm font-medium text-gray-700">
          Instruction (optional)
          <input
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
          />
        </label>
        <label className="col-span-2 block text-sm font-medium text-gray-700">
          Constraint (optional)
          <input
            className="mt-1 w-full rounded-md border border-gray-300 p-2"
            value={constraint}
            onChange={(e) => setConstraint(e.target.value)}
          />
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
        <span>
          Schemas:{' '}
          {schemas.isLoading
            ? 'loading…'
            : schemas.data?.length
              ? schemas.data.join(', ')
              : 'none'}
        </span>
        <span>
          Cases:{' '}
          {cases.isLoading
            ? 'loading…'
            : cases.data?.length
              ? cases.data.join(', ')
              : 'none'}
        </span>
      </div>

      <button
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        onClick={send}
        disabled={extract.isPending}
      >
        {extract.isPending ? (
          <Loader2 className="inline h-4 w-4 animate-spin" />
        ) : (
          'Run extraction'
        )}
      </button>

      {result && (
        <div className="space-y-3">
          <p
            className={
              result.status === 'success'
                ? 'text-sm font-medium text-green-700'
                : 'text-sm font-medium text-red-700'
            }
          >
            Status: {result.status}
            {result.id ? ` · id: ${result.id}` : ''}
            {result.error ? ` · ${result.error}` : ''}
          </p>
          <div>
            <p className="mb-1 text-sm font-medium text-gray-700">Answer JSON</p>
            <JsonView value={result.answer_json} />
          </div>
          <div>
            <p className="mb-1 text-sm font-medium text-gray-700">
              Triples ({Array.isArray(result.triples) ? result.triples.length : 0})
            </p>
            <JsonView value={result.triples} />
          </div>
        </div>
      )}
    </div>
  );
}

// ==================== Retrieve Result Tab ====================

function RetrieveTab() {
  const [id, setId] = useState('');
  const result = useOneKEResult(id || undefined);

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Fetch a previously stored extraction result by id.
      </p>
      <div className="flex flex-wrap gap-2">
        <input
          className="min-w-[240px] flex-1 rounded-md border border-gray-300 p-2"
          placeholder="result id"
          value={id}
          onChange={(e) => setId(e.target.value)}
        />
      </div>
      {result.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
      <JsonView value={result.data} />
    </div>
  );
}

// ==================== Page ====================

function OneKEPage() {
  return (
    <div className="p-6">
      <h1 className="mb-4 text-2xl font-bold text-gray-900">OneKE</h1>
      <Tab.Group>
        <Tab.List className="mb-4 flex gap-2 rounded-lg bg-gray-100 p-2">
          <TabButton label="Extract" />
          <TabButton label="Retrieve Result" />
        </Tab.List>
        <Tab.Panels>
          <Tab.Panel>
            <ExtractTab />
          </Tab.Panel>
          <Tab.Panel>
            <RetrieveTab />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
}
