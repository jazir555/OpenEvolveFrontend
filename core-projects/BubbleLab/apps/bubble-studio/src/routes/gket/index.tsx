import { createFileRoute } from '@tanstack/react-router';
import { Tab } from '@headlessui/react';
import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import {
  useGketExtract,
  useGketFetchExport,
  useGketGenerateModels,
  useGketHealth,
  useGketParseDoc,
} from '@/hooks/use-gket-api';
import type {
  GketCase,
  GketExportFormat,
  GketLlm,
  GketParser,
} from '@/services/gketApi';
import { BubbleConfigPanel } from '@/components/BubbleConfigPanel';

export const Route = createFileRoute('/gket/')({
  component: GketPage,
});

// ==================== Shared helpers ====================

function JsonView({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <p className="text-sm text-gray-500">No response yet.</p>;
  }
  return (
    <pre className="max-h-96 overflow-auto rounded-lg bg-gray-900 p-4 text-xs text-gray-100">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function Notes({ notes }: { notes?: string[] }) {
  if (!notes || notes.length === 0) return null;
  return (
    <ul className="space-y-1 rounded-md bg-amber-50 p-3 text-xs text-amber-800">
      {notes.map((note) => (
        <li key={note}>• {note}</li>
      ))}
    </ul>
  );
}

function TabButton({ label }: { label: string }) {
  return (
    <Tab className="rounded-md px-3 py-2 text-sm font-medium text-gray-600 ui-selected:bg-white ui-selected:text-gray-900 ui-selected:shadow">
      {label}
    </Tab>
  );
}

const inputClass = 'mt-1 w-full rounded-md border border-gray-300 p-2 text-sm';
const buttonClass =
  'rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50';

// ==================== Parse Tab ====================

function ParseTab() {
  const [fileRef, setFileRef] = useState('');
  const [parser, setParser] = useState<GketParser>('fast');
  const parse = useGketParseDoc();

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Parse a document by its path on the GKET server (PDF / DOCX). Fast uses
        PyMuPDF + python-docx; Docling uses the AI layout pipeline when installed.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-[280px] flex-1 text-sm font-medium text-gray-700">
          File path
          <input
            className={inputClass}
            placeholder="data/multi-type documents/invoice_1.pdf"
            value={fileRef}
            onChange={(e) => setFileRef(e.target.value)}
          />
        </label>
        <label className="text-sm font-medium text-gray-700">
          Parser
          <select
            className={inputClass}
            value={parser}
            onChange={(e) => setParser(e.target.value as GketParser)}
          >
            <option value="fast">fast</option>
            <option value="docling">docling</option>
          </select>
        </label>
        <button
          className={buttonClass}
          onClick={() => parse.mutate({ file_ref: fileRef, parser })}
          disabled={parse.isPending || !fileRef.trim()}
        >
          {parse.isPending ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Parse'
          )}
        </button>
      </div>

      <Notes notes={parse.data?.notes} />
      {parse.data?.error && (
        <p className="text-sm text-red-600">
          {parse.data.error} {parse.data.detail}
        </p>
      )}
      {parse.data?.text !== undefined && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500">
            {parse.data.file_name} — {parse.data.word_count} words,{' '}
            {parse.data.content_length} chars
          </p>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-lg bg-gray-50 p-4 text-xs text-gray-800">
            {parse.data.text || '(empty)'}
          </pre>
        </div>
      )}
    </div>
  );
}

// ==================== Generate Models Tab ====================

function GenerateModelsTab() {
  const [description, setDescription] = useState('');
  const [useCase, setUseCase] = useState('');
  const [llm, setLlm] = useState<GketLlm>('openai');
  const generate = useGketGenerateModels();

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Describe what to extract in plain language; the server parses it into a
        field config and a Pydantic model schema.
      </p>
      <textarea
        className="h-32 w-full rounded-md border border-gray-300 p-2 text-sm"
        placeholder="Extract the company name, the industry domain and the total revenue from each report."
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-[220px] flex-1 text-sm font-medium text-gray-700">
          Use case (optional)
          <input
            className={inputClass}
            placeholder="CompanyReports"
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
          />
        </label>
        <label className="text-sm font-medium text-gray-700">
          LLM
          <select
            className={inputClass}
            value={llm}
            onChange={(e) => setLlm(e.target.value as GketLlm)}
          >
            <option value="openai">openai</option>
            <option value="claude">claude</option>
          </select>
        </label>
        <button
          className={buttonClass}
          onClick={() =>
            generate.mutate({
              text_description: description,
              use_case: useCase,
              llm,
            })
          }
          disabled={generate.isPending || description.trim().length === 0}
        >
          {generate.isPending ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Generate models'
          )}
        </button>
      </div>

      <Notes notes={generate.data?.notes} />
      {generate.data?.error && (
        <p className="text-sm text-red-600">
          {generate.data.error} {generate.data.detail}
        </p>
      )}
      {generate.data?.json_schema && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900">
            {generate.data.model_name} schema
          </h3>
          <JsonView value={generate.data.json_schema} />
        </div>
      )}
      {generate.data?.model_code && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-900">Generated code</h3>
          <pre className="max-h-72 overflow-auto rounded-lg bg-gray-50 p-4 text-xs text-gray-800">
            {generate.data.model_code}
          </pre>
        </div>
      )}
    </div>
  );
}

// ==================== Extract Tab ====================

const CASE_LABELS: Record<GketCase, string> = {
  0: '0 — single type extraction',
  1: '1 — multi type classification & routing',
  2: '2 — hierarchical PO → BOM',
};

function RecordsTable({ records }: { records: Array<Record<string, unknown>> }) {
  const columns = Array.from(
    records.reduce<Set<string>>((acc, record) => {
      Object.keys(record).forEach((key) => acc.add(key));
      return acc;
    }, new Set<string>())
  );

  if (columns.length === 0) {
    return <p className="text-sm text-gray-500">No records returned.</p>;
  }

  const cell = (value: unknown) =>
    value === null || value === undefined
      ? ''
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);

  return (
    <div className="overflow-auto rounded-lg border border-gray-200">
      <table className="min-w-full divide-y divide-gray-200 text-xs">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                className="px-3 py-2 text-left font-semibold text-gray-700"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {records.map((record, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column} className="px-3 py-2 align-top text-gray-800">
                  {cell(record[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExtractTab({ onResultId }: { onResultId: (id: string) => void }) {
  const [extractionCase, setExtractionCase] = useState<GketCase>(0);
  const [llm, setLlm] = useState<GketLlm>('openai');
  const [input, setInput] = useState('');
  const [instruction, setInstruction] = useState('');
  const [schemaText, setSchemaText] = useState('');
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const extract = useGketExtract();

  const run = async () => {
    setSchemaError(null);
    let modelSchema: Record<string, unknown> | null = null;
    if (schemaText.trim()) {
      try {
        modelSchema = JSON.parse(schemaText) as Record<string, unknown>;
      } catch {
        setSchemaError('Schema must be valid JSON.');
        return;
      }
    }
    const result = await extract.mutateAsync({
      case: extractionCase,
      llm,
      text_or_file_ref: input,
      model_schema: modelSchema,
      instruction: instruction || undefined,
    });
    if (result.id) onResultId(result.id);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm font-medium text-gray-700">
          Case
          <select
            className={inputClass}
            value={extractionCase}
            onChange={(e) =>
              setExtractionCase(Number(e.target.value) as GketCase)
            }
          >
            {([0, 1, 2] as GketCase[]).map((value) => (
              <option key={value} value={value}>
                {CASE_LABELS[value]}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm font-medium text-gray-700">
          LLM
          <select
            className={inputClass}
            value={llm}
            onChange={(e) => setLlm(e.target.value as GketLlm)}
          >
            <option value="openai">openai</option>
            <option value="claude">claude</option>
          </select>
        </label>
        <button className={buttonClass} onClick={run} disabled={extract.isPending}>
          {extract.isPending ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Run extraction'
          )}
        </button>
      </div>

      <label className="block text-sm font-medium text-gray-700">
        Input text or server file path
        <textarea
          className="mt-1 h-28 w-full rounded-md border border-gray-300 p-2 text-sm"
          placeholder="Paste document text, or a path like data/purchase_orders/po_1.pdf"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      </label>

      <label className="block text-sm font-medium text-gray-700">
        Instruction / extraction description
        <input
          className={inputClass}
          placeholder="Extract purchase order items and their BOM details"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
        />
      </label>

      <label className="block text-sm font-medium text-gray-700">
        Model schema JSON (optional, case 0)
        <textarea
          className="mt-1 h-28 w-full rounded-md border border-gray-300 p-2 font-mono text-xs"
          placeholder='{"fields":[{"field_name":"total","field_type":"float","description":"Invoice total"}]}'
          value={schemaText}
          onChange={(e) => setSchemaText(e.target.value)}
        />
      </label>
      {schemaError && <p className="text-sm text-red-600">{schemaError}</p>}

      <Notes notes={extract.data?.notes} />
      {extract.data?.error && (
        <p className="text-sm text-red-600">
          {extract.data.error} {extract.data.detail}
        </p>
      )}
      {extract.data && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500">
            id: <code>{extract.data.id}</code> — status: {extract.data.status}
          </p>
          <RecordsTable records={extract.data.records ?? []} />
        </div>
      )}
    </div>
  );
}

// ==================== Export Tab ====================

function ExportTab({ resultId }: { resultId: string }) {
  const [id, setId] = useState(resultId);
  const [format, setFormat] = useState<GketExportFormat>('json');
  const fetchExport = useGketFetchExport();

  const download = async () => {
    const payload = await fetchExport.mutateAsync({ id, format });
    const body =
      typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
    const blob = new Blob([body], {
      type: format === 'csv' ? 'text/csv' : 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${id || 'gket-export'}.${format === 'xlsx' ? 'txt' : format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Fetch a stored <code>/extract</code> result by id. CSV downloads directly;
        JSON is shown below and can also be downloaded.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-[220px] flex-1 text-sm font-medium text-gray-700">
          Result id
          <input
            className={inputClass}
            placeholder="Paste an id from the Extract tab"
            value={id}
            onChange={(e) => setId(e.target.value)}
          />
        </label>
        <label className="text-sm font-medium text-gray-700">
          Format
          <select
            className={inputClass}
            value={format}
            onChange={(e) => setFormat(e.target.value as GketExportFormat)}
          >
            <option value="json">json</option>
            <option value="csv">csv</option>
            <option value="xlsx">xlsx</option>
          </select>
        </label>
        <button
          className={buttonClass}
          onClick={() => fetchExport.mutate({ id, format })}
          disabled={fetchExport.isPending || !id.trim()}
        >
          {fetchExport.isPending ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Fetch'
          )}
        </button>
        <button
          className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 disabled:opacity-50"
          onClick={download}
          disabled={fetchExport.isPending || !id.trim()}
        >
          Download
        </button>
      </div>

      {typeof fetchExport.data === 'string' ? (
        <pre className="max-h-96 overflow-auto rounded-lg bg-gray-50 p-4 text-xs text-gray-800">
          {fetchExport.data}
        </pre>
      ) : (
        <JsonView value={fetchExport.data} />
      )}
    </div>
  );
}

// ==================== Page ====================

function GketPage() {
  const health = useGketHealth();
  const [resultId, setResultId] = useState('');

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Knowledge Extraction</h1>
        <span className="text-xs text-gray-500">
          GKET API:{' '}
          {health.isPending
            ? 'checking…'
            : health.data?.status === 'ok'
              ? 'online'
              : 'offline'}
        </span>
      </div>
      <Tab.Group>
        <Tab.List className="mb-4 flex gap-2 rounded-lg bg-gray-100 p-2">
          <TabButton label="Parse Document" />
          <TabButton label="Generate Models" />
          <TabButton label="Extract" />
          <TabButton label="Export" />
          <TabButton label="Bubble Config" />
        </Tab.List>
        <Tab.Panels>
          <Tab.Panel>
            <ParseTab />
          </Tab.Panel>
          <Tab.Panel>
            <GenerateModelsTab />
          </Tab.Panel>
          <Tab.Panel>
            <ExtractTab onResultId={setResultId} />
          </Tab.Panel>
          <Tab.Panel>
            <ExportTab resultId={resultId} />
          </Tab.Panel>
          <Tab.Panel>
            <BubbleConfigPanel
              bubbleKey="gket"
              hint="Configure the GKET extraction bubble (parser, LLM, case classifier) before running."
            />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
}
