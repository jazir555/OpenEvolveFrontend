import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { openevolveApi } from '@/services/openevolveApi';

export const Route = createFileRoute('/openevolve/evolution')({
  component: EvolutionPage,
});

type WorkflowType = 'evolution' | 'adversarial' | 'sovereign';

interface EvolutionItem {
  key: string;
  label: string;
  workflowType: WorkflowType;
  description: string;
  parameters: Record<string, unknown>;
}

const COMPOSITIONS: EvolutionItem[] = [
  {
    key: 'pipeline',
    label: 'Evolution Pipeline',
    workflowType: 'evolution',
    description:
      'Standard one-shot evolution pipeline: decompose the problem, spawn solver/validator teams, run gauntlets, and refine until convergence.',
    parameters: { mode: 'pipeline' },
  },
  {
    key: 'continuous',
    label: 'Continuous Evolution',
    workflowType: 'evolution',
    description:
      'Long-running continuous evolution that keeps sampling and improving across rounds without a fixed stopping point.',
    parameters: { mode: 'continuous' },
  },
  {
    key: 'adaptive',
    label: 'Adaptive Evolution',
    workflowType: 'evolution',
    description:
      'Adaptive evolution that dynamically re-plans sub-problems and re-weights teams/gauntlets based on intermediate scores.',
    parameters: { mode: 'adaptive' },
  },
];

const BUBBLES: EvolutionItem[] = [
  {
    key: 'trigger',
    label: 'Trigger Bubble',
    workflowType: 'evolution',
    description:
      'Kick off an evolution run from an external event/signal. Registers the trigger and starts the downstream composition.',
    parameters: { bubble: 'trigger' },
  },
  {
    key: 'application',
    label: 'Application Bubble',
    workflowType: 'evolution',
    description:
      'Applies an evolution result to a target artifact/workload — the "apply the winner" stage of the loop.',
    parameters: { bubble: 'application' },
  },
  {
    key: 'validation',
    label: 'Validation Bubble',
    workflowType: 'evolution',
    description:
      'Runs the gauntlet/validation suite against a candidate, producing the score that gates further refinement.',
    parameters: { bubble: 'validation' },
  },
];

interface RunResult {
  status: 'idle' | 'running' | 'success' | 'error';
  definitionId?: string;
  instanceId?: string;
  data?: unknown;
  error?: string;
}

function EvolutionCard({ item }: { item: EvolutionItem }) {
  const [result, setResult] = useState<RunResult>({ status: 'idle' });

  const run = async () => {
    setResult({ status: 'running' });
    try {
      const def = await openevolveApi.createBubblelabsWorkflowDefinition({
        name: item.label,
        description: item.description,
        workflow_type: item.workflowType,
        parameters: item.parameters,
      });
      const inst = await openevolveApi.createBubblelabsWorkflowInstance({
        definition_id: def.definition_id,
        instance_name: `${item.label} ${new Date().toISOString()}`,
        inputs: { problem_statement: 'Auto-run from BubbleLab UI' },
        parameters: item.parameters,
      });
      const start = await openevolveApi.startBubblelabsWorkflowInstance(
        inst.instance_id
      );
      setResult({
        status: 'success',
        definitionId: def.definition_id,
        instanceId: inst.instance_id,
        data: start,
      });
    } catch (e) {
      setResult({
        status: 'error',
        error: e instanceof Error ? e.message : String(e),
      });
    }
  };

  return (
    <div className="flex flex-col rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-semibold text-gray-900">{item.label}</h3>
        <button
          className="shrink-0 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          onClick={run}
          disabled={result.status === 'running'}
        >
          {result.status === 'running' ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Run'
          )}
        </button>
      </div>
      <p className="mt-2 text-sm text-gray-600">{item.description}</p>
      <p className="mt-2 font-mono text-xs text-gray-400">
        workflow_type: {item.workflowType}
      </p>

      {result.status === 'success' && (
        <div className="mt-3 space-y-1 text-xs">
          <p className="font-medium text-green-700">Started ✓</p>
          <p className="text-gray-600">definition_id: {result.definitionId}</p>
          <p className="text-gray-600">instance_id: {result.instanceId}</p>
          <pre className="max-h-48 overflow-auto rounded bg-gray-900 p-3 text-gray-100">
            {JSON.stringify(result.data, null, 2)}
          </pre>
        </div>
      )}
      {result.status === 'error' && (
        <div className="mt-3 text-xs text-red-700">
          Error: {result.error}
        </div>
      )}
    </div>
  );
}

function Section({
  title,
  subtitle,
  items,
}: {
  title: string;
  subtitle: string;
  items: EvolutionItem[];
}) {
  return (
    <section className="mt-6">
      <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
      <p className="mb-3 text-sm text-gray-500">{subtitle}</p>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <EvolutionCard key={item.key} item={item} />
        ))}
      </div>
    </section>
  );
}

function EvolutionPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900">Evolution</h1>
      <p className="mt-1 text-sm text-gray-600">
        Launch the Evolution compositions and individual bubbles. Each
        &nbsp;Run creates a BubbleLabs workflow definition + instance and starts
        it against the OpenEvolve backend (requires the OpenEvolve API to be
        running).
      </p>

      <Section
        title="Compositions"
        subtitle="End-to-end evolution run modes."
        items={COMPOSITIONS}
      />
      <Section
        title="Bubbles"
        subtitle="Individual stages of the evolution loop."
        items={BUBBLES}
      />
    </div>
  );
}
