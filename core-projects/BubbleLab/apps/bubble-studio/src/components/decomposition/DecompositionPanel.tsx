import { useCallback, useEffect, useMemo, useState } from 'react';
import { Tab } from '@headlessui/react';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  WorkflowDecompositionPlan,
  WorkflowPlanResponse,
  WorkflowPlanUpdateRequest,
  WorkflowSubProblem,
  WorkflowSummary,
  WorkflowEngineResults,
} from '@/types/openevolve';
import { SubProblemCard } from './SubProblemCard';
import { DependencyGraph } from './DependencyGraph';
import { WorkflowSettingsPanel } from './WorkflowSettingsPanel';

/** Plan-level fields that the `PUT /decomposition-plan` body accepts. */
type PlanMeta = Omit<WorkflowDecompositionPlan, 'sub_problems' | 'problem_statement' | 'analyzed_context'>;

function emptySubProblem(index: number): WorkflowSubProblem {
  return {
    id: `sub-${Date.now().toString(36)}-${index}`,
    description: '',
    dependencies: [],
  };
}

function extractPlanMeta(plan: WorkflowDecompositionPlan): PlanMeta {
  const meta: PlanMeta = {};
  if (plan.max_refinement_loops !== undefined)
    meta.max_refinement_loops = plan.max_refinement_loops;
  if (plan.auto_approval_enabled !== undefined)
    meta.auto_approval_enabled = plan.auto_approval_enabled;
  if (plan.auto_approval_criteria !== undefined)
    meta.auto_approval_criteria = plan.auto_approval_criteria;
  if (plan.mdap_enabled !== undefined) meta.mdap_enabled = plan.mdap_enabled;
  if (plan.mdap_config !== undefined) meta.mdap_config = plan.mdap_config;
  if (plan.maker_enabled !== undefined) meta.maker_enabled = plan.maker_enabled;
  if (plan.maker_config !== undefined) meta.maker_config = plan.maker_config;
  if (plan.resource_limits !== undefined) meta.resource_limits = plan.resource_limits;
  if (plan.parallel_processing_enabled !== undefined)
    meta.parallel_processing_enabled = plan.parallel_processing_enabled;
  if (plan.max_parallel_sub_problems !== undefined)
    meta.max_parallel_sub_problems = plan.max_parallel_sub_problems;
  if (plan.learning_enabled !== undefined) meta.learning_enabled = plan.learning_enabled;
  if (plan.learning_config !== undefined) meta.learning_config = plan.learning_config;
  if (plan.content_analyzer_team_name !== undefined)
    meta.content_analyzer_team_name = plan.content_analyzer_team_name;
  if (plan.planner_team_name !== undefined) meta.planner_team_name = plan.planner_team_name;
  if (plan.assembler_team_name !== undefined) meta.assembler_team_name = plan.assembler_team_name;
  if (plan.final_red_team_gauntlet_name !== undefined)
    meta.final_red_team_gauntlet_name = plan.final_red_team_gauntlet_name;
  if (plan.final_gold_team_gauntlet_name !== undefined)
    meta.final_gold_team_gauntlet_name = plan.final_gold_team_gauntlet_name;
  if (plan.metadata !== undefined) meta.metadata = plan.metadata;
  return meta;
}

/**
 * Decomposition / Manual Review Panel.
 *
 * Picks a workflow via `listWorkflowSummaries()`, loads its `WorkflowPlanResponse`,
 * lets a human review/edit the sub-problems (the "Command" step), renders the
 * dependency graph, and saves via `updateWorkflowPlan` — re-fetching afterwards
 * because that endpoint returns only `{ message, execution_order }`.
 */
export function DecompositionPanel() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [workflowsError, setWorkflowsError] = useState<string | null>(null);
  const [workflowsLoading, setWorkflowsLoading] = useState(true);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>('');

  const [plan, setPlan] = useState<WorkflowPlanResponse | null>(null);
  const [subProblems, setSubProblems] = useState<WorkflowSubProblem[]>([]);
  const [planMeta, setPlanMeta] = useState<PlanMeta>({});
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // ---- Load workflow list ----
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setWorkflowsLoading(true);
      setWorkflowsError(null);
      try {
        const response = await openevolveApi.listWorkflowSummaries();
        if (cancelled) return;
        const summaries = response.workflows ?? [];
        setWorkflows(summaries);
        if (summaries.length > 0 && !selectedWorkflowId) {
          setSelectedWorkflowId(summaries[0].workflow_id);
        }
      } catch (error) {
        if (cancelled) return;
        setWorkflowsError(
          error instanceof Error ? error.message : 'Failed to load workflows'
        );
      } finally {
        if (!cancelled) setWorkflowsLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [selectedWorkflowId]);

  // ---- Load selected plan ----
  const loadPlan = useCallback(async (workflowId: string) => {
    setPlanLoading(true);
    setPlanError(null);
    setSaveMessage(null);
    setSaveError(null);
    try {
      const response = await openevolveApi.getWorkflowPlan(workflowId);
      setPlan(response);
      setSubProblems(response.plan.sub_problems ?? []);
      setPlanMeta(extractPlanMeta(response.plan));
    } catch (error) {
      setPlanError(error instanceof Error ? error.message : 'Failed to load decomposition plan');
      setPlan(null);
      setSubProblems([]);
    } finally {
      setPlanLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!selectedWorkflowId) return;
    void loadPlan(selectedWorkflowId);
  }, [selectedWorkflowId, loadPlan]);

  // ---- Editing handlers ----
  const updateSubProblem = useCallback((updated: WorkflowSubProblem) => {
    setSubProblems((current) =>
      current.map((sp) => (sp.id === updated.id ? updated : sp))
    );
  }, []);

  const removeSubProblem = useCallback((id: string) => {
    setSubProblems((current) => {
      const next = current.filter((sp) => sp.id !== id);
      // Drop dangling dependency references.
      return next.map((sp) => ({
        ...sp,
        dependencies: (sp.dependencies ?? []).filter((dep) => dep !== id),
      }));
    });
  }, []);

  const addSubProblem = useCallback(() => {
    setSubProblems((current) => [...current, emptySubProblem(current.length)]);
  }, []);

  // ---- Engine run results / telemetry / resource usage ----
  const [engineResults, setEngineResults] = useState<WorkflowEngineResults | null>(null);
  const [engineResultsNotRun, setEngineResultsNotRun] = useState(false);
  const [engineLoading, setEngineLoading] = useState(false);
  const [telemetry, setTelemetry] = useState<unknown>(null);
  const [resourceUsage, setResourceUsage] = useState<unknown>(null);

  const loadEngineData = useCallback(async (workflowId: string) => {
    if (!workflowId) return;
    setEngineLoading(true);
    try {
      const results = await openevolveApi.getWorkflowEngineResults(workflowId);
      setEngineResults(results ?? null);
      setEngineResultsNotRun(false);
    } catch {
      // Not-yet-run (or otherwise unavailable) engines: surface a friendly
      // prompt rather than an error toast.
      setEngineResults(null);
      setEngineResultsNotRun(true);
    }
    try {
      setTelemetry(await openevolveApi.getWorkflowTelemetry(workflowId));
    } catch {
      setTelemetry(null);
    }
    try {
      setResourceUsage(await openevolveApi.getWorkflowResourceUsage(workflowId));
    } catch {
      setResourceUsage(null);
    }
    setEngineLoading(false);
  }, []);

  useEffect(() => {
    if (!selectedWorkflowId) return;
    void loadEngineData(selectedWorkflowId);
  }, [selectedWorkflowId, loadEngineData]);

  // Live dependency graph derived from the editable sub-problems.
  const liveGraph = useMemo(() => {
    const edges: Record<string, string[]> = {};
    for (const sp of subProblems) {
      edges[sp.id] = sp.dependencies ?? [];
    }
    const nodes = subProblems.map((sp) => sp.id);
    return { nodes, edges, executionOrder: plan?.dependency_graph?.execution_order };
  }, [subProblems, plan]);

  const ids = useMemo(() => subProblems.map((sp) => sp.id), [subProblems]);

  // ---- Save ----
  const savePlan = useCallback(async () => {
    if (!selectedWorkflowId) return;
    setSaving(true);
    setSaveError(null);
    setSaveMessage(null);
    const payload: WorkflowPlanUpdateRequest = {
      sub_problems: subProblems,
      ...planMeta,
    };
    try {
      const result = await openevolveApi.updateWorkflowPlan(selectedWorkflowId, payload);
      setSaveMessage(result.message || 'Plan saved successfully.');
      // Backend returns only { message, execution_order }; re-fetch full plan.
      await loadPlan(selectedWorkflowId);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save decomposition plan');
    } finally {
      setSaving(false);
    }
  }, [selectedWorkflowId, subProblems, planMeta, loadPlan]);

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Decomposition / Manual Review</h1>
          <p className="mt-1 text-sm text-gray-400">
            Review and edit a workflow's decomposition plan (the "Command" step) before approval.
          </p>
        </div>
        <button
          type="button"
          onClick={() => selectedWorkflowId && void loadPlan(selectedWorkflowId)}
          disabled={!selectedWorkflowId || planLoading}
          className="rounded-md border border-[#3a3a3a] px-3 py-2 text-sm text-gray-200 hover:bg-[#1b1b1b] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {planLoading ? 'Reloading…' : 'Reload Plan'}
        </button>
      </div>

      {/* Workflow picker */}
      <div className="rounded-xl border border-[#2a2a2a] bg-[#111111] p-4">
        <label className="block">
          <span className="mb-2 block text-sm text-gray-300">Workflow</span>
          {workflowsLoading ? (
            <div className="text-sm text-gray-500">Loading workflows…</div>
          ) : workflowsError ? (
            <div className="rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
              {workflowsError}
            </div>
          ) : (
            <select
              value={selectedWorkflowId}
              onChange={(event) => setSelectedWorkflowId(event.target.value)}
              className="w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100"
            >
              {workflows.length === 0 && <option value="">No workflows available</option>}
              {workflows.map((wf) => (
                <option key={wf.workflow_id} value={wf.workflow_id}>
                  {wf.workflow_id} · {wf.status} · {wf.current_stage || '—'} ·{' '}
                  {Math.round((wf.progress ?? 0) * 100)}%
                </option>
              ))}
            </select>
          )}
        </label>
      </div>

      {planError && (
        <div className="rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
          {planError}
        </div>
      )}

      {planLoading && (
        <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-6 text-sm text-gray-500">
          Loading decomposition plan…
        </div>
      )}

      {!planLoading && !planError && plan && (
        <Tab.Group>
          <Tab.List className="mb-4 flex gap-2 rounded-lg bg-[#0d0d0d] p-2">
            <Tab className="rounded-md px-3 py-2 text-sm font-medium text-gray-400 ui-selected:bg-[#1b1b1b] ui-selected:text-white">
              Sub-Problems
            </Tab>
            <Tab className="rounded-md px-3 py-2 text-sm font-medium text-gray-400 ui-selected:bg-[#1b1b1b] ui-selected:text-white">
              Sovereign Settings
            </Tab>
            <Tab className="rounded-md px-3 py-2 text-sm font-medium text-gray-400 ui-selected:bg-[#1b1b1b] ui-selected:text-white">
              Engine Results
            </Tab>
          </Tab.List>
          <Tab.Panels>
            <Tab.Panel>
              <div className="grid gap-6 lg:grid-cols-2">
          {/* Sub-problems editor */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">
                Sub-Problems ({subProblems.length})
              </h2>
              <button
                type="button"
                onClick={addSubProblem}
                className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500"
              >
                + Add Sub-Problem
              </button>
            </div>

            {subProblems.length === 0 ? (
              <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-sm text-gray-500">
                No sub-problems. Use "Add Sub-Problem" to create one.
              </div>
            ) : (
              subProblems.map((sp, index) => (
                <SubProblemCard
                  key={sp.id}
                  subProblem={sp}
                  index={index}
                  allIds={ids}
                  canRemove={subProblems.length > 1}
                  onChange={updateSubProblem}
                  onRemove={() => removeSubProblem(sp.id)}
                />
              ))
            )}
          </div>

          {/* Dependency graph */}
          <div className="space-y-3">
            <DependencyGraph
              nodes={liveGraph.nodes}
              edges={liveGraph.edges}
              executionOrder={liveGraph.executionOrder}
            />

            <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4">
              <h3 className="text-base font-medium text-white">Plan Summary</h3>
              {plan.plan.problem_statement && (
                <p className="mt-2 text-sm text-gray-400">
                  <span className="text-gray-500">Problem statement:</span>{' '}
                  {plan.plan.problem_statement}
                </p>
              )}
              <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-gray-400">
                <div>
                  <dt className="text-gray-600">Workflow ID</dt>
                  <dd className="font-mono text-gray-200">{plan.workflow_id}</dd>
                </div>
                <div>
                  <dt className="text-gray-600">Sub-problems</dt>
                  <dd className="text-gray-200">{subProblems.length}</dd>
                </div>
                <div>
                  <dt className="text-gray-600">Parallel processing</dt>
                  <dd className="text-gray-200">
                    {planMeta.parallel_processing_enabled ? 'enabled' : 'disabled'}
                  </dd>
                </div>
                <div>
                  <dt className="text-gray-600">Max refinement loops</dt>
                  <dd className="text-gray-200">
                    {planMeta.max_refinement_loops ?? '—'}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
            </Tab.Panel>
            <Tab.Panel>
              <WorkflowSettingsPanel workflowId={selectedWorkflowId} />
            </Tab.Panel>
            <Tab.Panel>
              <EngineResultsView
                workflowId={selectedWorkflowId}
                results={engineResults}
                notRun={engineResultsNotRun}
                loading={engineLoading}
                telemetry={telemetry}
                resourceUsage={resourceUsage}
                onReload={() => void loadEngineData(selectedWorkflowId)}
              />
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      )}

      {/* Save bar */}
      {!planLoading && plan && (
        <div className="sticky bottom-0 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#2a2a2a] bg-[#111111] p-4">
          <div className="text-sm">
            {saveMessage && <span className="text-emerald-300">{saveMessage}</span>}
            {saveError && <span className="text-red-300">{saveError}</span>}
            {!saveMessage && !saveError && (
              <span className="text-gray-500">
                Edits are kept locally until you save the plan.
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => void savePlan()}
            disabled={saving}
            className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save Plan'}
          </button>
        </div>
      )}
    </section>
  );
}

/**
 * Renders the engine-backed decomposition-workflow run results, telemetry, and
 * resource usage fetched from the `:8000` proxy routes. Tolerant of the varying
 * shapes the engine returns and of the not-yet-run case.
 */
function EngineResultsView({
  workflowId,
  results,
  notRun,
  loading,
  telemetry,
  resourceUsage,
  onReload,
}: {
  workflowId: string;
  results: WorkflowEngineResults | null;
  notRun: boolean;
  loading: boolean;
  telemetry: unknown;
  resourceUsage: unknown;
  onReload: () => void;
}) {
  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Engine Run Results</h2>
        <button
          type="button"
          onClick={onReload}
          disabled={!workflowId || loading}
          className="rounded-md border border-[#3a3a3a] px-3 py-2 text-sm text-gray-200 hover:bg-[#1b1b1b] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Reload Engine Data'}
        </button>
      </div>

      {loading && (
        <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-6 text-sm text-gray-500">
          Loading engine results…
        </div>
      )}

      {!loading && notRun && (
        <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-6 text-sm text-gray-500">
          Run this workflow to see results.
        </div>
      )}

      {!loading && !notRun && results && (
        <div className="space-y-4">
          {results.error && (
            <div className="rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
              {results.error}
            </div>
          )}

          <section className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-200">Final Solution</h3>
            {results.final_solution == null ? (
              <p className="text-sm text-gray-500">No final solution available yet.</p>
            ) : (
              <SolutionView value={results.final_solution} />
            )}
          </section>

          <section className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-200">Sub-Problems</h3>
            <SubProblemsView value={results.sub_problems} />
          </section>

          {results.statistics != null && (
            <section className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-200">Statistics</h3>
              <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-xs text-gray-300">
                {JSON.stringify(results.statistics, null, 2)}
              </pre>
            </section>
          )}

          <section className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-200">Telemetry</h3>
            {telemetry == null ? (
              <p className="text-sm text-gray-500">No telemetry available.</p>
            ) : (
              <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-xs text-gray-300">
                {JSON.stringify(telemetry, null, 2)}
              </pre>
            )}
          </section>

          <section className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-200">Resource Usage</h3>
            {resourceUsage == null ? (
              <p className="text-sm text-gray-500">No resource usage available.</p>
            ) : (
              <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-xs text-gray-300">
                {JSON.stringify(resourceUsage, null, 2)}
              </pre>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

/** Renders a final_solution value that may be a string or a content-bearing object. */
function SolutionView({ value }: { value: unknown }) {
  if (typeof value === 'string') {
    return (
      <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-sm text-gray-300">
        {value}
      </pre>
    );
  }
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const content = typeof obj.content === 'string' ? obj.content : JSON.stringify(value, null, 2);
    const by = typeof obj.generated_by === 'string' ? obj.generated_by : undefined;
    return (
      <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4">
        {by && <p className="mb-2 text-xs text-gray-500">{by}</p>}
        <pre className="whitespace-pre-wrap text-sm text-gray-300">{content}</pre>
      </div>
    );
  }
  return (
    <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-sm text-gray-300">
      {String(value)}
    </pre>
  );
}

/** Renders sub_problems that may be an array of objects or a record keyed by id. */
function SubProblemsView({ value }: { value: unknown }) {
  if (value == null) {
    return <p className="text-sm text-gray-500">No sub-problem solutions available.</p>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <p className="text-sm text-gray-500">No sub-problem solutions available.</p>;
    }
    return (
      <div className="space-y-3">
        {value.map((item, index) => (
          <SolutionCard key={index} label={`#${index + 1}`} item={item} />
        ))}
      </div>
    );
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return <p className="text-sm text-gray-500">No sub-problem solutions available.</p>;
    }
    return (
      <div className="space-y-3">
        {entries.map(([key, item]) => (
          <SolutionCard key={key} label={key} item={item} />
        ))}
      </div>
    );
  }
  return (
    <pre className="whitespace-pre-wrap rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-sm text-gray-300">
      {String(value)}
    </pre>
  );
}

function SolutionCard({ label, item }: { label: string; item: unknown }) {
  let title = label;
  let body: unknown = item;
  if (item && typeof item === 'object') {
    const obj = item as Record<string, unknown>;
    if (typeof obj.id === 'string') title = obj.id;
    else if (typeof obj.description === 'string') title = `${label}: ${obj.description}`;
    if (obj.content != null) body = obj.content;
    else if (obj.solution != null) body = obj.solution;
  }
  return (
    <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4">
      <p className="mb-2 text-xs font-medium text-gray-400">{title}</p>
      {typeof body === 'string' ? (
        <pre className="whitespace-pre-wrap text-sm text-gray-300">{body}</pre>
      ) : (
        <pre className="whitespace-pre-wrap text-sm text-gray-300">
          {JSON.stringify(body, null, 2)}
        </pre>
      )}
    </div>
  );
}
