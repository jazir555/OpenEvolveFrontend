import { useCallback, useEffect, useMemo, useState } from 'react';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  WorkflowDecompositionPlan,
  WorkflowPlanResponse,
  WorkflowPlanUpdateRequest,
  WorkflowSubProblem,
  WorkflowSummary,
} from '@/types/openevolve';
import { SubProblemCard } from './SubProblemCard';
import { DependencyGraph } from './DependencyGraph';

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
