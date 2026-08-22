/**
 * WorkflowSettingsPanel
 *
 * Sovereign-Grade Decomposition Workflow settings editor. Loads the current
 * `WorkflowSettings` for a selected workflow, renders a grouped, typed control
 * surface (General, MAKER, MDAP, Auto-approval, Parallel & Resources, Learning,
 * Distributed, Knowledge Engine, Red-flag rules, Web3, Formal Verification), and
 * persists via `PUT /workflows/{id}/settings` (Save) or saves then runs via
 * `POST /workflows/{id}/run` (Save & Run).
 *
 * Styling mirrors `DecompositionPanel` (dark surfaces, `#303030` borders) and
 * reuses the HeadlessUI `Tab.Group` layout used elsewhere in BubbleLab.
 */

import { useEffect, useState } from 'react';
import { Tab } from '@headlessui/react';
import { ToggleSwitch } from '@/components/common/ToggleSwitch';
import {
  defaultWorkflowSettings,
  type WorkflowSettings,
} from '@/services/openevolveApi';
import {
  useRunWorkflow,
  useUpdateWorkflowSettings,
  useWorkflowSettings,
} from '@/hooks/use-workflow-settings';

const JSON_KEYS = [
  'mdap_config',
  'maker_config',
  'auto_approval_criteria',
  'learning_config',
] as const;

type JsonKey = (typeof JSON_KEYS)[number];

const inputClass =
  'w-full rounded-md border border-[#303030] bg-[#0f0f0f] px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500';
const labelClass = 'mb-1 block text-xs font-medium text-gray-400';
const sectionClass = 'space-y-4 rounded-xl border border-[#2a2a2a] bg-[#111111] p-4';

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className={labelClass}>{label}</span>
      {children}
    </label>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        className={inputClass}
        value={Number.isFinite(value) ? value : 0}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </Field>
  );
}

function TextInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <Field label={label}>
      <input
        type="text"
        className={inputClass}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </Field>
  );
}

function SelectInput({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <Field label={label}>
      <select className={inputClass} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </Field>
  );
}

function JsonArea({
  label,
  text,
  error,
  onChange,
}: {
  label: string;
  text: string;
  error?: string;
  onChange: (v: string) => void;
}) {
  return (
    <Field label={label}>
      <textarea
        className={`${inputClass} font-mono`}
        rows={5}
        value={text}
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
      />
      {error && <p className="mt-1 text-xs text-red-400">Invalid JSON: {error}</p>}
    </Field>
  );
}

function TabButton({ label }: { label: string }) {
  return (
    <Tab className="rounded-md px-3 py-2 text-xs font-medium text-gray-400 ui-selected:bg-[#1b1b1b] ui-selected:text-white">
      {label}
    </Tab>
  );
}

function WorkflowSettingsPanelInner({ workflowId }: { workflowId: string }) {
  const { data, isLoading, isError, error } = useWorkflowSettings(workflowId);
  const updateMutation = useUpdateWorkflowSettings(workflowId);
  const runMutation = useRunWorkflow();

  const [settings, setSettings] = useState<WorkflowSettings>(defaultWorkflowSettings);
  const [jsonText, setJsonText] = useState<Record<JsonKey, string>>({
    mdap_config: '{}',
    maker_config: '{}',
    auto_approval_criteria: '{}',
    learning_config: '{}',
  });
  const [jsonErrors, setJsonErrors] = useState<Record<string, string>>({});
  const [status, setStatus] = useState<{ kind: 'ok' | 'err'; msg: string } | null>(null);

  useEffect(() => {
    if (!data) return;
    const merged = { ...defaultWorkflowSettings, ...data };
    setSettings(merged);
    const next = {} as Record<JsonKey, string>;
    for (const k of JSON_KEYS) {
      next[k] = JSON.stringify((merged as Record<string, unknown>)[k] ?? {}, null, 2);
    }
    setJsonText(next);
    setJsonErrors({});
    setStatus(null);
  }, [data]);

  const update = (patch: Partial<WorkflowSettings>) =>
    setSettings((s) => ({ ...s, ...patch }));

  const updateGroup = <K extends keyof WorkflowSettings>(
    key: K,
    patch: Partial<WorkflowSettings[K]>
  ) =>
    setSettings((s) => ({
      ...s,
      [key]: { ...(s[key] as object), ...(patch as object) },
    }));

  const collect = (): WorkflowSettings | null => {
    const errors: Record<string, string> = {};
    const parsed: Record<string, unknown> = {};
    for (const k of JSON_KEYS) {
      const raw = jsonText[k]?.trim();
      if (!raw) {
        parsed[k] = {};
        continue;
      }
      try {
        parsed[k] = JSON.parse(raw);
      } catch (e) {
        errors[k] = e instanceof Error ? e.message : 'parse error';
      }
    }
    if (Object.keys(errors).length > 0) {
      setJsonErrors(errors);
      return null;
    }
    setJsonErrors({});
    return { ...settings, ...parsed } as WorkflowSettings;
  };

  const onSave = async () => {
    const payload = collect();
    if (!payload) return;
    setStatus(null);
    try {
      await updateMutation.mutateAsync(payload);
      setStatus({ kind: 'ok', msg: 'Settings saved.' });
    } catch (e) {
      setStatus({ kind: 'err', msg: e instanceof Error ? e.message : 'Save failed.' });
    }
  };

  const onSaveAndRun = async () => {
    const payload = collect();
    if (!payload) return;
    setStatus(null);
    try {
      const saved = await updateMutation.mutateAsync(payload);
      await runMutation.mutateAsync({ workflowId, config: saved });
      setStatus({ kind: 'ok', msg: 'Settings saved and workflow run started.' });
    } catch (e) {
      setStatus({ kind: 'err', msg: e instanceof Error ? e.message : 'Run failed.' });
    }
  };

  if (isLoading) {
    return (
      <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-6 text-sm text-gray-500">
        Loading workflow settings…
      </div>
    );
  }

  if (isError) {
    return (
      <div className="rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300">
        {error instanceof Error ? error.message : 'Failed to load settings.'}
      </div>
    );
  }

  const busy = updateMutation.isPending || runMutation.isPending;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-white">Sovereign Workflow Settings</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void onSave()}
            disabled={busy}
            className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {updateMutation.isPending ? 'Saving…' : 'Save settings'}
          </button>
          <button
            type="button"
            onClick={() => void onSaveAndRun()}
            disabled={busy}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {runMutation.isPending ? 'Running…' : 'Save & Run'}
          </button>
        </div>
      </div>

      {status && (
        <div
          className={
            status.kind === 'ok'
              ? 'rounded-md border border-emerald-900/60 bg-emerald-950/30 px-3 py-2 text-sm text-emerald-300'
              : 'rounded-md border border-red-900/60 bg-red-950/30 px-3 py-2 text-sm text-red-300'
          }
        >
          {status.msg}
        </div>
      )}

      <Tab.Group>
        <Tab.List className="flex flex-wrap gap-1 rounded-lg bg-[#0d0d0d] p-2">
          <TabButton label="General" />
          <TabButton label="MAKER" />
          <TabButton label="MDAP" />
          <TabButton label="Auto-approval" />
          <TabButton label="Parallel & Resources" />
          <TabButton label="Learning" />
          <TabButton label="Distributed" />
          <TabButton label="Knowledge Engine" />
          <TabButton label="Red-flag rules" />
          <TabButton label="Web3" />
          <TabButton label="Formal Verification" />
        </Tab.List>

        <Tab.Panels className="mt-4">
          {/* General */}
          <Tab.Panel>
            <div className={sectionClass}>
              <NumberInput
                label="Max refinement loops"
                value={settings.max_refinement_loops}
                onChange={(v) => update({ max_refinement_loops: v })}
              />
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Entanglement strict mode"
                  description="When on, sub-problem entanglements are enforced strictly."
                  checked={settings.entanglement_strict_mode}
                  onChange={(v) => update({ entanglement_strict_mode: v })}
                />
              </div>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3 text-xs text-gray-400">
                Circular dependency guard: <span className="text-gray-200">always on</span> (engine-enforced). This
                setting is read-only.
              </div>
            </div>
          </Tab.Panel>

          {/* MAKER */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="MAKER enabled"
                  checked={settings.maker_enabled}
                  onChange={(v) => update({ maker_enabled: v })}
                />
              </div>
              <JsonArea
                label="MAKER config (JSON)"
                text={jsonText.maker_config}
                error={jsonErrors.maker_config}
                onChange={(v) => setJsonText((t) => ({ ...t, maker_config: v }))}
              />
            </div>
          </Tab.Panel>

          {/* MDAP */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="MDAP enabled"
                  checked={settings.mdap_enabled}
                  onChange={(v) => update({ mdap_enabled: v })}
                />
              </div>
              <JsonArea
                label="MDAP config (JSON)"
                text={jsonText.mdap_config}
                error={jsonErrors.mdap_config}
                onChange={(v) => setJsonText((t) => ({ ...t, mdap_config: v }))}
              />
            </div>
          </Tab.Panel>

          {/* Auto-approval */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Auto-approval enabled"
                  checked={settings.auto_approval_enabled}
                  onChange={(v) => update({ auto_approval_enabled: v })}
                />
              </div>
              <JsonArea
                label="Auto-approval criteria (JSON)"
                text={jsonText.auto_approval_criteria}
                error={jsonErrors.auto_approval_criteria}
                onChange={(v) => setJsonText((t) => ({ ...t, auto_approval_criteria: v }))}
              />
            </div>
          </Tab.Panel>

          {/* Parallel & Resources */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Parallel processing enabled"
                  checked={settings.parallel_processing_enabled}
                  onChange={(v) => update({ parallel_processing_enabled: v })}
                />
              </div>
              <NumberInput
                label="Max parallel sub-problems"
                value={settings.max_parallel_sub_problems}
                onChange={(v) => update({ max_parallel_sub_problems: v })}
              />
              <div className="grid gap-4 sm:grid-cols-2">
                <NumberInput
                  label="Resource: total tokens"
                  value={settings.resource_limits.total_tokens}
                  onChange={(v) => updateGroup('resource_limits', { total_tokens: v })}
                />
                <NumberInput
                  label="Resource: total time (s)"
                  value={settings.resource_limits.total_time_seconds}
                  onChange={(v) => updateGroup('resource_limits', { total_time_seconds: v })}
                />
                <NumberInput
                  label="Resource: total steps"
                  value={settings.resource_limits.total_steps}
                  onChange={(v) => updateGroup('resource_limits', { total_steps: v })}
                />
                <NumberInput
                  label="Resource: max parallel"
                  value={settings.resource_limits.max_parallel}
                  onChange={(v) => updateGroup('resource_limits', { max_parallel: v })}
                />
                <NumberInput
                  label="Resource: tokens / sub-problem"
                  value={settings.resource_limits.tokens_per_sub_problem}
                  onChange={(v) => updateGroup('resource_limits', { tokens_per_sub_problem: v })}
                />
                <NumberInput
                  label="Resource: time / sub-problem (s)"
                  value={settings.resource_limits.time_per_sub_problem}
                  onChange={(v) =>
                    updateGroup('resource_limits', { time_per_sub_problem: v })
                  }
                />
                <NumberInput
                  label="Resource: steps / sub-problem"
                  value={settings.resource_limits.steps_per_sub_problem}
                  onChange={(v) =>
                    updateGroup('resource_limits', { steps_per_sub_problem: v })
                  }
                />
              </div>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Resource: allow overshoot"
                  checked={settings.resource_limits.allow_overshoot}
                  onChange={(v) => updateGroup('resource_limits', { allow_overshoot: v })}
                />
              </div>
            </div>
          </Tab.Panel>

          {/* Learning */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Learning enabled"
                  checked={settings.learning_enabled}
                  onChange={(v) => update({ learning_enabled: v })}
                />
              </div>
              <JsonArea
                label="Learning config (JSON)"
                text={jsonText.learning_config}
                error={jsonErrors.learning_config}
                onChange={(v) => setJsonText((t) => ({ ...t, learning_config: v }))}
              />
            </div>
          </Tab.Panel>

          {/* Distributed */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Distributed"
                  checked={settings.distributed}
                  onChange={(v) => update({ distributed: v })}
                />
              </div>
              <SelectInput
                label="Distributed backend"
                value={settings.distributed_backend}
                options={['local', 'multiprocessing']}
                onChange={(v) => update({ distributed_backend: v })}
              />
            </div>
          </Tab.Panel>

          {/* Knowledge Engine */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Knowledge engine enabled"
                  checked={settings.knowledge_engine_enabled}
                  onChange={(v) => update({ knowledge_engine_enabled: v })}
                />
              </div>
              <TextInput
                label="Knowledge engine path"
                value={settings.knowledge_engine_path}
                onChange={(v) => update({ knowledge_engine_path: v })}
                placeholder="/path/to/knowledge"
              />
            </div>
          </Tab.Panel>

          {/* Red-flag rules */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="grid gap-4 sm:grid-cols-2">
                <NumberInput
                  label="Red flag: max tokens"
                  value={settings.red_flag_rules.max_tokens}
                  onChange={(v) => updateGroup('red_flag_rules', { max_tokens: v })}
                />
                <NumberInput
                  label="Red flag: max characters"
                  value={settings.red_flag_rules.max_characters}
                  onChange={(v) => updateGroup('red_flag_rules', { max_characters: v })}
                />
                <NumberInput
                  label="Red flag: min confidence"
                  value={settings.red_flag_rules.min_confidence}
                  step={0.01}
                  onChange={(v) => updateGroup('red_flag_rules', { min_confidence: v })}
                />
              </div>
              <Field label="Red flag: blocked patterns (comma-separated)">
                <input
                  type="text"
                  className={inputClass}
                  value={settings.red_flag_rules.blocked_patterns.join(', ')}
                  onChange={(e) =>
                    updateGroup('red_flag_rules', {
                      blocked_patterns: e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </Field>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Red flag: require schema match"
                  checked={settings.red_flag_rules.require_schema_match}
                  onChange={(v) => updateGroup('red_flag_rules', { require_schema_match: v })}
                />
              </div>
            </div>
          </Tab.Panel>

          {/* Web3 */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Web3 enabled"
                  checked={settings.web3.enabled}
                  onChange={(v) => updateGroup('web3', { enabled: v })}
                />
              </div>
              <TextInput
                label="Web3: project path"
                value={settings.web3.project_path}
                onChange={(v) => updateGroup('web3', { project_path: v })}
                placeholder="/path/to/web3/project"
              />
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Web3: run fuzzing"
                  checked={settings.web3.run_fuzzing}
                  onChange={(v) => updateGroup('web3', { run_fuzzing: v })}
                />
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <NumberInput
                  label="Web3: slither timeout (s)"
                  value={settings.web3.slither_timeout_seconds}
                  onChange={(v) => updateGroup('web3', { slither_timeout_seconds: v })}
                />
                <NumberInput
                  label="Web3: forge timeout (s)"
                  value={settings.web3.forge_timeout_seconds}
                  onChange={(v) => updateGroup('web3', { forge_timeout_seconds: v })}
                />
              </div>
            </div>
          </Tab.Panel>

          {/* Formal Verification */}
          <Tab.Panel>
            <div className={sectionClass}>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Z3 enabled"
                  checked={settings.formal_verification.z3_enabled}
                  onChange={(v) => updateGroup('formal_verification', { z3_enabled: v })}
                />
              </div>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="LeanAide enabled"
                  checked={settings.formal_verification.leanaide_enabled}
                  onChange={(v) => updateGroup('formal_verification', { leanaide_enabled: v })}
                />
              </div>
              <div className="rounded-md border border-[#2a2a2a] bg-[#0d0d0d] p-3">
                <ToggleSwitch
                  label="Formal verification enabled"
                  checked={settings.formal_verification.formal_verification_enabled}
                  onChange={(v) =>
                    updateGroup('formal_verification', { formal_verification_enabled: v })
                  }
                />
              </div>
              <SelectInput
                label="Formal verification mode"
                value={settings.formal_verification.formal_verification_mode}
                options={['off', 'z3', 'leanaide', 'both']}
                onChange={(v) => updateGroup('formal_verification', { formal_verification_mode: v })}
              />
            </div>
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </section>
  );
}

export function WorkflowSettingsPanel({ workflowId }: { workflowId: string }) {
  if (!workflowId) {
    return (
      <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-6 text-sm text-gray-500">
        Select a workflow to configure its Sovereign-Grade settings.
      </div>
    );
  }
  return <WorkflowSettingsPanelInner workflowId={workflowId} />;
}
