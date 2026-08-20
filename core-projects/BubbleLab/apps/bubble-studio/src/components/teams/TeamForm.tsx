/**
 * TeamForm Component
 *
 * Controlled create/edit form for a canonical `Team` definition. Used by
 * `TeamManager` inside a modal. On submit it calls back with a fully-formed
 * `Team` payload (built from `createDefaultTeam` / `createDefaultModelConfig`
 * so every canonical field is present) for the parent to persist via
 * `openevolveApi.createTeamDefinition` / `updateTeam`.
 */

import { useState } from 'react';
import {
  createDefaultModelConfig,
  createDefaultTeam,
  type ModelConfig,
  type Team,
  type TeamRole,
} from '@/types/openevolve';
import { Input } from '@/components/common/Input';
import { Select } from '@/components/common/Select';
import { Button } from '@/components/common/Button';
import { Trash2, Plus } from 'lucide-react';

const TEAM_ROLES: TeamRole[] = ['Blue', 'Red', 'Gold'];

interface TeamFormProps {
  /** Existing team when editing; omitted when creating a new team. */
  team?: Team | null;
  /** True when editing an existing team (name is a fixed server key). */
  isEdit?: boolean;
  isSubmitting?: boolean;
  submitError?: string | null;
  onSubmit: (team: Team) => void;
  onCancel: () => void;
}

/** Coerce empty optional string fields to `undefined` for the wire payload. */
function normalizeMember(member: ModelConfig): ModelConfig {
  const next: ModelConfig = { ...member };
  if (next.api_base === '') next.api_base = undefined;
  return next;
}

export function TeamForm({
  team,
  isEdit = false,
  isSubmitting = false,
  submitError,
  onSubmit,
  onCancel,
}: TeamFormProps) {
  const [formTeam, setFormTeam] = useState<Team>(() =>
    team ? team : createDefaultTeam()
  );
  const [nameError, setNameError] = useState<string | null>(null);

  const updateField = <K extends keyof Team>(key: K, value: Team[K]) => {
    setFormTeam((prev) => ({ ...prev, [key]: value }));
  };

  const updateMember = (index: number, partial: Partial<ModelConfig>) => {
    setFormTeam((prev) => ({
      ...prev,
      members: prev.members.map((m, i) =>
        i === index ? { ...m, ...partial } : m
      ),
    }));
  };

  const addMember = () => {
    setFormTeam((prev) => ({
      ...prev,
      members: [...prev.members, createDefaultModelConfig()],
    }));
  };

  const removeMember = (index: number) => {
    setFormTeam((prev) => ({
      ...prev,
      members: prev.members.filter((_, i) => i !== index),
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formTeam.name.trim()) {
      setNameError('Team name is required.');
      return;
    }
    setNameError(null);

    const payload: Team = {
      ...formTeam,
      name: formTeam.name.trim(),
      description: formTeam.description?.trim() || undefined,
      members: formTeam.members.map(normalizeMember),
    };
    onSubmit(payload);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Core fields */}
      <div className="space-y-4">
        <Input
          label="Team Name"
          value={formTeam.name}
          onChange={(e) => updateField('name', e.target.value)}
          placeholder="e.g. blue-solver-team"
          disabled={isEdit}
          error={nameError ?? undefined}
          helperText={
            isEdit ? 'Team name is fixed and used as the server key.' : undefined
          }
        />

        <Select
          label="Role"
          value={formTeam.role}
          onChange={(e) => updateField('role', e.target.value as TeamRole)}
          options={TEAM_ROLES.map((role) => ({ value: role, label: role }))}
        />

        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
            Description
          </label>
          <textarea
            value={formTeam.description ?? ''}
            onChange={(e) => updateField('description', e.target.value)}
            rows={2}
            placeholder="Optional description of this team's purpose"
            className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          />
        </div>
      </div>

      {/* Members */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
            Members ({formTeam.members.length})
          </h4>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={addMember}
          >
            <Plus className="mr-1 h-4 w-4" />
            Add Member
          </Button>
        </div>

        {formTeam.members.length === 0 ? (
          <p className="rounded-md border border-dashed border-gray-300 p-4 text-center text-sm text-gray-500 dark:border-gray-600 dark:text-gray-400">
            No members yet. Add at least one model configuration.
          </p>
        ) : (
          <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
            {formTeam.members.map((member, index) => (
              <div
                key={index}
                className="rounded-lg border border-gray-200 bg-gray-50 p-3 dark:border-gray-700 dark:bg-gray-900/40"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Member {index + 1}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeMember(index)}
                    className="inline-flex items-center text-red-600 hover:text-red-700 dark:text-red-400"
                    aria-label={`Remove member ${index + 1}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Input
                    label="Model ID"
                    value={member.model_id}
                    onChange={(e) =>
                      updateMember(index, { model_id: e.target.value })
                    }
                    placeholder="e.g. gpt-4o"
                  />
                  <Input
                    label="API Base (optional)"
                    value={member.api_base ?? ''}
                    onChange={(e) =>
                      updateMember(index, { api_base: e.target.value })
                    }
                    placeholder="https://api.openai.com/v1"
                  />
                  <Input
                    label="Temperature"
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={member.temperature ?? ''}
                    onChange={(e) =>
                      updateMember(index, {
                        temperature: e.target.value === '' ? undefined : Number(e.target.value),
                      })
                    }
                  />
                  <Input
                    label="Max Tokens"
                    type="number"
                    step="1"
                    min="1"
                    value={member.max_tokens ?? ''}
                    onChange={(e) =>
                      updateMember(index, {
                        max_tokens: e.target.value === '' ? undefined : Number(e.target.value),
                      })
                    }
                  />
                  <div className="sm:col-span-2">
                    <Input
                      label="API Key (optional)"
                      type="password"
                      value={member.api_key ?? ''}
                      onChange={(e) =>
                        updateMember(index, { api_key: e.target.value })
                      }
                      placeholder="sk-..."
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {submitError && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {submitError}
        </div>
      )}

      <div className="flex justify-end gap-2 border-t border-gray-200 pt-4 dark:border-gray-700">
        <Button type="button" variant="ghost" onClick={onCancel} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" isLoading={isSubmitting}>
          {isEdit ? 'Save Changes' : 'Create Team'}
        </Button>
      </div>
    </form>
  );
}
