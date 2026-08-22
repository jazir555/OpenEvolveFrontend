/**
 * TeamManager Component
 *
 * Main view for managing AI teams (Blue/Red/Gold roles) used by the OpenEvolve
 * decomposition workflow. Loads `openevolveApi.listTeamSummaries()` on mount,
 * renders a table of teams with create/edit/delete actions, and persists
 * changes through the canonical team client methods.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  type Team,
  type TeamRole,
  type TeamSummary,
} from '@/types/openevolve';
import { openevolveApi } from '@/services/openevolveApi';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { Modal } from '@/components/common/Modal';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { TeamForm } from './TeamForm';
import { Plus, Pencil, Trash2, RefreshCw, Users } from 'lucide-react';

const ROLE_BADGE_VARIANT: Record<TeamRole, 'blue' | 'red' | 'yellow'> = {
  Blue: 'blue',
  Red: 'red',
  Gold: 'yellow',
};

export function TeamManager() {
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingTeam, setEditingTeam] = useState<Team | null>(null);
  const [editingTeamId, setEditingTeamId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [deletingTeam, setDeletingTeam] = useState<TeamSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const loadTeams = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await openevolveApi.listTeamSummaries();
      setTeams(response.teams ?? []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to load teams.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadTeams();
  }, [loadTeams]);

  const openCreate = () => {
    setEditingTeam(null);
    setSubmitError(null);
    setIsFormOpen(true);
  };

  const openEdit = async (summary: TeamSummary) => {
    setSubmitError(null);
    try {
      const full = await openevolveApi.getTeamDefinition(summary.id);
      setEditingTeam(full);
      setEditingTeamId(summary.id);
      setIsFormOpen(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to load team definition.'
      );
    }
  };

  const closeForm = () => {
    if (isSubmitting) return;
    setIsFormOpen(false);
    setEditingTeam(null);
    setEditingTeamId(null);
  };

  const handleSubmit = async (team: Team) => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (editingTeamId) {
        await openevolveApi.updateTeam(editingTeamId, team);
      } else {
        await openevolveApi.createTeamDefinition(team);
      }
      setIsFormOpen(false);
      setEditingTeam(null);
      setEditingTeamId(null);
      await loadTeams();
    } catch (err) {
      setSubmitError(
        err instanceof Error ? err.message : 'Failed to save team.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const confirmDelete = async () => {
    if (!deletingTeam) return;
    setIsDeleting(true);
    try {
      await openevolveApi.deleteTeam(deletingTeam.id);
      setDeletingTeam(null);
      await loadTeams();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete team.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Team Manager
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Create and manage AI teams used by the decomposition workflow.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void loadTeams()}
            disabled={loading}
          >
            <RefreshCw className="mr-1 h-4 w-4" />
            Refresh
          </Button>
          <Button variant="primary" size="sm" onClick={openCreate}>
            <Plus className="mr-1 h-4 w-4" />
            Create Team
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-16 animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700"
            />
          ))}
        </div>
      ) : teams.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-600">
          <Users className="mx-auto h-10 w-10 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No teams yet
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Get started by creating a new team.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Role
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Members
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Description
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-900">
              {teams.map((team) => (
                <tr key={team.name} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">
                    {team.name}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-sm">
                    <Badge variant={ROLE_BADGE_VARIANT[team.role]}>
                      {team.role}
                    </Badge>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-600 dark:text-gray-300">
                    {team.member_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-300">
                    {team.description || (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right text-sm">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => void openEdit(team)}
                        aria-label={`Edit ${team.name}`}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setDeletingTeam(team)}
                        aria-label={`Delete ${team.name}`}
                      >
                        <Trash2 className="h-4 w-4 text-red-600 dark:text-red-400" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Modal
        isOpen={isFormOpen}
        onClose={closeForm}
        title={editingTeam ? `Edit Team: ${editingTeam.name}` : 'Create Team'}
        size="xl"
      >
        {isFormOpen && (
          <TeamForm
            team={editingTeam}
            isEdit={!!editingTeam}
            isSubmitting={isSubmitting}
            submitError={submitError}
            onSubmit={(team) => void handleSubmit(team)}
            onCancel={closeForm}
          />
        )}
      </Modal>

      <ConfirmDialog
        isOpen={!!deletingTeam}
        onClose={() => setDeletingTeam(null)}
        onConfirm={() => void confirmDelete()}
        title="Delete Team"
        message={
          deletingTeam
            ? `Are you sure you want to delete "${deletingTeam.name}"? This action cannot be undone.`
            : ''
        }
        confirmLabel={isDeleting ? 'Deleting...' : 'Delete'}
        variant="danger"
      />
    </div>
  );
}
