/**
 * Teams Management Page
 * Create, edit, and delete teams
 */

import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { useTeams, useCreateTeam } from '../hooks/use-teams-api';
import { TeamList } from '../components/team/TeamList';
import { TeamEditorModal } from '../components/team/TeamEditorModal';
import { CreateTeamRequest } from '../types/api';

export const Route = createFileRoute('/oe-teams')({
  component: TeamsPage,
});

function TeamsPage() {
  const { data: teams, isLoading, error } = useTeams();
  const createTeam = useCreateTeam();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSaveTeam = async (teamData: CreateTeamRequest) => {
    await createTeam.mutateAsync(teamData);
    setIsModalOpen(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Teams</h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Configure OpenEvolve teams for workflow execution
          </p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z"
              clipRule="evenodd"
            />
          </svg>
          Create Team
        </button>
      </div>

      <TeamList teams={teams || []} isLoading={isLoading} error={error} />

      <TeamEditorModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveTeam}
        isSaving={createTeam.isPending}
      />
    </div>
  );
}
