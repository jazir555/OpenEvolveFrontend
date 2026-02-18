/**
 * Team List Component
 * Displays all teams with actions
 */

import { Link } from '@tanstack/react-router';
import { Team } from '../../types/api';

interface TeamListProps {
  teams: Team[];
  isLoading: boolean;
  error: Error | null;
}

export function TeamList({ teams, isLoading, error }: TeamListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="h-24 animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700"
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-900/20">
        <p className="text-red-800 dark:text-red-400">
          Error loading teams: {error.message}
        </p>
      </div>
    );
  }

  if (teams.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-600">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
          <path
            fillRule="evenodd"
            d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z"
            clipRule="evenodd"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
          No teams
        </h3>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Get started by creating a new team.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {teams.map((team) => (
        <TeamCard key={team.id} team={team} />
      ))}
    </div>
  );
}

function TeamCard({ team }: { team: Team }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition-shadow dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {team.name}
          </h3>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            {team.description || 'No description'}
          </p>
          <div className="mt-3 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>{team.members?.length || 0} members</span>
            <span>•</span>
            <span>Created {new Date(team.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex gap-2">
        <Link
          to="/oe-teams/$teamId"
          params={{ teamId: team.id }}
          className="inline-flex items-center rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          Edit
        </Link>
        <button className="inline-flex items-center rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700">
          Delete
        </button>
      </div>
    </div>
  );
}
