/**
 * Gauntlet List Component
 * Displays all gauntlets with actions
 */

import { Link } from '@tanstack/react-router';
import { Gauntlet } from '../../types/api';

interface GauntletListProps {
  gauntlets: Gauntlet[];
  isLoading: boolean;
  error: Error | null;
}

export function GauntletList({ gauntlets, isLoading, error }: GauntletListProps) {
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
          Error loading gauntlets: {error.message}
        </p>
      </div>
    );
  }

  if (gauntlets.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-600">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
          No gauntlets
        </h3>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Get started by creating a new gauntlet.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {gauntlets.map((gauntlet) => (
        <GauntletCard key={gauntlet.id} gauntlet={gauntlet} />
      ))}
    </div>
  );
}

function GauntletCard({ gauntlet }: { gauntlet: Gauntlet }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition-shadow dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {gauntlet.name}
          </h3>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            {gauntlet.description || 'No description'}
          </p>
          <div className="mt-3 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>{gauntlet.rounds?.length || 0} rounds</span>
            <span>•</span>
            <span>Created {new Date(gauntlet.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex gap-2">
        <Link
          to="/oe-gauntlets/$gauntletId"
          params={{ gauntletId: gauntlet.id }}
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
