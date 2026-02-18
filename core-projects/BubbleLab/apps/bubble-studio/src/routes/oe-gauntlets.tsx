/**
 * Gauntlets Management Page
 * Create, edit, and delete gauntlets
 */

import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { useGauntlets, useCreateGauntlet } from '../hooks/use-gauntlets-api';
import { GauntletList } from '../components/gauntlet/GauntletList';
import { GauntletEditorModal } from '../components/gauntlet/GauntletEditorModal';
import { CreateGauntletRequest } from '../types/api';

export const Route = createFileRoute('/oe-gauntlets')({
  component: GauntletsPage,
});

function GauntletsPage() {
  const { data: gauntlets, isLoading, error } = useGauntlets();
  const createGauntlet = useCreateGauntlet();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSaveGauntlet = async (gauntletData: CreateGauntletRequest) => {
    await createGauntlet.mutateAsync(gauntletData);
    setIsModalOpen(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Gauntlets</h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Configure validation gauntlets for quality assurance
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
          Create Gauntlet
        </button>
      </div>

      <GauntletList gauntlets={gauntlets || []} isLoading={isLoading} error={error} />

      <GauntletEditorModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveGauntlet}
        isSaving={createGauntlet.isPending}
      />
    </div>
  );
}
