/**
 * Settings Page
 * Application settings configuration
 */

import { createFileRoute } from '@tanstack/react-router';
import { SettingsPanel } from '../../components/settings/SettingsPanel';

export const Route = createFileRoute('/oe-settings')({
  component: SettingsPage,
});

function SettingsPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Configure your application preferences and LLM settings
        </p>
      </div>

      {/* Settings Panel */}
      <SettingsPanel />
    </div>
  );
}
