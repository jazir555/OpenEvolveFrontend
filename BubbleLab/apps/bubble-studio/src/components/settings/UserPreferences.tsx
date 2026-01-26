/**
 * UserPreferences Component
 * User-specific application settings
 */

import { useUIStore } from '../../stores/uiStore';
import { ToggleSwitch } from '../common/ToggleSwitch';
import { Select } from '../common/Select';

export function UserPreferences() {
  const {
    darkMode,
    setDarkMode,
    sidebarCollapsed,
    setSidebarCollapsed,
    notificationsEnabled,
    setNotificationsEnabled,
    autoSaveEnabled,
    setAutoSaveEnabled,
    theme,
    setTheme,
  } = useUIStore();

  const themeOptions = [
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' },
    { value: 'system', label: 'System' },
  ];

  return (
    <div className="space-y-6">
      {/* Appearance */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Appearance
        </h3>
        <div className="space-y-4">
          <ToggleSwitch
            checked={darkMode}
            onChange={setDarkMode}
            label="Dark Mode"
            description="Use dark theme across the application"
          />

          <Select
            label="Theme"
            value={theme}
            onChange={setTheme}
            options={themeOptions}
            description="Choose your preferred color theme"
          />

          <ToggleSwitch
            checked={sidebarCollapsed}
            onChange={setSidebarCollapsed}
            label="Collapse Sidebar"
            description="Start with sidebar collapsed by default"
          />
        </div>
      </div>

      {/* Behavior */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Behavior
        </h3>
        <div className="space-y-4">
          <ToggleSwitch
            checked={autoSaveEnabled}
            onChange={setAutoSaveEnabled}
            label="Auto-Save"
            description="Automatically save workflow configurations"
          />

          <ToggleSwitch
            checked={notificationsEnabled}
            onChange={setNotificationsEnabled}
            label="Notifications"
            description="Enable desktop notifications for workflow events"
          />
        </div>
      </div>
    </div>
  );
}
