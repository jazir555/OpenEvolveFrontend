/**
 * StatCard Component
 * Compact summary card used in the analytics dashboard.
 */

import type { ReactNode } from 'react';

export type StatAccent = 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'gray';

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  accent?: StatAccent;
  hint?: string;
}

const accentStyles: Record<StatAccent, string> = {
  blue: 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400',
  green: 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400',
  red: 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400',
  yellow: 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400',
  purple: 'text-purple-600 bg-purple-50 dark:bg-purple-900/20 dark:text-purple-400',
  gray: 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400',
};

export function StatCard({ label, value, icon, accent = 'blue', hint }: StatCardProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{label}</p>
          <p className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">
            {value}
          </p>
          {hint && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{hint}</p>
          )}
        </div>
        {icon && (
          <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${accentStyles[accent]}`}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
