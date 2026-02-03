/**
 * Quick Actions Component
 * Primary action buttons for common tasks
 */

import { Link } from '@tanstack/react-router';

export function QuickActions() {
  const actions = [
    {
      name: 'Create Workflow',
      description: 'Start a new workflow',
      href: '/workflows/create',
      icon: (
        <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" />
        </svg>
      ),
      color: 'bg-blue-600 hover:bg-blue-700',
    },
    {
      name: 'Create Team',
      description: 'Define a new AI team',
      href: '/teams/create',
      icon: (
        <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
        </svg>
      ),
      color: 'bg-green-600 hover:bg-green-700',
    },
    {
      name: 'Create Gauntlet',
      description: 'Configure a gauntlet',
      href: '/gauntlets/create',
      icon: (
        <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
      ),
      color: 'bg-purple-600 hover:bg-purple-700',
    },
  ];

  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {actions.map((action) => (
        <Link
          key={action.name}
          to={action.href as any}
          className="flex items-center gap-4 rounded-lg border border-gray-200 bg-white p-6 shadow-sm hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-800 dark:hover:bg-gray-700"
        >
          <div className={`flex-shrink-0 rounded-md p-3 text-white ${action.color}`}>
            {action.icon}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {action.name}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {action.description}
            </p>
          </div>
        </Link>
      ))}
    </div>
  );
}
