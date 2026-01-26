/**
 * Sidebar Component
 * Main navigation sidebar
 */

import { Link, useLocation } from '@tanstack/react-router';
import { useState } from 'react';
import { useUIState, useConfigStore } from '../../stores/configStore';
import {
  HomeIcon,
  DocumentTextIcon,
  UserGroupIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

export function Sidebar() {
  const collapsed = useUIState((state) => state.sidebarCollapsed);
  const setSidebarCollapsed = useConfigStore((state) => state.setSidebarCollapsed);
  const location = useLocation();

  const navigation = [
    {
      name: 'Dashboard',
      href: '/',
      icon: HomeIcon,
    },
    {
      name: 'Workflows',
      href: '/workflows',
      icon: DocumentTextIcon,
    },
    {
      name: 'Teams',
      href: '/teams',
      icon: UserGroupIcon,
    },
    {
      name: 'Gauntlets',
      href: '/gauntlets',
      icon: ShieldCheckIcon,
    },
    {
      name: 'Benchmarks',
      href: '/benchmarks',
      icon: ChartBarIcon,
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: ChartBarIcon,
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Cog6ToothIcon,
    },
  ];

  return (
    <aside
      className={`fixed left-0 top-0 z-40 h-screen flex-col border-r border-gray-200 bg-white transition-all duration-300 dark:border-gray-700 dark:bg-gray-800 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Logo Section */}
      <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4 dark:border-gray-700">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded bg-blue-600 text-white font-bold">
              OE
            </div>
            <span className="text-lg font-semibold text-gray-900 dark:text-white">
              OpenEvolve
            </span>
          </div>
        )}
        <button
          onClick={() => setSidebarCollapsed(!collapsed)}
          className="rounded p-1 text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <ChevronRightIcon className="h-5 w-5" />
          ) : (
            <ChevronLeftIcon className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto px-2 py-4">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`
                group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors
                ${
                  isActive
                    ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400'
                    : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                }
              `}
              title={collapsed ? item.name : undefined}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && <span>{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      {/* User Section (Bottom) */}
      <div className="border-t border-gray-200 p-4 dark:border-gray-700">
        {!collapsed && (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <p>OpenEvolve v1.0.0</p>
            <p className="mt-1">© 2024 OpenEvolve</p>
          </div>
        )}
      </div>
    </aside>
  );
}
