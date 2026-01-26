/**
 * Test Utilities
 * Helper functions for testing components
 */

import { render, RenderOptions } from '@testing-library/react';
import { ReactElement } from 'react';

/**
 * Custom render function with providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  // Add any providers here (Router, QueryClient, Theme, etc.)
  return render(ui, options);
}

/**
 * Mock data generators
 */
export const mockWorkflows = [
  {
    id: '1',
    name: 'Test Workflow 1',
    description: 'A test workflow',
    problem_statement: 'Solve this problem',
    content_type: 'text',
    teams: ['team-1'],
    gauntlets: ['gauntlet-1'],
    status: 'completed',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    user_id: 'user-1',
    tenant_id: 'tenant-1',
  },
  {
    id: '2',
    name: 'Test Workflow 2',
    description: 'Another test workflow',
    problem_statement: 'Solve another problem',
    content_type: 'code',
    teams: ['team-2'],
    gauntlets: [],
    status: 'running',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    user_id: 'user-1',
    tenant_id: 'tenant-1',
  },
];

export const mockTeams = [
  {
    id: 'team-1',
    name: 'Test Team',
    description: 'A test team',
    members: [
      {
        id: 'member-1',
        name: 'Member 1',
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        max_iterations: 5,
        role: 'analyst',
      },
    ],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    user_id: 'user-1',
    tenant_id: 'tenant-1',
  },
];

export const mockGauntlets = [
  {
    id: 'gauntlet-1',
    name: 'Test Gauntlet',
    description: 'A test gauntlet',
    rounds: [
      {
        id: 'round-1',
        name: 'Round 1',
        quorum_threshold: 0.7,
        confidence_threshold: 0.8,
        evaluation_type: 'majority_vote',
        required_consensus: true,
        max_iterations: 3,
      },
    ],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    user_id: 'user-1',
    tenant_id: 'tenant-1',
  },
];

/**
 * Wait for async operations
 */
export function waitFor(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Mock API responses
 */
export const mockApiResponse = <T>(data: T, delay = 100): Promise<T> => {
  return new Promise(resolve => setTimeout(() => resolve(data), delay));
};
