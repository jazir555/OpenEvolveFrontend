/**
 * Tests for useApi hooks
 */
import React, { type ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEvolutions, useEvolution, useAnalytics } from '../useApi';
import { useKnowledge } from '../useKnowledge';
import { apiClient } from '@/services/api/client';

// Mock apiClient
vi.mock('@/services/api/client', () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

// Mock auth store
vi.mock('@/stores/authStore', () => ({
  useAuthStore: (selector: (state: { token: string }) => string) =>
    selector({ token: 'mock-token' }),
}));

// Mock knowledge store
vi.mock('@/stores/knowledgeStore', () => ({
  useKnowledgeStore: vi.fn(() => ({
    setArtifacts: vi.fn(),
    addArtifact: vi.fn(),
    updateArtifact: vi.fn(),
    removeArtifact: vi.fn(),
  })),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe('useEvolutions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch evolutions successfully', async () => {
    const mockEvolutions = [
      { id: '1', name: 'Workflow 1', status: 'completed' },
      { id: '2', name: 'Workflow 2', status: 'running' },
    ];

    vi.mocked(apiClient.get).mockResolvedValueOnce({
      evolutions: mockEvolutions,
      total: 2,
    });

    const { result } = renderHook(() => useEvolutions(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toEqual({
      evolutions: mockEvolutions,
      total: 2,
    });
    expect(apiClient.get).toHaveBeenCalledWith('/evolution', undefined);
  });

  it('should handle errors', async () => {
    vi.mocked(apiClient.get).mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useEvolutions(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.error).toBeTruthy();
  });

  it('should use custom query params', async () => {
    vi.mocked(apiClient.get).mockResolvedValueOnce({
      evolutions: [],
      total: 0,
    });

    const { result } = renderHook(
      () => useEvolutions({ status: 'running', limit: 20 }),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(apiClient.get).toHaveBeenCalledWith('/evolution', {
      status: 'running',
      limit: 20,
    });
  });
});

describe('useEvolution', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should start evolution successfully', async () => {
    const mockResponse = {
      evolution_id: 'evol-123',
      status: 'running',
      websocket_url: 'ws://localhost:8000/ws/evolution/evol-123',
    };

    vi.mocked(apiClient.post).mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useEvolution(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      const response = await result.current.startEvolution({
        content: 'test content',
        mode: 'standard',
        parameters: { max_iterations: 10, population_size: 20 },
        models: [{ provider: 'openai', model: 'gpt-4', api_key: 'sk-test' }],
      });
    });

    expect(apiClient.post).toHaveBeenCalledWith('/evolution/start', {
      content: 'test content',
      mode: 'standard',
      parameters: { max_iterations: 10, population_size: 20 },
      models: [{ provider: 'openai', model: 'gpt-4', api_key: 'sk-test' }],
    });
  });

  it('should get evolution status', async () => {
    const mockStatus = {
      evolution_id: 'evol-123',
      status: 'running',
      progress: 50,
    };

    vi.mocked(apiClient.get).mockResolvedValueOnce(mockStatus);

    const { result } = renderHook(() => useEvolution('evol-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.evolution).toEqual(mockStatus);
    expect(apiClient.get).toHaveBeenCalledWith('/evolution/evol-123');
  });

  it('should pause evolution', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      evolution_id: 'evol-123',
      status: 'paused',
    });

    const { result } = renderHook(() => useEvolution(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.pauseEvolution('evol-123');
    });

    expect(apiClient.post).toHaveBeenCalledWith(
      '/evolution/evol-123/pause',
      {}
    );
  });
});

describe('useAnalytics', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should get analytics metrics', async () => {
    const mockMetrics = {
      metrics: {
        total_evolutions: 100,
        success_rate: 85,
        average_duration: 120,
      },
    };

    vi.mocked(apiClient.get)
      .mockResolvedValueOnce(mockMetrics)
      .mockResolvedValueOnce({ performance: [] });

    const { result } = renderHook(
      () =>
        useAnalytics({
          start_date: '2025-01-01',
          end_date: '2025-01-06',
          granularity: 'day',
        }),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metrics).toEqual(mockMetrics);
    expect(apiClient.get).toHaveBeenCalledWith('/analytics/metrics', {
      start_date: '2025-01-01',
      end_date: '2025-01-06',
      granularity: 'day',
    });
  });

  it('should get performance data', async () => {
    const mockPerformance = {
      performance: [
        { date: '2025-01-01', cpu: 45, memory: 60 },
        { date: '2025-01-02', cpu: 50, memory: 65 },
      ],
    };

    vi.mocked(apiClient.get).mockResolvedValueOnce(mockPerformance);

    const { result } = renderHook(() => useAnalytics(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.performance).toEqual(mockPerformance);
    expect(apiClient.get).toHaveBeenCalledWith('/analytics/performance');
  });
});

describe('useKnowledge', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should create artifact', async () => {
    const mockArtifact = {
      artifact_id: 'art-123',
      title: 'Test Artifact',
      content: 'Test content',
    };

    vi.mocked(apiClient.get).mockResolvedValue({ content: [] });
    vi.mocked(apiClient.post).mockResolvedValueOnce(mockArtifact);

    const { result } = renderHook(() => useKnowledge(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.createArtifact({
        title: 'Test Artifact',
        content: 'Test content',
      });
    });

    expect(apiClient.post).toHaveBeenCalledWith('/content', {
      title: 'Test Artifact',
      content: 'Test content',
    });
  });

  it('should fetch artifacts', async () => {
    const mockArtifacts = [
      { artifact_id: 'art-1', title: 'Python Tutorial', content: 'Intro' },
      { artifact_id: 'art-2', title: 'Python Advanced', content: 'Details' },
    ];

    vi.mocked(apiClient.get).mockResolvedValueOnce({ content: mockArtifacts });

    const { result } = renderHook(() => useKnowledge(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.artifacts).toEqual(mockArtifacts);
    expect(apiClient.get).toHaveBeenCalledWith('/content', undefined);
  });

  it('should delete artifact', async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ content: [] });
    vi.mocked(apiClient.delete).mockResolvedValueOnce(null);

    const { result } = renderHook(() => useKnowledge(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.deleteArtifact('art-123');
    });

    expect(apiClient.delete).toHaveBeenCalledWith('/content/art-123');
  });
});
