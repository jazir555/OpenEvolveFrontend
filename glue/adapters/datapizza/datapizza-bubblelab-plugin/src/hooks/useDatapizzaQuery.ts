// Datapizza Query Hook
// React hook for querying data with Datapizza
//
// INTEGRATION STATUS: Production Implementation
// - Uses DatapizzaClient for all API calls
// - Follows Federation Constitution laws
// - Configurable mock fallback for development (set DATAPIZZA_USE_MOCK=true)
//
// SETUP INSTRUCTIONS:
// 1. Configure DATAPIZZA_BASE_URL in environment
// 2. Configure DATAPIZZA_TIMEOUT_MS in environment
// 3. Set DATAPIZZA_USE_MOCK=true for development without API

import { useCallback, useState } from 'react';
import { DatapizzaQueryResult } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';

interface DatapizzaQueryOptions {
  dataSource?: string;
  maxResults?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

export function useDatapizzaQuery(client?: DatapizzaClient) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const queryData = useCallback(
    async (query: string, options: DatapizzaQueryOptions = {}): Promise<DatapizzaQueryResult> => {
      setIsLoading(true);
      setError(null);

      const startTime = Date.now();

      try {
        // Check if mock mode is enabled (for development)
        const useMock = process.env.DATAPIZZA_USE_MOCK === 'true';

        if (useMock) {
          console.warn('Datapizza mock mode enabled - set DATAPIZZA_USE_MOCK=false to use real API');

          // Simulate processing delay
          await new Promise(resolve => setTimeout(resolve, 600 + Math.random() * 400));

          return {
            success: true,
            query,
            results: [
              {
                id: `result_${Date.now()}_1`,
                score: 0.95 - Math.random() * 0.1,
                data: {
                  content: `Sample result for query: "${query}"`,
                  source: options.dataSource || 'default_source',
                  metadata: {
                    timestamp: new Date().toISOString(),
                    relevance: 'high',
                  },
                },
              },
              {
                id: `result_${Date.now()}_2`,
                score: 0.85 - Math.random() * 0.15,
                data: {
                  content: `Additional context for: "${query}"`,
                  source: options.dataSource || 'default_source',
                  metadata: {
                    timestamp: new Date().toISOString(),
                    relevance: 'medium',
                  },
                },
              },
            ],
            confidenceScore: 0.91 - Math.random() * 0.1,
            processingTime: Date.now() - startTime,
            errors: [],
            warnings: [
              'Using mock data - Datapizza API not configured',
              'Set DATAPIZZA_USE_MOCK=false and configure DATAPIZZA_BASE_URL',
            ],
            metadata: {
              timestamp: new Date().toISOString(),
              queryType: 'semantic',
              dataSources: [options.dataSource || 'default_source'],
              mock: true,
            },
            timestamp: new Date(),
          };
        }

        // Use real DatapizzaClient
        if (!client) {
          throw new Error(
            'DatapizzaClient not provided. Either pass a client to the hook or set DATAPIZZA_USE_MOCK=true for development.'
          );
        }

        const result = await client.queryData({
          query,
          dataSource: options.dataSource,
          limit: options.maxResults || 10,
          offset: 0,
        });

        // Map API response to hook format
        return {
          success: result.success,
          query,
          results: result.results.map(r => ({
            id: r.id,
            score: r.score,
            data: r.data,
          })),
          confidenceScore: result.results.length > 0
            ? Math.max(...result.results.map(r => r.score))
            : 0.0,
          processingTime: Date.now() - startTime,
          errors: [],
          warnings: [],
          metadata: {
            timestamp: new Date().toISOString(),
            queryType: 'semantic',
            dataSources: [options.dataSource || 'default_source'],
            totalCount: result.totalCount,
          },
          timestamp: new Date(),
        };
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);

        return {
          success: false,
          query,
          results: [],
          confidenceScore: 0.0,
          processingTime: Date.now() - startTime,
          errors: [errorMessage],
          warnings: [],
          metadata: {
            timestamp: new Date().toISOString(),
            queryType: 'unknown',
            dataSources: [],
          },
          timestamp: new Date(),
        };
      } finally {
        setIsLoading(false);
      }
    },
    [client]
  );

  return { queryData, isLoading, error };
}
