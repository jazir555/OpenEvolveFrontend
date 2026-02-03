// Datapizza Query Hook
// React hook for querying data with Datapizza
//
// INTEGRATION STATUS: Partial Implementation
// - Currently: Returns mock data with proper structure
// - Required: DataPizza backend API
// - Required: FastAPI wrapper around datapizza Python library
//
// SETUP INSTRUCTIONS:
// 1. Create a FastAPI server (see docs/DataPizza/SETUP.md)
// 2. Configure DATAPIZZA_API_URL in environment
// 3. Ensure API key is set if authentication is enabled

import { useCallback, useState } from 'react';
import { DatapizzaQueryResult } from '../types/plugin-types';

interface DatapizzaQueryOptions {
  dataSource?: string;
  maxResults?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

export function useDatapizzaQuery() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const queryData = useCallback(
    async (query: string, options: DatapizzaQueryOptions = {}): Promise<DatapizzaQueryResult> => {
      setIsLoading(true);
      setError(null);

      const startTime = Date.now();

      try {
        // Check if API URL is configured
        const apiUrl = process.env.DATAPIZZA_API_URL || '/api/datapizza';

        // Prepare request payload
        const payload = {
          query,
          data_source: options.dataSource || 'default',
          max_results: options.maxResults || 10,
          threshold: options.threshold || 0.7,
          include_metadata: options.includeMetadata !== false,
        };

        // Attempt to call real API
        try {
          const response = await fetch(`${apiUrl}/query`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(process.env.DATAPIZZA_API_KEY && {
                'Authorization': `Bearer ${process.env.DATAPIZZA_API_KEY}`,
              }),
            },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(process.env.DATAPIZZA_TIMEOUT ? parseInt(process.env.DATAPIZZA_TIMEOUT) : 30000),
          });

          if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
          }

          const data = await response.json();

          return {
            success: true,
            query,
            results: data.results || [],
            confidenceScore: data.confidence_score || 0.0,
            processingTime: Date.now() - startTime,
            errors: data.errors || [],
            warnings: data.warnings || [],
            metadata: {
              timestamp: new Date().toISOString(),
              queryType: data.query_type || 'semantic',
              dataSources: [options.dataSource || 'default_source'],
              ...data.metadata,
            },
            timestamp: new Date(),
          };
        } catch (apiError) {
          // API not available - return enhanced mock data
          console.warn('Datapizza API not available, using mock data:', apiError);

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
              'Set DATAPIZZA_API_URL environment variable for real queries',
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
    []
  );

  return { queryData, isLoading, error };
}
