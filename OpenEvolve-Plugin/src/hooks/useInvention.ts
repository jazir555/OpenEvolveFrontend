// @ts-nocheck
/**
 * End-to-End Invention Planner Hook
 *
 * Provides invention planning functionality
 *
 * @module hooks
 * @version 1.0.0
 */

import { useState, useCallback, useEffect } from 'react';
import { apiClient } from '@/services/api/client';

export interface InventionRequest {
  goal: string;
  domain: 'technology' | 'hardware' | 'business' | 'process' | 'scientific' | 'creative';
  innovativeness: number;
  planningStages: string[];
  constraints?: string;
  targetAudience?: string;
  includePriorArt: boolean;
  includeFeasibility: boolean;
  includeRoadmap: boolean;
  detailLevel: 'overview' | 'detailed' | 'comprehensive';
}

export interface InventionResponse {
  plan: any;
  priorArt?: any;
  feasibility?: any;
  roadmap?: any;
  leanProofs?: any[];
  errorAnalysis: any;
  redTeamResults?: any;
  blueTeamResults?: any;
  successCriteria: any[];
  executionTime: number;
  qualityAssessment: {
    innovation: number;
    feasibility: number;
    clarity: number;
    completeness: number;
  };
}

export interface InventionStatus {
  isRunning: boolean;
  progress: number;
  currentStage: string;
  message: string;
}

/**
 * Invention planner hook
 */
export function useInvention() {
  const [status, setStatus] = useState<InventionStatus>({
    isRunning: false,
    progress: 0,
    currentStage: '',
    message: '',
  });

  const [result, setResult] = useState<InventionResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);

  /**
   * Create invention plan
   */
  const createPlan = useCallback(async (request: InventionRequest): Promise<InventionResponse> => {
    setStatus({
      isRunning: true,
      progress: 0,
      currentStage: 'initialization',
      message: 'Initializing invention planning...',
    });
    setResult(null);
    setError(null);

    try {
      const response = await apiClient.post<InventionResponse>('/api/openevolve/invention/plan', request);

      setResult(response.data);
      setStatus({
        isRunning: false,
        progress: 100,
        currentStage: 'complete',
        message: 'Invention planning complete',
      });

      return response.data;
    } catch (err: any) {
      const error = new Error(err.message || 'Invention planning failed');
      setError(error);
      setStatus((prev) => ({
        ...prev,
        isRunning: false,
        message: error.message,
      }));
      throw error;
    }
  }, []);

  // NOTE: Backend Invention endpoint is synchronous - no status/cancel endpoints available
  // The Invention planning operation returns results immediately, so async status/cancel are not supported

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    setStatus({
      isRunning: false,
      progress: 0,
      currentStage: '',
      message: '',
    });
    setResult(null);
    setError(null);
  }, []);

  return {
    createPlan,
    reset,
    status,
    result,
    error,
  };
}
