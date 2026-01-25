/**
 * useKnowledgeExtraction Hook
 *
 * Provides knowledge extraction capabilities using the OpenEvolve Knowledge Engine.
 */

import { useMutation } from '@tanstack/react-query';
import { knowledgeApi } from '@/services/api/endpoints';

export interface ExtractionResult {
  entities: any[];
  relations: any[];
  events?: any[];
  timestamp: string;
}

export interface OneKEResult {
  schema_used: string;
  extracted_data: any;
  confidence: number;
  timestamp: string;
}

export function useKnowledgeExtraction() {
  const extractMutation = useMutation({
    mutationFn: async (data: { text: string; schema?: string[] }) => {
      return knowledgeApi.extract(data);
    },
  });

  const extractOneKEMutation = useMutation({
    mutationFn: async (data: { text: string; schema_name?: string }) => {
      return knowledgeApi.extractOneKE(data);
    },
  });

  const notifyWorkflowCompleteMutation = useMutation({
    mutationFn: async (data: { workflow_id: string; problem_statement: string; results: any }) => {
      return knowledgeApi.notifyWorkflowComplete(data);
    },
  });

  return {
    extract: extractMutation.mutateAsync,
    extractOneKE: extractOneKEMutation.mutateAsync,
    notifyWorkflowComplete: notifyWorkflowCompleteMutation.mutateAsync,
    isExtracting: extractMutation.isPending,
    isExtractingOneKE: extractOneKEMutation.isPending,
    isNotifying: notifyWorkflowCompleteMutation.isPending,
    error: extractMutation.error || extractOneKEMutation.error || notifyWorkflowCompleteMutation.error,
  };
}
