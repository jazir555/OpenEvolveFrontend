/**
 * OpenEvolve API Service Layer
 *
 * This file provides a comprehensive API client for connecting to the OpenEvolve backend.
 * It replaces all mock data interfaces across page components with actual API calls.
 *
 * @module OpenEvolveAPI
 */

import { apiClient } from './client';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// --------------------------------------------------------------------------
// Evolution Types
// --------------------------------------------------------------------------

export interface EvolutionConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  selectionMethod: 'tournament' | 'roulette' | 'rank' | 'uniform';
  elitismCount: number;
  tournamentSize: number;
  temperature: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}

export interface EvolutionRun {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  generation: number;
  bestFitness: number;
  avgFitness: number;
  startTime?: string;
  endTime?: string;
  config: EvolutionConfig;
}

export interface EvolutionCreateRequest {
  name: string;
  config: EvolutionConfig;
}

export interface EvolutionUpdateRequest {
  status?: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  config?: Partial<EvolutionConfig>;
}

// --------------------------------------------------------------------------
// Adversarial Types
// --------------------------------------------------------------------------

export interface AdversarialConfig {
  enabled: boolean;
  attackStrategy: 'fgsm' | 'pgd' | 'cw' | 'bim' | 'deepfool';
  numExamples: number;
  strength: number;
  stepSize: number;
  numSteps: number;
  defenseStrategy: 'robust' | 'certified' | 'detection' | 'randomization' | 'gradient_masking';
  robustnessThreshold: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}

export interface AdversarialRun {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  attackSuccessRate: number;
  defenseSuccessRate: number;
  startTime?: string;
  endTime?: string;
  config: AdversarialConfig;
}

export interface AdversarialCreateRequest {
  name: string;
  config: AdversarialConfig;
}

export interface AdversarialUpdateRequest {
  status?: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  config?: Partial<AdversarialConfig>;
}

// --------------------------------------------------------------------------
// Knowledge Base Types
// --------------------------------------------------------------------------

export interface KnowledgeEntry {
  id: string;
  title: string;
  content: string;
  tags: string[];
  category: string;
  createdAt: string;
  updatedAt: string;
  author: string;
  status: 'draft' | 'published' | 'archived';
}

export interface KnowledgeCategory {
  id: string;
  name: string;
  description: string;
  count: number;
}

export interface KnowledgeCreateRequest {
  title: string;
  content: string;
  category: string;
  tags: string[];
  status?: 'draft' | 'published' | 'archived';
}

export interface KnowledgeUpdateRequest {
  title?: string;
  content?: string;
  category?: string;
  tags?: string[];
  status?: 'draft' | 'published' | 'archived';
}

export interface KnowledgeQueryParams {
  search?: string;
  category?: string;
  tags?: string[];
  status?: 'draft' | 'published' | 'archived';
  limit?: number;
  offset?: number;
}

// --------------------------------------------------------------------------
// Workflow Types
// --------------------------------------------------------------------------

export interface WorkflowNode {
  id: string;
  type: 'start' | 'end' | 'evolution' | 'adversarial' | 'decomposition' | 'knowledge' | 'leanaide' | 'crewai' | 'custom';
  position: { x: number; y: number };
  data: {
    label: string;
    config?: any;
  };
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

export interface WorkflowDefinition {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  createdAt: string;
  updatedAt: string;
  status: 'draft' | 'published' | 'archived';
}

export interface WorkflowCreateRequest {
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  status?: 'draft' | 'published' | 'archived';
}

export interface WorkflowUpdateRequest {
  name?: string;
  description?: string;
  nodes?: WorkflowNode[];
  edges?: WorkflowEdge[];
  status?: 'draft' | 'published' | 'archived';
}

export interface WorkflowInstance {
  id: string;
  workflowId: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  startTime?: string;
  endTime?: string;
  results?: any;
}

// --------------------------------------------------------------------------
// Analytics Types
// --------------------------------------------------------------------------

export interface WorkflowPerformance {
  workflowId: string;
  name: string;
  startDate: string;
  endDate?: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  successRate: number;
  avgExecutionTime: number;
  totalSubProblems: number;
  solvedSubProblems: number;
}

export interface TeamPerformance {
  teamId: string;
  name: string;
  role: 'red' | 'blue' | 'gold';
  totalEvaluations: number;
  approvalRate: number;
  avgScore: number;
  avgTimePerEvaluation: number;
}

export interface GauntletPerformance {
  gauntletId: string;
  name: string;
  totalRuns: number;
  successRate: number;
  avgExecutionTime: number;
  avgScore: number;
  failureReasons: string[];
}

export interface SolutionQuality {
  solutionId: string;
  workflowId: string;
  qualityScore: number;
  completeness: number;
  correctness: number;
  efficiency: number;
  maintainability: number;
  submittedDate: string;
}

export interface KnowledgeStats {
  totalArtifacts: number;
  totalCategories: number;
  totalTags: number;
  weeklyGrowth: number;
  mostUsedCategory: string;
  mostActiveAuthor: string;
}

export interface AnalyticsQueryParams {
  startDate?: string;
  endDate?: string;
  workflowIds?: string[];
  teamIds?: string[];
  gauntletIds?: string[];
}

// --------------------------------------------------------------------------
// Decomposition Types
// --------------------------------------------------------------------------

export interface DecompositionProblem {
  id: string;
  title: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  status: 'pending' | 'decomposing' | 'decomposed' | 'failed';
  createdAt: string;
  subProblems: SubProblem[];
}

export interface SubProblem {
  id: string;
  parentProblemId: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'solved' | 'failed';
  priority: number;
  dependencies: string[];
  solution?: string;
}

export interface DecompositionRequest {
  title: string;
  description: string;
  complexity?: 'low' | 'medium' | 'high';
  maxDepth?: number;
  branchingFactor?: number;
}

// ============================================================================
// API SERVICE CLASS
// ============================================================================

export class OpenEvolveAPI {
  private apiClient = apiClient;

  // =========================================================================
  // EVOLUTION ENDPOINTS
  // =========================================================================

  /**
   * Get all evolution runs
   */
  async getEvolutionRuns(params?: { status?: string; limit?: number }): Promise<EvolutionRun[]> {
    return this.apiClient.get<EvolutionRun[]>('/evolution/runs', params);
  }

  /**
   * Get a specific evolution run by ID
   */
  async getEvolutionRun(runId: string): Promise<EvolutionRun> {
    return this.apiClient.get<EvolutionRun>(`/evolution/runs/${runId}`);
  }

  /**
   * Create a new evolution run
   */
  async createEvolutionRun(request: EvolutionCreateRequest): Promise<EvolutionRun> {
    return this.apiClient.post<EvolutionRun>('/evolution/runs', request);
  }

  /**
   * Update an evolution run
   */
  async updateEvolutionRun(runId: string, request: EvolutionUpdateRequest): Promise<EvolutionRun> {
    return this.apiClient.patch<EvolutionRun>(`/evolution/runs/${runId}`, request);
  }

  /**
   * Delete an evolution run
   */
  async deleteEvolutionRun(runId: string): Promise<void> {
    return this.apiClient.delete<void>(`/evolution/runs/${runId}`);
  }

  /**
   * Start an evolution run
   */
  async startEvolutionRun(runId: string): Promise<EvolutionRun> {
    return this.apiClient.post<EvolutionRun>(`/evolution/runs/${runId}/start`);
  }

  /**
   * Pause an evolution run
   */
  async pauseEvolutionRun(runId: string): Promise<EvolutionRun> {
    return this.apiClient.post<EvolutionRun>(`/evolution/runs/${runId}/pause`);
  }

  /**
   * Resume an evolution run
   */
  async resumeEvolutionRun(runId: string): Promise<EvolutionRun> {
    return this.apiClient.post<EvolutionRun>(`/evolution/runs/${runId}/resume`);
  }

  /**
   * Stop an evolution run
   */
  async stopEvolutionRun(runId: string): Promise<EvolutionRun> {
    return this.apiClient.post<EvolutionRun>(`/evolution/runs/${runId}/stop`);
  }

  /**
   * Get evolution configuration
   */
  async getEvolutionConfig(): Promise<EvolutionConfig> {
    return this.apiClient.get<EvolutionConfig>('/evolution/config');
  }

  /**
   * Update evolution configuration
   */
  async updateEvolutionConfig(config: Partial<EvolutionConfig>): Promise<EvolutionConfig> {
    return this.apiClient.put<EvolutionConfig>('/evolution/config', config);
  }

  // =========================================================================
  // ADVERSARIAL ENDPOINTS
  // =========================================================================

  /**
   * Get all adversarial runs
   */
  async getAdversarialRuns(params?: { status?: string; limit?: number }): Promise<AdversarialRun[]> {
    return this.apiClient.get<AdversarialRun[]>('/adversarial/runs', params);
  }

  /**
   * Get a specific adversarial run by ID
   */
  async getAdversarialRun(runId: string): Promise<AdversarialRun> {
    return this.apiClient.get<AdversarialRun>(`/adversarial/runs/${runId}`);
  }

  /**
   * Create a new adversarial run
   */
  async createAdversarialRun(request: AdversarialCreateRequest): Promise<AdversarialRun> {
    return this.apiClient.post<AdversarialRun>('/adversarial/runs', request);
  }

  /**
   * Update an adversarial run
   */
  async updateAdversarialRun(runId: string, request: AdversarialUpdateRequest): Promise<AdversarialRun> {
    return this.apiClient.patch<AdversarialRun>(`/adversarial/runs/${runId}`, request);
  }

  /**
   * Delete an adversarial run
   */
  async deleteAdversarialRun(runId: string): Promise<void> {
    return this.apiClient.delete<void>(`/adversarial/runs/${runId}`);
  }

  /**
   * Start an adversarial run
   */
  async startAdversarialRun(runId: string): Promise<AdversarialRun> {
    return this.apiClient.post<AdversarialRun>(`/adversarial/runs/${runId}/start`);
  }

  /**
   * Pause an adversarial run
   */
  async pauseAdversarialRun(runId: string): Promise<AdversarialRun> {
    return this.apiClient.post<AdversarialRun>(`/adversarial/runs/${runId}/pause`);
  }

  /**
   * Resume an adversarial run
   */
  async resumeAdversarialRun(runId: string): Promise<AdversarialRun> {
    return this.apiClient.post<AdversarialRun>(`/adversarial/runs/${runId}/resume`);
  }

  /**
   * Stop an adversarial run
   */
  async stopAdversarialRun(runId: string): Promise<AdversarialRun> {
    return this.apiClient.post<AdversarialRun>(`/adversarial/runs/${runId}/stop`);
  }

  /**
   * Get adversarial configuration
   */
  async getAdversarialConfig(): Promise<AdversarialConfig> {
    return this.apiClient.get<AdversarialConfig>('/adversarial/config');
  }

  /**
   * Update adversarial configuration
   */
  async updateAdversarialConfig(config: Partial<AdversarialConfig>): Promise<AdversarialConfig> {
    return this.apiClient.put<AdversarialConfig>('/adversarial/config', config);
  }

  // =========================================================================
  // KNOWLEDGE BASE ENDPOINTS
  // =========================================================================

  /**
   * Get knowledge entries with optional filtering
   */
  async getKnowledgeEntries(params?: KnowledgeQueryParams): Promise<KnowledgeEntry[]> {
    return this.apiClient.get<KnowledgeEntry[]>('/knowledge/entries', params);
  }

  /**
   * Get a specific knowledge entry by ID
   */
  async getKnowledgeEntry(entryId: string): Promise<KnowledgeEntry> {
    return this.apiClient.get<KnowledgeEntry>(`/knowledge/entries/${entryId}`);
  }

  /**
   * Create a new knowledge entry
   */
  async createKnowledgeEntry(request: KnowledgeCreateRequest): Promise<KnowledgeEntry> {
    return this.apiClient.post<KnowledgeEntry>('/knowledge/entries', request);
  }

  /**
   * Update a knowledge entry
   */
  async updateKnowledgeEntry(entryId: string, request: KnowledgeUpdateRequest): Promise<KnowledgeEntry> {
    return this.apiClient.patch<KnowledgeEntry>(`/knowledge/entries/${entryId}`, request);
  }

  /**
   * Delete a knowledge entry
   */
  async deleteKnowledgeEntry(entryId: string): Promise<void> {
    return this.apiClient.delete<void>(`/knowledge/entries/${entryId}`);
  }

  /**
   * Get all knowledge categories
   */
  async getKnowledgeCategories(): Promise<KnowledgeCategory[]> {
    return this.apiClient.get<KnowledgeCategory[]>('/knowledge/categories');
  }

  /**
   * Get knowledge statistics
   */
  async getKnowledgeStats(): Promise<KnowledgeStats> {
    return this.apiClient.get<KnowledgeStats>('/knowledge/stats');
  }

  /**
   * Search knowledge entries
   */
  async searchKnowledge(query: string, params?: KnowledgeQueryParams): Promise<KnowledgeEntry[]> {
    return this.apiClient.get<KnowledgeEntry[]>('/knowledge/search', { q: query, ...params });
  }

  // =========================================================================
  // WORKFLOW ENDPOINTS
  // =========================================================================

  /**
   * Get all workflow definitions
   */
  async getWorkflows(params?: { status?: string; limit?: number }): Promise<WorkflowDefinition[]> {
    return this.apiClient.get<WorkflowDefinition[]>('/workflows', params);
  }

  /**
   * Get a specific workflow by ID
   */
  async getWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    return this.apiClient.get<WorkflowDefinition>(`/workflows/${workflowId}`);
  }

  /**
   * Create a new workflow
   */
  async createWorkflow(request: WorkflowCreateRequest): Promise<WorkflowDefinition> {
    return this.apiClient.post<WorkflowDefinition>('/workflows', request);
  }

  /**
   * Update a workflow
   */
  async updateWorkflow(workflowId: string, request: WorkflowUpdateRequest): Promise<WorkflowDefinition> {
    return this.apiClient.patch<WorkflowDefinition>(`/workflows/${workflowId}`, request);
  }

  /**
   * Delete a workflow
   */
  async deleteWorkflow(workflowId: string): Promise<void> {
    return this.apiClient.delete<void>(`/workflows/${workflowId}`);
  }

  /**
   * Publish a workflow
   */
  async publishWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    return this.apiClient.post<WorkflowDefinition>(`/workflows/${workflowId}/publish`);
  }

  /**
   * Unpublish a workflow
   */
  async unpublishWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    return this.apiClient.post<WorkflowDefinition>(`/workflows/${workflowId}/unpublish`);
  }

  /**
   * Get workflow instances
   */
  async getWorkflowInstances(workflowId: string): Promise<WorkflowInstance[]> {
    return this.apiClient.get<WorkflowInstance[]>(`/workflows/${workflowId}/instances`);
  }

  /**
   * Run a workflow
   */
  async runWorkflow(workflowId: string, config?: any): Promise<WorkflowInstance> {
    return this.apiClient.post<WorkflowInstance>(`/workflows/${workflowId}/run`, { config });
  }

  /**
   * Get workflow templates
   */
  async getWorkflowTemplates(): Promise<WorkflowDefinition[]> {
    return this.apiClient.get<WorkflowDefinition[]>('/workflows/templates');
  }

  /**
   * Create workflow from template
   */
  async createWorkflowFromTemplate(templateId: string, name: string): Promise<WorkflowDefinition> {
    return this.apiClient.post<WorkflowDefinition>('/workflows/from-template', { templateId, name });
  }

  // =========================================================================
  // ANALYTICS ENDPOINTS
  // =========================================================================

  /**
   * Get workflow performance data
   */
  async getWorkflowPerformance(params?: AnalyticsQueryParams): Promise<WorkflowPerformance[]> {
    return this.apiClient.get<WorkflowPerformance[]>('/analytics/workflows', params);
  }

  /**
   * Get team performance data
   */
  async getTeamPerformance(params?: AnalyticsQueryParams): Promise<TeamPerformance[]> {
    return this.apiClient.get<TeamPerformance[]>('/analytics/teams', params);
  }

  /**
   * Get gauntlet performance data
   */
  async getGauntletPerformance(params?: AnalyticsQueryParams): Promise<GauntletPerformance[]> {
    return this.apiClient.get<GauntletPerformance[]>('/analytics/gauntlets', params);
  }

  /**
   * Get solution quality metrics
   */
  async getSolutionQuality(params?: AnalyticsQueryParams): Promise<SolutionQuality[]> {
    return this.apiClient.get<SolutionQuality[]>('/analytics/solutions', params);
  }

  /**
   * Get comprehensive analytics overview
   */
  async getAnalyticsOverview(params?: AnalyticsQueryParams): Promise<{
    workflows: WorkflowPerformance[];
    teams: TeamPerformance[];
    gauntlets: GauntletPerformance[];
    solutions: SolutionQuality[];
    knowledge: KnowledgeStats;
  }> {
    return this.apiClient.get<any>('/analytics/overview', params);
  }

  /**
   * Export analytics data
   */
  async exportAnalytics(params?: AnalyticsQueryParams): Promise<Blob> {
    const response = await fetch(`${apiClient['baseURL']}/analytics/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...apiClient['getAuthHeaders']?.() || {},
      },
      body: JSON.stringify(params),
    });
    return response.blob();
  }

  // =========================================================================
  // DECOMPOSITION ENDPOINTS
  // =========================================================================

  /**
   * Get all decomposition problems
   */
  async getDecompositionProblems(params?: { status?: string; limit?: number }): Promise<DecompositionProblem[]> {
    return this.apiClient.get<DecompositionProblem[]>('/decomposition/problems', params);
  }

  /**
   * Get a specific decomposition problem
   */
  async getDecompositionProblem(problemId: string): Promise<DecompositionProblem> {
    return this.apiClient.get<DecompositionProblem>(`/decomposition/problems/${problemId}`);
  }

  /**
   * Create a new decomposition problem
   */
  async createDecompositionProblem(request: DecompositionRequest): Promise<DecompositionProblem> {
    return this.apiClient.post<DecompositionProblem>('/decomposition/problems', request);
  }

  /**
   * Start decomposition of a problem
   */
  async startDecomposition(problemId: string): Promise<DecompositionProblem> {
    return this.apiClient.post<DecompositionProblem>(`/decomposition/problems/${problemId}/decompose`);
  }

  /**
   * Get sub-problems for a problem
   */
  async getSubProblems(problemId: string): Promise<SubProblem[]> {
    return this.apiClient.get<SubProblem[]>(`/decomposition/problems/${problemId}/subproblems`);
  }

  /**
   * Update sub-problem status
   */
  async updateSubProblem(subProblemId: string, status: SubProblem['status']): Promise<SubProblem> {
    return this.apiClient.patch<SubProblem>(`/decomposition/subproblems/${subProblemId}`, { status });
  }

  // =========================================================================
  // HEALTH & STATUS ENDPOINTS
  // =========================================================================

  /**
   * Get API health status
   */
  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    version: string;
    timestamp: string;
    services: {
      evolution: { status: string; latency: number };
      adversarial: { status: string; latency: number };
      knowledge: { status: string; latency: number };
      workflow: { status: string; latency: number };
      decomposition: { status: string; latency: number };
    };
  }> {
    return this.apiClient.get<any>('/health');
  }

  /**
   * Get system status
   */
  async getSystemStatus(): Promise<{
    activeEvolutions: number;
    activeAdversarialRuns: number;
    activeWorkflows: number;
    knowledgeBaseSize: number;
    uptime: number;
  }> {
    return this.apiClient.get<any>('/status');
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  /**
   * Convert API date strings to Date objects
   */
  private parseDate(dateString?: string): Date | undefined {
    return dateString ? new Date(dateString) : undefined;
  }

  /**
   * Convert Date objects to API date strings
   */
  private formatDate(date?: Date): string | undefined {
    return date ? date.toISOString() : undefined;
  }

  /**
   * Handle API errors with user-friendly messages
   */
  private handleApiError(error: any, context: string): never {
    console.error(`OpenEvolve API Error [${context}]:`, error);

    if (error.status === 401) {
      throw new Error('Authentication required. Please log in again.');
    } else if (error.status === 403) {
      throw new Error('You do not have permission to perform this action.');
    } else if (error.status === 404) {
      throw new Error('The requested resource was not found.');
    } else if (error.status === 500) {
      throw new Error('Server error. Please try again later.');
    } else if (error.message) {
      throw new Error(error.message);
    } else {
      throw new Error(`An unexpected error occurred: ${context}`);
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

/**
 * Singleton instance of the OpenEvolve API client
 */
export const openEvolveAPI = new OpenEvolveAPI();

/**
 * Export the class for testing purposes
 */
export { OpenEvolveAPI as OpenEvolveAPIClass };

/**
 * Default export
 */
export default openEvolveAPI;
