/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter
 *
 * Main adapter for integrating with all 7 ICR modes.
 * Provides idiomatic methods for each mode with proper error handling
 * and Federation Constitution compliance.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Idempotency: All operations safe to retry
 * - UTC: All timestamps in UTC ISO-8601 format
 * - Observability: All operations include correlation IDs
 */

import { v4 as uuidv4 } from 'uuid';
import {
  RefineModeRequest,
  RefineModeResponse,
  ReactModeRequest,
  ReactModeResponse,
  DeepthinkModeRequest,
  DeepthinkModeResponse,
  AdaptiveDeepthinkRequest,
  AdaptiveDeepthinkResponse,
  AgenticModeRequest,
  AgenticModeResponse,
  ContextualModeRequest,
  ContextualModeResponse,
  GenerativeUIModeRequest,
  GenerativeUIModeResponse,
  ICRHealthCheckResponse,
  ModeOptions
} from './icr-canonical';
import { ICRClient, icrClient } from './icr-client';

// ============================================================================
// REQUEST BUILDERS
// ============================================================================

/**
 * Build common metadata for requests
 */
function buildMetadata(correlationId?: string) {
  return {
    correlation_id: correlationId || uuidv4(),
    timestamp_utc: new Date().toISOString(),
    source_service: 'icr-adapter'
  };
}

// ============================================================================
// ICR ADAPTER
// ============================================================================

export interface ICRAdapterOptions {
  client?: ICRClient;
}

export class ICRAdapter {
  private readonly client: ICRClient;

  constructor(options?: ICRAdapterOptions) {
    this.client = options?.client || icrClient;
  }

  // ========================================================================
  // REFINE MODE
  // ========================================================================

  /**
   * Create a Refine mode request
   * Mode: Traditional iterative refinements with automated feature suggestion
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (temperature, evolution_mode, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Refine mode response
   */
  async createRefinementRequest(
    prompt: string,
    options?: ModeOptions & {
      evolution_mode?: 'novelty' | 'quality' | 'off';
      refinement_stages?: number;
    },
    correlationId?: string
  ): Promise<RefineModeResponse> {
    const cid = correlationId || uuidv4();

    const request: RefineModeRequest = {
      mode: 'refine',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        evolution_mode: options.evolution_mode,
        refinement_stages: options.refinement_stages
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // REACT MODE
  // ========================================================================

  /**
   * Create a React mode request
   * Mode: React application development with orchestrator-coordination
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (worker_count, enable_preview, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns React mode response
   */
  async createReactRequest(
    prompt: string,
    options?: ModeOptions & {
      worker_count?: number;
      enable_preview?: boolean;
    },
    correlationId?: string
  ): Promise<ReactModeResponse> {
    const cid = correlationId || uuidv4();

    const request: ReactModeRequest = {
      mode: 'react',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        worker_count: options.worker_count,
        enable_preview: options.enable_preview
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // DEEPTHINK MODE
  // ========================================================================

  /**
   * Create a Deepthink mode request
   * Mode: Complex problem-solving through strategic decomposition
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (strategy_count, sub_strategy_count, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Deepthink mode response
   */
  async createDeepthinkRequest(
    prompt: string,
    options?: ModeOptions & {
      strategy_count?: number;
      sub_strategy_count?: number;
      hypothesis_count?: number;
      enable_iterative_corrections?: boolean;
      enable_red_team?: boolean;
      red_team_aggressiveness?: 'low' | 'medium' | 'high';
    },
    correlationId?: string
  ): Promise<DeepthinkModeResponse> {
    const cid = correlationId || uuidv4();

    const request: DeepthinkModeRequest = {
      mode: 'deepthink',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        strategy_count: options.strategy_count,
        sub_strategy_count: options.sub_strategy_count,
        hypothesis_count: options.hypothesis_count,
        enable_iterative_corrections: options.enable_iterative_corrections,
        enable_red_team: options.enable_red_team,
        red_team_aggressiveness: options.red_team_aggressiveness
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // ADAPTIVE DEEPTHINK MODE
  // ========================================================================

  /**
   * Create an Adaptive Deepthink mode request
   * Mode: Full deepthink mode access to an agent
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (conversation_id, enable_streaming, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Adaptive Deepthink mode response
   */
  async createAdaptiveDeepthinkRequest(
    prompt: string,
    options?: ModeOptions & {
      conversation_id?: string;
      enable_streaming?: boolean;
    },
    correlationId?: string
  ): Promise<AdaptiveDeepthinkResponse> {
    const cid = correlationId || uuidv4();

    const request: AdaptiveDeepthinkRequest = {
      mode: 'adaptive_deepthink',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        conversation_id: options.conversation_id,
        enable_streaming: options.enable_streaming
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // AGENTIC MODE
  // ========================================================================

  /**
   * Create an Agentic mode request
   * Mode: General-purpose iterative refinement with tool-based manipulation
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (enable_diff_tools, enable_file_tools, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Agentic mode response
   */
  async createAgenticRequest(
    prompt: string,
    options?: ModeOptions & {
      conversation_id?: string;
      enable_diff_tools?: boolean;
      enable_file_tools?: boolean;
      enable_web_search?: boolean;
    },
    correlationId?: string
  ): Promise<AgenticModeResponse> {
    const cid = correlationId || uuidv4();

    const request: AgenticModeRequest = {
      mode: 'agentic',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        conversation_id: options.conversation_id,
        enable_diff_tools: options.enable_diff_tools,
        enable_file_tools: options.enable_file_tools,
        enable_web_search: options.enable_web_search
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // CONTEXTUAL MODE
  // ========================================================================

  /**
   * Create a Contextual mode request
   * Mode: Iterative refinement through specialized agent collaboration
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (enable_memory_agent, memory_compression_threshold, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Contextual mode response
   */
  async createContextualRequest(
    prompt: string,
    options?: ModeOptions & {
      conversation_id?: string;
      enable_memory_agent?: boolean;
      memory_compression_threshold?: number;
    },
    correlationId?: string
  ): Promise<ContextualModeResponse> {
    const cid = correlationId || uuidv4();

    const request: ContextualModeRequest = {
      mode: 'contextual',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        conversation_id: options.conversation_id,
        enable_memory_agent: options.enable_memory_agent,
        memory_compression_threshold: options.memory_compression_threshold
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // GENERATIVE UI MODE
  // ========================================================================

  /**
   * Create a Generative UI mode request
   * Mode: Interactive UI development with user interaction capture
   *
   * @param prompt - The user's prompt
   * @param options - Mode options (enable_interaction_capture, quality_threshold, etc.)
   * @param correlationId - Optional correlation ID for tracing
   * @returns Generative UI mode response
   */
  async createGenerativeUIRequest(
    prompt: string,
    options?: ModeOptions & {
      enable_interaction_capture?: boolean;
      quality_threshold?: number;
      max_iterations?: number;
    },
    correlationId?: string
  ): Promise<GenerativeUIModeResponse> {
    const cid = correlationId || uuidv4();

    const request: GenerativeUIModeRequest = {
      mode: 'generative_ui',
      prompt,
      options: options ? {
        temperature: options.temperature,
        top_p: options.top_p,
        max_iterations: options.max_iterations,
        model_name: options.model_name,
        provider: options.provider,
        enable_interaction_capture: options.enable_interaction_capture,
        quality_threshold: options.quality_threshold,
        max_iterations: options.max_iterations
      } : undefined,
      metadata: buildMetadata(cid)
    };

    return this.client.executeMode(request, cid);
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Perform health check on ICR system
   *
   * @param correlationId - Optional correlation ID for tracing
   * @returns Health check response
   */
  async healthCheck(correlationId?: string): Promise<ICRHealthCheckResponse> {
    const cid = correlationId || uuidv4();

    return this.client.healthCheck({ correlation_id: cid }, cid);
  }

  // ========================================================================
  // UTILITY METHODS
  // ========================================================================

  /**
   * Get circuit breaker state (for monitoring)
   */
  getCircuitBreakerState() {
    return this.client.getCircuitBreakerState();
  }

  /**
   * Reset circuit breaker (for recovery)
   */
  resetCircuitBreaker(): void {
    this.client.resetCircuitBreaker();
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

/**
 * Default ICR adapter instance
 */
export const icrAdapter = new ICRAdapter();
