/**
 * RagbitsBubbleLabIntegration
 * High-level facade for the RAGBits BubbleLab integration package.
 *
 * Provides a documented, stable entry point that wraps the underlying
 * engines (`RAGBitsWorkflowEngine`, `RAGBitsDocumentProcessor`,
 * `MonitoringService`). The README advertises `RagbitsBubbleLabIntegration`
 * and `RagbitsBubbleLabIntegration.getInstance()`; this class is the real,
 * functional implementation of that contract.
 */

import { createWorkflowEngine, RAGBitsWorkflowEngine } from './engine';
import { createProcessorIntegration, RAGBitsDocumentProcessor } from './integration';
import { createMonitoringService, MonitoringService } from './monitoring';
import type {
  BubbleLabWorkflowConfig,
  WorkflowExecutionOptions,
  WorkflowExecutionResult,
  ProcessorIntegrationConfig
} from './types';

/**
 * Workflow summary returned by `listWorkflows()`.
 */
export interface WorkflowSummary {
  id: string;
  name: string;
}

/**
 * RagbitsBubbleLabIntegration - Singleton facade consolidating the
 * RAGBits BubbleLab integration capabilities.
 *
 * Use `RagbitsBubbleLabIntegration.getInstance()` to obtain the shared
 * instance. The facade tracks the workflow engines it creates so callers
 * can later run, inspect, and dispose them through a single object.
 */
export class RagbitsBubbleLabIntegration {
  private static instance: RagbitsBubbleLabIntegration | null = null;

  private readonly engines: Map<string, RAGBitsWorkflowEngine>;
  private readonly lastResults: Map<string, WorkflowExecutionResult>;
  private readonly initialized: Set<string>;
  private processor: RAGBitsDocumentProcessor | null;
  private monitoring: MonitoringService | null;
  private readonly logger: Console;

  /**
   * Private constructor - use {@link getInstance} to obtain the singleton.
   */
  private constructor() {
    this.engines = new Map();
    this.lastResults = new Map();
    this.initialized = new Set();
    this.processor = null;
    this.monitoring = null;
    this.logger = console;
  }

  /**
   * Get the shared singleton instance.
   * @returns The process-wide `RagbitsBubbleLabIntegration` instance.
   */
  public static getInstance(): RagbitsBubbleLabIntegration {
    if (!RagbitsBubbleLabIntegration.instance) {
      RagbitsBubbleLabIntegration.instance = new RagbitsBubbleLabIntegration();
    }
    return RagbitsBubbleLabIntegration.instance;
  }

  /**
   * Reset the singleton instance. Intended for tests and advanced scenarios.
   */
  public static resetInstance(): void {
    const current = RagbitsBubbleLabIntegration.instance;
    if (current) {
      current.engines.clear();
      current.lastResults.clear();
      current.initialized.clear();
      current.processor = null;
      current.monitoring = null;
    }
    RagbitsBubbleLabIntegration.instance = null;
  }

  /**
   * Create (and register) a workflow engine for the given workflow config.
   * Mirrors the `createWorkflowEngine` example from the README.
   *
   * @param workflowConfig - BubbleLab workflow definition.
   * @param options - Optional execution options.
   * @param processor - Optional document processor (defaults to the
   *                     integration-wide processor, if one was set).
   * @returns The created `RAGBitsWorkflowEngine`, also tracked by this facade.
   */
  public createWorkflowEngine(
    workflowConfig: BubbleLabWorkflowConfig,
    options?: WorkflowExecutionOptions,
    processor?: RAGBitsDocumentProcessor
  ): RAGBitsWorkflowEngine {
    const engine = createWorkflowEngine(
      workflowConfig,
      options,
      processor ?? this.processor ?? undefined
    );
    this.engines.set(workflowConfig.id, engine);
    this.initialized.delete(workflowConfig.id);
    this.lastResults.delete(workflowConfig.id);
    this.logger.info(
      `[RagbitsBubbleLabIntegration] Registered workflow engine: ${workflowConfig.id}`
    );
    return engine;
  }

  /**
   * Get a previously registered workflow engine by id.
   * @param workflowId - Workflow identifier.
   * @returns The engine, or `undefined` if not registered.
   */
  public getWorkflowEngine(workflowId: string): RAGBitsWorkflowEngine | undefined {
    return this.engines.get(workflowId);
  }

  /**
   * Initialize a registered workflow engine (idempotent per registration).
   * @param workflowId - Workflow identifier.
   */
  public async initializeWorkflow(workflowId: string): Promise<void> {
    const engine = this.requireEngine(workflowId);
    if (!this.initialized.has(workflowId)) {
      await engine.initialize();
      this.initialized.add(workflowId);
    }
  }

  /**
   * Run (initialize if needed, then execute) a registered workflow.
   * The latest result is stored and retrievable via {@link getWorkflowStatus}.
   *
   * @param workflowId - Workflow identifier.
   * @param options - Optional execution options applied before running.
   * @returns The workflow execution result.
   */
  public async runWorkflow(
    workflowId: string,
    options?: WorkflowExecutionOptions
  ): Promise<WorkflowExecutionResult> {
    const engine = this.requireEngine(workflowId);
    if (options) {
      engine.setExecutionOptions(options);
    }
    if (!this.initialized.has(workflowId)) {
      await engine.initialize();
      this.initialized.add(workflowId);
    }
    const result = await engine.executeWorkflow();
    this.lastResults.set(workflowId, result);
    return result;
  }

  /**
   * Get the status (latest execution result) of a workflow.
   * @param workflowId - Workflow identifier.
   * @returns The latest execution result, or `null` if it has not run yet.
   */
  public getWorkflowStatus(workflowId: string): WorkflowExecutionResult | null {
    return this.lastResults.get(workflowId) ?? null;
  }

  /**
   * List the registered workflows.
   * @returns Array of workflow summaries ({ id, name }).
   */
  public listWorkflows(): WorkflowSummary[] {
    return Array.from(this.engines.values()).map(engine => {
      const config = engine.getWorkflowConfig();
      return { id: config.id, name: config.name };
    });
  }

  /**
   * List the execution history for a registered workflow.
   * @param workflowId - Workflow identifier.
   * @returns All recorded execution results for that workflow.
   */
  public listExecutions(workflowId: string): WorkflowExecutionResult[] {
    const engine = this.requireEngine(workflowId);
    return engine.getExecutionHistory();
  }

  /**
   * Create a standalone RAGBits document processor.
   * @param config - Optional processor integration configuration.
   * @returns A new `RAGBitsDocumentProcessor`.
   */
  public createDocumentProcessor(config?: ProcessorIntegrationConfig): RAGBitsDocumentProcessor {
    return createProcessorIntegration(config);
  }

  /**
   * Get the integration-wide document processor (if set).
   * @returns The shared processor, or `null`.
   */
  public getDocumentProcessor(): RAGBitsDocumentProcessor | null {
    return this.processor;
  }

  /**
   * Set an integration-wide document processor. Newly created workflow
   * engines automatically inherit it, and existing engines are updated.
   * @param processor - The processor instance to share.
   */
  public setDocumentProcessor(processor: RAGBitsDocumentProcessor): void {
    this.processor = processor;
    this.engines.forEach(engine => engine.setProcessor(processor));
  }

  /**
   * Get (lazily creating) the shared monitoring service.
   * @returns The integration's `MonitoringService`.
   */
  public getMonitoringService(): MonitoringService {
    if (!this.monitoring) {
      this.monitoring = createMonitoringService();
    }
    return this.monitoring;
  }

  /**
   * Dispose a registered workflow engine and remove it from the facade.
   * @param workflowId - Workflow identifier.
   */
  public async disposeWorkflow(workflowId: string): Promise<void> {
    const engine = this.requireEngine(workflowId);
    await engine.dispose();
    this.engines.delete(workflowId);
    this.initialized.delete(workflowId);
    this.lastResults.delete(workflowId);
  }

  /**
   * Resolve a registered engine or throw.
   * @param workflowId - Workflow identifier.
   */
  private requireEngine(workflowId: string): RAGBitsWorkflowEngine {
    const engine = this.engines.get(workflowId);
    if (!engine) {
      throw new Error(`No workflow engine registered with id: ${workflowId}`);
    }
    return engine;
  }
}
