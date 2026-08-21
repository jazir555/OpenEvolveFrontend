/**
 * Base Bubble Class
 * Abstract base class for all RAGBits bubble components
 */

import { BubbleConfig, BubbleLabNode } from '../types';
import { v4 as uuidv4 } from 'uuid';

/**
 * BaseBubble - Abstract base class for all bubble components
 * Provides common functionality for configuration, initialization, and lifecycle management
 */
export abstract class BaseBubble<T extends Record<string, any> = BubbleConfig> {
  protected config: T;
  protected initialized: boolean = false;
  protected node: BubbleLabNode | null = null;
  protected logger: Console;
  protected metrics: Record<string, any>;
  
  /**
   * Constructor
   * @param config - Bubble configuration
   */
  constructor(config: T) {
    this.config = this.validateConfig(config);
    this.logger = console;
    this.metrics = {
      executions: 0,
      successes: 0,
      failures: 0,
      lastExecutionTime: 0,
      avgExecutionTime: 0
    };
  }
  
  /**
   * Validate configuration
   * @param config - Configuration to validate
   * @returns Validated configuration
   * @throws Error if configuration is invalid
   */
  protected validateConfig(config: T): T {
    if (!config || typeof config !== 'object') {
      throw new Error('Configuration must be an object');
    }

    const identity = config as unknown as BubbleConfig;
    if (!identity.id || typeof identity.id !== 'string') {
      identity.id = uuidv4();
    }

    if (!identity.name || typeof identity.name !== 'string') {
      throw new Error('Configuration must include a valid name');
    }

    return config;
  }
  
  /**
   * Initialize the bubble
   * @param node - BubbleLab node associated with this bubble
   */
  public async initialize(node: BubbleLabNode): Promise<void> {
    this.node = node;
    this.initialized = true;
    this.log('info', `Bubble ${this.config.name} (${this.config.id}) initialized`);
  }
  
  /**
   * Execute the bubble's action
   * @param input - Input data for the action
   * @returns Promise with the action result
   */
  public abstract action(input: any): Promise<any>;
  
  /**
   * Dispose the bubble and clean up resources
   */
  public async dispose(): Promise<void> {
    this.initialized = false;
    this.log('info', `Bubble ${this.config.name} (${this.config.id}) disposed`);
  }
  
  /**
   * Check if bubble is initialized
   * @returns True if initialized, false otherwise
   */
  public isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Get bubble configuration
   * @returns Bubble configuration
   */
  public getConfig(): T {
    return this.config;
  }
  
  /**
   * Get bubble metrics
   * @returns Performance metrics
   */
  public getMetrics(): Record<string, any> {
    return this.metrics;
  }
  
  /**
   * Reset metrics
   */
  public resetMetrics(): void {
    this.metrics = {
      executions: 0,
      successes: 0,
      failures: 0,
      lastExecutionTime: 0,
      avgExecutionTime: 0
    };
  }
  
  /**
   * Log a message
   * @param level - Log level
   * @param message - Message to log
   * @param data - Additional data to log
   */
  protected log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: any): void {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${this.config.name}] [${level.toUpperCase()}] ${message}`;
    
    switch (level) {
      case 'debug':
        this.logger.debug(logMessage, data);
        break;
      case 'info':
        this.logger.info(logMessage, data);
        break;
      case 'warn':
        this.logger.warn(logMessage, data);
        break;
      case 'error':
        this.logger.error(logMessage, data);
        break;
    }
  }
  
  /**
   * Update execution metrics
   * @param success - Whether execution was successful
   * @param executionTime - Execution time in milliseconds
   */
  protected updateMetrics(success: boolean, executionTime: number): void {
    this.metrics.executions++;
    if (success) {
      this.metrics.successes++;
    } else {
      this.metrics.failures++;
    }
    
    this.metrics.lastExecutionTime = executionTime;
    this.metrics.avgExecutionTime = 
      (this.metrics.avgExecutionTime * (this.metrics.executions - 1) + executionTime) / this.metrics.executions;
  }
  
  /**
   * Handle errors
   * @param error - Error to handle
   * @param context - Context for error handling
   */
  protected handleError(error: Error, context: string): never {
    this.log('error', `Error in ${context}: ${error.message}`, {
      error: error.stack,
      context
    });
    throw error;
  }
  
  /**
   * Measure execution time
   * @param fn - Function to measure
   * @returns Promise with result and execution time
   */
  protected async measureExecutionTime<T>(fn: () => Promise<T>): Promise<{ result: T; time: number }> {
    const startTime = Date.now();
    try {
      const result = await fn();
      const endTime = Date.now();
      return { result, time: endTime - startTime };
    } catch (error) {
      const endTime = Date.now();
      this.log('error', `Execution failed after ${endTime - startTime}ms`, { error });
      throw error;
    }
  }
}