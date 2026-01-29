/**
 * Base Bubble Abstract Class for Ragbits + BubbleLab Integration
 */

import { BubbleConfig } from '../types';
import { Logger } from '../utils/common.utils';

export abstract class BaseBubble<ConfigType extends BubbleConfig, InputType, OutputType> {
  protected config: ConfigType;
  protected logger: Logger;

  constructor(config: ConfigType) {
    this.config = config;
    this.logger = new Logger({ level: 'info', prefix: config.name });
    this.validateConfig();
  }

  /**
   * Validates the configuration
   */
  protected validateConfig(): void {
    if (!this.config.id) {
      throw new Error('Bubble configuration must have an id');
    }
    if (!this.config.name) {
      throw new Error('Bubble configuration must have a name');
    }
  }

  /**
   * Initializes the bubble
   */
  async initialize(): Promise<void> {
    this.logger.info(`Initializing bubble: ${this.config.name}`);
  }

  /**
   * Performs the bubble's action
   */
  abstract action(input: InputType): Promise<OutputType>;

  /**
   * Disposes of resources used by the bubble
   */
  async dispose(): Promise<void> {
    this.logger.info(`Disposing bubble: ${this.config.name}`);
  }

  /**
   * Handles errors in the bubble
   */
  abstract handleError(error: unknown, context?: string): Error;

  /**
   * Add logging base functionality
   */
  protected log(level: 'debug' | 'info' | 'warn' | 'error', message: string): void {
    this.logger[level](message);
  }

  /**
   * Add metrics collection base functionality
   */
  protected collectMetric(name: string, value: number, tags?: Record<string, string>): void {
    // In a real implementation, this would send metrics to a metrics collection system
    this.logger.debug(`Metric collected - ${name}: ${value}${tags ? ` (tags: ${JSON.stringify(tags)})` : ''}`);
  }

  /**
   * Get bubble configuration
   */
  getConfig(): ConfigType {
    return this.config;
  }

  /**
   * Update bubble configuration
   */
  updateConfig(newConfig: Partial<ConfigType>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger.info(`Configuration updated for bubble: ${this.config.name}`);
  }

  /**
   * Check if bubble is ready to perform actions
   */
  isReady(): boolean {
    // In a real implementation, this would check if the bubble is properly initialized
    return true;
  }

  /**
   * Get bubble status
   */
  getStatus(): 'initialized' | 'ready' | 'busy' | 'disposed' {
    // In a real implementation, this would return the actual status
    return 'ready';
  }
}