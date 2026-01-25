/**
 * Base Bubble Abstract Class for Ragbits + BubbleLab Integration
 */

import { BubbleConfig } from '../types';

export abstract class BaseBubble<ConfigType extends BubbleConfig, InputType, OutputType> {
  protected config: ConfigType;

  constructor(config: ConfigType) {
    this.config = config;
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
    console.log(`Initializing bubble: ${this.config.name}`);
  }

  /**
   * Performs the bubble's action
   */
  abstract action(input: InputType): Promise<OutputType>;

  /**
   * Disposes of resources used by the bubble
   */
  async dispose(): Promise<void> {
    console.log(`Disposing bubble: ${this.config.name}`);
  }
}