/**
 * Assembly Node
 *
 * Combines multiple outputs into a unified deliverable using configurable
 * integration strategies and conflict resolution.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema,
} from './OpenEvolveBaseNode';

export type AssemblyStrategy = 'sequential' | 'parallel' | 'hierarchical' | 'adaptive';
export type IntegrationMethod = 'merge' | 'compose' | 'aggregate' | 'pipeline';
export type ConflictResolutionStrategy = 'priority' | 'merge' | 'voting' | 'custom';

export interface AssemblyNodeConfig {
  strategy?: AssemblyStrategy;
  integrationMethod?: IntegrationMethod;
  conflictResolution?: ConflictResolutionStrategy;
  optimizeAssembly?: boolean;
  optimizationObjectives?: string[];
  validateIntegration?: boolean;
  integrationTesting?: boolean;
  maxIterations?: number;
  generateDocumentation?: boolean;
}

export interface AssemblyResult {
  assembled: any;
  components: any[];
  metadata: {
    strategy: AssemblyStrategy;
    integrationMethod: IntegrationMethod;
    generatedAt: Date;
    executionTime: number;
  };
}

export class AssemblyNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Assembly';
  static readonly DESCRIPTION = 'Assemble multiple component outputs into an integrated deliverable';
  static readonly ICON = 'assembly';
  static readonly CATEGORY = 'integration';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: AssemblyNodeConfig = {}) {
    super(id, {
      strategy: 'sequential',
      integrationMethod: 'merge',
      conflictResolution: 'merge',
      optimizeAssembly: false,
      optimizationObjectives: [],
      validateIntegration: true,
      integrationTesting: false,
      maxIterations: 1,
      generateDocumentation: false,
      ...config,
    });
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const components = (inputs.components as any[]) || [];

      if (!Array.isArray(components) || components.length === 0) {
        return this.createErrorResult('At least one component is required for assembly');
      }

      context.updateProgress(20, 'Preparing assembly');

      const method = (inputs.integrationMethod as IntegrationMethod) ||
        (this.config.integrationMethod as IntegrationMethod);

      const assembled = this.assembleComponents(components, method);

      context.updateProgress(80, 'Finalizing assembly');

      const result: AssemblyResult = {
        assembled,
        components,
        metadata: {
          strategy: this.config.strategy as AssemblyStrategy,
          integrationMethod: method,
          generatedAt: new Date(),
          executionTime: Date.now() - startTime,
        },
      };

      context.updateProgress(100, 'Assembly complete');
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during assembly'
      );
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.components || !Array.isArray(inputs.components)) {
      errors.push({
        field: 'components',
        message: 'Components must be an array',
        severity: 'error',
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        integrationMethod: {
          type: 'string',
          description: 'Integration method to assemble components',
          enum: ['merge', 'compose', 'aggregate', 'pipeline'],
          default: 'merge',
        },
        conflictResolution: {
          type: 'string',
          description: 'Conflict resolution strategy',
          enum: ['priority', 'merge', 'voting', 'custom'],
          default: 'merge',
        },
        maxIterations: {
          type: 'number',
          description: 'Maximum assembly iterations',
          minimum: 1,
          maximum: 10,
          default: 1,
        },
        generateDocumentation: {
          type: 'boolean',
          description: 'Generate assembly documentation',
          default: false,
        },
      },
      required: [],
    };
  }

  private assembleComponents(components: any[], method: IntegrationMethod) {
    switch (method) {
      case 'compose':
        return this.composeComponents(components);
      case 'aggregate':
        return { items: components, count: components.length };
      case 'pipeline':
        return this.pipelineComponents(components);
      case 'merge':
      default:
        return this.mergeComponents(components);
    }
  }

  private mergeComponents(components: any[]) {
    return components.reduce((acc, item) => this.deepMerge(acc, item), {});
  }

  private composeComponents(components: any[]) {
    if (components.every((component) => typeof component === 'string')) {
      return components.join('\n\n');
    }

    if (components.every(Array.isArray)) {
      return components.flat();
    }

    return this.mergeComponents(components);
  }

  private pipelineComponents(components: any[]) {
    if (!components.length) {
      return null;
    }

    return components.reduce((acc, component) => {
      if (typeof component === 'function') {
        return component(acc);
      }
      return component;
    }, components[0]);
  }

  private deepMerge(target: any, source: any): any {
    if (Array.isArray(target) && Array.isArray(source)) {
      return [...target, ...source];
    }

    if (this.isObject(target) && this.isObject(source)) {
      const merged: Record<string, any> = { ...target };
      for (const [key, value] of Object.entries(source)) {
        if (key in merged) {
          merged[key] = this.deepMerge(merged[key], value);
        } else {
          merged[key] = value;
        }
      }
      return merged;
    }

    return source ?? target;
  }

  private isObject(value: any): value is Record<string, any> {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }
}

export default AssemblyNode;
