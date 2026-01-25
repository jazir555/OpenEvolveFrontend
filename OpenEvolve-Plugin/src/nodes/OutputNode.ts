/**
 * Output Node
 *
 * Formats and packages final results for delivery to downstream systems.
 * Supports JSON, YAML, Markdown, HTML, XML, and text outputs.
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

export type OutputFormat = 'json' | 'xml' | 'yaml' | 'markdown' | 'html' | 'text' | 'binary';
export type OutputDestination = 'file' | 'api' | 'database' | 'message_queue' | 'stream';

export interface OutputNodeConfig {
  format?: OutputFormat;
  includeMetadata?: boolean;
  includeMetrics?: boolean;
  includeArtifacts?: boolean;
  destination?: OutputDestination;
  compression?: 'none' | 'gzip' | 'zip';
  signOutput?: boolean;
  formattingOptions?: Record<string, unknown>;
}

export interface OutputResult {
  payload: string;
  format: OutputFormat;
  destination: OutputDestination;
  delivered: boolean;
  metadata: {
    generatedAt: Date;
    size: number;
  };
}

export class OutputNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Output Formatter';
  static readonly DESCRIPTION = 'Format and package workflow outputs for delivery';
  static readonly ICON = 'output';
  static readonly CATEGORY = 'output';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: OutputNodeConfig = {}) {
    super(id, {
      format: 'json',
      includeMetadata: true,
      includeMetrics: true,
      includeArtifacts: true,
      destination: 'stream',
      compression: 'none',
      signOutput: false,
      formattingOptions: {},
      ...config,
    });
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const data = inputs.data ?? inputs.payload ?? inputs;
      const format = (inputs.format as OutputFormat) || (this.config.format as OutputFormat);
      const destination =
        (inputs.destination as OutputDestination) ||
        (this.config.destination as OutputDestination);

      context.updateProgress(20, 'Formatting output');

      const payload = this.formatOutput(data, format);
      const delivered = await this.deliverOutput(payload, destination, inputs);

      const result: OutputResult = {
        payload,
        format,
        destination,
        delivered,
        metadata: {
          generatedAt: new Date(),
          size: payload.length,
        },
      };

      context.updateProgress(100, 'Output packaging complete');
      return this.createSuccessResult(result, { executionTime: Date.now() - startTime });
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during output formatting'
      );
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (inputs.data === undefined && inputs.payload === undefined) {
      errors.push({
        field: 'data',
        message: 'Output data is required',
        severity: 'error',
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        format: {
          type: 'string',
          description: 'Output format',
          enum: ['json', 'xml', 'yaml', 'markdown', 'html', 'text', 'binary'],
          default: 'json',
        },
        destination: {
          type: 'string',
          description: 'Output destination',
          enum: ['file', 'api', 'database', 'message_queue', 'stream'],
          default: 'stream',
        },
        includeMetadata: {
          type: 'boolean',
          description: 'Include metadata in the output payload',
          default: true,
        },
        includeMetrics: {
          type: 'boolean',
          description: 'Include metrics in the output payload',
          default: true,
        },
      },
      required: [],
    };
  }

  private formatOutput(data: any, format: OutputFormat): string {
    switch (format) {
      case 'json':
        return JSON.stringify(data, null, 2);
      case 'yaml':
        return this.toYaml(data);
      case 'markdown':
        return this.toMarkdown(data);
      case 'html':
        return this.toHtml(data);
      case 'xml':
        return this.toXml(data);
      case 'text':
      case 'binary':
      default:
        return typeof data === 'string' ? data : JSON.stringify(data);
    }
  }

  private async deliverOutput(
    payload: string,
    destination: OutputDestination,
    inputs: NodeInputs
  ): Promise<boolean> {
    const deliverFn = inputs.deliver as
      | ((payload: string, destination: OutputDestination) => Promise<boolean> | boolean)
      | undefined;

    if (deliverFn) {
      const result = await deliverFn(payload, destination);
      return Boolean(result);
    }

    // Default behavior is to return payload without side effects.
    return destination === 'stream';
  }

  private toYaml(value: any, indent = 0): string {
    if (value === null || value === undefined) {
      return 'null';
    }

    if (typeof value !== 'object') {
      return String(value);
    }

    const pad = '  '.repeat(indent);
    if (Array.isArray(value)) {
      return value
        .map((item) => `${pad}- ${this.toYaml(item, indent + 1).trimStart()}`)
        .join('\n');
    }

    return Object.entries(value)
      .map(([key, val]) => {
        const formatted = typeof val === 'object' && val !== null
          ? `\n${this.toYaml(val, indent + 1)}`
          : ` ${this.toYaml(val, 0)}`;
        return `${pad}${key}:${formatted}`;
      })
      .join('\n');
  }

  private toMarkdown(value: any): string {
    if (typeof value === 'string') {
      return value;
    }

    if (Array.isArray(value)) {
      return value.map((item) => `- ${this.toMarkdown(item)}`).join('\n');
    }

    if (this.isObject(value)) {
      return Object.entries(value)
        .map(([key, val]) => `**${key}**: ${this.toMarkdown(val)}`)
        .join('\n\n');
    }

    return String(value);
  }

  private toHtml(value: any): string {
    if (typeof value === 'string') {
      return `<p>${this.escapeHtml(value)}</p>`;
    }

    if (Array.isArray(value)) {
      return `<ul>${value.map((item) => `<li>${this.toHtml(item)}</li>`).join('')}</ul>`;
    }

    if (this.isObject(value)) {
      return Object.entries(value)
        .map(([key, val]) => `<section><h3>${this.escapeHtml(key)}</h3>${this.toHtml(val)}</section>`)
        .join('');
    }

    return `<span>${this.escapeHtml(String(value))}</span>`;
  }

  private toXml(value: any, nodeName = 'root'): string {
    if (!this.isObject(value)) {
      return `<${nodeName}>${this.escapeXml(String(value))}</${nodeName}>`;
    }

    const entries = Object.entries(value)
      .map(([key, val]) => this.toXml(val, key))
      .join('');
    return `<${nodeName}>${entries}</${nodeName}>`;
  }

  private escapeHtml(value: string): string {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  private escapeXml(value: string): string {
    return this.escapeHtml(value);
  }

  private isObject(value: any): value is Record<string, any> {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }
}

export default OutputNode;
