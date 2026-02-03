/**
 * Trace Exporter - Export traces to various backends
 *
 * This module provides support for exporting traces to different backends
 * including Jaeger, OpenTelemetry Collector, and console output.
 */

import {
  ConsoleSpanExporter,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';
import { OTLPTraceExporter as OTLPTraceExporterHTTP } from '@opentelemetry/exporter-trace-otlp-http';
import { SpanExporter } from '@opentelemetry/sdk-trace-base';
import {
  ExporterType,
  type OtlpExporterOptions,
  type ConsoleExporterOptions,
  type CollectorExporterOptions,
} from './types.js';
import { TraceLogger } from './trace-logger.js';

export { type ExporterType } from './types.js';

/**
 * Trace exporter with support for multiple backends
 */
export class TraceExporter {
  private exporter: SpanExporter | null = null;
  private type: ExporterType;
  private logger = new TraceLogger();

  constructor(config: { type: ExporterType; options: unknown }) {
    this.type = config.type;
    this.exporter = this.createExporter(config);
  }

  /**
   * Create the appropriate exporter based on configuration
   */
  private createExporter(config: {
    type: ExporterType;
    options: unknown;
  }): SpanExporter {
    switch (config.type) {
      case ExporterType.OTLP:
        return this.createOtlpExporter(config.options as OtlpExporterOptions);

      case ExporterType.CONSOLE:
        return this.createConsoleExporter(config.options as ConsoleExporterOptions);

      case ExporterType.COLLECTOR:
        return this.createCollectorExporter(config.options as CollectorExporterOptions);

      default:
        this.logger.warn(`Unknown exporter type: ${config.type}, falling back to console`);
        return new ConsoleSpanExporter();
    }
  }

  /**
   * Create OTLP exporter
   */
  private createOtlpExporter(options: OtlpExporterOptions): SpanExporter {
    // Determine if HTTP or gRPC based on URL
    const useHTTP = options.url.startsWith('http://') || options.url.startsWith('https://');

    let exporter: SpanExporter;

    if (useHTTP) {
      exporter = new OTLPTraceExporterHTTP({
        url: options.url,
        headers: options.headers,
      }) as unknown as SpanExporter;

      this.logger.info('Created OTLP HTTP exporter', {
        url: options.url,
      });
    } else {
      exporter = new OTLPTraceExporter({
        url: options.url,
        headers: options.headers,
      }) as unknown as SpanExporter;

      this.logger.info('Created OTLP gRPC exporter', {
        url: options.url,
      });
    }

    return exporter;
  }

  /**
   * Create console exporter (for debugging)
   */
  private createConsoleExporter(options: ConsoleExporterOptions): SpanExporter {
    this.logger.info('Created console exporter', options as Record<string, unknown>);
    return new ConsoleSpanExporter();
  }

  /**
   * Create collector exporter
   */
  private createCollectorExporter(
    options: CollectorExporterOptions
  ): SpanExporter {
    const exporter = new OTLPTraceExporterHTTP({
      url: options.endpoint,
      headers: options.headers,
    }) as unknown as SpanExporter;

    this.logger.info('Created collector exporter', {
      endpoint: options.endpoint,
    });

    return exporter;
  }

  /**
   * Get the underlying exporter
   */
  getExporter(): SpanExporter {
    if (!this.exporter) {
      throw new Error('Exporter not initialized');
    }
    return this.exporter;
  }

  /**
   * Get the exporter type
   */
  getExporterType(): ExporterType {
    return this.type;
  }

  /**
   * Shutdown the exporter
   */
  async shutdown(): Promise<void> {
    if (this.exporter) {
      await this.exporter.shutdown();
      this.exporter = null;
      this.logger.info('Trace exporter shut down');
    }
  }

  /**
   * Force flush any pending spans
   */
  async forceFlush(): Promise<void> {
    if (this.exporter && 'forceFlush' in this.exporter) {
      await (this.exporter as { forceFlush(): Promise<void> }).forceFlush();
    }
  }
}

/**
 * OTLP configuration helper
 */
export class OtlpConfigHelper {
  /**
   * Create OTLP exporter configuration for Honeycomb
   */
  static forHoneycomb(apiKey: string, dataset?: string): {
    type: ExporterType;
    options: OtlpExporterOptions;
  } {
    return {
      type: ExporterType.OTLP,
      options: {
        url: 'https://api.honeycomb.io:443/v1/traces',
        headers: {
          'x-honeycomb-team': apiKey,
          ...(dataset ? { 'x-honeycomb-dataset': dataset } : {}),
        },
      },
    };
  }

  /**
   * Create OTLP exporter configuration for New Relic
   */
  static forNewRelic(apiKey: string): {
    type: ExporterType;
    options: OtlpExporterOptions;
  } {
    return {
      type: ExporterType.OTLP,
      options: {
        url: 'https://otlp.nr-data.net:4317/v1/traces',
        headers: {
          'api-key': apiKey,
        },
      },
    };
  }

  /**
   * Create OTLP exporter configuration for custom collector
   */
  static forCollector(endpoint: string, headers?: Record<string, string>): {
    type: ExporterType;
    options: OtlpExporterOptions;
  } {
    return {
      type: ExporterType.COLLECTOR,
      options: {
        url: endpoint,
        headers,
      },
    };
  }
}
