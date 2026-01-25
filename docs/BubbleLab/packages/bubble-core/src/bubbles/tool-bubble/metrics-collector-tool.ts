import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * MetricsCollectorTool - Metrics collection and aggregation
 */
export class MetricsCollectorTool extends ToolBubble<MetricsCollectorParams, MetricsCollectorResult> {
  bubbleName = 'metrics-collector';
  type = 'tool';
  alias = 'metrics-collector';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<MetricsCollectorResult> {
    try {
      const result = await this.collect(input);
      return { success: true, metrics: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async collect(params: { source: string; metrics: string[] }): Promise<MetricsCollectorResult> {
    try {
      const collected = params.metrics.map(metric => ({
        name: metric,
        value: Math.random() * 100,
        timestamp: new Date().toISOString(),
        labels: { source: params.source }
      }));
      return { success: true, collected };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async aggregate(params: { metrics: any[]; operation: 'sum' | 'avg' | 'min' | 'max' }): Promise<MetricsCollectorResult> {
    try {
      const values = params.metrics.map(m => m.value);
      let result;
      switch (params.operation) {
        case 'sum':
          result = values.reduce((a, b) => a + b, 0);
          break;
        case 'avg':
          result = values.reduce((a, b) => a + b, 0) / values.length;
          break;
        case 'min':
          result = Math.min(...values);
          break;
        case 'max':
          result = Math.max(...values);
          break;
      }
      return { success: true, result, operation: params.operation };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async query(params: { metric: string; start: string; end: string }): Promise<MetricsCollectorResult> {
    try {
      const dataPoints = Array.from({ length: 10 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        value: Math.random() * 100
      }));
      return { success: true, dataPoints };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async export(params: { metrics: any[]; format: 'json' | 'csv' }): Promise<MetricsCollectorResult> {
    try {
      let exported;
      if (params.format === 'json') {
        exported = JSON.stringify(params.metrics, null, 2);
      } else if (params.format === 'csv') {
        const headers = Object.keys(params.metrics[0] || {}).join(',');
        const rows = params.metrics.map(m => Object.values(m).join(','));
        exported = [headers, ...rows].join('\n');
      }
      return { success: true, exported, format: params.format };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface MetricsCollectorParams {
  timeout?: number;
}

export interface MetricsCollectorResult {
  success: boolean;
  metrics?: any[];
  collected?: any[];
  result?: number;
  operation?: string;
  dataPoints?: any[];
  exported?: string;
  format?: string;
  error?: string;
}
